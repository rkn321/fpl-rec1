"""Client for the official FPL API.

Free, no auth, JSON. Two things matter here and both are handled below:

* **Caching.** `bootstrap-static/` is ~3MB and we hit it constantly; every
  response is cached to disk with a per-endpoint TTL.
* **Retry/backoff.** The API 503s around deadlines and at season launch, and
  rate-limits bursts of `element-summary/` calls. Retries are exponential with
  jitter and honour `Retry-After`.

Endpoint map (see the project brief):
    bootstrap-static/          players, teams, events, prices, chips
    fixtures/                  all fixtures + FPL difficulty ratings
    element-summary/{id}/      per-player gameweek history
    event/{gw}/live/           live per-player stat breakdown
    entry/{id}/history/        a manager's chips / transfers / ranks
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from ..config import Config, load_config

log = logging.getLogger(__name__)

# element_type -> position code. 1=GK, 2=DEF, 3=MID, 4=FWD.
ELEMENT_TYPE_TO_POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_TO_ELEMENT_TYPE = {v: k for k, v in ELEMENT_TYPE_TO_POSITION.items()}

_RETRY_STATUS = {429, 500, 502, 503, 504}


class FPLAPIError(RuntimeError):
    """Raised when the FPL API cannot be reached after exhausting retries."""


def _cache_key(path: str) -> str:
    """`element-summary/123/` -> `element-summary__123`."""
    return re.sub(r"[^A-Za-z0-9]+", "__", path.strip("/")) or "root"


def _ttl_family(path: str) -> str:
    """Map a request path onto a TTL bucket in config."""
    p = path.strip("/")
    if p.startswith("bootstrap-static"):
        return "bootstrap-static"
    if p.startswith("fixtures"):
        return "fixtures"
    if p.startswith("element-summary"):
        return "element-summary"
    if p.startswith("event") and p.endswith("live"):
        return "event-live"
    return "default"


class FPLClient:
    """Cached, retrying client for the official FPL API."""

    def __init__(self, config: Config | None = None, cache_dir: Path | None = None):
        self.config = config or load_config()
        api = self.config.api
        self.base_url: str = api["base_url"]
        self.timeout: float = api.get("timeout", 30)
        self.max_retries: int = api.get("max_retries", 5)
        self.backoff_base: float = api.get("backoff_base", 1.0)
        self.backoff_max: float = api.get("backoff_max", 30.0)
        self.max_workers: int = api.get("max_workers", 8)
        self._ttl: dict[str, int] = api.get("ttl", {})

        self.cache_dir = Path(cache_dir) if cache_dir else self.config.cache_dir / "api"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": api.get("user_agent", "fpl-pred1/0.1"),
                "Accept": "application/json",
            }
        )

    # -- plumbing ----------------------------------------------------------
    def _ttl_for(self, path: str) -> int:
        return int(self._ttl.get(_ttl_family(path), self._ttl.get("default", 3600)))

    def _cache_path(self, path: str) -> Path:
        return self.cache_dir / f"{_cache_key(path)}.json"

    def _read_cache(self, path: str, ttl: int | None) -> Any | None:
        fp = self._cache_path(path)
        if not fp.exists():
            return None
        ttl = self._ttl_for(path) if ttl is None else ttl
        if ttl >= 0 and (time.time() - fp.stat().st_mtime) > ttl:
            return None
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            log.warning("corrupt cache entry %s; refetching", fp)
            return None

    def _write_cache(self, path: str, payload: Any) -> None:
        fp = self._cache_path(path)
        tmp = fp.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tmp.replace(fp)  # atomic: a killed process never leaves a half-written cache

    def _sleep_for(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self.backoff_max)
            except ValueError:
                pass
        delay = min(self.backoff_base * (2**attempt), self.backoff_max)
        return delay * (0.5 + random.random() / 2)  # jitter, so parallel calls desync

    def get(self, path: str, *, ttl: int | None = None, force: bool = False) -> Any:
        """GET `path` relative to the API base, via the disk cache.

        `ttl=-1` means "cache forever" (used for finished gameweeks).
        `force=True` bypasses the cache read but still writes it.
        """
        if not force:
            cached = self._read_cache(path, ttl)
            if cached is not None:
                return cached

        url = self.base_url + path.lstrip("/")
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code in _RETRY_STATUS:
                    delay = self._sleep_for(attempt, resp.headers.get("Retry-After"))
                    log.warning(
                        "FPL API %s on %s (attempt %d/%d); retrying in %.1fs",
                        resp.status_code,
                        path,
                        attempt + 1,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                payload = resp.json()
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last_exc = exc
                delay = self._sleep_for(attempt, None)
                log.warning(
                    "FPL API error on %s (attempt %d/%d): %s; retrying in %.1fs",
                    path,
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue
            self._write_cache(path, payload)
            return payload

        # Exhausted retries: a stale cache entry beats no data at all.
        stale = self._read_cache(path, ttl=-1)
        if stale is not None:
            log.error("FPL API unreachable for %s; falling back to stale cache", path)
            return stale
        raise FPLAPIError(
            f"failed to fetch {url} after {self.max_retries} attempts"
        ) from last_exc

    # -- endpoints ---------------------------------------------------------
    def bootstrap_static(self, force: bool = False) -> dict[str, Any]:
        return self.get("bootstrap-static/", force=force)

    def fixtures(self, force: bool = False) -> list[dict[str, Any]]:
        return self.get("fixtures/", force=force)

    def element_summary(self, player_id: int, force: bool = False) -> dict[str, Any]:
        return self.get(f"element-summary/{int(player_id)}/", force=force)

    def event_live(self, gw: int, force: bool = False) -> dict[str, Any]:
        return self.get(f"event/{int(gw)}/live/", force=force)

    def entry_history(self, team_id: int, force: bool = False) -> dict[str, Any]:
        return self.get(f"entry/{int(team_id)}/history/", force=force)

    # -- tidy frames -------------------------------------------------------
    def players(self) -> pd.DataFrame:
        """One row per player, from `bootstrap-static`."""
        df = pd.DataFrame(self.bootstrap_static()["elements"])
        df["position"] = df["element_type"].map(ELEMENT_TYPE_TO_POSITION)
        return df

    def teams(self) -> pd.DataFrame:
        return pd.DataFrame(self.bootstrap_static()["teams"])

    def events(self) -> pd.DataFrame:
        """One row per gameweek, with the deadline parsed to UTC."""
        df = pd.DataFrame(self.bootstrap_static()["events"])
        df["deadline_time"] = pd.to_datetime(
            df["deadline_time"], utc=True, format="ISO8601"
        )
        return df

    def fixtures_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.fixtures())
        if "stats" in df.columns:
            df = df.drop(columns=["stats"])  # nested post-match detail; not used
        df["kickoff_time"] = pd.to_datetime(
            df["kickoff_time"], utc=True, format="ISO8601"
        )
        return df

    def current_gw(self) -> int | None:
        """The gameweek in progress or most recently finished."""
        ev = self.events()
        cur = ev.loc[ev["is_current"], "id"]
        if len(cur):
            return int(cur.iloc[0])
        finished = ev.loc[ev["finished"], "id"]
        return int(finished.max()) if len(finished) else None

    def next_gw(self) -> int | None:
        """The next gameweek whose deadline has not passed — what we predict."""
        ev = self.events()
        nxt = ev.loc[ev["is_next"], "id"]
        if len(nxt):
            return int(nxt.iloc[0])
        upcoming = ev.loc[~ev["finished"].astype(bool), "id"]
        return int(upcoming.min()) if len(upcoming) else None

    def player_histories(
        self, player_ids: Iterable[int] | None = None, force: bool = False
    ) -> pd.DataFrame:
        """Per-player gameweek history for the current season, all players.

        This is 600+ requests on a cold cache, so they run on a small thread
        pool. Each individual call still goes through the cache and retry path.
        """
        if player_ids is None:
            player_ids = self.players()["id"].tolist()
        ids = [int(i) for i in player_ids]

        def fetch(pid: int) -> list[dict[str, Any]]:
            try:
                return self.element_summary(pid, force=force).get("history", [])
            except FPLAPIError:
                log.error("giving up on element-summary/%d", pid)
                return []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            chunks = list(pool.map(fetch, ids))

        rows = [r for chunk in chunks for r in chunk]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["kickoff_time"] = pd.to_datetime(
            df["kickoff_time"], utc=True, format="ISO8601"
        )
        return df
