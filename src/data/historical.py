"""Loaders for the ready-made historical FPL datasets.

Source: **vaastav/Fantasy-Premier-League** — season-by-season CSVs scraped from
the official API and merged with Understat xG/xA, aligned to official FPL
element IDs. Don't re-scrape years by hand (brief §3.2).

`olbauday/FPL-Core-Insights` uses the same layout, so pointing
`historical.vaastav_base` in `config.yaml` at that repo is the only change
needed to swap sources.

Files used per season:
    gws/merged_gw.csv   every player-gameweek row for the season
    fixtures.csv        fixture list + FPL difficulty ratings
    teams.csv           team id <-> name crosswalk
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from ..config import Config, load_config
from .fpl_api import POSITION_TO_ELEMENT_TYPE
from .schema import ALL_COLS, POSITIONS, finalise_frame

log = logging.getLogger(__name__)

# Historical seasons are immutable once finished, so the cache never expires.
# The in-progress season is refreshed by deleting its cache directory.
_TIMEOUT = 60


class HistoricalDataError(RuntimeError):
    pass


class VaastavLoader:
    """Downloads (and caches) one season of the vaastav dataset."""

    def __init__(self, config: Config | None = None, cache_dir: Path | None = None):
        self.config = config or load_config()
        self.base_url: str = self.config.historical["vaastav_base"].rstrip("/")
        self.cache_dir = (
            Path(cache_dir) if cache_dir else self.config.cache_dir / "historical"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": self.config.api.get("user_agent", "fpl-pred1/0.1")}
        )

    def _fetch_csv(self, season: str, relpath: str, force: bool = False) -> pd.DataFrame:
        local = self.cache_dir / season / relpath
        if local.exists() and not force:
            return pd.read_csv(local, low_memory=False, encoding="utf-8")

        url = f"{self.base_url}/{season}/{relpath}"
        log.info("downloading %s", url)
        resp = self.session.get(url, timeout=_TIMEOUT)
        if resp.status_code == 404:
            raise HistoricalDataError(f"{url} not found (season {season} may not exist)")
        resp.raise_for_status()

        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(local.suffix + ".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(local)
        return pd.read_csv(local, low_memory=False, encoding="utf-8")

    # -- raw files ---------------------------------------------------------
    def merged_gw(self, season: str, force: bool = False) -> pd.DataFrame:
        return self._fetch_csv(season, "gws/merged_gw.csv", force=force)

    def fixtures(self, season: str, force: bool = False) -> pd.DataFrame:
        return self._fetch_csv(season, "fixtures.csv", force=force)

    def teams(self, season: str, force: bool = False) -> pd.DataFrame:
        return self._fetch_csv(season, "teams.csv", force=force)

    # -- normalised ---------------------------------------------------------
    def players_raw(self, season: str, force: bool = False) -> pd.DataFrame:
        return self._fetch_csv(season, "players_raw.csv", force=force)

    def set_piece_roles(self, season: str, force: bool = False) -> pd.DataFrame:
        """Penalty and corner/free-kick order per player, from the season snapshot.

        `players_raw.csv` is a single end-of-season row per player, so this is
        the *final* order rather than the order as it stood each gameweek. A
        taker who inherited the job in March is flagged for August too. The role
        is stable enough within a season that this is a fair training proxy — but
        it is a proxy, and it leans slightly toward the future. The live season
        uses the real, current value from the API instead.
        """
        raw = self.players_raw(season, force=force)
        cols = {"id": "player_id", "penalties_order": "penalties_order",
                "corners_and_indirect_freekicks_order": "setpiece_order"}
        have = [c for c in cols if c in raw.columns]
        out = raw[have].rename(columns=cols)
        for c in ("penalties_order", "setpiece_order"):
            if c not in out.columns:
                out[c] = pd.NA
        out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
        return out[["player_id", "penalties_order", "setpiece_order"]]

    def season_frame(self, season: str, force: bool = False) -> pd.DataFrame:
        """One season, normalised to the canonical `player_gameweek` schema."""
        gws = self.merged_gw(season, force=force)
        fixtures = self.fixtures(season, force=force)
        frame = normalise_merged_gw(gws, fixtures, season)
        roles = self.set_piece_roles(season, force=force)
        frame = frame.drop(columns=["penalties_order", "setpiece_order"]).merge(
            roles, on="player_id", how="left", validate="many_to_one"
        )
        return frame[[c for c in frame.columns]]


def _fixture_context(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Fixture id -> (home team, away team, both difficulty ratings).

    FPL publishes difficulty ratings with the fixture list, well before the
    deadline, so they are legitimate pre-match features.
    """
    cols = ["id", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
    missing = [c for c in cols if c not in fixtures.columns]
    if missing:
        raise HistoricalDataError(f"fixtures.csv missing columns: {missing}")
    out = fixtures[cols].copy()
    return out.rename(columns={"id": "fixture_id"})


def normalise_merged_gw(
    gws: pd.DataFrame, fixtures: pd.DataFrame, season: str
) -> pd.DataFrame:
    """Map a vaastav `merged_gw.csv` onto the canonical schema.

    The team a player lined up for is derived from the *fixture*, not from the
    `team` name column: that survives mid-season transfers and sidesteps the
    name-matching trap in brief §6.2.
    """
    df = gws.copy()

    rename = {"element": "player_id", "fixture": "fixture_id", "round": "gw"}
    df = df.rename(columns=rename)

    # Manager assets (2024-25) and any other non-player rows are not modelled.
    df = df[df["position"].isin(POSITIONS)].copy()

    df["season"] = season
    df["element_type"] = df["position"].map(POSITION_TO_ELEMENT_TYPE)
    df["was_home"] = df["was_home"].astype(bool)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True, format="ISO8601")

    ctx = _fixture_context(fixtures)
    df = df.merge(ctx, on="fixture_id", how="left", validate="many_to_one")

    home = df["was_home"]
    df["team_id"] = df["team_h"].where(home, df["team_a"])
    df["opponent_id"] = df["team_a"].where(home, df["team_h"])
    df["team_difficulty"] = df["team_h_difficulty"].where(home, df["team_a_difficulty"])
    df["opp_difficulty"] = df["team_a_difficulty"].where(home, df["team_h_difficulty"])

    # Goals for/against from that player's team's point of view.
    df["team_goals_for"] = df["team_h_score"].where(home, df["team_a_score"])
    df["team_goals_against"] = df["team_a_score"].where(home, df["team_h_score"])

    df = _mark_absent_xp(df)
    return finalise_frame(df)


def _mark_absent_xp(df: pd.DataFrame) -> pd.DataFrame:
    """Blank out gameweeks where FPL's expected points was never captured.

    The scrape is patchy — 27 of 38 gameweeks in 2025-26 have `xP` zero for
    every single player, which is absence recorded as a number. Left as zeros it
    silently corrupts anything that reads the column: as a baseline it predicts
    nobody scores, and as a feature it teaches the model that zero means zero.

    Marked NaN, a missing gameweek is visibly missing.
    """
    if "xP" not in df.columns:
        return df
    all_zero = df.groupby("gw")["xP"].transform(lambda x: (x.fillna(0) == 0).all())
    df.loc[all_zero, "xP"] = pd.NA
    return df


def load_seasons(
    seasons: list[str], config: Config | None = None, force: bool = False
) -> pd.DataFrame:
    """Concatenate several historical seasons into one canonical frame."""
    loader = VaastavLoader(config=config)
    frames = []
    for season in seasons:
        frame = loader.season_frame(season, force=force)
        log.info("%s: %d player-fixture rows", season, len(frame))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(ALL_COLS))
    return pd.concat(frames, ignore_index=True)
