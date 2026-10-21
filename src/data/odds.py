"""Market odds (Phase 4): the bookmakers' view of each match, as expected goals.

The model's sense of how many goals a team will score or concede comes from
lagged form and FPL's 1-5 difficulty rating. The betting market prices the same
thing with far more information — team strength, form, injuries and rotation
as they stand that week — and its price is public.

**Source.** football-data.co.uk: one CSV per Premier League season
(`mmz4281/2526/E0.csv`) with every match's odds, and `fixtures.csv` with the
odds for matches in the coming days. Free, no key.

**Which odds.** Each match is recorded twice: pre-match odds "collected on
Friday afternoons for weekend games and Tuesday afternoons for midweek games",
and closing odds (the `C` columns). The pre-match ones are used — taken at
about the FPL deadline, they know what a manager could have known. Closing
odds carry the team news that arrives after it.

**Conversion.** The market-average 1X2 and over/under 2.5 goals odds, with the
bookmaker's margin removed by normalising, give P(home win), P(draw), P(away
win) and P(over 2.5). The two Poisson means — home and away goals — whose
implied probabilities best match those four are each side's expected goals.
Independent Poissons understate draws a little; the fit absorbs that into
slightly lower means, which is the same for every match and so harmless to a
model that learns from them.

Each player row gets two columns: `odds_team_goals` (his side's expected
goals) and `odds_opp_goals` (their opponent's, i.e. expected goals conceded).
"""

from __future__ import annotations

import logging
import time
import unicodedata
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import requests

from ..config import Config

log = logging.getLogger(__name__)

SEASON_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
DIVISION = "E0"

# Market average first; Bet365 where the average is missing.
PRICE_SETS: tuple[tuple[str, str, str, str, str], ...] = (
    ("AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5"),
    ("B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"),
)

ODDS_COLS: tuple[str, str] = ("odds_team_goals", "odds_opp_goals")

# football-data's spelling -> FPL's, after normalising case and accents. Only
# the names that differ; the rest match as they are.
ALIASES: dict[str, str] = {
    "man united": "man utd",
    "manchester united": "man utd",
    "manchester city": "man city",
    "tottenham": "spurs",
    "sheffield united": "sheffield utd",
    "nottingham forest": "nott'm forest",
    "wolverhampton": "wolves",
    "west brom": "west brom",
    "newcastle united": "newcastle",
}

# Seasons that have finished never change; the current one fills in weekly,
# and the coming days' odds move hourly.
TTL_CURRENT = 6 * 3600
TTL_FIXTURES = 3600

# The grid the Poisson means are fitted on: 0.10 to 4.50 goals in 0.02 steps.
_GRID = np.round(np.arange(0.10, 4.5001, 0.02), 2)
_MAX_GOALS = 10


def season_code(season: str) -> str:
    """'2025-26' -> '2526', football-data's folder name."""
    start, end = season.split("-")
    return f"{start[-2:]}{end[-2:]}"


def _normalise(name: str) -> str:
    folded = unicodedata.normalize("NFKD", str(name))
    plain = "".join(c for c in folded if not unicodedata.combining(c)).lower().strip()
    return ALIASES.get(plain, plain)


class OddsLoader:
    """Downloads football-data.co.uk CSVs into the cache, with per-file TTLs."""

    def __init__(self, config: Config, session: requests.Session | None = None):
        self.config = config
        self.cache_dir = config.cache_dir / "odds"
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", config.api.get("user_agent", "fpl-pred1"))

    def _get(self, url: str, path: Path, ttl: float | None) -> pd.DataFrame | None:
        fresh = path.exists() and (ttl is None or time.time() - path.stat().st_mtime < ttl)
        if not fresh:
            try:
                resp = self.session.get(url, timeout=30)  # redirects to the bare domain; followed
                resp.raise_for_status()
                if not resp.content.strip():
                    raise ValueError("empty response")
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(".tmp")
                partial.write_bytes(resp.content)
                partial.replace(path)
            except (requests.RequestException, ValueError) as exc:
                if not path.exists():
                    log.warning("odds unavailable from %s: %s", url, exc)
                    return None
                log.warning("odds download failed (%s); using the cached copy of %s", exc, path.name)
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    def season(self, season: str, current: bool = False) -> pd.DataFrame | None:
        code = season_code(season)
        return self._get(SEASON_URL.format(code=code), self.cache_dir / f"E0_{code}.csv",
                         TTL_CURRENT if current else None)

    def upcoming(self) -> pd.DataFrame | None:
        frame = self._get(FIXTURES_URL, self.cache_dir / "fixtures.csv", TTL_FIXTURES)
        if frame is None or "Div" not in frame.columns:
            return None
        return frame[frame["Div"] == DIVISION]


def implied_probabilities(matches: pd.DataFrame) -> pd.DataFrame:
    """P(home), P(draw), P(away), P(over 2.5) with the margin removed; NaN where unpriced."""
    out = pd.DataFrame(index=matches.index, columns=["p_home", "p_draw", "p_away", "p_over"], dtype=float)
    for h, d, a, over, under in PRICE_SETS:
        if not {h, d, a, over, under} <= set(matches.columns):
            continue
        prices = matches[[h, d, a, over, under]].apply(pd.to_numeric, errors="coerce")
        inverse = 1.0 / prices.where(prices > 1.0)
        result = inverse[[h, d, a]].div(inverse[[h, d, a]].sum(axis=1), axis=0)
        total = inverse[[over, under]].sum(axis=1)
        todo = out["p_home"].isna() & result.notna().all(axis=1) & total.notna()
        out.loc[todo, ["p_home", "p_draw", "p_away"]] = result.loc[todo].to_numpy()
        out.loc[todo, "p_over"] = (inverse.loc[todo, over] / total[todo]).to_numpy()
    return out


def _grid_probabilities() -> np.ndarray:
    """For every (home mean, away mean) on the grid: P(home), P(draw), P(away), P(over 2.5)."""
    goals = np.arange(_MAX_GOALS + 1)
    log_fact = np.cumsum(np.log(np.maximum(goals, 1)))
    pmf = np.exp(goals[None, :] * np.log(_GRID[:, None]) - _GRID[:, None] - log_fact[None, :])
    joint = pmf[:, None, :, None] * pmf[None, :, None, :]          # [home mean, away mean, i, j]
    i, j = np.meshgrid(goals, goals, indexing="ij")
    return np.stack([
        (joint * (i > j)).sum(axis=(2, 3)),
        (joint * (i == j)).sum(axis=(2, 3)),
        (joint * (i < j)).sum(axis=(2, 3)),
        1.0 - (joint * (i + j <= 2)).sum(axis=(2, 3)),
    ], axis=-1).reshape(-1, 4)


_GRID_PROBS: np.ndarray | None = None


def implied_goals(probabilities: pd.DataFrame) -> pd.DataFrame:
    """Each match's home and away expected goals: the Poisson pair that best fits its odds."""
    global _GRID_PROBS
    if _GRID_PROBS is None:
        _GRID_PROBS = _grid_probabilities()
    out = pd.DataFrame(index=probabilities.index, columns=["goals_home", "goals_away"], dtype=float)
    known = probabilities[["p_home", "p_draw", "p_away", "p_over"]].notna().all(axis=1).to_numpy()
    target = probabilities.loc[known, ["p_home", "p_draw", "p_away", "p_over"]].to_numpy(dtype=float)
    best = np.empty(len(target), dtype=int)
    for start in range(0, len(target), 256):
        chunk = target[start:start + 256]
        err = ((chunk[:, None, :] - _GRID_PROBS[None, :, :]) ** 2).sum(axis=2)
        best[start:start + 256] = err.argmin(axis=1)
    n = len(_GRID)
    out.loc[known, "goals_home"] = _GRID[best // n]
    out.loc[known, "goals_away"] = _GRID[best % n]
    return out


def team_resolver(team_ids: Mapping[str, int]) -> Callable[[str], int | None]:
    """football-data team name -> FPL team id, or None when it cannot be told apart.

    `team_ids` maps FPL team names to that season's ids.
    """
    by_name = {_normalise(name): int(tid) for name, tid in team_ids.items()}

    def resolve(name: str) -> int | None:
        # Exact after normalising, else the one FPL name it starts: football-data
        # writes "Hull" where FPL has written both "Hull" and "Hull City".
        key = _normalise(name)
        if key in by_name:
            return by_name[key]
        starts = [tid for fpl, tid in by_name.items() if fpl.startswith(key + " ")]
        return starts[0] if len(starts) == 1 else None

    return resolve


def match_goals(odds: pd.DataFrame, team_ids: Mapping[str, int]) -> pd.DataFrame:
    """One row per priced match: home_id, away_id, goals_home, goals_away.

    `team_ids` maps FPL team names to that season's ids. A team that cannot be
    mapped is reported and its matches dropped, rather than joined to the
    wrong side.
    """
    resolve = team_resolver(team_ids)
    home = odds["HomeTeam"].map(resolve)
    away = odds["AwayTeam"].map(resolve)
    unknown = sorted(set(odds.loc[home.isna(), "HomeTeam"]) | set(odds.loc[away.isna(), "AwayTeam"]))
    if unknown:
        log.warning("odds for teams not in FPL's list, dropped: %s", unknown)
    goals = implied_goals(implied_probabilities(odds))
    out = pd.DataFrame({"home_id": home, "away_id": away}).join(goals)
    out = out.dropna()
    out[["home_id", "away_id"]] = out[["home_id", "away_id"]].astype(int)
    return out.drop_duplicates(["home_id", "away_id"], keep="first").reset_index(drop=True)


def attach(df: pd.DataFrame, season: str, goals: pd.DataFrame) -> pd.DataFrame:
    """Fill `odds_team_goals` / `odds_opp_goals` on `season`'s rows from `match_goals`.

    A pairing is unique within a season — each side hosts the other once — so
    (home, away) identifies the match without dates, which football-data and
    FPL record in different time zones.
    """
    out = df.copy()
    for col in ODDS_COLS:
        if col not in out.columns:
            out[col] = np.nan
    if goals is None or goals.empty:
        return out
    rows = (out["season"] == season).to_numpy()
    home = out.loc[rows, "was_home"].astype(bool).to_numpy()
    team = pd.to_numeric(out.loc[rows, "team_id"], errors="coerce").to_numpy()
    opp = pd.to_numeric(out.loc[rows, "opponent_id"], errors="coerce").to_numpy()
    key = pd.MultiIndex.from_arrays([np.where(home, team, opp), np.where(home, opp, team)])
    table = goals.set_index(["home_id", "away_id"])
    gh = table["goals_home"].reindex(key).to_numpy()
    ga = table["goals_away"].reindex(key).to_numpy()
    out.loc[rows, "odds_team_goals"] = np.where(home, gh, ga)
    out.loc[rows, "odds_opp_goals"] = np.where(home, ga, gh)
    return out


def attach_all(
    df: pd.DataFrame,
    config: Config,
    team_ids: Mapping[str, Mapping[str, int]],
    current: str,
    loader: OddsLoader | None = None,
) -> pd.DataFrame:
    """Odds for every season in `df`; the current season also takes the coming days' prices.

    `team_ids` is season -> {FPL team name: id}. Missing data — a download that
    fails, a season football-data does not have — leaves the columns NaN and
    the pipeline running; the coverage is logged either way.
    """
    loader = loader or OddsLoader(config)
    out = df
    for season in sorted(df["season"].dropna().unique()):
        names = team_ids.get(season)
        odds = loader.season(season, current=season == current)
        if season == current:
            coming = loader.upcoming()
            if coming is not None and len(coming):
                odds = coming if odds is None else pd.concat([odds, coming], ignore_index=True)
        if not names or odds is None or odds.empty:
            out = attach(out, season, None)
            continue
        out = attach(out, season, match_goals(odds, names))
        rows = out["season"] == season
        log.info("odds for %s: %.0f%% of rows", season, 100 * out.loc[rows, "odds_team_goals"].notna().mean())
    return out
