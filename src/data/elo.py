"""Team strength as Elo ratings, rebuilt from results the way ClubElo builds them.

The page judges fixtures weeks before any odds are published, and it did that
with FPL's 1-5 difficulty rating alone. ClubElo publishes a continuous
team-strength rating that could serve better, but its API went dark in October
2026: `api.clubelo.com` answers 502 to every ratings request and its fixtures
endpoint reports itself "deactivated". The method is public and small, and the
results it needs are already downloaded for the odds (football-data.co.uk), so
the ratings are computed here instead:

- the home side's expected score is 1 / (10^(-(Elo_home - Elo_away + HOME)/400) + 1)
- after the match each side moves by K x margin x (result - expected), with a
  result of 1, 0.5 or 0
- the margin factor is the square root of the goal difference (1 for a draw or
  a one-goal game), so a 4-0 moves the ratings twice as far as a 1-0
- between seasons every rating is pulled part of the way back to the mean, and
  a promoted side starts at the average its relegated predecessors finished on

K, HOME and CARRY were chosen on 2017-18 to 2021-22 (rated from a 2016-17
warm-up), before any season this project trains or tests on.

**Leak-safe by construction.** `EloBook.as_of(when)` replays only matches
played on an earlier *date* than `when`: football-data records dates without a
reliable time zone, so a match on the deadline's own day is left out rather
than risk counting one that kicked off after it.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ..config import Config
from .odds import OddsLoader, team_resolver

log = logging.getLogger(__name__)

# The first season replayed. Ratings start level here, so a few seasons of
# warm-up separate them before any season the model uses.
WARMUP_FROM = "2016-17"

# Chosen by the Brier score of each match's expected score over 2017-18 to
# 2021-22: 0.1556, against 0.1903 for always predicting the average result and
# 0.1571 for ClubElo's own K=20, home 60. The optimum is flat — K anywhere in
# 10-15 and home 35-45 are within 0.0002 of it.
START = 1500.0   # every side's rating in the first season replayed
K = 12.5         # points exchanged per unit of surprise
HOME = 45.0      # home advantage, in rating points
CARRY = 0.1      # share of the gap to the mean dropped between seasons


def seasons_between(first: str, last: str) -> list[str]:
    """'2016-17', '2018-19' -> ['2016-17', '2017-18', '2018-19']."""
    a, b = int(first[:4]), int(last[:4])
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(a, b + 1)]


def load_results(config: Config, last: str, loader: OddsLoader | None = None) -> pd.DataFrame:
    """Every played Premier League match from WARMUP_FROM to `last`, oldest first.

    Columns: season, date, home, away, home_goals, away_goals. Team names are
    football-data's, which stay the same from one season to the next — unlike
    FPL's team ids, which are renumbered every summer.
    """
    loader = loader or OddsLoader(config)
    frames = []
    for season in seasons_between(WARMUP_FROM, last):
        raw = loader.season(season, current=season == config.season_current)
        if raw is None or raw.empty:
            log.warning("no results for %s; Elo continues without it", season)
            continue
        played = raw.dropna(subset=["FTHG", "FTAG"])
        frames.append(pd.DataFrame({
            "season": season,
            "date": pd.to_datetime(played["Date"], dayfirst=True, format="mixed").dt.normalize(),
            "home": played["HomeTeam"].astype(str),
            "away": played["AwayTeam"].astype(str),
            "home_goals": played["FTHG"].astype(int),
            "away_goals": played["FTAG"].astype(int),
        }))
    if not frames:
        return pd.DataFrame(columns=["season", "date", "home", "away", "home_goals", "away_goals"])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "home"], kind="stable").reset_index(drop=True)


def expected_home(home_elo: float, away_elo: float, home: float = HOME) -> float:
    """The home side's expected score (win 1, draw 0.5) from the two ratings."""
    return 1.0 / (10 ** (-(home_elo - away_elo + home) / 400) + 1.0)


def margin_factor(goal_difference: int) -> float:
    return float(np.sqrt(max(1, abs(int(goal_difference)))))


@dataclass
class EloBook:
    """Ratings replayed through a run of results, queryable as of any date."""

    results: pd.DataFrame
    k: float = K
    home: float = HOME
    carry: float = CARRY
    # One entry per match day, after that day's matches: (date, ratings).
    _days: list[pd.Timestamp] = field(default_factory=list, init=False, repr=False)
    _after: list[dict[str, float]] = field(default_factory=list, init=False, repr=False)
    # Each match's pre-match expectation, for judging the parameters.
    expectations: list[tuple[str, float, float]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        ratings: dict[str, float] = {}
        # Home and away both: early in a season a promoted side may not have hosted anyone yet.
        season_teams = {
            s: set(g["home"]) | set(g["away"]) for s, g in self.results.groupby("season")
        }
        previous: set[str] = set()
        current_season = None

        for day, matches in self.results.groupby("date", sort=True):
            season = matches["season"].iloc[0]
            if season != current_season:
                teams = season_teams.get(season, set())
                ratings = self._new_season(ratings, previous, teams)
                previous, current_season = teams, season
                # The summer's adjustment needs only last season's ratings and
                # who is in the league, both known before a ball is kicked, so
                # it is dated to the close season: a GW1 deadline sees the
                # promoted sides seated and everyone pulled toward the mean.
                close = self._days[-1] + pd.Timedelta(days=1) if self._days else pd.Timestamp(day) - pd.Timedelta(days=1)
                self._days.append(close)
                self._after.append(dict(ratings))
            for m in matches.itertuples(index=False):
                h, a = ratings.setdefault(m.home, START), ratings.setdefault(m.away, START)
                expected = expected_home(h, a, self.home)
                result = 1.0 if m.home_goals > m.away_goals else 0.5 if m.home_goals == m.away_goals else 0.0
                self.expectations.append((season, expected, result))
                change = self.k * margin_factor(m.home_goals - m.away_goals) * (result - expected)
                ratings[m.home], ratings[m.away] = h + change, a - change
            self._days.append(pd.Timestamp(day))
            self._after.append(dict(ratings))

    def _new_season(self, ratings: dict[str, float], previous: set[str], teams: set[str]) -> dict[str, float]:
        """Regress to the mean, and seat promoted sides where the relegated ones finished."""
        if not ratings:
            return {t: START for t in teams}
        staying = [t for t in teams if t in ratings and t in previous]
        relegated = [ratings[t] for t in previous - teams if t in ratings]
        mean = float(np.mean([ratings[t] for t in previous])) if previous else START
        promoted_at = float(np.mean(relegated)) if relegated else mean
        out = dict(ratings)
        for t in staying:
            out[t] = mean + (ratings[t] - mean) * (1 - self.carry)
        for t in teams - set(staying):
            out[t] = mean + (promoted_at - mean) * (1 - self.carry)
        return out

    def as_of(self, when: pd.Timestamp | str) -> dict[str, float]:
        """Every rating after the matches played on dates strictly before `when`."""
        day = pd.Timestamp(when)
        if day.tzinfo is not None:
            day = day.tz_convert(None)
        i = bisect.bisect_left(self._days, day.normalize())
        return dict(self._after[i - 1]) if i else {}

    def brier(self, seasons: Iterable[str]) -> float:
        """Mean squared error of the expected score, over the given seasons' matches."""
        keep = set(seasons)
        errs = [(e - r) ** 2 for s, e, r in self.expectations if s in keep]
        return float(np.mean(errs)) if errs else float("nan")


def team_names(results: pd.DataFrame, season: str, team_ids: Mapping[str, int]) -> dict[int, str]:
    """FPL team id -> football-data name, for one season.

    `team_ids` maps that season's FPL team names to ids, as the odds join takes
    them. A side that cannot be matched is left out and logged.
    """
    resolve = team_resolver(team_ids)
    names = set(results.loc[results["season"] == season, "home"])
    out = {}
    for name in names:
        tid = resolve(name)
        if tid is not None:
            out[tid] = name
    missing = set(int(t) for t in team_ids.values()) - set(out)
    if missing and names:
        log.warning("Elo has no name for FPL team ids %s in %s", sorted(missing), season)
    return out


def fixture_gap(ratings: Mapping[str, float], team: str, opponent: str, home: bool,
                home_advantage: float = HOME) -> float:
    """Rating gap from `team`'s side of a fixture, home advantage included.

    Positive means `team` is the stronger side on the day. NaN when either side
    has no rating.
    """
    if team not in ratings or opponent not in ratings:
        return float("nan")
    return ratings[team] - ratings[opponent] + (home_advantage if home else -home_advantage)
