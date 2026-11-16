"""Elo ratings rebuilt from results, and the fixture multipliers fitted on them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from types import SimpleNamespace

from src.data.elo import (
    GAP_COL, HOME, START, EloBook, attach_gap_column, deadlines, expected_home, fixture_gap, margin_factor,
    team_names,
)
from src.evaluate_fixtures import ELO_CAP, fit
from src.frontend import _fixture_lookup


def results(rows: list[tuple[str, str, str, str, int, int]]) -> pd.DataFrame:
    """(season, date, home, away, home goals, away goals) -> the frame EloBook replays."""
    out = pd.DataFrame(rows, columns=["season", "date", "home", "away", "home_goals", "away_goals"])
    out["date"] = pd.to_datetime(out["date"])
    return out


SEASON = [
    ("2020-21", "2020-09-12", "A", "B", 3, 0),
    ("2020-21", "2020-09-19", "C", "D", 1, 1),
    ("2020-21", "2020-09-26", "B", "C", 0, 2),
    ("2020-21", "2020-10-03", "D", "A", 1, 4),
]


def test_expected_score_is_symmetric_without_home_advantage() -> None:
    assert expected_home(1600, 1500, home=0) + expected_home(1500, 1600, home=0) == pytest.approx(1.0)
    assert expected_home(1500, 1500, home=0) == pytest.approx(0.5)
    assert expected_home(1500, 1500, home=45) > 0.5


def test_margin_factor_grows_with_goal_difference() -> None:
    assert margin_factor(0) == margin_factor(1) == 1.0
    assert margin_factor(4) == pytest.approx(2.0)
    assert margin_factor(-4) == pytest.approx(2.0)


def test_ratings_are_exchanged_not_created() -> None:
    book = EloBook(results(SEASON), carry=0.0)
    after = book.as_of("2021-01-01")
    assert sum(after.values()) == pytest.approx(START * 4)
    assert after["A"] > START > after["B"]


def test_as_of_never_sees_the_day_itself_or_later() -> None:
    # The leakage guarantee: changing a result on or after the date cannot move
    # the ratings as of that date.
    base = EloBook(results(SEASON)).as_of("2020-09-26")
    flipped = [r if r[1] < "2020-09-26" else (*r[:4], r[5], r[4]) for r in SEASON]
    assert EloBook(results(flipped)).as_of("2020-09-26") == base
    # And on the day of the first match everyone is still level.
    assert EloBook(results(SEASON)).as_of("2020-09-12") == {t: START for t in "ABCD"}


def test_promoted_side_starts_where_the_relegated_one_finished() -> None:
    # E has only played away so far: it must still be seated at the season's start.
    nxt = [
        ("2021-22", "2021-08-14", "A", "E", 1, 0),   # B relegated, E promoted
        ("2021-22", "2021-08-14", "C", "D", 0, 0),
    ]
    book = EloBook(results(SEASON + nxt), carry=0.0)
    end = book.as_of("2020-10-04")      # the day after the last match
    start = book.as_of("2021-08-14")
    assert start["E"] == pytest.approx(end["B"])
    assert start["A"] == pytest.approx(end["A"])


def test_carry_pulls_ratings_toward_the_mean_between_seasons() -> None:
    nxt = [("2021-22", "2021-08-14", "A", "B", 1, 1), ("2021-22", "2021-08-14", "C", "D", 1, 1)]
    book = EloBook(results(SEASON + nxt), carry=0.5)
    end, start = book.as_of("2020-10-04"), book.as_of("2021-08-14")
    mean = np.mean(list(end.values()))
    assert start["A"] - mean == pytest.approx((end["A"] - mean) * 0.5)


def test_fixture_gap_takes_the_teams_side_and_home_advantage() -> None:
    ratings = {"A": 1600.0, "B": 1500.0}
    assert fixture_gap(ratings, "A", "B", True, 45) == pytest.approx(145)
    assert fixture_gap(ratings, "A", "B", False, 45) == pytest.approx(55)
    assert fixture_gap(ratings, "B", "A", True, 45) == pytest.approx(-55)
    assert np.isnan(fixture_gap(ratings, "A", "Z", True, 45))


def test_elo_multiplier_recovers_a_line_and_caps_it() -> None:
    rng = np.random.default_rng(0)
    gap = rng.uniform(-250, 250, 4000)
    rows = pd.DataFrame({
        "position": rng.choice(["GK", "DEF", "MID", "FWD"], 4000),
        "gap": gap,
        "ratio": 1.0 + 0.001 * gap + rng.normal(0, 0.05, 4000),
        "team_difficulty": 3,
    })
    m = fit(rows, "elo")
    assert m.table.loc["DEF", "b"] == pytest.approx(0.001, abs=1e-4)
    far = m(pd.Series(["DEF", "DEF"]), pd.Series([ELO_CAP, ELO_CAP * 3]))
    assert far[0] == pytest.approx(far[1])            # flat beyond the cap
    assert m(pd.Series(["MID"]), pd.Series([np.nan]))[0] == pytest.approx(m.overall)


class _Client:
    """Just enough of FPLClient for the fixture lookup."""

    def fixtures_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "event": [8, 9], "team_h": [1, 2], "team_a": [2, 1],
            "team_h_difficulty": [2, 4], "team_a_difficulty": [4, 2],
            "kickoff_time": ["2026-10-24T14:00:00Z", "2026-10-31T14:00:00Z"],
        })

    def teams(self) -> pd.DataFrame:
        return pd.DataFrame({"id": [1, 2], "short_name": ["ARS", "BUR"]})


def test_fixture_lookup_carries_each_sides_gap() -> None:
    fx = _fixture_lookup(_Client(), 8, horizon=2, gap=lambda h, a: 150.0 if h == 1 else -40.0)
    ars, bur = fx[1], fx[2]
    assert [f["elo"] for f in ars] == [150, 40]
    assert [f["elo"] for f in bur] == [-150, -40]
    # Without Elo the entries are exactly as before.
    assert "elo" not in _fixture_lookup(_Client(), 8, horizon=2)[1][0]


def test_the_close_season_adjustment_is_visible_before_the_first_match() -> None:
    nxt = [("2021-22", "2021-08-14", "A", "E", 1, 0), ("2021-22", "2021-08-14", "C", "D", 0, 0)]
    book = EloBook(results(SEASON + nxt))
    assert "E" in book.as_of("2021-08-14")         # the GW1 deadline's own day
    assert "E" not in book.as_of("2020-10-04")     # but not before last season had finished


# -- the gap as a column of the frame -------------------------------------------------

class _Loader:
    """football-data's season CSVs, for one season."""

    def __init__(self, played: list[tuple[str, str, str, int, int]]):
        self.frame = pd.DataFrame(played, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    def season(self, season: str, current: bool = False):
        return self.frame if season == "2020-21" else None


class _BrokenLoader:
    def season(self, season: str, current: bool = False):
        raise RuntimeError("football-data is down")


PLAYED = [
    ("12/09/2020", "Arsenal", "Burnley", 3, 0),
    ("19/09/2020", "Chelsea", "Fulham", 1, 1),
]
IDS = {"2020-21": {"Arsenal": 1, "Burnley": 2, "Chelsea": 3, "Fulham": 4}}
CONFIG = SimpleNamespace(season_current="2020-21")


def _frame(rows: list[tuple[int, int, int, bool, str]]) -> pd.DataFrame:
    """(gw, team, opponent, at home, kickoff) per row."""
    return pd.DataFrame(
        rows, columns=["gw", "team_id", "opponent_id", "was_home", "kickoff_time"],
    ).assign(season="2020-21", kickoff_time=lambda d: pd.to_datetime(d["kickoff_time"], utc=True))


def test_the_gap_column_is_the_teams_advantage_as_the_gameweek_began() -> None:
    frame = _frame([
        (1, 1, 2, True, "2020-09-12T14:00Z"),    # GW1: nothing played yet
        (2, 1, 3, True, "2020-09-26T14:00Z"),    # GW2: Arsenal have beaten Burnley 3-0
        (2, 3, 1, False, "2020-09-26T14:00Z"),   # the same fixture from Chelsea's side
    ])
    out = attach_gap_column(frame, CONFIG, IDS, loader=_Loader(PLAYED))
    book = EloBook(results([("2020-21", "2020-09-12", "Arsenal", "Burnley", 3, 0),
                            ("2020-21", "2020-09-19", "Chelsea", "Fulham", 1, 1)]))
    after = book.as_of("2020-09-26")

    assert out.loc[0, GAP_COL] == pytest.approx(HOME), "level ratings: only home advantage"
    assert out.loc[1, GAP_COL] == pytest.approx(after["Arsenal"] - after["Chelsea"] + HOME)
    assert out.loc[1, GAP_COL] == pytest.approx(-out.loc[2, GAP_COL]), "each side sees the other's gap, negated"
    assert out.loc[1, GAP_COL] > out.loc[0, GAP_COL]
    assert GAP_COL not in frame.columns, "the input frame is left alone"


def test_the_gap_never_sees_a_result_from_the_gameweeks_own_day() -> None:
    """A later match in the same gameweek must not move an earlier fixture's gap."""
    frame = _frame([(1, 1, 2, True, "2020-09-12T14:00Z")])
    base = attach_gap_column(frame, CONFIG, IDS, loader=_Loader(PLAYED))
    same_day_result = PLAYED + [("12/09/2020", "Chelsea", "Fulham", 5, 0)]
    again = attach_gap_column(frame, CONFIG, IDS, loader=_Loader(same_day_result))
    assert again.loc[0, GAP_COL] == base.loc[0, GAP_COL]


def test_the_gap_is_nan_when_a_team_cannot_be_rated_and_the_frame_survives_no_elo() -> None:
    frame = _frame([(2, 1, 99, True, "2020-09-26T14:00Z")])           # team 99 is in no table
    assert np.isnan(attach_gap_column(frame, CONFIG, IDS, loader=_Loader(PLAYED)).loc[0, GAP_COL])
    broken = attach_gap_column(frame, CONFIG, IDS, loader=_BrokenLoader())
    assert GAP_COL in broken.columns and broken[GAP_COL].isna().all()
    assert list(broken.drop(columns=GAP_COL).columns) == list(frame.columns)


def test_a_season_with_no_results_yet_has_no_gap_so_the_model_keeps_its_trees() -> None:
    """GW1 of a new season, before any result: nothing to name the clubs by, so NaN, not a guess."""
    frame = _frame([(1, 1, 2, True, "2021-08-14T14:00Z")]).assign(season="2021-22")
    out = attach_gap_column(frame, CONFIG, {**IDS, "2021-22": IDS["2020-21"]}, loader=_Loader(PLAYED))
    assert np.isnan(out.loc[0, GAP_COL])


def test_a_side_that_has_only_played_away_still_has_a_name_and_so_a_gap() -> None:
    """Early in a season not every club has hosted; it must not lose its rating for that."""
    played = results([("2020-21", "2020-09-12", "Arsenal", "Burnley", 3, 0)])
    assert team_names(played, "2020-21", IDS["2020-21"]) == {1: "Arsenal", 2: "Burnley"}


def test_a_gameweeks_deadline_day_is_its_first_kickoff() -> None:
    frame = _frame([
        (1, 1, 2, True, "2020-09-12T14:00Z"), (1, 3, 4, True, "2020-09-11T19:00Z"),
        (2, 1, 3, True, "2020-09-26T14:00Z"),
    ])
    assert deadlines(frame) == {("2020-21", 1): pd.Timestamp("2020-09-11"), ("2020-21", 2): pd.Timestamp("2020-09-26")}
