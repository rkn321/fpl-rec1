"""Elo ratings rebuilt from results, and the fixture multipliers fitted on them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.elo import START, EloBook, expected_home, fixture_gap, margin_factor
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
