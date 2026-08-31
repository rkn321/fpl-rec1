"""Unit tests for the lagged rolling helpers.

Small hand-built frames where the right answer can be written down by hand, so
a failure points at the helper rather than at the pipeline around it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.rolling import (
    as_of_previous_gameweek,
    expanding_stat,
    previous_fixture_value,
    rolling_stat,
    sort_canonical,
)

GROUP = ["season", "player_id"]


def frame(points: list[float], gws: list[int] | None = None) -> pd.DataFrame:
    gws = gws or list(range(1, len(points) + 1))
    return pd.DataFrame(
        {
            "season": "2099-00",
            "player_id": 1,
            "gw": gws,
            "fixture_id": range(1, len(points) + 1),
            "kickoff_time": pd.date_range("2099-08-01", periods=len(points), freq="7D", tz="UTC"),
            "points": points,
        }
    )


def test_rolling_mean_excludes_the_current_row() -> None:
    df = frame([1.0, 2.0, 3.0, 4.0, 5.0])
    got = rolling_stat(df, GROUP, ["points"], window=2)["points"]

    # row 0 has no history; row 3 sees rows 1-2 -> (2+3)/2
    expected = [np.nan, 1.0, 1.5, 2.5, 3.5]
    assert got.tolist()[1:] == expected[1:]
    assert np.isnan(got.iloc[0])


def test_expanding_mean_is_season_to_date_before_this_row() -> None:
    df = frame([2.0, 4.0, 6.0])
    got = expanding_stat(df, GROUP, ["points"])["points"]

    assert np.isnan(got.iloc[0])
    assert got.iloc[1] == pytest.approx(2.0)
    assert got.iloc[2] == pytest.approx(3.0)


def test_expanding_sum_matches_a_hand_count() -> None:
    df = frame([1.0, 1.0, 1.0, 1.0])
    got = expanding_stat(df, GROUP, ["points"], stat="sum")["points"]
    assert got.tolist()[1:] == [1.0, 2.0, 3.0]


def test_double_gameweek_rows_get_the_same_value() -> None:
    """Two fixtures in gameweek 3 must both look back only to gameweek 2."""
    df = frame([1.0, 2.0, 10.0, 20.0, 5.0], gws=[1, 2, 3, 3, 4])
    got = rolling_stat(df, GROUP, ["points"], window=3)["points"]

    # Both gameweek-3 rows see gameweeks 1-2 only: (1+2)/2
    assert got.iloc[2] == pytest.approx(1.5)
    assert got.iloc[3] == pytest.approx(1.5)
    # Gameweek 4 sees the last three fixtures: 2, 10, 20
    assert got.iloc[4] == pytest.approx((2.0 + 10.0 + 20.0) / 3)


def test_previous_fixture_value_skips_the_current_gameweek() -> None:
    df = frame([1.0, 2.0, 10.0, 20.0, 5.0], gws=[1, 2, 3, 3, 4])
    first = previous_fixture_value(df, GROUP, ["points"], n=1)["points"]
    second = previous_fixture_value(df, GROUP, ["points"], n=2)["points"]

    # Inside the double, "last time they played" is gameweek 2 for both legs.
    assert first.iloc[2] == pytest.approx(2.0)
    assert first.iloc[3] == pytest.approx(2.0)
    assert second.iloc[2] == pytest.approx(1.0)
    # After the double, the most recent fixture is the double's second leg.
    assert first.iloc[4] == pytest.approx(20.0)
    assert second.iloc[4] == pytest.approx(10.0)


def test_groups_do_not_bleed_into_each_other() -> None:
    a = frame([1.0, 2.0, 3.0])
    b = frame([100.0, 200.0, 300.0])
    b["player_id"] = 2
    df = sort_canonical(pd.concat([a, b], ignore_index=True))

    got = rolling_stat(df, GROUP, ["points"], window=5)["points"]
    df = df.assign(rolled=got)

    p2 = df[df["player_id"] == 2]["rolled"].tolist()
    assert np.isnan(p2[0])
    assert p2[1] == pytest.approx(100.0)
    assert p2[2] == pytest.approx(150.0)


def test_seasons_do_not_bleed_into_each_other() -> None:
    a = frame([1.0, 2.0])
    b = frame([50.0, 60.0])
    b["season"] = "2100-01"
    df = sort_canonical(pd.concat([a, b], ignore_index=True))

    got = expanding_stat(df, GROUP, ["points"])["points"]
    # First row of the later season has no history within that season.
    assert np.isnan(got[df["season"].eq("2100-01").to_numpy()].iloc[0])


def test_as_of_previous_gameweek_takes_the_last_fixture_of_that_gameweek() -> None:
    df = frame([1.0, 2.0, 3.0, 4.0], gws=[1, 1, 2, 3])
    running = df[["points"]]
    got = as_of_previous_gameweek(running, df, GROUP)["points"]

    assert np.isnan(got.iloc[0]) and np.isnan(got.iloc[1])
    assert got.iloc[2] == pytest.approx(2.0)  # last fixture of gameweek 1
    assert got.iloc[3] == pytest.approx(3.0)  # the single gameweek-2 fixture
