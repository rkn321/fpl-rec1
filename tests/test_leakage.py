"""Leakage tests — the guarantee promised in brief §9.

Two independent checks, both stated as properties of the feature builder rather
than as spot checks on individual columns, so they keep holding as features are
added in later phases:

1. **No future information.** Features for gameweek `t` must be identical
   whether or not gameweeks after `t` exist in the input.
2. **No same-gameweek information.** Features for gameweek `t` must be identical
   when every post-match outcome from gameweek `t` onward is replaced with
   garbage. This is the strong one: it catches a feature reading its own row's
   result, which check 1 cannot see.

Anything that fails these is using information that was not available at the
deadline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data.schema import ALL_COLS, CONTEXT_COLS, OUTCOME_COLS, TARGET
from src.features.build import build_features

KEY = ["season", "player_id", "fixture_id"]


def features_at_gw(df: pd.DataFrame, gw: int) -> pd.DataFrame:
    """Feature rows for one gameweek, keyed so two runs can be compared."""
    built, feature_cols = build_features(df)
    rows = built[built["gw"] == gw]
    return rows.set_index(KEY)[feature_cols].sort_index()


def corrupt_outcomes(df: pd.DataFrame, from_gw: int, seed: int = 1) -> pd.DataFrame:
    """Replace every post-match stat from `from_gw` onward with noise."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    mask = (out["gw"] >= from_gw).to_numpy()
    n = int(mask.sum())
    for col in OUTCOME_COLS + [TARGET]:
        if col not in out.columns:
            continue
        # Widen to float first: pandas 3 refuses to write noise into an int column.
        values = np.array(out[col].astype("float64").to_numpy(), copy=True)
        values[mask] = rng.uniform(-50, 50, size=n)
        out[col] = values
    return out


@pytest.mark.parametrize("gw", [3, 5, 8])
def test_no_future_leakage(synthetic: pd.DataFrame, gw: int) -> None:
    """Truncating the future must not change the present's features."""
    full = features_at_gw(synthetic, gw)
    truncated = features_at_gw(synthetic[synthetic["gw"] <= gw], gw)

    assert not full.empty, "test would pass vacuously with no rows"
    assert_frame_equal(full, truncated, check_dtype=False)


@pytest.mark.parametrize("gw", [3, 5, 8])
def test_no_same_gameweek_leakage(synthetic: pd.DataFrame, gw: int) -> None:
    """Corrupting gameweek `t`'s own results must not change its features."""
    clean = features_at_gw(synthetic, gw)
    corrupted = features_at_gw(corrupt_outcomes(synthetic, from_gw=gw), gw)

    assert not clean.empty
    assert_frame_equal(clean, corrupted, check_dtype=False)


def test_first_fixture_has_no_form(synthetic: pd.DataFrame) -> None:
    """A player's debut has no prior fixtures, so form features must be null."""
    built, _ = build_features(synthetic)
    first = built.sort_values(["season", "player_id", "kickoff_time"]).groupby(
        ["season", "player_id"], observed=True
    ).head(1)

    form_cols = ["minutes_r3", "total_points_r3", "minutes_lag1", "points_per_90_todate"]
    assert first[form_cols].isna().all().all()


def test_target_never_offered_as_a_feature(synthetic: pd.DataFrame) -> None:
    """The target and every raw outcome column stay out of the feature list."""
    _, feature_cols = build_features(synthetic)

    banned = set(OUTCOME_COLS) | {TARGET}
    assert banned.isdisjoint(feature_cols)

    # Only context columns are allowed through raw.
    raw_schema_cols = set(ALL_COLS).intersection(feature_cols)
    assert raw_schema_cols <= set(CONTEXT_COLS) | {"gw", "element_type"}


def test_fpl_expected_points_is_used_raw_and_that_is_deliberate() -> None:
    """`xP` is a feature for its own gameweek, unlike every other same-week column.

    It is FPL's published pre-deadline forecast, so using it for the row it sits
    on is legitimate — and it is the only input carrying team news. This test
    exists so the exception stays a decision rather than becoming an accident:
    if `xP` ever moves back to being post-match data, this is what should fail.
    """
    from src.data.schema import CONTEXT_COLS, OUTCOME_COLS

    assert "xP" in CONTEXT_COLS, "xP is knowable before the deadline"
    assert "xP" not in OUTCOME_COLS, "xP is not a match outcome"


def test_rolling_window_respects_the_lag(synthetic: pd.DataFrame) -> None:
    """`_r3` on fixture n must equal the mean of fixtures n-3..n-1, and nothing else.

    In this fixture list every team plays exactly once per gameweek, so the
    gameweek boundary and the fixture boundary coincide and the window is
    straightforward to state.
    """
    built, _ = build_features(synthetic)
    player = built[built["player_id"] == 1].sort_values("gw").reset_index(drop=True)

    for i in range(1, len(player)):
        window = player["total_points"].iloc[max(0, i - 3) : i]
        assert player["total_points_r3"].iloc[i] == pytest.approx(window.mean())


def test_double_gameweek_fixtures_share_features(double_gameweek: pd.DataFrame) -> None:
    """Both fixtures of a double must carry identical form features.

    You pick your team once, before the deadline. If the second fixture's
    features differed from the first's, they would be reading a result that had
    not happened yet at the moment of selection.
    """
    built, feature_cols = build_features(double_gameweek)

    doubles = built[built["gw"] == 5]
    per_player = doubles.groupby("player_id")["fixture_id"].nunique()
    assert (per_player == 2).all(), "the fixture used here is meant to be a double"

    # These describe the specific fixture rather than the player's form, and
    # every one of them is published before the deadline, so they are *supposed*
    # to differ between the two legs: different opponent, venue and rest.
    per_fixture = {"is_home", "team_difficulty", "opp_difficulty", "days_rest"}
    form_cols = [
        c for c in feature_cols if c not in per_fixture and not c.startswith("opp_form_")
    ]

    spread = doubles.groupby("player_id")[form_cols].nunique(dropna=False).max()
    varying = sorted(spread[spread > 1].index)
    assert not varying, f"form features differ between legs of a double: {varying}"


def test_double_gameweek_first_leg_does_not_leak(double_gameweek: pd.DataFrame) -> None:
    """Corrupting the double's own results must not move the double's features."""
    clean = features_at_gw(double_gameweek, 5)
    corrupted = features_at_gw(corrupt_outcomes(double_gameweek, from_gw=5), 5)

    assert not clean.empty
    assert_frame_equal(clean, corrupted, check_dtype=False)
