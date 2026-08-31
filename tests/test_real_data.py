"""The same leakage guarantees, checked against a real season.

The synthetic tests prove the feature builder is sound in principle. These prove
it on data with the messy bits that actually break pipelines: real double
gameweeks, players transferred mid-season, promoted clubs with no history, and
columns (DEFCON) that simply do not exist before 2025-26.

They read only from the local cache and skip when it is cold, so the suite still
runs offline and in CI. Populate the cache with:

    python -m src.cli build-features
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.config import load_config
from src.data.historical import VaastavLoader
from src.features.build import build_features
from tests.test_leakage import KEY, corrupt_outcomes, features_at_gw

SEASON = "2025-26"


def _cached_season() -> pd.DataFrame:
    config = load_config()
    cached = config.cache_dir / "historical" / SEASON / "gws" / "merged_gw.csv"
    if not cached.exists():
        pytest.skip(f"{cached} not cached — run `python -m src.cli build-features`")
    return VaastavLoader(config).season_frame(SEASON)


@pytest.fixture(scope="module")
def real_season() -> pd.DataFrame:
    return _cached_season()


def test_real_season_loads_on_the_canonical_schema(real_season: pd.DataFrame) -> None:
    assert len(real_season) > 20_000
    assert real_season["team_id"].notna().all()
    assert real_season["opponent_id"].notna().all()
    assert set(real_season["position"].unique()) <= {"GK", "DEF", "MID", "FWD"}
    # A player never plays themselves.
    assert (real_season["team_id"] != real_season["opponent_id"]).all()


def test_real_season_contains_double_gameweeks(real_season: pd.DataFrame) -> None:
    """If this ever fails the doubles tests below have gone vacuous."""
    per_player_gw = real_season.groupby(["player_id", "gw"], observed=True).size()
    assert (per_player_gw > 1).any(), "expected at least one double gameweek"


@pytest.mark.parametrize("gw", [10, 25, 34])
def test_no_future_leakage_on_real_data(real_season: pd.DataFrame, gw: int) -> None:
    full = features_at_gw(real_season, gw)
    truncated = features_at_gw(real_season[real_season["gw"] <= gw], gw)

    assert not full.empty
    assert_frame_equal(full, truncated, check_dtype=False)


@pytest.mark.parametrize("gw", [10, 25, 34])
def test_no_same_gameweek_leakage_on_real_data(real_season: pd.DataFrame, gw: int) -> None:
    clean = features_at_gw(real_season, gw)
    corrupted = features_at_gw(corrupt_outcomes(real_season, from_gw=gw), gw)

    assert not clean.empty
    assert_frame_equal(clean, corrupted, check_dtype=False)


def test_real_doubles_share_form_features(real_season: pd.DataFrame) -> None:
    """Both legs of every real double must carry the same form features."""
    built, feature_cols = build_features(real_season)

    counts = built.groupby(["player_id", "gw"], observed=True)["fixture_id"].transform("nunique")
    doubles = built[counts > 1]
    assert not doubles.empty

    per_fixture = {"is_home", "team_difficulty", "opp_difficulty", "days_rest"}
    form_cols = [
        c for c in feature_cols if c not in per_fixture and not c.startswith("opp_form_")
    ]

    spread = doubles.groupby(["player_id", "gw"], observed=True)[form_cols].nunique(dropna=False)
    varying = sorted(spread.columns[(spread > 1).any()])
    assert not varying, f"form features differ between legs of a real double: {varying}"


def test_promoted_clubs_get_null_form_not_zero(real_season: pd.DataFrame) -> None:
    """Cold start must read as 'unknown', never as a confident zero (brief §6.4).

    A zero would tell a model the player is known to be bad; NaN lets it learn
    what to do with an absence of evidence.
    """
    built, _ = build_features(real_season)
    debuts = built.sort_values(["player_id", "gw"]).groupby("player_id", observed=True).head(1)

    assert debuts["total_points_r3"].isna().all()
    assert debuts["minutes_todate"].isna().all()


def test_defcon_columns_are_present_for_this_season(real_season: pd.DataFrame) -> None:
    """DEFCON arrived in 2025-26; the loader must carry it, not drop it."""
    assert real_season["defensive_contribution"].notna().any()
    assert real_season["clearances_blocks_interceptions"].notna().any()


def test_earlier_season_has_no_defcon_but_still_builds() -> None:
    """2024-25 predates DEFCON — the columns must be absent-but-tolerated."""
    config = load_config()
    cached = config.cache_dir / "historical" / "2024-25" / "gws" / "merged_gw.csv"
    if not cached.exists():
        pytest.skip("2024-25 not cached")

    frame = VaastavLoader(config).season_frame("2024-25")
    assert frame["defensive_contribution"].isna().all()

    built, feature_cols = build_features(frame)
    assert "defensive_contribution_r3" in feature_cols
    assert built["defensive_contribution_r3"].isna().all()
