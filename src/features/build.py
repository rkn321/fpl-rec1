"""Assemble the `player_gameweek` feature frame.

Input: the canonical schema (see `src/data/schema.py`), one row per
(season, player_id, fixture_id). Output: the same rows widened with features
that are **all knowable before the deadline**.

Three families of feature:

1. **Player form** — lagged rolling means of every post-match stat over the
   last 3 / 5 / 10 fixtures plus season-to-date.
2. **Team & opponent form** — the same idea at team level, joined onto each
   player row twice: once for their own team, once for the opponent. This is
   the clean-sheet and goals-conceded signal.
3. **Fixture context** — home/away, FPL difficulty, rest days, double-gameweek
   flag, price, gameweek number. Published before the deadline, used raw.

Rows for a *future* gameweek (outcome columns all NA) can be concatenated onto
the played rows before calling `build_features`: the lagged machinery fills
their form features from real history and leaves the target NA.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..data.schema import CONTEXT_COLS, OUTCOME_COLS, TARGET, TEAM_LEVEL_COLS
from .rolling import (
    add_rolling_features,
    expanding_stat,
    previous_fixture_value,
    sort_canonical,
)

log = logging.getLogger(__name__)

# Player stats rolled into form features. Team goals are excluded here because
# they are not a player attribute — they get their own team-level treatment.
PLAYER_ROLL_COLS: list[str] = [c for c in OUTCOME_COLS if c not in TEAM_LEVEL_COLS]

# Team-level stats, one value per (team, fixture).
TEAM_ROLL_COLS: list[str] = [
    "team_goals_for",
    "team_goals_against",
    "team_clean_sheet",
    "team_xg",
    "team_xgc",
    "team_points_scored",
]

PLAYER_GROUP: list[str] = ["season", "player_id"]
TEAM_GROUP: list[str] = ["season", "team_id"]


def _team_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse player rows to one row per (season, team, fixture)."""
    agg = (
        df.groupby(["season", "team_id", "fixture_id"], sort=False, observed=True)
        .agg(
            gw=("gw", "first"),
            kickoff_time=("kickoff_time", "first"),
            team_goals_for=("team_goals_for", "first"),
            team_goals_against=("team_goals_against", "first"),
            team_xg=("expected_goals", "sum"),
            team_xgc=("expected_goals_conceded", "max"),
            team_points_scored=("total_points", "sum"),
        )
        .reset_index()
    )
    agg["team_clean_sheet"] = (agg["team_goals_against"] == 0).astype(float)
    # A future fixture has no result yet; keep those NA so they never enter a
    # rolling window as a zero.
    unplayed = agg["team_goals_for"].isna()
    agg.loc[unplayed, ["team_clean_sheet", "team_xg", "team_xgc", "team_points_scored"]] = np.nan
    return agg


def _team_form(df: pd.DataFrame, windows: list[int], expanding: bool) -> tuple[pd.DataFrame, list[str]]:
    """Lagged rolling team form, ready to join onto player rows."""
    teams = _team_frame(df)
    # Gameweek leads the sort: the lag is taken at the gameweek boundary, so the
    # ordering that defines "before" has to agree with it.
    teams = teams.sort_values(
        ["season", "team_id", "gw", "kickoff_time", "fixture_id"], kind="stable"
    )
    teams = teams.reset_index(drop=True)

    teams, added = add_rolling_features(
        teams,
        group_cols=TEAM_GROUP,
        cols=TEAM_ROLL_COLS,
        windows=windows,
        prefix="team_form_",
        expanding=expanding,
    )
    keep = ["season", "team_id", "fixture_id"] + added
    return teams[keep], added


def _fixture_context_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rest days and double-gameweek flags — both known from the fixture list."""
    df = df.copy()
    added: list[str] = []

    # Rest days come from the published schedule, not from results, so the
    # previous *fixture* is fair game here even inside a double gameweek —
    # you know both kickoff times before the deadline.
    prev_kickoff = df.groupby(PLAYER_GROUP, sort=False, observed=True)["kickoff_time"].shift(1)
    df["days_rest"] = (df["kickoff_time"] - prev_kickoff).dt.total_seconds() / 86400.0
    added.append("days_rest")

    fixtures_in_gw = df.groupby(["season", "team_id", "gw"], sort=False, observed=True)[
        "fixture_id"
    ].transform("nunique")
    df["team_fixtures_in_gw"] = fixtures_in_gw.astype(float)
    df["is_double_gw"] = (fixtures_in_gw > 1).astype(float)
    added += ["team_fixtures_in_gw", "is_double_gw"]

    df["is_home"] = df["was_home"].astype(float)
    added.append("is_home")

    return df, added


def _derived_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Ratios that need lagged sums rather than lagged means.

    Per-90 rates are built from expanding *sums* so a player with one 5-minute
    cameo doesn't get a wild rate; `appearances_todate` gives models the sample
    size to discount it further.
    """
    df = df.copy()
    added: list[str] = []

    df["_appearance"] = (df["minutes"] > 0).astype(float)
    sums = expanding_stat(
        df,
        PLAYER_GROUP,
        ["minutes", "total_points", "expected_goal_involvements", "_appearance", "starts"],
        stat="sum",
    )

    minutes_todate = sums["minutes"]
    per90 = (minutes_todate / 90.0).replace(0, np.nan)

    df["points_per_90_todate"] = sums["total_points"] / per90
    df["xgi_per_90_todate"] = sums["expected_goal_involvements"] / per90
    df["minutes_todate"] = minutes_todate
    df["appearances_todate"] = sums["_appearance"]
    df["starts_todate"] = sums["starts"]
    df["start_rate_todate"] = sums["starts"] / sums["_appearance"].replace(0, np.nan)
    added += [
        "points_per_90_todate",
        "xgi_per_90_todate",
        "minutes_todate",
        "appearances_todate",
        "starts_todate",
        "start_rate_todate",
    ]

    # "Last time they played", counting only fixtures before this gameweek.
    lag1 = previous_fixture_value(df, PLAYER_GROUP, ["minutes", "starts", "total_points"], n=1)
    df["minutes_lag1"] = lag1["minutes"]
    df["started_lag1"] = lag1["starts"]
    df["points_lag1"] = lag1["total_points"]
    lag2 = previous_fixture_value(df, PLAYER_GROUP, ["minutes"], n=2)
    df["minutes_lag2"] = lag2["minutes"]
    added += ["minutes_lag1", "started_lag1", "points_lag1", "minutes_lag2"]

    df = df.drop(columns=["_appearance"])
    return df, added


def build_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    expanding: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Widen a canonical frame with pre-deadline features.

    Returns `(frame, feature_cols)`. `feature_cols` is the contract: it is what
    models train on and what the leakage test checks. Nothing outside it should
    ever reach a model.
    """
    windows = windows or [3, 5, 10]
    df = sort_canonical(df)

    df, player_feats = add_rolling_features(
        df,
        group_cols=PLAYER_GROUP,
        cols=PLAYER_ROLL_COLS,
        windows=windows,
        prefix="",
        expanding=expanding,
    )

    df, derived_feats = _derived_features(df)
    df, ctx_feats = _fixture_context_features(df)

    team_form, team_feats = _team_form(df, windows, expanding)

    # Own-team form.
    df = df.merge(team_form, on=["season", "team_id", "fixture_id"], how="left")

    # Opponent form: same table, joined on the opponent's side of the fixture.
    opp = team_form.rename(
        columns={"team_id": "opponent_id", **{c: c.replace("team_form_", "opp_form_") for c in team_feats}}
    )
    opp_feats = [c.replace("team_form_", "opp_form_") for c in team_feats]
    df = df.merge(opp, on=["season", "opponent_id", "fixture_id"], how="left")

    static_feats = [
        "gw", "element_type", "team_difficulty", "opp_difficulty", "value",
        # Published before the deadline, so usable for the row it sits on.
        "xP",
    ]
    feature_cols = (
        static_feats + ctx_feats + player_feats + derived_feats + team_feats + opp_feats
    )
    feature_cols = [c for c in dict.fromkeys(feature_cols) if c in df.columns]

    _assert_no_raw_outcomes(feature_cols)
    log.info("built %d features over %d rows", len(feature_cols), len(df))
    return df, feature_cols


def _assert_no_raw_outcomes(feature_cols: list[str]) -> None:
    """Cheap guard: a raw post-match column must never be named as a feature."""
    banned = set(OUTCOME_COLS) | {TARGET}
    leaked = sorted(banned.intersection(feature_cols))
    if leaked:
        raise ValueError(f"raw post-match columns used as features: {leaked}")


def build_player_gameweek(
    df: pd.DataFrame, windows: list[int] | None = None, expanding: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """`build_features` plus a tidy column order for the stored parquet."""
    out, feature_cols = build_features(df, windows=windows, expanding=expanding)
    front = [
        "season",
        "player_id",
        "name",
        "position",
        "gw",
        "fixture_id",
        "team_id",
        "opponent_id",
        "kickoff_time",
        TARGET,
        "xP",
    ]
    front = [c for c in front if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest], feature_cols
