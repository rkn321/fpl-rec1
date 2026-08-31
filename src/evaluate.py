"""Walk-forward backtest harness.

The only honest way to test this: for each gameweek `t`, train on everything up
to and including `t-1` and predict `t`. Never shuffle across time (brief §7).

**Evaluation grain is (season, player_id, gameweek), not player-fixture.** FPL
pays you per gameweek, so in a double gameweek the two fixtures' predictions are
summed and compared against the summed actual. Predictions are still made per
fixture — that is where the features live.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .data.schema import TARGET
from .metrics import ACTUAL, PRED, score_gameweek, summarise
from .models.baselines import Predictor, default_baselines

log = logging.getLogger(__name__)

GROUP = ["season", "player_id", "gw"]


def to_gameweek_grain(rows: pd.DataFrame) -> pd.DataFrame:
    """Sum player-fixture predictions and actuals up to player-gameweek."""
    return (
        rows.groupby(GROUP, sort=False, observed=True)
        .agg(
            **{
                PRED: (PRED, "sum"),
                ACTUAL: (ACTUAL, "sum"),
                "position": ("position", "first"),
                "fixtures": ("fixture_id", "nunique"),
            }
        )
        .reset_index()
    )


def playing_filter(rows: pd.DataFrame, min_minutes_r3: float = 1.0) -> pd.Series:
    """Rows for players with any recent minutes.

    Reported alongside the all-players view because ~40% of the player pool
    never plays; their guaranteed zeros flatter MAE and tell you nothing about
    whether the model can pick a squad.
    """
    return rows["minutes_r3"].fillna(0.0) >= min_minutes_r3


def walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    predictors: dict[str, Predictor] | None = None,
    season: str | None = None,
    min_train_gws: int = 4,
    train_on_prior_seasons: bool = True,
) -> pd.DataFrame:
    """Run every predictor over every testable gameweek.

    Returns one row per (model, season, gameweek, fixture-row) prediction — the
    raw material for `score` below.
    """
    predictors = predictors or default_baselines()

    test_seasons = [season] if season else sorted(df["season"].unique())
    out: list[pd.DataFrame] = []

    for s in test_seasons:
        season_rows = df[df["season"] == s]
        gws = sorted(int(g) for g in season_rows["gw"].dropna().unique())
        testable = [g for g in gws if g > min_train_gws]
        if not testable:
            log.warning("season %s has no gameweek past the %d-GW warmup", s, min_train_gws)
            continue

        for gw in testable:
            if train_on_prior_seasons:
                train = df[(df["season"] < s) | ((df["season"] == s) & (df["gw"] < gw))]
            else:
                train = season_rows[season_rows["gw"] < gw]
            test = season_rows[season_rows["gw"] == gw]
            if test.empty:
                continue
            # Labels must be final; an unplayed fixture cannot be scored.
            test = test[test[TARGET].notna()]
            if test.empty:
                continue

            for name, predictor in predictors.items():
                predictor.fit(train, feature_cols)
                preds = np.asarray(predictor.predict(test), dtype=float)
                out.append(
                    pd.DataFrame(
                        {
                            "model": name,
                            "season": test["season"].to_numpy(),
                            "gw": test["gw"].to_numpy(),
                            "player_id": test["player_id"].to_numpy(),
                            "fixture_id": test["fixture_id"].to_numpy(),
                            "position": test["position"].to_numpy(),
                            "minutes_r3": test["minutes_r3"].to_numpy(),
                            PRED: preds,
                            ACTUAL: test[TARGET].to_numpy(dtype=float),
                        }
                    )
                )

    if not out:
        return pd.DataFrame(
            columns=["model", "season", "gw", "player_id", "fixture_id", "position", PRED, ACTUAL]
        )
    return pd.concat(out, ignore_index=True)


def score(predictions: pd.DataFrame, playing_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score raw predictions. Returns `(per_gameweek, summary)`."""
    rows = predictions
    if playing_only:
        rows = rows[playing_filter(rows)]

    per_gw: list[dict] = []
    for (model, season, gw), grp in rows.groupby(["model", "season", "gw"], sort=True, observed=True):
        # A model with no figure for this gameweek is skipped rather than
        # scored as though it had confidently predicted zero.
        grp = grp[grp[PRED].notna()]
        if grp.empty:
            continue
        gw_grain = to_gameweek_grain(grp)
        per_gw.append({"model": model, "season": season, "gw": int(gw), **score_gameweek(gw_grain)})

    per_gw_df = pd.DataFrame(per_gw)
    if per_gw_df.empty:
        return per_gw_df, pd.DataFrame()

    summary = (
        per_gw_df.groupby("model", observed=True)
        .apply(summarise, include_groups=False)
        .sort_values("mae")
    )
    return per_gw_df, summary


def run_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    predictors: dict[str, Predictor] | None = None,
    season: str | None = None,
    min_train_gws: int = 4,
) -> dict[str, pd.DataFrame]:
    """Backtest end to end, scored on both the full pool and likely starters."""
    predictions = walk_forward(
        df,
        feature_cols,
        predictors=predictors,
        season=season,
        min_train_gws=min_train_gws,
    )
    per_gw_all, summary_all = score(predictions, playing_only=False)
    per_gw_play, summary_play = score(predictions, playing_only=True)
    return {
        "predictions": predictions,
        "per_gw_all": per_gw_all,
        "summary_all": summary_all,
        "per_gw_playing": per_gw_play,
        "summary_playing": summary_play,
    }
