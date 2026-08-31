"""The three baselines every model has to beat (brief §7).

(a) `Last3Mean`      — average points over the player's last 3 fixtures.
(b) `FPLExpectedPoints` — FPL's own expected-points figure for the gameweek.
(c) `MinutesTimesPP90`  — expected minutes x season-to-date points per 90.

All three predict per *player-fixture*; doubles are handled by the evaluation
harness summing rows within a gameweek.

They share a `fit`/`predict` interface with real models so the walk-forward
harness treats them identically. `fit` is a no-op for all of them: each is a
closed-form function of already-lagged features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Predictor:
    """Minimal interface the walk-forward harness expects."""

    name = "predictor"

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> "Predictor":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class Last3Mean(Predictor):
    """Baseline (a). A player with no history yet scores 0."""

    name = "last3_mean"

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test["total_points_r3"].fillna(0.0).to_numpy(dtype=float)


class FPLExpectedPoints(Predictor):
    """Baseline (b): the game's own expected-points column.

    Caveat worth keeping in view: in the vaastav dataset this column is scraped
    around the gameweek rather than strictly at the deadline (brief §6.1), so it
    may be a slightly optimistic bar rather than a purely pre-deadline one. It
    is used only as a baseline, never as a feature.
    """

    name = "fpl_ep"

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        if "xP" not in test.columns:
            return np.zeros(len(test), dtype=float)
        return test["xP"].fillna(0.0).to_numpy(dtype=float)


class MinutesTimesPP90(Predictor):
    """Baseline (c): recent minutes x season-to-date points per 90."""

    name = "minutes_x_pp90"

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        minutes = test["minutes_r3"].fillna(0.0).to_numpy(dtype=float)
        pp90 = test["points_per_90_todate"].fillna(0.0).to_numpy(dtype=float)
        return np.clip(minutes, 0, 90) / 90.0 * pp90


class SeasonMean(Predictor):
    """Sanity floor: the player's season-to-date average points per fixture."""

    name = "season_mean"

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test["total_points_todate"].fillna(0.0).to_numpy(dtype=float)


def default_baselines() -> dict[str, Predictor]:
    """The bar for Phase 3, keyed by name."""
    preds = [Last3Mean(), FPLExpectedPoints(), MinutesTimesPP90(), SeasonMean()]
    return {p.name: p for p in preds}


def all_predictors() -> dict[str, Predictor]:
    """The baselines plus the trained component model.

    Imported lazily: the baselines must stay usable without lightgbm installed,
    and `component` imports this module.
    """
    from .component import ComponentModel

    preds = default_baselines()
    model = ComponentModel()
    preds[model.name] = model
    return preds
