"""Shared gradient-boosting plumbing for the component models.

Every component is the same shape of problem — a tabular regression or
classification over the lagged feature frame — so the estimator setup lives
here rather than being repeated five times.

Defaults are deliberately conservative. The training set is ~30k rows of a very
noisy target; a deep, hungry model memorises which players had lucky weeks. The
regularisation below was not tuned to death, and is documented as such.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SEED = 7

# Small trees, strong row/column subsampling, and a floor on leaf size. FPL
# points are heavy-tailed: without a leaf minimum the model happily carves out
# a leaf for one 20-point haul.
BASE_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 60,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}


class Component:
    """One fitted target: a LightGBM model plus the columns it was trained on.

    Holds a constant fallback so a component that cannot be trained — a target
    absent from the data, or a class that never occurs — still answers with
    something defensible rather than raising into the middle of a backtest.
    """

    def __init__(
        self,
        objective: str = "regression",
        classifier: bool = False,
        params: dict[str, Any] | None = None,
        fallback: float = 0.0,
        name: str = "component",
    ):
        self.objective = objective
        self.classifier = classifier
        self.params = {**BASE_PARAMS, **(params or {})}
        self.fallback = fallback
        self.name = name
        self.model: lgb.LGBMModel | None = None
        self.features: list[str] = []
        self.trained_on = 0

    def fit(
        self,
        frame: pd.DataFrame,
        features: Sequence[str],
        target: str | pd.Series,
        weight: pd.Series | None = None,
        min_rows: int = 400,
    ) -> "Component":
        y = frame[target] if isinstance(target, str) else target
        mask = y.notna()
        if mask.sum() < min_rows:
            log.debug("%s: only %d usable rows, using fallback", self.name, int(mask.sum()))
            self.model = None
            self.fallback = float(y[mask].mean()) if mask.any() else self.fallback
            return self

        X = frame.loc[mask, list(features)]
        y = y[mask].astype(float)

        # A target that never varies has nothing to learn; predict the constant.
        if y.nunique() < 2:
            self.model = None
            self.fallback = float(y.iloc[0])
            return self

        cls = lgb.LGBMClassifier if self.classifier else lgb.LGBMRegressor
        params = dict(self.params)
        if not self.classifier:
            params["objective"] = self.objective

        self.model = cls(**params)
        self.model.fit(X, y, sample_weight=None if weight is None else weight[mask])
        self.features = list(features)
        self.trained_on = int(mask.sum())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(frame), self.fallback, dtype=float)
        X = frame[self.features]
        if self.classifier:
            proba = self.model.predict_proba(X)
            return proba[:, 1].astype(float)
        return np.asarray(self.model.predict(X), dtype=float)


def recency_weights(frame: pd.DataFrame, half_life_gws: float = 40.0) -> pd.Series:
    """Weight recent player-fixtures more heavily, with a gentle half-life.

    Roles and prices change across seasons, so old rows should count for less —
    but not by much, since throwing away history is what leaves the model with
    two gameweeks of noise. Ordering is by absolute gameweek across seasons.
    """
    seasons = sorted(frame["season"].unique())
    season_index = {s: i for i, s in enumerate(seasons)}
    absolute = frame["season"].map(season_index) * 38 + frame["gw"].astype(float)
    age = absolute.max() - absolute
    return pd.Series(0.5 ** (age / half_life_gws), index=frame.index)


def poisson_tail_expectation(lmbda: np.ndarray, per: int = 2, cap: int = 24) -> np.ndarray:
    """E[floor(X / per)] for X ~ Poisson(lambda).

    Goals conceded pays -1 per *two* conceded, which is a step function, so the
    mean alone is not enough — E[floor(X/2)] is not floor(E[X]/2). Summed over
    the pmf rather than approximated.

    `cap` truncates the tail. At 24 the neglected mass is below 1e-12 for any
    plausible number of goals conceded, so the truncation is not a source of
    error worth thinking about.
    """
    lmbda = np.clip(np.asarray(lmbda, dtype=float), 1e-9, None)
    total = np.zeros_like(lmbda)
    pmf = np.exp(-lmbda)          # P(X = 0)
    for k in range(cap + 1):
        if k > 0:
            pmf = pmf * lmbda / k
        total += (k // per) * pmf
    return total
