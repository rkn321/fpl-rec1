"""Scoring metrics for the backtest.

Two things are measured, because they answer different questions (brief §7):

* **MAE / RMSE** — how wrong is the points estimate.
* **Spearman rank correlation within position** — does the model *order* players
  correctly. Squad selection only ever compares a midfielder to a midfielder, so
  ordering inside a position is what actually drives decisions; a model can win
  on MAE by predicting everyone near zero and still be useless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRED = "prediction"
ACTUAL = "actual"


def mae(actual: pd.Series, pred: pd.Series) -> float:
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(pred, dtype=float))))


def rmse(actual: pd.Series, pred: pd.Series) -> float:
    diff = np.asarray(actual, dtype=float) - np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def spearman(actual: pd.Series, pred: pd.Series) -> float:
    """Rank correlation; NaN when either side is constant (no ordering to get right).

    Computed as Pearson on average-ranked values, which is the definition of
    Spearman's rho and handles the heavy ties in FPL points (a third of the
    pool scores exactly 0) correctly. Done by hand so the package doesn't need
    scipy just for this.
    """
    a = pd.Series(np.asarray(actual, dtype=float))
    p = pd.Series(np.asarray(pred, dtype=float))
    if len(a) < 3 or a.nunique() < 2 or p.nunique() < 2:
        return float("nan")
    return float(a.rank(method="average").corr(p.rank(method="average")))


def spearman_by_position(df: pd.DataFrame, position_col: str = "position") -> float:
    """Mean within-position rank correlation, weighted by players in each position."""
    scores, weights = [], []
    for _, grp in df.groupby(position_col, observed=True):
        s = spearman(grp[ACTUAL], grp[PRED])
        if not np.isnan(s):
            scores.append(s)
            weights.append(len(grp))
    if not scores:
        return float("nan")
    return float(np.average(scores, weights=weights))


def score_gameweek(df: pd.DataFrame) -> dict[str, float]:
    """All metrics for one gameweek's predictions."""
    return {
        "n": int(len(df)),
        "mae": mae(df[ACTUAL], df[PRED]),
        "rmse": rmse(df[ACTUAL], df[PRED]),
        "spearman_overall": spearman(df[ACTUAL], df[PRED]),
        "spearman_by_position": spearman_by_position(df),
    }


def summarise(per_gw: pd.DataFrame) -> pd.Series:
    """Collapse per-gameweek metrics to one row per model.

    MAE and RMSE are weighted by row count so a small gameweek doesn't count as
    much as a full one; the rank correlations are averaged unweighted, since
    each gameweek is one independent test of ordering.
    """
    n = per_gw["n"].to_numpy(dtype=float)
    return pd.Series(
        {
            "gameweeks": int(len(per_gw)),
            # How many gameweeks actually produced a rank correlation. A model
            # whose data is missing for most of a season scores its Spearman on
            # the handful that remain, which is not comparable to one measured
            # on all of them — so the count travels with the number.
            "ranked_gws": int(per_gw["spearman_by_position"].notna().sum()),
            "rows": int(n.sum()),
            "mae": float(np.average(per_gw["mae"], weights=n)),
            "rmse": float(np.average(per_gw["rmse"], weights=n)),
            "spearman_overall": float(per_gw["spearman_overall"].mean(skipna=True)),
            "spearman_by_position": float(per_gw["spearman_by_position"].mean(skipna=True)),
        }
    )
