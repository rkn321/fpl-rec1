"""Tests for the walk-forward harness and metrics.

The harness is the thing that decides whether a model is any good, so its own
correctness matters more than most: a backtest that quietly trains on the future
would report a great model that loses money.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import to_gameweek_grain, walk_forward
from src.features.build import build_features
from src.metrics import ACTUAL, PRED, mae, rmse, spearman, spearman_by_position
from src.models.baselines import Predictor, default_baselines


class RecordingPredictor(Predictor):
    """Notes the gameweeks it was trained on, so the harness can be audited."""

    name = "recorder"

    def __init__(self) -> None:
        self.train_gws: list[set[int]] = []
        self.test_gws: list[set[int]] = []

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> "RecordingPredictor":
        self.train_gws.append(set(train["gw"].dropna().astype(int)))
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        self.test_gws.append(set(test["gw"].dropna().astype(int)))
        return np.zeros(len(test))


def test_walk_forward_never_trains_on_the_test_gameweek(synthetic: pd.DataFrame) -> None:
    built, feature_cols = build_features(synthetic)
    rec = RecordingPredictor()

    walk_forward(built, feature_cols, predictors={"recorder": rec}, min_train_gws=2)

    assert rec.train_gws, "harness produced no folds"
    for train_gws, test_gws in zip(rec.train_gws, rec.test_gws):
        test_gw = min(test_gws)
        assert all(g < test_gw for g in train_gws), (
            f"fold predicting gw {test_gw} trained on {sorted(train_gws)}"
        )


def test_walk_forward_respects_the_warmup(synthetic: pd.DataFrame) -> None:
    built, feature_cols = build_features(synthetic)
    preds = walk_forward(built, feature_cols, predictors=default_baselines(), min_train_gws=4)
    assert preds["gw"].min() == 5


def test_gameweek_grain_sums_a_double(synthetic: pd.DataFrame) -> None:
    """Two fixtures in one gameweek must collapse into one summed row."""
    rows = pd.DataFrame(
        {
            "season": ["2099-00"] * 3,
            "player_id": [1, 1, 2],
            "gw": [5, 5, 5],
            "fixture_id": [10, 11, 10],
            "position": ["MID", "MID", "DEF"],
            PRED: [2.0, 3.0, 1.0],
            ACTUAL: [6.0, 1.0, 2.0],
        }
    )
    out = to_gameweek_grain(rows)

    assert len(out) == 2
    doubled = out[out["player_id"] == 1].iloc[0]
    assert doubled[PRED] == pytest.approx(5.0)
    assert doubled[ACTUAL] == pytest.approx(7.0)
    assert doubled["fixtures"] == 2


def test_metrics_on_known_numbers() -> None:
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    pred = pd.Series([1.0, 3.0, 3.0, 6.0])

    assert mae(actual, pred) == pytest.approx(0.75)
    assert rmse(actual, pred) == pytest.approx(np.sqrt((0 + 1 + 0 + 4) / 4))

    # Strictly monotone predictions rank perfectly...
    assert spearman(actual, pd.Series([1.0, 2.5, 3.1, 9.0])) == pytest.approx(1.0)
    # ...while the tie at 3.0 above costs a little, as it should.
    assert spearman(actual, pred) == pytest.approx(0.9486832980505139)


def test_spearman_handles_ties_and_constants() -> None:
    # A third of the FPL pool scores exactly 0, so ties are the normal case and
    # a gameweek where nobody scored has no ordering to get right.
    assert np.isnan(spearman(pd.Series([0.0, 0.0, 0.0, 0.0]), pd.Series([1.0, 2.0, 3.0, 4.0])))
    assert np.isnan(spearman(pd.Series([1.0, 1.0, 1.0]), pd.Series([1.0, 2.0, 3.0])))

    # Heavy ties in the actuals still leave a usable signal.
    tied = spearman(pd.Series([0.0, 0.0, 0.0, 6.0]), pd.Series([1.0, 2.0, 3.0, 9.0]))
    assert 0.0 < tied <= 1.0

    perfectly_inverted = spearman(pd.Series([1.0, 2.0, 3.0]), pd.Series([3.0, 2.0, 1.0]))
    assert perfectly_inverted == pytest.approx(-1.0)


def test_spearman_by_position_weights_by_group_size() -> None:
    df = pd.DataFrame(
        {
            "position": ["MID"] * 4 + ["DEF"] * 4,
            ACTUAL: [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            PRED: [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    # One position perfectly ordered, the other perfectly inverted -> mean 0.
    assert spearman_by_position(df) == pytest.approx(0.0)


def test_baselines_produce_finite_predictions(synthetic: pd.DataFrame) -> None:
    built, feature_cols = build_features(synthetic)
    preds = walk_forward(built, feature_cols, predictors=default_baselines(), min_train_gws=3)

    assert len(preds) > 0
    assert np.isfinite(preds[PRED]).all()
    assert set(preds["model"]) == set(default_baselines())
