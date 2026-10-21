"""The one prediction path, and the copy of it that `fpl review` scores.

What matters: a double gameweek's points add while its probabilities do not;
the figure decomposes into terms that sum to it; a prediction is stored before
its deadline and never overwritten after; and what is stored reads back intact.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features.build import build_features
from src.models.combine import POINTS_TERMS
from src.predict import fit_predict, load_predictions, prediction_record, predictions_path, save_predictions

DEADLINE = pd.Timestamp("2099-09-18 17:30", tz="UTC")


@pytest.fixture
def config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""
        season:
          current: "2099-00"
          history: []
        paths:
          data_dir: "{(tmp_path / 'data').as_posix()}"
          cache_dir: "{(tmp_path / 'data' / 'cache').as_posix()}"
          processed_dir: "{(tmp_path / 'data' / 'processed').as_posix()}"
        api: {{base_url: "x", ttl: {{default: 1}}}}
        historical: {{vaastav_base: "y"}}
        features: {{windows: [3]}}
    """), encoding="utf-8")
    return load_config(cfg, use_local=False)


class FixedBreakdown:
    """A predictor whose breakdown is one point per term per fixture."""

    def predict_breakdown(self, target: pd.DataFrame) -> pd.DataFrame:
        n = len(target)
        out = pd.DataFrame({t: np.ones(n) for t in POINTS_TERMS}, index=target.index)
        out["expected_points"] = float(len(POINTS_TERMS))
        out["p_play"] = [0.9, 0.6, 0.8][:n]
        out["p_60"] = [0.7, 0.5, 0.6][:n]
        out["expected_minutes"] = [70.0, 50.0, 65.0][:n]
        return out


def target_rows() -> pd.DataFrame:
    """Player 1 has a double; player 2 a single."""
    return pd.DataFrame({
        "player_id": [1, 1, 2], "name": ["One", "One", "Two"], "position": ["MID", "MID", "FWD"],
        "team_id": [5, 5, 6], "value": [80, 80, 60], "fixture_id": [11, 12, 13],
        "xP": [7.5, 7.5, 3.0], "minutes_r3": [90.0, 90.0, 30.0],
    })


def test_a_double_adds_points_and_minutes_but_not_probabilities() -> None:
    record = prediction_record(target_rows(), FixedBreakdown())
    one = record.loc[1]
    assert one["fixtures"] == 2
    assert one["expected_points"] == 2 * len(POINTS_TERMS)
    assert one["goals"] == 2 and one["bonus"] == 2
    assert one["expected_minutes"] == 120
    assert one["p_play"] == 0.9 and one["p_60"] == 0.7, "the larger fixture's chance, not the sum"
    assert one["xp_fpl"] == 7.5, "FPL's figure already covers the whole gameweek"
    assert record.loc[2, "fixtures"] == 1


def test_a_model_with_no_breakdown_still_gives_a_record() -> None:
    class Flat:
        def predict(self, target):
            return np.full(len(target), 2.0)

    record = prediction_record(target_rows(), Flat())
    assert record.loc[1, "expected_points"] == 4.0
    assert "goals" not in record.columns


def test_stored_before_the_deadline_and_reads_back(config) -> None:
    record = prediction_record(target_rows(), FixedBreakdown())
    path = save_predictions(record, 5, config, "component", DEADLINE, now=DEADLINE - pd.Timedelta(hours=1))
    assert path == predictions_path(config, 5) and path.exists()

    loaded, meta = load_predictions(config, 5)
    assert meta["model"] == "component"
    assert pd.Timestamp(meta["generated_at"]) == DEADLINE - pd.Timedelta(hours=1)
    pd.testing.assert_frame_equal(loaded[record.columns], record, check_dtype=False)


def test_never_overwritten_after_the_deadline(config) -> None:
    before = prediction_record(target_rows(), FixedBreakdown())
    save_predictions(before, 5, config, "component", DEADLINE, now=DEADLINE - pd.Timedelta(hours=1))

    after = before.assign(expected_points=0.0)
    assert save_predictions(after, 5, config, "component", DEADLINE, now=DEADLINE) is None
    loaded, _ = load_predictions(config, 5)
    assert (loaded["expected_points"] > 0).all(), "the in-time prediction survives"


def test_rerunning_before_the_deadline_overwrites(config) -> None:
    early = prediction_record(target_rows(), FixedBreakdown())
    save_predictions(early, 5, config, "component", DEADLINE, now=DEADLINE - pd.Timedelta(days=1))
    late = early.assign(expected_points=1.0)
    save_predictions(late, 5, config, "component", DEADLINE, now=DEADLINE - pd.Timedelta(minutes=5))
    loaded, _ = load_predictions(config, 5)
    assert (loaded["expected_points"] == 1.0).all(), "the later run has the later team news"


def test_nothing_stored_reads_as_none(config) -> None:
    assert load_predictions(config, 9) is None


def test_fit_predict_decomposes_into_terms_that_sum(synthetic: pd.DataFrame) -> None:
    frame, features = build_features(synthetic)
    gw = 8
    record = fit_predict(frame, features, "2099-00", gw)

    expected_players = frame.loc[(frame["gw"] == gw), "player_id"].nunique()
    assert len(record) == expected_players and record.index.is_unique
    terms = record[list(POINTS_TERMS)].sum(axis=1)
    np.testing.assert_allclose(terms, record["expected_points"], atol=1e-9)
    assert record["xp_fpl"].notna().all()


def test_fit_predict_refuses_a_gameweek_with_no_rows(synthetic: pd.DataFrame) -> None:
    frame, features = build_features(synthetic)
    with pytest.raises(ValueError, match="no rows for gameweek 99"):
        fit_predict(frame, features, "2099-00", 99)


# -- the fit cache -------------------------------------------------------------------

class CountingPredictor:
    """Counts fits across instances; predicts each player's number of training rows."""

    fits = 0
    last_features: list[str] = []

    def fit(self, train, feature_cols):
        CountingPredictor.fits += 1
        CountingPredictor.last_features = list(feature_cols)
        self.rows = len(train)
        return self

    def predict(self, target):
        return np.full(len(target), float(self.rows))


@pytest.fixture
def counting(monkeypatch):
    import src.predict as predict_module

    CountingPredictor.fits = 0
    monkeypatch.setattr(predict_module, "all_predictors", lambda: {"counting": CountingPredictor()})
    return CountingPredictor


def test_an_identical_fit_comes_from_the_cache(synthetic, counting, tmp_path) -> None:
    frame, features = build_features(synthetic)
    first = fit_predict(frame, features, "2099-00", 8, model="counting", cache_dir=tmp_path)
    second = fit_predict(frame, features, "2099-00", 8, model="counting", cache_dir=tmp_path)
    assert counting.fits == 1
    pd.testing.assert_frame_equal(first, second)


def test_changed_training_rows_refit(synthetic, counting, tmp_path) -> None:
    frame, features = build_features(synthetic)
    fit_predict(frame, features, "2099-00", 8, model="counting", cache_dir=tmp_path)

    revised = frame.copy()
    earlier = revised.index[revised["gw"] < 8][0]
    revised.loc[earlier, "total_points"] += 1   # FPL corrects a past score
    fit_predict(revised, features, "2099-00", 8, model="counting", cache_dir=tmp_path)
    assert counting.fits == 2

    # The next gameweek's training window is different rows again.
    fit_predict(frame, features, "2099-00", 9, model="counting", cache_dir=tmp_path)
    assert counting.fits == 3


def test_a_corrupt_cached_model_is_refitted_over(synthetic, counting, tmp_path) -> None:
    frame, features = build_features(synthetic)
    fit_predict(frame, features, "2099-00", 8, model="counting", cache_dir=tmp_path)
    (cached,) = tmp_path.glob("counting-*.pkl")
    cached.write_bytes(b"half a pickle")

    record = fit_predict(frame, features, "2099-00", 8, model="counting", cache_dir=tmp_path)
    assert counting.fits == 2 and record["expected_points"].gt(0).all()
    fit_predict(frame, features, "2099-00", 8, model="counting", cache_dir=tmp_path)
    assert counting.fits == 2, "and the rewritten file is good"


def test_the_cache_keeps_only_the_latest_fits(synthetic, counting, tmp_path) -> None:
    from src.predict import KEEP_CACHED

    frame, features = build_features(synthetic)
    for gw in range(2, 2 + KEEP_CACHED + 2):
        fit_predict(frame, features, "2099-00", gw, model="counting", cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.pkl"))) == KEEP_CACHED


def test_no_cache_dir_always_fits(synthetic, counting) -> None:
    frame, features = build_features(synthetic)
    fit_predict(frame, features, "2099-00", 8, model="counting")
    fit_predict(frame, features, "2099-00", 8, model="counting")
    assert counting.fits == 2


def test_a_week_whose_odds_are_not_out_uses_the_no_odds_fit(synthetic, counting) -> None:
    frame, features = build_features(synthetic)
    frame = frame.assign(odds_team_goals=1.4, odds_opp_goals=1.1)
    features = features + ["odds_team_goals", "odds_opp_goals"]

    fit_predict(frame, features, "2099-00", 8, model="counting")
    assert "odds_team_goals" in counting.last_features, "priced week: the odds are used"

    unpriced = frame.copy()
    unpriced.loc[unpriced["gw"] == 8, ["odds_team_goals", "odds_opp_goals"]] = np.nan
    fit_predict(unpriced, features, "2099-00", 8, model="counting")
    assert not [f for f in counting.last_features if f.startswith("odds_")], (
        "a fit with the odds would read their absence as a signal"
    )
