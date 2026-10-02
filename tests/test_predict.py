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
