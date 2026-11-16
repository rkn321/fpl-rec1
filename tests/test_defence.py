"""The team-level defence models.

Two things are pinned: a clean sheet is a team event, so every player of a side must be
given the same odds of it, and where the betting market has priced the match its expected
goals, not a tree model's guess, drive the clean-sheet probability and goals conceded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import poisson_tail_expectation
from src.models.defence import DefenceModel, team_level

FEATURES = ["odds_opp_goals", "odds_team_goals", "is_home", "minutes_r3", "element_type", "team_form_team_goals_against_r3"]


def synthetic_fixtures(n: int = 800, seed: int = 1, elo: bool = False) -> pd.DataFrame:
    """Three players per team-fixture with different player-level features, one shared team outcome.

    With `elo`, goals against come from the team's rating gap rather than from the odds'
    own draw, and the odds carry the same expectation: stronger sides concede less.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        lam_opp, lam_team = rng.uniform(0.6, 2.4), rng.uniform(0.6, 2.4)
        gap = float(rng.uniform(-300, 300))
        if elo:
            lam_opp, lam_team = 1.4 * np.exp(-gap / 250), 1.4 * np.exp(gap / 250)
        against = rng.poisson(lam_opp)
        for k, (pos, form, etype) in enumerate((("GK", 90.0, 1), ("DEF", 90.0, 2), ("DEF", 12.0, 2))):
            rows.append({
                "season": "2099-00", "gw": 1 + i % 38, "team_id": i, "fixture_id": i, "player_id": 3 * i + k,
                "position": pos, "minutes": 90, "saves": rng.poisson(3) if pos == "GK" else 0,
                "penalties_saved": int(rng.random() < 0.02) if pos == "GK" else 0,
                "defensive_contribution": rng.integers(0, 14), "team_goals_against": against,
                "odds_opp_goals": lam_opp, "odds_team_goals": lam_team, "is_home": float(i % 2),
                "elo_gap": gap,
                "minutes_r3": form, "element_type": etype,
                "team_form_team_goals_against_r3": rng.uniform(0.5, 2.0),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_fixtures()


@pytest.fixture(scope="module")
def rated() -> pd.DataFrame:
    """Fixtures whose outcomes follow the Elo gap, for the weeks the market has not priced."""
    return synthetic_fixtures(n=1200, seed=3, elo=True)


NO_ODDS = [c for c in FEATURES if not c.startswith("odds_")]


def test_team_level_keeps_what_describes_a_team_and_drops_what_describes_a_player() -> None:
    cols = ["odds_opp_goals", "is_home", "team_form_team_xg_r5", "opp_form_team_goals_for_r3", "elo_gap",
            "minutes_r3", "element_type", "value", "goals_scored_r5", "penalties_order"]
    assert team_level(cols) == ["odds_opp_goals", "is_home", "team_form_team_xg_r5", "opp_form_team_goals_for_r3",
                                "elo_gap"]


def test_players_of_one_side_get_the_same_clean_sheet_odds(frame) -> None:
    train, test = frame[frame["team_id"] < 700], frame[frame["team_id"] >= 700]
    out = DefenceModel().fit(train, FEATURES).predict(test, np.ones(len(test)))
    spread = out.groupby(test["fixture_id"].to_numpy())[["p_clean_sheet", "expected_conceded"]].agg(np.ptp)
    assert (spread.to_numpy() < 1e-12).all(), "a keeper and a defender were given different odds of one team event"


def test_the_markets_expected_goals_drive_clean_sheets_and_goals_conceded(frame) -> None:
    model = DefenceModel().fit(frame, FEATURES)
    assert model.market_clean_sheet is not None
    probe = frame.iloc[[0, 0, 0]].copy()
    probe["odds_opp_goals"] = [0.8, 1.5, 2.4]
    out = model.predict(probe, np.ones(3))
    p, conceded = out["p_clean_sheet"].to_numpy(), out["expected_conceded"].to_numpy()
    assert p[0] > p[1] > p[2], "a weaker attack against us must raise the clean-sheet chance"
    assert p == pytest.approx(np.exp(-np.array([0.8, 1.5, 2.4])), abs=0.05)   # the data were Poisson draws
    assert conceded == pytest.approx([0.8, 1.5, 2.4], abs=0.2)
    assert np.diff(out["conceded_penalty"].to_numpy()).min() > 0


def test_a_row_without_odds_falls_back_to_the_team_level_trees(frame) -> None:
    model = DefenceModel().fit(frame, FEATURES)
    probe = frame.iloc[[0, 1]].copy()
    probe[["odds_opp_goals", "odds_team_goals"]] = np.nan
    out = model.predict(probe, np.ones(2))
    assert np.isfinite(out.to_numpy()).all()
    assert out["p_clean_sheet"].between(0, 1).all()


def test_saves_pay_per_three_as_a_step_not_a_third_of_the_mean(frame) -> None:
    """E[floor(X/3)] sits well below mean/3, and only a keeper is paid for saves."""
    model = DefenceModel().fit(frame, FEATURES)
    test = frame.head(30)
    out = model.predict(test, np.ones(len(test)))
    mean = np.clip(model.saves.predict(test), 0.0, None)
    keeper = (test["position"] == "GK").to_numpy()
    assert keeper.any() and (~keeper).any()

    step = poisson_tail_expectation(mean, per=3) + model.penalty_save_points
    assert out["expected_save_points"].to_numpy()[keeper] == pytest.approx(step[keeper])
    assert (out["expected_save_points"].to_numpy()[keeper] < mean[keeper] / 3).all()
    assert (out["expected_save_points"].to_numpy()[~keeper] == 0).all()


def test_a_save_payout_scales_with_the_chance_of_playing(frame) -> None:
    model = DefenceModel().fit(frame, FEATURES)
    test = frame.head(30)
    full = model.predict(test, np.ones(len(test)))["expected_save_points"].to_numpy()
    half = model.predict(test, np.full(len(test), 0.5))["expected_save_points"].to_numpy()
    assert half == pytest.approx(full / 2)


def test_penalty_saves_are_a_flat_base_rate_from_the_training_keepers(frame) -> None:
    model = DefenceModel().fit(frame, FEATURES)
    keepers = frame[frame["position"] == "GK"]
    assert model.penalty_save_points == pytest.approx(5 * keepers["penalties_saved"].mean())
    # Not a column the page or a thin frame is guaranteed to carry: absent means no term, not an error.
    bare = DefenceModel().fit(frame.drop(columns=["penalties_saved"]), FEATURES)
    assert bare.penalty_save_points == 0.0


def test_the_odds_free_fit_uses_no_market_layer(frame) -> None:
    """The page falls back to a fit without odds columns when the week's prices are not out."""
    features = [c for c in FEATURES if not c.startswith("odds_")]
    model = DefenceModel().fit(frame, features)
    assert model.market_clean_sheet is None
    out = model.predict(frame.head(6), np.ones(6))
    assert out["p_clean_sheet"].between(0, 1).all()


# -- Elo, for a match with no price ------------------------------------------------------

def _unpriced(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.assign(odds_opp_goals=np.nan, odds_team_goals=np.nan)


def test_without_a_price_the_rating_gap_sets_the_clean_sheet_and_goals_conceded(rated) -> None:
    model = DefenceModel().fit(rated, NO_ODDS)
    assert model.market_clean_sheet is None and model.elo_clean_sheet is not None
    probe = _unpriced(rated.iloc[[0, 0, 0]].copy())
    probe["elo_gap"] = [-250.0, 0.0, 250.0]
    out = model.predict(probe, np.ones(3))
    p, conceded = out["p_clean_sheet"].to_numpy(), out["expected_conceded"].to_numpy()
    assert p[0] < p[1] < p[2], "the stronger side keeps more clean sheets"
    assert conceded[0] > conceded[1] > conceded[2]
    truth = 1.4 * np.exp(np.array([250.0, 0.0, -250.0]) / 250)       # goals against, by the data's own rule
    assert p == pytest.approx(np.exp(-truth), abs=0.07)
    assert conceded == pytest.approx(truth, abs=0.25)


def test_elo_gives_every_player_of_a_side_the_same_odds_and_stays_a_probability(rated) -> None:
    train, test = rated[rated["team_id"] < 1000], _unpriced(rated[rated["team_id"] >= 1000])
    out = DefenceModel().fit(train, NO_ODDS).predict(test, np.ones(len(test)))
    spread = out.groupby(test["fixture_id"].to_numpy())[["p_clean_sheet", "expected_conceded"]].agg(np.ptp)
    assert (spread.to_numpy() < 1e-12).all()
    assert out["p_clean_sheet"].between(0, 1).all() and (out["expected_conceded"] > 0).all()


def test_a_week_the_market_priced_is_untouched_by_elo(rated) -> None:
    """The market outranks Elo, so adding the gap column cannot move a priced row."""
    with_gap = DefenceModel().fit(rated, FEATURES).predict(rated.head(60), np.ones(60))
    bare = rated.drop(columns="elo_gap")
    without = DefenceModel().fit(bare, FEATURES).predict(bare.head(60), np.ones(60))
    pd.testing.assert_frame_equal(with_gap, without)


def test_a_row_with_no_price_and_no_rating_keeps_the_trees(rated) -> None:
    model = DefenceModel().fit(rated, NO_ODDS)
    blind = _unpriced(rated.iloc[[0, 1]].copy()).assign(elo_gap=np.nan)
    trees = DefenceModel().fit(rated.drop(columns="elo_gap"), NO_ODDS).predict(
        blind.drop(columns="elo_gap"), np.ones(2)
    )
    pd.testing.assert_frame_equal(model.predict(blind, np.ones(2)), trees)


def test_no_rating_column_means_no_elo_layer(frame) -> None:
    bare = frame.drop(columns="elo_gap")
    model = DefenceModel().fit(bare, NO_ODDS)
    assert model.elo_clean_sheet is None and model.elo_conceded is None
    assert np.isfinite(model.predict(bare.head(6), np.ones(6)).to_numpy()).all()
