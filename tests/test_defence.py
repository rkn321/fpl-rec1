"""The team-level defence models.

Two things are pinned: a clean sheet is a team event, so every player of a side must be
given the same odds of it, and where the betting market has priced the match its expected
goals, not a tree model's guess, drive the clean-sheet probability and goals conceded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.defence import DefenceModel, team_level

FEATURES = ["odds_opp_goals", "odds_team_goals", "is_home", "minutes_r3", "element_type", "team_form_team_goals_against_r3"]


def synthetic_fixtures(n: int = 800, seed: int = 1) -> pd.DataFrame:
    """Three players per team-fixture with different player-level features, one shared team outcome."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        lam_opp, lam_team = rng.uniform(0.6, 2.4), rng.uniform(0.6, 2.4)
        against = rng.poisson(lam_opp)
        for k, (pos, form, etype) in enumerate((("GK", 90.0, 1), ("DEF", 90.0, 2), ("DEF", 12.0, 2))):
            rows.append({
                "season": "2099-00", "gw": 1 + i % 38, "team_id": i, "fixture_id": i, "player_id": 3 * i + k,
                "position": pos, "minutes": 90, "saves": rng.poisson(3) if pos == "GK" else 0,
                "defensive_contribution": rng.integers(0, 14), "team_goals_against": against,
                "odds_opp_goals": lam_opp, "odds_team_goals": lam_team, "is_home": float(i % 2),
                "minutes_r3": form, "element_type": etype,
                "team_form_team_goals_against_r3": rng.uniform(0.5, 2.0),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_fixtures()


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


def test_the_odds_free_fit_uses_no_market_layer(frame) -> None:
    """The page falls back to a fit without odds columns when the week's prices are not out."""
    features = [c for c in FEATURES if not c.startswith("odds_")]
    model = DefenceModel().fit(frame, features)
    assert model.market_clean_sheet is None
    out = model.predict(frame.head(6), np.ones(6))
    assert out["p_clean_sheet"].between(0, 1).all()
