"""Tests for the Phase 3 component models.

The pieces worth pinning down are the ones where a plausible-looking shortcut
is wrong: the step functions in the scoring rules, the places playing time has
to be applied, and the fact that combining is arithmetic on expectations rather
than a second model.

These run on the synthetic season, so they are fast and need no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import build_features
from src.models.base import poisson_tail_expectation
from src.models.combine import card_rates_from, combine
from src.models.component import ComponentModel
from src.models.scoring import ASSIST_POINTS, GOAL_POINTS


@pytest.fixture(scope="module")
def built(synthetic: pd.DataFrame):
    frame, features = build_features(synthetic)
    return frame, features


def test_poisson_step_expectation_is_not_the_floor_of_the_mean() -> None:
    """-1 per two conceded is a step function, so E[floor(X/2)] != floor(E[X]/2).

    Taking the mean first is the obvious shortcut and it is wrong: a team
    expected to concede 1.0 still concedes two or more about a quarter of the
    time, which the floor of the mean prices at zero.
    """
    # A team expected to concede 1.0 still loses 0.28 points to the rule on
    # average, because it concedes two or more about a quarter of the time.
    # Taking the floor of the mean prices that at exactly zero.
    assert poisson_tail_expectation(np.array([1.0]), per=2)[0] == pytest.approx(0.2838, abs=1e-3)
    assert np.floor(1.0 / 2) == 0.0

    # Monotone, and matches a brute-force sum over the pmf.
    lam = np.array([0.5, 1.0, 2.0, 3.0])
    got = poisson_tail_expectation(lam, per=2)
    assert np.all(np.diff(got) > 0)

    from math import exp, factorial

    for i, l in enumerate(lam):
        brute = sum((k // 2) * exp(-l) * l**k / factorial(k) for k in range(40))
        assert got[i] == pytest.approx(brute, abs=1e-6)


def test_model_trains_and_predicts_sane_points(built) -> None:
    frame, features = built
    train = frame[frame["gw"] <= 7]
    test = frame[frame["gw"] == 8]

    model = ComponentModel().fit(train, features)
    breakdown = model.predict_breakdown(test)

    assert len(breakdown) == len(test)
    assert np.isfinite(breakdown["expected_points"]).all()
    # Nobody is expected to score 40, and expected points are rarely negative:
    # only cards and goals conceded push down, and both are small.
    assert breakdown["expected_points"].between(-2, 25).all()
    assert (breakdown["p_play"].between(0, 1)).all()
    assert (breakdown["p_60"].between(0, 1)).all()


def test_probability_of_sixty_never_exceeds_probability_of_playing(built) -> None:
    """The two classifiers are fitted separately and can disagree at the margin."""
    frame, features = built
    model = ComponentModel().fit(frame[frame["gw"] <= 7], features)
    out = model.predict_breakdown(frame[frame["gw"] == 8])

    assert (out["p_60"] <= out["p_play"] + 1e-9).all()


def test_expected_minutes_respects_playing_probability(built) -> None:
    frame, features = built
    model = ComponentModel().fit(frame[frame["gw"] <= 7], features)
    out = model.predict_breakdown(frame[frame["gw"] == 8])

    assert (out["expected_minutes"] >= -1e-9).all()
    assert (out["expected_minutes"] <= 90 + 1e-9).all()
    # A player who will not play cannot accumulate minutes.
    idle = out[out["p_play"] < 0.02]
    if len(idle):
        assert (idle["expected_minutes"] < 10).all()


def test_combine_applies_position_multipliers() -> None:
    """A goal is worth 6 to a defender and 4 to a forward; combining must know."""
    frame = pd.DataFrame(
        {
            "position": ["DEF", "FWD"],
            "season": ["2099-00"] * 2,
            "fixture_id": [1, 1],
        }
    )
    minutes = pd.DataFrame({"p_play": [1.0, 1.0], "p_60": [1.0, 1.0], "expected_minutes": [90, 90]})
    attack = pd.DataFrame({"expected_goals": [1.0, 1.0], "expected_assists": [1.0, 1.0]})
    defence = pd.DataFrame(
        {
            "p_clean_sheet": [0.0, 0.0],
            "expected_conceded": [0.0, 0.0],
            "conceded_penalty": [0.0, 0.0],
            "expected_save_points": [0.0, 0.0],
            "p_defcon": [0.0, 0.0],
        }
    )
    bonus = pd.DataFrame({"expected_bps": [0.0, 0.0], "expected_bonus": [0.0, 0.0]})

    out = combine(frame, minutes, attack, defence, bonus)

    assert out["goals"].tolist() == [GOAL_POINTS["DEF"], GOAL_POINTS["FWD"]]
    assert out["assists"].tolist() == [ASSIST_POINTS, ASSIST_POINTS]
    # 2 appearance points each, since p_play and p_60 are both 1.
    assert out["appearance"].tolist() == [2.0, 2.0]


def test_clean_sheet_needs_sixty_minutes_not_expected_minutes() -> None:
    """Someone certain to play but unlikely to last 60 earns almost no clean sheet."""
    frame = pd.DataFrame({"position": ["DEF"], "season": ["2099-00"], "fixture_id": [1]})
    minutes = pd.DataFrame({"p_play": [1.0], "p_60": [0.1], "expected_minutes": [45.0]})
    attack = pd.DataFrame({"expected_goals": [0.0], "expected_assists": [0.0]})
    defence = pd.DataFrame(
        {
            "p_clean_sheet": [1.0],
            "expected_conceded": [0.0],
            "conceded_penalty": [0.0],
            "expected_save_points": [0.0],
            "p_defcon": [0.0],
        }
    )
    bonus = pd.DataFrame({"expected_bps": [0.0], "expected_bonus": [0.0]})

    out = combine(frame, minutes, attack, defence, bonus)

    # A certain team clean sheet, but only a 10% chance of being on for it.
    assert out["clean_sheet"].iloc[0] == pytest.approx(0.4)


def test_only_keepers_and_defenders_are_docked_for_goals_conceded() -> None:
    frame = pd.DataFrame(
        {"position": ["GK", "DEF", "MID", "FWD"], "season": ["2099-00"] * 4, "fixture_id": [1] * 4}
    )
    n = len(frame)
    minutes = pd.DataFrame({"p_play": [1.0] * n, "p_60": [1.0] * n, "expected_minutes": [90.0] * n})
    attack = pd.DataFrame({"expected_goals": [0.0] * n, "expected_assists": [0.0] * n})
    defence = pd.DataFrame(
        {
            "p_clean_sheet": [0.0] * n,
            "expected_conceded": [2.0] * n,
            "conceded_penalty": [1.0] * n,
            "expected_save_points": [0.0] * n,
            "p_defcon": [0.0] * n,
        }
    )
    bonus = pd.DataFrame({"expected_bps": [0.0] * n, "expected_bonus": [0.0] * n})

    out = combine(frame, minutes, attack, defence, bonus)

    assert out["goals_conceded"].tolist() == [-1.0, -1.0, 0.0, 0.0]


def test_card_rates_are_per_appearance_by_position(built) -> None:
    frame, _ = built
    rates = card_rates_from(frame)

    assert set(rates.columns) == {"yellow", "red"}
    assert (rates["yellow"] >= 0).all()
    assert (rates["yellow"] <= 1).all()


def test_breakdown_terms_sum_to_the_total(built) -> None:
    """The total has to be the sum of its parts, or the breakdown is decoration."""
    frame, features = built
    model = ComponentModel().fit(frame[frame["gw"] <= 7], features)
    out = model.predict_breakdown(frame[frame["gw"] == 8])

    parts = out[
        ["appearance", "goals", "assists", "clean_sheet", "goals_conceded",
         "saves", "defcon", "bonus", "cards"]
    ].sum(axis=1)
    pd.testing.assert_series_equal(parts, out["expected_points"], check_names=False)
