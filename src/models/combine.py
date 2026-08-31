"""Combine component predictions into expected points (Phase 3).

This is why the brief insists on components rather than one regression on total
points: the position multipliers and the DEFCON and clean-sheet thresholds
behave so differently that a single model fits all of them badly. Each component
predicts something with its own natural shape, and the scoring rules — the one
part of this problem that is known exactly — do the combining.

Everything here is an expectation, and expectations add. The subtlety is *which*
playing-time probability each term needs:

    appearance      p(play) + p(60), because the second point is a second cliff
    goals, assists  scaled by expected minutes, done in the attack model
    clean sheet     p(60), not E[minutes] — it is a hard 60-minute threshold
    goals conceded  p(60) for the same reason
    DEFCON, saves   p(60), applied in the defence model
    cards           p(play), and modelled as a rate rather than predicted

Cards, own goals and penalties are small, close to irreducible noise, and would
each need their own model to move the needle by a tenth of a point. They are
taken as historical per-appearance rates by position instead, which is stated
here rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import (
    APPEARANCE_POINTS,
    ASSIST_POINTS,
    CLEAN_SHEET_POINTS,
    CONCEDES_GOALS,
    DEFCON_POINTS,
    GOAL_POINTS,
    RED_CARD_POINTS,
    YELLOW_CARD_POINTS,
)


def combine(
    frame: pd.DataFrame,
    minutes: pd.DataFrame,
    attack: pd.DataFrame,
    defence: pd.DataFrame,
    bonus: pd.DataFrame,
    card_rates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expected points per player-fixture, with every term kept separate.

    The breakdown is returned alongside the total on purpose: a number you
    cannot decompose is a number you cannot argue with, and "why is he rated
    highly" should be answerable.
    """
    position = frame["position"]

    p_play = minutes["p_play"].to_numpy()
    p_60 = minutes["p_60"].to_numpy()

    # 1 point for turning up, a second for reaching 60.
    appearance = APPEARANCE_POINTS * p_play + APPEARANCE_POINTS * p_60

    goal_value = position.map(GOAL_POINTS).fillna(0).to_numpy()
    goals = attack["expected_goals"].to_numpy() * goal_value
    assists = attack["expected_assists"].to_numpy() * ASSIST_POINTS

    cs_value = position.map(CLEAN_SHEET_POINTS).fillna(0).to_numpy()
    clean_sheet = defence["p_clean_sheet"].to_numpy() * p_60 * cs_value

    concedes = position.isin(CONCEDES_GOALS).to_numpy()
    conceded = np.where(concedes, -defence["conceded_penalty"].to_numpy() * p_60, 0.0)

    saves = defence["expected_save_points"].to_numpy()
    defcon = defence["p_defcon"].to_numpy() * DEFCON_POINTS
    bonus_points = bonus["expected_bonus"].to_numpy()

    if card_rates is not None:
        yellows = frame["position"].map(card_rates["yellow"]).fillna(0).to_numpy()
        reds = frame["position"].map(card_rates["red"]).fillna(0).to_numpy()
        cards = (yellows * YELLOW_CARD_POINTS + reds * RED_CARD_POINTS) * p_play
    else:
        cards = np.zeros(len(frame))

    total = appearance + goals + assists + clean_sheet + conceded + saves + defcon + bonus_points + cards

    return pd.DataFrame(
        {
            "appearance": appearance,
            "goals": goals,
            "assists": assists,
            "clean_sheet": clean_sheet,
            "goals_conceded": conceded,
            "saves": saves,
            "defcon": defcon,
            "bonus": bonus_points,
            "cards": cards,
            "expected_points": total,
        },
        index=frame.index,
    )


def card_rates_from(train: pd.DataFrame) -> pd.DataFrame:
    """Per-appearance card rates by position.

    Not worth a model: cards are rare, weakly predictable from anything in this
    feature frame, and worth at most a point. A positional base rate captures
    the part that is real — defenders and defensive midfielders are booked more
    than forwards — without pretending to more.
    """
    played = train[train["minutes"] > 0]
    if played.empty:
        return pd.DataFrame({"yellow": {}, "red": {}})
    grouped = played.groupby("position", observed=True)
    return pd.DataFrame(
        {
            "yellow": grouped["yellow_cards"].mean(),
            "red": grouped["red_cards"].mean(),
        }
    )
