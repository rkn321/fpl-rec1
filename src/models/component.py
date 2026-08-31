"""The component model, assembled and wired into the evaluation harness.

Predicts each scoring component separately and combines them through the
2026/27 rules, rather than regressing total points in one shot.

The ordering matters: minutes are predicted first and fed to everything else, so
playing time is estimated in exactly one place instead of being relearned by
each component from the same features.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from .attack import AttackModel
from .baselines import Predictor
from .bonus import BonusModel
from .combine import card_rates_from, combine
from .defence import DefenceModel
from .minutes import MinutesModel

log = logging.getLogger(__name__)


class ComponentModel(Predictor):
    """Gradient-boosted component models combined via the scoring rules."""

    name = "component"

    def __init__(self, use_cards: bool = True):
        self.minutes = MinutesModel()
        self.attack = AttackModel()
        self.defence = DefenceModel()
        self.bonus = BonusModel()
        self.use_cards = use_cards
        self.card_rates: pd.DataFrame | None = None
        self.breakdown: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> "ComponentModel":
        # Rows with no outcome cannot teach anything: future fixtures ride along
        # in the frame so their features get built the same way.
        train = train[train["total_points"].notna()]
        if train.empty:
            return self

        features = [c for c in feature_cols if c in train.columns]

        self.minutes.fit(train, features)
        self.attack.fit(train, features)
        self.defence.fit(train, features)
        self.bonus.fit(train, features)
        self.card_rates = card_rates_from(train) if self.use_cards else None
        return self

    def predict_breakdown(self, test: pd.DataFrame) -> pd.DataFrame:
        """Expected points with every scoring term kept separate."""
        minutes = self.minutes.predict(test)
        attack = self.attack.predict(test, minutes["expected_minutes"].to_numpy())
        defence = self.defence.predict(test, minutes["p_60"].to_numpy())
        bonus = self.bonus.predict(test, minutes["p_play"].to_numpy())

        breakdown = combine(test, minutes, attack, defence, bonus, self.card_rates)
        return pd.concat([minutes, breakdown], axis=1)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        self.breakdown = self.predict_breakdown(test)
        return self.breakdown["expected_points"].to_numpy(dtype=float)
