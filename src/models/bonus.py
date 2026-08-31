"""Bonus points (Phase 3).

Bonus is a *within-match tournament*, not a per-player quantity: the top three
BPS scores in each fixture take 3 / 2 / 1, shared on ties. So a player's bonus
depends on who else played in that match, and predicting it directly from their
own features is asking the wrong question — a 30-BPS game wins bonus in a quiet
match and wins nothing in a wild one.

Two steps instead:

1. Predict BPS per player-fixture, which *is* a per-player quantity.
2. Convert predicted BPS into expected bonus by position within its fixture.

Step 2 uses an empirical table learned from the model's own ranks on the
training data rather than the idealised 3/2/1. That matters: with perfect
ranking, first place would be worth 3.0, but our ranking is imperfect, and the
table absorbs exactly that error. The player we rank first in a fixture is
historically worth well under three points, which is the honest number.

The table is built in-sample, so it is mildly optimistic. Bonus is capped at 3
points, so the size of that optimism is bounded.

For 2026/27 the BPS weights were rebalanced to reduce overlap with DEFCON, now
favouring attackers and full-backs (brief §4), so coefficients learned on older
seasons are stale. Recency weighting handles some of this; a season indicator
would handle more.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .base import Component, recency_weights

# Ranks beyond this never earn bonus, so there is nothing to learn about them.
MAX_RANK = 5


class BonusModel:
    """Expected bonus points, via predicted BPS ranked within each fixture."""

    def __init__(self) -> None:
        self.bps = Component(fallback=0.0, name="bps")
        # rank within fixture -> mean bonus actually awarded
        self.rank_to_bonus: dict[int, float] = {}
        self.default_bonus = 0.0

    @staticmethod
    def _rank_within_fixture(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
        """Dense rank of `values` inside each fixture, 1 = highest."""
        scored = pd.Series(values, index=frame.index)
        return (
            scored.groupby([frame["season"], frame["fixture_id"]], observed=True)
            .rank(ascending=False, method="min")
            .to_numpy()
        )

    def fit(self, train: pd.DataFrame, features: Sequence[str]) -> "BonusModel":
        played = train[train["minutes"] > 0]
        if played.empty:
            return self

        self.bps.fit(played, features, played["bps"].astype(float), recency_weights(played))

        # What our own ranking is worth, rather than what a perfect one would be.
        predicted = self.bps.predict(played)
        ranks = self._rank_within_fixture(played, predicted)
        bonus = played["bonus"].astype(float).to_numpy()

        table = pd.DataFrame({"rank": ranks, "bonus": bonus})
        means = table[table["rank"] <= MAX_RANK].groupby("rank")["bonus"].mean()
        self.rank_to_bonus = {int(r): float(v) for r, v in means.items()}
        self.default_bonus = float(table.loc[table["rank"] > MAX_RANK, "bonus"].mean() or 0.0)
        return self

    def predict(self, test: pd.DataFrame, p_play: np.ndarray) -> pd.DataFrame:
        predicted_bps = self.bps.predict(test)
        ranks = self._rank_within_fixture(test, predicted_bps)

        expected = np.array(
            [self.rank_to_bonus.get(int(r), self.default_bonus) for r in ranks], dtype=float
        )
        # Bonus needs minutes on the pitch like everything else.
        expected = expected * np.clip(np.asarray(p_play, dtype=float), 0.0, 1.0)

        return pd.DataFrame(
            {"expected_bps": predicted_bps, "expected_bonus": expected}, index=test.index
        )
