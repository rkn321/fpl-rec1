"""Goals and assists (Phase 3).

Modelled as *rates per 90*, then scaled by expected minutes, rather than as
counts per fixture. Two reasons, both practical:

* The rate is the stable quantity. Realised goals are a Poisson draw around it,
  and a model fitted on counts spends its capacity explaining variance that is
  irreducible.
* It keeps playing time in exactly one place. Minutes are already modelled; a
  count model would have to relearn them from the same features and would then
  disagree with the minutes component.

Both use a Poisson objective. The target is a non-negative rate with variance
growing in the mean, which is what Poisson deviance is for — squared error would
weight a 3-goal haul about nine times too heavily.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .base import Component, recency_weights

# Minutes below this make a per-90 rate meaningless: a goal in a 4-minute cameo
# is 22.5 per 90, which is not a rate, it is an artefact.
MIN_MINUTES_FOR_RATE = 20


class AttackModel:
    """Expected goals and assists for a player-fixture."""

    def __init__(self) -> None:
        self.goals_per90 = Component(objective="poisson", fallback=0.0, name="goals/90")
        self.assists_per90 = Component(objective="poisson", fallback=0.0, name="assists/90")

    def fit(self, train: pd.DataFrame, features: Sequence[str]) -> "AttackModel":
        played = train[train["minutes"] >= MIN_MINUTES_FOR_RATE].copy()
        if played.empty:
            return self

        per90 = 90.0 / played["minutes"].astype(float)
        weight = recency_weights(played)

        # Weight by minutes as well as recency: a full match is more evidence of
        # a player's rate than a cameo, and should count for more.
        minutes_weight = weight * (played["minutes"].astype(float) / 90.0)

        self.goals_per90.fit(
            played, features, played["goals_scored"].astype(float) * per90, minutes_weight
        )
        self.assists_per90.fit(
            played, features, played["assists"].astype(float) * per90, minutes_weight
        )
        return self

    def predict(self, test: pd.DataFrame, expected_minutes: np.ndarray) -> pd.DataFrame:
        share = np.clip(np.asarray(expected_minutes, dtype=float), 0.0, 90.0) / 90.0
        goals = np.clip(self.goals_per90.predict(test), 0.0, None) * share
        assists = np.clip(self.assists_per90.predict(test), 0.0, None) * share
        return pd.DataFrame(
            {"expected_goals": goals, "expected_assists": assists}, index=test.index
        )
