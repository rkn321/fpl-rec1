"""Minutes model (Phase 3) — the one that matters most.

Every other component gets multiplied by playing time, so error here propagates
into all of them (brief §6.3). It is also the only component where the target is
close to deterministic given team news, and the feature frame does not have team
news — so this is where the model is most obviously limited by its inputs.

Three quantities, because the scoring rules need three different things:

    p_play    P(minutes >= 1)   — the appearance point, and a gate on everything
    p_60      P(minutes >= 60)  — the second appearance point, and clean sheets
    minutes   E[minutes]        — scales goal and assist rates

`p_60` is modelled directly rather than derived from a minutes regression. The
distribution of minutes is strongly bimodal — a starter plays ~90, a substitute
plays ~15, and almost nobody plays 55 — so a conditional mean lands in a gap
where few players actually are, and thresholding it is worse than asking the
question directly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .base import Component, recency_weights


class MinutesModel:
    """p(play), p(60+) and expected minutes for a player-fixture."""

    def __init__(self) -> None:
        self.p_play = Component(classifier=True, fallback=0.0, name="p_play")
        self.p_60 = Component(classifier=True, fallback=0.0, name="p_60")
        self.minutes_if_played = Component(fallback=0.0, name="minutes|played")

    def fit(self, train: pd.DataFrame, features: Sequence[str]) -> "MinutesModel":
        weight = recency_weights(train)

        self.p_play.fit(train, features, (train["minutes"] > 0).astype(float), weight)
        self.p_60.fit(train, features, (train["minutes"] >= 60).astype(float), weight)

        # Conditional on appearing, so the model is not spending capacity
        # relearning "most of the pool does not play".
        played = train[train["minutes"] > 0]
        self.minutes_if_played.fit(
            played, features, played["minutes"].astype(float), recency_weights(played)
        )
        return self

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        p_play = np.clip(self.p_play.predict(test), 0.0, 1.0)
        p_60 = np.clip(self.p_60.predict(test), 0.0, 1.0)
        # Playing 60 minutes implies playing at all; the two classifiers are
        # fitted independently and can disagree at the margins.
        p_60 = np.minimum(p_60, p_play)

        conditional = np.clip(self.minutes_if_played.predict(test), 0.0, 90.0)
        expected = p_play * conditional

        return pd.DataFrame(
            {"p_play": p_play, "p_60": p_60, "expected_minutes": expected},
            index=test.index,
        )
