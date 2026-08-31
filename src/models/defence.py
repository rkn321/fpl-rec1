"""Clean sheets, goals conceded, saves and DEFCON (Phase 3).

Four payouts with genuinely different shapes, so four treatments:

* **Clean sheet** is a *team* event. It is modelled once per team-fixture rather
  than once per player, because eleven players of the same side share one
  outcome — fitting it per player would relearn the same thing eleven times and
  let two team-mates disagree about whether their team kept a clean sheet.
* **Goals conceded** pays -1 per *two* conceded. A step function needs the whole
  distribution, not its mean: E[floor(X/2)] is not floor(E[X]/2). A Poisson mean
  is fitted and the expectation summed over the pmf.
* **Saves** pay 1 per 3, so the same argument applies, and only to keepers.
* **DEFCON** is a threshold on a count — 10 combined defensive actions for a
  defender, 12 for everyone else. Modelled as P(threshold reached) directly,
  because regressing the count and thresholding the mean is badly biased: a
  player averaging 8 with real variance clears 10 far more often than a point
  estimate of 8 suggests.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .base import Component, poisson_tail_expectation, recency_weights
from .scoring import DEFCON_THRESHOLD, GOALS_CONCEDED_PER_PENALTY, SAVES_PER_POINT


class DefenceModel:
    """Team clean sheet, goals conceded, saves and DEFCON probability."""

    def __init__(self) -> None:
        self.team_clean_sheet = Component(classifier=True, fallback=0.25, name="team_cs")
        self.team_conceded = Component(objective="poisson", fallback=1.4, name="team_gc")
        self.saves = Component(objective="poisson", fallback=0.0, name="saves")
        self.defcon = Component(classifier=True, fallback=0.0, name="defcon")

    # -- team level ---------------------------------------------------------
    @staticmethod
    def _team_rows(frame: pd.DataFrame) -> pd.DataFrame:
        """One row per team-fixture — the grain a clean sheet actually happens at."""
        return frame.drop_duplicates(subset=["season", "team_id", "fixture_id"])

    def fit(self, train: pd.DataFrame, features: Sequence[str]) -> "DefenceModel":
        teams = self._team_rows(train)
        team_weight = recency_weights(teams)

        self.team_clean_sheet.fit(
            teams, features, (teams["team_goals_against"] == 0).astype(float), team_weight
        )
        self.team_conceded.fit(
            teams, features, teams["team_goals_against"].astype(float), team_weight
        )

        keepers = train[(train["position"] == "GK") & (train["minutes"] >= 60)]
        self.saves.fit(keepers, features, keepers["saves"].astype(float), recency_weights(keepers))

        # DEFCON only exists from 2025-26; earlier rows are NaN and drop out.
        played = train[(train["minutes"] >= 60) & train["defensive_contribution"].notna()]
        if not played.empty:
            threshold = played["position"].map(DEFCON_THRESHOLD).fillna(12)
            hit = (played["defensive_contribution"].astype(float) >= threshold).astype(float)
            self.defcon.fit(played, features, hit, recency_weights(played))
        return self

    def predict(self, test: pd.DataFrame, p_60: np.ndarray) -> pd.DataFrame:
        p_60 = np.clip(np.asarray(p_60, dtype=float), 0.0, 1.0)

        p_cs = np.clip(self.team_clean_sheet.predict(test), 0.0, 1.0)
        conceded = np.clip(self.team_conceded.predict(test), 0.0, None)

        # Only paid out if the player is on for 60 minutes.
        conceded_penalty = poisson_tail_expectation(conceded, per=GOALS_CONCEDED_PER_PENALTY)

        saves = np.clip(self.saves.predict(test), 0.0, None)
        is_gk = (test["position"] == "GK").to_numpy()
        save_points = np.where(is_gk, saves / SAVES_PER_POINT, 0.0) * p_60

        p_defcon = np.clip(self.defcon.predict(test), 0.0, 1.0) * p_60

        return pd.DataFrame(
            {
                "p_clean_sheet": p_cs,
                "expected_conceded": conceded,
                "conceded_penalty": conceded_penalty,
                "expected_save_points": save_points,
                "p_defcon": p_defcon,
            },
            index=test.index,
        )
