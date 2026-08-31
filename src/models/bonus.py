"""Bonus points (Phase 3).

The top 3 BPS scores in each match take 3 / 2 / 1, shared on ties. BPS is built
from ~32 match stats, so the tractable route is to predict a player's BPS and
then model where it lands in their match's ranking — bonus is a *within-match
tournament*, not an independent per-player quantity.

Sketch: predict E[BPS] per player-fixture, then simulate or rank within each
fixture to get p(1st), p(2nd), p(3rd), and take the expectation.

For 2026/27 the BPS weights were rebalanced to reduce overlap with DEFCON: they
now favour attackers and full-backs, and players are no longer penalised for
being tackled or dispossessed (brief §4). BPS coefficients learned on earlier
seasons are therefore stale — fit on recent data only, or give the model a
season indicator.
"""

from __future__ import annotations


class BonusModel:
    def fit(self, train, feature_cols):
        raise NotImplementedError("Phase 3: see module docstring")

    def predict(self, test):
        raise NotImplementedError("Phase 3: see module docstring")
