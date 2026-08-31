"""Goals and assists (Phase 3).

Predict rates, not totals: xG and xA per 90 are far more stable than realised
goals, so model the rate and multiply by expected minutes from `minutes.py`.

    E[goals]   = xG_per_90 * E[minutes] / 90
    E[assists] = xA_per_90 * E[minutes] / 90

Adjust for opponent (the `opp_form_*` features) and venue. Penalty takers need
separate treatment: penalty xG is ~0.79 per shot and concentrated in one player
per side, so a team penalty rate times a taker share beats folding it into
open-play xG.

Points depend on position — 6 for GK/DEF, 5 for MID, 4 for FWD, assists 3 flat —
and `combine.py` applies those multipliers.
"""

from __future__ import annotations


class AttackModel:
    def fit(self, train, feature_cols):
        raise NotImplementedError("Phase 3: see module docstring")

    def predict(self, test):
        raise NotImplementedError("Phase 3: see module docstring")
