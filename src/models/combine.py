"""Combine component predictions into expected points (Phase 3).

This is why the brief insists on component models: position multipliers and
DEFCON thresholds behave so differently that a single black box on total points
fits all of them badly. Predict the components, then apply the scoring rules —
which live in `src/models/scoring.py`, so there is exactly one copy of them.

    E[points] = appearance + goals + assists + clean sheet + goals conceded
              + saves + DEFCON + bonus + cards

Every attacking and defensive component is scaled by playing-time probability
from `minutes.py`. The appearance bonus and the clean-sheet term specifically
need p(60+ minutes) rather than E[minutes], since both are step functions at 60.
"""

from __future__ import annotations


def combine(components, positions):
    raise NotImplementedError("Phase 3: see module docstring")
