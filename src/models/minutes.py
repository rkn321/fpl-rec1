"""Minutes model (Phase 3) — build this one first.

Minutes are the make-or-break feature: every other component is multiplied by
playing time, so errors here dominate everything downstream (brief §6.3).

Two targets:
    p_start(player, fixture)           probability of starting
    expected_minutes(player, fixture)  minutes, integrating over start/bench/cameo

Suggested shape: classifiers for p(start) and p(any minutes), then

    E[minutes] = p_start * E[min | start] + p_cameo * E[min | cameo]

Features already in the frame: `start_rate_todate`, `minutes_r3/r5/r10`,
`minutes_lag1/lag2`, `appearances_todate`. Add `chance_of_playing_next_round`
from bootstrap-static, and the injury feed once it exists.

Calibration matters more than accuracy here — track it (brief §7). Note that
p(60+ minutes) is needed separately from E[minutes], because the appearance
bonus and clean-sheet points are step functions at 60.
"""

from __future__ import annotations


class MinutesModel:
    def fit(self, train, feature_cols):
        raise NotImplementedError("Phase 3: see module docstring")

    def predict_minutes(self, test):
        raise NotImplementedError("Phase 3: see module docstring")
