"""Clean sheets, goals conceded, saves and DEFCON (Phase 3).

Four separate things, and all but goals conceded need 60+ minutes to pay out:

* **Clean sheet** — p(opponent scores 0). A team-level Poisson on expected goals
  conceded works well; odds-implied CS probability (Phase 4) is stronger still.
  Worth 4 to GK/DEF, 1 to MID, 0 to FWD.
* **Goals conceded** — -1 per 2 conceded for GK/DEF. Needs the whole conceded
  distribution, not its mean, because the penalty is a step function.
* **Saves** — 1 point per 3 saves, so again a distribution rather than a mean.
* **DEFCON** — model explicitly (brief §4). Defenders need 10 combined
  clearances, blocks, interceptions and tackles (CBIT); mids and forwards need
  12 of those plus ball recoveries (CBIRT). Worth +2, capped at 2 per match
  (reaching double the threshold still pays 2), repeatable, and independent of
  clean sheets. Model p(threshold reached) directly from the per-90 rate and
  expected minutes — regressing the mean count and then thresholding it would be
  badly biased.

The `defensive_contribution`, `clearances_blocks_interceptions`, `recoveries`
and `tackles` columns exist from 2025-26 onward and are NaN before that.
"""

from __future__ import annotations

# Combined defensive actions needed for the +2, by position.
DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12, "GK": 12}
DEFCON_POINTS = 2


class DefenceModel:
    def fit(self, train, feature_cols):
        raise NotImplementedError("Phase 3: see module docstring")

    def predict(self, test):
        raise NotImplementedError("Phase 3: see module docstring")
