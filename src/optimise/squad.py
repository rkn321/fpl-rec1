"""Squad optimiser (Phase 5, optional).

An integer linear program (e.g. `pulp`) maximising expected points subject to
the game's constraints (brief §4):

    15 players, budget 100.0m
    exactly 2 GK, 5 DEF, 5 MID, 3 FWD
    at most 3 players per club
    starting XI: 1 GK, >=3 DEF, >=2 MID, >=1 FWD
    captain scores double

Extensions worth having, in rough order of value:

* **Multi-week planning** — optimise over a horizon with a decay factor instead
  of one gameweek at a time, so fixture swings are anticipated rather than
  chased.
* **Transfers and hits** — 1 free transfer per gameweek rolling up to 5; extra
  transfers cost -4. A hit is worth taking only if the horizon gain clears 4.
* **Selling price** — you get purchase price plus half of any rise, rounded down
  to 0.1m, so realisable budget is not the displayed squad value.
* **Bench and autosubs** — bench players carry option value; a 0-minute starter
  is replaced by the first bench player that keeps a legal formation.
* **Chips** — two sets of four, the first expiring at the GW19 deadline, one per
  gameweek. Bench Boost and Triple Captain are the two an optimiser can price
  directly.
"""

from __future__ import annotations

SQUAD_SIZE = 15
BUDGET = 1000  # tenths of a million, matching the API's `value` units
POSITION_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MIN_STARTERS = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
MAX_PER_CLUB = 3
TRANSFER_HIT_COST = 4
MAX_ROLLED_TRANSFERS = 5


def pick_squad(expected_points, budget: int = BUDGET):
    raise NotImplementedError("Phase 5: see module docstring")
