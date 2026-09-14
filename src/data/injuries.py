"""Injury, suspension and predicted-lineup feed (Phase 4).

The hardest input to source and the most valuable: a haul is worthless if the
player is benched (brief §3.6). There is no clean free API — predicted lineups
and injury news come from RotoWire, Fantasy Football Scout and Physioroom, by
scraping or paid tier.

Until that exists, minutes are modelled from rolling `minutes` in the FPL API
(which `src/models/minutes.py` does) plus `chance_of_playing_next_round` from
`bootstrap-static`, which is free, official and already available.
"""

from __future__ import annotations


def load_availability(gameweek: int):
    raise NotImplementedError("Phase 4: see module docstring")
