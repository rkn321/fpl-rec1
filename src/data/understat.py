"""Understat xG / shot-level data (Phase 4).

Underlying numbers are the real predictive juice: xG, xA, npxG, xGChain,
xGBuildup, and shot coordinates, big-5 leagues from 2014/15 (brief §3.3).

Planned approach: pull via the `understat` package or `soccerdata` rather than a
hand-rolled scraper. Understat has light bot protection; FBref sits behind
Cloudflare and should go through `soccerdata` / `worldfootballR`.

**Join on the FPL element id via the crosswalk, never on player name** — names
differ across FPL / Understat / FBref through accents, nicknames and transfers
(brief §6.2). The vaastav dataset already carries Understat xG merged against
official ids, so Phases 1-3 get much of this for free; this module is for the
shot-level detail the merged dataset does not carry.
"""

from __future__ import annotations


def load_shots(season: str):
    raise NotImplementedError("Phase 4: see module docstring")
