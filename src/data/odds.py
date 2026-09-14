"""Market odds (Phase 4).

Implied clean-sheet probability and team goal expectancy are strong, underused
features, and they price in team news the model cannot otherwise see (brief §3.4).

Sources: the-odds-api.com for live odds (free tier, needs a key in config), and
football-data.co.uk for historical closing odds going back many seasons.

Conversion sketch: de-vig the 1X2 and over/under markets to recover each team's
expected goals, then map those to p(clean sheet) and to the goals-conceded bands
the FPL scoring rules actually pay on.
"""

from __future__ import annotations


def load_odds(season: str):
    raise NotImplementedError("Phase 4: see module docstring")
