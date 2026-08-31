"""The 2026/27 FPL scoring rules, in one place.

`combine.py` turns component predictions into expected points by applying these,
and the optimiser prices decisions with them. Keeping exactly one copy means a
rule change is a one-line edit rather than a hunt through the codebase.

Everything here is deterministic given a realised stat line, which also makes it
testable against actual gameweek data — see `tests/test_scoring.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Goals are worth different amounts by position; assists are flat.
GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
ASSIST_POINTS = 3

# Clean sheets need 60+ minutes played.
CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
CLEAN_SHEET_MIN_MINUTES = 60

APPEARANCE_POINTS = 1
APPEARANCE_60_POINTS = 2  # total, not additional

SAVES_PER_POINT = 3
PENALTY_SAVE_POINTS = 5
PENALTY_MISS_POINTS = -2
GOALS_CONCEDED_PER_PENALTY = 2  # -1 per 2 conceded, GK/DEF only
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3
OWN_GOAL_POINTS = -2

# Defensive contributions. Defenders count clearances, blocks, interceptions and
# tackles (CBIT); everyone else adds ball recoveries (CBIRT). The award is +2 and
# is capped at 2 per match — reaching double the threshold still pays 2.
DEFCON_THRESHOLD = {"GK": 12, "DEF": 10, "MID": 12, "FWD": 12}
DEFCON_POINTS = 2

CONCEDES_GOALS = {"GK", "DEF"}


@dataclass(frozen=True)
class StatLine:
    """One player's realised stats in one fixture."""

    position: str
    minutes: int = 0
    goals_scored: int = 0
    assists: int = 0
    clean_sheet: bool = False
    goals_conceded: int = 0
    saves: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    own_goals: int = 0
    defensive_contribution: int = 0
    bonus: int = 0


def appearance_points(minutes: int) -> int:
    if minutes <= 0:
        return 0
    return APPEARANCE_60_POINTS if minutes >= 60 else APPEARANCE_POINTS


def clean_sheet_points(position: str, minutes: int, clean_sheet: bool) -> int:
    """Clean sheets only pay from 60 minutes; a 59-minute shutout is worth nothing."""
    if not clean_sheet or minutes < CLEAN_SHEET_MIN_MINUTES:
        return 0
    return CLEAN_SHEET_POINTS.get(position, 0)


def goals_conceded_points(position: str, goals_conceded: int) -> int:
    if position not in CONCEDES_GOALS:
        return 0
    return -(goals_conceded // GOALS_CONCEDED_PER_PENALTY)


def saves_points(saves: int) -> int:
    return saves // SAVES_PER_POINT


def defcon_points(position: str, defensive_contribution: int) -> int:
    """+2 once the threshold is reached, and no more than +2 however far past it."""
    threshold = DEFCON_THRESHOLD.get(position, 12)
    return DEFCON_POINTS if defensive_contribution >= threshold else 0


def score(stat_line: StatLine) -> int:
    """Total FPL points for a realised stat line."""
    s = stat_line
    if s.minutes <= 0:
        # A player who does not appear scores nothing, whatever else is recorded.
        return 0

    total = appearance_points(s.minutes)
    total += s.goals_scored * GOAL_POINTS.get(s.position, 0)
    total += s.assists * ASSIST_POINTS
    total += clean_sheet_points(s.position, s.minutes, s.clean_sheet)
    total += goals_conceded_points(s.position, s.goals_conceded)
    total += saves_points(s.saves)
    total += s.penalties_saved * PENALTY_SAVE_POINTS
    total += s.penalties_missed * PENALTY_MISS_POINTS
    total += defcon_points(s.position, s.defensive_contribution)
    total += s.yellow_cards * YELLOW_CARD_POINTS
    total += s.red_cards * RED_CARD_POINTS
    total += s.own_goals * OWN_GOAL_POINTS
    total += s.bonus
    return total
