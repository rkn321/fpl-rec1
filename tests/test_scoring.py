"""Tests for the 2026/27 scoring rules.

The unit tests pin the rules that are easy to get subtly wrong (the 60-minute
cliff, the DEFCON cap, integer division on saves and goals conceded). The last
test is the real check: reconstruct `total_points` for a whole real season from
its component columns and confirm it matches what FPL actually awarded.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config
from src.data.historical import VaastavLoader
from src.models.scoring import (
    DEFCON_THRESHOLD,
    StatLine,
    appearance_points,
    clean_sheet_points,
    defcon_points,
    goals_conceded_points,
    saves_points,
    score,
)


def test_appearance_cliff_at_60_minutes() -> None:
    assert appearance_points(0) == 0
    assert appearance_points(1) == 1
    assert appearance_points(59) == 1
    assert appearance_points(60) == 2
    assert appearance_points(90) == 2


def test_clean_sheet_needs_60_minutes() -> None:
    assert clean_sheet_points("DEF", 90, True) == 4
    assert clean_sheet_points("DEF", 59, True) == 0, "a 59-minute shutout pays nothing"
    assert clean_sheet_points("MID", 90, True) == 1
    assert clean_sheet_points("FWD", 90, True) == 0
    assert clean_sheet_points("GK", 90, False) == 0


def test_goals_conceded_only_hits_keepers_and_defenders() -> None:
    assert goals_conceded_points("GK", 2) == -1
    assert goals_conceded_points("DEF", 3) == -1, "-1 per *two* conceded, rounded down"
    assert goals_conceded_points("DEF", 4) == -2
    assert goals_conceded_points("MID", 4) == 0
    assert goals_conceded_points("FWD", 4) == 0


def test_saves_are_one_point_per_three() -> None:
    assert saves_points(2) == 0
    assert saves_points(3) == 1
    assert saves_points(8) == 2


def test_defcon_is_capped_at_two() -> None:
    assert defcon_points("DEF", 9) == 0
    assert defcon_points("DEF", 10) == 2
    assert defcon_points("DEF", 25) == 2, "double the threshold is still only +2"
    # Mids and forwards need 12, and count recoveries toward it.
    assert defcon_points("MID", 10) == 0
    assert defcon_points("MID", 12) == 2
    assert DEFCON_THRESHOLD["DEF"] == 10 and DEFCON_THRESHOLD["MID"] == 12


def test_a_player_who_does_not_appear_scores_nothing() -> None:
    assert score(StatLine(position="DEF", minutes=0, clean_sheet=True, bonus=3)) == 0


def test_a_full_stat_line() -> None:
    # Defender: 90 mins (2) + goal (6) + assist (3) + clean sheet (4)
    # + DEFCON (2) + bonus (3) - yellow (1) = 19
    line = StatLine(
        position="DEF",
        minutes=90,
        goals_scored=1,
        assists=1,
        clean_sheet=True,
        defensive_contribution=11,
        yellow_cards=1,
        bonus=3,
    )
    assert score(line) == 19


def test_keeper_stat_line() -> None:
    # 90 mins (2) + 6 saves (2) + penalty save (5) - 2 conceded (1) = 8
    line = StatLine(position="GK", minutes=90, saves=6, penalties_saved=1, goals_conceded=2)
    assert score(line) == 8


def test_scoring_rules_reconstruct_a_real_season() -> None:
    """Rebuild `total_points` for 2025-26 from its components.

    This is the test that would catch a misread rule: it checks the whole
    rulebook against ~30,000 real, officially-scored player-fixtures.
    """
    config = load_config()
    cached = config.cache_dir / "historical" / "2025-26" / "gws" / "merged_gw.csv"
    if not cached.exists():
        pytest.skip("2025-26 not cached — run `python -m src.cli build-features`")

    df = VaastavLoader(config).season_frame("2025-26")
    df = df[df["minutes"] > 0]

    rebuilt = [
        score(
            StatLine(
                position=row.position,
                minutes=int(row.minutes),
                goals_scored=int(row.goals_scored),
                assists=int(row.assists),
                clean_sheet=bool(row.clean_sheets),
                goals_conceded=int(row.goals_conceded),
                saves=int(row.saves),
                penalties_saved=int(row.penalties_saved),
                penalties_missed=int(row.penalties_missed),
                yellow_cards=int(row.yellow_cards),
                red_cards=int(row.red_cards),
                own_goals=int(row.own_goals),
                defensive_contribution=int(row.defensive_contribution),
                bonus=int(row.bonus),
            )
        )
        for row in df.itertuples()
    ]

    got = pd.Series(rebuilt, index=df.index)
    mismatches = df.loc[got != df["total_points"]]

    # It reconstructs exactly, all 11,498 of them. Asserting exactness rather
    # than a tolerance means a mid-season rule change shows up here as a
    # failure, which is precisely when we would want to hear about it.
    assert mismatches.empty, (
        f"{len(mismatches)} of {len(df)} stat lines did not reconstruct; "
        f"first: {mismatches.head(1).to_dict('records')}"
    )
