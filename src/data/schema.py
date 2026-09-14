"""The canonical `player_gameweek` schema.

Every data source (the live FPL API, the vaastav historical CSVs) is normalised
into the same long frame before features are built.

**Row grain is `(season, player_id, fixture_id)`, not `(season, player_id, gw)`.**
Blanks and doubles are real: a player can have 0 or 2 fixtures in a gameweek, so
keying on gameweek alone silently drops or collapses rows (brief §6.5).

Columns are split into three groups, and the split is what the leakage test
enforces:

* `IDENTITY_COLS`  — keys and joins.
* `CONTEXT_COLS`   — knowable *before* the deadline (home/away, FPL fixture
                     difficulty, the price you would pay). Safe to use raw.
* `OUTCOME_COLS`   — the result of the match. **Never** usable raw as a feature
                     for that same row; only via lagged/rolling transforms.
"""

from __future__ import annotations

IDENTITY_COLS: list[str] = [
    "season",
    "player_id",
    "gw",
    "fixture_id",
    "team_id",
    "opponent_id",
    "kickoff_time",
    "name",
    "position",
    "element_type",
]

CONTEXT_COLS: list[str] = [
    "was_home",
    "team_difficulty",
    "opp_difficulty",
    "value",
    # FPL's own expected points for the gameweek, published before the deadline.
    # It is the one input here that carries team news — it moves with the
    # `chance_of_playing` flags FPL sets from press conferences — which is
    # exactly what a model built on lagged appearances cannot reconstruct.
    #
    # Checked before being trusted (brief §6.1 warns it may be post-hoc). On
    # 2024-25, where the scrape is 91% complete, players who had started the
    # previous three matches and then did not play average 1.40 against 3.55 for
    # those who did: reduced, not zeroed. A column computed after the match
    # would know they were absent and put them at 0.
    "xP",
    # Set-piece duty: 1 = first choice, 2 = second, ... NA = not in the order.
    # Published in bootstrap-static and stable within a season, so it is a
    # legitimate pre-deadline input. Penalty xG is ~0.79 a shot and concentrated
    # in one player per side; corner duty drives a large share of assists.
    "penalties_order",
    "setpiece_order",
    # FPL's own availability flag, 0-100, from press conferences. It has no
    # per-gameweek history, so it is never a training feature — it is applied as
    # a ceiling on the minutes prediction for the gameweek being predicted.
    "chance_of_playing",
]

# Post-match player stats. These are the model's raw material but are only ever
# consumed lagged. `total_points` is also the prediction target.
OUTCOME_COLS: list[str] = [
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "penalties_saved",
    "penalties_missed",
    "own_goals",
    "yellow_cards",
    "red_cards",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    # Defensive contributions (DEFCON). Only present from 2025-26 onward;
    # NaN for earlier seasons.
    "defensive_contribution",
    "clearances_blocks_interceptions",
    "recoveries",
    "tackles",
    # Team-level match result, carried on every player row of that fixture.
    "team_goals_for",
    "team_goals_against",
    # Transfer-market activity settled at that gameweek's deadline. Lagged like
    # everything else — cheap insurance, since the exact snapshot time varies.
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
]

# `xP` doubles as baseline (b) in the evaluation (brief §7). It now also serves
# as a feature, which makes "beats baseline (b)" partly circular — the backtest
# reports both with and without it so the comparison stays legible.
BASELINE_COLS: list[str] = []

TARGET = "total_points"

ALL_COLS: list[str] = IDENTITY_COLS + CONTEXT_COLS + OUTCOME_COLS + BASELINE_COLS

# Columns that carry a single value for the whole fixture rather than per player.
TEAM_LEVEL_COLS: list[str] = ["team_goals_for", "team_goals_against"]

POSITIONS: tuple[str, ...] = ("GK", "DEF", "MID", "FWD")


def finalise_frame(df):  # type: ignore[no-untyped-def]
    """Coerce any source frame to the canonical schema: columns, dtypes, order.

    Missing columns are filled with NA rather than raising — DEFCON stats don't
    exist before 2025-26, and the live API has no `xP` column — so downstream
    code can rely on the full column set always being present.
    """
    import pandas as pd

    df = df.copy()
    for col in ALL_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    out = df[ALL_COLS].copy()

    int_like = ["player_id", "gw", "fixture_id", "team_id", "opponent_id", "element_type"]
    for col in int_like:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    non_numeric = {"season", "name", "position", "kickoff_time", "was_home", *int_like}
    for col in ALL_COLS:
        if col not in non_numeric:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["was_home"] = out["was_home"].astype(bool)
    out = out.dropna(subset=["player_id", "gw", "fixture_id"])
    out = out.sort_values(["season", "player_id", "kickoff_time", "fixture_id"])
    return out.reset_index(drop=True)
