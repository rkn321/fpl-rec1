"""Shared test fixtures.

The leakage tests run on a *synthetic* season so they are fast, deterministic
and offline. A parallel set of tests runs the same checks against real cached
data, and skips when the cache is cold.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.data.schema import finalise_frame

N_TEAMS = 6
PLAYERS_PER_TEAM = 10
POSITION_CYCLE = ["GK", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "MID", "MID", "FWD"]
ELEMENT_TYPES = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}


def _round_robin(n_teams: int) -> list[tuple[int, tuple[int, int]]]:
    """A double round robin by the circle method: every team plays once per gameweek.

    Realism matters here — if a team accidentally played twice in a gameweek the
    leakage tests would be testing a fixture list that cannot occur.
    """
    teams = list(range(1, n_teams + 1))
    rounds: list[tuple[int, tuple[int, int]]] = []
    rotation = teams[:]
    for r in range(n_teams - 1):
        for i in range(n_teams // 2):
            home, away = rotation[i], rotation[n_teams - 1 - i]
            if i % 2 == 0:
                home, away = away, home
            rounds.append((r + 1, (home, away)))
        rotation = [rotation[0]] + [rotation[-1]] + rotation[1:-1]
    # Reverse leg: same pairings, home and away swapped.
    second = [(gw + n_teams - 1, (a, h)) for gw, (h, a) in rounds]
    return rounds + second


def make_synthetic_season(season: str = "2099-00", seed: int = 0) -> pd.DataFrame:
    """A small, fully-formed season on the canonical schema.

    Six teams playing a double round robin: 30 fixtures across 10 gameweeks.
    Outcomes are random but seeded, and correlated with a per-player skill so
    rolling features have real signal to pick up.
    """
    rng = np.random.default_rng(seed)

    fixtures = []
    for gw, (home, away) in _round_robin(N_TEAMS):
        fixtures.append(
            {
                "fixture_id": len(fixtures) + 1,
                "gw": gw,
                "team_h": home,
                "team_a": away,
                "kickoff_time": pd.Timestamp("2099-08-01", tz="UTC") + pd.Timedelta(days=7 * gw),
                "team_h_difficulty": int(rng.integers(2, 6)),
                "team_a_difficulty": int(rng.integers(2, 6)),
                "goals_h": int(rng.integers(0, 4)),
                "goals_a": int(rng.integers(0, 3)),
            }
        )
    fx = pd.DataFrame(fixtures)

    players = []
    for team in range(1, N_TEAMS + 1):
        for j in range(PLAYERS_PER_TEAM):
            pid = (team - 1) * PLAYERS_PER_TEAM + j + 1
            position = POSITION_CYCLE[j]
            players.append(
                {
                    "player_id": pid,
                    "team_id": team,
                    "name": f"Player {pid}",
                    "position": position,
                    "element_type": ELEMENT_TYPES[position],
                    "skill": float(rng.uniform(0.2, 1.0)),
                    "value": int(rng.integers(40, 130)),
                }
            )
    pl = pd.DataFrame(players)

    rows = []
    for _, f in fx.iterrows():
        for home in (True, False):
            team = f["team_h"] if home else f["team_a"]
            gf = f["goals_h"] if home else f["goals_a"]
            ga = f["goals_a"] if home else f["goals_h"]
            squad = pl[pl["team_id"] == team]
            for _, p in squad.iterrows():
                minutes = int(rng.choice([0, 20, 90], p=[0.3, 0.2, 0.5]))
                goals = int(rng.poisson(p["skill"] * 0.3)) if minutes > 45 else 0
                assists = int(rng.poisson(p["skill"] * 0.2)) if minutes > 45 else 0
                rows.append(
                    {
                        "season": season,
                        "player_id": int(p["player_id"]),
                        "gw": int(f["gw"]),
                        "fixture_id": int(f["fixture_id"]),
                        "team_id": int(team),
                        "opponent_id": int(f["team_a"] if home else f["team_h"]),
                        "kickoff_time": f["kickoff_time"],
                        "name": p["name"],
                        "position": p["position"],
                        "element_type": int(p["element_type"]),
                        "was_home": home,
                        "team_difficulty": int(
                            f["team_h_difficulty"] if home else f["team_a_difficulty"]
                        ),
                        "opp_difficulty": int(
                            f["team_a_difficulty"] if home else f["team_h_difficulty"]
                        ),
                        "value": int(p["value"]),
                        "minutes": minutes,
                        "starts": int(minutes >= 60),
                        "goals_scored": goals,
                        "assists": assists,
                        "expected_goals": float(rng.gamma(1.0, p["skill"] * 0.2)),
                        "expected_assists": float(rng.gamma(1.0, p["skill"] * 0.1)),
                        "expected_goal_involvements": float(rng.gamma(1.0, p["skill"] * 0.3)),
                        "expected_goals_conceded": float(rng.gamma(1.0, 0.5)),
                        "clean_sheets": int(ga == 0 and minutes >= 60),
                        "goals_conceded": int(ga),
                        "saves": int(rng.integers(0, 5)) if p["position"] == "GK" else 0,
                        "penalties_saved": 0,
                        "penalties_missed": 0,
                        "own_goals": 0,
                        "yellow_cards": int(rng.random() < 0.1),
                        "red_cards": 0,
                        "bonus": int(rng.choice([0, 0, 0, 1, 2, 3])),
                        "bps": int(rng.integers(0, 40)),
                        "influence": float(rng.uniform(0, 50)),
                        "creativity": float(rng.uniform(0, 50)),
                        "threat": float(rng.uniform(0, 50)),
                        "ict_index": float(rng.uniform(0, 15)),
                        "defensive_contribution": float(rng.integers(0, 3)) * 2,
                        "clearances_blocks_interceptions": float(rng.integers(0, 12)),
                        "recoveries": float(rng.integers(0, 12)),
                        "tackles": float(rng.integers(0, 6)),
                        "team_goals_for": int(gf),
                        "team_goals_against": int(ga),
                        "selected": int(rng.integers(1000, 500000)),
                        "transfers_in": int(rng.integers(0, 50000)),
                        "transfers_out": int(rng.integers(0, 50000)),
                        "transfers_balance": int(rng.integers(-50000, 50000)),
                        "xP": float(rng.uniform(0, 8)),
                    }
                )

    df = pd.DataFrame(rows)
    # Points roughly follow the real scoring rules so the target is not noise.
    goal_pts = df["position"].map({"GK": 6, "DEF": 6, "MID": 5, "FWD": 4})
    cs_pts = df["position"].map({"GK": 4, "DEF": 4, "MID": 1, "FWD": 0})
    df["total_points"] = (
        (df["minutes"] > 0).astype(int)
        + (df["minutes"] >= 60).astype(int)
        + df["goals_scored"] * goal_pts
        + df["assists"] * 3
        + df["clean_sheets"] * cs_pts
        + df["bonus"]
        - df["yellow_cards"]
    )
    return finalise_frame(df)


def make_double_gameweek_season(seed: int = 3) -> pd.DataFrame:
    """A synthetic season with a real double gameweek folded in.

    Gameweek 6's fixtures are relabelled as gameweek 5, so every team plays
    twice in gameweek 5 — including on different days, which is what makes
    fixture-level lagging unsafe.
    """
    df = make_synthetic_season(seed=seed).copy()
    moved = df["gw"] == 6
    df.loc[moved, "gw"] = 5
    # The second fixture kicks off three days later, as a real double does.
    df.loc[moved, "kickoff_time"] = df.loc[moved, "kickoff_time"] + pd.Timedelta(days=3)
    return df.sort_values(["season", "player_id", "gw", "kickoff_time"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def synthetic() -> pd.DataFrame:
    return make_synthetic_season()


@pytest.fixture(scope="session")
def double_gameweek() -> pd.DataFrame:
    return make_double_gameweek_season()
