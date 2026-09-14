"""The in-progress season, straight from the live FPL API.

Two frames come out of here, both on the canonical schema:

* `played_frame`   — finished player-fixtures so far this season. Same shape as
                     a historical season, so features are built identically.
* `upcoming_frame` — one row per (player, fixture) for a *future* gameweek, with
                     every outcome column left NA. This is what predictions are
                     produced for; blanks and doubles fall out naturally because
                     rows come from the fixture list, not from the player list.
"""

from __future__ import annotations

import logging

import pandas as pd

from .fpl_api import ELEMENT_TYPE_TO_POSITION, FPLClient
from .schema import finalise_frame

log = logging.getLogger(__name__)


def _player_meta(client: FPLClient) -> pd.DataFrame:
    """id -> name / position / current team. Joined on element id, never name."""
    players = client.players()
    meta = players[["id", "element_type", "team", "web_name", "first_name", "second_name"]].copy()
    meta["name"] = meta["first_name"].fillna("") + " " + meta["second_name"].fillna("")
    meta["name"] = meta["name"].str.strip().where(lambda s: s.ne(""), meta["web_name"])
    meta["position"] = meta["element_type"].map(ELEMENT_TYPE_TO_POSITION)
    return meta.rename(columns={"id": "player_id", "team": "current_team_id"})[
        ["player_id", "name", "position", "element_type", "current_team_id"]
    ]


def _fixture_context(client: FPLClient) -> pd.DataFrame:
    fx = client.fixtures_frame()
    cols = ["id", "event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty", "kickoff_time"]
    return fx[cols].rename(columns={"id": "fixture_id"})


def _attach_fixture_context(df: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """Derive team/opponent/difficulty from the fixture and the home flag."""
    df = df.merge(
        ctx.drop(columns=["event", "kickoff_time"]),
        on="fixture_id",
        how="left",
        validate="many_to_one",
    )
    home = df["was_home"].astype(bool)
    df["team_id"] = df["team_h"].where(home, df["team_a"])
    df["opponent_id"] = df["team_a"].where(home, df["team_h"])
    df["team_difficulty"] = df["team_h_difficulty"].where(home, df["team_a_difficulty"])
    df["opp_difficulty"] = df["team_a_difficulty"].where(home, df["team_h_difficulty"])
    return df


def played_frame(client: FPLClient, season: str, force: bool = False) -> pd.DataFrame:
    """Every finished player-fixture of the current season.

    Warning from brief §4: a gameweek's scores are not final until 09:00 UK the
    day after its last match — bonus and DEFCON still move. Rows for a gameweek
    that has not been marked finished are dropped, so provisional numbers never
    become training labels.
    """
    hist = client.player_histories(force=force)
    if hist.empty:
        return finalise_frame(pd.DataFrame({"player_id": [], "gw": [], "fixture_id": []}))

    df = hist.rename(columns={"element": "player_id", "fixture": "fixture_id", "round": "gw"})
    df["season"] = season
    df["was_home"] = df["was_home"].astype(bool)

    df = df.merge(_player_meta(client), on="player_id", how="left", validate="many_to_one")
    df = _attach_fixture_context(df, _fixture_context(client))
    roles = client.players()[["id", "penalties_order", "corners_and_indirect_freekicks_order"]].rename(
        columns={"id": "player_id", "corners_and_indirect_freekicks_order": "setpiece_order"}
    )
    df = df.merge(roles, on="player_id", how="left", validate="many_to_one")

    home = df["was_home"]
    df["team_goals_for"] = df["team_h_score"].where(home, df["team_a_score"])
    df["team_goals_against"] = df["team_a_score"].where(home, df["team_h_score"])

    finished = set(client.events().loc[lambda d: d["finished"].astype(bool), "id"].astype(int))
    before = len(df)
    df = df[df["gw"].astype(int).isin(finished)]
    if len(df) < before:
        log.info("dropped %d rows from unfinished gameweeks (scores not final)", before - len(df))

    return finalise_frame(df)


def upcoming_frame(client: FPLClient, gw: int, season: str) -> pd.DataFrame:
    """Prediction rows for gameweek `gw`: every player x their team's fixtures.

    Outcome columns are NA by construction — there is nothing to know yet.
    A player whose team blanks gets no row; a double gives them two.
    """
    ctx = _fixture_context(client)
    fixtures = ctx[ctx["event"] == gw]
    if fixtures.empty:
        raise ValueError(f"no fixtures scheduled for gameweek {gw}")

    meta = _player_meta(client)

    home = fixtures.assign(team_id=fixtures["team_h"], was_home=True)
    away = fixtures.assign(team_id=fixtures["team_a"], was_home=False)
    sides = pd.concat([home, away], ignore_index=True)

    df = meta.merge(sides, left_on="current_team_id", right_on="team_id", how="inner")
    df["season"] = season
    df["gw"] = gw

    df["opponent_id"] = df["team_a"].where(df["was_home"], df["team_h"])
    df["team_difficulty"] = df["team_h_difficulty"].where(df["was_home"], df["team_a_difficulty"])
    df["opp_difficulty"] = df["team_a_difficulty"].where(df["was_home"], df["team_h_difficulty"])

    # Price and FPL's own expected points are both published before the
    # deadline, so both are legitimate inputs for the gameweek being predicted.
    # `ep_next` is the live equivalent of the historical `xP` column.
    live = client.players()[[
        "id", "now_cost", "ep_next", "penalties_order", "corners_and_indirect_freekicks_order",
        "chance_of_playing_next_round",
    ]].rename(columns={
        "id": "player_id", "now_cost": "value", "ep_next": "xP",
        "corners_and_indirect_freekicks_order": "setpiece_order",
        "chance_of_playing_next_round": "chance_of_playing",
    })
    live["xP"] = pd.to_numeric(live["xP"], errors="coerce")
    df = df.merge(live, on="player_id", how="left", validate="many_to_one")

    return finalise_frame(df)
