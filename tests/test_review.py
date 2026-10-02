"""The post-gameweek review.

What matters: actual points split into the model's terms and reconcile with
FPL's total; auto-subs and the captain's armband follow FPL's rules; the log
replaces a gameweek rather than double-counting it; and the comparison against
FPL's own figure is computed on the same players for both.
"""

from __future__ import annotations

import textwrap

import pandas as pd
import pytest

from src.config import load_config
from src.models.combine import POINTS_TERMS
from src.review import (
    ACTUAL_TERMS,
    Pick,
    Review,
    Squad,
    append_log,
    auto_subs,
    best_xi,
    calibration,
    head_to_head,
    join_actuals,
    parse_live,
    render,
    settle,
    squad_from_picks,
    squad_table,
)


@pytest.fixture
def config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""
        season:
          current: "2099-00"
          history: []
        paths:
          data_dir: "{(tmp_path / 'data').as_posix()}"
          cache_dir: "{(tmp_path / 'data' / 'cache').as_posix()}"
          processed_dir: "{(tmp_path / 'data' / 'processed').as_posix()}"
        api: {{base_url: "x", ttl: {{default: 1}}}}
        historical: {{vaastav_base: "y"}}
        features: {{windows: [3]}}
    """), encoding="utf-8")
    return load_config(cfg, use_local=False)


# -- what happened ---------------------------------------------------------------

LIVE = {
    "elements": [
        {   # a forward who scored, got bonus and an own goal
            "id": 9,
            "stats": {"minutes": 90, "total_points": 8, "goals_scored": 1, "bonus": 3, "own_goals": 1},
            "explain": [{"fixture": 1, "stats": [
                {"identifier": "minutes", "points": 2, "value": 90},
                {"identifier": "goals_scored", "points": 4, "value": 1},
                {"identifier": "bonus", "points": 3, "value": 3},
                {"identifier": "own_goals", "points": -2, "value": 1},
                {"identifier": "yellow_cards", "points": 1 - 2, "value": 1},
            ]}],
        },
        {   # a double: two fixtures' worth of explain
            "id": 4,
            "stats": {"minutes": 180, "total_points": 10, "clean_sheets": 1},
            "explain": [
                {"fixture": 2, "stats": [{"identifier": "minutes", "points": 2, "value": 90},
                                          {"identifier": "clean_sheets", "points": 4, "value": 1}]},
                {"fixture": 3, "stats": [{"identifier": "minutes", "points": 2, "value": 90},
                                          {"identifier": "defensive_contribution", "points": 2, "value": 11}]},
            ],
        },
        {"id": 5, "stats": {"minutes": 0, "total_points": 0}, "explain": []},
    ]
}


def test_actual_points_split_into_terms_and_reconcile() -> None:
    actual = parse_live(LIVE)
    nine = actual.loc[9]
    assert nine["act_goals"] == 4 and nine["act_bonus"] == 3 and nine["act_appearance"] == 2
    assert nine["act_other"] == -2, "own goals have no term; they still have to be counted"
    assert nine["act_cards"] == -1

    four = actual.loc[4]
    assert four["act_appearance"] == 4, "both fixtures of a double"
    assert four["act_clean_sheet"] == 4 and four["act_defcon"] == 2

    terms = actual[[f"act_{t}" for t in ACTUAL_TERMS]].sum(axis=1)
    assert terms.loc[4] == actual.loc[4, "actual"]
    assert actual.loc[5, "minutes"] == 0


def test_a_predicted_player_missing_from_live_counts_as_no_minutes() -> None:
    record = pd.DataFrame(
        {"name": ["A", "B"], "position": ["MID", "MID"], "team_id": [1, 1], "expected_points": [3.0, 1.0]},
        index=pd.Index([9, 77], name="player_id"),
    )
    table = join_actuals(record, parse_live(LIVE), {1: "XYZ (H) 1-0"})
    assert table.loc[77, "minutes"] == 0 and table.loc[77, "actual"] == 0
    assert table.loc[9, "diff"] == 8 - 3
    assert table.loc[9, "fixture"] == "XYZ (H) 1-0"


# -- the squad ---------------------------------------------------------------------

def squad_frame() -> pd.DataFrame:
    """A legal 15 with descending expected points within each position."""
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    xp = [5, 1, 6, 5, 4, 3, 1, 8, 7, 6, 5, 1, 9, 4, 2]
    return pd.DataFrame(
        {"position": positions, "expected_points": xp, "name": [f"p{i}" for i in range(15)]},
        index=pd.Index(range(1, 16), name="player_id"),
    )


def test_best_xi_is_legal_and_takes_the_best() -> None:
    frame = squad_frame()
    xi = best_xi(frame)
    shape = frame.loc[xi, "position"].value_counts().to_dict()
    assert len(xi) == 11 and shape["GK"] == 1
    assert shape["DEF"] >= 3 and shape["MID"] >= 2 and shape["FWD"] >= 1
    assert 2 not in xi, "the 1-point keeper sits"
    assert 13 in xi, "the best player plays"


def test_auto_sub_is_keeper_for_keeper_and_keeps_the_shape() -> None:
    positions = {1: "GK", 2: "DEF", 3: "DEF", 4: "DEF", 5: "MID", 6: "MID", 7: "MID", 8: "MID",
                 9: "MID", 10: "FWD", 11: "FWD", 12: "GK", 13: "FWD", 14: "MID", 15: "DEF"}
    xi = list(range(1, 12))
    bench = [12, 13, 14, 15]
    minutes = {i: 90 for i in range(1, 16)}

    # Keeper out: only the bench keeper may come on.
    subs = auto_subs(xi, bench, positions, {**minutes, 1: 0})
    assert subs == {1: 12}

    # A defender out with three at the back: only a defender keeps it legal,
    # even though a forward and a midfielder sit higher on the bench.
    subs = auto_subs(xi, bench, positions, {**minutes, 2: 0})
    assert subs == {2: 15}

    # A midfielder out: first outfield bench player who played.
    subs = auto_subs(xi, bench, positions, {**minutes, 5: 0, 13: 0})
    assert subs == {5: 14}, "the forward on the bench did not play, so he is skipped"

    # No one eligible: the starter stays, scoring nothing.
    assert auto_subs(xi, bench, positions, {**minutes, 1: 0, 12: 0}) == {}


def test_the_vice_takes_the_armband_and_its_multiplier() -> None:
    picks = [
        Pick(1, multiplier=3, slot=1, is_captain=True),     # triple captain, did not play
        Pick(2, multiplier=1, slot=2, is_vice=True),
        Pick(3, multiplier=1, slot=3, subbed_out=True),
        Pick(4, multiplier=0, slot=12, subbed_in=True),
    ]
    notes = settle(picks, {1: 0, 2: 90, 3: 0, 4: 30})
    by_id = {p.player_id: p.multiplier for p in picks}
    assert by_id == {1: 1, 2: 3, 3: 0, 4: 1}
    assert notes and "vice" in notes[0]

    # Already settled, as FPL may return it: nothing moves twice.
    settle(picks, {1: 0, 2: 90, 3: 0, 4: 30})
    assert {p.player_id: p.multiplier for p in picks} == by_id


def test_no_handover_when_the_vice_did_not_play_either() -> None:
    picks = [Pick(1, multiplier=2, slot=1, is_captain=True), Pick(2, multiplier=1, slot=2, is_vice=True)]
    assert settle(picks, {1: 0, 2: 0}) == []
    assert [p.multiplier for p in picks] == [2, 1]


class PicksClient:
    def entry_picks(self, team_id, gw):
        return {
            "active_chip": None,
            "automatic_subs": [{"element_in": 12, "element_out": 1}],
            "entry_history": {"points": 61, "points_on_bench": 4, "event_transfers_cost": 4},
            "picks": (
                [{"element": i, "position": i, "multiplier": 2 if i == 10 else 1,
                  "is_captain": i == 10, "is_vice_captain": i == 11} for i in range(1, 12)]
                + [{"element": i, "position": i, "multiplier": 0, "is_captain": False,
                    "is_vice_captain": False} for i in range(12, 16)]
            ),
        }


def test_squad_from_picks_carries_fpls_record() -> None:
    minutes = {i: 90 for i in range(1, 16)} | {1: 0}
    squad = squad_from_picks(PicksClient(), 123, 5, minutes)
    by_id = {p.player_id: p for p in squad.picks}
    assert squad.points == 61 and squad.transfer_cost == 4 and squad.bench_points == 4
    assert by_id[1].subbed_out and by_id[1].multiplier == 0
    assert by_id[12].subbed_in and by_id[12].multiplier == 1
    assert by_id[10].multiplier == 2


# -- the log and its summaries ----------------------------------------------------

def gameweek_table(gw: int, model_bias: float = 0.0) -> pd.DataFrame:
    """22 players, two of each position-ish, where FPL's figure is noisier than the model's."""
    positions = (["GK"] * 2 + ["DEF"] * 6 + ["MID"] * 8 + ["FWD"] * 6)
    n = len(positions)
    actual = pd.Series([float((i * 7 + gw) % 9) for i in range(n)]).to_numpy()
    table = pd.DataFrame({
        "name": [f"p{i}" for i in range(n)],
        "position": positions,
        "minutes_r3": [90.0] * (n - 2) + [0.0, 0.0],   # two without recent minutes
        "expected_points": actual + model_bias,
        "xp_fpl": actual[::-1],
        "actual": actual,
        "minutes": [90] * n,
    }, index=pd.Index(range(100, 100 + n), name="player_id"))
    for t in POINTS_TERMS:
        table[t] = 0.0
        table[f"act_{t}"] = 0.0
    table["bonus"], table["act_bonus"] = 0.5, 1.0
    table["act_other"] = 0.0
    return table


def test_the_log_replaces_a_reviewed_gameweek_rather_than_adding_it(config) -> None:
    append_log(config, 5, "2099-00", gameweek_table(5))
    append_log(config, 6, "2099-00", gameweek_table(6))
    again = append_log(config, 5, "2099-00", gameweek_table(5, model_bias=1.0))
    assert len(again) == 2 * len(gameweek_table(5))
    week5 = again[again["gw"] == 5]
    assert (week5["expected_points"] - week5["actual"]).round(9).eq(1.0).all(), "the second review won"


def test_calibration_sums_each_term(config) -> None:
    log_rows = append_log(config, 5, "2099-00", gameweek_table(5))
    c = calibration(log_rows)
    assert c.loc["bonus", "predicted"] == pytest.approx(0.5 * 22)
    assert c.loc["bonus", "ratio"] == pytest.approx(2.0)
    assert pd.isna(c.loc["goals", "ratio"]), "nothing predicted, so no ratio"


def test_head_to_head_scores_both_on_the_same_players(config) -> None:
    append_log(config, 5, "2099-00", gameweek_table(5))
    log_rows = append_log(config, 6, "2099-00", gameweek_table(6))
    h = head_to_head(log_rows)
    assert list(h.index) == ["5", "6", "all"]
    assert (h.loc[["5", "6"], "n"] == 20).all(), "the two without recent minutes are left out of both"
    assert h.loc["5", "model_mae"] == 0 and h.loc["5", "fpl_mae"] > 0
    assert h.loc["5", "model_rank"] == pytest.approx(1.0)
    assert h.loc["all", "model_xi"] == h.loc["5", "model_xi"] + h.loc["6", "model_xi"]


# -- rendering ------------------------------------------------------------------------

def test_render_covers_every_section_and_survives_a_cp1252_console(config) -> None:
    table = gameweek_table(5)
    table["fixture"] = "ABC (H) 1-0"
    table["diff"] = table["actual"] - table["expected_points"]
    log_rows = append_log(config, 5, "2099-00", table)
    picks = [Pick(pid, multiplier=1, slot=i + 1) for i, pid in enumerate(table.index[:15])]
    picks[0].is_captain, picks[0].multiplier = True, 2
    squad = Squad(picks=picks, source="test", points=40)
    review = Review(
        gw=5, season="2099-00", model="component", generated_at="2099-09-18T15:16:00+00:00",
        provisional=True, table=table, squad=squad,
        squad_rows=squad_table(squad, table, pd.DataFrame()),
        calibration=calibration(log_rows), head_to_head=head_to_head(log_rows),
    )
    text = render(review)
    for section in ("Gameweek 5 review", "Your squad", "Biggest over-deliveries",
                    "Model against FPL", "Scoring terms", "bonus can still move"):
        assert section in text, section
    assert "18 Sep 15:16" in text
    text.encode("cp1252")   # no arrows or other characters a Windows console cannot print
