"""Pre-deadline snapshots and how they reach the current season's training rows.

What matters: a snapshot round-trips; one captured after its deadline is
refused on the way in and ignored on the way out; and when overlaid on played
rows it fills `xP` for exactly the gameweeks it covers, replaces the set-piece
orders there, and leaves everything else as the live API had it.
"""

from __future__ import annotations

import textwrap

import pandas as pd
import pytest

from src.config import load_config
from src.data.current import played_frame
from src.data.snapshots import (
    apply_fixture_ratings,
    apply_snapshots,
    deadline_state,
    load_fixture_ratings,
    load_snapshots,
    snapshot_path,
    write_snapshot,
)

DEADLINES = {
    1: pd.Timestamp("2099-08-15 17:30", tz="UTC"),
    2: pd.Timestamp("2099-08-22 17:30", tz="UTC"),
    3: pd.Timestamp("2099-08-29 17:30", tz="UTC"),
}


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


def payload(*players: dict) -> dict:
    """A bootstrap-static payload with only the fields the snapshot keeps."""
    base = {
        "web_name": "x", "team": 1, "element_type": 3, "now_cost": 50, "ep_next": None,
        "ep_this": None, "chance_of_playing_next_round": None, "status": "a", "news": "",
        "penalties_order": None, "corners_and_indirect_freekicks_order": None, "form": "0.0",
    }
    return {"elements": [{**base, **p} for p in players]}


# -- writing and reading -------------------------------------------------------

def test_snapshot_round_trips_as_context_columns(config) -> None:
    before = DEADLINES[1] - pd.Timedelta(hours=3)
    write_snapshot(
        payload(
            {"id": 7, "ep_next": "6.0", "penalties_order": 2, "chance_of_playing_next_round": 100},
            {"id": 8, "ep_next": "3.5", "corners_and_indirect_freekicks_order": 1},
        ),
        gw=1, config=config, deadline=DEADLINES[1], captured_at=before,
    )
    assert snapshot_path(config, 1).exists()

    snaps = load_snapshots(config, deadlines=DEADLINES).set_index("player_id")
    assert list(snaps.columns) == ["gw", "xP", "value", "penalties_order", "setpiece_order", "chance_of_playing"]
    assert snaps.loc[7, "value"] == 50
    assert snaps.loc[7, "xP"] == 6.0, "ep_next is stored as a string by FPL; it comes back numeric"
    assert snaps.loc[7, "penalties_order"] == 2
    assert snaps.loc[7, "chance_of_playing"] == 100
    assert snaps.loc[8, "setpiece_order"] == 1
    assert pd.isna(snaps.loc[8, "penalties_order"]), "not in the order is NA, not 0"


def test_writing_after_the_deadline_is_refused(config) -> None:
    late = DEADLINES[1] + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="after it is not pre-deadline"):
        write_snapshot(payload({"id": 7}), gw=1, config=config, deadline=DEADLINES[1], captured_at=late)
    assert not snapshot_path(config, 1).exists()


def test_a_snapshot_captured_after_its_deadline_is_ignored_on_load(config) -> None:
    # Written without a deadline check, as an older version of the command could.
    write_snapshot(payload({"id": 7, "ep_next": "9.9"}), gw=1, config=config,
                   captured_at=DEADLINES[1] + pd.Timedelta(hours=2))
    write_snapshot(payload({"id": 7, "ep_next": "4.0"}), gw=2, config=config,
                   captured_at=DEADLINES[2] - pd.Timedelta(hours=2))

    snaps = load_snapshots(config, deadlines=DEADLINES)
    assert snaps["gw"].tolist() == [2], "the post-deadline gameweek is dropped, the other kept"
    assert snaps["xP"].tolist() == [4.0]


def test_no_snapshot_directory_is_an_empty_frame(config) -> None:
    snaps = load_snapshots(config, deadlines=DEADLINES)
    assert snaps.empty
    assert "xP" in snaps.columns


def test_rerunning_overwrites_the_same_gameweek(config) -> None:
    early = DEADLINES[1] - pd.Timedelta(days=2)
    later = DEADLINES[1] - pd.Timedelta(hours=1)
    write_snapshot(payload({"id": 7, "ep_next": "5.0"}), gw=1, config=config, captured_at=early)
    write_snapshot(payload({"id": 7, "ep_next": "2.0"}), gw=1, config=config, captured_at=later)
    snaps = load_snapshots(config, deadlines=DEADLINES)
    assert snaps["xP"].tolist() == [2.0], "the later capture wins: it has the later team news"


# -- overlaying onto played rows ----------------------------------------------

def played_rows() -> pd.DataFrame:
    """Two players, two gameweeks, with today's roles already merged on."""
    return pd.DataFrame({
        "gw": [1, 1, 2, 2],
        "player_id": [7, 8, 7, 8],
        "xP": [None] * 4,
        "penalties_order": [1, None, 1, None],   # today's live values
        "setpiece_order": [None, 1, None, 1],
        "chance_of_playing": [None] * 4,
        "value": [50, 50, 50, 50],
        "minutes": [90, 60, 90, 0],
    })


def test_overlay_fills_only_the_gameweeks_a_snapshot_covers() -> None:
    snaps = pd.DataFrame({
        "gw": [1, 1], "player_id": [7, 8],
        "xP": [6.0, 3.5], "value": [55.0, 45.0], "penalties_order": [2.0, None], "setpiece_order": [None, None],
        "chance_of_playing": [100.0, None],
    })
    out = apply_snapshots(played_rows(), snaps)

    assert out.index.equals(played_rows().index) and len(out) == 4
    gw1 = out[out["gw"] == 1].set_index("player_id")
    gw2 = out[out["gw"] == 2].set_index("player_id")

    assert gw1.loc[7, "xP"] == 6.0 and gw1.loc[8, "xP"] == 3.5
    assert gw2["xP"].isna().all(), "no snapshot for gameweek 2, so no xP"
    assert gw1.loc[7, "chance_of_playing"] == 100

    # Roles: replaced where covered — including NA, which is "not in the order
    # at that deadline" — and left as the live value where not.
    assert gw1.loc[7, "penalties_order"] == 2, "snapshot's order displaces today's"
    assert pd.isna(gw1.loc[8, "setpiece_order"]), "snapshot says not a taker then; live says one now"
    assert gw2.loc[7, "penalties_order"] == 1 and gw2.loc[8, "setpiece_order"] == 1

    # Nothing else moved — including the price, which the played rows already had.
    assert out["minutes"].tolist() == [90, 60, 90, 0]
    assert out["value"].tolist() == [50, 50, 50, 50], "a present price is never overwritten by the snapshot's"


def test_a_player_missing_from_the_snapshot_keeps_live_roles() -> None:
    snaps = pd.DataFrame({
        "gw": [1], "player_id": [7], "xP": [6.0], "value": [55.0], "penalties_order": [2.0],
        "setpiece_order": [None], "chance_of_playing": [None],
    })
    out = apply_snapshots(played_rows(), snaps).set_index(["gw", "player_id"])
    assert pd.isna(out.loc[(1, 8), "xP"])
    assert out.loc[(1, 8), "setpiece_order"] == 1, "joined after the snapshot: today's role is the best guess"


def test_empty_snapshots_leave_the_frame_alone() -> None:
    rows = played_rows()
    out = apply_snapshots(rows, pd.DataFrame(columns=["gw", "player_id", "xP"]))
    pd.testing.assert_frame_equal(out, rows)


# -- end to end through played_frame -------------------------------------------

class FakeClient:
    """Just the surface `played_frame` touches, for one team pair over two gameweeks."""

    def __init__(self, config):
        self.config = config

    def players(self) -> pd.DataFrame:
        return pd.DataFrame({
            "id": [7, 8], "element_type": [3, 4], "team": [1, 2],
            "web_name": ["Seven", "Eight"], "first_name": ["S", "E"], "second_name": ["Seven", "Eight"],
            "penalties_order": [1, None], "corners_and_indirect_freekicks_order": [None, 1],
        })

    def fixtures_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "id": [101, 102], "event": [1, 2], "team_h": [1, 2], "team_a": [2, 1],
            "team_h_difficulty": [3, 3], "team_a_difficulty": [3, 3],
            "kickoff_time": pd.to_datetime(["2099-08-16 14:00", "2099-08-23 14:00"], utc=True),
        })

    def events(self) -> pd.DataFrame:
        return pd.DataFrame({
            "id": [1, 2, 3], "finished": [True, True, False],
            "deadline_time": [DEADLINES[1], DEADLINES[2], DEADLINES[3]],
        })

    def player_histories(self, force: bool = False) -> pd.DataFrame:
        rows = []
        for pid, home in ((7, True), (8, False)):
            for gw, fixture in ((1, 101), (2, 102)):
                rows.append({
                    "element": pid, "fixture": fixture, "round": gw, "was_home": home if gw == 1 else not home,
                    "kickoff_time": pd.Timestamp("2099-08-16 14:00", tz="UTC") + pd.Timedelta(days=7 * (gw - 1)),
                    "team_h_score": 2, "team_a_score": 1, "minutes": 90, "total_points": 5, "value": 50,
                })
        return pd.DataFrame(rows)


def test_played_frame_carries_snapshot_xp_for_covered_gameweeks(config) -> None:
    write_snapshot(
        payload({"id": 7, "ep_next": "6.0", "penalties_order": 2}, {"id": 8, "ep_next": "3.5"}),
        gw=1, config=config, deadline=DEADLINES[1], captured_at=DEADLINES[1] - pd.Timedelta(hours=1),
    )
    frame = played_frame(FakeClient(config), season="2099-00").set_index(["gw", "player_id"])

    assert frame.loc[(1, 7), "xP"] == 6.0
    assert frame.loc[(1, 8), "xP"] == 3.5
    assert frame.loc[(2, 7), "xP"] is pd.NA or pd.isna(frame.loc[(2, 7), "xP"]), "gameweek 2 has no snapshot"
    assert frame.loc[(1, 7), "penalties_order"] == 2, "the order as it stood at that deadline"
    assert frame.loc[(2, 7), "penalties_order"] == 1, "today's order where nothing better exists"
    assert frame.loc[(1, 7), "total_points"] == 5, "outcomes untouched"


# -- fixture ratings and the deadline state -------------------------------------

FIXTURES = [
    {"id": 101, "event": 1, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 5},
    {"id": 102, "event": 2, "team_h": 2, "team_a": 1, "team_h_difficulty": 4, "team_a_difficulty": 3},
]


def test_fixture_ratings_round_trip_and_are_optional(config) -> None:
    before = DEADLINES[1] - pd.Timedelta(hours=1)
    write_snapshot(payload({"id": 7}), gw=1, config=config, captured_at=before, fixtures=FIXTURES)
    ratings = load_fixture_ratings(config, 1, deadlines=DEADLINES)
    assert ratings.set_index("id").loc[101, "team_a_difficulty"] == 5

    write_snapshot(payload({"id": 7}), gw=2, config=config, captured_at=DEADLINES[2] - pd.Timedelta(hours=1))
    assert load_fixture_ratings(config, 2, deadlines=DEADLINES).empty, "older snapshots carry no ratings"
    assert load_fixture_ratings(config, 3, deadlines=DEADLINES).empty, "no snapshot at all"


def test_each_side_takes_its_own_rating() -> None:
    rows = pd.DataFrame({
        "fixture_id": [101, 101, 999], "was_home": [True, False, True],
        "team_difficulty": [3, 3, 3], "opp_difficulty": [3, 3, 3],
    })
    out = apply_fixture_ratings(rows, pd.DataFrame(FIXTURES))
    assert out.loc[0, ["team_difficulty", "opp_difficulty"]].tolist() == [2, 5], "home side"
    assert out.loc[1, ["team_difficulty", "opp_difficulty"]].tolist() == [5, 2], "away side"
    assert out.loc[2, ["team_difficulty", "opp_difficulty"]].tolist() == [3, 3], "unknown fixture untouched"


def deadline_rows() -> pd.DataFrame:
    """Gameweek 1 played, gameweek 2 being predicted, carrying today's values."""
    return pd.DataFrame({
        "season": ["2099-00"] * 4, "gw": [1, 1, 2, 2], "player_id": [7, 8, 7, 8],
        "fixture_id": [101, 101, 102, 102], "was_home": [True, False, False, True],
        "team_difficulty": [9, 9, 9, 9], "opp_difficulty": [9, 9, 9, 9],
        "xP": [None, None, 9.9, 9.9], "value": [50, 50, 99, 99], "chance_of_playing": [None, None, 0, 0],
        "penalties_order": [3, 3, 3, 3], "setpiece_order": [3, 3, 3, 3],
        "total_points": [5, 2, None, None],
    })


def test_deadline_state_replays_what_a_live_build_saw(config) -> None:
    write_snapshot(
        payload({"id": 7, "ep_next": "6.0", "now_cost": 55, "penalties_order": 1},
                {"id": 8, "ep_next": "2.0", "now_cost": 45, "corners_and_indirect_freekicks_order": 2}),
        gw=2, config=config, captured_at=DEADLINES[2] - pd.Timedelta(hours=1), fixtures=FIXTURES,
    )
    out, notes = deadline_state(deadline_rows(), "2099-00", 2, config, deadlines=DEADLINES)
    assert notes == []
    out = out.set_index(["gw", "player_id"])

    # The prediction rows: the snapshot's xP, price and availability, nothing of today's.
    assert out.loc[(2, 7), "xP"] == 6.0 and out.loc[(2, 7), "value"] == 55
    assert pd.isna(out.loc[(2, 7), "chance_of_playing"]), "no flag at the deadline; today's 0 must not leak back"

    # Earlier rows with no snapshot of their own take the deadline's roles...
    assert out.loc[(1, 7), "penalties_order"] == 1 and pd.isna(out.loc[(1, 8), "penalties_order"])
    assert out.loc[(1, 8), "setpiece_order"] == 2
    # ...but keep their own outcomes and their missing xP.
    assert out.loc[(1, 7), "total_points"] == 5 and pd.isna(out.loc[(1, 7), "xP"])

    # Every fixture takes the deadline's ratings, each side its own.
    assert out.loc[(1, 7), ["team_difficulty", "opp_difficulty"]].tolist() == [2, 5]
    assert out.loc[(2, 7), ["team_difficulty", "opp_difficulty"]].tolist() == [3, 4]


def test_deadline_state_says_when_ratings_are_todays(config) -> None:
    write_snapshot(payload({"id": 7, "ep_next": "6.0"}), gw=2, config=config,
                   captured_at=DEADLINES[2] - pd.Timedelta(hours=1))
    out, notes = deadline_state(deadline_rows(), "2099-00", 2, config, deadlines=DEADLINES)
    assert len(notes) == 1 and "difficulty is today's" in notes[0]
    assert (out["team_difficulty"] == 9).all()


def test_deadline_state_needs_a_snapshot(config) -> None:
    with pytest.raises(FileNotFoundError):
        deadline_state(deadline_rows(), "2099-00", 2, config, deadlines=DEADLINES)
