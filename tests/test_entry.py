"""Importing a manager's real team from FPL by team id."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.entry import estimate_free_transfers, import_team


def rows(*transfers: int) -> list[dict]:
    """Entry-history rows for GW1.., one per gameweek, with these transfer counts."""
    return [{"event": gw, "event_transfers": n} for gw, n in enumerate(transfers, start=1)]


def test_one_free_transfer_a_week_banks_up_to_five() -> None:
    # Squad picked in GW1, then nothing: GW2 has 1, and each week adds one, capped at 5.
    assert estimate_free_transfers(rows(0), {}, started=1, through=1) == 1
    assert estimate_free_transfers(rows(0, 0), {}, started=1, through=2) == 2
    assert estimate_free_transfers(rows(*[0] * 9), {}, started=1, through=9) == 5


def test_transfers_spend_the_bank_and_a_hit_leaves_none() -> None:
    assert estimate_free_transfers(rows(0, 0, 2), {}, started=1, through=3) == 1   # 2 banked, both used
    assert estimate_free_transfers(rows(0, 3), {}, started=1, through=2) == 1      # 1 free + 2 hits


def test_wildcard_and_free_hit_keep_the_bank() -> None:
    chips = {3: "wildcard", 5: "freehit"}
    # Even if FPL counted the chip weeks' transfers, they cost nothing.
    assert estimate_free_transfers(rows(0, 0, 9, 0, 9, 1), chips, started=1, through=6) == 5


def test_a_late_starter_counts_from_their_first_week() -> None:
    assert estimate_free_transfers(rows(0, 0, 0, 0), {}, started=3, through=4) == 2


class FakeClient:
    """FPL's team endpoints for one manager, who played a Free Hit in GW5."""

    def __init__(self, chips: list[dict] | None = None):
        self.chips = chips if chips is not None else [{"name": "freehit", "event": 5}]

    def entry(self, team_id):
        return {"name": "Test XI", "started_event": 1}

    def entry_history(self, team_id):
        return {
            "current": [{"event": gw, "event_transfers": 1 if gw == 4 else 0, "event_transfers_cost": 4 if gw == 4 else 0,
                         "value": 1000 + gw, "bank": gw} for gw in range(1, 6)],
            "chips": self.chips,
        }

    def entry_transfers(self, team_id):
        return [
            {"element_in": 2, "element_in_cost": 61, "element_out": 9, "event": 4, "time": "2026-09-10"},
            {"element_in": 7, "element_in_cost": 80, "element_out": 2, "event": 5, "time": "2026-09-17"},  # Free Hit
        ]

    def entry_picks(self, team_id, gw):
        squad = [1, 2, 3] if gw == 4 else [1, 7, 3]          # GW5 is the Free Hit squad
        return {"picks": [{"element": pid, "position": i + 1} for i, pid in enumerate(squad)],
                "entry_history": {"bank": 12 if gw == 4 else 3}}

    def players(self):
        return pd.DataFrame({"id": [1, 2, 3, 7], "now_cost": [45, 64, 100, 82]})

    def element_summary(self, pid):
        return {"history": [{"round": 1, "value": {1: 40, 2: 60, 3: 95, 7: 78}[pid]}]}


def test_a_free_hit_week_imports_the_squad_it_reverts_to() -> None:
    team = import_team(FakeClient(), 99, last_gw=5)
    assert team.picks == [1, 2, 3]            # GW4's squad, not the Free Hit's
    assert team.bank == 12                    # and GW4's bank
    assert team.notes and "Free Hit" in team.notes[0]


def test_purchase_prices_come_from_transfers_then_the_first_squad() -> None:
    team = import_team(FakeClient(), 99, last_gw=5)
    assert team.buy_prices[2] == 61           # bought in GW4
    assert team.buy_prices[1] == 40           # kept since GW1: his GW1 price
    assert team.buy_prices[3] == 95
    assert 7 not in team.buy_prices           # the Free Hit signing reverted


def test_chips_history_and_free_transfers_come_through() -> None:
    team = import_team(FakeClient(chips=[{"name": "wildcard", "event": 3}]), 99, last_gw=5)
    assert team.chips_used == {"1:wildcard": 3}
    assert [h["gw"] for h in team.history] == [1, 2, 3, 4, 5]
    assert team.history[3]["hit"] == 4
    # In hand for GW2 1, GW3 2, GW4 3 (the Wildcard keeps them), GW5 3 (one used), GW6 4.
    assert team.free_transfers == 4
    assert team.picks == [1, 7, 3]            # no Free Hit, so GW5's own squad
    assert team.to_json()["buy_prices"]["7"] == 80


def test_no_squad_before_the_manager_started() -> None:
    class Late(FakeClient):
        def entry(self, team_id):
            return {"name": "Late", "started_event": 6}
    with pytest.raises(ValueError):
        import_team(Late(), 99, last_gw=5)
