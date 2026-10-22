"""A manager's real team, imported from FPL by team id.

The page used to start from fifteen names typed into `config.local.yaml`, a
bank typed by hand and a free-transfer count remembered by the manager. FPL
publishes nearly all of it for any team id, without signing in, as it stood at
the last deadline:

- the fifteen, from that gameweek's picks — or, if it was a Free Hit week, the
  gameweek before, because the Free Hit squad reverts
- the bank, from the same gameweek's entry history
- what was paid for each player, from the transfer list (skipping Free Hit
  weeks, whose transfers reverted), or for one kept since the squad was first
  picked, his price in the gameweek the manager started. The page sells at
  FPL's rule — half the rise, rounded down — so this is what makes a sale's
  proceeds right, not just the bank's total
- the chips played, and each gameweek's transfers and hits

Two things it cannot see. Transfers made for the coming gameweek are private
until its deadline, so a team imported mid-week is last deadline's team. And
FPL does not publish the free-transfer count: it is worked out from the
transfer history by the season's rules (one a week, up to five banked, kept
through a Wildcard or Free Hit), which misses any one-off top-up FPL hands out,
so the page labels it an estimate and lets it be corrected.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from .fpl_api import FPLClient

log = logging.getLogger(__name__)

MAX_FREE_TRANSFERS = 5
FIRST_HALF_LAST_GW = 19          # chips come in two sets; the first expires here
TRANSFER_KEEPING_CHIPS = {"wildcard", "freehit"}


@dataclass
class ImportedTeam:
    team_id: int
    name: str
    gw: int                       # the deadline this squad stood at
    picks: list[int]              # FPL's order: 1-11 the XI, 12-15 the bench
    bank: int                     # tenths of a million
    free_transfers: int           # estimated for the coming gameweek
    buy_prices: dict[int, int]    # player id -> what was paid, in tenths
    chips_used: dict[str, int] = field(default_factory=dict)   # "1:wildcard" -> gameweek
    history: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["buy_prices"] = {str(k): v for k, v in self.buy_prices.items()}
        return out


def chip_key(name: str, gw: int) -> str:
    return f"{1 if gw <= FIRST_HALF_LAST_GW else 2}:{name}"


def estimate_free_transfers(
    history: list[dict[str, Any]], chips: dict[int, str], started: int, through: int,
) -> int:
    """Free transfers in hand for the gameweek after `through`.

    The squad picked in the starting gameweek is free; each gameweek after it
    adds one transfer, banking up to five. Transfers made under a Wildcard or
    Free Hit cost none, and the bank survives the chip. Beyond the allowance a
    transfer is a hit and leaves none banked.
    """
    by_gw = {int(r["event"]): int(r.get("event_transfers", 0) or 0) for r in history}
    ft = 1
    for gw in range(started + 1, through + 1):
        used = 0 if chips.get(gw) in TRANSFER_KEEPING_CHIPS else by_gw.get(gw, 0)
        ft = min(MAX_FREE_TRANSFERS, max(0, ft - used) + 1)
    return ft


def import_team(client: FPLClient, team_id: int, last_gw: int) -> ImportedTeam:
    """The team as it stood at the `last_gw` deadline, ready to plan the next one."""
    summary = client.entry(team_id)
    history = client.entry_history(team_id)
    chips = {int(c["event"]): str(c["name"]) for c in history.get("chips", [])}
    started = int(summary.get("started_event") or 1)
    if last_gw < started:
        raise ValueError(f"team {team_id} started in GW{started}; there is no squad at GW{last_gw} yet")

    notes: list[str] = []
    squad_gw = last_gw
    if chips.get(last_gw) == "freehit":
        squad_gw = last_gw - 1
        notes.append(f"GW{last_gw} was a Free Hit, so this is the GW{squad_gw} squad it reverts to")

    picks_data = client.entry_picks(team_id, squad_gw)
    picks = [int(p["element"]) for p in sorted(picks_data["picks"], key=lambda p: int(p["position"]))]
    bank = int((picks_data.get("entry_history") or {}).get("bank", 0) or 0)

    # The latest real purchase of each player in the squad. Free Hit weeks'
    # transfers are in the list too, but they reverted, so they do not count.
    paid: dict[int, int] = {}
    for t in sorted(client.entry_transfers(team_id), key=lambda t: (int(t["event"]), str(t.get("time", "")))):
        gw = int(t["event"])
        if gw <= squad_gw and chips.get(gw) != "freehit":
            paid[int(t["element_in"])] = int(t["element_in_cost"])

    prices = client.players().set_index("id")["now_cost"].astype(int).to_dict()
    buy: dict[int, int] = {}
    for pid in picks:
        if pid in paid:
            buy[pid] = paid[pid]
            continue
        # Kept since the first squad: his price in the gameweek the manager started.
        rows = client.element_summary(pid).get("history", [])
        first = [r for r in rows if int(r.get("round", 0)) >= started]
        buy[pid] = int(first[0]["value"]) if first else int(prices.get(pid, 0))

    rows = sorted(history.get("current", []), key=lambda r: int(r["event"]))
    season = [
        {
            "gw": int(r["event"]),
            "transfers": int(r.get("event_transfers", 0) or 0),
            "hit": int(r.get("event_transfers_cost", 0) or 0),
            "chip": chips.get(int(r["event"])),
            "value": int(r.get("value", 0) or 0),
        }
        for r in rows if int(r["event"]) <= last_gw
    ]

    return ImportedTeam(
        team_id=int(team_id),
        name=str(summary.get("name") or f"Team {team_id}"),
        gw=last_gw,
        picks=picks,
        bank=bank,
        free_transfers=estimate_free_transfers(rows, chips, started, last_gw),
        buy_prices=buy,
        chips_used={chip_key(name, gw): gw for gw, name in chips.items()},
        history=season,
        notes=notes,
    )
