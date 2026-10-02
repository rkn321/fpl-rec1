"""Pre-deadline snapshots of `bootstrap-static`, and how they reach training.

The model's most valuable inputs — FPL's own expected points (`ep_next`), the
availability flag, the set-piece orders — exist in the API only as *current*
values. The historical dataset carries `xP` for some gameweeks and not others
(27 of 38 are missing in 2025-26), and the live season's played rows carry none
at all, because `element-summary` has no expected-points history. Left alone,
the model trains on the season in progress blind to its most important feature
— and those are the only rows scored under this season's rules.

`fpl snapshot` fixes that going forward. Run before a deadline, it stores the
per-player fields under the gameweek about to be played. `load_snapshots` reads
them back and `apply_snapshots` overlays them onto that gameweek's played rows,
so from the first snapshot on, the current season trains with `xP` present.

A snapshot also keeps FPL's fixture difficulty ratings as they stood. FPL
revises those as the season goes and the fixtures endpoint rewrites them for
matches already played, so a gameweek rebuilt later sees its own fixtures — and
every earlier one — rated with hindsight. On GW5 of 2026-27 that, with the
set-piece orders, moved a rebuilt prediction by up to 1.8 points against what
the page showed at the deadline. `deadline_state` hands a rebuild what a live
build would have seen: the player fields, the roles and the ratings.

The one rule: a snapshot is only ever pre-deadline. `write_snapshot` refuses a
gameweek whose deadline has passed and `load_snapshots` drops a file captured
after its deadline, so a training row's `xP` is pre-deadline by construction
rather than by assumption — the doubt the brief raises about the historical
scrape (§6.1) does not apply here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..config import Config

log = logging.getLogger(__name__)

SNAPSHOT_SUBDIR = "snapshots"

# Per-player fields kept from the payload: what the model can use, plus enough
# to join and to read the file by eye.
SNAPSHOT_FIELDS: list[str] = [
    "id", "web_name", "team", "element_type", "now_cost", "ep_next", "ep_this",
    "chance_of_playing_next_round", "chance_of_playing_this_round", "status", "news",
    "penalties_order", "corners_and_indirect_freekicks_order", "form",
    "selected_by_percent", "transfers_in_event", "transfers_out_event",
]

# Per-fixture fields kept, for the difficulty ratings as they stood.
FIXTURE_FIELDS: list[str] = ["id", "event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]

# Snapshot field -> canonical context column it feeds.
SNAPSHOT_TO_CONTEXT: dict[str, str] = {
    "ep_next": "xP",
    "now_cost": "value",
    "penalties_order": "penalties_order",
    "corners_and_indirect_freekicks_order": "setpiece_order",
    "chance_of_playing_next_round": "chance_of_playing",
}

# Columns a snapshot *replaces* on the rows it covers, rather than fills. NA in
# a snapshot's set-piece order means "not in the order at that deadline", which
# is information; the live value it displaces is only "in the order today".
REPLACED_COLS: tuple[str, ...] = ("penalties_order", "setpiece_order")
# Columns a snapshot fills only where the row has nothing. Played rows already
# carry their price from `element-summary`; prediction rows being rebuilt for a
# past deadline (see `review.reconstruct_predictions`) are blanked first.
FILLED_COLS: tuple[str, ...] = ("xP", "value", "chance_of_playing")


def snapshot_dir(config: Config) -> Path:
    return config.data_dir / SNAPSHOT_SUBDIR


def snapshot_path(config: Config, gw: int) -> Path:
    return snapshot_dir(config) / f"bootstrap_gw{int(gw):02d}.json"


def write_snapshot(
    payload: dict[str, Any],
    gw: int,
    config: Config,
    deadline: pd.Timestamp | None = None,
    captured_at: pd.Timestamp | None = None,
    fixtures: list[dict[str, Any]] | None = None,
) -> Path:
    """Store the per-player fields of a `bootstrap-static` payload under `gw`,
    with the difficulty ratings from a `fixtures/` payload if one is given.

    Idempotent per gameweek: a second run before the same deadline overwrites,
    which is what you want — the later capture has the later team news.
    Refuses a gameweek whose deadline has already passed.
    """
    captured_at = captured_at or pd.Timestamp.now(tz="UTC")
    if deadline is not None and captured_at >= deadline:
        raise ValueError(
            f"gameweek {gw} deadline was {deadline:%a %d %b %H:%M UTC}; a snapshot "
            "taken after it is not pre-deadline and would not be used"
        )

    body: dict[str, Any] = {
        "gameweek": int(gw),
        "captured_at": captured_at.isoformat(),
        "players": [{k: p.get(k) for k in SNAPSHOT_FIELDS} for p in payload["elements"]],
    }
    if fixtures is not None:
        body["fixtures"] = [{k: f.get(k) for k in FIXTURE_FIELDS} for f in fixtures]
    path = snapshot_path(config, gw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _read_one(path: Path) -> tuple[int, pd.Timestamp, pd.DataFrame, pd.DataFrame | None] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        gw = int(raw["gameweek"])
        captured = pd.Timestamp(raw["captured_at"])
        players = pd.DataFrame(raw["players"])
        fixtures = pd.DataFrame(raw["fixtures"]) if raw.get("fixtures") else None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("skipping unreadable snapshot %s: %s", path.name, exc)
        return None
    if captured.tzinfo is None:
        captured = captured.tz_localize("UTC")
    return gw, captured, players, fixtures


def _pre_deadline(
    path: Path, gw: int, captured: pd.Timestamp, deadlines: Mapping[int, pd.Timestamp] | None
) -> bool:
    deadline = (deadlines or {}).get(gw)
    if deadline is not None and captured >= deadline:
        log.warning(
            "snapshot %s was captured %s, after the gameweek %d deadline %s; ignored",
            path.name, captured.strftime("%Y-%m-%d %H:%M"), gw,
            pd.Timestamp(deadline).strftime("%Y-%m-%d %H:%M"),
        )
        return False
    return True


def load_snapshots(
    config: Config, deadlines: Mapping[int, pd.Timestamp] | None = None
) -> pd.DataFrame:
    """Every snapshot on disk as one frame: a row per (gw, player_id).

    Columns are `gw`, `player_id` and the context columns in
    `SNAPSHOT_TO_CONTEXT`. With `deadlines` (gameweek -> deadline, UTC), a
    snapshot captured at or after its own deadline is dropped with a warning:
    whatever it holds may already know something about the matches.
    """
    empty = pd.DataFrame(columns=["gw", "player_id", *SNAPSHOT_TO_CONTEXT.values()])
    folder = snapshot_dir(config)
    if not folder.is_dir():
        return empty

    frames: list[pd.DataFrame] = []
    for path in sorted(folder.glob("bootstrap_gw*.json")):
        read = _read_one(path)
        if read is None:
            continue
        gw, captured, players, _ = read
        if not _pre_deadline(path, gw, captured, deadlines):
            continue

        frame = players.reindex(columns=["id", *SNAPSHOT_TO_CONTEXT]).rename(
            columns={"id": "player_id", **SNAPSHOT_TO_CONTEXT}
        )
        frame.insert(0, "gw", gw)
        frames.append(frame)

    if not frames:
        return empty

    out = pd.concat(frames, ignore_index=True)
    out["gw"] = out["gw"].astype(int)
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    for col in SNAPSHOT_TO_CONTEXT.values():
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["player_id"]).drop_duplicates(["gw", "player_id"], keep="last")
    log.info("loaded %d snapshot gameweeks: %s", out["gw"].nunique(), sorted(int(g) for g in out["gw"].unique()))
    return out.reset_index(drop=True)


def apply_snapshots(df: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Overlay snapshot context onto played rows, matched on `(gw, player_id)`.

    `xP` and `chance_of_playing` are filled where the row has a snapshot. The
    set-piece orders are *replaced* on covered rows (see `REPLACED_COLS`) and
    left as the live values everywhere else — a gameweek with no snapshot, or a
    player who joined after it was taken, keeps the best available guess.
    """
    if snapshots is None or snapshots.empty or df.empty:
        return df

    snap = snapshots.rename(columns={c: f"_snap_{c}" for c in SNAPSHOT_TO_CONTEXT.values()})
    left = df.copy()
    left["_gw_key"] = pd.to_numeric(left["gw"], errors="coerce").astype("Int64")
    left["_pid_key"] = pd.to_numeric(left["player_id"], errors="coerce").astype("Int64")
    snap = snap.rename(columns={"gw": "_gw_key", "player_id": "_pid_key"})
    snap["_gw_key"] = snap["_gw_key"].astype("Int64")
    snap["_covered"] = True

    merged = left.merge(snap, on=["_gw_key", "_pid_key"], how="left", validate="many_to_one")
    covered = merged["_covered"].fillna(False).astype(bool)

    for col in FILLED_COLS:
        if col not in merged.columns:
            merged[col] = pd.NA
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged[col] = merged[col].where(merged[col].notna(), merged[f"_snap_{col}"])
    for col in REPLACED_COLS:
        if col not in merged.columns:
            merged[col] = pd.NA
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged[col] = merged[f"_snap_{col}"].where(covered, merged[col])

    n_rows = int(covered.sum())
    if n_rows:
        log.info(
            "snapshots cover %d of %d played rows (%d gameweeks)",
            n_rows, len(merged), merged.loc[covered, "_gw_key"].nunique(),
        )
    drop = [c for c in merged.columns if c.startswith("_snap_")] + ["_covered", "_gw_key", "_pid_key"]
    merged = merged.drop(columns=drop)
    merged.index = df.index
    return merged


def load_fixture_ratings(
    config: Config, gw: int, deadlines: Mapping[int, pd.Timestamp] | None = None
) -> pd.DataFrame:
    """Difficulty ratings for every fixture, as they stood at `gw`'s deadline.

    Empty when `gw` has no snapshot, or one taken before snapshots kept them.
    """
    empty = pd.DataFrame(columns=FIXTURE_FIELDS)
    path = snapshot_path(config, gw)
    read = _read_one(path) if path.exists() else None
    if read is None:
        return empty
    snap_gw, captured, _, fixtures = read
    if fixtures is None or not _pre_deadline(path, snap_gw, captured, deadlines):
        return empty
    out = fixtures.reindex(columns=FIXTURE_FIELDS)
    for col in FIXTURE_FIELDS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["id"]).reset_index(drop=True)


def apply_fixture_ratings(df: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Set `team_difficulty` / `opp_difficulty` from `ratings` wherever the fixture is in it.

    Each side's rating is its own: at home a player's difficulty is
    `team_h_difficulty` and the opponent's is `team_a_difficulty`, and the
    other way round away — the same derivation `src/data/current.py` uses.
    """
    if ratings is None or ratings.empty or df.empty:
        return df
    by_id = ratings.set_index(ratings["id"].astype(int))
    fixture = pd.to_numeric(df["fixture_id"], errors="coerce")
    known = fixture.isin(by_id.index).to_numpy()
    if not known.any():
        return df
    out = df.copy()
    rows = by_id.loc[fixture[known].astype(int).to_numpy()]
    home = out.loc[known, "was_home"].astype(bool).to_numpy()
    h = rows["team_h_difficulty"].to_numpy(dtype=float)
    a = rows["team_a_difficulty"].to_numpy(dtype=float)
    out.loc[known, "team_difficulty"] = np.where(home, h, a)
    out.loc[known, "opp_difficulty"] = np.where(home, a, h)
    return out


def deadline_state(
    df: pd.DataFrame,
    season: str,
    gw: int,
    config: Config,
    deadlines: Mapping[int, pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Put `season`'s rows back the way a live build at `gw`'s deadline saw them.

    `df` is the raw canonical frame with `gw`'s prediction rows in it. A live
    build takes today's set-piece orders for every played row without a
    snapshot of its own, and today's difficulty ratings for every fixture.
    Rebuilt later, "today" has moved on, so both come from `gw`'s snapshot
    instead, and the prediction rows also take its `xP`, price and
    availability. Returns the frame and caveats worth printing: a snapshot
    missing a piece means that piece is today's value, not the deadline's.
    """
    snapshots = load_snapshots(config, deadlines=deadlines)
    own = snapshots[snapshots["gw"] == int(gw)]
    if own.empty:
        raise FileNotFoundError(f"no pre-deadline snapshot for gameweek {gw}")

    in_season = (df["season"] == season).to_numpy()
    gws = pd.to_numeric(df["gw"], errors="coerce").to_numpy()
    target = in_season & (gws == gw)

    # The prediction rows: everything the snapshot holds, nothing of today's.
    rows = df[target].copy()
    for col in FILLED_COLS:
        rows[col] = np.nan
    out = pd.concat([df[~target], apply_snapshots(rows, own)]).loc[df.index]

    # Earlier rows with no snapshot of their own: a live build gave them the
    # roles of the day, which is what this gameweek's snapshot recorded.
    covered = {int(g) for g in snapshots["gw"].unique()}
    earlier = in_season & (gws < gw) & ~np.isin(gws, list(covered))
    roles = own.set_index(own["player_id"].astype(int))
    pid = out.loc[earlier, "player_id"].astype(int).to_numpy()
    present = np.isin(pid, roles.index)
    for col in REPLACED_COLS:
        values = pd.to_numeric(out.loc[earlier, col], errors="coerce").to_numpy(dtype=float, copy=True)
        values[present] = roles[col].reindex(pid[present]).to_numpy(dtype=float)
        out.loc[earlier, col] = values

    notes: list[str] = []
    ratings = load_fixture_ratings(config, gw, deadlines=deadlines)
    if ratings.empty:
        notes.append(
            f"the gameweek {gw} snapshot predates fixture ratings being kept, so difficulty is "
            "today's; FPL revises it with hindsight, so figures can differ from what was shown"
        )
    else:
        out.loc[in_season] = apply_fixture_ratings(out.loc[in_season], ratings)
    return out, notes


def local_captured_at(config: Config, gw: int) -> pd.Timestamp | None:
    """When the snapshot on disk for `gw` was taken, or None if there is none."""
    path = snapshot_path(config, gw)
    read = _read_one(path) if path.exists() else None
    return None if read is None else read[1]


def keep_if_newer(config: Config, raw: bytes) -> int | None:
    """Store a snapshot captured elsewhere if it is newer than the one on disk.

    The same rule as rerunning `fpl snapshot` before a deadline: for each
    gameweek the latest pre-deadline capture wins, because it has the latest
    team news. Returns the gameweek written, or None if the local copy was
    already as new.
    """
    body = json.loads(raw.decode("utf-8"))
    gw = int(body["gameweek"])
    captured = pd.Timestamp(body["captured_at"])
    if captured.tzinfo is None:
        captured = captured.tz_localize("UTC")
    if not isinstance(body.get("players"), list) or not body["players"]:
        raise ValueError(f"snapshot for gameweek {gw} has no players")

    local = local_captured_at(config, gw)
    if local is not None and local >= captured:
        return None
    path = snapshot_path(config, gw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")
    return gw
