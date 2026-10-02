"""Post-gameweek review: what the model said before the deadline, against what happened.

Two jobs. The first is the Monday question — how did my players do, and where
was the model wrong — answered from the prediction the model actually made
before the deadline (`predictions_gw{N}.csv`, stored by `src/predict.py`
whenever the page or `fpl predict` is built in time), never from one rebuilt
afterwards. For gameweeks before predictions were stored, the model is refitted
as of that deadline with the gameweek's snapshot supplying the pre-deadline
inputs, and the output says so.

The second is evidence. Every review appends predicted and actual points per
scoring term for every player to `data/processed/review_log.csv`, and prints
two season-to-date summaries from it: which scoring terms run hot or cold, and
how the model is doing against FPL's own figure on the same players. One
gameweek is an anecdote; the log is what a claim like "the bonus term
overweights X" should be checked against before anything is changed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import Config, load_config
from .data.fpl_api import FPLAPIError, FPLClient
from .data.schema import POSITIONS
from .data.snapshots import deadline_state
from .evaluate import playing_filter
from .metrics import ACTUAL, PRED, mae, spearman_by_position
from .models.combine import POINTS_TERMS
from .predict import PREDICTIONS_NAME, fit_predict, load_predictions

log = logging.getLogger(__name__)

LOG_NAME = "review_log.csv"

# How FPL labels each scoring event in `event/{gw}/live/`, mapped onto the
# model's terms. Own goals and missed penalties have no term — the model treats
# them as noise — so they land in `other` and the actual total still reconciles.
EXPLAIN_TO_TERM: dict[str, str] = {
    "minutes": "appearance",
    "goals_scored": "goals",
    "assists": "assists",
    "clean_sheets": "clean_sheet",
    "goals_conceded": "goals_conceded",
    "saves": "saves",
    "penalties_saved": "saves",
    "defensive_contribution": "defcon",
    "bonus": "bonus",
    "yellow_cards": "cards",
    "red_cards": "cards",
    "own_goals": "other",
    "penalties_missed": "other",
}
ACTUAL_TERMS: tuple[str, ...] = (*POINTS_TERMS, "other")

# Legal starting shapes: one keeper, then 3-5 / 2-5 / 1-3.
FORMATIONS: tuple[tuple[int, int, int], ...] = tuple(
    (d, m, f)
    for d in range(3, 6)
    for m in range(2, 6)
    for f in range(1, 4)
    if d + m + f == 10
)
MIN_STARTERS = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}


# -- the prediction as made ----------------------------------------------------

def reconstruct_predictions(
    config: Config, client: FPLClient, gw: int, model: str = "component"
) -> tuple[pd.DataFrame, list[str]]:
    """Refit the model as of `gw`'s deadline, when no stored prediction exists.

    Training is everything before `gw`; every current-season row is put back
    the way a live build that day saw it (`deadline_state`) — the gameweek's
    `xP`, price, availability and set-piece orders, and the fixture difficulty
    ratings before FPL revised them. Without a snapshot there is no honest way
    to do this, so it refuses rather than guess. Returns the record and any
    caveats about how faithful the rebuild is.
    """
    from . import pipeline
    from .data.current import deadlines
    from .features.build import build_player_gameweek

    season = config.season_current
    raw = pipeline.load_raw(config, include_current=True, upcoming_gw=gw, client=client)
    current = (raw["season"] == season).to_numpy()
    at_or_after = (pd.to_numeric(raw["gw"], errors="coerce").fillna(-1) >= gw).to_numpy()
    known = raw["total_points"].notna().to_numpy()
    # Results from this gameweek on were not known at the deadline.
    raw = raw[~(current & at_or_after & known)].reset_index(drop=True)

    try:
        raw, notes = deadline_state(raw, season, gw, config, deadlines=deadlines(client))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"no stored prediction for gameweek {gw}, and {exc} to rebuild it from. From now "
            f"on, building the page or running `fpl predict` before a deadline stores "
            f"{PREDICTIONS_NAME.format(gw=gw)}."
        ) from None

    frame, feature_cols = build_player_gameweek(
        raw, windows=list(config.features["windows"]),
        expanding=bool(config.features.get("expanding", True)),
    )
    return fit_predict(frame, feature_cols, season, gw, model=model), notes


# -- what happened ----------------------------------------------------------------

def parse_live(live: Mapping[str, Any]) -> pd.DataFrame:
    """Per player: points scored, split by scoring term, plus the raw stats.

    The split comes from FPL's own `explain` block, so it reconciles with the
    total exactly — including the points no term predicts, which go to `other`.
    """
    rows: list[dict[str, Any]] = []
    for element in live.get("elements", []):
        stats = element.get("stats", {})

        def stat(key: str) -> int:
            return int(float(stats.get(key, 0) or 0))

        row: dict[str, Any] = {
            "player_id": int(element["id"]),
            "minutes": stat("minutes"),
            "actual": float(stats.get("total_points", 0) or 0),
            "goals_scored": stat("goals_scored"),
            "assists_n": stat("assists"),
            "clean_sheets": stat("clean_sheets"),
            "bonus_n": stat("bonus"),
            "bps": stat("bps"),
            "yellow": stat("yellow_cards"),
            "red": stat("red_cards"),
            "saves_n": stat("saves"),
            "xg": float(stats.get("expected_goals", 0) or 0),
            "xa": float(stats.get("expected_assists", 0) or 0),
        }
        for term in ACTUAL_TERMS:
            row[f"act_{term}"] = 0.0
        for fixture in element.get("explain", []):
            for item in fixture.get("stats", []):
                term = EXPLAIN_TO_TERM.get(item.get("identifier"), "other")
                row[f"act_{term}"] += float(item.get("points", 0) or 0)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["minutes", "actual"], index=pd.Index([], name="player_id"))
    return pd.DataFrame(rows).set_index("player_id")


def fixture_labels(client: FPLClient, gw: int) -> dict[int, str]:
    """team_id -> 'SUN (H) 5-3', fixtures joined with ', ' on a double."""
    teams = client.teams().set_index("id")["short_name"].to_dict()
    fx = client.fixtures_frame()
    fx = fx[fx["event"] == gw]
    labels: dict[int, list[str]] = {}
    for f in fx.itertuples():
        played = pd.notna(f.team_h_score) and pd.notna(f.team_a_score)
        home = f"{int(f.team_h_score)}-{int(f.team_a_score)}" if played else "—"
        away = f"{int(f.team_a_score)}-{int(f.team_h_score)}" if played else "—"
        labels.setdefault(int(f.team_h), []).append(f"{teams.get(int(f.team_a), '?')} (H) {home}")
        labels.setdefault(int(f.team_a), []).append(f"{teams.get(int(f.team_h), '?')} (A) {away}")
    return {t: ", ".join(v) for t, v in labels.items()}


def join_actuals(record: pd.DataFrame, actual: pd.DataFrame, labels: Mapping[int, str]) -> pd.DataFrame:
    """Every predicted player with what they scored; no row in `actual` means no minutes."""
    table = record.join(actual, how="left")
    table["minutes"] = table["minutes"].fillna(0).astype(int)
    table["actual"] = table["actual"].fillna(0.0)
    for term in ACTUAL_TERMS:
        col = f"act_{term}"
        table[col] = table[col].fillna(0.0) if col in table.columns else 0.0
    table["fixture"] = table["team_id"].map(lambda t: labels.get(int(t), "—") if pd.notna(t) else "—")
    table["diff"] = table["actual"] - table["expected_points"]
    return table


# -- the squad --------------------------------------------------------------------

@dataclass
class Pick:
    player_id: int
    multiplier: int          # 0 bench, 1 started, 2 captain, 3 triple captain
    slot: int                # 1-11 the XI, 12-15 bench order
    is_captain: bool = False
    is_vice: bool = False
    subbed_in: bool = False  # came off the bench by auto-sub
    subbed_out: bool = False


@dataclass
class Squad:
    picks: list[Pick]
    source: str                      # where the lineup came from, for the header
    points: int | None = None        # FPL's own total for the week, if known
    bench_points: int | None = None
    transfer_cost: int = 0
    chip: str | None = None
    notes: list[str] = field(default_factory=list)


def best_xi(frame: pd.DataFrame, col: str = "expected_points") -> list[int]:
    """The eleven with the most of `col` in a legal shape."""
    by_pos = {
        pos: frame[frame["position"] == pos].sort_values(col, ascending=False).index.tolist()
        for pos in POSITIONS
    }
    best: tuple[float, list[int]] | None = None
    for d, m, f in FORMATIONS:
        if len(by_pos["GK"]) < 1 or len(by_pos["DEF"]) < d or len(by_pos["MID"]) < m or len(by_pos["FWD"]) < f:
            continue
        ids = by_pos["GK"][:1] + by_pos["DEF"][:d] + by_pos["MID"][:m] + by_pos["FWD"][:f]
        total = float(frame.loc[ids, col].sum())
        if best is None or total > best[0]:
            best = (total, ids)
    return best[1] if best else frame.index[:11].tolist()


def auto_subs(
    xi: list[int], bench: list[int], positions: Mapping[int, str], minutes: Mapping[int, int]
) -> dict[int, int]:
    """FPL's auto-sub rule: `{starter_out: bench_in}`.

    A starter with no minutes is replaced by the first bench player, in bench
    order, who played and keeps the XI legal — keeper for keeper, and at least
    three defenders, two midfielders and a forward.
    """
    def played(pid: int) -> bool:
        return int(minutes.get(pid, 0) or 0) > 0

    shape: dict[str, int] = {}
    for pid in xi:
        shape[positions[pid]] = shape.get(positions[pid], 0) + 1

    subs: dict[int, int] = {}
    for out_id in xi:
        if played(out_id):
            continue
        out_pos = positions[out_id]
        for in_id in bench:
            if in_id in subs.values() or not played(in_id):
                continue
            in_pos = positions[in_id]
            if (out_pos == "GK") != (in_pos == "GK"):
                continue
            trial = dict(shape)
            trial[out_pos] -= 1
            trial[in_pos] = trial.get(in_pos, 0) + 1
            if all(trial.get(p, 0) >= n for p, n in MIN_STARTERS.items()):
                subs[out_id] = in_id
                shape = trial
                break
    return subs


def settle(picks: list[Pick], minutes: Mapping[int, int]) -> list[str]:
    """Apply FPL's end-of-gameweek rules to the multipliers, in place.

    A captain with no minutes hands the armband — and whatever multiplier it
    carried, so a triple captain stays triple — to the vice, if the vice
    played. Players auto-subbed off count for nothing and players auto-subbed
    on count once. Idempotent: multipliers FPL has already settled pass
    through unchanged. Returns notes worth printing.
    """
    def played(pid: int) -> bool:
        return int(minutes.get(pid, 0) or 0) > 0

    notes: list[str] = []
    captain = next((p for p in picks if p.is_captain), None)
    vice = next((p for p in picks if p.is_vice), None)
    if (
        captain is not None and vice is not None
        and captain.multiplier > 1 and vice.multiplier <= 1
        and not played(captain.player_id) and played(vice.player_id)
    ):
        vice.multiplier, captain.multiplier = captain.multiplier, 1
        notes.append("captain did not play; the armband passed to the vice")
    for p in picks:
        if p.subbed_out:
            p.multiplier = 0
        elif p.subbed_in and p.multiplier == 0:
            p.multiplier = 1
    return notes


def squad_from_picks(client: FPLClient, team_id: int, gw: int, minutes: Mapping[int, int]) -> Squad:
    """The team as FPL recorded it: XI, bench order, armbands, auto-subs and score."""
    data = client.entry_picks(team_id, gw)
    swapped_in = {int(s["element_in"]) for s in data.get("automatic_subs", [])}
    swapped_out = {int(s["element_out"]) for s in data.get("automatic_subs", [])}
    picks = [
        Pick(
            player_id=int(p["element"]), multiplier=int(p["multiplier"]), slot=int(p["position"]),
            is_captain=bool(p.get("is_captain")), is_vice=bool(p.get("is_vice_captain")),
            subbed_in=int(p["element"]) in swapped_in, subbed_out=int(p["element"]) in swapped_out,
        )
        for p in data.get("picks", [])
    ]
    notes = settle(picks, minutes)
    history = data.get("entry_history", {}) or {}
    return Squad(
        picks=picks,
        source=f"FPL team {team_id}",
        points=history.get("points"),
        bench_points=history.get("points_on_bench"),
        transfer_cost=int(history.get("event_transfers_cost", 0) or 0),
        chip=data.get("active_chip"),
        notes=notes,
    )


def squad_from_config(
    config: Config, client: FPLClient, table: pd.DataFrame, minutes: Mapping[int, int]
) -> Squad:
    """The 15 in `config.local.yaml`, started as the model would have started them.

    The saved squad is whoever you hold *now*, so after a transfer this reviews
    a team you did not field; setting `squad.team_id` reviews the real one.
    Bench order is the model's, keeper first as FPL requires.
    """
    from .frontend import _normalise, resolve_squad

    names = config.squad_players
    if not names:
        raise ValueError("no squad: set `squad.players` in config.local.yaml, or `squad.team_id`")
    ids = resolve_squad(names, client)
    have = [i for i in ids if i in table.index]
    notes = ["your saved squad as it stands now, started as the model would — set squad.team_id "
             "in config.local.yaml to review the team you actually fielded"]
    missing = [n for n, i in zip(names, ids) if i not in table.index]
    if missing:
        notes.append(f"no prediction for {', '.join(missing)} (no fixture, or not in the pool at the deadline)")

    squad = table.loc[have]
    xi = best_xi(squad)
    ranked = squad.drop(index=xi).sort_values("expected_points", ascending=False)
    bench = ranked.index[ranked["position"] == "GK"].tolist() + ranked.index[ranked["position"] != "GK"].tolist()

    web = client.players().set_index("id")["web_name"].to_dict()

    def armband(name: str | None) -> int | None:
        if not name:
            return None
        want = _normalise(name)
        return next(
            (i for i in xi if want in _normalise(web.get(i, "")) or _normalise(web.get(i, "")) in want), None
        )

    captain = armband(config.squad_captain) or max(xi, key=lambda i: squad.loc[i, "expected_points"])
    vice = armband(config.squad_vice)
    if vice == captain:
        vice = None

    subs = auto_subs(xi, bench, squad["position"].to_dict(), minutes)
    picks = [
        Pick(pid, multiplier=2 if pid == captain else 1, slot=slot, is_captain=pid == captain,
             is_vice=pid == vice, subbed_out=pid in subs)
        for slot, pid in enumerate(xi, start=1)
    ] + [
        Pick(pid, multiplier=0, slot=slot, subbed_in=pid in subs.values())
        for slot, pid in enumerate(bench, start=12)
    ]
    notes += settle(picks, minutes)
    return Squad(picks=picks, source="config.local.yaml", notes=notes)


def squad_table(squad: Squad, table: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """One row per pick, in slot order, with prediction and outcome attached."""
    rows = []
    for p in sorted(squad.picks, key=lambda p: p.slot):
        if p.player_id in table.index:
            r = table.loc[p.player_id].to_dict()
        else:
            # Owned but not predicted: a blank, or a player who left the pool.
            r = {
                "name": f"#{p.player_id}", "position": "?", "expected_points": float("nan"),
                "actual": float(actual["actual"].get(p.player_id, 0.0)) if "actual" in actual else 0.0,
                "minutes": int(actual["minutes"].get(p.player_id, 0)) if "minutes" in actual else 0,
                "fixture": "no fixture",
            }
        r.update({
            "player_id": p.player_id, "slot": p.slot, "multiplier": p.multiplier,
            "is_captain": p.is_captain, "is_vice": p.is_vice,
            "subbed_in": p.subbed_in, "subbed_out": p.subbed_out,
        })
        rows.append(r)
    return pd.DataFrame(rows).set_index("player_id")


# -- the running log ---------------------------------------------------------------

LOG_ID_COLS = ["season", "gw", "player_id", "name", "position"]
LOG_VALUE_COLS = ["minutes_r3", "xp_fpl", "expected_points", "actual", "minutes"]


def log_path(config: Config) -> Path:
    return config.processed_dir / LOG_NAME


def append_log(config: Config, gw: int, season: str, table: pd.DataFrame) -> pd.DataFrame:
    """Add this gameweek's predicted and actual terms to the running log.

    Idempotent per gameweek: reviewing it again — after bonus is confirmed,
    say — replaces its rows rather than counting them twice.
    """
    rows = table.rename_axis("player_id").reset_index()
    rows["season"], rows["gw"] = season, int(gw)
    cols = LOG_ID_COLS + LOG_VALUE_COLS
    cols += [t for t in POINTS_TERMS] + [f"act_{t}" for t in ACTUAL_TERMS]
    rows = rows.reindex(columns=cols)

    path = log_path(config)
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[~((existing["season"] == season) & (existing["gw"] == int(gw)))]
        rows = pd.concat([existing, rows], ignore_index=True)
    rows = rows.sort_values(["season", "gw", "player_id"]).reset_index(drop=True)
    config.ensure_dirs()
    rows.to_csv(path, index=False)
    return rows


def calibration(log_rows: pd.DataFrame) -> pd.DataFrame:
    """Predicted against actual points per scoring term, summed over the log.

    Sums rather than means, because the question is "does the model hand out
    the right amount of bonus over a season", and a ratio near 1.0 is the
    answer. Over every predicted player, including those who did not play —
    the appearance term already carries their chance of playing.
    """
    rows = []
    for term in ACTUAL_TERMS:
        pred = float(log_rows[term].sum()) if term in log_rows else 0.0
        rows.append({"term": term, "predicted": pred, "actual": float(log_rows[f"act_{term}"].sum())})
    rows.append({
        "term": "total",
        "predicted": float(log_rows["expected_points"].sum()),
        "actual": float(log_rows["actual"].sum()),
    })
    out = pd.DataFrame(rows).set_index("term")
    out["ratio"] = out["actual"] / out["predicted"].where(out["predicted"].abs() > 1e-9)
    return out


def head_to_head(log_rows: pd.DataFrame) -> pd.DataFrame:
    """The model against FPL's own figure, gameweek by gameweek, on the same players.

    The backtest's three measures, scored on what happened this season:
    MAE and within-position rank correlation over players with recent
    minutes (the backtest's playing filter), and the real points of each
    one's top XI picked from the whole pool. A row per gameweek plus `all`,
    which weights MAE by players, averages the rank correlations, and sums
    the XI points.
    """
    out = []
    for gw, g in log_rows.groupby("gw"):
        playing = g[playing_filter(g) & g["xp_fpl"].notna()]
        row: dict[str, Any] = {"gw": int(gw), "n": len(playing)}
        for who, col in (("model", "expected_points"), ("fpl", "xp_fpl")):
            scored = playing.rename(columns={col: PRED, "actual": ACTUAL})
            row[f"{who}_mae"] = mae(scored[ACTUAL], scored[PRED]) if len(scored) else float("nan")
            row[f"{who}_rank"] = spearman_by_position(scored[[PRED, ACTUAL, "position"]]) if len(scored) else float("nan")
            pool = g.set_index("player_id")
            pool = pool[pool[col].notna()]
            row[f"{who}_xi"] = float(pool.loc[best_xi(pool, col), "actual"].sum()) if len(pool) >= 11 else float("nan")
        out.append(row)
    if not out:
        return pd.DataFrame()
    per_gw = pd.DataFrame(out).set_index("gw")
    n = per_gw["n"].clip(lower=1)
    total = {"n": int(per_gw["n"].sum())}
    for who in ("model", "fpl"):
        total[f"{who}_mae"] = float(np.average(per_gw[f"{who}_mae"].fillna(0), weights=n))
        total[f"{who}_rank"] = float(per_gw[f"{who}_rank"].mean(skipna=True))
        total[f"{who}_xi"] = float(per_gw[f"{who}_xi"].sum(skipna=True))
    per_gw.index = per_gw.index.astype(str)
    return pd.concat([per_gw, pd.DataFrame([total], index=["all"])])


# -- putting it together ----------------------------------------------------------

@dataclass
class Review:
    gw: int
    season: str
    model: str
    generated_at: str | None        # when the stored prediction was made; None if rebuilt
    provisional: bool               # FPL has not marked the gameweek final
    table: pd.DataFrame             # every predicted player, joined with actuals
    squad: Squad | None
    squad_rows: pd.DataFrame | None
    calibration: pd.DataFrame       # season to date, per scoring term
    head_to_head: pd.DataFrame      # season to date, model against FPL
    notes: list[str] = field(default_factory=list)

    @property
    def reconstructed(self) -> bool:
        return self.generated_at is None


def latest_started_gw(client: FPLClient) -> int:
    """The most recent gameweek with a match kicked off."""
    fx = client.fixtures_frame()
    started = fx.loc[fx["started"].fillna(False).astype(bool) & fx["event"].notna(), "event"]
    if started.empty:
        raise ValueError("no gameweek has kicked off yet")
    return int(started.max())


def run(
    config: Config | None = None,
    client: FPLClient | None = None,
    gw: int | None = None,
    model: str = "component",
    team_id: int | None = None,
) -> Review:
    config = config or load_config()
    client = client or FPLClient(config)
    season = config.season_current
    gw = int(gw or latest_started_gw(client))

    events = client.events()
    match = events.loc[events["id"] == gw, "finished"]
    if match.empty:
        raise ValueError(f"no gameweek {gw}")
    provisional = not bool(match.fillna(False).iloc[0])

    notes: list[str] = []
    stored = load_predictions(config, gw)
    if stored is not None:
        record, meta = stored
        model, generated_at = meta.get("model", model), meta.get("generated_at")
    else:
        log.info("no stored prediction for gameweek %d; rebuilding it from the snapshot", gw)
        (record, notes), generated_at = reconstruct_predictions(config, client, gw, model=model), None

    # Short names, as on the page: "Raya", not "David Raya Martín".
    web = client.players().set_index("id")["web_name"]
    record = record.assign(name=record.index.to_series().map(web).fillna(record["name"]))

    actual = parse_live(client.event_live(gw))
    if actual.empty or int(actual["minutes"].sum()) == 0:
        raise ValueError(f"gameweek {gw} has no minutes played yet")
    table = join_actuals(record, actual, fixture_labels(client, gw))

    # Every predicted player goes in the log; the squad is a view onto it.
    logged = append_log(config, gw, season, table)
    this_season = logged[logged["season"] == season]

    minutes = actual["minutes"].to_dict()
    squad = None
    team_id = team_id or config.squad_team_id
    if team_id:
        try:
            squad = squad_from_picks(client, team_id, gw, minutes)
        except FPLAPIError as exc:
            log.warning("could not fetch team %s's picks (%s); using the saved squad", team_id, exc)
    if squad is None and config.squad_players:
        squad = squad_from_config(config, client, table, minutes)

    return Review(
        gw=gw, season=season, model=model, generated_at=generated_at, provisional=provisional,
        table=table, squad=squad,
        squad_rows=squad_table(squad, table, actual) if squad is not None else None,
        calibration=calibration(this_season), head_to_head=head_to_head(this_season),
        notes=notes,
    )


# -- text ----------------------------------------------------------------------------

def what_happened(row: Mapping[str, Any]) -> str:
    """A compact line of what the player actually did: `90' G 2A CS 3b`."""
    minutes = int(row.get("minutes", 0) or 0)
    if minutes == 0:
        return "did not play"
    bits = [f"{minutes}'"]
    g, a = int(row.get("goals_scored", 0) or 0), int(row.get("assists_n", 0) or 0)
    if g:
        bits.append("G" if g == 1 else f"{g}G")
    if a:
        bits.append("A" if a == 1 else f"{a}A")
    if row.get("act_clean_sheet", 0):
        bits.append("CS")
    if row.get("position") == "GK" and int(row.get("saves_n", 0) or 0) >= 3:
        bits.append(f"{int(row['saves_n'])}sv")
    if row.get("act_defcon", 0):
        bits.append("DC")
    if int(row.get("bonus_n", 0) or 0):
        bits.append(f"{int(row['bonus_n'])}b")
    if int(row.get("yellow", 0) or 0):
        bits.append("YC")
    if int(row.get("red", 0) or 0):
        bits.append("RC")
    if row.get("act_other", 0):
        bits.append(f"{int(row['act_other']):+d} other")
    return " ".join(bits)


def _num(x: float, nd: int = 1) -> str:
    return "—" if pd.isna(x) else f"{x:.{nd}f}"


def _squad_lines(review: Review) -> list[str]:
    sq, rows = review.squad, review.squad_rows
    assert sq is not None and rows is not None
    out = ["", f"Your squad ({sq.source})"]
    if sq.chip:
        out.append(f"  chip: {sq.chip}")
    for n in sq.notes:
        out.append(f"  note: {n}")
    out.append("  C captain, V vice, + auto-subbed on, x auto-subbed off. The armband's figures are"
               " already multiplied; the bench is shown at face value.")
    out.append(f"  {'':2} {'':3} {'player':<16} {'fixture':<20} {'xP':>5} {'pts':>5} {'diff':>5}  what happened")

    def line(r: pd.Series) -> str:
        mark = "C" if r["is_captain"] else ("V" if r["is_vice"] else " ")
        arrow = "+" if r["subbed_in"] else ("x" if r["subbed_out"] else " ")
        mult = int(r["multiplier"])
        scale = mult if mult else 1  # the bench is shown at face value
        pts = float(r["actual"]) * scale
        xp = float(r["expected_points"]) * scale if pd.notna(r["expected_points"]) else float("nan")
        diff = pts - xp if (mult and pd.notna(xp)) else float("nan")
        shown = f"{pts:.0f}"
        return (
            f"  {mark}{arrow} {str(r['position']):<3} {str(r['name'])[:16]:<16} {str(r['fixture'])[:20]:<20} "
            f"{_num(xp):>5} {shown:>5} {'' if pd.isna(diff) else f'{diff:+.1f}':>5}  {what_happened(r)}"
        )

    for _, r in rows[rows["slot"] <= 11].iterrows():
        out.append(line(r))
    out.append("  " + "-" * 76)
    for _, r in rows[rows["slot"] > 11].iterrows():
        out.append(line(r))

    counted = rows[rows["multiplier"] > 0]
    xp_total = float((counted["expected_points"].fillna(0) * counted["multiplier"]).sum())
    pts_total = float((counted["actual"] * counted["multiplier"]).sum())
    summary = f"  scored {pts_total:.0f} against {xp_total:.1f} expected"
    if sq.points is not None:
        extras = [f"FPL: {sq.points} pts"]
        if sq.transfer_cost:
            extras.append(f"-{sq.transfer_cost} for transfers")
        if sq.bench_points is not None:
            extras.append(f"{sq.bench_points} left on the bench")
        summary += "   (" + ", ".join(extras) + ")"
    out += ["", summary]
    return out


def render(review: Review, top: int = 8) -> str:
    title = f"Gameweek {review.gw} review — {review.season}"
    out = [title, "=" * len(title)]
    if review.reconstructed:
        out.append(f"prediction: rebuilt from the gameweek {review.gw} snapshot (none was stored before the deadline)")
    else:
        made = pd.Timestamp(review.generated_at)
        out.append(f"prediction: as stored {made:%a %d %b %H:%M} UTC, model {review.model}")
    for note in review.notes:
        out.append(f"note: {note}")
    if review.provisional:
        out.append("note: FPL has not marked this gameweek final — bonus can still move; review again tomorrow")

    if review.squad is not None:
        out += _squad_lines(review)

    t = review.table
    played = t[t["minutes"] > 0]
    out += ["", "Biggest over-deliveries:"]
    for _, r in played.sort_values("diff", ascending=False).head(top).iterrows():
        out.append(f"  {str(r['name'])[:18]:<18} {str(r['position']):<3} {str(r['fixture'])[:20]:<20} "
                   f"xP {r['expected_points']:4.1f} -> {r['actual']:3.0f}   {what_happened(r)}")
    out += ["", "Biggest under-deliveries (of the model's top 60):"]
    rated = t.sort_values("expected_points", ascending=False).head(60)
    for _, r in rated.sort_values("diff").head(top).iterrows():
        out.append(f"  {str(r['name'])[:18]:<18} {str(r['position']):<3} {str(r['fixture'])[:20]:<20} "
                   f"xP {r['expected_points']:4.1f} -> {r['actual']:3.0f}   {what_happened(r)}")

    h = review.head_to_head
    if not h.empty:
        week, season = h.loc[str(review.gw)], h.loc["all"]
        span = f"{len(h) - 1} gameweek{'s' if len(h) > 2 else ''}"
        out += [
            "",
            "Model against FPL's own figure (players with recent minutes):",
            f"  {'':<24}{'this week':>18}{'season (' + span + ')':>24}",
            f"  {'':<24}{'model':>9}{'FPL':>9}{'model':>12}{'FPL':>12}",
            f"  {'MAE (lower is better)':<24}{week.model_mae:>9.3f}{week.fpl_mae:>9.3f}"
            f"{season.model_mae:>12.3f}{season.fpl_mae:>12.3f}",
            f"  {'rank within position':<24}{_num(week.model_rank, 3):>9}{_num(week.fpl_rank, 3):>9}"
            f"{_num(season.model_rank, 3):>12}{_num(season.fpl_rank, 3):>12}",
            f"  {'top XI, real points':<24}{_num(week.model_xi, 0):>9}{_num(week.fpl_xi, 0):>9}"
            f"{_num(season.model_xi, 0):>12}{_num(season.fpl_xi, 0):>12}",
        ]

    c = review.calibration
    out += [
        "",
        "Scoring terms, season to date (every predicted player):",
        f"  {'term':<16}{'predicted':>10}{'actual':>10}{'actual/pred':>13}",
    ]
    for term, r in c.iterrows():
        if term == "total":
            out.append("  " + "-" * 49)
        ratio = "" if pd.isna(r["ratio"]) else f"{r['ratio']:.2f}"
        out.append(f"  {term:<16}{r['predicted']:>10.0f}{r['actual']:>10.0f}{ratio:>13}")
    out += ["", f"  log: {LOG_NAME} — one gameweek is an anecdote; the log is the evidence"]
    return "\n".join(out)
