"""Export the squad-picker frontend as a single self-contained HTML file.

The page is static: player pool, prices, expected points and the upcoming
fixtures are baked in at build time, and all the squad and transfer logic runs
in the browser. That keeps it shareable — one file, no server — at the cost of
needing a rebuild when prices move. Rebuild before each deadline:

    python -m src.cli export-frontend

Squad state (the 15, the bench order, chips, free transfers) lives in the
viewer's own browser storage, so the page can enforce the transfer rules across
gameweeks without any backend.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config, load_config
from .data.fpl_api import ELEMENT_TYPE_TO_POSITION, FPLAPIError, FPLClient
from .models.baselines import Predictor, default_baselines

log = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "frontend" / "template.html"
OUTPUT_NAME = "squad-picker.html"
DATA_PLACEHOLDER = "/*__FPL_DATA__*/null"

# Name-matching confidence. Above `_GOOD_MATCH` the squad's shape is allowed to
# break ties; below `_MIN_MATCH` nothing is accepted at all.
_GOOD_MATCH = 0.7
_MIN_MATCH = 0.62

# FPL's availability codes.
STATUS_LABEL = {
    "a": "available",
    "d": "doubtful",
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
    "n": "ineligible",
}


# Letters whose diacritic is part of the glyph rather than a combining mark, so
# NFKD leaves them alone. Ødegaard and Fabiański are exactly the names people
# type unaccented, so these have to be mapped by hand.
_TRANSLITERATE = str.maketrans(
    {
        "ø": "o", "Ø": "o",
        "æ": "ae", "Æ": "ae",
        "œ": "oe", "Œ": "oe",
        "ł": "l", "Ł": "l",
        "đ": "d", "Đ": "d",
        "ð": "d", "Ð": "d",
        "þ": "th", "Þ": "th",
        "ß": "ss",
        "ı": "i",
        "'": "", "'": "", "`": "",
    }
)


def _normalise(text: str) -> str:
    """Strip accents, ligatures and case so `Ødegaard` matches `odegaard`."""
    folded = str(text).translate(_TRANSLITERATE)
    decomposed = unicodedata.normalize("NFKD", folded)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()


def resolve_squad(names: list[str], client: FPLClient) -> list[int]:
    """Turn a list of typed player names into FPL element ids.

    Matching is deliberately shape-aware. Surnames collide — there are two
    Jameses and two Ouattaras in the pool, in different positions — so after
    scoring each name independently this walks the ambiguous ones and prefers
    the candidate that keeps the squad legal (2 GK, 5 DEF, 5 MID, 3 FWD).
    A bare name lookup gets those wrong roughly as often as it gets them right.
    """
    players = client.players()
    players = players.assign(
        full=(players["first_name"].fillna("") + " " + players["second_name"].fillna("")).str.strip()
    )

    def candidates(name: str) -> list[tuple[float, int, str]]:
        want = _normalise(name)
        scored: list[tuple[float, int, str]] = []
        for row in players.itertuples():
            keys = [_normalise(row.web_name), _normalise(row.full), _normalise(row.second_name)]
            score = max(SequenceMatcher(None, want, k).ratio() for k in keys)
            if any(want == k for k in keys):
                score = 1.0
            elif any(want in k for k in keys):
                score = max(score, 0.93)
            scored.append((score, int(row.id), ELEMENT_TYPE_TO_POSITION[int(row.element_type)]))
        scored.sort(key=lambda x: -x[0])
        return scored[:8]

    all_candidates = [(name, candidates(name)) for name in names]

    # Settle unambiguous names first, then let the remaining quota decide the rest.
    chosen: dict[str, tuple[int, str]] = {}
    pending: list[tuple[str, list[tuple[float, int, str]]]] = []
    for name, cands in all_candidates:
        top = [c for c in cands if c[0] >= 0.999]
        if len(top) == 1:
            chosen[name] = (top[0][1], top[0][2])
        else:
            pending.append((name, cands))

    quota = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for _, pos in chosen.values():
        quota[pos] -= 1

    for name, cands in pending:
        pick = next(
            (c for c in cands if c[0] >= _GOOD_MATCH and quota.get(c[2], 0) > 0),
            next((c for c in cands if c[0] >= _MIN_MATCH), None),
        )
        if pick is None:
            # Better to stop than to quietly sign someone else. A player who has
            # left the league scores ~0.5 against whoever happens to look most
            # like them, and that is not a squad anyone meant to pick.
            suggestions = ", ".join(
                f"{_web_name(players, pid)} ({score:.2f})" for score, pid, _ in cands[:3]
            )
            raise ValueError(
                f"no confident match for {name!r} — closest were: {suggestions}. "
                "Check the spelling, or whether they are still in the Premier League."
            )
        chosen[name] = (pick[1], pick[2])
        quota[pick[2]] -= 1

    return [chosen[name][0] for name in names]


def _web_name(players: pd.DataFrame, player_id: int) -> str:
    row = players.loc[players["id"] == player_id, "web_name"]
    return str(row.iloc[0]) if len(row) else str(player_id)


def price_prior(config: Config, seasons: list[str] | None = None) -> dict[str, list[list[float]]]:
    """What a player at a given price is worth per gameweek, by position.

    Two gameweeks of a new season is far too little to judge anyone on, so
    suggestions shrink a player's own record toward this prior. It answers
    "what does a £5.5m defender normally do?" using a full finished season.

    The rate is per *gameweek*, not per appearance, and is computed over every
    player rather than only those who played — so the risk of not playing at all
    is priced in, which is most of what separates a £4.0m bench filler from a
    £5.5m starter.

    Returned as price/value knots per position for interpolation. A linear fit
    was the obvious thing to do and is badly wrong: prices within a position
    span a narrow range, so extrapolating puts an £8.0m keeper at eight points a
    gameweek. Binned medians cannot run away like that.
    """
    from .data.historical import load_seasons

    seasons = seasons or config.season_history[-1:]
    df = load_seasons(seasons, config=config)
    if df.empty:
        return {}

    n_gws = max(1, int(df.groupby("season")["gw"].nunique().mean()))
    agg = (
        df.groupby(["season", "player_id", "position"], observed=True)
        .agg(pts=("total_points", "sum"), price=("value", "mean"))
        .reset_index()
    )
    agg["rate"] = agg["pts"] / n_gws

    # Fixed £0.5m price bands, not quantiles. Quantile bins put every knot in the
    # crowded cheap end and leave nothing above about £7m, so a £15m striker
    # would inherit a mid-price knot. Bands cover the actual range.
    band = 5  # tenths of a million
    knots: dict[str, list[list[float]]] = {}

    for position, sub in agg.groupby("position", observed=True):
        if len(sub) < 12:
            continue
        sub = sub.assign(band=(sub["price"] // band).astype(int))
        grouped = (
            sub.groupby("band", observed=True)
            .agg(price=("price", "median"), rate=("rate", "mean"), n=("rate", "size"))
            .reset_index()
            .sort_values("price")
        )

        # Merge thin bands forward so no knot rests on one or two players.
        rows: list[list[float]] = []
        carry_n, carry_sum, carry_price = 0, 0.0, 0.0
        for _, row in grouped.iterrows():
            carry_n += int(row["n"])
            carry_sum += float(row["rate"]) * int(row["n"])
            carry_price = float(row["price"])
            if carry_n >= 4:
                rows.append([carry_price, carry_sum / carry_n])
                carry_n, carry_sum = 0, 0.0
        if carry_n and rows:
            rows[-1] = [carry_price, (rows[-1][1] * 1 + carry_sum / carry_n) / 2]

        # Sample noise can invert two adjacent bands; a dearer player should
        # never be priced below a cheaper one.
        running = 0.0
        cleaned: list[list[float]] = []
        for price, rate in rows:
            running = max(running, rate)
            cleaned.append([round(price, 1), round(running, 3)])
        if cleaned:
            knots[str(position)] = cleaned

    return knots


# A club counts as an elite defence at this multiple of the league's mean clean
# sheet rate. Last season that put Arsenal (0.50) and Man City (0.42) above the
# line and everyone else below it, with the next club on 0.32 — a real gap
# rather than a cut chosen to produce a tidy number.
ELITE_DEFENCE_RATIO = 1.35


def team_defence(config: Config, client: FPLClient) -> dict[str, Any]:
    """Last season's clean-sheet record per club, keyed by *current* team id.

    Clubs are joined across seasons on short name, not id. Team ids are
    reassigned like player ids, but the twenty short names are stable and
    unambiguous, so this is the one place where a name join is the right call.
    Promoted clubs have no Premier League record and are simply absent.
    """
    from .data.historical import VaastavLoader

    season = config.season_history[-1]
    loader = VaastavLoader(config)
    try:
        df = loader.season_frame(season)
        past_teams = loader.teams(season)
    except Exception:  # noqa: BLE001 - a missing season must not break the export
        log.warning("no historical season available for team defence")
        return {}

    id_to_short = past_teams.set_index("id")["short_name"].to_dict()
    per_fixture = df.drop_duplicates(subset=["team_id", "fixture_id"])

    agg = per_fixture.groupby("team_id").agg(
        games=("fixture_id", "nunique"),
        clean_sheets=("team_goals_against", lambda s: int((s == 0).sum())),
        conceded=("team_goals_against", "sum"),
    )
    agg = agg[agg["games"] > 0]
    agg["cs_rate"] = agg["clean_sheets"] / agg["games"]

    # Round first, then decide. The page only ever sees the rounded numbers, so
    # deciding on full precision could publish a club shown at exactly the
    # threshold yet flagged as not elite — a page that contradicts itself.
    league_mean = round(float(agg["cs_rate"].mean()), 3)
    threshold = round(league_mean * ELITE_DEFENCE_RATIO, 3)

    current = client.teams().set_index("short_name")["id"].to_dict()

    out: dict[str, Any] = {
        "leagueMeanCsRate": league_mean,
        "eliteThreshold": threshold,
        "teams": {},
    }
    for old_id, row in agg.iterrows():
        short = id_to_short.get(int(old_id))
        team_id = current.get(short)
        if team_id is None:
            continue   # relegated since
        cs_rate = round(float(row["cs_rate"]), 3)
        out["teams"][str(int(team_id))] = {
            "short": short,
            "cs": int(row["clean_sheets"]),
            "games": int(row["games"]),
            "csRate": cs_rate,
            "concededPerGame": round(float(row["conceded"]) / int(row["games"]), 2),
            "elite": bool(cs_rate >= threshold),
        }
    return out


def last_season_records(client: FPLClient) -> dict[int, dict[str, int]]:
    """Each player's most recent completed Premier League season.

    `element-summary/{id}/` carries a `history_past` block keyed by the player's
    *current* element id, which quietly solves the cross-season crosswalk the
    brief flags in §6.2: FPL renumbers players every season, so joining last
    season's data by id is wrong and joining it by name is the trap. The API
    has already done the mapping.

    This is a far better prior than price. Price says "the market rates this
    player at £6.0m"; this says "he scored 179 points in 3150 minutes".
    """
    records: dict[int, dict[str, int]] = {}
    for player_id in client.players()["id"].tolist():
        try:
            past = client.element_summary(int(player_id)).get("history_past", [])
        except FPLAPIError:
            continue
        if not past:
            continue   # new signing or promoted club — no PL history to lean on
        last = past[-1]
        records[int(player_id)] = {
            "pts": int(last.get("total_points", 0)),
            "mins": int(last.get("minutes", 0)),
        }
    return records


def _fixture_lookup(
    client: FPLClient, first_gw: int, horizon: int = 1
) -> dict[int, list[dict[str, Any]]]:
    """team id -> its fixtures across `horizon` gameweeks from `first_gw`.

    A list per team, and the gameweek is carried on every entry, because the
    two things that most change a transfer's value are exactly the ones a
    single-gameweek view hides: a blank (no entry for that gameweek) and a
    double (two entries for it).
    """
    fixtures = client.fixtures_frame()
    teams = client.teams().set_index("id")["short_name"].to_dict()
    window = range(first_gw, first_gw + horizon)

    out: dict[int, list[dict[str, Any]]] = {}
    selected = fixtures[fixtures["event"].isin(list(window))].sort_values(["event", "kickoff_time"])
    for _, f in selected.iterrows():
        home, away, gw = int(f["team_h"]), int(f["team_a"]), int(f["event"])
        out.setdefault(home, []).append(
            {
                "gw": gw,
                "opp": teams.get(away, "?"),
                "home": True,
                "difficulty": int(f["team_h_difficulty"]),
            }
        )
        out.setdefault(away, []).append(
            {
                "gw": gw,
                "opp": teams.get(home, "?"),
                "home": False,
                "difficulty": int(f["team_a_difficulty"]),
            }
        )
    return out


def build_player_data(
    config: Config | None = None,
    client: FPLClient | None = None,
    gw: int | None = None,
    model: str = "minutes_x_pp90",
    squad: list[str] | None = None,
    bank: int | None = None,
    horizon: int = 5,
) -> dict[str, Any]:
    """Player pool, prices, model expected points and next-gameweek fixtures."""
    from . import pipeline

    config = config or load_config()
    client = client or FPLClient(config)
    gw = gw or client.next_gw()
    if gw is None:
        raise ValueError("no upcoming gameweek found")

    frame, feature_cols = pipeline.build(config, upcoming_gw=gw, client=client)
    target = frame[(frame["season"] == config.season_current) & (frame["gw"] == gw)].copy()

    predictor: Predictor = default_baselines()[model]
    predictor.fit(frame[frame["gw"] < gw], feature_cols)
    target["ep"] = predictor.predict(target)

    # Sum across fixtures: a double gameweek pays for both.
    ep = target.groupby("player_id", observed=True)["ep"].sum()

    players = client.players()
    teams = client.teams().set_index("id")["short_name"].to_dict()
    # A few gameweeks beyond the horizon, so the page still has fixtures to
    # work with if a deadline passes before the next rebuild.
    fixtures = _fixture_lookup(client, gw, horizon=horizon + 4)
    last_season = last_season_records(client)

    rows = []
    for _, p in players.iterrows():
        pid = int(p["id"])
        team_id = int(p["team"])
        rows.append(
            {
                "id": pid,
                "name": str(p["web_name"]),
                "pos": ELEMENT_TYPE_TO_POSITION[int(p["element_type"])],
                "team": teams.get(team_id, "?"),
                "teamId": team_id,
                # Price in tenths of a million, as the API stores it. All the
                # money arithmetic stays in integers so 0.1m rounding is exact.
                "price": int(p["now_cost"]),
                "ep": round(float(ep.get(pid, 0.0)), 2),
                "epFpl": round(float(p["ep_next"] or 0), 2),
                "pts": int(p["total_points"]),
                "ppg": float(p["points_per_game"] or 0),
                "minutes": int(p["minutes"]),
                "starts": int(p["starts"]),
                "form": float(p["form"] or 0),
                "owned": float(p["selected_by_percent"] or 0),
                "status": STATUS_LABEL.get(str(p["status"]), str(p["status"])),
                "chance": (
                    None
                    if pd.isna(p["chance_of_playing_next_round"])
                    else int(p["chance_of_playing_next_round"])
                ),
                "last": last_season.get(pid),
                "fixtures": fixtures.get(team_id, []),
            }
        )

    preset = resolve_squad(squad, client) if squad else None

    # Resolve armbands within the squad only: "haaland" should never match some
    # other Haaland-ish name from the wider pool.
    captain_id = vice_id = None
    if preset and squad:
        picked = {_normalise(client.players().set_index("id").loc[pid, "web_name"]): pid
                  for pid in preset}
        for name, target in ((config.squad_captain, "c"), (config.squad_vice, "v")):
            if not name:
                continue
            match = next(
                (pid for key, pid in picked.items() if _normalise(name) in key or key in _normalise(name)),
                None,
            )
            if match is None:
                log.warning("armband name %r is not in the squad; ignoring", name)
            elif target == "c":
                captain_id = match
            else:
                vice_id = match

    events = client.events()
    deadline = events.loc[events["id"] == gw, "deadline_time"]

    # Every remaining deadline, so the page can work out the current gameweek
    # from its own clock instead of waiting to be rebuilt. The whole calendar is
    # published in advance, so this needs no network at runtime.
    upcoming_events = [
        {"gw": int(row.id), "deadline": row.deadline_time.isoformat()}
        for row in events.itertuples()
        if int(row.id) >= gw and pd.notna(row.deadline_time)
    ]

    return {
        "season": config.season_current,
        "gameweek": gw,
        "deadline": deadline.iloc[0].isoformat() if len(deadline) else None,
        "model": model,
        "generatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "events": upcoming_events,
        "presetSquad": preset,
        "captain": captain_id,
        "vice": vice_id,
        "freeTransfers": config.squad_free_transfers,
        "bank": bank,
        "pricePrior": price_prior(config),
        "teamDefence": team_defence(config, client),
        "seasonGws": 38,
        "gwsPlayed": max(0, gw - 1),
        "horizon": horizon,
        "horizonGws": list(range(gw, gw + horizon)),
        "teams": [{"id": int(k), "short": v} for k, v in sorted(teams.items())],
        "players": rows,
    }


def export(
    config: Config | None = None,
    client: FPLClient | None = None,
    gw: int | None = None,
    model: str = "minutes_x_pp90",
    squad: list[str] | None = None,
    bank: int | None = None,
    horizon: int = 5,
    output: Path | None = None,
) -> Path:
    """Write the self-contained HTML page and return its path."""
    config = config or load_config()
    data = build_player_data(
        config=config, client=client, gw=gw, model=model, squad=squad,
        bank=bank, horizon=horizon,
    )

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if DATA_PLACEHOLDER not in template:
        raise ValueError(f"{TEMPLATE_PATH} is missing the {DATA_PLACEHOLDER} placeholder")

    payload = json.dumps(data, separators=(",", ":"))
    # `</script>` inside a JSON string would close the tag early.
    payload = payload.replace("</", "<\\/")
    html = template.replace(DATA_PLACEHOLDER, payload)

    output = output or (TEMPLATE_PATH.parent / OUTPUT_NAME)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    log.info("wrote %s (%d players, gameweek %d)", output, len(data["players"]), data["gameweek"])
    return output
