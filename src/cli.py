"""Command line entry points.

    python -m src.cli build-features    # pull data, build + store the feature frame
    python -m src.cli backtest          # walk-forward baselines
    python -m src.cli predict           # expected points for the next gameweek
    python -m src.cli export-frontend   # build the squad / transfer page
    python -m src.cli serve             # build it, then serve it so it can save your squad
    python -m src.cli snapshot          # capture live inputs before a deadline

`export-frontend` writes one self-contained HTML file with the player pool
baked into it, and the page does the rest in the browser — so "running the
frontend" can mean simply opening that file. `serve` puts a small local helper
behind the same page so that "Save team" can write `config.local.yaml`, which a
page opened as a file has no way to do. Re-run either whenever prices,
fixtures or your squad change.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import pipeline
from .config import load_config
from .data.fpl_api import FPLClient
from .evaluate import run_backtest, training_window
from .models.baselines import all_predictors, default_baselines


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def cmd_build_features(args: argparse.Namespace) -> int:
    config = load_config(args.config, use_local=not args.no_local)
    df, feature_cols = pipeline.build(config, include_current=not args.history_only)
    parquet_path, features_path = pipeline.save(df, feature_cols, config)

    print(f"rows      : {len(df):,}")
    print(f"features  : {len(feature_cols)}")
    print(f"seasons   : {', '.join(sorted(df['season'].unique()))}")
    print(f"written   : {parquet_path}")
    print(f"            {features_path}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    config = load_config(args.config, use_local=not args.no_local)
    if args.rebuild:
        df, feature_cols = pipeline.build(config)
        pipeline.save(df, feature_cols, config)
    else:
        df, feature_cols = pipeline.load_processed(config)

    season = args.season or config.season_history[-1]
    predictors = default_baselines() if args.baselines_only else all_predictors()
    results = run_backtest(
        df,
        feature_cols,
        predictors=predictors,
        season=season,
        min_train_gws=int(config.evaluate["min_train_gws"]),
    )

    pd.set_option("display.width", 200)
    print(f"\nWalk-forward backtest — season {season}")
    print("Train on gameweeks < t, predict t. Scored per player-gameweek.\n")
    print("== all players ==")
    print(results["summary_all"].round(4).to_string())
    print("\n== players with recent minutes ==")
    print(results["summary_playing"].round(4).to_string())

    config.ensure_dirs()
    out = config.processed_dir / f"backtest_{season}.csv"
    results["per_gw_all"].to_csv(out, index=False)
    print(f"\nper-gameweek detail: {out}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    config = load_config(args.config, use_local=not args.no_local)
    client = FPLClient(config)

    gw = args.gw or client.next_gw()
    if gw is None:
        print("no upcoming gameweek found", file=sys.stderr)
        return 1

    df, feature_cols = pipeline.build(config, upcoming_gw=gw, client=client)

    target = df[(df["season"] == config.season_current) & (df["gw"] == gw)].copy()
    if target.empty:
        print(f"no rows for gameweek {gw}", file=sys.stderr)
        return 1

    predictor = all_predictors()[args.model]
    predictor.fit(training_window(df, config.season_current, gw), feature_cols)
    target["expected_points"] = predictor.predict(target)

    # A double gameweek means two fixtures; FPL pays you for both.
    per_player = (
        target.groupby(["player_id", "name", "position"], observed=True)
        .agg(
            expected_points=("expected_points", "sum"),
            fixtures=("fixture_id", "nunique"),
            price=("value", "first"),
        )
        .reset_index()
        .sort_values("expected_points", ascending=False)
    )
    per_player["price"] = per_player["price"] / 10.0

    config.ensure_dirs()
    out = config.processed_dir / f"expected_points_gw{gw}.csv"
    per_player.to_csv(out, index=False)

    print(f"\nExpected points — gameweek {gw} ({config.season_current}), model: {args.model}")
    print(f"{len(per_player)} players | written to {out}\n")
    print(per_player.head(args.top).round(2).to_string(index=False))
    return 0


def cmd_export_frontend(args: argparse.Namespace) -> int:
    from . import frontend

    config = load_config(args.config, use_local=not args.no_local)
    client = FPLClient(config)
    gw = args.gw or client.next_gw()

    # Flags win; otherwise fall back to whatever `config.yaml` remembers, so the
    # common case is a bare `export-frontend` with no arguments at all.
    if args.squad:
        squad = [n.strip() for n in args.squad.split(",") if n.strip()]
    elif args.squad_file:
        text = Path(args.squad_file).read_text(encoding="utf-8")
        squad = [n.strip() for n in text.replace(",", "\n").splitlines() if n.strip()]
    else:
        squad = config.squad_players or None

    bank = round(args.bank * 10) if args.bank is not None else config.squad_bank
    horizon = args.horizon if args.horizon is not None else (config.squad_horizon or 5)

    output = Path(args.output) if args.output else None
    path = frontend.export(
        config=config, client=client, gw=gw, model=args.model, squad=squad,
        bank=bank, horizon=horizon, output=output,
    )

    size_kb = path.stat().st_size / 1024
    print(f"gameweek  : {gw}")
    print(f"model     : {args.model}")
    print(f"written   : {path} ({size_kb:.0f} KB)")
    if squad:
        from .data.fpl_api import ELEMENT_TYPE_TO_POSITION

        players = client.players().set_index("id")
        ids = frontend.resolve_squad(squad, client)
        print(f"\nsquad loaded ({len(ids)} players):")
        for pid in ids:
            row = players.loc[pid]
            pos = ELEMENT_TYPE_TO_POSITION[int(row["element_type"])]
            print(f"  {pos:<3} {row['web_name']:<20} £{row['now_cost'] / 10:>4.1f}m")
        total = sum(int(players.loc[i, "now_cost"]) for i in ids)
        print(f"  {'':<3} {'total':<20} £{total / 10:>4.1f}m")
        if bank is None:
            # No --bank given, so this is only what today's prices imply. Your
            # real balance depends on what you paid, which this never sees.
            print(
                f"  {'':<3} {'bank (implied)':<20} £{(1000 - total) / 10:>4.1f}m"
                "   — pass --bank to set your actual balance"
            )
        else:
            print(f"  {'':<3} {'bank':<20} £{bank / 10:>4.1f}m")
    if args.open:
        import webbrowser

        webbrowser.open(path.resolve().as_uri())
        print("\nOpened in your browser.")
    else:
        print(f"\nOpen it with:  start {path}")
    print("Rebuild before each deadline — prices and fixtures are baked in.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Build the page, then serve it locally so it can save your squad.

    Opened as a plain file the page cannot write to disk, which is why
    `config.local.yaml` had to be filled in by hand. Served from here it gets a
    "Save team" button that writes the file — and on a fresh clone, where the
    file does not exist yet, it asks for a team and a bank on the first visit.
    """
    from . import frontend, serve

    config = load_config(args.config, use_local=not args.no_local)

    if args.no_build:
        page = frontend.TEMPLATE_PATH.parent / frontend.OUTPUT_NAME
        if not page.exists():
            print(f"{page} does not exist yet — run without --no-build first.")
            return 1
    else:
        client = FPLClient(config)
        gw = args.gw or client.next_gw()
        page = frontend.export(
            config=config, client=client, gw=gw, model=args.model,
            squad=config.squad_players or None, bank=config.squad_bank,
            horizon=config.squad_horizon or 5,
        )
        print(f"gameweek  : {gw}")
        print(f"written   : {page}")

    serve.serve(
        config, directory=page.parent, page=page.name,
        port=args.port, open_browser=not args.no_open,
    )
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Store bootstrap-static as it stands right now, keyed by gameweek.

    The model's most valuable inputs — FPL's expected points, the availability
    flag, set-piece orders — are only ever *current* in the API. Their history
    in the vaastav dataset is patchy and only probably pre-deadline. Run this
    before each deadline and next season trains on data that is pre-deadline by
    construction, with none of the "was this scraped after the match" doubt.

    Idempotent per gameweek: rerunning before the same deadline overwrites.
    """
    import json

    config = load_config(args.config, use_local=not args.no_local)
    client = FPLClient(config)
    payload = client.bootstrap_static(force=True)
    gw = args.gw or client.next_gw()
    if gw is None:
        print("no upcoming gameweek to snapshot", file=sys.stderr)
        return 1

    out_dir = config.data_dir / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC")
    path = out_dir / f"bootstrap_gw{gw:02d}.json"

    # Keep only the per-player fields the model can use, plus enough to join.
    keep = [
        "id", "web_name", "team", "element_type", "now_cost", "ep_next", "ep_this",
        "chance_of_playing_next_round", "chance_of_playing_this_round", "status", "news",
        "penalties_order", "corners_and_indirect_freekicks_order", "form",
        "selected_by_percent", "transfers_in_event", "transfers_out_event",
    ]
    players = [{k: p.get(k) for k in keep} for p in payload["elements"]]
    path.write_text(
        json.dumps({"gameweek": gw, "captured_at": stamp.isoformat(), "players": players}),
        encoding="utf-8",
    )
    flagged = sum(1 for p in players if p["chance_of_playing_next_round"] is not None)
    print(f"gameweek  : {gw}")
    print(f"captured  : {stamp.strftime('%a %d %b %H:%M UTC')}")
    print(f"players   : {len(players)}  ({flagged} carrying an availability flag)")
    print(f"written   : {path}")
    print("\nRun this before every deadline. A scheduled task on Friday afternoons works.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fpl", description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")

    # Shared by every subcommand, so it can be typed after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--no-local",
        action="store_true",
        help="ignore config.local.yaml — how the copy of the page that gets "
             "committed is built, so it ships without a personal squad in it",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-features", parents=[common], help="pull data and build the feature frame")
    p_build.add_argument(
        "--history-only",
        action="store_true",
        help="skip the live API and use historical seasons only",
    )
    p_build.set_defaults(func=cmd_build_features)

    p_bt = sub.add_parser("backtest", parents=[common], help="walk-forward baseline backtest")
    p_bt.add_argument("--season", default=None, help="season to test (default: latest historical)")
    p_bt.add_argument("--rebuild", action="store_true", help="rebuild features first")
    p_bt.add_argument(
        "--baselines-only", action="store_true",
        help="skip the trained component model (much faster)",
    )
    p_bt.set_defaults(func=cmd_backtest)

    p_pred = sub.add_parser("predict", parents=[common], help="expected points for the upcoming gameweek")
    p_pred.add_argument("--gw", type=int, default=None, help="gameweek (default: next)")
    p_pred.add_argument(
        "--model", default="component",
        help="predictor name: component (trained) or a baseline",
    )
    p_pred.add_argument("--top", type=int, default=25, help="rows to print")
    p_pred.set_defaults(func=cmd_predict)

    p_front = sub.add_parser("export-frontend", parents=[common], help="build the squad-picker HTML page")
    p_front.add_argument("--gw", type=int, default=None, help="gameweek (default: next)")
    p_front.add_argument(
        "--model", default="component",
        help="predictor supplying the model xPts column",
    )
    p_front.add_argument(
        "--squad", default=None,
        help="comma-separated player names (default: squad.players in config.yaml)"
    )
    p_front.add_argument(
        "--squad-file", default=None, help="file of player names, one per line"
    )
    p_front.add_argument(
        "--bank", type=float, default=None,
        help="money in the bank, in millions, e.g. 1.5 (default: squad.bank in "
             "config.yaml). Cannot be derived: it depends on what you paid for "
             "your players, not what they cost now.",
    )
    p_front.add_argument(
        "--horizon", type=int, default=None,
        help="gameweeks of fixtures to load (default: squad.horizon in config.yaml, else 5)",
    )
    p_front.add_argument(
        "--open", action="store_true", help="open the page in your browser once built"
    )
    p_front.add_argument("--output", default=None, help="output path")
    p_front.set_defaults(func=cmd_export_frontend)

    p_serve = sub.add_parser(
        "serve", parents=[common],
        help="build the page and serve it locally, so it can save your squad to config.local.yaml",
    )
    p_serve.add_argument("--gw", type=int, default=None, help="gameweek (default: next)")
    p_serve.add_argument("--model", default="component", help="predictor supplying the model xPts column")
    p_serve.add_argument("--port", type=int, default=8765, help="local port (default: 8765)")
    p_serve.add_argument("--no-open", action="store_true", help="do not open the browser")
    p_serve.add_argument(
        "--no-build", action="store_true",
        help="serve the page as last built instead of rebuilding it first",
    )
    p_serve.set_defaults(func=cmd_serve)

    p_snap = sub.add_parser(
        "snapshot", parents=[common],
        help="store the live player data for this gameweek, so next season trains on clean inputs",
    )
    p_snap.add_argument("--gw", type=int, default=None, help="gameweek to label it (default: next)")
    p_snap.set_defaults(func=cmd_snapshot)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
