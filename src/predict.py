"""Predicting one gameweek — the single path every entry point uses.

`fpl predict`, the page (`export-frontend` and `serve`) and the post-gameweek
review all need the same thing: the model fitted on what was known before a
deadline, its figure for every player in that gameweek, and that figure
decomposed into the scoring terms it was built from. Doing it in one place
means the page, the CSV and the review cannot disagree about what the model
said.

Every prediction made before its deadline is stored as
`data/processed/predictions_gw{N}.csv`. A rerun before the same deadline
overwrites — the later run has the later team news — and a run after it never
does, so the file is always what the model said in time to act on. That file
is what `fpl review` scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .data.fpl_api import FPLClient
from .evaluate import training_window
from .models.baselines import Predictor, all_predictors
from .models.combine import POINTS_TERMS

log = logging.getLogger(__name__)

PREDICTIONS_NAME = "predictions_gw{gw}.csv"

# Quantities that add across a double gameweek, alongside the points terms.
SUMMED = ("expected_points", *POINTS_TERMS, "expected_minutes")
# Probabilities, which do not: two 90% chances of playing are not 180%.
MAXED = ("p_play", "p_60")


@dataclass
class GameweekPrediction:
    gw: int
    model: str
    record: pd.DataFrame            # one row per player, indexed by player_id
    deadline: pd.Timestamp | None
    stored: Path | None             # where it was saved, or None if after the deadline


def prediction_record(target: pd.DataFrame, predictor: Predictor) -> pd.DataFrame:
    """One row per player: the model's figure for the gameweek, decomposed.

    Points terms and expected minutes sum across a double gameweek — FPL pays
    for both fixtures — while the playing-time probabilities take the larger
    fixture's value. FPL's own figure (`xp_fpl`) is per gameweek already, as is
    `minutes_r3`, the recent-minutes signal the backtest's playing filter uses.
    """
    keep = ["player_id", "name", "position", "team_id", "value", "fixture_id", "xP", "minutes_r3"]
    rows = target[[c for c in keep if c in target.columns]].copy()
    if hasattr(predictor, "predict_breakdown"):
        breakdown = predictor.predict_breakdown(target)
        rows = pd.concat([rows, breakdown.drop(columns=[c for c in breakdown if c in rows])], axis=1)
    else:
        rows["expected_points"] = np.asarray(predictor.predict(target), dtype=float)

    agg: dict[str, str] = {c: "first" for c in ("name", "position", "team_id", "value", "xP", "minutes_r3") if c in rows}
    agg["fixture_id"] = "nunique"
    agg.update({c: "sum" for c in SUMMED if c in rows})
    agg.update({c: "max" for c in MAXED if c in rows})
    out = (
        rows.groupby("player_id", observed=True)
        .agg(agg)
        .rename(columns={"fixture_id": "fixtures", "xP": "xp_fpl"})
    )
    out.index = out.index.astype(int)
    out.index.name = "player_id"
    return out


def fit_predict(
    frame: pd.DataFrame, feature_cols: list[str], season: str, gw: int, model: str = "component"
) -> pd.DataFrame:
    """Fit on everything known before `gw` and return its prediction record.

    `frame` must already hold the gameweek's prediction rows with their
    features built; `training_window` is the one definition of "known before".
    """
    target = frame[(frame["season"] == season) & (frame["gw"] == gw)]
    if target.empty:
        raise ValueError(f"no rows for gameweek {gw} of {season}")
    predictor: Predictor = all_predictors()[model]
    predictor.fit(training_window(frame, season, gw), feature_cols)
    return prediction_record(target.copy(), predictor)


def predictions_path(config: Config, gw: int) -> Path:
    return config.processed_dir / PREDICTIONS_NAME.format(gw=int(gw))


def save_predictions(
    record: pd.DataFrame,
    gw: int,
    config: Config,
    model: str,
    deadline: pd.Timestamp | None,
    now: pd.Timestamp | None = None,
) -> Path | None:
    """Store the record, unless its deadline has passed. Returns the path written, or None.

    After the deadline the API's inputs — `xP` above all — describe the *next*
    gameweek, so a rerun would overwrite what the model said in time with
    something it could not have said. Refusing is the only way the file stays
    worth reviewing.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    if deadline is not None and now >= deadline:
        log.info("gameweek %d deadline has passed; keeping the prediction stored before it", gw)
        return None
    config.ensure_dirs()
    path = predictions_path(config, gw)
    out = record.reset_index()
    out["model"] = model
    out["generated_at"] = now.isoformat()
    out.to_csv(path, index=False)
    log.info("stored gameweek %d predictions: %s", gw, path)
    return path


def load_predictions(config: Config, gw: int) -> tuple[pd.DataFrame, dict[str, str]] | None:
    """The stored record for `gw` and its `{model, generated_at}`, or None."""
    path = predictions_path(config, gw)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    meta = {k: str(df[k].iloc[0]) for k in ("model", "generated_at") if k in df and len(df)}
    df = df.drop(columns=[c for c in ("model", "generated_at") if c in df])
    df = df.set_index(df["player_id"].astype(int)).drop(columns=["player_id"])
    df.index.name = "player_id"
    return df, meta


def predict_gameweek(
    config: Config | None = None,
    client: FPLClient | None = None,
    gw: int | None = None,
    model: str = "component",
    save: bool = True,
) -> GameweekPrediction:
    """Build the frame, fit, predict `gw` (default: next), and store it if still in time."""
    from . import pipeline
    from .data.current import deadlines

    config = config or load_config()
    client = client or FPLClient(config)
    gw = gw or client.next_gw()
    if gw is None:
        raise ValueError("no upcoming gameweek found")
    gw = int(gw)

    frame, feature_cols = pipeline.build(config, upcoming_gw=gw, client=client)
    record = fit_predict(frame, feature_cols, config.season_current, gw, model=model)

    deadline = deadlines(client).get(gw)
    stored = save_predictions(record, gw, config, model, deadline) if save else None
    return GameweekPrediction(gw=gw, model=model, record=record, deadline=deadline, stored=stored)
