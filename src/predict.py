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

Fitting is the slow part — about 45 seconds for the component model — and the
training data only changes when a gameweek finishes, so a fitted model is kept
under `data/cache/models/`, keyed on a hash of the training rows, the feature
list, the model code and the LightGBM version. Change any of them and the next
build refits; change none and `serve` starts in seconds rather than a minute.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass
from importlib import metadata
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

# The betting market's prices are published a few days before a gameweek. A
# build earlier than that has none for the week it predicts, and a model fitted
# with them would read the gap as a signal; it falls back to the same model
# fitted without them, which is the variant measured against it.
ODDS_PREFIX = "odds_"
ODDS_MIN_COVERAGE = 0.5

MODEL_CACHE = "models"
# A season's worth of rebuilds is a few dozen files of a few MB; keep the latest.
KEEP_CACHED = 6


@dataclass
class GameweekPrediction:
    gw: int
    model: str
    record: pd.DataFrame            # one row per player, indexed by player_id
    deadline: pd.Timestamp | None
    stored: Path | None             # where it was saved, or None if after the deadline
    odds_priced: bool = True        # False: the week's odds were not out, so the no-odds fit was used


def odds_priced(target: pd.DataFrame) -> bool:
    """Whether the gameweek being predicted has its betting odds yet."""
    cols = [c for c in target.columns if c.startswith(ODDS_PREFIX)]
    return bool(cols) and float(target[cols[0]].notna().mean()) >= ODDS_MIN_COVERAGE


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


def _code_fingerprint() -> str:
    """The code a fitted model came from, so editing it is a refit rather than a stale model."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted((root / "models").glob("*.py")) + [root / "evaluate.py"]:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fit_key(model: str, train: pd.DataFrame, feature_cols: list[str]) -> str:
    """Everything a fit depends on, hashed: same key, same fitted model."""
    try:
        lgb_version = metadata.version("lightgbm")
    except metadata.PackageNotFoundError:
        lgb_version = "none"
    digest = hashlib.sha256()
    for part in (model, "\n".join(feature_cols), "\n".join(map(str, train.columns)),
                 _code_fingerprint(), lgb_version):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    digest.update(pd.util.hash_pandas_object(train, index=False).to_numpy().tobytes())
    return digest.hexdigest()[:24]


def fitted(
    model: str, train: pd.DataFrame, feature_cols: list[str], cache_dir: Path | None = None
) -> Predictor:
    """`model` fitted on `train`, from the cache when the same fit has been done before."""
    if cache_dir is None:
        predictor: Predictor = all_predictors()[model]
        predictor.fit(train, feature_cols)
        return predictor

    path = cache_dir / f"{model}-{fit_key(model, train, feature_cols)}.pkl"
    if path.exists():
        try:
            predictor = pickle.loads(path.read_bytes())
            log.info("fitted %s model from cache: %s", model, path.name)
            path.touch()  # most recently used, so pruning keeps it
            return predictor
        except Exception as exc:  # a half-written or incompatible file: refit over it
            log.warning("ignoring unreadable cached model %s: %s", path.name, exc)

    predictor = all_predictors()[model]
    predictor.fit(train, feature_cols)
    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".tmp")
    partial.write_bytes(pickle.dumps(predictor))
    partial.replace(path)
    for stale in sorted(cache_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime)[:-KEEP_CACHED]:
        stale.unlink(missing_ok=True)
    return predictor


def fit_predict(
    frame: pd.DataFrame,
    feature_cols: list[str],
    season: str,
    gw: int,
    model: str = "component",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fit on everything known before `gw` and return its prediction record.

    `frame` must already hold the gameweek's prediction rows with their
    features built; `training_window` is the one definition of "known before".
    With `cache_dir`, a fit already done on identical training rows is reused.
    """
    target = frame[(frame["season"] == season) & (frame["gw"] == gw)]
    if target.empty:
        raise ValueError(f"no rows for gameweek {gw} of {season}")
    if not odds_priced(target):
        feature_cols = [c for c in feature_cols if not c.startswith(ODDS_PREFIX)]
        log.info("gameweek %d odds not published yet; predicting with the no-odds fit", gw)
    predictor = fitted(model, training_window(frame, season, gw), feature_cols, cache_dir)
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

    After the deadline the API's inputs — prices, availability, FPL's own
    figure — describe the *next* gameweek, so a rerun would overwrite what the model said in time with
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
    record = fit_predict(
        frame, feature_cols, config.season_current, gw, model=model,
        cache_dir=config.cache_dir / MODEL_CACHE,
    )

    deadline = deadlines(client).get(gw)
    stored = save_predictions(record, gw, config, model, deadline) if save else None
    priced = odds_priced(frame[(frame["season"] == config.season_current) & (frame["gw"] == gw)])
    return GameweekPrediction(gw=gw, model=model, record=record, deadline=deadline, stored=stored,
                              odds_priced=priced)
