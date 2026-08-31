"""Assembling the full dataset, and the parquet it is stored in.

Historical seasons come from the vaastav CSVs; the season in progress comes from
the live API. Both are normalised to the same schema first, so from here down
nothing knows or cares which source a row came from.
"""

from __future__ import annotations

import logging

import pandas as pd

from .config import Config, load_config
from .data.current import played_frame, upcoming_frame
from .data.fpl_api import FPLClient
from .data.historical import load_seasons
from .features.build import build_player_gameweek

log = logging.getLogger(__name__)

FEATURES_PARQUET = "player_gameweek.parquet"
FEATURE_LIST = "feature_columns.txt"


def load_raw(
    config: Config | None = None,
    include_current: bool = True,
    upcoming_gw: int | None = None,
    client: FPLClient | None = None,
) -> pd.DataFrame:
    """Every player-fixture row we have, on the canonical schema.

    `upcoming_gw` appends prediction rows for a future gameweek: same schema,
    outcome columns empty. They ride through feature building with everything
    else so their form features are built from exactly the same code path.
    """
    config = config or load_config()
    frames = [load_seasons(config.season_history, config=config)]

    if include_current or upcoming_gw is not None:
        client = client or FPLClient(config)
        current = played_frame(client, config.season_current)
        if not current.empty:
            log.info("current season %s: %d played rows", config.season_current, len(current))
            frames.append(current)

        if upcoming_gw is not None:
            future = upcoming_frame(client, upcoming_gw, config.season_current)
            log.info("gameweek %d: %d prediction rows", upcoming_gw, len(future))
            frames.append(future)

    return pd.concat(frames, ignore_index=True)


def build(
    config: Config | None = None,
    include_current: bool = True,
    upcoming_gw: int | None = None,
    client: FPLClient | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load everything and build the feature frame."""
    config = config or load_config()
    raw = load_raw(
        config, include_current=include_current, upcoming_gw=upcoming_gw, client=client
    )
    return build_player_gameweek(
        raw,
        windows=list(config.features["windows"]),
        expanding=bool(config.features.get("expanding", True)),
    )


def save(df: pd.DataFrame, feature_cols: list[str], config: Config | None = None) -> tuple:
    """Write the feature frame and its feature list side by side.

    The feature list is stored with the data on purpose: which columns are
    legal model inputs is part of the dataset's contract, not something to be
    re-derived from column-name patterns later.
    """
    config = config or load_config()
    config.ensure_dirs()

    parquet_path = config.processed_dir / FEATURES_PARQUET
    features_path = config.processed_dir / FEATURE_LIST

    df.to_parquet(parquet_path, index=False)
    features_path.write_text("\n".join(feature_cols), encoding="utf-8")

    log.info("wrote %s (%d rows, %d cols)", parquet_path, len(df), df.shape[1])
    return parquet_path, features_path


def load_processed(config: Config | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Read back what `save` wrote."""
    config = config or load_config()
    parquet_path = config.processed_dir / FEATURES_PARQUET
    features_path = config.processed_dir / FEATURE_LIST

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found — run `python -m src.cli build-features` first"
        )

    df = pd.read_parquet(parquet_path)
    feature_cols = features_path.read_text(encoding="utf-8").split("\n")
    return df, [c for c in feature_cols if c]
