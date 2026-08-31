"""Lagged rolling-window helpers.

Every function here is **lagged to the gameweek deadline**, which is a stricter
rule than "lag by one row" and is the one that matches how FPL is actually
played: you pick a team once, before the deadline, and it stands for the whole
gameweek.

The consequence is subtle but matters. In a double gameweek a player has two
fixtures. Lagging by one *row* would let the second fixture's features see the
first fixture's result — information you did not have when you picked. So every
feature on a row in gameweek `g` is computed from fixtures in gameweeks
**strictly before `g`**, and both fixtures of a double therefore carry identical
form features.

There is no non-lagged variant of anything here, deliberately: the only way to
build a form feature is the safe way (brief §6.1).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

# Sort order that defines "before". Gameweek first, then kickoff and fixture id
# to break ties deterministically.
SORT_COLS: list[str] = ["season", "player_id", "gw", "kickoff_time", "fixture_id"]

# The unit the lag is taken at. Nothing from the current gameweek is visible.
PERIOD_COL = "gw"


def sort_canonical(df: pd.DataFrame, sort_cols: Sequence[str] = SORT_COLS) -> pd.DataFrame:
    cols = [c for c in sort_cols if c in df.columns]
    return df.sort_values(cols, kind="stable").reset_index(drop=True)


def _flatten(result: pd.DataFrame, n_levels: int, index: pd.Index) -> pd.DataFrame:
    """groupby().rolling() returns a MultiIndex of (group keys..., row index)."""
    result = result.copy()
    result.index = result.index.droplevel(list(range(n_levels)))
    return result.reindex(index)


def as_of_previous_gameweek(
    inclusive: pd.DataFrame,
    df: pd.DataFrame,
    group_cols: Sequence[str],
    period_col: str = PERIOD_COL,
) -> pd.DataFrame:
    """Carry each row the value the series held at the end of the previous gameweek.

    `inclusive` holds a running quantity computed *including* each row. This
    takes its value at the last fixture of the preceding gameweek and stamps it
    on every row of the current one — so a double gameweek's two fixtures get
    the same, deadline-legal value.
    """
    group_cols = list(group_cols)
    keys = group_cols + [period_col]
    value_cols = list(inclusive.columns)

    tmp = inclusive.copy()
    for k in keys:
        tmp[k] = df[k].to_numpy()

    # sort=True puts gameweeks in ascending order, so shift(1) is "previous gameweek".
    per_gw = tmp.groupby(keys, sort=True, observed=True)[value_cols].last()
    previous = per_gw.groupby(level=list(range(len(group_cols))), sort=False).shift(1)

    lookup = previous.reindex(pd.MultiIndex.from_frame(df[keys]))
    lookup.index = df.index
    return lookup


def rolling_stat(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    cols: Sequence[str],
    window: int,
    stat: str = "mean",
    min_periods: int = 1,
) -> pd.DataFrame:
    """`stat` over the last `window` fixtures, as known at this gameweek's deadline."""
    group_cols = list(group_cols)
    cols = list(cols)
    grouped = df.groupby(group_cols, sort=False, observed=True)[cols]
    inclusive = getattr(grouped.rolling(window, min_periods=min_periods), stat)()
    inclusive = _flatten(inclusive, len(group_cols), df.index)
    return as_of_previous_gameweek(inclusive, df, group_cols)


def expanding_stat(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    cols: Sequence[str],
    stat: str = "mean",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Season-to-date `stat`, as known at this gameweek's deadline."""
    group_cols = list(group_cols)
    cols = list(cols)
    grouped = df.groupby(group_cols, sort=False, observed=True)[cols]
    inclusive = getattr(grouped.expanding(min_periods=min_periods), stat)()
    inclusive = _flatten(inclusive, len(group_cols), df.index)
    return as_of_previous_gameweek(inclusive, df, group_cols)


def previous_fixture_value(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    cols: Sequence[str],
    n: int = 1,
) -> pd.DataFrame:
    """Value from the `n`-th most recent fixture *before* this row's gameweek.

    `n=1` is "last time they played", which for both fixtures of a double is the
    same fixture in the preceding gameweek.
    """
    group_cols = list(group_cols)
    cols = list(cols)
    if n < 1:
        raise ValueError("n must be >= 1")

    shifted = df.groupby(group_cols, sort=False, observed=True)[cols].shift(n - 1)
    return as_of_previous_gameweek(shifted, df, group_cols)


def add_rolling_features(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    cols: Sequence[str],
    windows: Iterable[int],
    prefix: str = "",
    expanding: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach `{prefix}{col}_r{w}` means plus `{prefix}{col}_todate`.

    Returns the widened frame and the list of column names added, so callers
    never have to guess the feature set from string patterns.
    """
    cols = [c for c in cols if c in df.columns]
    added: list[str] = []
    new: dict[str, pd.Series] = {}

    for w in windows:
        rolled = rolling_stat(df, group_cols, cols, window=w, stat="mean")
        for c in cols:
            name = f"{prefix}{c}_r{w}"
            new[name] = rolled[c]
            added.append(name)

    if expanding:
        todate = expanding_stat(df, group_cols, cols, stat="mean")
        for c in cols:
            name = f"{prefix}{c}_todate"
            new[name] = todate[c]
            added.append(name)

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1), added
