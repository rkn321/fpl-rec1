"""How well a fixture rating predicts points weeks ahead: FPL's difficulty against Elo.

The page projects every week after the next one as "this player's usual rate,
times how kind the fixture is": the odds behind next week's prediction do not
exist yet that far out. This measures the second factor on its own, with the
first held fixed for every method:

- **usual rate** — the player's points per start this season before the as-of
  deadline (three starts at least), which carries no particular fixture
- **fixture factor** — a multiplier per position, fitted the way the page's
  difficulty table was: each starter's points against their own season
  average, pulled toward the all-position figure by n / (n + 300). Fitted only
  on seasons before the one being tested.
- **target** — the points of every player who started (60+ minutes) 0 to 4
  gameweeks after the as-of deadline, with Elo taken as it stood *at that
  deadline*. Lead 0 is the week the odds already cover; 1-4 are the weeks
  this is for.

**FPL's difficulty is flattered here.** The historical record holds the ratings
as FPL last revised them, after the season had played out (the snapshots notes
in the README show the revisions are real), so in this test it has seen later
results and Elo has not. A tie would favour Elo in practice.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .data.elo import EloBook, load_results, team_names

log = logging.getLogger(__name__)

POSITIONS = ("GK", "DEF", "MID", "FWD")
STARTER_MINUTES = 60
MIN_SEASON_STARTS = 5      # to have a season average worth dividing by
MIN_PRIOR_STARTS = 3       # to have a usual rate at the as-of deadline
SHRINK = 300               # starts at which a position's cell is trusted half
# Elo gaps are capped here before the line is applied: past about 300 points
# (a title side at home to a relegation side) there are too few matches to
# say the effect keeps growing.
ELO_CAP = 300.0
LEADS = (0, 1, 2, 3, 4)
METHODS = ("none", "fdr", "elo")


def starters(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[(df["minutes"] >= STARTER_MINUTES) & df["position"].isin(POSITIONS)]
    return rows[["season", "gw", "player_id", "name", "position", "team_id", "opponent_id",
                 "was_home", "kickoff_time", "team_difficulty", "total_points"]].copy()


def deadlines(df: pd.DataFrame) -> dict[tuple[str, int], pd.Timestamp]:
    """(season, gw) -> the day of that gameweek's first kickoff, a stand-in for its deadline."""
    first = df.groupby(["season", "gw"])["kickoff_time"].min()
    return {k: v.tz_convert(None).normalize() for k, v in first.items()}


def attach_gap(rows: pd.DataFrame, book: EloBook, names: dict[str, dict[int, str]],
               when: pd.Series) -> pd.Series:
    """Each row's Elo gap (see `fixture_gap`), with ratings as they stood on `when`."""
    def name(season: str, tid: int) -> str | None:
        return names.get(season, {}).get(int(tid))

    team = pd.Series([name(s, t) for s, t in zip(rows["season"], rows["team_id"])], index=rows.index)
    opp = pd.Series([name(s, t) for s, t in zip(rows["season"], rows["opponent_id"])], index=rows.index)
    home = pd.Series(np.where(rows["was_home"].astype(bool), book.home, -book.home), index=rows.index)
    out = pd.Series(np.nan, index=rows.index)
    for day, idx in when.groupby(when).groups.items():
        ratings = book.as_of(day)
        out.loc[idx] = team.loc[idx].map(ratings) - opp.loc[idx].map(ratings) + home.loc[idx]
    return out


def season_ratio(rows: pd.DataFrame) -> pd.Series:
    """Points over the player's own average per start that season; NaN below MIN_SEASON_STARTS."""
    g = rows.groupby(["season", "player_id"])["total_points"]
    mean, n = g.transform("mean"), g.transform("size")
    return (rows["total_points"] / mean.where((n >= MIN_SEASON_STARTS) & (mean > 0)))


@dataclass
class Multipliers:
    """A fitted fixture factor, by position.

    `fdr`: a table, position x difficulty 1-5. `elo`: a line per position,
    columns `a` and `b`, applied as a + b x gap with the gap capped at ELO_CAP.
    The line replaced eight bins of gap: the bins were not monotone (a
    defender's 1.04 at a gap of -24 against 0.99 at +26 is noise, not
    football) and on the backtest neither beat the other by more than 0.06.
    """

    kind: str
    table: pd.DataFrame
    overall: float

    def __call__(self, position: pd.Series, value: pd.Series) -> np.ndarray:
        out = np.full(len(position), self.overall)
        if self.kind == "none":
            return out
        for pos in POSITIONS:
            mask = (position == pos).to_numpy() & value.notna().to_numpy()
            if not mask.any() or pos not in self.table.index:
                continue
            if self.kind == "fdr":
                lookup = self.table.loc[pos].to_dict()
                out[mask] = [lookup.get(int(v), self.overall) for v in value[mask]]
            else:
                a, b = self.table.loc[pos, "a"], self.table.loc[pos, "b"]
                out[mask] = a + b * np.clip(value[mask].to_numpy(dtype=float), -ELO_CAP, ELO_CAP)
        return out


def _shrunk(cell: float, every: float, n: int) -> float:
    """A position's figure pulled toward the all-position one, by n / (n + SHRINK)."""
    w = n / (n + SHRINK)
    return w * cell + (1 - w) * every


def fit(train: pd.DataFrame, kind: str) -> Multipliers:
    """Fit a fixture factor on starter rows that carry `ratio` (and `gap` for Elo)."""
    t = train.dropna(subset=["ratio"])
    overall = float(t["ratio"].mean())
    if kind == "none":
        return Multipliers(kind, pd.DataFrame(), overall)

    if kind == "fdr":
        every = t.groupby("team_difficulty")["ratio"].mean()
        cells = t.groupby(["position", "team_difficulty"])["ratio"].agg(["mean", "size"])
        table: dict[str, dict] = {}
        for (pos, k), row in cells.iterrows():
            table.setdefault(pos, {})[int(k)] = _shrunk(row["mean"], every[k], int(row["size"]))
        return Multipliers(kind, pd.DataFrame(table).T.sort_index(axis=1), overall)

    t = t.dropna(subset=["gap"])
    x = t["gap"].clip(-ELO_CAP, ELO_CAP)
    b_all, a_all = np.polyfit(x, t["ratio"], 1)
    lines = {}
    for pos, g in t.groupby("position"):
        b, a = np.polyfit(g["gap"].clip(-ELO_CAP, ELO_CAP), g["ratio"], 1)
        lines[pos] = {"a": _shrunk(a, a_all, len(g)), "b": _shrunk(b, b_all, len(g))}
    return Multipliers(kind, pd.DataFrame(lines).T, overall)


def _spearman_by_position(g: pd.DataFrame, col: str) -> float:
    vals = [s[col].rank().corr(s["total_points"].rank()) for _, s in g.groupby("position") if len(s) >= 10]
    vals = [v for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else float("nan")


def run(df: pd.DataFrame, config: Config, seasons: Iterable[str],
        team_ids: dict[str, dict[str, int]], leads: Iterable[int] = LEADS,
        first_gw: int = 6) -> dict[str, pd.DataFrame]:
    """Every (season, as-of gameweek, lead) prediction, and the summary per method."""
    seasons = list(seasons)
    results = load_results(config, max(df["season"]))
    book = EloBook(results)
    names = {s: team_names(results, s, ids) for s, ids in team_ids.items()}
    dl = deadlines(df)

    rows = starters(df)
    rows["ratio"] = season_ratio(rows)
    # For fitting, a fixture is described as it stood at its own deadline.
    rows["gap"] = attach_gap(rows, book, names,
                             pd.Series([dl[(s, int(g))] for s, g in zip(rows["season"], rows["gw"])], index=rows.index))

    preds = []
    for season in seasons:
        train = rows[rows["season"] < season]
        if train.empty:
            log.warning("no earlier season to fit on for %s; skipped", season)
            continue
        fitted = {kind: fit(train, kind) for kind in METHODS}
        test = rows[rows["season"] == season]
        last_gw = int(test["gw"].max())
        for t in range(first_gw, last_gw + 1):
            if (season, t) not in dl:
                continue
            before = test[test["gw"] < t].groupby("player_id")["total_points"].agg(["mean", "size"])
            rate = before.loc[before["size"] >= MIN_PRIOR_STARTS, "mean"]
            target = test[test["gw"].isin([t + L for L in leads]) & test["player_id"].isin(rate.index)].copy()
            if target.empty:
                continue
            target["as_of"], target["lead"] = t, target["gw"] - t
            target["rate"] = target["player_id"].map(rate)
            target["gap_then"] = attach_gap(target, book, names, pd.Series(dl[(season, t)], index=target.index))
            for kind, m in fitted.items():
                value = target["team_difficulty"] if kind == "fdr" else target["gap_then"]
                target[f"pred_{kind}"] = target["rate"] * m(target["position"], value)
            preds.append(target)

    every = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return {"predictions": every, "summary": summarise(every)}


def summarise(every: pd.DataFrame, draws: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Per season and lead: squared and absolute error, ranking, and Elo minus FPL with a 95% interval.

    The interval resamples whole as-of gameweeks, since the rows within one
    share their ratings and their weekend.
    """
    if every.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    out = []
    for (season, lead), g in every.groupby(["season", "lead"]):
        row = {"season": season, "lead": int(lead), "starts": len(g)}
        for kind in METHODS:
            err = g[f"pred_{kind}"] - g["total_points"]
            row[f"mse_{kind}"] = float((err ** 2).mean())
            row[f"mae_{kind}"] = float(err.abs().mean())
            row[f"rank_{kind}"] = float(np.nanmean([_spearman_by_position(w, f"pred_{kind}")
                                                    for _, w in g.groupby("as_of")]))
        per_week = g.assign(
            d=(g["pred_elo"] - g["total_points"]) ** 2 - (g["pred_fdr"] - g["total_points"]) ** 2
        ).groupby("as_of")["d"].agg(["sum", "size"])
        sums, sizes = per_week["sum"].to_numpy(), per_week["size"].to_numpy()
        idx = rng.integers(0, len(sums), size=(draws, len(sums)))
        boot = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
        row["elo_minus_fdr_mse"] = float(sums.sum() / sizes.sum())
        row["ci_low"], row["ci_high"] = (float(x) for x in np.quantile(boot, [0.025, 0.975]))
        out.append(row)
    return pd.DataFrame(out)
