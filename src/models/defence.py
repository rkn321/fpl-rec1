"""Clean sheets, goals conceded, saves and DEFCON (Phase 3).

Four payouts with genuinely different shapes, so four treatments:

* **Clean sheet** is a *team* event. It is modelled once per team-fixture rather
  than once per player, because eleven players of the same side share one
  outcome — fitting it per player would relearn the same thing eleven times and
  let two team-mates disagree about whether their team kept a clean sheet.
  That means the model may only see what describes a team in a fixture
  (`team_level`): it was once trained on an arbitrary squad member's row per
  match, then predicted for every player from his own row, so a keeper and a
  defender on the same side got different odds of the same event.
* **Goals conceded** pays -1 per *two* conceded. A step function needs the whole
  distribution, not its mean: E[floor(X/2)] is not floor(E[X]/2). A Poisson mean
  is fitted and the expectation summed over the pmf.
* **Saves** pay 1 per 3, so the same argument applies, and only to keepers: the
  expected payout is E[floor(X/3)] over a Poisson of the fitted mean, not the mean
  divided by three, which paid every keeper about 0.3 a match too much (out of fold
  over 2,045 keeper-matches: bias +0.30 and RMSE 0.743, against -0.03 and 0.679).
  Penalty saves pay 5 and are too rare to model — 1.7% of keeper starts — so they
  are a flat base rate, the way cards are.
* **DEFCON** is a threshold on a count — 10 combined defensive actions for a
  defender, 12 for everyone else. Modelled as P(threshold reached) directly,
  because regressing the count and thresholding the mean is badly biased: a
  player averaging 8 with real variance clears 10 far more often than a point
  estimate of 8 suggests.

**The market answers the team questions where it has priced the match.** The
bookmakers' expected goals for and against are the best single predictor of a
clean sheet there is, and a tree model fitted to ~3,000 noisy team rows does
worse than a constant guess: on 498 keeper-matches the tree's Brier score was
0.197 and a constant 24% scored 0.180, while a Poisson from the market's expected
goals, exp(-xG against), scored 0.166. So where a row carries both odds columns,
the clean-sheet probability is a two-feature logistic fit on them and the goals
conceded are the market's expected goals against, scaled by one fitted constant.
Rows without odds (a week whose prices are not out, and the odds-free fit the
page falls back to) used to keep the tree models above, and those are worse than
a constant guess: out of fold over 2,202 keeper-matches the odds-free trees scored
0.186 against 0.177 for the constant, and gave a side as much as 86% to keep a
clean sheet (the market-based fit's highest, over the same three seasons, was 54%). **Where the match has no
price, Elo stands in for it.** A logistic on the two sides' rating gap (home
advantage included) scores 0.169, within 0.002 of the market's 0.167. The forty
sides it rated most likely to keep one averaged 45% and kept it 47.5% of the time;
the trees' forty averaged 72% and kept it 32.5%. Goals conceded are a Poisson fitted
on the same gap. The gap is the `elo_gap` column (`src/data/elo.py`), read here and
by nothing else. The market still wins where it has priced the match, and a row
with neither keeps the trees.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from .base import Component, poisson_tail_expectation, recency_weights
from .scoring import (
    DEFCON_THRESHOLD,
    GOALS_CONCEDED_PER_PENALTY,
    PENALTY_SAVE_POINTS,
    SAVES_PER_POINT,
)

# What describes a team in a fixture, and nothing about any one player in it.
TEAM_FEATURES = frozenset({
    "gw", "team_difficulty", "opp_difficulty", "odds_team_goals", "odds_opp_goals",
    "is_home", "is_double_gw", "team_fixtures_in_gw",
})
TEAM_FEATURE_PREFIXES = ("team_form_", "opp_form_", "elo_")

# The market's expected goals for and against the side, as the odds join names them.
MARKET_COLS = ("odds_opp_goals", "odds_team_goals")
MIN_MARKET_ROWS = 400

# Team strength for a match with no price: the rating gap, in hundreds of points.
ELO_COL = "elo_gap"
ELO_SCALE = 100.0


def team_level(features: Sequence[str]) -> list[str]:
    """The subset of `features` that is the same for every player of a team in a fixture."""
    return [c for c in features if c in TEAM_FEATURES or c.startswith(TEAM_FEATURE_PREFIXES)]


class DefenceModel:
    """Team clean sheet, goals conceded, saves and DEFCON probability."""

    def __init__(self) -> None:
        self.team_clean_sheet = Component(classifier=True, fallback=0.25, name="team_cs")
        self.team_conceded = Component(objective="poisson", fallback=1.4, name="team_gc")
        self.saves = Component(objective="poisson", fallback=0.0, name="saves")
        self.defcon = Component(classifier=True, fallback=0.0, name="defcon")
        self.market_clean_sheet: LogisticRegression | None = None
        self.market_scale = 1.0
        self.elo_clean_sheet: LogisticRegression | None = None
        self.elo_conceded: PoissonRegressor | None = None
        # Points a starting keeper collects from penalty saves, a flat base rate.
        self.penalty_save_points = 0.0

    # -- team level ---------------------------------------------------------
    @staticmethod
    def _team_rows(frame: pd.DataFrame) -> pd.DataFrame:
        """One row per team-fixture — the grain a clean sheet actually happens at."""
        return frame.drop_duplicates(subset=["season", "team_id", "fixture_id"])

    def _fit_market(self, teams: pd.DataFrame, weight: pd.Series, features: Sequence[str]) -> None:
        """Clean sheet and goals conceded from the odds, when the odds were part of the fit."""
        self.market_clean_sheet, self.market_scale = None, 1.0
        cols = list(MARKET_COLS)
        if not all(c in features and c in teams.columns for c in cols):
            return
        priced = teams.dropna(subset=cols)
        if len(priced) < MIN_MARKET_ROWS:
            return
        w = weight.loc[priced.index].to_numpy(dtype=float)
        kept = (priced["team_goals_against"] == 0).to_numpy()
        if kept.all() or not kept.any():
            return
        self.market_clean_sheet = LogisticRegression(C=100.0, max_iter=500).fit(
            priced[cols].to_numpy(dtype=float), kept, sample_weight=w
        )
        # Scored goals match the market on average (within 0.1 in every season);
        # the constant absorbs whatever small gap there is.
        self.market_scale = float(
            np.average(priced["team_goals_against"].astype(float), weights=w)
            / np.average(priced["odds_opp_goals"].astype(float), weights=w)
        )

    def _fit_elo(self, teams: pd.DataFrame, weight: pd.Series) -> None:
        """Clean sheet and goals conceded from the Elo gap, for matches the market has not priced.

        Fitted on what happened, not on the odds, so it needs no odds columns and the
        odds-free fit has it too.
        """
        self.elo_clean_sheet, self.elo_conceded = None, None
        if ELO_COL not in teams.columns:
            return
        rated = teams.dropna(subset=[ELO_COL])
        if len(rated) < MIN_MARKET_ROWS:
            return
        kept = (rated["team_goals_against"] == 0).to_numpy()
        if kept.all() or not kept.any():
            return
        w = weight.loc[rated.index].to_numpy(dtype=float)
        gap = rated[[ELO_COL]].to_numpy(dtype=float) / ELO_SCALE
        self.elo_clean_sheet = LogisticRegression(C=100.0, max_iter=500).fit(gap, kept, sample_weight=w)
        self.elo_conceded = PoissonRegressor(alpha=0.0, max_iter=300).fit(
            gap, rated["team_goals_against"].astype(float), sample_weight=w
        )

    def fit(self, train: pd.DataFrame, features: Sequence[str]) -> "DefenceModel":
        teams = self._team_rows(train)
        team_weight = recency_weights(teams)
        team_features = team_level(features)

        self.team_clean_sheet.fit(
            teams, team_features, (teams["team_goals_against"] == 0).astype(float), team_weight
        )
        self.team_conceded.fit(
            teams, team_features, teams["team_goals_against"].astype(float), team_weight
        )
        self._fit_market(teams, team_weight, features)
        self._fit_elo(teams, team_weight)

        keepers = train[(train["position"] == "GK") & (train["minutes"] >= 60)]
        self.saves.fit(keepers, features, keepers["saves"].astype(float), recency_weights(keepers))
        self.penalty_save_points = 0.0
        if "penalties_saved" in keepers and keepers["penalties_saved"].notna().any():
            self.penalty_save_points = PENALTY_SAVE_POINTS * float(keepers["penalties_saved"].mean())

        # DEFCON only exists from 2025-26; earlier rows are NaN and drop out.
        played = train[(train["minutes"] >= 60) & train["defensive_contribution"].notna()]
        if not played.empty:
            threshold = played["position"].map(DEFCON_THRESHOLD).fillna(12)
            hit = (played["defensive_contribution"].astype(float) >= threshold).astype(float)
            self.defcon.fit(played, features, hit, recency_weights(played))
        return self

    def _team_predictions(self, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """P(clean sheet) and expected goals conceded for each row's team in its fixture."""
        p_cs = np.clip(self.team_clean_sheet.predict(test), 0.0, 1.0)
        conceded = np.clip(self.team_conceded.predict(test), 0.0, None)

        # Elo, where the rating gap is known: a row with no price is better served
        # by team strength than by the trees. The market below overrides it.
        if self.elo_clean_sheet is not None and ELO_COL in test.columns:
            rated = test[ELO_COL].notna().to_numpy()
            if rated.any():
                gap = test.loc[rated, [ELO_COL]].to_numpy(dtype=float) / ELO_SCALE
                p_cs[rated] = self.elo_clean_sheet.predict_proba(gap)[:, 1]
                conceded[rated] = np.clip(self.elo_conceded.predict(gap), 0.0, None)

        if self.market_clean_sheet is not None:
            cols = list(MARKET_COLS)
            priced = test[cols].notna().all(axis=1).to_numpy()
            if priced.any():
                p_cs[priced] = self.market_clean_sheet.predict_proba(
                    test.loc[priced, cols].to_numpy(dtype=float)
                )[:, 1]
                conceded[priced] = np.clip(
                    test.loc[priced, "odds_opp_goals"].to_numpy(dtype=float) * self.market_scale, 0.0, None
                )
        return p_cs, conceded

    def predict(self, test: pd.DataFrame, p_60: np.ndarray) -> pd.DataFrame:
        p_60 = np.clip(np.asarray(p_60, dtype=float), 0.0, 1.0)

        p_cs, conceded = self._team_predictions(test)

        # Only paid out if the player is on for 60 minutes.
        conceded_penalty = poisson_tail_expectation(conceded, per=GOALS_CONCEDED_PER_PENALTY)

        saves = np.clip(self.saves.predict(test), 0.0, None)
        is_gk = (test["position"] == "GK").to_numpy()
        # A step payout again: one point per three saves is E[floor(X / 3)], which
        # is below the mean over three, and the shortfall is the whole tail.
        save_points = np.where(
            is_gk,
            poisson_tail_expectation(saves, per=SAVES_PER_POINT) + self.penalty_save_points,
            0.0,
        ) * p_60

        p_defcon = np.clip(self.defcon.predict(test), 0.0, 1.0) * p_60

        return pd.DataFrame(
            {
                "p_clean_sheet": p_cs,
                "expected_conceded": conceded,
                "conceded_penalty": conceded_penalty,
                "expected_save_points": save_points,
                "p_defcon": p_defcon,
            },
            index=test.index,
        )
