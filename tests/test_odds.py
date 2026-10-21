"""Market odds -> expected goals -> two columns on every player row.

What matters: the bookmaker's margin is removed; the Poisson fit recovers the
means that produced a set of prices; football-data's club names reach the right
FPL ids; each row gets its own side's goals the right way round; and a failed
download leaves the columns empty instead of stopping the pipeline.
"""

from __future__ import annotations

import os
import textwrap
from math import exp, factorial

import numpy as np
import pandas as pd
import pytest
import requests

from src.config import load_config
from src.data.odds import (
    OddsLoader,
    attach,
    implied_goals,
    implied_probabilities,
    match_goals,
    season_code,
)


def poisson_prices(home: float, away: float, margin: float = 1.05) -> dict[str, float]:
    """Decimal odds a bookmaker with `margin` would quote for these Poisson means."""
    pmf = lambda lam, k: exp(-lam) * lam**k / factorial(k)
    ph = pd_ = pa = under = 0.0
    for i in range(12):
        for j in range(12):
            p = pmf(home, i) * pmf(away, j)
            if i > j:
                ph += p
            elif i == j:
                pd_ += p
            else:
                pa += p
            if i + j <= 2:
                under += p
    over = 1 - under
    return {"AvgH": 1 / (ph * margin), "AvgD": 1 / (pd_ * margin), "AvgA": 1 / (pa * margin),
            "Avg>2.5": 1 / (over * margin), "Avg<2.5": 1 / (under * margin)}


def test_season_codes() -> None:
    assert season_code("2025-26") == "2526"
    assert season_code("2099-00") == "9900"


def test_the_margin_is_removed() -> None:
    p = implied_probabilities(pd.DataFrame([poisson_prices(1.6, 1.1, margin=1.07)]))
    assert p[["p_home", "p_draw", "p_away"]].sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert 0 < p["p_over"].iloc[0] < 1


def test_bet365_stands_in_when_the_average_is_missing() -> None:
    prices = poisson_prices(1.6, 1.1)
    row = {k.replace("Avg", "B365"): v for k, v in prices.items()}
    row.update({"AvgH": np.nan, "AvgD": np.nan, "AvgA": np.nan, "Avg>2.5": np.nan, "Avg<2.5": np.nan})
    p = implied_probabilities(pd.DataFrame([row]))
    assert p.notna().all(axis=1).iloc[0]


@pytest.mark.parametrize("home, away", [(1.6, 1.1), (2.8, 0.7), (0.9, 1.9), (1.3, 1.3)])
def test_the_fit_recovers_the_means_behind_the_prices(home, away) -> None:
    goals = implied_goals(implied_probabilities(pd.DataFrame([poisson_prices(home, away)])))
    assert goals["goals_home"].iloc[0] == pytest.approx(home, abs=0.03)
    assert goals["goals_away"].iloc[0] == pytest.approx(away, abs=0.03)


def test_an_unpriced_match_stays_empty() -> None:
    goals = implied_goals(implied_probabilities(pd.DataFrame([{"AvgH": np.nan}])))
    assert goals.isna().all(axis=None)


def odds_rows() -> pd.DataFrame:
    rows = [{"HomeTeam": "Man United", "AwayTeam": "Tottenham", **poisson_prices(1.8, 1.2)},
            {"HomeTeam": "Tottenham", "AwayTeam": "Man United", **poisson_prices(1.5, 1.4)},
            {"HomeTeam": "Atlantis FC", "AwayTeam": "Man United", **poisson_prices(1.0, 1.0)}]
    return pd.DataFrame(rows)


def test_club_names_reach_fpl_ids_and_strangers_are_dropped() -> None:
    goals = match_goals(odds_rows(), {"Man Utd": 14, "Spurs": 18})
    assert len(goals) == 2, "the unknown club's match is dropped, not guessed"
    first = goals.set_index(["home_id", "away_id"]).loc[(14, 18)]
    assert first["goals_home"] == pytest.approx(1.8, abs=0.03)


def test_a_short_club_name_matches_the_one_fpl_name_it_starts() -> None:
    rows = pd.DataFrame([{"HomeTeam": "Hull", "AwayTeam": "Man United", **poisson_prices(1.2, 1.5)},
                         {"HomeTeam": "Man", "AwayTeam": "Hull", **poisson_prices(1.2, 1.5)}])
    goals = match_goals(rows, {"Hull City": 11, "Man Utd": 14, "Man City": 13})
    assert goals[["home_id", "away_id"]].values.tolist() == [[11, 14]], "'Man' starts two names: no match"


def test_each_side_gets_its_own_goals() -> None:
    goals = match_goals(odds_rows(), {"Man Utd": 14, "Spurs": 18})
    rows = pd.DataFrame({
        "season": ["2099-00"] * 4,
        "team_id": [14, 18, 18, 14], "opponent_id": [18, 14, 14, 18],
        "was_home": [True, False, True, False],
    })
    out = attach(rows, "2099-00", goals)
    assert out.loc[0, "odds_team_goals"] == pytest.approx(1.8, abs=0.03)   # United at home
    assert out.loc[1, "odds_team_goals"] == pytest.approx(1.2, abs=0.03)   # Spurs away there
    assert out.loc[1, "odds_opp_goals"] == pytest.approx(1.8, abs=0.03)
    assert out.loc[2, "odds_team_goals"] == pytest.approx(1.5, abs=0.03)   # the return fixture


def test_other_seasons_are_left_alone() -> None:
    goals = match_goals(odds_rows(), {"Man Utd": 14, "Spurs": 18})
    rows = pd.DataFrame({"season": ["2098-99"], "team_id": [14], "opponent_id": [18], "was_home": [True]})
    assert attach(rows, "2099-00", goals)["odds_team_goals"].isna().all()


# -- downloading -------------------------------------------------------------------

@pytest.fixture
def config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""
        season:
          current: "2099-00"
          history: []
        paths:
          data_dir: "{(tmp_path / 'data').as_posix()}"
          cache_dir: "{(tmp_path / 'data' / 'cache').as_posix()}"
          processed_dir: "{(tmp_path / 'data' / 'processed').as_posix()}"
        api: {{base_url: "x", ttl: {{default: 1}}, user_agent: "test"}}
        historical: {{vaastav_base: "y"}}
        features: {{windows: [3]}}
    """), encoding="utf-8")
    return load_config(cfg, use_local=False)


class Response:
    def __init__(self, body: bytes, status: int = 200):
        self.content, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class Session:
    def __init__(self, *responses):
        self.responses, self.calls, self.headers = list(responses), [], {}

    def get(self, url, timeout=None):
        self.calls.append(url)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


CSV = b"\xef\xbb\xbfDiv,HomeTeam,AwayTeam,AvgH\nE0,Arsenal,Spurs,1.5\n"


def test_a_finished_season_is_downloaded_once(config) -> None:
    session = Session(Response(CSV))
    loader = OddsLoader(config, session=session)
    assert loader.season("2097-98")["HomeTeam"].tolist() == ["Arsenal"], "the byte-order mark is not in the column name"
    loader.season("2097-98")
    assert len(session.calls) == 1 and session.calls[0].endswith("/9798/E0.csv")


def test_a_failed_download_falls_back_to_the_cached_copy_or_to_nothing(config) -> None:
    loader = OddsLoader(config, session=Session(requests.ConnectionError("offline")))
    assert loader.season("2099-00", current=True) is None, "nothing cached: no odds, no exception"

    loader = OddsLoader(config, session=Session(Response(CSV), requests.ConnectionError("offline")))
    loader.season("2099-00", current=True)
    cached = loader.cache_dir / "E0_9900.csv"
    os.utime(cached, (0, 0))   # long stale, so the next call tries the network
    assert loader.season("2099-00", current=True)["HomeTeam"].tolist() == ["Arsenal"]


def test_upcoming_keeps_only_the_premier_league(config) -> None:
    body = b"Div,HomeTeam,AwayTeam\nE0,Arsenal,Spurs\nEC,Woking,Yeovil\n"
    loader = OddsLoader(config, session=Session(Response(body)))
    assert loader.upcoming()["HomeTeam"].tolist() == ["Arsenal"]
