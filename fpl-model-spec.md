# FPL Prediction Model — Project Brief

> Hand this file to Claude Code as the starting spec. Rename to `CLAUDE.md` if you
> want it auto-loaded as persistent context. Everything below reflects the
> **2026/27 FPL season** rules.

## 1. Objective

Build a system that predicts **expected FPL points per player per gameweek**, and
(optionally) an optimiser that picks the best legal 15-man squad / starting XI
under the game's constraints.

Predicting total points as a single black box works poorly because the position
multipliers and DEFCON thresholds behave very differently. **Predict the scoring
components separately, then combine them** through the scoring rules below.

Primary target: `expected_points[player, gameweek]`
Secondary targets (component models): `p(start)`, `expected_minutes`,
`expected_goals`, `expected_assists`, `p(clean_sheet)`, `p(defcon)`, `expected_bonus`.

## 2. Tech stack

- **Python 3.11+**, `pandas`, `numpy`, `scikit-learn`, `polars` (optional, for speed).
- Gradient boosting: `lightgbm` or `xgboost` for the component models.
- Data pull: `requests` / `httpx` for the official API; `soccerdata` for
  FBref/Understat; `understat` for shot-level xG.
- Storage: start with local parquet files; add DuckDB if the joins get heavy.
- Config: a single `config.yaml`. No secrets needed for the core FPL API.
- Testing: `pytest`. Reproducibility: pin versions, seed everything.

## 3. Data sources

### 3.1 Official FPL API (backbone — free, no auth, JSON)
Base URL: `https://fantasy.premierleague.com/api/`

| Endpoint | Gives you |
|---|---|
| `bootstrap-static/` | All players, teams, gameweeks: prices, ownership, form, ICT, position (`element_type`), season totals, chip availability, full fixture list |
| `fixtures/` | All fixtures + FPL difficulty ratings (`team_h_difficulty`, `team_a_difficulty`) |
| `element-summary/{player_id}/` | Per-player gameweek history + upcoming fixtures |
| `event/{gw}/live/` | Live per-player stat breakdown for a gameweek |
| `entry/{team_id}/history/` | A manager's chip usage (`chips` array), transfers, past ranks |

Notes: cache `bootstrap-static` ~hourly; it can 503 during deadlines/launch, so
add retry + backoff. `element_type`: 1=GK, 2=DEF, 3=MID, 4=FWD.

### 3.2 Ready-made historical datasets (don't re-scrape years yourself)
- **vaastav/Fantasy-Premier-League** (GitHub) — season-by-season CSVs from the
  official API, merged with Understat xG/xA. Read raw via
  `raw.githubusercontent.com`.
- **olbauday/FPL-Core-Insights** (GitHub) — includes 2026/27, fuses the API with
  match stats + dynamic team Elo, aligned to official FPL IDs, same layout as
  vaastav's so tooling only needs a path change.

### 3.3 Underlying performance data (xG — the real predictive juice)
- **Understat** — xG, xA, npxG, xGChain, xGBuildup, shot-level with coordinates,
  big-5 leagues from 2014/15. Scrapeable without heavy bot protection. Use the
  `understat` Python package or `soccerdata`.
- **FBref** (StatsBomb-powered) — deeper metrics (progressive carries, SCA/GCA,
  defensive actions). Behind Cloudflare; pull via `soccerdata` / `worldfootballR`,
  not a hand-rolled scraper.

### 3.4 Odds / market data (strong, underused features)
Implied clean-sheet probability and team goal expectancy are excellent features.
- **the-odds-api.com** — free tier, live odds via API.
- **football-data.co.uk** — historical CSVs incl. closing odds, many seasons back.

### 3.5 Fixtures, results, team strength
- **football-data.org** — free API (needs a key) for schedules/results.
- **clubelo.com** — free API for team Elo ratings.

### 3.6 Minutes / lineups / injuries (hardest and most important)
A haul is worthless if the player is benched, so this drives everything. No clean
free API. Predicted lineups + injury news come from RotoWire, Fantasy Football
Scout, Physioroom (scraping or paid tiers). Realistically: **model minutes yourself**
from rolling `minutes` in the API, supplemented by a scraped injury/suspension feed.

## 4. Domain rules the model must encode (2026/27)

### Squad & selection
- 15 players for £100.0m: **2 GK, 5 DEF, 5 MID, 3 FWD**. Max **3 per club**.
- Starting XI: 1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD (valid shapes 3-4-3 … 5-4-1).
- Captain scores double; if captain plays 0 minutes in the whole gameweek it
  passes to the vice-captain. Judged over the full gameweek (matters in doubles).
- Autosub: a 0-minute starter is replaced by the first bench player that keeps a
  legal formation (bench GK only replaces GK).

### Transfers & prices
- 1 free transfer per gameweek, roll up to **5**. Extra transfers cost **−4** each.
- Prices drift with net transfer activity. On sale you get purchase price + half of
  any rise (rounded down to 0.1m) — you don't capture full profit.

### Scoring
| Action | Points |
|---|---|
| Played ≤60 min | 1 |
| Played 60+ min | 2 |
| Goal — DEF / MID / FWD | 6 / 5 / 4 (GK highest, rare) |
| Assist (any position) | 3 |
| Clean sheet — GK/DEF / MID (needs 60+ min) | 4 / 1 |
| GK saves | 1 per 3 saves |
| Penalty save | +5 |
| Penalty miss | −2 |
| Goals conceded (GK/DEF) | −1 per 2 conceded |
| Yellow / red card | −1 / −3 |
| Own goal | −2 |

### Defensive contributions (DEFCON) — model this explicitly
- **Defender:** +2 for reaching **10** combined clearances, blocks, interceptions,
  tackles (CBIT) in a match.
- **Mid/Fwd:** +2 for reaching **12** combined clearances, blocks, interceptions,
  tackles + ball recoveries (CBIRT).
- Capped at 2 per match (double the threshold ≠ 4). Repeatable, clean-sheet-independent.

### Bonus (BPS)
Top 3 in each match get 3 / 2 / 1 (ties shared). BPS is computed from ~32 match
stats. For 2026/27 the BPS was re-weighted to reduce overlap with DEFCON (now
favours attackers/full-backs), and players are no longer punished for being
tackled/dispossessed.

### Timing gotcha
Gameweek scores become **final at 09:00 UK the day after** the gameweek's last
match — bonus and DEFCON can shift after full time. Don't treat provisional
live scores as final when building training labels.

### Chips (two sets — one per half of the season, 8 total)
Wildcard (unlimited permanent transfers), Free Hit (one-week unlimited reshuffle
that reverts, unavailable GW1), Triple Captain (captain ×3), Bench Boost (all 15
score). First set expires at the GW19 deadline; only one chip per gameweek.
Chip state is in `bootstrap-static` and `entry/{id}/history`.

## 5. Suggested project structure

```
fpl-model/
├── config.yaml
├── src/
│   ├── data/
│   │   ├── fpl_api.py        # official API client (cache + retry/backoff)
│   │   ├── historical.py     # vaastav / olbauday loaders
│   │   ├── understat.py      # xG / shot-level
│   │   ├── odds.py           # market data
│   │   └── injuries.py       # scraped minutes/lineup/injury feed
│   ├── features/
│   │   ├── build.py          # assemble player-gameweek feature frame
│   │   └── rolling.py        # form/rolling-window features (lagged!)
│   ├── models/
│   │   ├── minutes.py        # p(start), expected_minutes
│   │   ├── attack.py         # xG/xA -> goals/assists
│   │   ├── defence.py        # clean sheet, DEFCON, saves
│   │   ├── bonus.py          # expected bonus
│   │   └── combine.py        # components -> expected points via scoring rules
│   ├── optimise/
│   │   └── squad.py          # ILP: pick squad under constraints (optional)
│   └── evaluate.py
├── data/                     # parquet cache (gitignored)
├── notebooks/
└── tests/
```

## 6. Critical gotchas (read before writing features)

1. **No data leakage.** Only feed the model what was knowable *before* the deadline.
   The vaastav dataset's `xP` column is scraped post-gameweek and may reflect
   post-match info — shift it by one within each player group, or drop it. Same
   principle for every rolling feature: lag them.
2. **Join on FPL element IDs, not names.** Names differ across FPL / Understat /
   FBref (accents, nicknames, transfers). Both community datasets align to official
   IDs — lean on that. Maintain an ID crosswalk table.
3. **Minutes are the make-or-break feature.** Build `p(start)` and
   `expected_minutes` first; multiply component outputs by playing-time probability.
4. **Promoted-team data is thin.** New/promoted clubs and new signings have little
   PL history — handle cold-start (priors from odds, team strength, transfer fee).
5. **Fixtures aren't fixed.** Postponements, blanks and doubles happen; key on
   (player, gameweek, fixture_id), not (player, gameweek) alone.

## 7. Evaluation

- Backtest **walk-forward** by gameweek (train on GW ≤ t, predict t+1). Never
  shuffle across time.
- Metrics: MAE / RMSE on points; but also **rank correlation** (Spearman) within
  position, since selection cares about ordering more than absolute error.
- Baselines to beat: (a) last-3-GW average points, (b) the FPL API's own `ep_next`
  expected-points field, (c) a naive minutes × season points-per-90.
- Track calibration on the probability components (clean sheet, start, DEFCON).

## 8. Phased build plan

**Phase 1 — Data foundation**
Build the API client (cache + backoff), load one historical season, establish the
ID crosswalk, and materialise a tidy `player_gameweek` parquet with correctly
**lagged** features. Deliverable: a clean feature frame + a leakage check test.

**Phase 2 — Baselines**
Implement the three baselines in §7 and the walk-forward harness. This is the bar
every model must clear.

**Phase 3 — Component models**
Minutes first, then attack (xG/xA-driven), defence (CS/DEFCON/saves), bonus.
Combine via the scoring rules into expected points.

**Phase 4 — Enrichment**
Add Understat xG, odds-implied clean-sheet/goal expectancy, team Elo, and the
injury/lineup feed. Re-run the backtest and measure lift over Phase 3.

**Phase 5 — Optimiser (optional)**
Integer linear program (e.g. `pulp`) that maximises expected points subject to
budget, positional quota, 3-per-club, and formation constraints — extendable to
multi-week planning and transfer/hit decisions.

## 9. Definition of done

- One command pulls fresh data and outputs `expected_points` for the upcoming
  gameweek for every player.
- Walk-forward backtest beats all three baselines on MAE and within-position rank
  correlation.
- No leakage: an automated test confirms no feature uses same-or-future-gameweek
  information.
- README documents data sources, refresh cadence, and how to run a backtest.

## 10. Start here (first task for Claude Code)

Scaffold the repo in §5, implement `src/data/fpl_api.py` with a cached client for
`bootstrap-static/`, `fixtures/`, and `element-summary/{id}/`, and produce a
`player_gameweek` parquet for the current season with lagged rolling-form features
and a passing leakage test. Then implement the Phase 2 baselines and the
walk-forward evaluation harness.
