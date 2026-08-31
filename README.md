# FPL Prediction Model

Predicts expected FPL points per player per gameweek for the 2026/27 season.

Built against [`fpl-model-spec.md`](fpl-model-spec.md). **Phases 1 and 2 are
complete**: the data foundation, a leakage-safe feature frame, the three
baselines, and the walk-forward evaluation harness. Phases 3–5 (component
models, enrichment, optimiser) are scaffolded with design notes and not yet
implemented.

## Running it

**There is no server, and nothing to keep running.** The architecture is a
batch backend and a static frontend:

- **Backend** — a Python CLI. It runs on demand, pulls from the FPL API, and
  writes files: a parquet feature frame, backtest CSVs, and the HTML page.
- **Frontend** — one self-contained HTML file with the player pool baked into
  it. All the squad and transfer logic runs in the browser. You open the file;
  there is no port, no `npm`, and no build step beyond the command that writes it.

The consequence worth remembering: because the data is baked in at build time,
the page does not update itself. Re-run the export before each deadline.

### One-time setup

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`fpl.cmd` in the repo root runs the CLI through that virtualenv, so nothing
needs activating: `.\fpl <command>`.

### Backend

```powershell
.\fpl build-features
```

Pulls the historical seasons and the live season, builds the feature frame, and
writes `data/processed/player_gameweek.parquet`. Everything downloaded is cached
under `data/cache/`, so later runs are fast.

```powershell
.\fpl backtest
```

```powershell
.\fpl predict
```

`predict` writes `data/processed/expected_points_gw{N}.csv` and prints the top of
the list.

### Frontend

Your squad, bank and horizon live in the `squad:` block of `config.yaml`, so the
weekly command takes no arguments. `fpl.cmd` wraps the virtualenv, and `--open`
launches the page when it is built:

```powershell
.\fpl export-frontend --open
```

That is the whole ritual before a deadline. It runs the pipeline itself, so
`build-features` is not a prerequisite.

**The gameweek follows the real FPL calendar, not a button.** Every remaining
deadline is baked into the page, so it works out which gameweek it is from the
clock — cross a deadline and it rolls over on its own, granting the free
transfer and reverting a Free Hit, with no rebuild and no network. A page left
open across a Friday evening notices within the minute.

Saving your team settles the transfers and books any hit; it does not move time
forward. What a rebuild *is* still needed for is the data: prices, projections
and fixtures are stamped at build time, so once a deadline passes the page says
so and asks to be rebuilt.

**Transfers you apply on the page are not overwritten by a rebuild.** The page
keeps your squad in browser storage and prefers it over the list in
`config.yaml`, which only ever seeds a *first* visit. To push the other way —
a fresh browser, cleared storage, a second machine — hit **Copy squad** on the
page and paste the result into `squad.players`, which accepts that
comma-separated form directly.

Every setting can still be overridden per run:

| Flag | Does |
|---|---|
| `--squad "name, name, ..."` | override `squad.players`; names are fuzzy-matched and disambiguated by squad shape |
| `--squad-file players.txt` | the same, one name per line |
| `--bank 1.5` | override `squad.bank`, in millions. **Cannot be derived** — it depends on what you paid, not on today's prices |
| `--horizon 5` | override `squad.horizon` — gameweeks of fixtures to load |
| `--gw 4` | target a specific gameweek instead of the next one |
| `--model` | which predictor supplies the model xPts column |
| `--open` | open the page in your browser once built |

With no squad configured and no `--squad`, the page opens empty and you pick a
squad by hand.

## What is here

```
config.yaml               all settings: seasons, paths, API TTLs, windows
src/
  config.py               config loading
  cli.py                  build-features / backtest / predict
  pipeline.py             assemble + store the feature frame
  metrics.py              MAE, RMSE, Spearman (overall and within position)
  evaluate.py             walk-forward backtest harness
  data/
    schema.py             the canonical player_gameweek schema
    fpl_api.py            official API client — disk cache, retry/backoff
    historical.py         vaastav season CSVs
    current.py            the in-progress season, live
    understat.py          Phase 4 — stub with design notes
    odds.py               Phase 4 — stub
    injuries.py           Phase 4 — stub
  features/
    rolling.py            lagged rolling helpers
    build.py              feature assembly
  models/
    scoring.py            2026/27 scoring rules (implemented + verified)
    baselines.py          the three baselines
    minutes.py            Phase 3 — stub with design notes
    attack.py             Phase 3 — stub
    defence.py            Phase 3 — stub
    bonus.py              Phase 3 — stub
    combine.py            Phase 3 — stub
  optimise/squad.py       Phase 5 — stub
fpl.cmd                   CLI wrapper — .\fpl <command>
frontend/template.html    the page source; squad-picker.html is generated
tests/                    65 tests; leakage checks on synthetic and real data
data/                     parquet + API cache (gitignored)
```

## Data sources and refresh cadence

| Source | Used for | Refresh |
|---|---|---|
| [FPL API](https://fantasy.premierleague.com/api/) `bootstrap-static/` | players, teams, gameweeks, prices | hourly (cached) |
| FPL API `fixtures/` | fixture list + difficulty ratings | hourly |
| FPL API `element-summary/{id}/` | current-season per-gameweek history | hourly |
| [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) | historical seasons, with Understat xG merged | once per season (immutable) |

Everything is cached to `data/cache/`. TTLs are per endpoint in `config.yaml`;
the client retries with exponential backoff and jitter, honours `Retry-After`,
and falls back to a stale cache entry rather than failing when the API is down
(it does 503 around deadlines).

To point at `olbauday/FPL-Core-Insights` instead, change
`historical.vaastav_base` in `config.yaml` — the layouts match.

## The leakage guarantee

The rule: **a feature may only use information available before that gameweek's
deadline.** Two automated tests enforce it as a property of the feature builder,
so it keeps holding as features are added:

1. `test_no_future_leakage` — features for gameweek `t` are identical whether or
   not gameweeks after `t` exist in the input.
2. `test_no_same_gameweek_leakage` — features for gameweek `t` are identical when
   every post-match outcome from `t` onward is replaced with random noise. This
   is the strong one: it catches a feature reading its own row's result.

Both run on synthetic data (fast, offline) and again on a real cached season.

**Lagging is done at the gameweek boundary, not by row.** This matters in double
gameweeks. Lagging by one row would let a player's second fixture see the first
fixture's result — but you pick your team once, before the deadline, so nothing
from that gameweek is knowable. Both legs of a double therefore carry identical
form features, and `test_real_doubles_share_form_features` checks that on real
doubles. Fixture context (venue, opponent, difficulty, rest days) does differ
between legs, because the schedule is published in advance.

Other gotchas from §6 that are handled:

- **Row grain is `(season, player_id, fixture_id)`**, never `(player, gameweek)` —
  otherwise blanks and doubles silently collapse.
- **Joins are on FPL element ids**, never names. The team a player lined up for is
  derived from the fixture, so mid-season transfers stay correct.
- **Cold start reads as null, not zero.** A promoted club's players get NaN form,
  which tells a model "unknown" rather than "known to be bad".
- **`xP` is never a feature.** It is held aside purely as baseline (b).
- **Provisional scores are not labels.** The current-season loader drops
  gameweeks the API has not marked finished, since bonus and DEFCON still move
  until 09:00 UK the day after the last match.

## Scoring rules

`src/models/scoring.py` is the single copy of the 2026/27 rules. It reconstructs
`total_points` **exactly for all 11,498 played player-fixtures of 2025-26** from
the component stats — which also confirms the DEFCON reading: the API's
`defensive_contribution` is the raw CBIT/CBIRT count, the thresholds are 10 for
defenders and 12 for everyone else, and the award caps at +2 however far past the
threshold a player goes.

## Backtest results

Walk-forward over 2025-26: train on gameweeks `< t`, predict `t`, scored per
player-gameweek (double gameweeks summed).

**All players** (~780 rows per gameweek):

| model | MAE | RMSE | Spearman (overall) | Spearman (within position) |
|---|---|---|---|---|
| `minutes_x_pp90` | **1.003** | 2.034 | 0.745 | 0.746 |
| `last3_mean` | 1.043 | 2.165 | 0.735 | 0.735 |
| `season_mean` | 1.056 | 2.040 | 0.692 | 0.689 |
| `fpl_ep` | 1.092 | 2.425 | **0.787** | **0.784** |

**Players with recent minutes** (~357 per gameweek):

| model | MAE | RMSE | Spearman (overall) | Spearman (within position) |
|---|---|---|---|---|
| `season_mean` | **2.014** | 2.894 | 0.362 | 0.359 |
| `minutes_x_pp90` | 2.076 | 2.923 | 0.402 | 0.400 |
| `last3_mean` | 2.161 | 3.123 | 0.365 | 0.365 |
| `fpl_ep` | 2.248 | 3.510 | **0.689** | **0.687** |

Read these as the bar Phase 3 has to clear, and note the split verdict: no single
baseline wins both. FPL's own `ep` has the *worst* MAE and by far the *best*
ranking, which is the more useful property — you pick players by ordering them,
not by their absolute predicted score. The honest target for a component model is
to beat `minutes_x_pp90` on MAE **and** `fpl_ep` on within-position Spearman.

Two caveats on those numbers:

- The all-players view is flattered by the ~55% of the pool who do not play. Their
  guaranteed zeros are easy and inflate both MAE and rank correlation. The
  likely-playing split is the more honest read.
- `fpl_ep` uses the vaastav `xP` column, which is scraped around the gameweek
  rather than strictly at the deadline (§6.1), so it may be a slightly generous
  bar. It is used only as a baseline and never as a feature.

## Running the tests

```bash
python -m pytest
```

Tests that need real data skip cleanly when `data/cache/` is cold, so the suite
runs offline. Populate the cache with `python -m src.cli build-features` to
enable them.

## Current state and next step

The feature frame covers 2024-25, 2025-26 and the current season: 57,650
player-fixtures and 189 features. DEFCON columns exist only from 2025-26 and are
NaN before that, which is carried explicitly rather than filled.

Next, per §8 Phase 3: **build the minutes model first**. Every other component
gets multiplied by playing time, so it dominates the error budget. The features
it needs (`start_rate_todate`, `minutes_r3/r5/r10`, `minutes_lag1/lag2`) are
already in the frame; `chance_of_playing_next_round` from `bootstrap-static`
should be added alongside them.

One caveat on predicting right now: the 2026/27 season is one gameweek old, so
`points_per_90_todate` is computed from a single match and `predict` output is
close to "whoever scored well in GW1". That is the baselines behaving correctly
on one gameweek of evidence, not a bug — it is also exactly the cold-start
problem Phase 4's priors (odds, team strength) are meant to solve.
