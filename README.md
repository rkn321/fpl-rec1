# FPL Prediction Model

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.2%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Season](https://img.shields.io/badge/season-2026%2F27-00694E)](https://fantasy.premierleague.com/)
[![Data](https://img.shields.io/badge/data-official%20FPL%20API-37003C)](https://fantasy.premierleague.com/api/bootstrap-static/)
[![Repo](https://img.shields.io/badge/github-rkn321%2Ffpl--rec1-181717?logo=github&logoColor=white)](https://github.com/rkn321/fpl-rec1)

Predicts expected FPL points per player per gameweek for the 2026/27 season, and
turns that into ranked transfer advice for a real squad.

**Repository:** https://github.com/rkn321/fpl-rec1

Built against [`fpl-model-spec.md`](fpl-model-spec.md). **Phases 1–3 are
complete**: the data foundation, a leakage-safe feature frame, the baselines and
walk-forward harness, and gradient-boosted component models for minutes, attack,
defence and bonus, combined through the scoring rules. Phase 4 (Understat xG,
odds, the injury and lineup feed) and Phase 5 (the optimiser) are scaffolded with
design notes and not yet implemented.

The component model beats every baseline on MAE and beats the naive baselines on
ranking, but is still behind FPL's own expected points at ordering players — see
[what this does and does not clear](#what-this-does-and-does-not-clear).

On top of that sits a self-contained transfer tool — see
[Frontend](#frontend) — which scores every legal swap by its effect on your
starting XI and tells you when the best move on the board is not worth the
4-point hit.

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
git clone https://github.com/rkn321/fpl-rec1.git
```

```powershell
cd fpl-rec1
```

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

Your squad, bank and armbands live in **`config.local.yaml`**, which is
gitignored and deep-merged over `config.yaml` at load time — so personal state
stays out of the repository and a fresh clone opens an empty pitch rather than
someone else's team. Copy the example to start:

```powershell
copy config.local.yaml.example config.local.yaml
```

With that in place the weekly command takes no arguments. `fpl.cmd` wraps the virtualenv, and `--open`
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
`config.local.yaml`, which only ever seeds a *first* visit. To push the other way —
a fresh browser, cleared storage, a second machine — hit **Copy squad** on the
page and paste the result into `squad.players`, which accepts that
comma-separated form directly.

The built page in the repo is generated with `--no-local`, so it ships without
a squad baked in. Rebuilding normally puts *your* team in it, which will show as
a modified file — regenerate with `fpl export-frontend --no-local` before
committing if you would rather it stayed neutral.

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
| `--no-local` | ignore `config.local.yaml` — how the committed page is built |

With no squad configured and no `--squad`, the page opens empty and you pick a
squad by hand.

## What is here

```
config.yaml               all settings: seasons, paths, API TTLs, windows
config.local.yaml         your squad and bank (gitignored; see .example)
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
    base.py               shared LightGBM wrapper, recency weights, Poisson tails
    minutes.py            p(play), p(60+), expected minutes
    attack.py             goals and assists as per-90 rates (Poisson)
    defence.py            team clean sheet, goals conceded, saves, DEFCON
    bonus.py              BPS, ranked within fixture -> expected bonus
    combine.py            components -> expected points via the scoring rules
    component.py          the assembled model, wired into the harness
  optimise/squad.py       Phase 5 — stub
fpl.cmd                   CLI wrapper — .\fpl <command>
frontend/template.html    the page source; squad-picker.html is generated
tests/                    72 tests; leakage checks on synthetic and real data
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

## What the transfer advice is calibrated on

Two constants in the page drive every suggestion, and both were originally
picked by hand. Both have since been measured against 2025-26.

**Fixture difficulty.** Comparing every starter's points in a match against
their own season average — within a player, so it is not just "good players get
easy fixtures":

| FDR | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| measured | 1.27 | 1.18 | 1.00 | 0.88 | 0.64 |

A difficulty-5 fixture costs a starter about a third of their normal return.
This is the single largest swing in the suggestions, and it is real.

**How far to shrink current form toward a prior.** This one was wrong. The
weight was set at 4 gameweeks on the reasoning that two games is too little to
judge anyone on. Replaying the season and ranking players by a blend of
season-to-date form and last season's record says the opposite — every increase
in prior weight made the ordering worse:

| prior weight | Spearman (predicting GW2–5) |
|---|---|
| form only | **0.574** |
| 1 gameweek | 0.554 |
| 4 gameweeks | 0.477 |
| prior only | 0.245 |

Recent form carries what last season cannot: whether a player is first choice
right now, and whether the team plays through them. The weight is now 1 — nearly
all of the ranking accuracy, while still damping magnitudes so a single 15-point
haul is not projected across a horizon as though it were the norm.

## Backtest results

Walk-forward over **2024-25**: train on gameweeks `< t`, predict `t`, scored per
player-gameweek (double gameweeks summed).

2024-25 rather than 2025-26 on purpose. The historical `xP` scrape is patchy —
27 of 38 gameweeks in 2025-26 have it zero for *every* player — so a backtest
there scores `fpl_ep` on the handful of gameweeks that survive while every other
model is scored on all of them. `ranked_gws` now travels with each row so that
mismatch is visible rather than silent.

**Players with recent minutes** — the honest view, since ~55% of the pool never
plays and their guaranteed zeros flatter every metric:

| model | ranked GWs | MAE | RMSE | Spearman (overall) | Spearman (within position) |
|---|---|---|---|---|---|
| **`component`** | 34 | **1.295** | **2.034** | **0.690** | **0.695** |
| `fpl_ep` | 31 | 1.513 | 2.302 | 0.676 | 0.665 |
| `season_mean` | 34 | 1.844 | 2.751 | 0.402 | 0.387 |
| `minutes_x_pp90` | 34 | 1.914 | 2.774 | 0.419 | 0.404 |
| `last3_mean` | 34 | 2.003 | 2.978 | 0.387 | 0.375 |

**All players:**

| model | ranked GWs | MAE | RMSE | Spearman (overall) | Spearman (within position) |
|---|---|---|---|---|---|
| **`component`** | 34 | **0.707** | **1.474** | **0.775** | **0.770** |
| `fpl_ep` | 31 | 0.882 | 1.690 | 0.758 | 0.751 |
| `minutes_x_pp90` | 34 | 1.016 | 2.013 | 0.716 | 0.709 |
| `last3_mean` | 34 | 1.061 | 2.155 | 0.703 | 0.697 |
| `season_mean` | 34 | 1.059 | 2.020 | 0.664 | 0.656 |

### What this clears, and the asterisk on it

The brief's definition of done asks the model to beat all three baselines on MAE
**and** within-position rank correlation. It now does, on both views.

But `xP` — FPL's own expected points — is both baseline (b) *and* one of the
model's features, so "beats baseline (b)" needs stating precisely. The claim is
not "a model built from scratch beats FPL's". It is:

> FPL's published forecast, corrected by a model trained on lagged form,
> fixtures and team strength, beats that forecast used raw.

That is still a real result — the corrections add information rather than
reproducing what was already there — and the size of it is the honest measure of
what the model contributes:

| | MAE | Spearman (within position) |
|---|---|---|
| `fpl_ep` alone | 1.513 | 0.665 |
| component **without** `xP` | 1.718 | 0.468 |
| component **with** `xP` | **1.295** | **0.695** |

The middle row is the model on its own inputs, and it loses to FPL. That gap was
never a modelling failure: FPL's figure is computed with **team news** — press
conferences, predicted lineups, injury flags — and nothing in the lagged feature
frame reconstructs a team sheet. Minutes are the make-or-break input (brief
§6.3) and the lineup feed is the hardest thing to source (§3.6).

### The residual doubt

The brief warns that `xP` may be scraped post-gameweek (§6.1), which would make
this circular. Checked rather than assumed, on 2024-25 where the scrape is 91%
complete: players who started the previous three matches and then did not play
average **1.40** expected points against **3.55** for those who did. Reduced,
not zeroed — which is what FPL's *pre-deadline* `chance_of_playing` flags look
like, and not what a column computed after the whistle would look like.

That is evidence, not proof. `xP` correlates 0.67 with same-gameweek minutes,
and some of that could be knowledge rather than forecasting. The way to settle
it is to snapshot `bootstrap-static`'s `ep_next` before each deadline from here
on, building a training set that is provably pre-deadline, and re-run this
comparison against it.

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
