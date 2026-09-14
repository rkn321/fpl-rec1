# FPL Prediction Model

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.2%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Season](https://img.shields.io/badge/season-2026%2F27-00694E)](https://fantasy.premierleague.com/)
[![Data](https://img.shields.io/badge/data-official%20FPL%20API-37003C)](https://fantasy.premierleague.com/api/bootstrap-static/)
[![Repo](https://img.shields.io/badge/github-rkn321%2Ffpl--rec1-181717?logo=github&logoColor=white)](https://github.com/rkn321/fpl-rec1)

Predicts expected FPL points per player per gameweek for the 2026/27 season, and
turns that into ranked transfer advice for a real squad.

> **Just want to run it?** Skip everything else and follow the
> **[Quick start](#quick-start)** — four steps, about five minutes.

**Repository:** https://github.com/rkn321/fpl-rec1

Built against [`fpl-model-spec.md`](fpl-model-spec.md). **Phases 1–3 are
complete**: the data foundation, a leakage-safe feature frame, the baselines and
walk-forward harness, and gradient-boosted component models for minutes, attack,
defence and bonus, combined through the scoring rules. Phase 4 (Understat xG,
odds, the injury and lineup feed) and Phase 5 (the optimiser) are scaffolded with
design notes and not yet implemented.

The component model beats every baseline, including FPL's own expected points,
on both MAE and within-position ranking — with an asterisk, since FPL's figure
is also one of its inputs. See [what this clears](#what-this-clears-and-the-asterisk-on-it).

On top of that sits a self-contained transfer tool — see
[Frontend](#frontend) — which scores every legal swap by its effect on your
starting XI and tells you when the best move on the board is not worth the
4-point hit.

## Quick start

This is everything, start to finish, on a **Windows** laptop. It takes about
five minutes, and you only do Steps 1–3 once.

All the commands go in a **PowerShell terminal**. The easiest way to get one is
to open **VS Code** and press **Ctrl + `** (the key above Tab); or search
Windows for "PowerShell". Type each command exactly as shown and press Enter.

### Step 1 — Check you have Python and Git

```powershell
python --version
```

```powershell
git --version
```

- If the first prints **`Python 3.11`** or higher **and** the second prints a
  Git version, you are set — go to **Step 2**.
- If either says *"not recognized"* (or typing `python` opens the Microsoft
  Store), install what is missing, then **close and reopen the terminal**:
  - **Python** — https://www.python.org/downloads/windows/ — on the first
    installer screen tick **"Add python.exe to PATH"** before clicking Install.
  - **Git** — https://git-scm.com/download/win — the defaults are fine.

### Step 2 — Download the project

```powershell
git clone https://github.com/rkn321/fpl-rec1.git
```

```powershell
cd fpl-rec1
```

You are now inside the project folder. Every command from here on is run from
inside it.

### Step 3 — Install what it needs (once)

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The second command downloads the libraries and takes a minute or two. A lot of
text scrolling past is normal.

### Step 4 — Run it

```powershell
.\fpl serve
```

Wait until it prints `serving   : http://127.0.0.1:8765/...` — your browser then
opens the page by itself. The first run takes a little while, because it
downloads the latest player data and builds the predictions.

**The first time**, the page shows an empty pitch and a box saying **"First
time here?"**. Pick your 15 players from the list on the right, type your
**bank** and **free transfers** into the box, and click **Save team**. That is
it — your team is saved on your laptop, in a file called `config.local.yaml`
that only you have.

### Every time after

Before each deadline, run **Step 4 again** — just that one command:

```powershell
.\fpl serve
```

The page opens with your team already in it, plus the latest prices and
fixtures. Leave the terminal open while you use the page, and press **Ctrl+C**
in it when you are done.

### If something goes wrong

| What you see | What to do |
|---|---|
| `python` is not recognized | Install Python (Step 1), tick **"Add python.exe to PATH"**, then close and reopen the terminal. |
| `.\fpl` is not recognized, or "cannot be loaded" | Check you are inside the `fpl-rec1` folder (`cd fpl-rec1`). If PowerShell still refuses, run `.venv\Scripts\python.exe -m src.cli serve` instead — it does exactly the same thing. |
| The page says the deadline has passed and asks to be rebuilt | Press **Ctrl+C** in the terminal and run `.\fpl serve` again. |
| The browser did not open by itself | Open it yourself and go to `http://127.0.0.1:8765/squad-picker.html`. |

## Running it

**Nothing to host, and nothing to keep running between deadlines.** The
architecture is a batch backend and a static frontend:

- **Backend** — a Python CLI. It runs on demand, pulls from the FPL API, and
  writes files: a parquet feature frame, backtest CSVs, and the HTML page.
- **Frontend** — one self-contained HTML file with the player pool baked into
  it. All the squad and transfer logic runs in the browser. `fpl serve` keeps a
  small local helper up only while the page is open, so that **Save team** can
  write your `config.local.yaml` — a page opened as a plain file cannot write to
  disk on its own.

The consequence worth remembering: because the data is baked in at build time,
the page does not update itself. Re-run `fpl serve` before each deadline.

### Setup

The install and run commands live in one place — the **[Quick start](#quick-start)**
above — so they cannot drift out of step. The one thing worth knowing behind
them: `fpl.cmd` in the repo root runs the CLI through the project's virtualenv,
so nothing ever needs activating and `.\fpl <command>` is all you type. If
PowerShell refuses to run it (a locked-down execution policy on some machines),
`.venv\Scripts\python.exe -m src.cli <command>` is the identical long form.

### Backend

**You do not need any of these to use the page** — `fpl serve` runs everything
it needs by itself. They are for working on the model.

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

### Capturing training data (optional)

Not needed to use the page. This is for improving next season's model:

```powershell
.\fpl snapshot
```

Stores the live player data — FPL's expected points, availability flags,
set-piece orders — labelled by gameweek under `data/snapshots/`. These inputs
are only ever *current* in the API; capturing them before each deadline is what
makes them trainable next season without the "was this scraped after the match"
doubt. A Friday-afternoon scheduled task is the natural home for it.

### Frontend

Your squad, bank and armbands live in **`config.local.yaml`**, which is
gitignored and deep-merged over `config.yaml` at load time — so personal state
stays out of the repository and a fresh clone opens an empty pitch rather than
someone else's team. You never need to edit it by hand: the page writes it.

The weekly command builds the page and serves it locally:

```powershell
.\fpl serve
```

Served this way the page gets a **Save team** button. Whether
`config.local.yaml` exists is how it tells a first visit from a returning one:
absent, it shows a **"First time here?"** strip asking for a team, a bank and a
free-transfer count; present, those come pre-filled and **Save team** simply
updates them after you make transfers. A browser page cannot write to disk on
its own, which is what the small helper behind `serve` is for — it listens on
`127.0.0.1` only and stops when you close the terminal (Ctrl+C).

That is the whole ritual before a deadline. It runs the pipeline itself, so
`build-features` is not a prerequisite. If you would rather just open the file
with no helper running, `.\fpl export-frontend --open` still does that — the
page works the same, minus saving to the yaml (use **Copy squad** and paste
instead).

**The gameweek follows the real FPL calendar, not a button.** Every remaining
deadline is baked into the page, so it works out which gameweek it is from the
clock — cross a deadline and it rolls over on its own, granting the free
transfer and reverting a Free Hit, with no rebuild and no network. A page left
open across a Friday evening notices within the minute.

Saving your team settles the transfers and books any hit; it does not move time
forward. What a rebuild *is* still needed for is the data: prices, projections
and fixtures are stamped at build time, so once a deadline passes the page says
so and asks to be rebuilt.

**A rebuild picks up whatever is in `config.local.yaml`.** The page keeps your
working squad in browser storage between visits, and a rebuild whose team, bank
or free transfers differ from what the browser last saw adopts the new ones —
so after **Save team**, or after editing the file by hand, the next `fpl serve`
shows that team, while your chip and hit history is kept. Without the helper,
**Copy squad** puts the 15 names on the clipboard in the comma-separated form
`squad.players` accepts, for pasting into the file yourself.

The built page in the repo is generated with `--no-local`, so it ships without
a squad baked in. Rebuilding normally puts *your* team in it, which will show as
a modified file — regenerate with `fpl export-frontend --no-local` before
committing if you would rather it stayed neutral.

`serve` takes `--port` (default 8765), `--no-open`, and `--no-build` to serve
the page as last built without rebuilding. Every setting of `export-frontend`
can still be overridden per run:

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
  cli.py                  build-features / backtest / predict / export-frontend / serve
  serve.py                local helper behind `fpl serve` — lets the page write config.local.yaml
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

### Team news, applied rather than learned

Two of the inputs the roadmap ranked highest have no per-gameweek history —
they exist only as end-of-season snapshots or not at all — so they cannot be
trained on. They are used anyway, in the two ways that are honest about that:

**Set-piece duty** (`penalties_order`, `corners_and_indirect_freekicks_order`)
is stable enough within a season that the season snapshot is a fair training
proxy, so it is a feature. First-choice penalty takers scored 0.359 goals per 90
in 2025-26 against 0.105 for everyone else. And yet: **on the 2024-25 backtest
it moved nothing** — MAE 1.2945 to 1.2967, Spearman 0.6946 to 0.6936, inside
noise. The likely reason is that `xP` already encodes penalty duty, because FPL
knows who takes them. Kept as a hedge for the gameweeks where `xP` is missing;
not claimed as a gain.

**Availability** (`chance_of_playing_next_round`) has no history at all, so it
is applied as a *ceiling* on the minutes prediction for the gameweek being
predicted rather than fed to the model. A player FPL puts at 25% cannot be given
a 90% chance of playing by a model that only knows he started last month. On the
GW5 predictions this changed 139 of 258 flagged players; Dean Henderson went
from 52% (recent minutes) to 0% (injured), and from 1.9 expected points to 0.

`fpl snapshot` stores the live player data before a deadline, so that next
season these inputs can be trained on properly rather than approximated.

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
