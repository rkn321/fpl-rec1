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
defence and bonus, combined through the scoring rules. Phase 4 is partly
built — betting odds are in ([below](#betting-odds)) — while Understat xG, the
injury and lineup feed and Phase 5 (the optimiser) are scaffolded with design
notes and not yet implemented.

The component model beats every simple baseline on MAE and within-position
ranking, on both 2024-25 and 2025-26. It does **not** use FPL's own expected
points: the historical record of that figure turned out to be captured after
the matches, and an earlier version that trained on it backtested far better
than it could ever play. See [the leak](#the-leak-in-fpls-historical-expected-points)
and [what this clears](#what-this-clears).

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

### Already downloaded it before? Get the latest version

If you cloned the project earlier, pull the newest version first, then run it.
Do the same any time you are told there is an update:

```powershell
cd fpl-rec1
```

```powershell
git pull
```

```powershell
.\fpl serve
```

Your saved team is not affected — it lives in `config.local.yaml`, which
`git pull` never touches.

If you never ran the install step the first time round, run these two once,
after `git pull` and before `.\fpl serve`:

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

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
.\fpl backtest-fixtures
```

Scores the page's fixture weighting — Elo against FPL's difficulty rating —
from the week the odds cover to four weeks past it. See
[fixtures beyond next week](#fixtures-beyond-next-week-elo-not-fpls-difficulty).

```powershell
.\fpl predict
```

`predict` prints the top of the list. It, the page and `serve` all predict
through one function (`src/predict.py`), and each stores what it predicted, with
every scoring term, as `data/processed/predictions_gw{N}.csv` — but only while
the deadline is still ahead. A rerun before the deadline overwrites (later
runs have later team news); a run after it never does, so the file is always
what the model said in time to act on. That file is what `fpl review` scores.

The fit itself is cached under `data/cache/models/`, keyed on the training rows,
features, model code and LightGBM version, so only the first build after a
gameweek finishes (or after the model changes) pays for fitting — a rebuild of
the page otherwise skips the 45-second fit.

### Capturing training data (recommended)

Not needed to use the page, but it is what lets this season's results train
the model properly:

```powershell
.\fpl snapshot
```

Stores the live player data — FPL's expected points, availability flags,
set-piece orders, prices — labelled by gameweek under `data/snapshots/`. These
inputs are only ever *current* in the API and are overwritten at every
deadline, so once a gameweek has passed there is no other record of them as
they stood. Once a gameweek has a snapshot, its played rows carry the set-piece
orders as they stood at that deadline. FPL's `ep_next` captured this way is the
only expected-points record that is pre-deadline by construction — the
historical one is not ([the leak](#the-leak-in-fpls-historical-expected-points))
— so it is what `fpl review` scores the model against, and a season of it is
what a future model could fairly learn from. The command refuses to write a
snapshot after its gameweek's deadline, and the loader ignores one, so nothing
in these rows can have been influenced by the matches.

It also keeps FPL's fixture difficulty ratings as they stood. FPL revises those
during the season and rewrites them for matches already played, so a gameweek
rebuilt later sees every fixture rated with hindsight: rebuilding GW5 of
2026-27 two weeks on moved predictions by up to 1.8 points from what the page
had shown, and Emersonn from 6.14 to 5.80. With the deadline's ratings and
set-piece orders put back, the rebuild matches the page to within 0.05 for 656
of 659 players. That is what `fpl review` relies on for a gameweek with no
stored prediction.

It labels the capture by the *upcoming* gameweek and overwrites, so the
simplest schedule is also the right one: run it regularly, and the last run
before each deadline is automatically the freshest capture.

**In the cloud.** A missed deadline is a gameweek whose pre-deadline state —
roles, ratings, FPL's own figure — is gone for good, so capturing cannot depend
on a laptop being awake.
`.github/workflows/snapshot.yml` runs `fpl snapshot` on GitHub Actions every
six hours and keeps each capture as a private workflow artifact named for its
gameweek. Artifacts expire after 90 days, so they are a way home, not a place
to keep things: `fpl fetch-snapshots` downloads any cloud capture newer than
the one in `data/snapshots/`, and `snapshot`, `serve`, `predict`,
`export-frontend` and `review` all do the same quietly before they start.
Running the project at least once every three months is enough to lose
nothing. Downloading artifacts needs a signed-in GitHub CLI, even for a public
repository — once:

```powershell
gh auth login
```

The schedule runs from the default branch only, and GitHub pauses schedules
in a public repository after 60 days without a commit; it emails first, and
the Actions tab re-enables it.

**On this machine, too.** Optional now, but a second source costs nothing, and
it is what keeps fetching while the project is idle. This registers a Windows
scheduled task that runs `fpl snapshot` every six hours — capturing, then
fetching whatever the cloud has that is newer — catching up on wake if the
laptop was asleep, with output in `data\snapshots\snapshot.log` (paste into
PowerShell from inside the project folder):

```powershell
$repo = (Get-Location).Path; $a = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c call fpl.cmd snapshot >> data\snapshots\snapshot.log 2>&1" -WorkingDirectory $repo; $t = New-ScheduledTaskTrigger -Daily -At "00:00"; $t.Repetition = (New-ScheduledTaskTrigger -Once -At "00:00" -RepetitionInterval (New-TimeSpan -Hours 6) -RepetitionDuration (New-TimeSpan -Hours 23 -Minutes 59)).Repetition; $s = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew; Register-ScheduledTask -TaskName "FPL snapshot" -Action $a -Trigger $t -Settings $s -Force | Out-Null; Start-ScheduledTask -TaskName "FPL snapshot"; "registered and run once"
```

To remove it: `Unregister-ScheduledTask -TaskName "FPL snapshot" -Confirm:$false`.

### After a gameweek

```powershell
.\fpl review
```

How the last gameweek to kick off went, against what the model said before
its deadline: your squad line by line (expected against actual, auto-subs and
the armband applied by FPL's rules), the biggest over- and under-deliveries
across the pool, the model against FPL's own figure on the same players, and
every scoring term's predicted total against its actual one.

The two summaries run season to date. Each review appends every player's
predicted and actual terms to `data/processed/review_log.csv` (reviewing a
gameweek again replaces its rows), and those totals — not one Saturday — are
what a claim like "the bonus term overweights small samples" should be tested
against.

It scores the stored prediction where there is one. Where there is not (any
gameweek before stored predictions existed) it rebuilds the model as of that
deadline from the gameweek's snapshot, and says so; with no snapshot either, it
refuses rather than guess. Run it after FPL marks the gameweek final — the
morning after the last match — or it will warn that bonus can still move.

By default it reviews the 15 in `config.local.yaml` as the model would have
started them, which after a transfer is not the team you fielded. Set
`squad.team_id` (the number in your team's URL on the FPL site) and it scores
the real XI, bench order, armband and chip instead.

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

**Live points.** During a gameweek each card shows the points its player has
scored so far, once his match is under way — a running total for the XI sits
in the ledger, captain doubled. Players whose match has not kicked off show
nothing; a bench player whose match is over without him getting on shows a
dashed 0. The gameweek shown is the one most recently past its deadline: in
play over the weekend, final by the time the next deadline arrives. A browser
cannot ask FPL for this itself, so it comes through the `fpl serve` helper,
refreshed every minute — opened as a plain file the page simply omits it.

**Player sheets.** Click any player's name or xPts figure, in the pool or on
the pitch, for the figure taken apart: his chance of playing and of 60
minutes, each scoring term it is built from (largest first, adding up to the
total), his fixtures ahead, and — once gameweeks have been reviewed — what the
model and FPL each predicted for him against what he scored.

**Track record.** At the foot of the squad panel, the season so far as `fpl
review` scored it: the model's average miss, within-position ranking and
best-XI points against FPL's own figure, gameweek by gameweek, and every
scoring term's predicted total against its actual one. It is read from
`review_log.csv` at build time, so run `fpl review` after a gameweek and the
next `fpl serve` shows it.

**Availability, in words.** Cards and pool rows say who might not play: FPL's
own flag first — `75%` in amber for a doubt, `INJ`, `SUS` or `OUT` in red, with
the card itself tinted — and, for a player FPL calls fit, the minutes model's
view: `ROT` under a 60% chance of lasting an hour, `BENCH` under a 30% chance
of playing at all. The thresholds are `ROTATION_P60` and `BENCH_P_PLAY` in the
template.

**Fixtures ahead.** Each card shows this gameweek's fixture and, under it, a
strip of the four after it (home in capitals, away in lower case, ×2 for a
double, a dash for a blank). The **fixture ticker** under the chip advice lays
every club's next eight gameweeks side by side, ranked by how kind the next 3,
5 or 8 are, in average games: each fixture counts its Elo multiplier, so an
even game is 1.0, a double counts twice and a blank nothing. Clicking a club
filters the player pool to it.

**Fixture colours and weights are Elo, not FPL's difficulty.** Every fixture
carries its Elo gap — the two sides' ratings, home advantage included — and
that is what the projections, the transfer advice, the chip timing and the
ticker weight it by. The chips are coloured by an Elo tier on FPL's familiar
1–5 scale, cut so each tier turns up as often as FPL's own number does; hover
one for both ratings. Why: [fixtures beyond next
week](#fixtures-beyond-next-week-elo-not-fpls-difficulty). A page built
without the results to rate from falls back to FPL's difficulty.

**Chip timing.** Under the chip bar the page says when to play each chip, from
the same projections taken one gameweek at a time — a double gameweek is where
Triple Captain and Bench Boost pay, a blank is where a Free Hit does. Each chip
gets its best week, the number behind it, and the reason: Triple Captain is the
extra ×1 of your best starter; Bench Boost is what your bench would score; Free
Hit only fires for a week whose *shape* demands it (a blank that empties your
XI, or a widespread double), never for an ordinary week's paper gain; and the
Wildcard is valued against what one free transfer a week would reach anyway,
so its figure is what doing the whole rebuild *now* is worth, not the gap to a
perfect squad. Players with fewer than 90 minutes this season are kept out of
any "best squad" — no evidence, no place in a dream team. A set-one chip
unused near the GW19 deadline is flagged as use-it-or-lose-it.

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
  cli.py                  build-features / backtest / backtest-fixtures / predict / export-frontend / serve / snapshot / review
  predict.py              the one predict path; stores each gameweek's pre-deadline prediction
  review.py               `fpl review`: a played gameweek against that prediction, and the season log
  serve.py                local helper behind `fpl serve` — lets the page write config.local.yaml
  pipeline.py             assemble + store the feature frame
  metrics.py              MAE, RMSE, Spearman (overall and within position)
  evaluate.py             walk-forward backtest harness
  evaluate_fixtures.py    `fpl backtest-fixtures`: Elo against FPL difficulty, weeks ahead
  data/
    schema.py             the canonical player_gameweek schema
    fpl_api.py            official API client — disk cache, retry/backoff
    historical.py         vaastav season CSVs
    current.py            the in-progress season, live
    snapshots.py          pre-deadline captures, and the deadline state a rebuild replays
    cloud_snapshots.py    downloads the captures GitHub Actions takes
    understat.py          Phase 4 — stub with design notes
    odds.py               betting odds -> each side's expected goals
    elo.py                team Elo ratings from results, built the ClubElo way
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
tests/                    ~180 tests; leakage checks on synthetic and real data
.github/workflows/        snapshot.yml captures before every deadline; tests.yml runs the suite
data/                     parquet + API cache (gitignored)
```

## Data sources and refresh cadence

| Source | Used for | Refresh |
|---|---|---|
| [FPL API](https://fantasy.premierleague.com/api/) `bootstrap-static/` | players, teams, gameweeks, prices | hourly (cached) |
| FPL API `fixtures/` | fixture list + difficulty ratings | hourly |
| FPL API `element-summary/{id}/` | current-season per-gameweek history | hourly |
| [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) | historical seasons, with Understat xG merged | once per season (immutable) |
| `fpl snapshot` (`bootstrap-static/` + `fixtures/`) | pre-deadline `xP`, prices, availability, set-piece orders, difficulty ratings | every six hours, on GitHub Actions and optionally locally |
| FPL API `event/{gw}/live/` | actual points by scoring term, for `fpl review` | five minutes |
| [football-data.co.uk](https://www.football-data.co.uk/) | match odds (the model), and results since 2016-17 for the Elo ratings (the page) | six hours in season; finished seasons once |

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
- **`xP` is never a feature.** The historical column was recorded after the
  matches ([the leak](#the-leak-in-fpls-historical-expected-points)), and
  `tests/test_leakage.py` fails if it reaches a model.
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
easy fixtures" — over 2022-23 to 2025-26, by position:

| FDR | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| GK | 1.30 | 1.10 | 1.00 | 0.88 | 0.74 |
| DEF | 1.39 | 1.19 | 1.00 | 0.82 | 0.64 |
| MID | 1.27 | 1.08 | 1.00 | 0.87 | 0.75 |
| FWD | 1.31 | 1.09 | 1.00 | 0.83 | 0.73 |

This is the single largest swing in the suggestions, and it is real: ignoring
fixtures altogether costs 0.23 in squared error per start. It is also not the
same for everyone. A defender's return rides on clean sheets and swings from
1.39 to 0.64; a midfielder's runs 1.27 to 0.75. An easy fixture was worth more
to a defender than to a midfielder in each of the four seasons.

It began as one table for every position, measured on 2025-26 alone (1.27,
1.18, 1.00, 0.88, 0.64). Held out a season at a time, the position tables beat
it on all three seasons it was not measured on — by 0.027, 0.022 and 0.026 in
squared error per start, about a tenth of the whole fixture effect — and lost
by 0.004 only on 2025-26, which that table was fitted to. Each position's cell
is pulled toward the all-position figure by weight n / (n + 300) on its number
of starts; without that pull the difficulty-5 cells, a few hundred matches
each, made the position tables lose in 2023-24.

**Superseded, and overstated.** The ratings this table was measured on are the
ones FPL rewrote after the results were in, and the swing it found is mostly
that hindsight: rated before kickoff, a fixture is worth 0.03–0.06 in squared
error per start (Elo, and the market itself), not 0.23, and a defender's honest swing is about 0.75 to 1.26, not
0.64 to 1.39. The page now weights fixtures by Elo and keeps this table only as
a fallback — see [fixtures beyond next
week](#fixtures-beyond-next-week-elo-not-fpls-difficulty).

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

Walk-forward over **2024-25** and **2025-26**: train on gameweeks `< t`, predict `t`, scored per
player-gameweek (double gameweeks summed). "Train on gameweeks `< t`" means
prior seasons in full plus the current one up to `t` — the rule lives in one
place, `training_window`, and the page and `predict` fit on exactly it, so the
figures shown are the figures validated here. (They did not, until GW5 of
2026-27: a bare `gw < t` had been keeping only the first weeks of *every*
season, an eighth of the data.)

History is four seasons, 2022-23 to 2025-26 — every season FPL has published
xG for. Older rows count for less through the 40-gameweek recency half-life,
which is what lets a 2022-23 row help without a 2022-23 role being mistaken for
a current one. (The third and fourth seasons were each measured as a gain when
added, but with the leaked `xP` in the model; see below.)

**Players with recent minutes** — the honest view, since ~55% of the pool never
plays and their guaranteed zeros flatter every metric:

| model | 2024-25 MAE | 2024-25 Spearman within position | 2025-26 MAE | 2025-26 Spearman within position |
|---|---|---|---|---|
| **`component`** | **1.716** | **0.484** | **1.837** | **0.493** |
| `season_mean` | 1.844 | 0.387 | 2.014 | 0.359 |
| `minutes_x_pp90` | 1.914 | 0.404 | 2.076 | 0.400 |
| `last3_mean` | 2.003 | 0.375 | 2.161 | 0.365 |

**All players:**

| model | 2024-25 MAE | 2024-25 Spearman within position | 2025-26 MAE | 2025-26 Spearman within position |
|---|---|---|---|---|
| **`component`** | **0.949** | 0.707 | **0.922** | 0.726 |
| `minutes_x_pp90` | 1.016 | **0.709** | 1.003 | **0.746** |
| `last3_mean` | 1.061 | 0.697 | 1.043 | 0.735 |
| `season_mean` | 1.059 | 0.656 | 1.056 | 0.689 |

`fpl_ep` — FPL's own figure — is left out of both tables. On historical seasons
it knows the result, so it is not a bar anything honest can be measured
against; the harness still prints it, and it still "wins".

On the walk-forward that decided the odds (every other gameweek of 2024-25),
the model's top XI scored **5.9 real points a player** a gameweek, against 4.8
for the best baseline's. The figure this README used to quote, 9.5, was the
leak.

### What this clears

The brief's definition of done asks the model to beat all three baselines on MAE
**and** within-position rank correlation. On the players-with-recent-minutes
view it does, in both seasons and by a clear margin: ranking 0.48–0.49 against
0.36–0.40. On the all-players view it has the best MAE in both seasons but only
ties the minutes baseline on ranking in 2024-25 and trails it in 2025-26 (0.726
against 0.746). Ranking the whole pool is mostly ranking who plays, which is
exactly what that baseline measures and what the model, without team news, can
only infer from lagged minutes.

Whether it beats FPL's own *pre-deadline* figure is not yet known. There is one
clean gameweek of that figure (2026-27 GW5, from the snapshot); `fpl review`
keeps the running comparison from there on.

### The leak in FPL's historical expected points

The brief warned that the historical `xP` might be scraped after the gameweek
(§6.1). An earlier check — players who went on to miss a match still had
nonzero `xP` — showed it was not computed after the whistle *in full*, and it
was taken as settled. It was not. `fpl snapshot` produced the first
expected-points figure that is pre-deadline by construction (2026-27 GW5), and
against it three checks on the historical column, 2022-23 to 2025-26, agree:

| rank correlation with points, players with recent minutes | last week | the week forecast | the week after |
|---|---|---|---|
| snapshot `ep_next`, 2026-27 GW5 | **0.71** | 0.53 | — |
| historical `xP`, each season | 0.62–0.66 | **0.68–0.72** | 0.41–0.42 |

A forecast is built from the past, so it tracks the past better than the week
it predicts; the snapshot does. The historical column does the reverse in every
season and then falls away a week later — the signature of a number recorded
after the matches.

- Among regular starters on the same middling form (2–4 points a game), it gave
  those about to score 10+ **6.0–6.5** and those about to blank **2.3**, in
  every season. No forecast can know which is which.
- A top XI picked on it scored 7.5–8.5 real points a player; picked on the
  snapshot, 6.0.

The model had made `xP` its largest input, so it learned the result through
it. Backtests reported MAE 1.205, within-position Spearman 0.720 and a top XI
of 9.5–10.3 points a player — 85% of what perfect hindsight would have scored —
and the live model over-trusted FPL's real pre-deadline figure, which knows far
less. Every number in this README measured with `xP` in the model is affected;
the ones that matter have been re-measured above.

`xP` stays in the data — the page shows it, and `fpl review` scores the model
against the live and snapshot values, which are genuine — but it is not a
feature, and `tests/test_leakage.py` fails if it becomes one.

### Betting odds

The market's pre-match prices for every match since 2022-23, from
football-data.co.uk (`src/data/odds.py`): the odds collected on the Friday
afternoon for weekend games — about the FPL deadline — not the closing ones,
which carry later team news. Win/draw/loss and over/under 2.5 goals, margin
removed, become each side's expected goals, and every player row gets his
side's expected goals for and against. They average within 0.1 of actual goals
in every season.

Against the same model without them, on every other gameweek of 2024-25 and
GW2–6 of 2025-26: accuracy unchanged (MAE +0.001, ranking +0.002), and the top
XI **+0.26 real points a player** (95% interval +0.05 to +0.48), changing about
3 of the 11 picks a week. Adopted.

A gameweek's prices appear a few days before it, so a build earlier in the week
has none for the week it predicts. Rather than hand a model trained with odds a
blank, `predict` falls back to the same model fitted without them — the variant
measured above — and the page and `fpl predict` say so. Rebuild once the week's
odds are out.

### Fixtures beyond next week: Elo, not FPL's difficulty

The odds only reach the coming gameweek. Every week after it — the 2–5 week
horizon, chip timing, the ticker — was weighted by FPL's 1–5 difficulty rating
through the table above. The plan was to replace that with ClubElo's ratings,
but ClubElo's API went dark in October 2026 (`api.clubelo.com` answers 502 and
its fixtures endpoint calls itself "deactivated"), so `src/data/elo.py` rebuilds
them the ClubElo way from the football-data.co.uk results already downloaded
for the odds: expected score from the rating gap plus home advantage, a move of
K × √(goal margin) × surprise after each match, a light pull to the mean each
summer, promoted sides seated where the relegated ones finished. K = 12.5, home
advantage 45 and the pull were chosen on 2017-18 to 2021-22, before any season
used here; the match-by-match Brier score is 0.156 against 0.190 for always
guessing the average. A rating used for a deadline only knows matches played on
an earlier day.

`fpl backtest-fixtures` holds a player's usual rate fixed (his points per start
before the as-of deadline) and scores each fixture weighting on every start
0–4 gameweeks later, with tables fitted only on earlier seasons. Squared error
per start, leads 1–4, 2023-24 to 2025-26 weighted together:

| weighting | squared error | within-position ranking |
|---|---|---|
| none | 10.589 | 0.104 |
| Elo, as of the deadline | 10.558 | 0.142 |
| FPL difficulty, as recorded | **10.342** | **0.155** |

On its face FPL's rating wins, in every season and at every lead (Elo minus
FPL +0.13 to +0.31, intervals clear of zero). It cannot be taken at face value.
The historical record holds the difficulty ratings as FPL last revised them,
after the results — and given the same rate, those ratings beat **the betting
market's own pre-match view** on 2025-26 (10.47 against 10.68 at lead 0). No
rating set before kickoff does that. The revisions are real: between the GW5 and
GW6 deadlines of 2026-27, FPL changed 37 of the 180 difficulty ratings on GW5–13.

The fair test is what each rating said before kickoff, against the market at
kickoff. The snapshots hold FPL's ratings as they stood at each deadline since
GW5. Rank correlation with the market's goal difference:

| | at the deadline (n=60) | set 1–2 weeks earlier (n=60) | 2022-23 to 2025-26, as recorded |
|---|---|---|---|
| FPL difficulty | −0.65 | −0.57 | −0.63 (with hindsight) |
| Elo | +0.86 | **+0.87** | **+0.95** |

So the page weights fixtures by Elo: a line per position, a + b × gap with
the gap capped at ±300, fitted on every completed season (eight bins did no
better and were not monotone). It is a smaller effect than the old table
claimed — ignoring fixtures entirely costs only 0.03 per start against Elo,
0.06 against the market where it exists — and Elo's honest swing is about 0.75–1.26 for a defender and
0.86–1.14 for a midfielder.

One loose end this exposes: the component model's `team_difficulty` features
are trained on the same revised history, so it too learned a fixture effect
stronger than the ratings will show at a deadline.

### Against the crowd

Does the model pick better than FPL managers do? Each gameweek's ten most-bought
players are what the crowd decided before the deadline: net transfers track the
*previous* week's points (rank correlation 0.18) more than the week bought for
(0.10). For each, the model's alternative is the player it rated highest at the
same position within £0.5m. On every other gameweek of 2024-25 and GW2–6 of
2025-26, where the two disagreed (141 of 200 buys):

| | real points |
|---|---|
| the crowd's buy | 4.56 |
| the model's same-price alternative | 4.45 |

A draw: the model's pick won 55, lost 65 and tied 21, and the 95% interval on
the difference runs from −0.99 to +0.87. The crowd chases last week — its top
buys averaged 8.0 points the week before and 4.9 in the week bought for — but
the model does no better at the same price. The buys the model *agreed* with
averaged 5.46, a hint (interval −0.40 to +2.29) that it is more use as a filter
on popular picks than as a rival to them.

The same comparison with the leaked `xP` in the model gave its alternative 9.6
points to the crowd's 4.0 — which is how the leak was found.

### Team news, applied rather than learned

Two of the inputs the roadmap ranked highest have no per-gameweek history —
they exist only as end-of-season snapshots or not at all — so they cannot be
trained on. They are used anyway, in the two ways that are honest about that:

**Set-piece duty** (`penalties_order`, `corners_and_indirect_freekicks_order`)
is stable enough within a season that the season snapshot is a fair training
proxy, so it is a feature. First-choice penalty takers scored 0.359 goals per 90
in 2025-26 against 0.105 for everyone else. And yet: **on the 2024-25 backtest
it moved nothing** — MAE 1.2945 to 1.2967, Spearman 0.6946 to 0.6936, inside
noise. That was measured with the leaked `xP` in the model, which already knew
who had scored the penalties; it is kept, and worth re-measuring on the clean
model.

**Availability** (`chance_of_playing_next_round`) has no history at all, so it
is applied as a *ceiling* on the minutes prediction for the gameweek being
predicted rather than fed to the model. A player FPL puts at 25% cannot be given
a 90% chance of playing by a model that only knows he started last month. On the
GW5 predictions this changed 139 of 258 flagged players; Dean Henderson went
from 52% (recent minutes) to 0% (injured), and from 1.9 expected points to 0.

`fpl snapshot` stores the live player data before a deadline, and the played
rows of any gameweek with a snapshot train on the set-piece orders as they
stood then rather than as they stand today.

### Measured and not adopted

Plausible improvements, tested on the walk-forward harness and not adopted.

**Read these with care.** Every one was measured with the leaked `xP` in the
model, on test weeks whose `xP` knew the result, so any variant that leaned
less on `xP` was bound to lose — the xG-only rates and the cross-season prior
in particular deserve a clean re-test. They are kept as a record of what was
tried, not as settled verdicts.

**Scoring rates from xG only, with points and FPL's `xP` removed.** The
motivation was real: four gameweeks in, a £5.6m midfielder on a hot streak and
Haaland were predicted the same goals despite a threefold gap in xG per game,
because `xP` — which early in a season is simply FPL's recent-points average —
was the goals model's largest input. Removing points and `xP` from the goal and
assist rate models fixed that pair exactly as intended, and made the model
worse everywhere else:

| | MAE, recent-minutes players | Spearman within position | Actual points of the model's top XI |
|---|---|---|---|
| cold start, 2025-26 GW2–6 — current | **1.210** | **0.764** | **10.4** |
| cold start — xG-only rates | 1.463 | 0.708 | 9.7 |
| full season, 2024-25 — current | **1.297** | **0.694** | **9.5** |
| full season — xG-only rates | 1.432 | 0.666 | 8.6 |

Worse in every gameweek including GW2, and an ensemble of the two only lands in
between. Part of the reason is that xG is no longer the only route to points:
a midfielder with 0.09 xGI per 90 was rated 6.6 because he collects defensive
contributions, and the model was right about him. The Groß/Haaland compression
at the very top is real but is a magnitude issue, not a ranking one, and every
hand fix cost more than it recovered.

**`xP` out of the scoring-rate and bonus models only, with xG constrained.** A
more surgical version of the first idea, prompted by a defensive midfielder
rated 8.2 off one freak 14-point game: his `xP` and points form had leaked into
a *goals* prediction (0.44 expected, on 0.10 xG a game) and a *bonus* prediction
(ranked above Palmer for BPS in his own fixture). Keep `xP` in the minutes
model, remove it from goals, assists and BPS, and make the rates monotone in
xG, xA and threat. Every piece lost. The monotone constraint changed nothing
(the trees were not violating it). Removing `xP` from the rates alone cost
about 0.2 MAE and 0.04 in ranking. Removing it from the BPS model as well
*halved* the real points scored by the model's top XI (10.4 to 5.5 a player at
the cold start, 9.5 to 5.7 over 2024-25): without it the BPS model cannot rank
players within a fixture, and the rank-to-bonus table then hands the top bonus
figure to the wrong people. The individual failure is real; the fix for it is
not this. It is most likely a cross-season prior on underlying *rates* (last
season's xG per 90, minutes share) — the one idea on this list not yet tested.

**A cross-season prior on underlying rates.** The idea left open above: last
season's xG per 90, xA per 90, minutes share, start rate, defensive
contribution, threat and creativity, joined by name (the one join FPL's yearly
renumbering leaves), null below five games or with no prior season. Rates
rather than points, so a player whose situation changed is read afresh. Built,
tested for leakage across seasons, and run on identical folds with and without
the eight features — with a third season of history added so the priors were
populated in training. A wash: MAE 1.004 to 1.032 and 1.197 to 1.199 at the two
cold starts, 1.218 to 1.225 over the full season, ranking within a hundredth
either way, and the top XI's real points *down* 10.7 to 10.4 on the window
that matters most. On individual players it did what was intended — Groß, a
creator with 0.08 prior xG per 90, moved 5.1 to 4.0; Haaland 6.9 to 7.2 — but
the midfielder whose 8.2 prompted it had no prior season at all, and no prior
can reach a player with no history. Not adopted; the extra season was.

**A last-season points prior.** Rejected before testing, on the grounds that a
player whose situation has changed — a new club, a first-choice role — is
exactly who it would misrate.

**Opponent form instead of static fixture difficulty.** The intuition was that a
big club in a slump is an easier fixture than its rating says. Across two
seasons of player-matches, an opponent's recent form does carry a signal on its
own (leaky recent defences lift a player's points by about 11%, tight ones cut
them 10%) — but that signal is entirely explained by its correlation with the
static rating. Within a difficulty bucket, splitting opponents by their form
moves the outcome by zero: 5-week, 10-week and season-to-date windows, by xG
conceded or by points, all give mean gaps between −0.03 and +0.01. The reverse
does not hold — within a form band, static difficulty still separates 1.22 from
0.82 — so the rating carries information form does not, and FPL's own revisions
already fold in whatever form matters. The page keeps the measured difficulty
multipliers. *Since found to be compromised:* the "static" ratings in this test
were FPL's after-the-fact revisions, which have seen the form being tested
([fixtures beyond next week](#fixtures-beyond-next-week-elo-not-fpls-difficulty)),
so the verdict deserves a re-test against Elo.

**Head-to-head history.** Prompted by a content creator's table of goal
involvements and clean sheets for each pairing of clubs. Measured on every
match since 2022-23 with at least three earlier meetings (1,562 team-matches),
against the betting market's pre-match view of the same match. A side's
clean-sheet record against this opponent correlates +0.05 with keeping one,
against +0.23 for the market; with the market taken out, −0.04 (noise is ±0.05).
Goals: +0.02 beyond the market. For players — attackers with two or more
earlier meetings and 900 prior minutes, 5,854 starts — a goal-involvement rate
against this opponent above his own usual rate predicts +0.03 of the same
excess (noise ±0.03); the fifth with the best records against an opponent,
+0.58 involvements per 90 above their usual, scored −0.002 above it. What a
head-to-head record knows, the odds and the player's own rate already do. Not
adopted, and no new data source needed to check: it is all in the feature frame.

**Shrinking the bonus term for short samples.** Prompted by a promoted-side
forward handed 1.93 expected bonus off four appearances, then blanking while
the striker the model ranked fifth in the same fixture scored. Checked before
changing anything, by fitting the bonus model walk-forward on early-season
gameweeks (5–10) of all four seasons with `xP` present and comparing predicted
to actual bonus by sample size. There is nothing to shrink: players with three
or four prior appearances predicted 1.75+ bonus averaged 2.36 predicted and
2.36 actual (n=47), those with seven to nine 2.37 and 2.47, and the regression
slope of actual on predicted BPS sits at 0.9–1.0 in every sample-size band. The
model's top-ranked player in a fixture blanks 9% of the time whether he has four
games behind him or twenty; the forward was one of the 9%. Nor was his short
record what the model was reading: of his 25 predicted BPS, 11.8 came from
FPL's own `xP` of 6.0 and under 3 from his own BPS history. A shrinkage factor
would halve a figure that is already right on average. Not adopted.

The check did surface the term's real weakness, which is different: the
rank-to-bonus table is built in-sample, and it holds only as long as the
out-of-sample ranking is as good as the in-sample one. On the 2025-26 gameweeks
where `xP` is missing the BPS ranking degrades (correlation 0.3 against 0.75)
and the table promises 2.2 for first place while delivering 1.2. With `xP`
present the two agree (2.34 against 2.29). Building the table out-of-fold would
make it honest when the inputs are thin; it would not have changed this call.

**Correction.** This audit ran with the leaked `xP` in the BPS model and on
test weeks whose `xP` knew the result, so "calibrated" describes that model,
not a fair one. Emersonn's bonus came from FPL's figure because the model had
learned that FPL's figure knew who got bonus. The rebuilt model does not read
`xP`; its bonus term should be audited afresh.

## Running the tests

```bash
python -m pytest
```

Tests that need real data skip cleanly when `data/cache/` is cold, so the suite
runs offline. Populate the cache with `python -m src.cli build-features` to
enable them.

`.github/workflows/tests.yml` runs the same suite on every push and pull
request, on Python 3.11 (the oldest supported) and 3.14 (what the project is
developed on). There the cache is always cold, so it is the offline suite —
140 tests, with the 14 that need real data skipped.

## Current state and next step

The feature frame covers 2022-23 to 2025-26 and the current season: 115,819
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
