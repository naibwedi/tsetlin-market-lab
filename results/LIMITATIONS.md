# Limitations & open problems

Honest account of what constrains the current results (v0.1–v0.4).

## 1. The data — the binding constraint

**The Beat-The-Bookie dataset is 2015–2016 and sampled hourly.** Both are
problems, and everything downstream inherits them:

- **Wrong decade.** Betting markets have professionalised heavily since 2016 —
  more sharp capital, exchange dominance, faster pricing. A lead/lag structure
  that held in 2015 may not hold now.
- **Wrong resolution.** Information propagates between bookmakers in *minutes*
  (the project premise guessed 18–46 seconds). At 1-hour granularity we only see
  "book X updated sometime this hour" — the microstructure timing that was the
  whole point is gone. The 0.9 pp CLV signal in the economic branch is measured
  at a resolution that almost certainly blurs the real exploitable window.
- **Proxy truth.** The backtests treat the cross-book *consensus* as "true
  probability". If the consensus is biased, every CLV number is biased. We have
  real match results in the data and only started using them (v0.4 calibration
  check) — that check should gate how much to trust the economic branch.
- **One 6-month season per league.** No cross-season test, no regime-change
  check. The time-split is clean but narrow.
- **Bookmaker identity is an assumption.** `BOOKS` in `btb.py` maps column index
  → name by the order in the source paper. If that order is wrong, the
  sharp-book rules (`bet365 leads`, etc.) are mislabelled.

**Our own live collector** is real and modern but tiny: hourly-ish, ~40 books,
9 EPL matches, a handful of snapshots. The free Odds-API tier (500 req/month)
caps it at ~2-hour spacing. Weeks to accumulate anything usable; a Betfair
delayed key or a paid feed is the actual unlock for the economic branch.

## 2. The Tsetlin Machine

- **It ties the baselines, never beats them.** Synthetic 0.77 vs XGBoost 0.78;
  real EPL v0.1 0.74 vs 0.76; v0.2 features 0.71 — now *below* the baselines. The
  `max_included_literals=5` budget that made the clauses readable cost ~3 AUC
  points.
- **Training is unstable.** On v0.2 features the per-epoch AUC oscillated
  0.72 → 0.59 → 0.71. `T` and `s` are untuned; no hyper-parameter search has been
  run. The green-tsetlin `HyperparameterSearch` or a proper grid is needed.
- **The tooling is fragile.** Five rounds of environment fights: `tmu` needs a
  CUDA GPU (CPU hung a 6-hour CI runner); `green-tsetlin`'s pip build crashes on
  its own bundled example on Linux; Colab moved to Python 3.13, which `tmu` has
  no wheel for. Every TM run now = Colab notebook + T4 + a throwaway Python 3.11
  venv. It cannot run in CI. A research programme needs this on a stable box.
- **Clause reading is crude.** We dump clauses with ≤6 literals and rank by a
  CUDA off-by-one-safe loop; clause *weights* (which clauses actually carry the
  vote) aren't read, so the "top clauses" list is approximate.

## 3. Compute & engineering

- **Single machine, pandas.** The 8-league panel build hits 5+ minutes and ~5 GB
  RAM; `pivot_table` hung outright on 15 M rows (now `.pivot`), `_stale_counter`
  was O(n²) (now vectorised). This does not scale to "all 550 leagues" or years
  of data. A columnar engine (polars / DuckDB) or chunked processing is needed.
- **No experiment tracking.** Results overwrite `results/*.md` each run. No
  versioned datasets, no run log, no way to diff two feature sets cleanly.
- **Colab friction.** ~90-min idle kill, ~12-h cap, GPU allocation queues. Every
  TM iteration is a 15–20-min round trip through a fragile stack.
- **Auth is a workaround.** GitHub and Kaggle are driven by pulling tokens from
  the OS credential store / env vars, not a proper service account.

## 4. Method — what is not yet validated

- **The backtest is directional evidence, not P&L.** No commission, no stake
  model, no check that the flagged price/size was actually bettable. The 0.9 pp
  "edge" could vanish after costs or be a measurement artefact.
- **Only `p_home` is modelled.** Draw and away are ignored; other markets
  (Asian handicap, over/under) untouched.
- **The "offside" universe** (book ≥1 % longer than consensus) hasn't been
  audited — some of those are genuine value, some are one book being slow to
  open a market.

## 5. Process risk

v0.1 → v0.4 landed in a single session. Each branch (rule discovery, economic)
is plausibly a month of work. The temptation is to keep adding branches (GraphTM,
news signals, more leagues) before any one result is deeply validated. The
calibration check (v0.4) and a real costed backtest should come before new
features.

## The one-line version

**The data is hourly and a decade old, so the project cannot yet answer the
question it was designed around.** The rule-discovery branch produces a
believable description of the 2015 market; the economic branch shows a faint
signal that needs modern sub-hourly data to become real.
