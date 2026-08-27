# Tsetlin Sports Market Microstructure Lab — v0.1

Research question:

> **From the current state of the multi-bookmaker market for a football match,
> can a Tsetlin Machine predict which bookmaker will change its price in the next
> 5-minute snapshot — better than baselines — and emit human-readable rules?**

Not match results. Not betting tips. **The next bookmaker movement.**

The goal is *not* "prove TM is better" — it is to find **where** an interpretable,
clause-producing model has a real edge over XGBoost / LightGBM / plain rules.

## Pipeline

```
ingest  ─►  panel  ─►  features  ─►  bakeoff  ─►  report
(Odds API)  (tidy)     (~150 Bool   (TM vs 7      (clause_report.md,
                        literals)    baselines)    summary.md)
```

| Stage | Module | Output |
|---|---|---|
| ingest | `src/ingest/odds_api.py` | `data/raw/{sport}/{yyyy-mm}.parquet` |
| panel | `src/panel/build_panel.py` | `data/panel/panel.parquet` |
| features | `src/features/booleanize.py` | `data/features/{X,meta}.parquet`, `features.json` |
| bakeoff | `src/models/bakeoff.py` | `results/bakeoff_*.json`, `results/summary.md` |
| report | `src/analysis/clauses.py` | `results/clause_report.md` |

Config lives in `config/*.yaml`.

## Quick start (dry run, no API key)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
make all       # synth -> panel -> features -> bakeoff -> report  (synthetic data)
make test
```

`src/common/synthetic.py` fabricates a raw dataset with a **known** lead/lag
structure (pinnacle leads, followers copy 1–2 snapshots later, one lazy book).
It is the sanity target: a model that learns "follower moves next iff the leader
just moved" must beat the majority baseline. The pipeline is validated end-to-end
against it before the real feed is wired up.

## Real data

1. Subscribe to **The Odds API — Business** ($99/mo). Historical archive is
   included at no extra credit cost; 5-min snapshots since Sep 2022; Pinnacle +
   50+ EU/UK books.
2. `cp .env.example .env` and set `ODDS_API_KEY`.
3. Set the date range in `config/ingest.yaml`.
4. **Check the quota math first** (`src/ingest/odds_api.py` docstring): 6 leagues
   × ~40 matches/mo × 288 snapshots ≈ 69k requests / month of history vs a 200k
   req/month plan. Narrow `window_hours_before_kickoff` or `leagues` if needed.
5. `make ingest && make panel && make features && make bakeoff && make report`.
6. Downgrade the plan once the archive is downloaded.

## Tsetlin Machine

`tmu` builds C/CUDA extensions and is awkward to build natively on Windows.
Run the modeling stages in **WSL2 / Docker / Linux CI**:

```bash
pip install -e ".[tm]"
```

`bakeoff.py` and `clauses.py` auto-skip the TM models when `tmu` is missing, so
the baseline bake-off still runs on Windows.

## Leakage discipline

- Split is **time-ordered by kickoff**; matches never straddle a boundary.
- `test_pipeline.py` includes a **shuffled-target control** — every model must
  collapse to ROC-AUC ≈ 0.5.
- Target uses a forward `shift(-h)` per (match, book); rows without a future
  snapshot are dropped.

## Roadmap (later phases — not built yet)

P2 3-class + "which book moves *first*" · P3 GraphTM lead/lag graph ·
P4 news/LLM literals · P5 regime + cross-market · P6 backtest → paper trade →
CLV · P7 sub-minute microstructure via Betfair Exchange Stream.

See `C:\Users\naib\.claude\plans\peaceful-beaming-snowflake.md` for the full plan.
