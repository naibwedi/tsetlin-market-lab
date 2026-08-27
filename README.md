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

## Where to run it (all free, no local setup needed)

| Environment | Use for | Free allowance |
|---|---|---|
| **GitHub Codespaces** | main dev — VS Code in browser, `.devcontainer/` auto-builds with `tmu` | 120 core-hours + 15 GB / month |
| **GitHub Actions** (`ci.yml`) | tests + full synthetic bake-off on every push | 2000 min/mo private, unlimited public |
| **GitHub Actions** (`collect.yml`) | unattended odds collection every 30 min | same |
| **Kaggle Notebooks** | heavy Tsetlin grid searches (GPU) | 30 GPU-h/week, 20 GB datasets |
| **Google Colab** | quick TM runs — `notebooks/tm_bakeoff_colab.ipynb` | free GPU, ephemeral disk |
| local venv / Docker | offline fallback | — |

Recommended: **push the repo to GitHub → open a Codespace → `make all`.** Nothing runs on your machine.

## Quick start (dry run, no API key)

```bash
pip install -e ".[dev]"
make all       # synth -> panel -> features -> bakeoff -> report  (synthetic data)
make test
```

Local venv on Windows: `py -3.12 -m venv .venv && .venv\Scripts\activate` first.
Docker: `docker build -t tml . && docker run -it -v ${PWD}:/workspace tml make all`.

`src/common/synthetic.py` fabricates a raw dataset with a **known** lead/lag
structure (pinnacle leads, followers copy 1–2 snapshots later, one lazy book).
It is the sanity target: a model that learns "follower moves next iff the leader
just moved" must beat the majority baseline. The pipeline is validated end-to-end
against it before the real feed is wired up.

## Real data — free, self-collected (no spend)

We bank our own history for free while developing on synthetic data.

1. Get a **free The Odds API key** (500 req/month): https://the-odds-api.com
2. `cp .env.example .env`, set `ODDS_API_KEY`. *(Optional: Betfair delayed key —
   free, real order book + volume — if a Betfair account is available in your
   country. Set `BETFAIR_*` and enable the adapter in `config/collect.yaml`.)*
3. Local test: `make collect` (appends to `data/raw/live/`).
4. Unattended: add `ODDS_API_KEY` as a GitHub repo secret. The
   `.github/workflows/collect.yml` cron runs every 30 min and commits snapshots
   back to the repo — stays inside the 500 req/month free tier.
5. After a few weeks: `make panel && make features && make bakeoff`.

### Fast-track (later, optional, paid)
**The Odds API Business ($99/mo)** ships a 5-min historical archive back to 2022
— one month's subscription bulk-downloads years of data. Use once the approach
is proven; see `src/ingest/odds_api.py`.

## Free tech stack

| Need | Free tool |
|---|---|
| multi-book odds | The Odds API free tier + our poller |
| exchange microstructure | Betfair delayed app key (free) |
| closing-line context | football-data.co.uk CSVs |
| scheduled collection | GitHub Actions cron |
| Tsetlin Machine compute | Google Colab / Kaggle free GPU (`notebooks/tm_bakeoff_colab.ipynb`) |
| everything else | scikit-learn, xgboost, lightgbm, tmu, ruff, pytest |

## Tsetlin Machine

`tmu` 0.8.3 ships a prebuilt Windows wheel (no compiler needed) **but is not
NumPy 2.x compatible**. Keep it in its own env pinned to `numpy<2`:

```bash
make setup-tm       # creates .venv-tm
make bakeoff-tm     # runs bake-off + clause report with TM included
```

`bakeoff.py` / `clauses.py` auto-skip the TM models if `tmu` is missing, so the
main `.venv` still runs the baseline bake-off. The `pycuda` warnings on startup
are harmless — it falls back to CPU clause banks.

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
