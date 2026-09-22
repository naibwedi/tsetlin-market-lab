# src/ingest/ index
- `collect.py` — live poller (used by collect.yml); writes `data/raw/live/<sport>/<date>.parquet`. Paces credits along a linear burn-down of the month (quota resets on the 1st), skips a league when the free `/events` endpoint shows nothing kicking off, and asks for an explicit 10-book list (billed as one region) so Pinnacle and Betfair EU stay in.
- `footballdata.py` — football-data.co.uk seasons (2021/22 on): opening + closing odds per book, no intraday. Writes `data/raw/footballdata/`. Opening time is an approximated Friday/Tuesday capture.
- `oddspapi.py` — OddsPapi free-tier ingest: real sub-hourly price history (median ~16 min gap, confirmed 2026-09-22) for finished Premier League matches, 2 sharp books by default. Writes `data/raw/oddspapi/` (gitignored, never commit — their terms forbid redistributing raw data). `--max-matches`/`--days-back`/`--books` to tune. Run via `oddspapi-ingest.yml` (manual dispatch only, quota is 250/month). The two books update asynchronously (never share a timestamp) — run `../panel/resample_ticks.py` on the output before `build_panel.py`, or every panel row gets `n_books=1`.
- `odds_api.py` — The Odds API adapter (free tier 500 credits/month).
- `betfair.py` — optional Betfair delayed-key adapter (may be unavailable in Norway).
- `btb.py` — Beat The Bookie Kaggle ingest (hourly data, 2015-16) plus match results.
- `make_synthetic.py` — synthetic lead/lag data for pipeline tests.
