"""Ingest OddsPapi's /historical-odds -> our raw long schema.

Confirmed 2026-09-22 (results/data_sources.md): free-tier /historical-odds
returns real sub-hourly price history for real sharp books on real top-5
leagues (bet365's Full Time Result market on a Premier League fixture:
median gap ~16 min, 96.9% of gaps under 1h). This is the first data source
this project has found that answers the *resolution* problem, not just the
*decade* problem football-data.co.uk fixed.

Two calls per match: /historical-odds (price history) and /scores (final
result, needed to label the target the same way btb.py/footballdata.py do).
Market 101 = "Full Time Result" with outcomes 101/102/103 = home/draw/away
(OddsPapi's GET /markets reference; not fixture-specific, so hardcoded here
rather than spending a request on it every run).

TERMS OF SERVICE: "You may not resell, repackage, or redistribute our data
as a standalone product" (oddspapi.io/us/legal/terms). This repo is public,
so raw price data is written only to data/raw/oddspapi/ (gitignored, same
treatment as the BTB Kaggle dump and the football-data CSVs) -- never commit
the parquet this script writes.

    ODDSPAPI_KEY=... python -m src.ingest.oddspapi --max-matches 15
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

from src.common.config import resolve

BASE = "https://api.oddspapi.io/v4"
OUT_DIR = resolve("data/raw/oddspapi")
REQUEST_DELAY_S = 3.0

WINNER_MARKET_KEY = "101"                              # "Full Time Result"
OUTCOME_NAME = {"101": "Home", "102": "Draw", "103": "Away"}

# OddsPapi bookmaker key -> the name used across this project's other sources
BOOK_NAMES = {
    "bet365": "bet365", "pinnacle": "pinnacle", "williamhill": "williamhill",
    "marathonbet": "marathonbet", "paddypower": "paddypower", "ladbrokes": "ladbrokes",
    "betvictor": "betvictor", "unibet": "unibet", "bwin": "bwin",
}


def get(session: requests.Session, key: str, path: str, **params) -> requests.Response:
    time.sleep(REQUEST_DELAY_S)
    r = session.get(f"{BASE}{path}", params={"apiKey": key, **params}, timeout=60)
    if r.status_code == 429:
        wait = float(r.headers.get("Retry-After", 15))
        print(f"  (429, retrying once after {wait:.0f}s)")
        time.sleep(wait)
        r = session.get(f"{BASE}{path}", params={"apiKey": key, **params}, timeout=60)
    return r


def _fixtures(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


MAX_WINDOW_DAYS = 10   # /fixtures 400s on wider ranges (same limit the vendor
                       # documents for /historical-odds date-range queries)


def find_finished_fixtures(session: requests.Session, key: str, days_back: int,
                            tournament: str, category: str, max_matches: int) -> list[dict]:
    """Sweep backward in <=10-day windows (widest range the API accepts)
    until max_matches is found or days_back is exhausted."""
    matches: list[dict] = []
    window_end = date.today()
    swept = 0
    while swept < days_back and len(matches) < max_matches:
        window_days = min(MAX_WINDOW_DAYS, days_back - swept)
        window_start = window_end - timedelta(days=window_days)
        r = get(session, key, "/fixtures", sportId=10,
                **{"from": window_start.isoformat(), "to": window_end.isoformat()})
        if not r.ok:
            print(f"  /fixtures {window_start}..{window_end}: HTTP {r.status_code} {r.text[:200]}")
            break
        fx = _fixtures(r.json())
        found = [f for f in fx
                 if str(f.get("tournamentName", "")).strip().lower() == tournament.lower()
                 and category.lower() in str(f.get("categoryName", "")).lower()
                 and f.get("statusName") == "Finished"]
        matches.extend(found)
        window_end = window_start
        swept += window_days
    matches.sort(key=lambda f: f.get("startTime", ""), reverse=True)
    # de-dupe (a fixture could theoretically straddle a window boundary)
    seen: set[str] = set()
    uniq = []
    for m in matches:
        if m["fixtureId"] not in seen:
            seen.add(m["fixtureId"])
            uniq.append(m)
    return uniq[:max_matches]


def _result_from_scores_body(body: dict) -> str | None:
    """H / D / A from the full-time score.

    Shape confirmed live 2026-09-22: {"scores": {"periods": {"fulltime": {...},
    "result": {...}, "p1": {...}}}} - a dict keyed by named periods (not the
    numbered-list shape the docs implied). "fulltime" is the 90-minute score;
    fall back to "result" (same value in every fixture checked so far).
    """
    periods = ((body.get("scores") or {}).get("periods") or {})
    final = periods.get("fulltime") or periods.get("result")
    if not final:
        return None
    h, a = final.get("participant1Score"), final.get("participant2Score")
    if h is None or a is None:
        return None
    return "H" if h > a else ("A" if a > h else "D")


def fetch_result(session: requests.Session, key: str, fixture_id: str) -> str | None:
    r = get(session, key, "/scores", fixtureId=fixture_id)
    if not r.ok:
        return None
    return _result_from_scores_body(r.json())


def parse_history(hist: dict, fixture: dict, books: list[str], result: str | None) -> pd.DataFrame:
    home, away = fixture.get("participant1Name"), fixture.get("participant2Name")
    commence = pd.to_datetime(fixture["startTime"], utc=True)
    match_id = f"{commence:%Y%m%d}_{fixture.get('tournamentSlug', 'oddspapi')}_{fixture['fixtureId']}"
    sport = f"oddspapi_{fixture.get('tournamentSlug', 'unknown')}"

    rows = []
    bookmakers = hist.get("bookmakers") or {}
    for book in books:
        b = bookmakers.get(book)
        if not b:
            continue
        market = (b.get("markets") or {}).get(WINNER_MARKET_KEY)
        if not market:
            continue
        for outcome_key, name in OUTCOME_NAME.items():
            outcome = (market.get("outcomes") or {}).get(outcome_key)
            if not outcome:
                continue
            for series in (outcome.get("players") or {}).values():
                for p in series:
                    if p.get("createdAt") is None or p.get("price") is None:
                        continue
                    rows.append({
                        "sport": sport, "snapshot_ts": p["createdAt"], "match_id": match_id,
                        "commence_time": commence, "bookmaker": BOOK_NAMES.get(book, book),
                        "home_team": home, "away_team": away,
                        "book_last_update": p["createdAt"], "market": "h2h",
                        "result": result, "outcome_name": name, "price": float(p["price"]),
                    })
    return pd.DataFrame(rows)


def run(days_back: int, max_matches: int, books: list[str], tournament: str, category: str) -> None:
    key = os.environ.get("ODDSPAPI_KEY", "")
    if not key:
        raise SystemExit("set ODDSPAPI_KEY")

    session = requests.Session()
    matches = find_finished_fixtures(session, key, days_back, tournament, category, max_matches)
    print(f"found {len(matches)} finished {tournament} fixtures in the last {days_back} days")

    frames = []
    for f in matches:
        label = f"{f.get('participant1Name')} v {f.get('participant2Name')}"
        r = get(session, key, "/historical-odds", fixtureId=f["fixtureId"], bookmakers=",".join(books))
        if not r.ok:
            print(f"  {label}: history HTTP {r.status_code} {r.text[:200]}, skipped")
            continue
        result = fetch_result(session, key, f["fixtureId"])
        df = parse_history(r.json(), f, books, result)
        print(f"  {label}: {len(df):,} rows, result={result}")
        if not df.empty:
            frames.append(df)

    if not frames:
        print("no data collected")
        return
    out = pd.concat(frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "oddspapi.parquet"
    if path.exists():
        prior = pd.read_parquet(path)
        out = pd.concat([prior, out], ignore_index=True).drop_duplicates(
            ["match_id", "bookmaker", "outcome_name", "snapshot_ts"])
    out.to_parquet(path, index=False)
    print(f"\n-> {path}  rows={len(out):,}  matches={out['match_id'].nunique()}  "
          f"(NOT committed to git - data/raw/ is gitignored per OddsPapi's terms)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-back", type=int, default=60)
    ap.add_argument("--max-matches", type=int, default=15)
    ap.add_argument("--books", default="pinnacle,bet365")
    ap.add_argument("--tournament", default="Premier League")
    ap.add_argument("--category", default="England")
    a = ap.parse_args()
    run(a.days_back, a.max_matches, a.books.split(","), a.tournament, a.category)
