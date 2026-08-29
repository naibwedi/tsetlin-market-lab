"""Ingest the 'Beat The Bookie' Kaggle dataset -> our raw long schema.

  austro/beat-the-bookie-worldwide-football-dataset

Real multi-book odds: 32 bookmakers x 72 hourly snapshots per match, from ~72h
before kickoff to kickoff. Covers 2015-09 .. 2016-06 (two batches: odds_series,
odds_series_b). We reshape the very wide `{outcome}_b{book}_{t}` columns into
one row per (snapshot_ts, match_id, bookmaker, outcome).

Auth: set KAGGLE_API_TOKEN=KGAT_... in the environment (never commit it).

    python -m src.ingest.btb --leagues "England: Premier League" --min-books 6
"""
from __future__ import annotations

import argparse
import re
from datetime import timedelta

import numpy as np
import pandas as pd

from src.common.config import resolve

# b1..b32 in the order listed in Kaufman et al. (arXiv:1710.02824, p.5)
BOOKS = [
    "interwetten", "bwin", "bet_at_home", "unibet", "stan_james", "expekt", "10bet",
    "williamhill", "bet365", "pinnacle", "doxxbet", "betsafe", "betway", "888sport",
    "ladbrokes", "betclic", "sportingbet", "mybet", "betsson", "188bet", "jetbull",
    "paddypower", "tipico", "coral", "sbobet", "betvictor", "12bet", "titanbet",
    "youwin", "comeon", "betadonis", "betfair_ex_eu",
]
N_STEPS = 72
BTB_DIR = resolve("data/raw/btb")
OUT_DIR = resolve("data/raw/btb_long")

TOP_LEAGUES = [
    "England: Premier League", "Spain: Primera Division", "Germany: Bundesliga",
    "Italy: Serie A", "France: Ligue 1", "Netherlands: Eredivisie",
    "Portugal: Primeira Liga", "Europe: Champions League",
]


def _load_matches() -> pd.DataFrame:
    frames = []
    for name in ("odds_series_matches", "odds_series_b_matches"):
        p = BTB_DIR / f"{name}.csv.gz"
        if p.exists():
            df = pd.read_csv(p, encoding="latin-1")
            df.columns = [c.strip() for c in df.columns]
            frames.append(df)
    m = pd.concat(frames, ignore_index=True).drop_duplicates("match_id")
    m["league"] = m["league"].str.strip()
    m["commence_time"] = pd.to_datetime(m["match_datetime"], errors="coerce", utc=True)
    return m.dropna(subset=["commence_time"])


def _series_files() -> list:
    return [p for p in (BTB_DIR / "odds_series.csv.gz", BTB_DIR / "odds_series_b.csv.gz") if p.exists()]


def _reshape_match(row: pd.Series, meta: pd.Series) -> list[dict]:
    """One wide series row -> long rows, dropping missing/degenerate quotes."""
    out: list[dict] = []
    ct = meta["commence_time"]
    for bi, book in enumerate(BOOKS, start=1):
        for t in range(N_STEPS):
            prices = {}
            for slot, name in (("home", "Home"), ("draw", "Draw"), ("away", "Away")):
                col = f"{slot}_b{bi}_{t}"
                v = row.get(col, np.nan)
                if pd.notna(v) and float(v) > 1.0:
                    prices[name] = float(v)
            if len(prices) < 3:
                continue
            ts = (ct - timedelta(hours=N_STEPS - 1 - t)).strftime("%Y-%m-%dT%H:%M:%SZ")
            for name, price in prices.items():
                out.append({
                    "sport": "soccer_btb",
                    "snapshot_ts": ts,
                    "match_id": str(row["match_id"]),
                    "commence_time": ct.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "home_team": meta.get("home_team", ""),
                    "away_team": meta.get("away_team", ""),
                    "bookmaker": book,
                    "book_last_update": ts,
                    "market": "h2h",
                    "outcome_name": name,
                    "price": price,
                })
    return out


def run(leagues: list[str], min_books: int, max_matches: int | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matches = _load_matches().set_index("match_id")
    if leagues:
        matches = matches[matches["league"].isin(leagues)]
    print(f"{len(matches)} matches in {matches['league'].nunique()} leagues")

    wanted = set(matches.index.astype(str))
    per_league: dict[str, list[dict]] = {}
    seen = 0
    for f in _series_files():
        for chunk in pd.read_csv(f, encoding="latin-1", chunksize=500):
            chunk["match_id"] = chunk["match_id"].astype(str)
            chunk = chunk[chunk["match_id"].isin(wanted)]
            for _, row in chunk.iterrows():
                meta = matches.loc[int(row["match_id"])]
                rows = _reshape_match(row, meta)
                if not rows:
                    continue
                n_books = len({r["bookmaker"] for r in rows})
                if n_books < min_books:
                    continue
                per_league.setdefault(_slug(meta["league"]), []).extend(rows)
                seen += 1
                if max_matches and seen >= max_matches:
                    break
            if max_matches and seen >= max_matches:
                break

    for slug, rows in per_league.items():
        df = pd.DataFrame(rows)
        path = OUT_DIR / f"{slug}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {path.name}: {len(df):,} rows, {df.match_id.nunique()} matches, "
              f"{df.bookmaker.nunique()} books")
    print(f"done: {seen} matches reshaped")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", default="England: Premier League",
                    help="comma-separated exact league names, or 'top' for the top-8 European")
    ap.add_argument("--min-books", type=int, default=6)
    ap.add_argument("--max-matches", type=int, default=0)
    a = ap.parse_args()
    leagues = TOP_LEAGUES if a.leagues.strip() == "top" else [
        s.strip() for s in a.leagues.split(",") if s.strip()]
    run(leagues, a.min_books, a.max_matches or None)
