"""Ingest the 'Beat The Bookie' Kaggle dataset -> our raw long schema.

  austro/beat-the-bookie-worldwide-football-dataset

Real multi-book odds: 32 bookmakers x 72 hourly snapshots per match, from ~72h
before kickoff to kickoff. Covers 2015-09 .. 2016-06 (two batches: odds_series,
odds_series_b). We reshape the very wide `{outcome}_b{book}_{t}` columns into
one row per (snapshot_ts, match_id, bookmaker, outcome).

Auth: set KAGGLE_API_TOKEN=KGAT_... in the environment (never commit it).

    python -m src.ingest.btb --leagues "England: Premier League" --min-books 6
    python -m src.ingest.btb --leagues top          # 8 European leagues
"""
from __future__ import annotations

import argparse
import re

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


def _reshape_chunk(chunk: pd.DataFrame, mid2ct: dict, mid2meta: dict) -> pd.DataFrame:
    """Vectorised: wide series rows (one per match) -> long rows."""
    mids = chunk["match_id"].to_numpy()
    ct = pd.DatetimeIndex([mid2ct[m] for m in mids])
    home = np.array([mid2meta[m][0] for m in mids])
    away = np.array([mid2meta[m][1] for m in mids])
    frames = []
    for bi, book in enumerate(BOOKS, start=1):
        cols = {s: [f"{s}_b{bi}_{t}" for t in range(N_STEPS)] for s in ("home", "draw", "away")}
        if cols["home"][0] not in chunk.columns:
            continue
        arr = {s: chunk[cols[s]].to_numpy(dtype=float) for s in cols}  # each (n_matches, 72)
        valid = (arr["home"] > 1.0) & (arr["draw"] > 1.0) & (arr["away"] > 1.0)
        mi, ti = np.where(valid)
        if mi.size == 0:
            continue
        base = ct.take(mi)
        ts = base - pd.to_timedelta(N_STEPS - 1 - ti, unit="h")
        for slot, name in (("home", "Home"), ("draw", "Draw"), ("away", "Away")):
            frames.append(pd.DataFrame({
                "sport": "soccer_btb",
                "snapshot_ts": ts,
                "match_id": mids[mi].astype(str),
                "commence_time": base,
                "home_team": home[mi], "away_team": away[mi],
                "bookmaker": book, "book_last_update": ts,
                "market": "h2h", "outcome_name": name, "price": arr[slot][mi, ti],
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run(leagues: list[str], min_books: int, max_matches: int | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matches = _load_matches()
    if leagues:
        matches = matches[matches["league"].isin(leagues)]
    print(f"{len(matches)} matches in {matches['league'].nunique()} leagues")

    mid2ct = dict(zip(matches["match_id"], matches["commence_time"], strict=False))
    mid2meta = {r.match_id: (r.home_team, r.away_team) for r in matches.itertuples()}
    mid2league = dict(zip(matches["match_id"], matches["league"], strict=False))
    wanted = set(matches["match_id"])

    buf: dict[str, list[pd.DataFrame]] = {}
    seen = 0
    for f in _series_files():
        for chunk in pd.read_csv(f, encoding="latin-1", chunksize=400):
            chunk = chunk[chunk["match_id"].isin(wanted)]
            if chunk.empty:
                continue
            long = _reshape_chunk(chunk, mid2ct, mid2meta)
            if long.empty:
                continue
            nb = long.groupby("match_id")["bookmaker"].nunique()
            keep = set(nb[nb >= min_books].index)
            long = long[long["match_id"].isin(keep)]
            long["_league"] = long["match_id"].astype(int).map(mid2league).map(_slug)
            for slug, part in long.groupby("_league"):
                buf.setdefault(slug, []).append(part.drop(columns="_league"))
            seen += len(keep)
            if max_matches and seen >= max_matches:
                break
        if max_matches and seen >= max_matches:
            break

    for slug, parts in buf.items():
        df = pd.concat(parts, ignore_index=True).drop_duplicates(
            ["match_id", "snapshot_ts", "bookmaker", "outcome_name"])
        path = OUT_DIR / f"{slug}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {path.name}: {len(df):,} rows, {df.match_id.nunique()} matches, "
              f"{df.bookmaker.nunique()} books")
    print(f"done: ~{seen} matches reshaped")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", default="England: Premier League",
                    help="comma-separated exact league names, or 'top' for the top-8 European")
    ap.add_argument("--min-books", type=int, default=6)
    ap.add_argument("--max-matches", type=int, default=0)
    a = ap.parse_args()
    ls = TOP_LEAGUES if a.leagues.strip() == "top" else [
        s.strip() for s in a.leagues.split(",") if s.strip()]
    run(ls, a.min_books, a.max_matches or None)
