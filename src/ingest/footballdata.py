"""Ingest football-data.co.uk season CSVs -> our raw long schema.

Free public dataset, modern seasons (to the current one), ~8 bookmakers with an
**opening** and a **closing** price per match. No intraday series, so this fixes
the *data age* problem (2015-16 -> today), not the *resolution* problem. It is
the test bed for: does the closing-consensus forecast signal survive on modern
seasons?

Each match becomes two snapshots per bookmaker:
  open  = when the site captured odds: Friday afternoon for weekend games,
          Tuesday afternoon for midweek games (per the site's notes). This is an
          APPROXIMATION of the real capture time.
  close = kickoff minus 5 minutes.

    python -m src.ingest.footballdata --seasons 2122 2223 2324 2425 2526 \
        --divs E0 SP1 D1 I1 F1
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from src.common.config import resolve

URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
CSV_DIR = resolve("data/raw/fd_csv")
OUT_DIR = resolve("data/raw/footballdata")

# football-data column prefix -> bookmaker name used across the lab
BOOKS = {
    "B365": "bet365", "PS": "pinnacle", "WH": "williamhill", "BW": "bwin",
    "IW": "interwetten", "VC": "betvictor", "LB": "ladbrokes", "SB": "sportingbet",
    "GB": "gamebookers", "BS": "bluesquare", "SJ": "stanjames", "BFE": "betfair_ex_eu",
}
MIN_GAP_H = 3.0


def opening_time(kickoff: datetime) -> datetime:
    """Approximate when the site captured 'opening' odds for this kickoff."""
    d = kickoff.replace(minute=0, second=0, microsecond=0)
    wd = d.weekday()                      # Mon=0 .. Sun=6
    if wd in (5, 6, 0):                   # Sat, Sun, Mon -> Friday before
        back = {5: 1, 6: 2, 0: 3}[wd]
        open_ = (d - timedelta(days=back)).replace(hour=15)
    elif wd == 4:                         # Friday -> same-day afternoon
        open_ = d.replace(hour=12)
    else:                                 # Tue, Wed, Thu -> Tuesday afternoon
        open_ = (d - timedelta(days=wd - 1)).replace(hour=15)
    if kickoff - open_ < timedelta(hours=MIN_GAP_H):
        open_ = kickoff - timedelta(hours=24)
    return open_


def _download(season: str, div: str) -> pd.DataFrame | None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    path = CSV_DIR / f"{season}_{div}.csv"
    if not path.exists():
        r = requests.get(URL.format(season=season, div=div), timeout=30)
        if r.status_code != 200 or len(r.content) < 200:
            print(f"  {season} {div}: not available (HTTP {r.status_code})")
            return None
        path.write_bytes(r.content)
        time.sleep(0.5)                   # be polite to a free public site
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    return df.dropna(subset=["HomeTeam", "AwayTeam", "Date"])


def _rows(df: pd.DataFrame, season: str, div: str) -> pd.DataFrame:
    date = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    tm = df["Time"].fillna("15:00") if "Time" in df.columns else pd.Series("15:00", index=df.index)
    ko = pd.to_datetime(date.dt.strftime("%Y-%m-%d") + " " + tm.astype(str), errors="coerce", utc=True)
    df = df.assign(_ko=ko).dropna(subset=["_ko"])

    out = []
    for code, book in BOOKS.items():
        for phase in ("open", "close"):
            cols = [f"{code}{'C' if phase == 'close' else ''}{o}" for o in "HDA"]
            if not all(c in df.columns for c in cols):
                continue
            sub = df[["_ko", "HomeTeam", "AwayTeam", "FTR", *cols]].copy()
            sub[cols] = sub[cols].apply(pd.to_numeric, errors="coerce")
            sub = sub[(sub[cols] > 1.0).all(axis=1)]
            if sub.empty:
                continue
            ko_ts = sub["_ko"]
            snap = ko_ts - pd.Timedelta(minutes=5) if phase == "close" else ko_ts.map(
                lambda k: pd.Timestamp(opening_time(k.to_pydatetime())))
            base = pd.DataFrame({
                "sport": f"fd_{div}", "snapshot_ts": snap, "commence_time": ko_ts,
                "match_id": ko_ts.dt.strftime("%Y%m%d") + f"_{div}_" + sub["HomeTeam"].str.replace(" ", "")
                            + "_" + sub["AwayTeam"].str.replace(" ", ""),
                "home_team": sub["HomeTeam"], "away_team": sub["AwayTeam"],
                "bookmaker": book, "book_last_update": snap, "market": "h2h",
                "result": sub["FTR"],
            })
            for o, name in zip(cols, ("Home", "Draw", "Away"), strict=True):
                out.append(base.assign(outcome_name=name, price=sub[o].to_numpy(float)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run(seasons: list[str], divs: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for season in seasons:
        for div in divs:
            df = _download(season, div)
            if df is None:
                continue
            r = _rows(df, season, div)
            if r.empty:
                continue
            print(f"  {season} {div}: {r.match_id.nunique()} matches, {r.bookmaker.nunique()} books")
            parts.append(r)
    if not parts:
        raise SystemExit("nothing downloaded")
    allr = pd.concat(parts, ignore_index=True)
    allr = allr.drop_duplicates(["match_id", "snapshot_ts", "bookmaker", "outcome_name"])
    path = OUT_DIR / "footballdata.parquet"
    allr.to_parquet(path, index=False)
    print(f"-> {path}: {len(allr):,} rows, {allr.match_id.nunique()} matches, "
          f"{allr.bookmaker.nunique()} books, {allr.commence_time.min():%Y-%m-%d} .. "
          f"{allr.commence_time.max():%Y-%m-%d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=["2122", "2223", "2324", "2425", "2526"])
    ap.add_argument("--divs", nargs="+", default=["E0", "SP1", "D1", "I1", "F1"])
    a = ap.parse_args()
    run(a.seasons, a.divs)
