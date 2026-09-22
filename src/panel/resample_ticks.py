"""Resample genuine async tick data onto a shared time grid.

build_panel.py computes cross-book consensus/dispersion/n_books by grouping
rows on (match_id, snapshot_ts) -- correct for sources where every book is
sampled on the same grid (BTB's hourly snapshots, football-data's open/close,
the live collector's poll times), wrong for real tick data. Checked on the
OddsPapi ingest (2026-09-22): Pinnacle and bet365 update independently and
NEVER share an exact snapshot_ts (0 of 18,916 timestamp groups have more than
one bookmaker) -- every panel row ends up with n_books=1, so "consensus"
degenerates to "your own price" and the whole lead/lag feature set goes dead.

Fix: forward-fill each (match, bookmaker, outcome)'s last known price onto a
regular grid (default every 5 min, comfortably finer than the ~16 min median
update gap measured in scripts/oddspapi_resolution.py), spanning from the
earliest observation across either book to kickoff. Now every book has a
price at every grid timestamp, so build_panel.py's groupby actually finds
multiple books per snapshot.

    python -m src.panel.resample_ticks --raw-glob "data/raw/oddspapi/*.parquet" \
        --out data/raw/oddspapi_resampled/oddspapi_resampled.parquet --grid-minutes 5
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import resolve


def resample(raw: pd.DataFrame, grid_minutes: int) -> pd.DataFrame:
    raw = raw.copy()
    raw["snapshot_ts"] = pd.to_datetime(raw["snapshot_ts"], utc=True)
    raw["commence_time"] = pd.to_datetime(raw["commence_time"], utc=True)

    out_frames = []
    key_cols = ["match_id", "bookmaker", "outcome_name"]
    meta_cols = ["sport", "match_id", "commence_time", "bookmaker", "home_team",
                "away_team", "market", "result", "outcome_name"]

    for _match_id, g in raw.groupby("match_id", sort=False):
        commence = g["commence_time"].iloc[0]
        # ceil (not floor): guarantees the first grid point is >= the earliest
        # real observation, so merge_asof(backward) always has something to
        # find there instead of dropping it as NaN.
        grid_start = g["snapshot_ts"].min().ceil(f"{grid_minutes}min")
        grid_end = min(g["snapshot_ts"].max(), commence)
        if grid_end <= grid_start:
            continue
        grid = pd.date_range(grid_start, grid_end, freq=f"{grid_minutes}min", tz="UTC")
        grid_df = pd.DataFrame({"snapshot_ts": grid})

        for _, sub in g.groupby(key_cols, sort=False):
            row_meta = sub[meta_cols].iloc[0]
            sub = sub.sort_values("snapshot_ts")[["snapshot_ts", "price"]]
            merged = pd.merge_asof(grid_df, sub, on="snapshot_ts", direction="backward")
            merged = merged.dropna(subset=["price"])
            if merged.empty:
                continue
            for col in meta_cols:
                merged[col] = row_meta[col]
            merged["book_last_update"] = merged["snapshot_ts"]
            out_frames.append(merged)

    if not out_frames:
        return pd.DataFrame(columns=[*meta_cols, "snapshot_ts", "book_last_update", "price"])
    return pd.concat(out_frames, ignore_index=True)


def run(raw_glob: str, out_path: str, grid_minutes: int) -> None:
    paths = sorted(resolve(".").glob(raw_glob))
    frames = [pd.read_parquet(p) for p in paths]
    if not frames:
        raise SystemExit(f"no raw parquet matching {raw_glob}")
    raw = pd.concat(frames, ignore_index=True)
    out = resample(raw, grid_minutes)
    out_p = resolve(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_p, index=False)
    print(f"resample -> {out_p}  rows={len(out):,}  matches={out['match_id'].nunique()}  "
          f"grid={grid_minutes}min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-glob", default="data/raw/oddspapi/*.parquet")
    ap.add_argument("--out", default="data/raw/oddspapi_resampled/oddspapi_resampled.parquet")
    ap.add_argument("--grid-minutes", type=int, default=5)
    a = ap.parse_args()
    run(a.raw_glob, a.out, a.grid_minutes)
