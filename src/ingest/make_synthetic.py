"""Write a synthetic raw dataset to data/raw/ so the pipeline can be dry-run
without a live Odds API key."""
from __future__ import annotations

import argparse

from src.common.config import resolve
from src.common.synthetic import make_raw


def run(n_matches: int = 40, seed: int = 0) -> None:
    df = make_raw(n_matches=n_matches, seed=seed)
    out = resolve("data/raw") / "soccer_epl"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "2026-03.parquet"
    df.to_parquet(path, index=False)
    print(f"synthetic raw -> {path}  rows={len(df):,}  matches={df.match_id.nunique()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-matches", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.n_matches, a.seed)
