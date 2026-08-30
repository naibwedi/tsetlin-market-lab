"""Sanity check the whole economic premise: is the market consensus actually a
good estimate of the true outcome probability?

Uses the real match results (`result` column, from btb.py). For each match we
take the consensus fair p_home at the last pre-kickoff snapshot, bin matches by
that probability, and compare the predicted rate to the realised home-win rate.
Also reports Brier score and log-loss of the closing consensus vs a flat prior.

Run:  python -m scripts.calibration --panel data/panel_allbtb.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common.config import resolve


def run(panel_path: str) -> None:
    p = pd.read_parquet(resolve(panel_path))
    if "result" not in p.columns:
        raise SystemExit("panel has no `result` column - re-ingest with the updated btb.py")
    p = p[p["minutes_to_kickoff"] >= 0].sort_values("minutes_to_kickoff")

    close = p.groupby("match_id").agg(
        p_home=("cons_fp_home", "first"), p_draw=("cons_fp_draw", "first"),
        result=("result", "first"),
    ).dropna()
    close["p_away"] = (1 - close["p_home"] - close["p_draw"]).clip(0, 1)
    y_home = (close["result"] == "H").astype(int)

    # calibration table
    bins = np.arange(0, 1.01, 0.1)
    close["bin"] = pd.cut(close["p_home"], bins)
    cal = close.groupby("bin", observed=True).agg(
        n=("result", "size"), pred=("p_home", "mean"),
        actual=("result", lambda s: (s == "H").mean()),
    ).round(3)

    brier = float(((close["p_home"] - y_home) ** 2).mean())
    base = float(y_home.mean())
    brier_base = float(((base - y_home) ** 2).mean())
    eps = 1e-12
    ll = float(-(y_home * np.log(close["p_home"] + eps)
                 + (1 - y_home) * np.log(1 - close["p_home"] + eps)).mean())

    lines = [
        "# Is the consensus a good probability estimate?",
        "",
        f"- matches with a result: {len(close):,}  |  home-win base rate: {base:.3f}",
        "",
        "## Calibration of the closing consensus p(home win)",
        "```", cal.to_string(), "```",
        "",
        f"- Brier score (closing consensus): **{brier:.4f}**  (always-predict-base-rate: {brier_base:.4f})",
        f"- log-loss: {ll:.4f}",
        "",
        "If `pred` ~= `actual` down the calibration table and Brier < base, the "
        "consensus is a well-calibrated probability -- which is what the economic "
        "branch assumes when it treats consensus as proxy truth.",
    ]
    out = resolve("results/calibration.md")
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panel_allbtb.parquet")
    run(ap.parse_args().panel)
