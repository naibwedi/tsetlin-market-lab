"""Costed backtest for the closing-consensus forecast's "home will shorten" flag.

Earlier CLV checks (in consensus_forecast.py) only measured whether the consensus
*moved the right way* -- a proxy, not money. This settles real bets: stake 1 unit
at the best home price available across books at the flagged snapshot, decided by
the actual match result. For a fixed-odds bookmaker the price already has the vig
built in, so no separate commission applies; pass --commission for an exchange-style
backtest (e.g. Betfair's ~5%) taken off net winnings only.

Run:  python -m scripts.costed_backtest --config config/consensus.yaml
      python -m scripts.costed_backtest --config config/consensus.fd.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common.config import load_yaml, resolve
from src.models.consensus_forecast import _split


def _stat(d: pd.DataFrame, commission: float) -> dict:
    if d.empty:
        return {"n": 0, "avg_odds": None, "win_rate": None, "roi_pct": None,
                "total_pnl_units": None}
    win = d["result"] == "H"
    gross = np.where(win, d["entry_odds"] - 1.0, -1.0)
    pnl = np.where(win, (d["entry_odds"] - 1.0) * (1 - commission), -1.0)
    return {
        "n": len(d), "avg_odds": round(float(d["entry_odds"].mean()), 3),
        "win_rate": round(float(win.mean()), 3),
        "roi_pct": round(100 * float(pnl.mean()), 2),
        "total_pnl_units": round(float(pnl.sum()), 2),
        "gross_roi_pct_no_commission": round(100 * float(gross.mean()), 2),
    }


def run(config_path: str, thr: float = 0.005, commission: float = 0.0,
        out_name: str | None = None) -> None:
    cfg = load_yaml(config_path)
    d = resolve(cfg["out_dir"])
    X = pd.read_parquet(d / "X.parquet")
    meta = pd.read_parquet(d / "meta.parquet")
    feat = json.loads((d / "features.json").read_text())
    y = meta["y"].to_numpy()

    tr, te = _split(meta, cfg)
    Xtr, Xte = X[feat].to_numpy()[tr], X[feat].to_numpy()[te]
    ytr = y[tr]

    from xgboost import XGBRegressor
    xgb = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
    xgb.fit(Xtr, ytr)
    yhat = xgb.predict(Xte)

    te_meta = meta[te].reset_index(drop=True)
    te_meta["yhat"] = yhat

    panel = pd.read_parquet(resolve(cfg["panel_path"]))
    entry = (panel.groupby(["match_id", "snapshot_ts"])
              .agg(entry_odds=("o_home", "max")).reset_index())
    result = panel.groupby("match_id")["result"].first().rename("result")

    j = te_meta.merge(entry, on=["match_id", "snapshot_ts"], how="left")
    j = j.merge(result, on="match_id", how="left")
    j = j.dropna(subset=["entry_odds", "result"])

    flagged = j[j["yhat"] > thr]
    eligible = j  # everything in the test set is a candidate snapshot
    rng = np.random.default_rng(0)
    rand = eligible.iloc[rng.choice(len(eligible), min(len(flagged), len(eligible)), replace=False)] \
        if len(flagged) else eligible.iloc[:0]

    rows = {
        "model-flagged (yhat > thr)": _stat(flagged, commission),
        "random same-n from test set": _stat(rand, commission),
        "bet every test-set row": _stat(eligible, commission),
    }
    tbl = pd.DataFrame(rows).T

    lines = [
        "# Costed backtest (real odds, real results)", "",
        f"- config: `{config_path}` &middot; flag threshold: predicted consensus shortens > "
        f"{thr:.1%} &middot; commission: {commission:.0%}",
        f"- stake: 1 unit on home at the best price on offer across books at that snapshot",
        f"- test-set snapshots: {len(j):,} &middot; flagged: {len(flagged):,}",
        "", "```", tbl.to_string(), "```", "",
        "**roi_pct** is mean return per unit staked, net of commission (if any). "
        "**gross_roi_pct_no_commission** ignores commission -- for fixed-odds books "
        "this is the number that matters, since the bookmaker's margin is already "
        "baked into the odds. If 'model-flagged' roi_pct is not clearly above "
        "'random same-n', there is no betting edge here once real prices and results "
        "replace the consensus proxy.", "",
        cfg.get("data_note", ""),
    ]
    out_name = out_name or ("costed_backtest_fd.md" if "fd" in config_path else "costed_backtest.md")
    resolve(cfg["results_dir"]).mkdir(exist_ok=True)
    (resolve(cfg["results_dir"]) / out_name).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/consensus.yaml")
    ap.add_argument("--thr", type=float, default=0.005)
    ap.add_argument("--commission", type=float, default=0.0)
    a = ap.parse_args()
    run(a.config, a.thr, a.commission)
