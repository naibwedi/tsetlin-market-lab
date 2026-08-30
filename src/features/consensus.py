"""Economic branch: forecast where the market *closes*.

Grain = one row per (match, snapshot) at least `min_hours` before kickoff.
Target = closing_consensus_p_home - consensus_p_home_now  (how much further the
market will move, and which way). Beating the current consensus at predicting
the closing consensus is closing-line value.

Output -> data/consensus/{X,y,meta}.parquet
Run:  python -m src.features.consensus
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.common.config import load_yaml, resolve

SHARP = ["pinnacle", "betfair_ex_eu", "bet365", "williamhill"]


def build(panel: pd.DataFrame, min_hours: float, horizon_h: float | None) -> pd.DataFrame:
    p = panel[panel["minutes_to_kickoff"] >= 0].copy()
    p = p.sort_values(["match_id", "snapshot_ts"])

    snap = p.groupby(["match_id", "snapshot_ts"], sort=False)
    m = snap.agg(
        cons_home=("cons_fp_home", "first"),
        cons_draw=("cons_fp_draw", "first"),
        disp=("disp_fp_home", "first"),
        n_books=("bookmaker", "count"),
        n_moved_prev=("moved", "sum"),
        mtk=("minutes_to_kickoff", "first"),
        cons_move_dir=("cons_move_dir", "first"),
    ).reset_index()
    m["hours_to_kickoff"] = m["mtk"] / 60.0

    # sharp-book deviations from consensus, and their recent moves
    for b in SHARP:
        sub = (p[p["bookmaker"] == b]
               .set_index(["match_id", "snapshot_ts"])[["dev_fp_home", "moved", "move_dir"]])
        idx = pd.MultiIndex.from_frame(m[["match_id", "snapshot_ts"]])
        m[f"{b}_dev"] = idx.map(sub["dev_fp_home"].to_dict()).astype("float").fillna(0.0)
        m[f"{b}_moved"] = idx.map(sub["moved"].to_dict()).astype("float").fillna(0.0)
        m[f"{b}_dir"] = idx.map(sub["move_dir"].to_dict()).astype("float").fillna(0.0)
    m["sharp_mean_dev"] = m[[f"{b}_dev" for b in SHARP]].mean(axis=1)

    # closing consensus per match = last pre-kickoff snapshot
    closing = (m.sort_values("mtk").groupby("match_id")["cons_home"].first()
               .rename("closing_cons_home"))
    m = m.merge(closing, on="match_id")

    if horizon_h:
        g = m.sort_values("snapshot_ts").groupby("match_id", sort=False)
        # consensus `horizon_h` hours later (approx: shift by horizon steps of 1h)
        m["fwd_cons"] = g["cons_home"].shift(-int(horizon_h))
        m["y"] = m["fwd_cons"] - m["cons_home"]
    else:
        m["y"] = m["closing_cons_home"] - m["cons_home"]

    m = m[(m["hours_to_kickoff"] >= min_hours) & m["y"].notna() & (m["n_books"] >= 4)]
    return m.reset_index(drop=True)


def run(config_path: str = "config/consensus.yaml") -> None:
    cfg = load_yaml(config_path)
    panel = pd.read_parquet(resolve(cfg["panel_path"]))
    m = build(panel, cfg.get("min_hours", 3.0), cfg.get("horizon_hours"))

    feat = [c for c in m.columns if c not in
            ("match_id", "snapshot_ts", "y", "mtk", "closing_cons_home", "fwd_cons")]
    out = resolve(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    m[feat].to_parquet(out / "X.parquet", index=False)
    m[["match_id", "snapshot_ts", "y", "cons_home", "closing_cons_home",
       "sharp_mean_dev", "hours_to_kickoff"]].to_parquet(out / "meta.parquet", index=False)
    (out / "features.json").write_text(json.dumps(feat, indent=2))
    print(f"consensus features -> {out}  rows={len(m):,}  feats={len(feat)}  "
          f"y std={m['y'].std():.4f}  |y|>0.005: {(m['y'].abs() > 0.005).mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/consensus.yaml")
    run(ap.parse_args().config)
