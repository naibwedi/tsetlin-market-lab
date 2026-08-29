"""Boolean feature factory + target construction.

Output (saved to ``data/features/``):
    X.parquet       uint8 literal matrix  (one row per panel row kept)
    meta.parquet    match_id, snapshot_ts, bookmaker, commence_time, y, y3
    features.json   ordered list of literal names

Row grain = (snapshot_ts, match_id, bookmaker): "will THIS book move next?"
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common.config import load_yaml, resolve

MOVE_EPS = 1e-4

# books generally regarded as sharp / market-leading
SHARP_BOOKS = {
    "pinnacle", "betfair_ex_eu", "betfair_ex_uk", "betfair_sb_uk", "sbobet",
    "bet365", "williamhill", "marathonbet", "betvictor", "matchbook", "smarkets",
}


def _target(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    slot = cfg["target"]["outcome"]
    thr = float(cfg["target"]["move_threshold_prob"])
    h = int(cfg["target"]["horizon_snapshots"])
    g = panel.sort_values("snapshot_ts").groupby(["match_id", "bookmaker"], sort=False)
    fut = g[f"fp_{slot}"].shift(-h)
    delta = fut - panel[f"fp_{slot}"]
    y = (delta.abs() >= thr).astype("Int8")
    y3 = pd.Series(np.select([delta >= thr, delta <= -thr], [2, 0], default=1), index=panel.index)
    y3[fut.isna()] = pd.NA
    return pd.DataFrame({"y": y, "y3": y3, "_future_exists": fut.notna()})


def _literals(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = panel
    lit: dict[str, np.ndarray] = {}

    disp_hi = p["disp_fp_home"].quantile(cfg["dispersion_high_quantile"])
    lit["dispersion_high"] = (p["disp_fp_home"] >= disp_hi).to_numpy()
    lit["dispersion_very_high"] = (p["disp_fp_home"] >= p["disp_fp_home"].quantile(0.9)).to_numpy()

    ac = float(cfg["above_consensus_prob"])
    lit["thisbook_above_consensus"] = (p["dev_fp_home"] >= ac).to_numpy()
    lit["thisbook_below_consensus"] = (p["dev_fp_home"] <= -ac).to_numpy()
    lit["thisbook_at_consensus"] = (p["dev_fp_home"].abs() < ac).to_numpy()

    st = int(cfg["stale_snapshots"])
    lit["thisbook_stale"] = (p["stale_snaps"] >= st).to_numpy()
    lit["thisbook_very_stale"] = (p["stale_snaps"] >= 2 * st).to_numpy()
    lit["thisbook_moved_last"] = p["moved"].to_numpy()
    lit["thisbook_moved_up_last"] = (p["move_dir"] > 0).to_numpy()
    lit["thisbook_moved_down_last"] = (p["move_dir"] < 0).to_numpy()

    for mins in cfg["kickoff_buckets_minutes"]:
        lit[f"kickoff_lt_{mins}m"] = (p["minutes_to_kickoff"] <= mins).to_numpy()
    lit["kickoff_gt_180m"] = (p["minutes_to_kickoff"] > 180).to_numpy()

    lit["n_books_moved_prev_ge_3"] = (p["n_books_moved_prev"] >= 3).to_numpy()
    lit["n_books_moved_prev_ge_6"] = (p["n_books_moved_prev"] >= 6).to_numpy()
    lit["n_books_moved_prev_0"] = (p["n_books_moved_prev"] == 0).to_numpy()

    for pct in cfg["move_pct_bins"]:
        lit[f"thisbook_absmove_ge_{int(pct*100)}pct"] = (p["move_rel"].abs() >= pct).to_numpy()

    # "offside": this book sits on the dear side of consensus AND the market just
    # drifted further from it -> a candidate to be corrected next.
    lit["thisbook_offside_high"] = (
        (p["dev_fp_home"] >= ac) & (p["cons_move_dir"] < 0)
    ).to_numpy()
    lit["thisbook_offside_low"] = (
        (p["dev_fp_home"] <= -ac) & (p["cons_move_dir"] > 0)
    ).to_numpy()
    lit["thisbook_stale_while_market_moved"] = (
        (p["stale_snaps"] >= st) & (p["n_books_moved_prev"] >= 2)
    ).to_numpy()

    # --- reference-book lead/lag literals ------------------------------------
    for col, vals in _reference_book_state(p, cfg["reference_books"]).items():
        lit[col] = vals

    # --- book identity ----------------------------------------------------------
    mode = cfg.get("book_identity", "sharp")
    is_sharp = p["bookmaker"].isin(SHARP_BOOKS).to_numpy()
    if mode != "none":
        lit["thisbook_is_sharp"] = is_sharp
        lit["thisbook_is_soft"] = ~is_sharp
    if mode == "sharp":
        for b in cfg["reference_books"]:
            lit[f"book_is_{b}"] = (p["bookmaker"] == b).to_numpy()
    elif mode == "all":
        for b in sorted(p["bookmaker"].unique()):
            lit[f"book_is_{b}"] = (p["bookmaker"] == b).to_numpy()

    X = pd.DataFrame({k: np.asarray(v).astype(np.uint8) for k, v in lit.items()}, index=p.index)
    return X


def _reference_book_state(p: pd.DataFrame, refs: list[str]) -> dict[str, np.ndarray]:
    """For each sharp reference book r, broadcast its last-snapshot move onto
    every book's row and derive lag/chase literals."""
    key = ["match_id", "snapshot_ts"]
    idx = pd.MultiIndex.from_arrays([p["match_id"], p["snapshot_ts"]])
    this_moved = p["moved"].to_numpy()
    this_dir = p["move_dir"].to_numpy()
    out: dict[str, np.ndarray] = {}
    for r in refs:
        sub = p[p["bookmaker"] == r].set_index(key)
        moved = np.nan_to_num(idx.map(sub["moved"].to_dict()).astype("float")).astype(bool)
        rdir = np.nan_to_num(idx.map(sub["move_dir"].to_dict()).astype("float"))
        out[f"ref_{r}_moved_last"] = moved
        out[f"ref_{r}_moved_up_last"] = rdir > 0
        out[f"ref_{r}_moved_down_last"] = rdir < 0
        # this book has not yet followed the reference's move
        out[f"thisbook_lags_{r}"] = moved & (~this_moved)
        out[f"thisbook_lags_{r}_up"] = (rdir > 0) & (this_dir <= 0)
        out[f"thisbook_lags_{r}_down"] = (rdir < 0) & (this_dir >= 0)
    # did ANY sharp reference move last snapshot?
    any_ref = np.zeros(len(p), dtype=bool)
    for r in refs:
        any_ref |= out[f"ref_{r}_moved_last"]
    out["any_sharp_moved_last"] = any_ref
    out["thisbook_lags_any_sharp"] = any_ref & (~this_moved)
    return out


def run(config_path: str = "config/features.yaml") -> None:
    cfg = load_yaml(config_path)
    panel = pd.read_parquet(resolve(cfg["panel_path"]))

    if cfg.get("anchor_only_pre_kickoff", True):
        panel = panel[panel["minutes_to_kickoff"] >= 0]
    panel = panel[panel["n_books"] >= int(cfg["min_books_per_snapshot"])]
    panel = panel.sort_values(["snapshot_ts", "match_id", "bookmaker"]).reset_index(drop=True)

    tgt = _target(panel, cfg)
    X = _literals(panel, cfg)

    keep = tgt["_future_exists"].to_numpy()
    X, panel, tgt = X[keep].reset_index(drop=True), panel[keep].reset_index(drop=True), tgt[keep].reset_index(drop=True)

    out = resolve(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    X.to_parquet(out / "X.parquet", index=False)
    meta = panel[["match_id", "snapshot_ts", "commence_time", "bookmaker"]].copy()
    meta["y"] = tgt["y"].astype(int).to_numpy()
    meta["y3"] = tgt["y3"].astype(int).to_numpy()
    meta.to_parquet(out / "meta.parquet", index=False)
    (out / "features.json").write_text(json.dumps(list(X.columns), indent=2))
    print(
        f"features -> {out}  rows={len(X):,}  literals={X.shape[1]}  "
        f"positive_rate={meta['y'].mean():.3f}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/features.yaml")
    run(ap.parse_args().config)
