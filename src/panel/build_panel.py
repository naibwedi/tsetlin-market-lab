"""Reshape raw (snapshot x match x book x outcome) rows into an analysis panel.

Panel grain: one row per (snapshot_ts, match_id, bookmaker).
Columns:
    raw implied probs           : ip_home / ip_draw / ip_away
    vig-removed fair probs       : fp_home / fp_draw / fp_away
    book margin                  : overround
    timing                       : minutes_to_kickoff
    consensus (median of books)  : cons_fp_home / cons_fp_draw / cons_fp_away
    dispersion across books       : disp_fp_home (IQR)
    deviation from consensus      : dev_fp_home = fp_home - cons_fp_home
    move vs previous snapshot     : d_fp_home, moved (bool), move_dir (-1/0/1),
                                    move_rel (relative odds change)
    staleness                     : stale_snaps (snapshots since this book moved)
    market activity               : n_books, n_books_moved_prev
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common.config import load_yaml, resolve
from src.common.odds import implied_prob, overround, remove_vig_proportional

_OUTCOME_MAP = {"home": ["Home"], "draw": ["Draw"], "away": ["Away"]}
MOVE_EPS = 1e-4


def _pivot_outcomes(raw: pd.DataFrame) -> pd.DataFrame:
    """Wide by outcome: one row per (snapshot_ts, match_id, bookmaker)."""
    raw = raw.copy()
    name = raw["outcome_name"].str.lower()
    # Odds API uses team names for Home/Away; synthetic uses "Home"/"Away". Map both.
    is_draw = name.eq("draw")
    is_home = name.eq(raw["home_team"].str.lower()) | name.eq("home")
    raw["slot"] = np.where(is_draw, "draw", np.where(is_home, "home", "away"))

    keys = ["sport", "snapshot_ts", "match_id", "commence_time", "bookmaker"]
    raw = raw.drop_duplicates([*keys, "slot"], keep="last")
    wide = raw.pivot(index=keys, columns="slot", values="price").reset_index()
    wide.columns.name = None
    return wide.rename(columns={"home": "o_home", "draw": "o_draw", "away": "o_away"})


def build(raw: pd.DataFrame) -> pd.DataFrame:
    w = _pivot_outcomes(raw).dropna(subset=["o_home", "o_draw", "o_away"])
    if "result" in raw.columns:
        res = raw[["match_id", "result"]].drop_duplicates("match_id")
        w = w.merge(res, on="match_id", how="left")

    ip = np.column_stack(
        [implied_prob(w["o_home"]), implied_prob(w["o_draw"]), implied_prob(w["o_away"])]
    )
    fp = remove_vig_proportional(ip)
    w["ip_home"], w["ip_draw"], w["ip_away"] = ip[:, 0], ip[:, 1], ip[:, 2]
    w["fp_home"], w["fp_draw"], w["fp_away"] = fp[:, 0], fp[:, 1], fp[:, 2]
    w["overround"] = overround(ip)

    w["snapshot_ts"] = pd.to_datetime(w["snapshot_ts"], utc=True)
    w["commence_time"] = pd.to_datetime(w["commence_time"], utc=True)
    w["minutes_to_kickoff"] = (
        (w["commence_time"] - w["snapshot_ts"]).dt.total_seconds() / 60.0
    )
    w = w.sort_values(["match_id", "bookmaker", "snapshot_ts"]).reset_index(drop=True)

    # --- per-book move vs previous snapshot ------------------------------------
    g = w.groupby(["match_id", "bookmaker"], sort=False)
    for slot in ("home", "draw", "away"):
        w[f"d_fp_{slot}"] = g[f"fp_{slot}"].diff()
    w["move_rel"] = g["o_home"].pct_change()
    w["moved"] = w["d_fp_home"].abs().fillna(0) > MOVE_EPS
    w["move_dir"] = np.sign(w["d_fp_home"].fillna(0)).astype(int)

    # staleness: snapshots since this book last moved
    w["stale_snaps"] = _stale_counter(w)

    # --- cross-book consensus / dispersion per (match, snapshot) --------------
    snap = w.groupby(["match_id", "snapshot_ts"], sort=False)
    for slot in ("home", "draw", "away"):
        w[f"cons_fp_{slot}"] = snap[f"fp_{slot}"].transform("median")
        w[f"dev_fp_{slot}"] = w[f"fp_{slot}"] - w[f"cons_fp_{slot}"]
    w["disp_fp_home"] = snap["fp_home"].transform(lambda s: s.quantile(0.75) - s.quantile(0.25))
    w["n_books"] = snap["bookmaker"].transform("count")
    w["n_books_moved_prev"] = snap["moved"].transform("sum")

    # how the consensus itself moved vs the previous snapshot (per match)
    cons = (w[["match_id", "snapshot_ts", "cons_fp_home"]]
            .drop_duplicates(["match_id", "snapshot_ts"])
            .sort_values(["match_id", "snapshot_ts"]))
    cons["cons_d_fp_home"] = cons.groupby("match_id", sort=False)["cons_fp_home"].diff()
    w = w.merge(cons[["match_id", "snapshot_ts", "cons_d_fp_home"]],
                on=["match_id", "snapshot_ts"], how="left")
    w["cons_move_dir"] = np.sign(w["cons_d_fp_home"].fillna(0)).astype(int)

    return w


def _stale_counter(w: pd.DataFrame) -> pd.Series:
    """Rows since this (match, bookmaker) last moved. Vectorised: for each row,
    position minus the position of the last 'reset' (group start or a move)."""
    n = len(w)
    grp = w.groupby(["match_id", "bookmaker"], sort=False).ngroup().to_numpy()
    moved = w["moved"].to_numpy(dtype=bool)
    pos = np.arange(n)
    reset = moved.copy()
    reset[0] = True
    reset[1:] |= grp[1:] != grp[:-1]
    last_reset = np.maximum.accumulate(np.where(reset, pos, -1))
    return pd.Series(pos - last_reset, index=w.index)


def run(config_path: str = "config/features.yaml") -> pd.DataFrame:
    cfg = load_yaml(config_path)
    # `raw_glob` (in features.yaml) picks which raw sources to use; default = all.
    # e.g. "data/raw/btb_long/*.parquet" to analyse only Beat-The-Bookie.
    pattern = cfg.get("raw_glob", "data/raw/**/*.parquet")
    paths = sorted(resolve(".").glob(pattern))
    frames = [pd.read_parquet(p) for p in paths]
    if not frames:
        raise SystemExit(f"no raw parquet matching {pattern}")
    print(f"panel sources: {len(paths)} files matching {pattern}")
    panel = build(pd.concat(frames, ignore_index=True))
    out = resolve(cfg["panel_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    print(f"panel -> {out}  rows={len(panel):,}  matches={panel.match_id.nunique()}")
    return panel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/features.yaml")
    run(ap.parse_args().config)
