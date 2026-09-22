"""Per-league closing-consensus forecast on modern football-data.co.uk seasons.

Beat The Bookie (2015-16 hourly) found Netherlands and Portugal markedly more
predictable than the big-5 leagues for "which book moves next". This checks
whether that thinner-market pattern also shows up in the *consensus forecast*
signal on modern data (open->close only, no intraday).

Run: python -m scripts.per_league_fd
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import load_yaml, resolve
from src.features.consensus import build
from src.models.consensus_forecast import _scores, _split

DIV_NAME = {
    "fd_E0": "england", "fd_SP1": "spain", "fd_D1": "germany",
    "fd_I1": "italy", "fd_F1": "france", "fd_N1": "netherlands", "fd_P1": "portugal",
}


def run(config_path: str = "config/consensus.fd.yaml") -> None:
    cfg = load_yaml(config_path)
    panel = pd.read_parquet(resolve(cfg["panel_path"]))

    rows = []
    for sport, name in DIV_NAME.items():
        sub = panel[panel["sport"] == sport]
        if sub.empty:
            continue
        m = build(sub, cfg.get("min_hours", 3.0), cfg.get("horizon_hours"))
        if len(m) < 50:
            rows.append({"league": name, "matches": m["match_id"].nunique(), "rows": len(m),
                         "note": "too few rows, skipped"})
            continue
        feat = [c for c in m.columns if c not in
                ("match_id", "snapshot_ts", "y", "mtk", "closing_cons_home", "fwd_cons")]
        tr, te = _split(m, cfg)
        if te.sum() < 20:
            rows.append({"league": name, "matches": m["match_id"].nunique(), "rows": len(m),
                         "note": "too few test rows, skipped"})
            continue
        X = m[feat].to_numpy()
        y = m["y"].to_numpy()
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        sharp_dev_te = m["sharp_mean_dev"].to_numpy()[te]

        no_change = _scores(yte, np.zeros_like(yte))
        toward_sharp = _scores(yte, sharp_dev_te)
        try:
            from xgboost import XGBRegressor
            xgb = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
            xgb.fit(Xtr, ytr)
            xgb_sc = _scores(yte, xgb.predict(Xte))
        except Exception:  # noqa: BLE001
            xgb_sc = {"rmse": None, "dir_acc_on_moves": None, "spearman": None}

        rows.append({
            "league": name, "matches": m["match_id"].nunique(), "rows": len(m),
            "test_rows": int(te.sum()), "y_std": round(float(y.std()), 4),
            "no_change_rmse": no_change["rmse"],
            "toward_sharp_dir_acc": toward_sharp["dir_acc_on_moves"],
            "toward_sharp_spearman": toward_sharp["spearman"],
            "xgb_dir_acc": xgb_sc["dir_acc_on_moves"],
            "xgb_spearman": xgb_sc["spearman"],
        })

    tbl = pd.DataFrame(rows)
    lines = ["# Per-league closing-consensus forecast (football-data.co.uk, modern seasons)", "",
             "Same forecast as [consensus_forecast_fd.md](consensus_forecast_fd.md), split out "
             "by league. Checks whether the Beat The Bookie finding that thinner markets "
             "(Netherlands, Portugal) are more predictable also holds for the consensus-forecast "
             "signal on modern open->close data.", "",
             "```", tbl.to_string(index=False), "```", "",
             "**dir_acc**: of moves > 0.5%, how often the sign was called right (0.5 = coin flip). "
             "**spearman**: rank correlation between predicted and realised drift.", "",
             "_football-data.co.uk, 2021/22-2026/27, open (approx capture) -> close only._"]
    out = resolve(cfg["results_dir"]) / "per_league_fd.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    run()
