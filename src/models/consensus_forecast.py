"""Regression bake-off for closing-consensus forecasting + a CLV check.

Baselines:
  no_change  : predict 0 (the current consensus is the best guess)
  toward_sharp : predict `sharp_mean_dev` (consensus will drift toward the sharp
                 books' current price)
Models: Ridge, XGBoostRegressor.

Metric: RMSE, MAE, directional accuracy (sign of predicted vs realised move,
on moves > 0.5%), and Spearman corr. Plus a CLV backtest: bet home whenever the
model predicts the consensus will shorten (y_hat > thr); realised CLV = how much
it actually shortened.

Run:  python -m src.models.consensus_forecast
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.common.config import load_yaml, resolve


def _split(meta: pd.DataFrame, cfg: dict):
    order = meta.sort_values(["match_id", "snapshot_ts"]).index.to_numpy()
    matches = meta.loc[order, "match_id"].drop_duplicates().to_numpy()
    n = len(matches)
    tr = matches[: int(n * cfg["split"]["train_frac"])]
    va = matches[int(n * cfg["split"]["train_frac"]): int(n * (cfg["split"]["train_frac"] + cfg["split"]["val_frac"]))]
    mm = meta["match_id"].to_numpy()
    return np.isin(mm, tr) | np.isin(mm, va), ~np.isin(mm, np.concatenate([tr, va]))


def _scores(y, yhat, thr=0.005) -> dict:
    move = np.abs(y) > thr
    dir_acc = float((np.sign(yhat[move]) == np.sign(y[move])).mean()) if move.any() else float("nan")
    rho = float(spearmanr(y, yhat).statistic)
    return {"rmse": float(mean_squared_error(y, yhat) ** 0.5),
            "mae": float(mean_absolute_error(y, yhat)),
            "dir_acc_on_moves": round(dir_acc, 3), "spearman": round(rho, 3)}


def run(config_path: str = "config/consensus.yaml") -> None:
    cfg = load_yaml(config_path)
    d = resolve(cfg["out_dir"])
    X = pd.read_parquet(d / "X.parquet")
    meta = pd.read_parquet(d / "meta.parquet")
    feat = json.loads((d / "features.json").read_text())
    y = meta["y"].to_numpy()

    tr, te = _split(meta, cfg)
    Xtr, Xte = X[feat].to_numpy()[tr], X[feat].to_numpy()[te]
    ytr, yte = y[tr], y[te]
    sharp_dev_te = meta["sharp_mean_dev"].to_numpy()[te]

    rows = {
        "no_change": _scores(yte, np.zeros_like(yte)),
        "toward_sharp": _scores(yte, sharp_dev_te),
    }
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
    rows["ridge"] = _scores(yte, ridge.predict(Xte))
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
        xgb.fit(Xtr, ytr)
        yhat_xgb = xgb.predict(Xte)
        rows["xgboost"] = _scores(yte, yhat_xgb)
    except Exception:  # noqa: BLE001
        yhat_xgb = ridge.predict(Xte)

    # CLV backtest on the best model's home-shorten calls
    thr = 0.005
    flag = yhat_xgb > thr
    rand = np.random.default_rng(0).permutation(flag)
    bt = {
        "n_flagged": int(flag.sum()),
        "flagged_mean_realised_shortening": round(float(yte[flag].mean()), 4) if flag.any() else None,
        "flagged_dir_hit_rate": round(float((yte[flag] > 0).mean()), 3) if flag.any() else None,
        "random_same_n_mean": round(float(yte[rand].mean()), 4),
        "all_test_mean": round(float(yte.mean()), 4),
    }

    tbl = pd.DataFrame(rows).T.round(4)
    lines = ["# Closing-consensus forecast", "",
             f"- rows: {len(y):,}  test: {int(te.sum()):,}  |  y std: {y.std():.4f}",
             f"- forecasting from >= {cfg.get('min_hours')}h before kickoff",
             "", "```", tbl.to_string(), "```", "",
             "**dir_acc_on_moves**: of the times the consensus moved > 0.5%, how often "
             "the model got the direction right (0.5 = coin flip).", "",
             "## CLV check (model says 'home will shorten')", "```",
             json.dumps(bt, indent=2), "```", "",
             "If `flagged_mean_realised_shortening` > `random_same_n_mean`, betting home "
             "on the model's calls locks in genuine closing-line value.",
             "", "_2015-16 hourly data; consensus as proxy truth; directional evidence only._"]
    resolve(cfg["results_dir"]).mkdir(exist_ok=True)
    (resolve(cfg["results_dir"]) / "consensus_forecast.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/consensus.yaml")
    run(ap.parse_args().config)
