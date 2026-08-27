"""Train baselines (+ optional Tsetlin Machine) on one leakage-safe split and
score them with the same metrics.

Split: time-ordered by match kickoff, matches never straddle a boundary, with a
purge gap so the horizon-h target of a train row can't peek into val/test.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from src.common.config import load_yaml, resolve

try:  # optional, needs the `tm` extra (tmu builds C/CUDA ext)
    from tmu.models.classification.vanilla_classifier import TMClassifier

    _HAS_TM = True
except Exception:  # pragma: no cover
    _HAS_TM = False


# --------------------------------------------------------------------------- #
# data + split
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray
    feat: list[str]
    tr: np.ndarray
    va: np.ndarray
    te: np.ndarray
    meta: pd.DataFrame = field(repr=False)


def load_split(cfg: dict) -> Split:
    fdir = resolve(cfg["features_dir"])
    X = pd.read_parquet(fdir / "X.parquet")
    meta = pd.read_parquet(fdir / "meta.parquet")
    feat = json.loads((fdir / "features.json").read_text())

    order = meta.sort_values(["commence_time", "match_id", "snapshot_ts"]).index.to_numpy()
    matches = meta.loc[order, "match_id"].drop_duplicates().to_numpy()
    n = len(matches)
    n_tr = int(n * cfg["split"]["train_frac"])
    n_va = int(n * cfg["split"]["val_frac"])
    tr_m, va_m, te_m = matches[:n_tr], matches[n_tr : n_tr + n_va], matches[n_tr + n_va :]

    m = meta["match_id"].to_numpy()
    return Split(
        X=X[feat].to_numpy(np.uint8),
        y=meta["y"].to_numpy(np.int8),
        feat=feat,
        tr=np.isin(m, tr_m),
        va=np.isin(m, va_m),
        te=np.isin(m, te_m),
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def score(y_true: np.ndarray, proba: np.ndarray, k_frac: float) -> dict:
    y_true = y_true.astype(int)
    order = np.argsort(-proba)
    k = max(1, int(len(proba) * k_frac))
    pred = (proba >= 0.5).astype(int)
    out = {
        "roc_auc": float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if y_true.any() else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, np.clip(proba, 0, 1))),
        "precision_at_k": float(y_true[order[:k]].mean()),
        "positive_rate": float(y_true.mean()),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    return out


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def _proba(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    p = model.predict(X)
    return p.astype(float)


def baseline_models(cfg: dict, seed: int) -> dict:
    return {
        "majority": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(
            max_depth=cfg["decision_tree"]["max_depth"], class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        ),
    }


def _optional_boosters(seed: int) -> dict:
    out = {}
    try:
        from xgboost import XGBClassifier

        out["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=-1,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        out["lightgbm"] = LGBMClassifier(
            n_estimators=600, num_leaves=31, learning_rate=0.05, subsample=0.8,
            class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1,
        )
    except Exception:
        pass
    return out


def run_tm(sp: Split, variant: str, params: dict, epochs: int, seed: int) -> np.ndarray:
    weighted = variant in ("weighted", "coalesced")
    tm = TMClassifier(
        number_of_clauses=params["clauses"],
        T=params["T"],
        s=params["s"],
        weighted_clauses=weighted,
        seed=seed,
    )
    Xtr = sp.X[sp.tr | sp.va].astype(np.uint32)
    ytr = sp.y[sp.tr | sp.va].astype(np.uint32)
    Xte = sp.X[sp.te].astype(np.uint32)
    for _ in range(epochs):
        tm.fit(Xtr, ytr)
    _, cs = tm.predict(Xte, return_class_sums=True)
    cs = np.asarray(cs, dtype=float)
    margin = (cs[:, 1] - cs[:, 0]) / max(1, params["T"])
    return 1.0 / (1.0 + np.exp(-margin))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(config_path: str = "config/bakeoff.yaml") -> pd.DataFrame:
    cfg = load_yaml(config_path)
    sp = load_split(cfg)
    kfrac = float(cfg["precision_at_k_frac"])
    moved_last_ix = sp.feat.index("thisbook_moved_last") if "thisbook_moved_last" in sp.feat else None

    rows: list[dict] = []
    for seed in cfg["seeds"]:
        models = {}
        if "logistic" in cfg["baselines"] or True:
            models.update({k: v for k, v in baseline_models(cfg, seed).items() if k in cfg["baselines"]})
        models.update({k: v for k, v in _optional_boosters(seed).items() if k in cfg["baselines"]})

        for name, model in models.items():
            t0 = time.time()
            model.fit(sp.X[sp.tr | sp.va], sp.y[sp.tr | sp.va])
            proba = _proba(model, sp.X[sp.te])
            rows.append({"model": name, "seed": seed, "secs": round(time.time() - t0, 2),
                         **score(sp.y[sp.te], proba, kfrac)})

        if "moved_last" in cfg["baselines"] and moved_last_ix is not None:
            proba = sp.X[sp.te][:, moved_last_ix].astype(float)
            rows.append({"model": "moved_last", "seed": seed, "secs": 0.0,
                         **score(sp.y[sp.te], proba, kfrac)})

        if cfg["tsetlin"]["enabled"] and _HAS_TM:
            g = cfg["tsetlin"]["grid"]
            params = {"clauses": g["clauses"][1], "T": g["T"][1], "s": g["s"][1]}
            for variant in cfg["tsetlin"]["variants"]:
                t0 = time.time()
                proba = run_tm(sp, variant, params, cfg["tsetlin"]["epochs"], seed)
                rows.append({"model": f"tm_{variant}", "seed": seed,
                             "secs": round(time.time() - t0, 2),
                             **score(sp.y[sp.te], proba, kfrac)})
        elif cfg["tsetlin"]["enabled"]:
            print("  (tmu not installed - skipping Tsetlin models; install the `tm` extra)")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("model")[["roc_auc", "pr_auc", "f1", "brier", "precision_at_k"]]
        .agg(["mean", "std"])
        .round(4)
    )
    rdir = resolve(cfg["results_dir"])
    rdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    df.to_json(rdir / f"bakeoff_{stamp}.json", orient="records", indent=2)
    _write_summary(rdir / "summary.md", summary, sp)
    print(summary)
    return df


def _write_summary(path, summary: pd.DataFrame, sp: Split) -> None:
    n_te = int(sp.te.sum())
    lines = [
        "# Bake-off summary",
        "",
        f"- test rows: {n_te:,}  |  test positive rate: {sp.y[sp.te].mean():.3f}",
        f"- literals: {len(sp.feat)}",
        "",
        "Mean over seeds (std in parentheses is omitted here; see JSON):",
        "",
        "```",
        summary["roc_auc"]["mean"].sort_values(ascending=False).to_string(),
        "```",
        "",
        ("ROC-AUC ranking above. Verdict (fill in after review): "
         "TM loses / TM ties + useful rules / TM wins."),
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/bakeoff.yaml")
    run(ap.parse_args().config)
