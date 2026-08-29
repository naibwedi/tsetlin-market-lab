"""Tsetlin bake-off entry using the green-tsetlin C++ engine (fast).

Runs on the features produced by src/features/booleanize.py, on the same
time-ordered split as src/models/bakeoff.py. Writes:
    results/tm_result.json   metrics for the TM vs the split
    results/tm_clauses.txt    human-readable clauses

Meant for Linux (Codespaces / CI / Colab) - green-tsetlin builds a C++ backend.
Usage:  python -m scripts.tm_gt --clauses 800 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

from src.common.config import load_yaml, resolve
from src.models.bakeoff import load_split


def _scores_for_rows(tm, X: np.ndarray) -> np.ndarray:
    """Per-row signed vote margin (class1 - class0), sigmoid-squashed -> [0,1]."""
    predictor = tm.get_predictor()
    out = np.zeros(len(X), dtype=float)
    for i, row in enumerate(X):
        predictor.predict(row)
        v = np.asarray(predictor.get_votes(), dtype=float)
        out[i] = v[1] - v[0] if v.size >= 2 else float(v.ravel()[0])
    s = out.std() or 1.0
    return 1.0 / (1.0 + np.exp(-out / s))


def _extract_clauses(tm, feat: list[str], limit: int = 60) -> list[str]:
    try:
        c = np.asarray(tm._state.c)          # (n_clauses, 2*n_literals) int8
        w = np.asarray(tm._state.w)          # (n_clauses, n_classes) int16
    except Exception as e:  # noqa: BLE001
        return [f"(clause extraction unavailable: {type(e).__name__}: {e})"]
    n_lit = len(feat)
    lines: list[str] = []
    for ci in range(c.shape[0]):
        pos = [feat[k] for k in range(n_lit) if c[ci, k] > 0]
        neg = [f"NOT {feat[k]}" for k in range(n_lit) if c[ci, n_lit + k] > 0]
        lits = pos + neg
        if not (0 < len(lits) <= 6):
            continue
        cls = int(np.argmax(w[ci])) if w.ndim == 2 else (1 if w[ci] > 0 else 0)
        verdict = "MOVE" if cls == 1 else "NO-MOVE"
        weight = int(w[ci].max()) if w.ndim == 2 else int(w[ci])
        lines.append(f"[{verdict} w={weight}] IF " + " AND ".join(lits))
    lines.sort(key=lambda s: -int(s.split("w=")[1].split("]")[0]))
    return lines[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/bakeoff.yaml")
    ap.add_argument("--clauses", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--threshold", type=int, default=400)
    ap.add_argument("--literal-budget", type=int, default=6)
    a = ap.parse_args()

    import green_tsetlin as gt

    resolve("results").mkdir(exist_ok=True)
    n_jobs = max(1, min(4, os.cpu_count() or 1))  # green-tsetlin needs a positive int
    cfg = load_yaml(a.config)
    sp = load_split(cfg)
    Xtr = np.ascontiguousarray(sp.X[sp.tr | sp.va], dtype=np.uint8)
    ytr = np.ascontiguousarray(sp.y[sp.tr | sp.va], dtype=np.uint32)
    Xte = np.ascontiguousarray(sp.X[sp.te], dtype=np.uint8)
    yte = sp.y[sp.te].astype(int)
    n_lit = Xtr.shape[1]
    print(f"train={len(Xtr)} test={len(Xte)} literals={n_lit} pos_rate={yte.mean():.3f}", flush=True)

    tm = gt.TsetlinMachine(n_literals=n_lit, n_clauses=a.clauses, n_classes=2,
                           s=a.s, threshold=a.threshold, literal_budget=a.literal_budget)
    trainer = gt.Trainer(tm, n_epochs=a.epochs, seed=0, n_jobs=n_jobs, progress_bar=False)
    trainer.set_train_data(Xtr, ytr)
    trainer.set_eval_data(Xte, yte.astype(np.uint32))
    t0 = time.time()
    res = trainer.train()
    train_s = time.time() - t0
    print(f"trained {a.epochs} epochs in {train_s:.0f}s  best_eval_score={res.get('best_eval_score')}",
          flush=True)

    proba = _scores_for_rows(tm, Xte)
    order = np.argsort(-proba)
    k = max(1, int(len(proba) * float(cfg.get("precision_at_k_frac", 0.1))))
    result = {
        "engine": "green-tsetlin",
        "n_clauses": a.clauses, "s": a.s, "threshold": a.threshold,
        "literal_budget": a.literal_budget, "epochs": a.epochs,
        "train_seconds": round(train_s, 1),
        "eval_accuracy": float(res.get("best_eval_score", float("nan"))),
        "roc_auc": float(roc_auc_score(yte, proba)) if len(set(yte)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(yte, proba)),
        "f1": float(f1_score(yte, (proba >= 0.5).astype(int), zero_division=0)),
        "brier": float(brier_score_loss(yte, np.clip(proba, 0, 1))),
        "precision_at_k": float(yte[order[:k]].mean()),
        "positive_rate": float(yte.mean()),
    }
    resolve("results").mkdir(exist_ok=True)
    (resolve("results/tm_result.json")).write_text(json.dumps(result, indent=2))
    clauses = _extract_clauses(tm, sp.feat)
    (resolve("results/tm_clauses.txt")).write_text("\n".join(clauses))
    print(json.dumps(result, indent=2), flush=True)
    print(f"\n{len(clauses)} clauses -> results/tm_clauses.txt", flush=True)
    for line in clauses[:12]:
        print("  " + line, flush=True)


if __name__ == "__main__":
    main()
