"""Real Tsetlin run (tmu) on the built features.

Runs as its OWN process so a notebook kernel with a half-downgraded numpy can't
break it. Writes results/tm_result.json and results/tm_clauses.txt, and prints
progress to stdout.

    python -m scripts.tm_run                 # auto: CUDA if available else CPU
    python -m scripts.tm_run --cpu           # force CPU
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

logging.disable(logging.CRITICAL)
from tmu.models.classification.vanilla_classifier import TMClassifier  # noqa: E402

from src.common.config import load_yaml, resolve  # noqa: E402
from src.models.bakeoff import load_split  # noqa: E402

# CUDA is fast; if we fall back to CPU, do much less so it still finishes.
CLAUSES, T, S, EPOCHS, EVAL_EVERY = 600, 32, 5.0, 15, 3
CPU_CLAUSES, CPU_EPOCHS = 250, 10
MAX_LITERALS = 5   # keep clauses short and readable
# TM epoch time scales with rows; cap the training set so a run finishes in ~10 min.
MAX_TRAIN_ROWS = 120_000


def _build(cpu: bool):
    if not cpu:
        try:
            m = TMClassifier(number_of_clauses=CLAUSES, T=T, s=S, weighted_clauses=True,
                             max_included_literals=MAX_LITERALS, platform="CUDA", seed=0)
            return m, "CUDA", CLAUSES, EPOCHS
        except Exception as e:  # noqa: BLE001
            print(f"CUDA unavailable ({e}); using CPU", flush=True)
    m = TMClassifier(number_of_clauses=CPU_CLAUSES, T=T, s=S, weighted_clauses=True,
                     max_included_literals=MAX_LITERALS, seed=0)
    return m, "CPU", CPU_CLAUSES, CPU_EPOCHS


def _clauses(tm, feat: list[str], n_clauses: int, limit: int = 80) -> list[str]:
    """Read each clause's included literals.

    tmu layout: TMClassifier.get_ta_action(clause, ta) with ta in [0, 2n):
    ta<n -> positive literal ta, ta>=n -> negated literal (ta-n). The first
    half of the clauses are positive-polarity (vote 'move'), the rest negative.
    The CUDA clause bank throws IndexError on the very last clause - skip it.
    """
    n = len(feat)
    out: list[str] = []
    half = n_clauses // 2
    for c in range(n_clauses):
        try:
            lits = [feat[k] for k in range(n) if tm.get_ta_action(c, k)]
            lits += ["NOT " + feat[k] for k in range(n) if tm.get_ta_action(c, n + k)]
        except Exception:  # noqa: BLE001 - tmu CUDA edge cases
            continue
        if 0 < len(lits) <= 6:
            verdict = "MOVE" if c < half else "NO-MOVE"
            out.append(f"[{verdict}] IF " + " AND ".join(lits))
    return out[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--clauses", type=int, default=0, help="override clause count")
    ap.add_argument("--epochs", type=int, default=0, help="override epoch count")
    args = ap.parse_args()
    cpu = args.cpu

    resolve("results").mkdir(exist_ok=True)
    sp = load_split(load_yaml("config/bakeoff.yaml"))
    Xtr = sp.X[sp.tr | sp.va].astype(np.uint32)
    ytr = sp.y[sp.tr | sp.va].astype(np.uint32)
    Xte = sp.X[sp.te].astype(np.uint32)
    yte = sp.y[sp.te].astype(int)

    if len(Xtr) > MAX_TRAIN_ROWS:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(Xtr), MAX_TRAIN_ROWS, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
        print(f"subsampled train to {MAX_TRAIN_ROWS} rows", flush=True)
    print(f"train={len(Xtr)} test={len(Xte)} literals={Xtr.shape[1]} "
          f"pos_rate={yte.mean():.3f}", flush=True)

    tm, backend, n_clauses, n_epochs = _build(cpu)
    if args.clauses:
        n_clauses = args.clauses
        tm = TMClassifier(number_of_clauses=n_clauses, T=T, s=S, weighted_clauses=True,
                          max_included_literals=MAX_LITERALS,
                          platform=("CUDA" if backend == "CUDA" else "CPU"), seed=0)
    if args.epochs:
        n_epochs = args.epochs
    print(f"backend: {backend}  clauses={n_clauses}  epochs={n_epochs}  "
          f"(first fit compiles kernels - ~30-60s)", flush=True)

    # during training, evaluate on a small slice of the test set (cheap); full test at the end
    ev = slice(None) if len(Xte) <= 40000 else np.random.default_rng(1).choice(len(Xte), 40000, replace=False)

    def _proba(X):
        _, cs = tm.predict(X, return_class_sums=True)
        cs = np.asarray(cs, dtype=float)
        return 1 / (1 + np.exp(-(cs[:, 1] - cs[:, 0]) / T))

    t0 = time.time()
    for e in range(n_epochs):
        tm.fit(Xtr, ytr)
        if (e + 1) % EVAL_EVERY == 0 or e == 0:
            p = _proba(Xte[ev])
            print(f"epoch {e+1:2d}/{n_epochs}  AUC={roc_auc_score(yte[ev], p):.3f}  "
                  f"PR-AUC={average_precision_score(yte[ev], p):.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    proba = _proba(Xte)
    order = np.argsort(-proba)
    k = max(1, int(len(proba) * 0.1))
    result = {
        "engine": "tmu", "backend": backend, "clauses": n_clauses, "T": T, "s": S, "epochs": n_epochs,
        "roc_auc": float(roc_auc_score(yte, proba)),
        "pr_auc": float(average_precision_score(yte, proba)),
        "f1": float(f1_score(yte, (proba >= 0.5).astype(int), zero_division=0)),
        "brier": float(brier_score_loss(yte, np.clip(proba, 0, 1))),
        "precision_at_k": float(yte[order[:k]].mean()),
        "positive_rate": float(yte.mean()),
        "train_seconds": round(time.time() - t0, 1),
    }
    resolve("results/tm_result.json").write_text(json.dumps(result, indent=2))
    print("\n=== Tsetlin Machine ===\n" + json.dumps(result, indent=2), flush=True)

    try:
        cl = _clauses(tm, sp.feat, n_clauses)
        resolve("results/tm_clauses.txt").write_text("\n".join(cl))
        print(f"\n{len(cl)} short clauses -> results/tm_clauses.txt", flush=True)
        for line in cl[:15]:
            print("  " + line, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"clause extraction failed: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
