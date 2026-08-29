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

CLAUSES, T, S, EPOCHS, EVAL_EVERY = 1000, 32, 5.0, 40, 8


def _build(cpu: bool):
    if not cpu:
        try:
            m = TMClassifier(number_of_clauses=CLAUSES, T=T, s=S, weighted_clauses=True,
                             platform="CUDA", seed=0)
            return m, "CUDA"
        except Exception as e:  # noqa: BLE001
            print(f"CUDA unavailable ({e}); using CPU", flush=True)
    return TMClassifier(number_of_clauses=CLAUSES, T=T, s=S, weighted_clauses=True, seed=0), "CPU"


def _clauses(tm, feat: list[str], limit: int = 60) -> list[str]:
    n = len(feat)
    out: list[str] = []
    for cls in (1, 0):
        for c in range(CLAUSES):
            lits: list[str] = []
            for k in range(n):
                try:
                    if tm.get_ta_action(clause=c, ta=k, the_class=cls, polarity=0):
                        lits.append(feat[k])
                    if tm.get_ta_action(clause=c, ta=k, the_class=cls, polarity=1):
                        lits.append("NOT " + feat[k])
                except TypeError:  # older tmu: negated literals live at ta = k + n
                    if tm.get_ta_action(clause=c, ta=k, the_class=cls):
                        lits.append(feat[k])
                    if tm.get_ta_action(clause=c, ta=k + n, the_class=cls):
                        lits.append("NOT " + feat[k])
            if 0 < len(lits) <= 6:
                out.append(f"[{'MOVE' if cls else 'NO-MOVE'}] IF " + " AND ".join(lits))
    return out[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    cpu = ap.parse_args().cpu

    resolve("results").mkdir(exist_ok=True)
    sp = load_split(load_yaml("config/bakeoff.yaml"))
    Xtr = sp.X[sp.tr | sp.va].astype(np.uint32)
    ytr = sp.y[sp.tr | sp.va].astype(np.uint32)
    Xte = sp.X[sp.te].astype(np.uint32)
    yte = sp.y[sp.te].astype(int)
    print(f"train={len(Xtr)} test={len(Xte)} literals={Xtr.shape[1]} "
          f"pos_rate={yte.mean():.3f}", flush=True)

    tm, backend = _build(cpu)
    print(f"backend: {backend}  (first fit compiles kernels - ~30-60s)", flush=True)

    t0 = time.time()
    proba = None
    for e in range(EPOCHS):
        tm.fit(Xtr, ytr)
        if (e + 1) % EVAL_EVERY == 0 or e == 0:
            _, cs = tm.predict(Xte, return_class_sums=True)
            cs = np.asarray(cs, dtype=float)
            proba = 1 / (1 + np.exp(-(cs[:, 1] - cs[:, 0]) / T))
            print(f"epoch {e+1:2d}/{EPOCHS}  AUC={roc_auc_score(yte, proba):.3f}  "
                  f"PR-AUC={average_precision_score(yte, proba):.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    order = np.argsort(-proba)
    k = max(1, int(len(proba) * 0.1))
    result = {
        "engine": "tmu", "backend": backend, "clauses": CLAUSES, "T": T, "s": S, "epochs": EPOCHS,
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
        cl = _clauses(tm, sp.feat)
        resolve("results/tm_clauses.txt").write_text("\n".join(cl))
        print(f"\n{len(cl)} short clauses -> results/tm_clauses.txt", flush=True)
        for line in cl[:15]:
            print("  " + line, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"clause extraction failed: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
