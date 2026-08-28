"""Real Tsetlin run on the built features -> results/tm_result.json + tm_clauses.txt.
Writes progress line-by-line so it can be watched. .venv-tm/Scripts/python -m scripts.tm_run
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

logging.disable(logging.CRITICAL)
from tmu.models.classification.vanilla_classifier import TMClassifier  # noqa: E402

from src.common.config import load_yaml, resolve  # noqa: E402
from src.models.bakeoff import load_split  # noqa: E402

CLAUSES, T, S, EPOCHS = 300, 16, 5.0, 20
LOG = resolve("results/tm_progress.txt")


def w(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')}  {msg}\n")


def main() -> None:
    resolve("results").mkdir(exist_ok=True)
    open(LOG, "w").close()
    cfg = load_yaml("config/bakeoff.yaml")
    w("loading split...")
    sp = load_split(cfg)
    Xtr = sp.X[sp.tr | sp.va].astype(np.uint32)
    ytr = sp.y[sp.tr | sp.va].astype(np.uint32)
    Xte = sp.X[sp.te].astype(np.uint32)
    yte = sp.y[sp.te].astype(int)
    w(f"train={len(Xtr)} test={len(Xte)} literals={Xtr.shape[1]} pos_rate={yte.mean():.3f}")

    tm = TMClassifier(number_of_clauses=CLAUSES, T=T, s=S, weighted_clauses=True, seed=0)
    t0 = time.time()
    proba = None
    best = 0.0
    for e in range(EPOCHS):
        tm.fit(Xtr, ytr)
        _, cs = tm.predict(Xte, return_class_sums=True)
        cs = np.asarray(cs, float)
        proba = 1 / (1 + np.exp(-(cs[:, 1] - cs[:, 0]) / T))
        auc = roc_auc_score(yte, proba)
        best = max(best, auc)
        w(f"epoch {e+1:2d}/{EPOCHS}  AUC={auc:.3f}  PR-AUC={average_precision_score(yte, proba):.3f}  "
          f"({time.time()-t0:.0f}s)")

    order = np.argsort(-proba)
    k = max(1, int(len(proba) * 0.1))
    result = {
        "model": "tm_weighted", "clauses": CLAUSES, "T": T, "s": S, "epochs": EPOCHS,
        "roc_auc": float(roc_auc_score(yte, proba)),
        "pr_auc": float(average_precision_score(yte, proba)),
        "f1": float(f1_score(yte, (proba >= 0.5).astype(int), zero_division=0)),
        "brier": float(brier_score_loss(yte, np.clip(proba, 0, 1))),
        "precision_at_k": float(yte[order[:k]].mean()),
        "positive_rate": float(yte.mean()),
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(resolve("results/tm_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    w(f"FINAL {json.dumps(result)}")

    n_lit = len(sp.feat)
    lines = []
    for cls in (1, 0):
        for c in range(CLAUSES):
            pos = [sp.feat[k2] for k2 in range(n_lit)
                   if tm.get_ta_action(clause=c, ta=k2, the_class=cls, polarity=0)]
            neg = ["NOT " + sp.feat[k2] for k2 in range(n_lit)
                   if tm.get_ta_action(clause=c, ta=k2, the_class=cls, polarity=1)]
            lits = pos + neg
            if 0 < len(lits) <= 6:
                lines.append(f"[{'MOVE' if cls else 'NO-MOVE'}] IF " + " AND ".join(lits))
    resolve("results/tm_clauses.txt").write_text("\n".join(lines[:60]))
    w(f"wrote {len(lines)} short clauses")


if __name__ == "__main__":
    main()
