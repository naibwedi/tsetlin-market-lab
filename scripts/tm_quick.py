"""Quick single-TM run on the already-built features, for a fast real result.
Usage:  .venv-tm/Scripts/python -m scripts.tm_quick
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

logging.disable(logging.CRITICAL)
from tmu.models.classification.vanilla_classifier import TMClassifier  # noqa: E402

from src.common.config import load_yaml, resolve  # noqa: E402
from src.models.bakeoff import load_split, score  # noqa: E402

CLAUSES, T, S, EPOCHS = 160, 16, 5.0, 8


def main() -> None:
    cfg = load_yaml("config/bakeoff.yaml")
    sp = load_split(cfg)
    Xtr = sp.X[sp.tr | sp.va].astype(np.uint32)
    ytr = sp.y[sp.tr | sp.va].astype(np.uint32)
    Xte = sp.X[sp.te].astype(np.uint32)
    yte = sp.y[sp.te].astype(int)

    tm = TMClassifier(number_of_clauses=CLAUSES, T=T, s=S, weighted_clauses=True, seed=0)
    t0 = time.time()
    proba = None
    for e in range(EPOCHS):
        tm.fit(Xtr, ytr)
        _, cs = tm.predict(Xte, return_class_sums=True)
        cs = np.asarray(cs, float)
        proba = 1 / (1 + np.exp(-(cs[:, 1] - cs[:, 0]) / T))
        line = (f"  epoch {e+1:2d}  AUC={roc_auc_score(yte, proba):.3f}  "
                f"PR-AUC={average_precision_score(yte, proba):.3f}  ({time.time()-t0:.0f}s)")
        print(line, flush=True)
        resolve("results").mkdir(exist_ok=True)
        with open(resolve("results/tm_quick.json"), "w") as fh:
            json.dump({"epoch": e + 1, "auc": roc_auc_score(yte, proba),
                       "pr_auc": average_precision_score(yte, proba),
                       "secs": time.time() - t0}, fh, indent=2)

    m = score(sp.y[sp.te], proba, float(cfg["precision_at_k_frac"]))
    print("FINAL", json.dumps({k: round(v, 4) for k, v in m.items()}))

    # top clauses
    n_lit = len(sp.feat)
    out = []
    for cls in (1, 0):
        for c in range(CLAUSES):
            lits = []
            for k in range(n_lit):
                if tm.get_ta_action(clause=c, ta=k, the_class=cls, polarity=0):
                    lits.append(sp.feat[k])
                if tm.get_ta_action(clause=c, ta=k, the_class=cls, polarity=1):
                    lits.append("NOT " + sp.feat[k])
            if 0 < len(lits) <= 5:
                out.append(("MOVE" if cls else "NO-MOVE", lits))
    resolve("results").mkdir(exist_ok=True)
    with open(resolve("results/tm_quick.json"), "w") as fh:
        json.dump({"metrics": m, "clauses": [{"verdict": v, "lits": ls} for v, ls in out[:40]]}, fh, indent=2)
    print(f"\n{len(out)} short clauses; sample:")
    for v, ls in out[:12]:
        print(f"  IF {' AND '.join(ls)}  -> {v}")


if __name__ == "__main__":
    main()
