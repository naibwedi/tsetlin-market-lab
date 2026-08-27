"""Extract human-readable rules from a trained Tsetlin Machine and compare the
dominant literals against the decision-tree splits and gradient-boosting gains.

Produces results/clause_report.md - the qualitative "where/why" deliverable.
Requires the `tm` extra; if tmu is missing it still writes the tree/GBM side.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from src.common.config import load_yaml, resolve
from src.models.bakeoff import load_split

try:
    from tmu.models.classification.vanilla_classifier import TMClassifier

    _HAS_TM = True
except Exception:
    _HAS_TM = False


def _tm_clauses(sp, params, epochs, seed, top=25) -> list[str]:
    tm = TMClassifier(
        number_of_clauses=params["clauses"], T=params["T"], s=params["s"],
        weighted_clauses=True, seed=seed,
    )
    Xtr = sp.X[sp.tr | sp.va].astype(np.uint32)
    ytr = sp.y[sp.tr | sp.va].astype(np.uint32)
    for _ in range(epochs):
        tm.fit(Xtr, ytr)

    n_lit = len(sp.feat)
    out: list[str] = []
    for cls in (1, 0):
        for c in range(params["clauses"]):
            lits = []
            for k in range(n_lit):
                if tm.get_ta_action(clause=c, ta=k, the_class=cls, polarity=0):
                    lits.append(sp.feat[k])
                if tm.get_ta_action(clause=c, ta=k, the_class=cls, polarity=1):
                    lits.append(f"NOT {sp.feat[k]}")
            if lits:
                verdict = "MOVE" if cls == 1 else "NO-MOVE"
                out.append(f"IF {' AND '.join(lits)}  ->  {verdict}  [{len(lits)} lits]")
    return out[:top]


def _tree_rules(sp) -> str:
    t = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=0)
    t.fit(sp.X[sp.tr | sp.va], sp.y[sp.tr | sp.va])
    return export_text(t, feature_names=list(sp.feat))


def _gbm_importance(sp) -> pd.Series | None:
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss")
    m.fit(sp.X[sp.tr | sp.va], sp.y[sp.tr | sp.va])
    return pd.Series(m.feature_importances_, index=sp.feat).sort_values(ascending=False).head(20)


def run(config_path: str = "config/bakeoff.yaml") -> None:
    cfg = load_yaml(config_path)
    sp = load_split(cfg)
    parts = ["# Clause report", ""]

    parts += ["## Decision tree (depth 4)", "```", _tree_rules(sp), "```", ""]

    imp = _gbm_importance(sp)
    if imp is not None:
        parts += ["## XGBoost feature importance (top 20)", "```", imp.to_string(), "```", ""]

    if _HAS_TM:
        g = cfg["tsetlin"]["grid"]
        params = {"clauses": g["clauses"][1], "T": g["T"][1], "s": g["s"][1]}
        clauses = _tm_clauses(sp, params, cfg["tsetlin"]["epochs"], cfg["seeds"][0])
        parts += ["## Tsetlin Machine clauses (top 25 by presence)", "```",
                  "\n".join(clauses), "```", ""]
    else:
        parts += ["## Tsetlin Machine clauses", "_tmu not installed - run with the `tm` extra._", ""]

    parts += [
        "## Verdict (fill in)",
        "- Where does TM win / lose vs tree & GBM?",
        "- Any literal combination TM surfaced that the others did not?",
        "- Recommended next phase.",
    ]
    out = resolve(cfg["results_dir"]) / "clause_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    print(f"clause report -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/bakeoff.yaml")
    run(ap.parse_args().config)
