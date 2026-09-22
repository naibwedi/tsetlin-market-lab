"""Does pooling BTB (2015-16, hourly) with football-data.co.uk (modern, open/close
only) into one training panel help the "which book moves next" classifier, or does
the resolution mismatch between the two sources dominate?

Builds data/panel_combined.parquet + data/features_combined/ (via
config/features.combined.yaml, which adds an `is_modern_data` literal), then runs
the same baseline models bakeoff.py uses -- as a standalone script so it doesn't
touch bakeoff.py or overwrite results/summary.md (another session's work).

Run: python -m src.panel.build_panel --config config/features.combined.yaml
     python -m src.features.booleanize --config config/features.combined.yaml
     python -m scripts.combined_era_bakeoff
"""
from __future__ import annotations

import pandas as pd

from src.common.config import resolve
from src.models.bakeoff import baseline_models, load_split, score


def run(config_path: str = "config/bakeoff.ci.yaml",
        features_config: str = "config/features.combined.yaml") -> None:
    import yaml
    cfg = yaml.safe_load(open(resolve(config_path)))
    cfg["features_dir"] = "data/features_combined"
    sp = load_split(cfg)

    era_ix = sp.feat.index("is_modern_data") if "is_modern_data" in sp.feat else None
    rows = []
    models = baseline_models(cfg, seed=0)
    for name in ["logistic", "decision_tree", "random_forest"]:
        model = models[name]
        model.fit(sp.X[sp.tr | sp.va], sp.y[sp.tr | sp.va])
        proba = model.predict_proba(sp.X[sp.te])[:, 1] if hasattr(model, "predict_proba") \
            else model.decision_function(sp.X[sp.te])
        rows.append({"model": name, **score(sp.y[sp.te], proba, 0.1)})

    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                        eval_metric="logloss")
    xgb.fit(sp.X[sp.tr | sp.va], sp.y[sp.tr | sp.va])
    proba_xgb = xgb.predict_proba(sp.X[sp.te])[:, 1]
    rows.append({"model": "xgboost", **score(sp.y[sp.te], proba_xgb, 0.1)})

    # xgboost fit again, WITHOUT the era literal, for comparison
    if era_ix is not None:
        keep = [i for i in range(len(sp.feat)) if i != era_ix]
        xgb2 = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             eval_metric="logloss")
        xgb2.fit(sp.X[sp.tr | sp.va][:, keep], sp.y[sp.tr | sp.va])
        proba_xgb2 = xgb2.predict_proba(sp.X[sp.te][:, keep])[:, 1]
        rows.append({"model": "xgboost (no is_modern_data)", **score(sp.y[sp.te], proba_xgb2, 0.1)})
        era_importance = float(xgb.feature_importances_[era_ix])
        era_rank = int((-xgb.feature_importances_).argsort().tolist().index(era_ix)) + 1
    else:
        era_importance, era_rank = None, None

    # split-by-era: how well does the SAME model do on each source alone?
    is_modern_te = sp.X[sp.te][:, era_ix].astype(bool) if era_ix is not None else None
    by_era = {}
    if is_modern_te is not None:
        for label, mask in [("btb only", ~is_modern_te), ("modern only", is_modern_te)]:
            if mask.sum() > 20:
                by_era[label] = score(sp.y[sp.te][mask], proba_xgb[mask], 0.1)

    tbl = pd.DataFrame(rows).set_index("model").round(4)
    by_era_tbl = pd.DataFrame(by_era).T.round(4) if by_era else None

    lines = [
        "# Combined-panel bake-off: does pooling eras help?", "",
        "- combined panel: BTB EPL (2015-16, hourly) + football-data.co.uk "
        "(2021/22-2026/27, 7 leagues, open/close only)",
        f"- rows: {len(sp.y):,}  test: {int(sp.te.sum()):,}  |  positive rate: {sp.y.mean():.3f}  "
        f"|  test positive rate: {sp.y[sp.te].mean():.3f}",
        f"- literals: {len(sp.feat)}", "",
        "```", tbl.to_string(), "```", "",
    ]
    if era_importance is not None:
        lines += [
            f"**`is_modern_data` XGBoost importance:** {era_importance:.4f} "
            f"(rank {era_rank} of {len(sp.feat)} literals).", "",
        ]
    if by_era_tbl is not None:
        lines += [
            "Same xgboost model, scored separately on each source's test rows. Note: "
            "the time-grouped split puts all of 2015-16 BTB in train (it is chronologically "
            "first), so the test set here is 100% modern data -- this table is really just "
            "confirming that combined-panel test performance is identical to the modern-only "
            "run, not a genuine per-source comparison:",
            "", "```", by_era_tbl.to_string(), "```", "",
        ]
    lines += [
        "**Reading.** Compare `xgboost` (era literal included) to "
        "`xgboost (no is_modern_data)`: if dropping the era literal costs several "
        "AUC points, the model is substantially leaning on telling the two sources "
        "apart rather than on a shared microstructure signal -- expected here, since "
        "BTB's target is an hourly move and football-data's target is a "
        "multi-day-to-kickoff move (opening capture to closing), a very different "
        "base rate for the same threshold. Pooling adds rows but not comparable rows.",
        "",
        "_Compare to results/FINDINGS.md v0.1 (BTB-only, AUC 0.765) and "
        "results/per_league_fd.md (modern-only) for the un-pooled baselines._",
    ]
    resolve("results").mkdir(exist_ok=True)
    resolve("results/combined_era.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    run()
