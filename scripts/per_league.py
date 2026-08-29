"""Per-league bake-off: run the pipeline separately on each Beat-The-Bookie
league parquet and compare. Writes results/per_league.md.

    python -m scripts.per_league
"""
from __future__ import annotations

import shutil

import pandas as pd
import yaml

from src.common.config import load_yaml, resolve
from src.features import booleanize
from src.models import bakeoff
from src.panel import build_panel

LONG_DIR = resolve("data/raw/btb_long")
COLS = ["league", "matches", "rows", "pos_rate", "xgboost", "logistic",
        "decision_tree", "moved_last"]


def _one(parquet, feat_cfg: dict) -> dict:
    tmp = resolve("data/_pl_raw/src")
    if tmp.parent.exists():
        shutil.rmtree(tmp.parent)
    tmp.mkdir(parents=True)
    shutil.copy(parquet, tmp / parquet.name)

    cfg = {**feat_cfg, "raw_glob": "data/_pl_raw/src/*.parquet",
           "panel_path": "data/_pl_panel.parquet", "out_dir": "data/_pl_features"}
    (resolve("config/_pl.yaml")).write_text(yaml.safe_dump(cfg))
    build_panel.run("config/_pl.yaml")
    booleanize.run("config/_pl.yaml")

    bcfg = {**load_yaml("config/bakeoff.ci.yaml"), "features_dir": "data/_pl_features"}
    (resolve("config/_plb.yaml")).write_text(yaml.safe_dump(bcfg))
    g = bakeoff.run("config/_plb.yaml").groupby("model")["roc_auc"].mean()
    meta = pd.read_parquet(resolve("data/_pl_features/meta.parquet"))
    return {
        "league": parquet.stem, "matches": meta["match_id"].nunique(), "rows": len(meta),
        "pos_rate": round(meta["y"].mean(), 3),
        **{m: round(g.get(m, float("nan")), 3)
           for m in ("xgboost", "logistic", "decision_tree", "moved_last")},
    }


def main() -> None:
    feat_cfg = load_yaml("config/features.yaml")
    rows = []
    for p in sorted(LONG_DIR.glob("*.parquet")):
        print(f"\n=== {p.stem} ===")
        try:
            rows.append(_one(p, feat_cfg))
        except Exception as e:  # noqa: BLE001
            print(f"  skipped {p.stem}: {e}")

    lines = ["# Per-league bake-off (Beat The Bookie)", "",
             "| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in COLS) + " |")
    lines += ["", "ROC-AUC, time-split (`bakeoff.ci.yaml`). `moved_last` = naive persistence."]
    resolve("results/per_league.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
