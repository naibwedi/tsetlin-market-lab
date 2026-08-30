"""A closing-line-value backtest for the 'which book moves next' model.

Idea: a book offering **longer odds on the home team than the market consensus**
is a value-bet candidate (if the consensus is close to the true probability).
The bet is only good if that price *converges* toward consensus before kickoff
rather than sitting mispriced or drifting away. Does the model's "this book will
move" flag pick out the candidates that actually converge?

Universe : test-set rows where the book is 'offside' on home
           (its fair p_home is >= `edge` below consensus -> longer home odds).
Signal   : model P(move) in the top `top_frac` of that universe.
Outcome  : realized closing-line value =
           closing_consensus_p_home / this_book_p_home_now - 1
           (> 0  => we locked in a better price than the closing market).

Writes results/backtest.md. Run:  python -m src.backtest.clv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.common.config import load_yaml, resolve
from src.models.bakeoff import load_split


def run(config_path: str = "config/bakeoff.ci.yaml",
        features_config: str = "config/features.yaml",
        edge: float = 0.01, top_frac: float = 0.25) -> None:
    cfg = load_yaml(config_path)
    sp = load_split(cfg)
    tr = sp.tr | sp.va

    # quick, honest probability model (same leakage-safe split as the bake-off)
    lr = LogisticRegression(max_iter=500, class_weight="balanced").fit(sp.X[tr], sp.y[tr])
    proba = lr.predict_proba(sp.X[sp.te])[:, 1]

    meta = sp.meta[sp.te].reset_index(drop=True)
    panel = pd.read_parquet(resolve(load_yaml(features_config)["panel_path"]))
    panel = panel.sort_values(["snapshot_ts", "match_id", "bookmaker"]).reset_index(drop=True)

    # align: features.booleanize keeps rows with a future snapshot, in this order
    key = ["match_id", "snapshot_ts", "bookmaker"]
    m = meta[key].copy()
    m["proba"] = proba
    j = m.merge(panel[key + ["fp_home", "cons_fp_home", "dev_fp_home",
                             "minutes_to_kickoff"]], on=key, how="left")

    pre = panel[panel["minutes_to_kickoff"] >= 0].sort_values("minutes_to_kickoff")
    # closing consensus per match, and each book's own closing fair prob
    closing = pre.groupby("match_id")["cons_fp_home"].first().rename("closing_cons_p_home")
    book_close = (pre.groupby(["match_id", "bookmaker"])["fp_home"].first()
                  .rename("book_close_p_home"))
    j = j.merge(closing, on="match_id", how="left")
    j = j.merge(book_close, on=["match_id", "bookmaker"], how="left")

    # CLV vs the closing market
    j["clv"] = j["closing_cons_p_home"] / j["fp_home"] - 1.0
    # did THIS book's price actually converge toward consensus by close?
    gap0 = (j["cons_fp_home"] - j["fp_home"]).abs()
    gap1 = (j["closing_cons_p_home"] - j["book_close_p_home"]).abs()
    j["converged_frac"] = 1.0 - (gap1 / gap0).clip(upper=1.0)

    offside = j[(j["dev_fp_home"] <= -edge) & j["clv"].notna()
                & j["converged_frac"].notna()].copy()
    if offside.empty:
        resolve("results/backtest.md").write_text("# Backtest\n\nNo offside opportunities found.\n")
        print("no offside opportunities")
        return

    thr = offside["proba"].quantile(1 - top_frac)
    flagged = offside[offside["proba"] >= thr]
    rest = offside[offside["proba"] < thr]

    def stat(d: pd.DataFrame) -> dict:
        return {"n": len(d), "mean_clv_pct": round(100 * d["clv"].mean(), 3),
                "converged_frac": round(d["converged_frac"].mean(), 3),
                "win_rate": round((d["clv"] > 0).mean(), 3)}

    rng = np.random.default_rng(0)
    rand = offside.iloc[rng.choice(len(offside), len(flagged), replace=False)]

    lines = [
        "# Closing-line-value backtest",
        "",
        f"- config: {config_path} &middot; edge={edge} &middot; top_frac={top_frac}",
        f"- offside universe: {len(offside):,} (book >= {edge:.0%} below consensus on home)",
        f"- test matches: {offside['match_id'].nunique()}",
        "",
        "| slice | n | mean CLV % | converged frac | CLV win rate |",
        "|---|--:|--:|--:|--:|",
        f"| model-flagged (top {top_frac:.0%}) | {stat(flagged)['n']:,} | "
        f"{stat(flagged)['mean_clv_pct']} | {stat(flagged)['converged_frac']} | {stat(flagged)['win_rate']} |",
        f"| rest of universe | {stat(rest)['n']:,} | {stat(rest)['mean_clv_pct']} | "
        f"{stat(rest)['converged_frac']} | {stat(rest)['win_rate']} |",
        f"| random same-size sample | {stat(rand)['n']:,} | {stat(rand)['mean_clv_pct']} | "
        f"{stat(rand)['converged_frac']} | {stat(rand)['win_rate']} |",
        "",
        "**Read:** *converged frac* is how much of an offside book's gap to consensus "
        "actually closed by kickoff (1.0 = fully corrected, 0 = didn't budge / drifted). "
        "If 'model-flagged' > 'random' there, the model really does spot the stale prices.",
        "",
        "_Caveat: hourly 2015-16 data, no stake sizing, no commission, consensus "
        "used as a proxy for true probability. Directional evidence only._",
    ]
    resolve("results/backtest.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/bakeoff.ci.yaml")
    ap.add_argument("--features-config", default="config/features.yaml")
    ap.add_argument("--edge", type=float, default=0.01)
    ap.add_argument("--top-frac", type=float, default=0.25)
    a = ap.parse_args()
    run(a.config, a.features_config, a.edge, a.top_frac)
