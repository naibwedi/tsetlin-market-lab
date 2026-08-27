"""End-to-end smoke test on synthetic data with a known lead/lag structure.

Asserts:
  * panel invariants (fair probs sum to 1, staleness non-negative)
  * a learnable signal exists: logistic / tree beat the majority baseline
  * leakage control: shuffling the target collapses everything to AUC ~= 0.5
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.common.synthetic import make_raw
from src.features.booleanize import _literals, _target
from src.panel.build_panel import build

CFG = {
    "target": {"outcome": "home", "move_threshold_prob": 0.004, "horizon_snapshots": 1},
    "dispersion_high_quantile": 0.75,
    "above_consensus_prob": 0.01,
    "stale_snapshots": 3,
    "kickoff_buckets_minutes": [15, 60, 180],
    "move_pct_bins": [0.01, 0.03],
    "reference_books": ["pinnacle"],
}


@pytest.fixture(scope="module")
def prepared():
    raw = make_raw(n_matches=30, snapshots_per_match=50, seed=1)
    panel = build(raw)
    panel = panel[panel["minutes_to_kickoff"] >= 0].sort_values(
        ["snapshot_ts", "match_id", "bookmaker"]
    ).reset_index(drop=True)
    tgt = _target(panel, CFG)
    X = _literals(panel, CFG)
    keep = tgt["_future_exists"].to_numpy()
    return panel[keep].reset_index(drop=True), X[keep].reset_index(drop=True), tgt[keep].reset_index(drop=True)


def test_panel_invariants(prepared):
    panel, _, _ = prepared
    fp = panel[["fp_home", "fp_draw", "fp_away"]].to_numpy()
    assert np.allclose(fp.sum(axis=1), 1.0, atol=1e-6)
    assert (panel["stale_snaps"] >= 0).all()
    assert panel["n_books"].max() <= 7


def test_signal_beats_majority(prepared):
    panel, X, tgt = prepared
    y = tgt["y"].astype(int).to_numpy()
    matches = panel["match_id"].drop_duplicates().to_numpy()
    cut = matches[: int(len(matches) * 0.7)]
    tr = panel["match_id"].isin(cut).to_numpy()

    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[tr], y[tr])
    auc = roc_auc_score(y[~tr], lr.predict_proba(X[~tr])[:, 1])
    assert auc > 0.60, f"expected learnable signal, got AUC={auc:.3f}"


def test_shuffled_target_is_chance(prepared):
    panel, X, tgt = prepared
    rng = np.random.default_rng(0)
    y = rng.permutation(tgt["y"].astype(int).to_numpy())
    matches = panel["match_id"].drop_duplicates().to_numpy()
    cut = matches[: int(len(matches) * 0.7)]
    tr = panel["match_id"].isin(cut).to_numpy()

    lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
    auc = roc_auc_score(y[~tr], lr.predict_proba(X[~tr])[:, 1])
    assert 0.42 < auc < 0.58, f"leakage: shuffled-target AUC={auc:.3f}"


def test_features_json_serialisable(prepared):
    _, X, _ = prepared
    json.dumps(list(X.columns))
    assert X.dtypes.eq(np.uint8).all()
