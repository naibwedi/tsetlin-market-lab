# Findings

## v0.1 — real data (Beat The Bookie, 433 EPL matches, 2015–16)

**Question:** from the multi-bookmaker state, predict whether a given book changes
its price in the next hour.

**Data:** `src/ingest/btb.py` — 433 EPL matches, 29 bookmakers, 72 hourly
snapshots/match. 746k rows, 57 boolean literals. Base rate (a book moves next
hour): **7.1%**.

**Split:** time-ordered by kickoff, matches never straddle the boundary
(train 60% / val 20% / test 20% = 142k test rows).

### Leaderboard (ROC-AUC on the held-out test set)

| Model | ROC-AUC | PR-AUC | Precision@10% |
|---|--:|--:|--:|
| XGBoost | **0.765** | 0.221 | 0.230 |
| LightGBM | 0.763 | 0.218 | 0.231 |
| Logistic regression | 0.761 | 0.215 | 0.229 |
| Decision tree (d=5) | 0.740 | 0.178 | 0.209 |
| Random forest | 0.667 | 0.148 | 0.173 |
| "moved last snapshot" rule | 0.608 | 0.094 | 0.152 |
| Majority / coin flip | 0.500 | 0.062 | 0.056 |
| Tsetlin Machine | _pending (Colab GPU)_ | | |

### Leakage control

Shuffling the target collapses logistic regression to **AUC 0.502**. The 0.76 is
signal, not leakage.

### Read

1. **The core hypothesis holds on real data.** The multi-book state genuinely
   predicts which book moves next hour — XGBoost 0.765 vs 0.500 chance, and vs
   0.608 for naive persistence.
2. **~3.3× lift where it matters.** Of the books the model is most confident
   about (top 10%), 23% actually move, against a 7% base rate.
3. **Real books have real momentum** — "moved last" jumped from 0.55 (synthetic)
   to 0.61 (real) — but the learned models beat it by ~15 AUC points, so the
   signal is much richer than "it moved, so it'll move again".
4. Weaker than synthetic (0.765 vs ~0.78) as expected — real markets are noisier
   and stickier.

### Caveats

- Data is 2015–16 and hourly (not sub-hour). A modern, faster snapshot cadence
  (our live collector, or a paid feed) may show a different / stronger signal.
- EPL only so far. `btb.py` can pull any of ~550 leagues.
- Bookmaker index → name mapping (`BOOKS` in `btb.py`) assumes the paper's column
  order; spot-check before trusting book-specific clauses.

### Next

- Tsetlin Machine on this exact split (Colab) → its AUC + the rules it learns.
- More leagues; per-league models.
- Roadmap P2+ (which book moves *first*, GraphTM, news, backtest).
