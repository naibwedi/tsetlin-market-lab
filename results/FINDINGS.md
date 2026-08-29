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
| **Tsetlin Machine** (600 clauses, 120k-row subsample, Colab T4) | **0.742** | 0.196 | 0.211 |

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

### Tsetlin Machine: ties, and the clauses need work

**AUC 0.742** — a tie with the decision tree (0.740), a hair behind logistic/GBM.
Same verdict as synthetic: it matches the black boxes, doesn't beat them, so its
rules come "for free" — *if* the rules are good.

They are not, yet. The dominant learned clauses are single-literal, per-book:
`IF book_is_betway -> MOVE`, `IF book_is_jetbull -> MOVE`, ... The machine mostly
learned *which books are twitchier than others* — true, but not insight.

The lead/lag clauses we actually want do appear, just rarely:

```
IF ref_marathonbet_moved_up_last                          -> MOVE
IF ref_betfair_ex_eu_moved_last AND NOT book_is_188bet
                               AND NOT book_is_betfred_uk  -> MOVE
IF n_books_moved_prev_0                                    -> MOVE  (someone breaks the silence)
```

---

## v0.2 — feature rework + a backtest (EPL, same 433 matches)

### Feature rework (`book_identity: sharp`)

Dropped the per-book one-hot literals for soft books (kept a coarse
`thisbook_is_sharp/soft` split + one-hots for the 4 sharp reference books), and
added directional lead/lag literals (`thisbook_lags_pinnacle_up`,
`any_sharp_moved_last`, …), "offside" literals (dear side of consensus while the
consensus drifts away), and a `max_included_literals=5` budget on the TM.

| Feature set | XGBoost | Logistic | Decision tree |
|---|--:|--:|--:|
| v0.1 (all book one-hots) | 0.765 | 0.761 | 0.740 |
| v0.2 (sharp identity only) | 0.750 | 0.747 | 0.741 |

**The per-book base rate is worth ~1 AUC point** — a real effect (some books
*are* twitchier), but not the mechanism. Trading it away for clauses that are
about lead/lag rather than "book X exists" is the right call for the research
goal. (TM on v0.2 features: run in Colab, notebook picks up the new config.)

### Closing-line-value backtest (`src/backtest/clv.py`)

Universe: test rows where a book's home odds are ≥1% longer than consensus
(a value candidate) — **18,064 opportunities across 88 matches**. Outcome:
`closing_consensus_p_home / this_book_p_home_now − 1` (>0 = we beat the close).

| slice | n | mean CLV % | win rate |
|---|--:|--:|--:|
| model-flagged (top 25% P(move)) | 4,546 | **+4.19** | 0.92 |
| rest of universe | 13,518 | +4.55 | 0.90 |
| random same-size sample | 4,546 | +4.48 | 0.90 |

**The model does not help the bet.** The edge is in the *offside* condition
itself (~+4.5% CLV, 90% win) — being a lagging book is the value. "Will move"
does not beat random for picking which offside books to back; its picks convert
slightly *worse*.

**Why:** the target is "moves", not "moves favourably". A book flagged as
volatile can move either way. **v0.3: switch to a directional target** — "will
this book's home odds *shorten toward* consensus" — and re-run the backtest.
(The +4.5% raw number is inflated by the proxy-truth / no-commission caveats;
treat it as directional, not a P&L.)

### Caveats

- Data is 2015–16 and hourly. A modern sub-hour feed may look different.
- Bookmaker index → name mapping (`BOOKS` in `btb.py`) assumes the paper's column
  order; spot-check before trusting book-specific clauses.

### Next (v0.3)

- **Directional target** (shorten-toward-consensus) + re-run the backtest.
- Multi-league bake-off (`btb.py --leagues top` pulls 8 European leagues).
- Tsetlin on v0.2 features — did the clauses get better?
- Roadmap P3+ (GraphTM lead/lag graph, news signals, paper trading).
