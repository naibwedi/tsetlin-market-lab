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
*are* twitchier), but not the mechanism.

**Tsetlin on v0.2 features (Colab T4): AUC 0.713** — now *below* the baselines
(0.74–0.75). The `max_included_literals=5` budget cost accuracy and made training
oscillate (epoch AUCs 0.72 → 0.59 → 0.71). **But the rulebook is now real:**

```
IF ref_bet365_moved_last                       -> MOVE   (x10 - strongest signal)
IF ref_betfair_ex_eu_moved_up_last             -> MOVE
IF ref_williamhill_moved_last                  -> MOVE
IF thisbook_lags_williamhill                   -> MOVE
IF thisbook_lags_bet365_up                     -> MOVE
IF n_books_moved_prev_0                        -> MOVE   (x6 - quiet hour precedes a move)
IF thisbook_offside_high                       -> MOVE
IF NOT kickoff_gt_180m                         -> MOVE   (movement concentrates near kickoff)
IF thisbook_very_stale AND n_books_moved_prev_ge_3  -> MOVE
IF thisbook_lags_any_sharp AND thisbook_is_soft AND NOT (absmove_3pct | offside_high | betfair_up) -> MOVE
```

**This is the v0.2 win:** bet365 / Betfair / William Hill lead; soft books lag
them; a quiet hour tends to break; movement clusters near kickoff. That reads
like an actual description of the market. The ~3-point accuracy cost (TM 0.742 →
0.713) bought it. `max_included_literals` and `T`/`s` want tuning to recover
accuracy without losing the readability.

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

---

## v0.3 — directional target + a better backtest metric

### Directional target (`target.mode: converges`)

New label: does the book move ≥ threshold **toward consensus** over the next
hour? (rows already at consensus are dropped). 365k rows, positive rate **5.7%**.

| target | XGBoost | Logistic | naive |
|---|--:|--:|--:|
| "moves" (v0.2) | 0.749 | 0.747 | 0.608 |
| "converges" (v0.3) | 0.743 | 0.741 | 0.597 |

**Predicting *favourable* movement is just as learnable as predicting movement.**

### Backtest, redone

Added a real skill metric: **converged frac** = how much of an offside book's gap
to consensus actually closed by kickoff (1.0 = fully corrected).

| slice | mean CLV % | converged frac | CLV win rate |
|---|--:|--:|--:|
| model-flagged (converges model, top 25%) | 4.15 | **0.46** | 0.91 |
| model-flagged (moves model, top 25%) | 4.19 | 0.46 | 0.92 |
| random same-size sample | 4.48 | **0.46** | 0.90 |

**Neither model helps.** Offside books close ~46% of their gap to consensus on
average — but the model (either target) cannot pick *which* ones will close more
than random. Its 0.74 accuracy is about *when books update* (near kickoff, after
sharps move), not *which mispriced prices are exploitable*.

### Where this leaves it

- **The descriptive rulebook is the deliverable.** v0.2's clauses genuinely
  describe the EPL market's lead/lag structure. That has standalone value.
- **A betting edge is not established.** Hourly resolution likely washes out the
  exploitable window; the model sees *update timing*, not *mispricing*.
- **v0.4 ideas:** sub-hourly data (the live collector, or a paid feed); reframe
  as *forecast the closing consensus from the early cross-book picture* (a
  regression, not a movement classifier); or accept the rulebook as the product.

### Caveats

- Data is 2015–16 and hourly. A modern sub-hour feed may look different.
- Bookmaker index → name mapping (`BOOKS` in `btb.py`) assumes the paper's column
  order; spot-check before trusting book-specific clauses.

---

## v0.4 — two branches

**Decision (user, 2026-08-30):** keep "which book moves next" as the *rule-discovery*
branch (not optimised for profit). New *economic* branch = **forecast the closing
consensus** + get modern sub-hourly data.

### Rule-discovery branch — per-league (`scripts/per_league.py`)

| league | matches | XGBoost AUC | naive |
|---|--:|--:|--:|
| Portugal Primeira Liga | 357 | **0.805** | 0.634 |
| Netherlands Eredivisie | 386 | 0.792 | 0.613 |
| France Ligue 1 | 442 | 0.775 | 0.622 |
| Champions League | 265 | 0.756 | 0.615 |
| Italy Serie A | 457 | 0.755 | 0.611 |
| England Premier League | 433 | 0.750 | 0.608 |
| Germany Bundesliga | 355 | 0.749 | 0.602 |
| Spain Primera Division | 344 | 0.746 | 0.618 |

**The signal travels** — every league 0.75–0.80. And the *less efficient* markets
(Portugal, Netherlands — smaller leagues) are the *most* predictable: books lag
each other more where the market is thinner.

### Economic branch — closing-consensus forecast (`src/models/consensus_forecast.py`)

Grain: one row per (match, snapshot) ≥3h before kickoff (EPL, 30k rows).
Target: `closing_consensus_p_home − consensus_now`. Beating "no change" at
predicting the close *is* closing-line value.

| model | RMSE | dir. acc (moves >0.5%) |
|---|--:|--:|
| no change (current consensus) | **0.0221** | — |
| toward sharp books | 0.0224 | 0.553 |
| Ridge | 0.0229 | 0.508 |
| XGBoost | 0.0238 | 0.539 |

On raw RMSE **the current consensus is unbeatable** — you can't predict the
*magnitude* of the drift. But direction has a modest edge, and the CLV check
shows a real (small) signal:

```
model flags "home will shorten" on 1,826 test cases:
  realised shortening   0.0091   (0.9 pp)
  random same-size      0.0051
  direction hit-rate    0.62
```

**When the model says "home will shorten", the consensus shortens ~1.8× more
than random.** That is genuine closing-line value — bet home now at the longer
price, the market comes to you. Small (0.9 pp, proxy-truth, hourly 2015 data) but
it is the first result with an economic signal, and it validates the reframe:
*forecasting the close* beats *classifying movements*.

### Next

- Re-run the consensus forecast on all 8 leagues (~240k rows) &mdash; more data,
  per-league edges.
- Sub-hourly modern data (live collector, or a paid feed) &mdash; the drift
  window is probably finer than one hour.
- Size the CLV edge properly: stake model, commission, real bet availability.
- Tune the rule-discovery TM (`max_included_literals`, `T`, `s`).
