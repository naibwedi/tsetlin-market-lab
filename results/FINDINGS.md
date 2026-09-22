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

> **Correction (2026-09-19 audit, see below):** the "someone breaks the silence"
> reading of `n_books_moved_prev_0 -> MOVE` is backwards. Tested directly against
> the BTB data, a quiet snapshot (nobody moved) is followed by a move only **2.6%**
> of the time, against a **7.1%** base rate (0.37x) — quiet snapshots predict *more*
> quiet, not a break. The clause fires correctly (its NO-MOVE cases are common) but
> the plain-English gloss had the direction wrong.

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
IF n_books_moved_prev_0                        -> MOVE   (x6 - see correction below)
IF thisbook_offside_high                       -> MOVE
IF NOT kickoff_gt_180m                         -> MOVE   (movement concentrates near kickoff)
IF thisbook_very_stale AND n_books_moved_prev_ge_3  -> MOVE
IF thisbook_lags_any_sharp AND thisbook_is_soft AND NOT (absmove_3pct | offside_high | betfair_up) -> MOVE
```

**This is the v0.2 win:** bet365 / Betfair / William Hill lead; soft books lag
them; movement clusters near kickoff. That reads like an actual description of
the market. The ~3-point accuracy cost (TM 0.742 → 0.713) bought it.
`max_included_literals` and `T`/`s` want tuning to recover accuracy without
losing the readability. (The "quiet hour tends to break" reading of the
`n_books_moved_prev_0` clause does **not** hold — see the correction below and
v0.8: quiet snapshots predict more quiet, the opposite of what was written here.)

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

---

## v0.5 — the same forecast on modern data (2026-09-19)

Question: is the weak closing-consensus signal just a 2015-16 artefact?

Data (`src/ingest/footballdata.py`, free, no key): football-data.co.uk, 5 leagues,
2021/22 to 2026/27, 9,110 matches, 8 books. Two snapshots per match: **opening**
(an approximated Friday/Tuesday capture, median 29 h before kickoff, range 3 to 79 h)
and **closing**. 7,937 usable rows (at least 4 books). Date split; the test set is
1,588 matches from 2025-01-19 to 2026-05-24. Full output in `consensus_forecast_fd.md`.

| model | RMSE | direction on moves >0.5% | Spearman |
|---|--:|--:|--:|
| no change | 0.0270 | n/a | n/a |
| toward sharp books | 0.0270 | 0.525 | 0.104 |
| Ridge | 0.0269 | 0.559 | 0.076 |
| XGBoost | 0.0275 | 0.550 | 0.110 |

2015-16 EPL for comparison: toward-sharp 0.553 / 0.084, XGBoost 0.539 / 0.113.

CLV check, 453 test cases flagged "home will shorten": mean realised shortening
**+0.05 pp** against **-0.41 pp** for random picks; only **46%** of flagged cases
shortened (0.62 in 2015-16).

**What replicates.** Direction skill near 0.55 and a rank correlation near 0.1, in
both decades and at very different resolution. The weak lead is not only a data-age
artefact.

**What does not.** The absolute edge. In 2015-16 the flagged set shortened by 0.9 pp
and won 62% of the time. Here it shortens by about zero and wins 46%. The +0.46 pp
gap to random (2015-16: +0.40 pp) is measured against a test period in which home
probability drifted *down* 0.41 pp on average, while the training period averaged
+0.04 pp. The sign of the average drift flipped between periods, so the model may be
separating from a drift rather than finding value. By my rough estimate the gap is
about 2.5 standard errors (n = 453, drift std 2.7 pp), so it is not nothing, but it
is not a bettable edge.

**Checks.** Only 2.7% of matches have a different set of books at open and close, so
the drift is not a book-mix artefact.

**Limits of this test.** Opening is 1 to 3 days out, not an hourly path into the last
three hours; 5 to 6 books, and the capture time is approximated; consensus is still a
proxy for truth. This tests the *decade* question. The *resolution* question stays
open and needs sub-hourly data.

## v0.6 — do thinner markets still lead, on modern data?

The 2015-16 per-league bake-off (`results/per_league.md`) found Netherlands
(0.792 AUC) and Portugal (0.805) clearly ahead of the big-5 leagues (0.75-0.78) for
"which book moves next". Pulled those two divisions from football-data.co.uk
(11,968 matches now across 7 leagues, up from 9,110/5) and re-ran the closing-
consensus forecast per league (`scripts/per_league_fd.py` -> `results/per_league_fd.md`):

| league | dir_acc (toward-sharp) | spearman | dir_acc (XGBoost) | spearman (XGBoost) |
|---|---|---|---|---|
| netherlands | 0.579 | 0.123 | 0.545 | 0.115 |
| italy | 0.560 | 0.121 | 0.641 | 0.016 |
| portugal | 0.558 | 0.089 | 0.494 | -0.016 |
| spain | 0.534 | 0.114 | 0.530 | 0.145 |
| england | 0.539 | 0.081 | 0.532 | 0.083 |
| france | 0.512 | 0.062 | 0.556 | 0.095 |
| germany | 0.480 | 0.051 | 0.555 | 0.104 |

**Partial replication.** Netherlands is again the best-performing league on the
simple toward-sharp baseline (0.579 direction accuracy, the highest of the seven),
consistent with 2015-16. Portugal is mid-table here, not a standout — on this
2-snapshot, open/close-only frame it no longer separates itself, so the earlier
Portugal result may lean on the hourly path 2015-16 had and this test does not.
XGBoost is noisier per league (small per-league test sets, 270-340 rows) and its
ranking does not track the simple baseline's, so read the XGBoost column as
unstable rather than a second confirmation. Germany is the weakest league in both
tests.

**Reading.** "Thinner market -> more predictable" gets one real data point of
support (Netherlands) and one that does not hold up (Portugal) once the frame
changes from hourly moves to a 2-point open/close forecast. Not a clean win.

## v0.7 — costed backtest: real odds, real results, no edge

Every CLV check so far used the consensus as a proxy for the true outcome (did the
price move the "right" way). `scripts/costed_backtest.py` replaces that with real
money: stake 1 unit on home at the best price on offer across books, whenever the
model flags "home will shorten" (yhat > 0.5%), settle on the actual match result.
No commission (fixed-odds books already bake their margin into the price).

| | n | avg odds | win rate | ROI % |
|---|--:|--:|--:|--:|
| **2015-16 (BTB)** model-flagged | 1,826 | 2.32 | 0.465 | **-9.03** |
| 2015-16 random same-n | 1,826 | 2.79 | 0.416 | -7.46 |
| 2015-16 bet every row | 6,003 | 2.79 | 0.414 | -4.21 |
| **modern (football-data)** model-flagged | 453 | 2.07 | 0.561 | **+0.38** |
| modern random same-n | 453 | 2.95 | 0.450 | +1.75 |
| modern bet every row | 1,588 | 2.86 | 0.436 | -4.01 |

**Reading.** In both eras the model-flagged bets do **not** beat a random same-size
sample from the test set — on 2015-16 data they are clearly worse (-9.0% vs -7.5%),
and on modern data they are also worse (+0.4% vs +1.8%, though both are noisy at
n=453). "Bet every row" landing around -4% in both eras is the expected fixed-odds
home-bias/vig baseline, a useful sanity check that the backtest mechanics are sound.

This directly contradicts the optimistic read in v0.4/v0.5 (flagged set shortens
more than random, wins more often). The difference is what "right" means: the
consensus-proxy CLV check rewards the model for calling the *direction of the
consensus move*, which is not the same as calling *which side wins the match* at a
*specific bookmaker's price*. The model is weakly right about where the market is
headed and this does not convert into betting profit. **Verdict: no economic edge
survives contact with real prices and real results, in either era.**

## v0.8 — Rule Explorer audit: closing out the "clauses need work" thread

`scripts/extract_rules.py` (2026-09-19) tests every clause pulled from the v0.2
decision tree and the live-data Tsetlin export directly against data, rather than
trusting the plain-English gloss written at the time. 44 rules: **20 supported,
14 weak (right direction, small lift), 8 contradicted, 2 untested.**

The 8 contradicted rules are all versions of the same mistake: reading "no book
moved last snapshot" as "a quiet market is about to break" when the data says the
opposite — a quiet snapshot is followed by a move only 2.6-6.0% of the time
against a 7.1-7.9% base rate (0.37x-0.85x lift, i.e. *less* likely, not more).
This affects the v0.1 and v0.2 clause write-ups above (both used the
`n_books_moved_prev_0` literal and called it "someone breaks the silence" /
"a quiet hour tends to break") — corrected in place above. Full detail per rule:
`results/rules.json`.

**Reading.** The Tsetlin clauses were directionally wrong on this one point, not
fabricated — the underlying literal (`n_books_moved_prev_0`) is a real, useful
predictor of the *opposite* class (NO-MOVE), and several of the "weak" (label
correct, small lift) rules already say that correctly. The error was in how a
human (me) glossed a MOVE-class clause containing that literal, not in the
Tsetlin Machine's classification. Worth remembering when reading any TM clause
report: check the class the clause votes for, not just the literals in it.
## v0.9 — testing the moves-classifier itself on modern data (2026-09-22)

Every prior modern-data test (v0.5-v0.7) used the *consensus-forecast regression*.
The project's actual subject — "which book moves next", the classifier the
Tsetlin Machine was built for — had never been run on football-data.co.uk at all.
Built `config/features.fd.moves.yaml` + `config/bakeoff.fd.yaml` to do that, and
ran the 7 non-TM baselines locally first (CPU, free, no Colab needed) before
spending GPU time.

```
model            roc_auc
majority          0.5000
moved_last        0.5000
logistic          0.4939
random_forest     0.4874
lightgbm          0.4866
decision_tree     0.4861
xgboost           0.4849
```
(57,907 rows, positive rate **0.832** — full table: `results/bakeoff_fd_moves_summary.md`)

**No baseline beats chance.** Every trained model scores at or *below* 0.50
ROC-AUC — logistic regression, random forest, XGBoost and LightGBM all land
between 0.485 and 0.494. This is a materially different (worse) result than any
prior test on this data: the consensus-forecast regression (v0.5) found a real,
if weak, direction-calling signal (~0.55). The classifier framing finds nothing.

**Why the two framings disagree.** football-data.co.uk has only 2 snapshots per
book per match (opening capture, closing), so "will this book move by the next
snapshot" collapses into "did the price differ from open to close" — a single
1-to-3-day-ahead prediction, not a true snapshot-to-snapshot classifier. The
85%+ positive rate confirms it: almost every price differs somewhat over that
gap, so the useful information is *how much and which way* (what the regression
target measures), not binary *whether* (what this classifier target measures).
The "which book moves next" framing needs real intraday snapshots to mean
anything; football-data's 2-point structure cannot support it.

**Decision: skip the Colab TM run on this framing.** `notebooks/tm_bakeoff_fd_colab.ipynb`
is built and ready (mirrors `tm_bakeoff_colab.ipynb`, points at the configs above,
writes `results/tm_fd_result.json`/`tm_fd_clauses.txt` via `scripts/tm_run.py
--config config/bakeoff.fd.yaml --out-prefix tm_fd`), but running the actual
Tsetlin Machine would cost ~10 min of GPU time to most likely reconfirm ~0.49
AUC — every baseline model already agrees there is nothing here to find. The
Tsetlin Machine's actual test on modern data remains open only in the narrow
sense of "not literally run"; the evidence that it would find nothing new is
already about as strong as it gets short of running it.

### Does pooling BTB + modern data help? (`scripts/combined_era_bakeoff.py`)

Combined BTB EPL (2015-16, hourly, 433 matches) with all modern football-data
matches (12,401 matches total, 803,994 rows) into one training panel, added an
`is_modern_data` literal, and reran the classifier.

```
model                         roc_auc
logistic                       0.4930
decision_tree                  0.4933
random_forest                  0.4881
xgboost                        0.4892
xgboost (no is_modern_data)    0.4961
```

**No help, and a confound.** `is_modern_data` is XGBoost's single most important
literal (importance 0.57, rank 1 of 55) — the model leans on telling the two
sources apart more than on anything else. Test performance (0.489) is
statistically the same as the modern-only run (0.485-0.494) because the
time-ordered split puts all of 2015-16 in train — BTB is chronologically first,
so the test set here is 100% modern rows regardless. **Pooling adds rows, not
comparable rows**: the two sources have different target semantics (hourly move
vs multi-day move for the same 0.5% threshold), so mixing them does not create a
bigger, more powerful training set — it creates a training set the model mostly
learns to segment by source. Full detail: `results/combined_era.md`.

## v0.10 — real sub-hourly data restores the signal (n=50 matches)

v0.9 found zero signal for "which book moves next" on football-data.co.uk,
and traced it to resolution: only 2 snapshots a match can't support a
snapshot-to-snapshot classifier. OddsPapi's free tier (see
`results/data_sources.md`) answers that directly — real per-book price ticks,
median ~16 min apart. Built the ingest (`src/ingest/oddspapi.py`).

**A real methodological problem showed up first.** Pinnacle and bet365 update
asynchronously and never share an exact timestamp (0 of 18,916 raw
timestamps had more than one book). `build_panel.py`'s consensus/dispersion
logic groups by `(match_id, snapshot_ts)`, so every row had `n_books=1` and
consensus silently collapsed to "your own price" — the whole lead/lag feature
set was dead on arrival. Fixed with a new step, `src/panel/resample_ticks.py`:
forward-fill each (match, book, outcome) onto a shared 5-minute grid (well
under the ~16 min median gap) before the panel step.

**First run, 15 matches:** logistic 0.761 / XGBoost 0.758 AUC, test set 3
matches, leakage control (shuffled target) 0.477 (chance). Promising, but
thin — one manual ingest run capped by `--max-matches 15`.

**Scaled up to 50 matches** (`--max-matches 50 --days-back 90`, ~100 of the
250 monthly requests, checked via the free `/account` endpoint first: 50/250
used before this run, comfortable headroom) — 126,855 raw price points,
637,995 panel rows after resampling:

```
model            roc_auc
xgboost           0.769
logistic          0.768
decision_tree     0.764
lightgbm          0.754
random_forest     0.738
moved_last        0.578
majority          0.500
```

**The result held and got slightly stronger with more data.** Test set is
now 10 matches (70,932 rows, up from 3/19,258), and **XGBoost 0.769 edges
past the 2015-16 BTB result (0.765)** — on real modern data, at real
sub-hourly resolution, with real sharp books. Leakage control: shuffling the
target gives **0.511** (chance) on this larger split too.

**Still a caveat, smaller than before.** 10 test matches is far better than 3
but still not a large evaluation set — treat this as a strong, corroborated
signal rather than a final number. Scaling further is mostly a matter of
running the ingest again (quota allowing) and re-running the pipeline; both
steps are proven.

**Reading.** Put together with v0.9, the picture is now coherent rather than
contradictory: the moves-classifier needs real intraday resolution to work at
all (v0.9: zero signal without it), and when that resolution is present
(v0.10), the signal comes back at 2015-16 strength or slightly above it. The
"is the signal a 2015-16 artefact" question from v0.5-v0.9 has a clear
answer: no — it's a resolution requirement, and modern data has it, just not
from the two free sources (BTB's descendant data, football-data.co.uk) tried
first.

## v0.11 — the Tsetlin Machine, finally run on the best data — and it loses

Every v0.10 number was baselines only. Ran the actual Tsetlin Machine
(`notebooks/tm_bakeoff_oddspapi_colab.ipynb`, Colab T4, same 600 clauses /
T=32 / s=5.0 / `max_included_literals=5` config as every other TM run) on the
50-match OddsPapi panel — the project's best data, on the project's actual
subject, for the first time.

```
epoch  1/15  AUC=0.712
epoch  3/15  AUC=0.691
epoch  6/15  AUC=0.730
epoch  9/15  AUC=0.629
epoch 12/15  AUC=0.615
epoch 15/15  AUC=0.566
final test AUC: 0.572   (PR-AUC 0.015, train 881s)
```

**TM does not just tie here — it loses, clearly, and gets worse as it
trains.** Every baseline beat it: logistic 0.768, decision tree 0.764,
XGBoost 0.764, LightGBM 0.754, random forest 0.738. TM's final 0.572 barely
edges the naive "moved last snapshot" heuristic (0.578) — the exact pattern
the `moved_last` baseline exists to catch, TM did not clear it. This is a
real regression from the 2015-16 result, where TM tied the baselines (0.742
vs XGBoost's 0.765, only ~2 points back). On this data it is ~19 points back
and dropping across epochs, not converging.

**Why, probably.** The training positive rate here is **0.78%**, far more
imbalanced than 2015-16's 7.1% (a 5-minute grid gives many more "did nothing"
steps for the same move threshold). The same epoch-instability pattern
appeared in v0.2 on far less imbalanced BTB data ("epoch AUCs 0.72 -> 0.59 ->
0.71") and was blamed there on the `max_included_literals=5` budget; at
0.78% positive rate that instability looks to have gotten materially worse,
not better. Untuned `T`/`s`/clause-count for this imbalance level is the
most likely fix, not yet tried.

**The clauses are also less interesting than v0.2's.** 80 short clauses came
out, almost all single-literal and near-duplicates of the model's own most
obvious features (`any_sharp_moved_last` appears standalone 8+ times,
`n_books_moved_prev_0` negated 8+ times) — much less structure than the
BTB run's lead/lag rulebook. One clause (`IF is_modern_data`) is a known
bug, not a finding: `is_modern_data` is meant to distinguish BTB from
football-data.co.uk rows in the *combined* panel (v0.9); on a single-source
OddsPapi panel it is constant (always false) and carries no information —
`booleanize.py` should skip it when the panel has only one source. Harmless
here (a constant literal can't change any prediction) but worth fixing
before the next combined-panel run.

**Reading.** This closes the "has TM been tried on the best data" question
honestly: yes, and it lost. The baselines' 0.769 AUC (v0.10) is real and
holds up; the Tsetlin Machine specifically needs either more tuning for this
imbalance level or is just a worse fit for 5-minute-grid data than for the
hourly BTB grid it was tuned against. Matches this project's stated goal —
find where TM does and doesn't have an edge, not prove it's better — cleanly:
here, it doesn't.

