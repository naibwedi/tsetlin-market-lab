# Pre-registration: is there a betting edge in the odds-move signal?

Written 2026-11-21, **before** any new OddsPapi matches are collected for this test.
Everything below is fixed in advance. Changes after data arrives go in the
"Deviations" log at the bottom, with a date, and cannot rescue a failed test.

## Question

Can a model that flags "this book is about to move" (Tsetlin Machine or
XGBoost) pick bets whose price beats Pinnacle's closing fair price, after the
book's own margin, on matches it has never seen?

## Why this test exists

FINDINGS v0.12 (10 test matches, 2 books): AUC about 0.76 for every model, but
flagged bets lost about as much as random stale-side bets (EV -3 to -6%).
The only hint is the top 1% of the `converges` target (about 2 points better
than random, n = 60-74, interval spans zero). This test decides whether that
hint is real.

## Data (new, held out)

- OddsPapi free tier, finished EPL matches, **books: pinnacle, bet365, paddypower**
  (3 books fit in one history call, so no extra quota).
- Matches are taken in date order from newest backwards until the quota budget
  is spent (at most 200 requests in this round). No picking or dropping of matches.
- The current 50-match panel becomes train + validation. **Only matches not in
  it count as test.**
- Raw odds stay local (OddsPapi terms). Only aggregates are published.

## Frozen method

- Features: the current Boolean features (`config/features.oddspapi*.yaml`),
  plus the Paddy Power literals the same code produces. No new feature ideas
  after seeing test results.
- Primary target: `converges` (move toward the other book). Secondary: `moves`.
- Models: Tsetlin Machine (500 clauses, class-weighted, best validation epoch,
  early stopping 6) and XGBoost. T, epochs and every hyper-parameter are chosen
  on validation only. **Fit once, then score the test matches once.**
- Bet rule: score each (match, book, 5-min) row; keep rows in the top **1%** of
  test scores; back HOME at that book only when its price is the longest of the
  books at that moment (the stale side); one bet per (match, book, hour), the
  highest-scored row; stake 1.
- Value measure: `EV = entry_odds x Pinnacle closing fair P(home) - 1`, closing
  = Pinnacle's last pre-kickoff snapshot, no-vig. The book's margin is already in
  `entry_odds`, so no extra cost is added here.
- Control: random stale-side bets, same number of bets, 200 draws.
- Uncertainty: bootstrap over **matches** (2,000 resamples), never over rows.

## Primary test (one decision)

Primary = Tsetlin Machine, `converges` target, top 1%. Everything else
(XGBoost, `moves`, top 5%) is reported but cannot decide the result.

**PASS** only if ALL hold at the single analysis point:

1. at least **40 new test matches** and at least **300 primary bets**;
2. mean EV > 0 and its 95% match-bootstrap interval lower bound > 0;
3. mean EV beats the random stale-side control by at least 2 points, with that
   difference's 95% interval lower bound > 0;
4. mean EV > 0 in both chronological halves of the new matches.

**FAIL** (close the question as a clean negative, no retries with new
features or targets on the same data): the minimums in (1) are met and
(2), (3) or (4) is not.

**INCONCLUSIVE**: fewer than 40 matches or 300 bets. Collect more; do not look
at EV until the minimums are met (the AUC and bet counts may be checked).

One analysis at the minimums, one more if the quota allows a second batch. No
early stop on a good-looking result.

## What a PASS would and would not mean

A PASS is a paper edge at the quoted price. It does not include limits, stale
prices that are no longer available by the time an order lands, account
restrictions or latency. After a PASS the next step is a paper-trade trial with
a realistic fill haircut, not live money.

## Deviations log

- 2026-11-21 (before any test data was collected): the third book is **paddypower**,
  not williamhill. A 2-match trial ingest returned no William Hill price history
  for current fixtures (Paddy Power and Marathonbet both returned data). Same
  call slot, no extra quota. Wherever this document says "William Hill", read
  "Paddy Power". Test matches are those kicking off after 2026-09-21.
