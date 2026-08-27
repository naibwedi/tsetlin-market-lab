# Clause report

## Decision tree (depth 4)
```
|--- dispersion_high <= 0.50
|   |--- ref_pinnacle_moved_last <= 0.50
|   |   |--- book_is_pinnacle <= 0.50
|   |   |   |--- n_books_moved_prev_0 <= 0.50
|   |   |   |   |--- class: 0
|   |   |   |--- n_books_moved_prev_0 >  0.50
|   |   |   |   |--- class: 0
|   |   |--- book_is_pinnacle >  0.50
|   |   |   |--- ref_betfair_ex_eu_moved_up_last <= 0.50
|   |   |   |   |--- class: 1
|   |   |   |--- ref_betfair_ex_eu_moved_up_last >  0.50
|   |   |   |   |--- class: 0
|   |--- ref_pinnacle_moved_last >  0.50
|   |   |--- book_is_slowbook <= 0.50
|   |   |   |--- book_is_pinnacle <= 0.50
|   |   |   |   |--- class: 1
|   |   |   |--- book_is_pinnacle >  0.50
|   |   |   |   |--- class: 1
|   |   |--- book_is_slowbook >  0.50
|   |   |   |--- thisbook_very_stale <= 0.50
|   |   |   |   |--- class: 0
|   |   |   |--- thisbook_very_stale >  0.50
|   |   |   |   |--- class: 0
|--- dispersion_high >  0.50
|   |--- book_is_slowbook <= 0.50
|   |   |--- thisbook_absmove_ge_1pct <= 0.50
|   |   |   |--- book_is_pinnacle <= 0.50
|   |   |   |   |--- class: 1
|   |   |   |--- book_is_pinnacle >  0.50
|   |   |   |   |--- class: 1
|   |   |--- thisbook_absmove_ge_1pct >  0.50
|   |   |   |--- ref_pinnacle_moved_last <= 0.50
|   |   |   |   |--- class: 1
|   |   |   |--- ref_pinnacle_moved_last >  0.50
|   |   |   |   |--- class: 1
|   |--- book_is_slowbook >  0.50
|   |   |--- n_books_moved_prev_ge_6 <= 0.50
|   |   |   |--- thisbook_below_consensus <= 0.50
|   |   |   |   |--- class: 0
|   |   |   |--- thisbook_below_consensus >  0.50
|   |   |   |   |--- class: 0
|   |   |--- n_books_moved_prev_ge_6 >  0.50
|   |   |   |--- thisbook_lags_pinnacle <= 0.50
|   |   |   |   |--- class: 1
|   |   |   |--- thisbook_lags_pinnacle >  0.50
|   |   |   |   |--- class: 0

```

## XGBoost feature importance (top 20)
```
dispersion_high               0.230242
n_books_moved_prev_0          0.135243
book_is_slowbook              0.103259
ref_pinnacle_moved_last       0.094203
book_is_pinnacle              0.089885
thisbook_absmove_ge_1pct      0.047384
dispersion_very_high          0.034665
thisbook_stale                0.032774
thisbook_at_consensus         0.020878
thisbook_above_consensus      0.018567
thisbook_lags_pinnacle        0.016717
n_books_moved_prev_ge_6       0.013964
n_books_moved_prev_ge_3       0.013722
thisbook_absmove_ge_3pct      0.013439
thisbook_moved_last           0.010454
ref_pinnacle_moved_up_last    0.008198
kickoff_lt_60m                0.008177
book_is_bet365                0.008159
kickoff_lt_180m               0.008082
ref_marathonbet_moved_last    0.007409
```

## Tsetlin Machine clauses
_tmu not installed - run with the `tm` extra._

## Verdict (fill in)
- Where does TM win / lose vs tree & GBM?
- Any literal combination TM surfaced that the others did not?
- Recommended next phase.