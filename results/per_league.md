# Per-league bake-off (Beat The Bookie)

| league | matches | rows | pos_rate | xgboost | logistic | decision_tree | moved_last |
|---|---|---|---|---|---|---|---|
| england_premier_league | 433 | 746087 | 0.071 | 0.75 | 0.747 | 0.741 | 0.608 |
| europe_champions_league | 265 | 446821 | 0.092 | 0.756 | 0.755 | 0.747 | 0.615 |
| france_ligue_1 | 442 | 741598 | 0.064 | 0.775 | 0.772 | 0.765 | 0.622 |
| germany_bundesliga | 355 | 602352 | 0.062 | 0.749 | 0.745 | 0.737 | 0.602 |
| italy_serie_a | 457 | 781099 | 0.059 | 0.755 | 0.751 | 0.742 | 0.611 |
| netherlands_eredivisie | 386 | 590223 | 0.047 | 0.792 | 0.789 | 0.781 | 0.613 |
| portugal_primeira_liga | 357 | 554071 | 0.047 | 0.805 | 0.804 | 0.8 | 0.634 |
| spain_primera_division | 344 | 577502 | 0.063 | 0.746 | 0.746 | 0.74 | 0.618 |

ROC-AUC, time-split (`bakeoff.ci.yaml`). `moved_last` = naive persistence.