# Bake-off summary

- test rows: 4,956  |  test positive rate: 0.116
- literals: 35

Mean over seeds (std in parentheses is omitted here; see JSON):

```
model
xgboost          0.7980
decision_tree    0.7861
lightgbm         0.7784
logistic         0.7763
random_forest    0.7180
moved_last       0.5190
majority         0.5000
```

ROC-AUC ranking above. Verdict (fill in after review): TM loses / TM ties + useful rules / TM wins.