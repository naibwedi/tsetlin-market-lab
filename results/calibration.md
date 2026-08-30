# Is the consensus a good probability estimate?

- matches with a result: 433  |  home-win base rate: 0.432

## Calibration of the closing consensus p(home win)
```
             n   pred  actual
bin                          
(0.1, 0.2]  48  0.163   0.188
(0.2, 0.3]  46  0.255   0.370
(0.3, 0.4]  95  0.354   0.284
(0.4, 0.5]  91  0.451   0.396
(0.5, 0.6]  70  0.555   0.586
(0.6, 0.7]  43  0.649   0.605
(0.7, 0.8]  38  0.750   0.763
(0.8, 0.9]   2  0.838   1.000
```

- Brier score (closing consensus): **0.2188**  (always-predict-base-rate: 0.2454)
- log-loss: 0.6279

If `pred` ~= `actual` down the calibration table and Brier < base, the consensus is a well-calibrated probability -- which is what the economic branch assumes when it treats consensus as proxy truth.