# Experiment Results

## Smoke Test (3 epochs, synthetic PEMS04 data)

```
Smoke test: batch=16 mem=128 epochs=3
Device: cuda | PEMS04 d=128 mem=128
Generating synthetic PEMS04 data (59 days, 307 sensors)
Dataset sizes: train=10172, val=3375, test=3376
Model parameters: 1,797,134
  Ep1 B100/635: loss=0.5261
  Ep1 B200/635: loss=0.4458
  Ep1 B300/635: loss=0.4133
  Ep1 B400/635: loss=0.3966
  Ep1 B500/635: loss=0.3856
  Ep1 B600/635: loss=0.3783
Ep1/3: loss=0.3763 | val_MAE=8.5966 RMSE=13.2356 MAPE=14.20% | 37.3s
  Ep2 B100/635: loss=0.3419
  Ep2 B200/635: loss=0.3407
  Ep2 B300/635: loss=0.3395
  Ep2 B400/635: loss=0.3395
  Ep2 B500/635: loss=0.3397
  Ep2 B600/635: loss=0.3393
Ep2/3: loss=0.3390 | val_MAE=8.0742 RMSE=12.2755 MAPE=13.37% | 37.2s
  Ep3 B100/635: loss=0.3348
  Ep3 B200/635: loss=0.3347
  Ep3 B300/635: loss=0.3342
  Ep3 B400/635: loss=0.3335
  Ep3 B500/635: loss=0.3328
  Ep3 B600/635: loss=0.3325
Ep3/3: loss=0.3326 | val_MAE=8.0909 RMSE=12.3542 MAPE=13.27% | 37.0s

=== TEST EVALUATION ===
  mae_avg: 8.0550
  rmse_avg: 12.1389
  mape_avg: 13.8719
SMOKE TEST PASSED
Test MAE: 8.055042266845703
```
