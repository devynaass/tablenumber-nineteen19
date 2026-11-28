# Hasil Komparasi Sprint 1

Metrik regresi dihitung dari berkas metrics.json. Tanda — berarti metrik tidak tersedia.

| NIM | Model | Seed | Val RMSE | Val MAE | Val R² | Test RMSE | Test MAE | Test R² | Durasi (s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | baseline | 1 | 0.4101 | 0.2944 | 0.0159 | — | — | — | 2.7 |  valid |
| 123450007 | cnn | 1 | 0.4309 | 0.2278 | -0.0863 | — | — | — | 1100.7 | valid |
| 123450064 | trf | 1 | 0.4773 | 0.4120 | -0.3329 | — | — | — | 8.7 | valid |
| 123450075 | transfer_learning_resnet18 | 1 | 0.4019 | 0.3018 | 0.0550 | 0.6335 | 0.4911 | -0.0734 | 597.7 | valid |
| 123450085 | rnn | 1 | 0.6613 | 0.5132 | -1.5588 | — | — | — | 46.9 | valid |
| 123450090 | ae | 1 | 0.9116 | 0.7473 | -3.8626 | — | — | — | 891.0 | valid |
