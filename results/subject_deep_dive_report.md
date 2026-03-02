# Subject Deep-Dive Analysis Report

Generated: 2026-03-01 21:41:30
Subjects: S10, S20, S05, S21


---

# S10

## 1. Per-Session Overview

| Session | Runs | Trials | Labels | Amp Max | Amp P99.9 | Artifact% | Ch Var Mean | SNR (dB) |
|---------|------|--------|--------|---------|-----------|-----------|-------------|----------|
| OfflineImagery | 30 | 600 | 1:150, 2:150, 3:150, 4:150 | 267903.8 | 100008.0 | 25.0% | 88135968.00 | -23.6 |
| OnlineImagery_Sess01_2class_Base | 8 | 160 | 1:80, 4:80 | 1151.7 | 167.7 | 4.4% | 239.82 | n/a |
| OnlineImagery_Sess01_2class_Finetune | 8 | 160 | 1:80, 4:80 | 1343.0 | 78.1 | 1.9% | 107.56 | n/a |
| OnlineImagery_Sess01_3class_Base | 8 | 240 | 1:80, 2:80, 4:80 | 3950.3 | 244.3 | 3.8% | 593.18 | n/a |
| OnlineImagery_Sess01_3class_Finetune | 8 | 240 | 1:80, 2:80, 4:80 | 759.5 | 142.8 | 2.9% | 159.61 | n/a |
| OnlineImagery_Sess02_2class_Base | 8 | 160 | 1:80, 4:80 | 322.7 | 98.4 | 0.0% | 111.72 | n/a |
| OnlineImagery_Sess02_2class_Finetune | 8 | 160 | 1:80, 4:80 | 6972.4 | 149.8 | 4.4% | 349.45 | n/a |
| OnlineImagery_Sess02_3class_Base | 8 | 240 | 1:80, 2:80, 4:80 | 4267.6 | 174.4 | 1.7% | 292.93 | n/a |
| OnlineImagery_Sess02_3class_Finetune | 8 | 240 | 1:80, 2:80, 4:80 | 744.5 | 101.4 | 0.8% | 115.61 | n/a |

## 2. Run-Level Amplitude Profile

### Top 10 Worst Runs (by max amplitude)

| Session | Run | Trials | Amp Max | Amp Mean | Trial P95 | >500 | >1000 |
|---------|-----|--------|---------|----------|----------|------|-------|
| OfflineImagery | 30 | 20 | 267903.8 | 13314.27 | 254050.6 | 20 | 20 |
| OfflineImagery | 28 | 20 | 257780.5 | 13329.50 | 252649.1 | 20 | 20 |
| OfflineImagery | 27 | 20 | 254949.8 | 13367.84 | 243999.0 | 20 | 20 |
| OfflineImagery | 29 | 20 | 241538.8 | 13336.62 | 216337.2 | 20 | 20 |
| OfflineImagery | 26 | 20 | 235053.2 | 13391.43 | 232795.2 | 20 | 20 |
| OnlineImagery_Sess02_2class_Finetune | 3 | 20 | 6972.4 | 6.93 | 650.3 | 1 | 1 |
| OnlineImagery_Sess02_3class_Base | 8 | 30 | 4267.6 | 6.95 | 1731.2 | 2 | 2 |
| OnlineImagery_Sess01_3class_Base | 5 | 30 | 3950.3 | 14.91 | 1322.1 | 3 | 3 |
| OnlineImagery_Sess01_3class_Base | 8 | 30 | 3742.2 | 10.41 | 687.1 | 2 | 1 |
| OnlineImagery_Sess01_3class_Base | 6 | 30 | 3649.9 | 10.29 | 1987.8 | 2 | 2 |

### Per-Session Run Summary

| Session | N Runs | Median AmpMax | Max AmpMax | Total >500 | Total >1000 |
|---------|--------|---------------|------------|-----------|-------------|
| OfflineImagery | 30 | 1627.9 | 267903.8 | 150 | 133 |
| OnlineImagery_Sess01_2class_Base | 8 | 399.2 | 1151.7 | 7 | 1 |
| OnlineImagery_Sess01_2class_Finetune | 8 | 120.4 | 1343.0 | 3 | 2 |
| OnlineImagery_Sess01_3class_Base | 8 | 572.8 | 3950.3 | 9 | 6 |
| OnlineImagery_Sess01_3class_Finetune | 8 | 547.0 | 759.5 | 7 | 0 |
| OnlineImagery_Sess02_2class_Base | 8 | 192.8 | 322.7 | 0 | 0 |
| OnlineImagery_Sess02_2class_Finetune | 8 | 1014.6 | 6972.4 | 7 | 5 |
| OnlineImagery_Sess02_3class_Base | 8 | 401.9 | 4267.6 | 4 | 3 |
| OnlineImagery_Sess02_3class_Finetune | 8 | 303.2 | 744.5 | 2 | 0 |

## 3. Channel Quality Map

Sampled 500 trials.

### Worst 10 Channels by Variance

| Rank | Ch Idx | Label | Mean Var | Region |
|------|--------|-------|----------|--------|
| 1 | 43 | B12 | 165372256.0000 | peripheral |
| 2 | 54 | B23 | 164802304.0000 | peripheral |
| 3 | 66 | C3 | 140709968.0000 | central |
| 4 | 13 | A14 | 138442640.0000 | peripheral |
| 5 | 21 | A22 | 127098904.0000 | peripheral |
| 6 | 59 | B28 | 125846640.0000 | peripheral |
| 7 | 25 | A26 | 125657896.0000 | peripheral |
| 8 | 7 | A8 | 123214120.0000 | peripheral |
| 9 | 72 | C9 | 122475224.0000 | peripheral |
| 10 | 75 | C12 | 120164608.0000 | peripheral |

### Worst 10 Channels by Max Amplitude

| Rank | Ch Idx | Label | Mean Max Amp | Region |
|------|--------|-------|-------------|--------|
| 1 | 54 | B23 | 9520.34 | peripheral |
| 2 | 43 | B12 | 9255.84 | peripheral |
| 3 | 75 | C12 | 8500.21 | peripheral |
| 4 | 13 | A14 | 8410.22 | peripheral |
| 5 | 21 | A22 | 8352.47 | peripheral |
| 6 | 66 | C3 | 8224.86 | central |
| 7 | 7 | A8 | 8147.27 | peripheral |
| 8 | 72 | C9 | 8137.43 | peripheral |
| 9 | 59 | B28 | 8067.39 | peripheral |
| 10 | 64 | C1 | 7971.02 | central |

**Central vs Peripheral**:
- Variance: central=19432660.0000, peripheral=28840474.0000 (ratio=1.48x)
- Max Amp: central=2543.77, peripheral=3036.28
- Mu/Beta Ratio: central=0.495, peripheral=0.494

## 4. Per-Session Class Discriminability

| Session | Classes | Fisher Mean | AUROC Mean (top5) | Class Counts |
|---------|---------|-----------|-----------------|-------------|
| OfflineImagery | 4 | 0.0001 | 0.5354 | 1:150, 2:150, 3:150, 4:150 |
| OnlineImagery_Sess01_2class_Base | 2 | 0.0095 | 0.6098 | 1:80, 4:80 |
| OnlineImagery_Sess01_2class_Finetune | 2 | 0.0257 | 0.6722 | 1:80, 4:80 |
| OnlineImagery_Sess01_3class_Base | 3 | 0.0127 | 0.6097 | 1:80, 2:80, 4:80 |
| OnlineImagery_Sess01_3class_Finetune | 3 | 0.0104 | 0.5855 | 1:80, 2:80, 4:80 |
| OnlineImagery_Sess02_2class_Base | 2 | 0.0145 | 0.6026 | 1:80, 4:80 |
| OnlineImagery_Sess02_2class_Finetune | 2 | 0.0562 | 0.7060 | 1:80, 4:80 |
| OnlineImagery_Sess02_3class_Base | 3 | 0.0193 | 0.6022 | 1:80, 2:80, 4:80 |
| OnlineImagery_Sess02_3class_Finetune | 3 | 0.0132 | 0.5865 | 1:80, 2:80, 4:80 |

## 5. Per-Session Spectral Profile

| Session | Theta | Mu | Low-β | High-β | γ | Mu/β Ratio |
|---------|-------|-----|-------|--------|---|-----------|
| OfflineImagery | 1981371.8750 | 1988307.3750 | 2036132.7500 | 2034679.8750 | 2035581.7500 | 0.488 |
| OnlineImagery_Sess01_2class_Base | 3.1341 | 0.8680 | 0.4225 | 0.2958 | 0.2612 | 1.208 |
| OnlineImagery_Sess01_2class_Finetune | 2.1245 | 0.7511 | 0.3519 | 0.1925 | 0.1343 | 1.380 |
| OnlineImagery_Sess01_3class_Base | 3.7811 | 1.1033 | 0.5425 | 0.3993 | 0.3860 | 1.172 |
| OnlineImagery_Sess01_3class_Finetune | 2.1792 | 0.7386 | 0.3811 | 0.2323 | 0.1845 | 1.204 |
| OnlineImagery_Sess02_2class_Base | 2.8344 | 0.7618 | 0.4501 | 0.2756 | 0.1820 | 1.050 |
| OnlineImagery_Sess02_2class_Finetune | 5.4999 | 1.4342 | 0.7145 | 0.5089 | 0.3614 | 1.172 |
| OnlineImagery_Sess02_3class_Base | 2.6789 | 0.7266 | 0.3687 | 0.2228 | 0.1406 | 1.228 |
| OnlineImagery_Sess02_3class_Finetune | 2.6245 | 0.7456 | 0.4281 | 0.3277 | 0.2107 | 0.987 |

## 6. Train/Test Distribution Comparison

- Train trials: 1400, Test trials: 800
- Train global mean: -0.0000 (std=46.7494)
- Test global mean: 0.0000 (std=5.0250)
- **Mean KS statistic**: 0.1197
- Median KS statistic: 0.1201
- Max KS statistic: 0.1914
- Channels where train/test indistinguishable (p>0.05): **1/128**

### Most Similar Channels (lowest KS)

| Ch Idx | Label | KS Stat | p-value |
|--------|-------|---------|---------|
| 90 | C27 | 0.0496 | 0.1573 |
| 83 | C20 | 0.0630 | 0.0335 |
| 89 | C26 | 0.0641 | 0.0292 |
| 126 | D31 | 0.0657 | 0.0235 |
| 11 | A12 | 0.0659 | 0.0230 |

### Most Different Channels (highest KS)

| Ch Idx | Label | KS Stat | p-value |
|--------|-------|---------|---------|
| 117 | D22 | 0.1914 | 0.0000 |
| 93 | C30 | 0.1786 | 0.0000 |
| 4 | A5 | 0.1761 | 0.0000 |
| 29 | A30 | 0.1745 | 0.0000 |
| 94 | C31 | 0.1711 | 0.0000 |

## 7. Salvageability Analysis

**Amplitude threshold**: 500 µV
**Overall**: 2011/2200 trials clean (**91.4%**)

| Session | Total | Clean | Clean% | Amp P50 | Amp P90 | Amp P99 |
|---------|-------|-------|--------|---------|---------|---------|
| OfflineImagery | 600 | 450 | 75.0% | 120.2 | 200184.0 | 243499.3 |
| OnlineImagery_Sess01_2class_Base | 160 | 153 | 95.6% | 72.5 | 140.7 | 754.8 |
| OnlineImagery_Sess01_2class_Finetune | 160 | 157 | 98.1% | 65.8 | 95.1 | 871.9 |
| OnlineImagery_Sess01_3class_Base | 240 | 231 | 96.2% | 103.4 | 297.1 | 3497.9 |
| OnlineImagery_Sess01_3class_Finetune | 240 | 233 | 97.1% | 64.7 | 132.8 | 614.1 |
| OnlineImagery_Sess02_2class_Base | 160 | 160 | 100.0% | 131.0 | 195.6 | 292.0 |
| OnlineImagery_Sess02_2class_Finetune | 160 | 153 | 95.6% | 145.9 | 195.5 | 1891.4 |
| OnlineImagery_Sess02_3class_Base | 240 | 236 | 98.3% | 130.4 | 280.7 | 2002.0 |
| OnlineImagery_Sess02_3class_Finetune | 240 | 238 | 99.2% | 116.9 | 186.8 | 344.1 |


---

# S20

## 1. Per-Session Overview

| Session | Runs | Trials | Labels | Amp Max | Amp P99.9 | Artifact% | Ch Var Mean | SNR (dB) |
|---------|------|--------|--------|---------|-----------|-----------|-------------|----------|
| OfflineImagery | 32 | 640 | 1:160, 2:160, 3:160, 4:160 | 558.8 | 156.7 | 0.3% | 228.11 | -22.0 |
| OnlineImagery_Sess01_2class_Base | 8 | 160 | 1:80, 4:80 | 467.5 | 146.3 | 0.0% | 176.00 | n/a |
| OnlineImagery_Sess01_2class_Finetune | 8 | 160 | 1:80, 4:80 | 319.5 | 171.4 | 0.0% | 215.71 | n/a |
| OnlineImagery_Sess01_3class_Base | 8 | 240 | 1:80, 2:80, 4:80 | 5924.6 | 136.6 | 1.2% | 201.61 | n/a |
| OnlineImagery_Sess01_3class_Finetune | 8 | 240 | 1:80, 2:80, 4:80 | 13965.6 | 362.3 | 2.9% | 4901.37 | n/a |
| OnlineImagery_Sess02_2class_Base | 8 | 160 | 1:80, 4:80 | 281.8 | 132.9 | 0.0% | 131.01 | n/a |
| OnlineImagery_Sess02_2class_Finetune | 8 | 160 | 1:80, 4:80 | 429.9 | 132.6 | 0.0% | 150.97 | n/a |
| OnlineImagery_Sess02_3class_Base | 8 | 240 | 1:80, 2:80, 4:80 | 501.8 | 124.6 | 0.4% | 132.34 | n/a |
| OnlineImagery_Sess02_3class_Finetune | 8 | 240 | 1:80, 2:80, 4:80 | 2124.1 | 149.5 | 0.4% | 226.10 | n/a |

## 2. Run-Level Amplitude Profile

### Top 10 Worst Runs (by max amplitude)

| Session | Run | Trials | Amp Max | Amp Mean | Trial P95 | >500 | >1000 |
|---------|-----|--------|---------|----------|----------|------|-------|
| OnlineImagery_Sess01_3class_Finetune | 6 | 30 | 13965.6 | 21.24 | 9607.8 | 5 | 4 |
| OnlineImagery_Sess01_3class_Base | 6 | 30 | 5924.6 | 6.13 | 257.2 | 1 | 1 |
| OnlineImagery_Sess02_3class_Finetune | 1 | 30 | 2124.1 | 7.94 | 270.0 | 1 | 1 |
| OnlineImagery_Sess01_3class_Base | 1 | 30 | 1678.4 | 6.63 | 274.5 | 1 | 1 |
| OnlineImagery_Sess01_3class_Finetune | 2 | 30 | 986.2 | 9.36 | 256.5 | 1 | 0 |
| OnlineImagery_Sess01_3class_Finetune | 4 | 30 | 830.7 | 8.18 | 380.1 | 1 | 0 |
| OnlineImagery_Sess01_3class_Base | 3 | 30 | 643.4 | 6.42 | 214.8 | 1 | 0 |
| OfflineImagery | 13 | 20 | 558.8 | 10.50 | 539.0 | 2 | 0 |
| OnlineImagery_Sess02_3class_Base | 3 | 30 | 501.8 | 7.62 | 283.1 | 1 | 0 |
| OnlineImagery_Sess01_3class_Finetune | 1 | 30 | 489.1 | 6.94 | 286.7 | 0 | 0 |

### Per-Session Run Summary

| Session | N Runs | Median AmpMax | Max AmpMax | Total >500 | Total >1000 |
|---------|--------|---------------|------------|-----------|-------------|
| OfflineImagery | 32 | 304.9 | 558.8 | 2 | 0 |
| OnlineImagery_Sess01_2class_Base | 8 | 279.3 | 467.5 | 0 | 0 |
| OnlineImagery_Sess01_2class_Finetune | 8 | 289.2 | 319.5 | 0 | 0 |
| OnlineImagery_Sess01_3class_Base | 8 | 352.6 | 5924.6 | 3 | 2 |
| OnlineImagery_Sess01_3class_Finetune | 8 | 406.8 | 13965.6 | 7 | 4 |
| OnlineImagery_Sess02_2class_Base | 8 | 267.4 | 281.8 | 0 | 0 |
| OnlineImagery_Sess02_2class_Finetune | 8 | 248.1 | 429.9 | 0 | 0 |
| OnlineImagery_Sess02_3class_Base | 8 | 254.6 | 501.8 | 1 | 0 |
| OnlineImagery_Sess02_3class_Finetune | 8 | 259.7 | 2124.1 | 1 | 1 |

## 3. Channel Quality Map

Sampled 500 trials.

### Worst 10 Channels by Variance

| Rank | Ch Idx | Label | Mean Var | Region |
|------|--------|-------|----------|--------|
| 1 | 123 | D28 | 2296.5361 | central |
| 2 | 92 | C29 | 1469.5548 | peripheral |
| 3 | 79 | C16 | 1464.6771 | peripheral |
| 4 | 80 | C17 | 1403.9598 | peripheral |
| 5 | 91 | C28 | 1179.8838 | peripheral |
| 6 | 93 | C30 | 1158.7762 | peripheral |
| 7 | 78 | C15 | 997.4325 | peripheral |
| 8 | 81 | C18 | 936.6161 | peripheral |
| 9 | 94 | C31 | 815.5151 | peripheral |
| 10 | 71 | C8 | 717.6181 | peripheral |

### Worst 10 Channels by Max Amplitude

| Rank | Ch Idx | Label | Mean Max Amp | Region |
|------|--------|-------|-------------|--------|
| 1 | 79 | C16 | 175.92 | peripheral |
| 2 | 92 | C29 | 167.35 | peripheral |
| 3 | 80 | C17 | 166.64 | peripheral |
| 4 | 91 | C28 | 149.82 | peripheral |
| 5 | 93 | C30 | 142.52 | peripheral |
| 6 | 78 | C15 | 141.06 | peripheral |
| 7 | 81 | C18 | 137.04 | peripheral |
| 8 | 71 | C8 | 119.93 | peripheral |
| 9 | 94 | C31 | 117.85 | peripheral |
| 10 | 72 | C9 | 114.34 | peripheral |

**Central vs Peripheral**:
- Variance: central=132.4246, peripheral=214.3855 (ratio=1.62x)
- Max Amp: central=27.42, peripheral=50.79
- Mu/Beta Ratio: central=2.006, peripheral=1.767

## 4. Per-Session Class Discriminability

| Session | Classes | Fisher Mean | AUROC Mean (top5) | Class Counts |
|---------|---------|-----------|-----------------|-------------|
| OfflineImagery | 4 | 0.0298 | 0.6154 | 1:160, 2:160, 3:160, 4:160 |
| OnlineImagery_Sess01_2class_Base | 2 | 0.0338 | 0.6547 | 1:80, 4:80 |
| OnlineImagery_Sess01_2class_Finetune | 2 | 0.0615 | 0.6392 | 1:80, 4:80 |
| OnlineImagery_Sess01_3class_Base | 3 | 0.0297 | 0.6441 | 1:80, 2:80, 4:80 |
| OnlineImagery_Sess01_3class_Finetune | 3 | 0.0144 | 0.5961 | 1:80, 2:80, 4:80 |
| OnlineImagery_Sess02_2class_Base | 2 | 0.0191 | 0.6201 | 1:80, 4:80 |
| OnlineImagery_Sess02_2class_Finetune | 2 | 0.0384 | 0.6221 | 1:80, 4:80 |
| OnlineImagery_Sess02_3class_Base | 3 | 0.0157 | 0.5940 | 1:80, 2:80, 4:80 |
| OnlineImagery_Sess02_3class_Finetune | 3 | 0.0174 | 0.6141 | 1:80, 2:80, 4:80 |

## 5. Per-Session Spectral Profile

| Session | Theta | Mu | Low-β | High-β | γ | Mu/β Ratio |
|---------|-------|-----|-------|--------|---|-----------|
| OfflineImagery | 6.0669 | 1.9798 | 0.5884 | 0.4825 | 0.4584 | 1.849 |
| OnlineImagery_Sess01_2class_Base | 6.0517 | 1.1569 | 0.4451 | 0.4340 | 0.4604 | 1.316 |
| OnlineImagery_Sess01_2class_Finetune | 6.1816 | 1.7342 | 0.5260 | 0.4023 | 0.3534 | 1.868 |
| OnlineImagery_Sess01_3class_Base | 3.9275 | 1.1004 | 0.4194 | 0.3758 | 0.4037 | 1.384 |
| OnlineImagery_Sess01_3class_Finetune | 158.4372 | 50.1272 | 23.0079 | 13.3544 | 9.4689 | 1.379 |
| OnlineImagery_Sess02_2class_Base | 4.8941 | 0.9692 | 0.4602 | 0.4089 | 0.3719 | 1.115 |
| OnlineImagery_Sess02_2class_Finetune | 4.8917 | 0.9770 | 0.4880 | 0.4013 | 0.3787 | 1.099 |
| OnlineImagery_Sess02_3class_Base | 4.6890 | 0.8954 | 0.4262 | 0.3472 | 0.3179 | 1.158 |
| OnlineImagery_Sess02_3class_Finetune | 5.9418 | 1.0463 | 0.4868 | 0.4013 | 0.4030 | 1.178 |

## 6. Train/Test Distribution Comparison

- Train trials: 1440, Test trials: 800
- Train global mean: -0.0000 (std=2.1340)
- Test global mean: -0.0000 (std=12.4631)
- **Mean KS statistic**: 0.0764
- Median KS statistic: 0.0720
- Max KS statistic: 0.1824
- Channels where train/test indistinguishable (p>0.05): **50/128**

### Most Similar Channels (lowest KS)

| Ch Idx | Label | KS Stat | p-value |
|--------|-------|---------|---------|
| 53 | B22 | 0.0275 | 0.8204 |
| 102 | D7 | 0.0292 | 0.7619 |
| 79 | C16 | 0.0303 | 0.7208 |
| 101 | D6 | 0.0306 | 0.7103 |
| 59 | B28 | 0.0312 | 0.6839 |

### Most Different Channels (highest KS)

| Ch Idx | Label | KS Stat | p-value |
|--------|-------|---------|---------|
| 85 | C22 | 0.1824 | 0.0000 |
| 65 | C2 | 0.1693 | 0.0000 |
| 97 | D2 | 0.1572 | 0.0000 |
| 74 | C11 | 0.1546 | 0.0000 |
| 27 | A28 | 0.1539 | 0.0000 |

## 7. Salvageability Analysis

**Amplitude threshold**: 500 µV
**Overall**: 2226/2240 trials clean (**99.4%**)

| Session | Total | Clean | Clean% | Amp P50 | Amp P90 | Amp P99 |
|---------|-------|-------|--------|---------|---------|---------|
| OfflineImagery | 640 | 638 | 99.7% | 226.4 | 282.9 | 372.1 |
| OnlineImagery_Sess01_2class_Base | 160 | 160 | 100.0% | 196.4 | 274.5 | 326.9 |
| OnlineImagery_Sess01_2class_Finetune | 160 | 160 | 100.0% | 236.4 | 276.1 | 315.8 |
| OnlineImagery_Sess01_3class_Base | 240 | 237 | 98.8% | 172.7 | 245.4 | 533.5 |
| OnlineImagery_Sess01_3class_Finetune | 240 | 233 | 97.1% | 223.5 | 273.9 | 3834.5 |
| OnlineImagery_Sess02_2class_Base | 160 | 160 | 100.0% | 212.3 | 258.0 | 274.4 |
| OnlineImagery_Sess02_2class_Finetune | 160 | 160 | 100.0% | 208.8 | 243.5 | 268.3 |
| OnlineImagery_Sess02_3class_Base | 240 | 239 | 99.6% | 179.8 | 239.9 | 325.8 |
| OnlineImagery_Sess02_3class_Finetune | 240 | 239 | 99.6% | 214.1 | 246.9 | 274.0 |


---

# S05

## 1. Per-Session Overview

| Session | Runs | Trials | Labels | Amp Max | Amp P99.9 | Artifact% | Ch Var Mean | SNR (dB) |
|---------|------|--------|--------|---------|-----------|-----------|-------------|----------|
| OfflineImagery | 32 | 640 | 1:160, 2:160, 3:160, 4:160 | 554.7 | 33.5 | 0.3% | 36.49 | -18.7 |
| OnlineImagery_Sess01_2class_Base | 8 | 160 | 1:80, 4:80 | 293.2 | 34.1 | 0.0% | 29.16 | n/a |
| OnlineImagery_Sess01_2class_Finetune | 8 | 160 | 1:80, 4:80 | 299.3 | 24.0 | 0.0% | 28.18 | n/a |
| OnlineImagery_Sess01_3class_Base | 8 | 240 | 1:80, 2:80, 4:80 | 527.8 | 55.5 | 0.4% | 38.13 | n/a |
| OnlineImagery_Sess01_3class_Finetune | 8 | 240 | 1:80, 2:80, 4:80 | 251.7 | 24.6 | 0.0% | 26.71 | n/a |
| OnlineImagery_Sess02_2class_Base | 8 | 160 | 1:80, 4:80 | 324.7 | 69.8 | 0.0% | 57.93 | n/a |
| OnlineImagery_Sess02_2class_Finetune | 8 | 160 | 1:80, 4:80 | 368.0 | 70.2 | 0.0% | 59.66 | n/a |
| OnlineImagery_Sess02_3class_Base | 8 | 240 | 1:80, 2:80, 4:80 | 1088.9 | 74.9 | 0.4% | 63.91 | n/a |
| OnlineImagery_Sess02_3class_Finetune | 8 | 240 | 1:80, 2:80, 4:80 | 276.6 | 81.2 | 0.0% | 70.73 | n/a |

## 2. Run-Level Amplitude Profile

### Top 10 Worst Runs (by max amplitude)

| Session | Run | Trials | Amp Max | Amp Mean | Trial P95 | >500 | >1000 |
|---------|-----|--------|---------|----------|----------|------|-------|
| OnlineImagery_Sess02_3class_Base | 1 | 30 | 1088.9 | 4.58 | 160.3 | 1 | 1 |
| OfflineImagery | 8 | 20 | 554.7 | 4.27 | 213.2 | 1 | 0 |
| OnlineImagery_Sess01_3class_Base | 7 | 30 | 527.8 | 4.11 | 297.4 | 1 | 0 |
| OfflineImagery | 21 | 20 | 514.9 | 4.36 | 77.7 | 1 | 0 |
| OfflineImagery | 1 | 20 | 487.2 | 4.21 | 287.2 | 0 | 0 |
| OfflineImagery | 12 | 20 | 421.5 | 4.15 | 135.3 | 0 | 0 |
| OfflineImagery | 29 | 20 | 400.8 | 4.46 | 141.6 | 0 | 0 |
| OnlineImagery_Sess02_2class_Finetune | 3 | 20 | 368.0 | 4.95 | 126.3 | 0 | 0 |
| OfflineImagery | 13 | 20 | 368.0 | 4.37 | 247.0 | 0 | 0 |
| OnlineImagery_Sess02_2class_Base | 4 | 20 | 324.7 | 4.75 | 192.1 | 0 | 0 |

### Per-Session Run Summary

| Session | N Runs | Median AmpMax | Max AmpMax | Total >500 | Total >1000 |
|---------|--------|---------------|------------|-----------|-------------|
| OfflineImagery | 32 | 218.1 | 554.7 | 2 | 0 |
| OnlineImagery_Sess01_2class_Base | 8 | 219.0 | 293.2 | 0 | 0 |
| OnlineImagery_Sess01_2class_Finetune | 8 | 71.8 | 299.3 | 0 | 0 |
| OnlineImagery_Sess01_3class_Base | 8 | 261.8 | 527.8 | 1 | 0 |
| OnlineImagery_Sess01_3class_Finetune | 8 | 126.3 | 251.7 | 0 | 0 |
| OnlineImagery_Sess02_2class_Base | 8 | 222.6 | 324.7 | 0 | 0 |
| OnlineImagery_Sess02_2class_Finetune | 8 | 208.2 | 368.0 | 0 | 0 |
| OnlineImagery_Sess02_3class_Base | 8 | 224.8 | 1088.9 | 1 | 1 |
| OnlineImagery_Sess02_3class_Finetune | 8 | 236.0 | 276.6 | 0 | 0 |

## 3. Channel Quality Map

Sampled 500 trials.

### Worst 10 Channels by Variance

| Rank | Ch Idx | Label | Mean Var | Region |
|------|--------|-------|----------|--------|
| 1 | 92 | C29 | 215.7702 | peripheral |
| 2 | 80 | C17 | 165.6514 | peripheral |
| 3 | 93 | C30 | 155.4534 | peripheral |
| 4 | 13 | A14 | 150.4839 | peripheral |
| 5 | 91 | C28 | 147.3457 | peripheral |
| 6 | 81 | C18 | 144.2194 | peripheral |
| 7 | 14 | A15 | 136.2285 | peripheral |
| 8 | 79 | C16 | 133.6628 | peripheral |
| 9 | 78 | C15 | 124.1814 | peripheral |
| 10 | 94 | C31 | 95.6499 | peripheral |

### Worst 10 Channels by Max Amplitude

| Rank | Ch Idx | Label | Mean Max Amp | Region |
|------|--------|-------|-------------|--------|
| 1 | 92 | C29 | 48.68 | peripheral |
| 2 | 93 | C30 | 44.63 | peripheral |
| 3 | 80 | C17 | 43.57 | peripheral |
| 4 | 91 | C28 | 41.34 | peripheral |
| 5 | 81 | C18 | 39.52 | peripheral |
| 6 | 79 | C16 | 38.81 | peripheral |
| 7 | 78 | C15 | 37.63 | peripheral |
| 8 | 94 | C31 | 34.91 | peripheral |
| 9 | 71 | C8 | 33.61 | peripheral |
| 10 | 82 | C19 | 30.39 | peripheral |

**Central vs Peripheral**:
- Variance: central=24.0749, peripheral=51.1670 (ratio=2.13x)
- Max Amp: central=15.80, peripheral=22.91
- Mu/Beta Ratio: central=3.751, peripheral=3.592

## 4. Per-Session Class Discriminability

| Session | Classes | Fisher Mean | AUROC Mean (top5) | Class Counts |
|---------|---------|-----------|-----------------|-------------|
| OfflineImagery | 4 | 0.0265 | 0.6280 | 1:160, 2:160, 3:160, 4:160 |
| OnlineImagery_Sess01_2class_Base | 2 | 0.0866 | 0.7422 | 1:80, 4:80 |
| OnlineImagery_Sess01_2class_Finetune | 2 | 0.0514 | 0.7226 | 1:80, 4:80 |
| OnlineImagery_Sess01_3class_Base | 3 | 0.0123 | 0.6014 | 1:80, 2:80, 4:80 |
| OnlineImagery_Sess01_3class_Finetune | 3 | 0.0516 | 0.6612 | 1:80, 2:80, 4:80 |
| OnlineImagery_Sess02_2class_Base | 2 | 0.2932 | 0.8800 | 1:80, 4:80 |
| OnlineImagery_Sess02_2class_Finetune | 2 | 0.2988 | 0.8614 | 1:80, 4:80 |
| OnlineImagery_Sess02_3class_Base | 3 | 0.0947 | 0.6927 | 1:80, 2:80, 4:80 |
| OnlineImagery_Sess02_3class_Finetune | 3 | 0.2107 | 0.7623 | 1:80, 2:80, 4:80 |

## 5. Per-Session Spectral Profile

| Session | Theta | Mu | Low-β | High-β | γ | Mu/β Ratio |
|---------|-------|-----|-------|--------|---|-----------|
| OfflineImagery | 1.2556 | 2.3748 | 0.3341 | 0.1755 | 0.0768 | 4.661 |
| OnlineImagery_Sess01_2class_Base | 0.9371 | 1.2213 | 0.3146 | 0.1306 | 0.0737 | 2.744 |
| OnlineImagery_Sess01_2class_Finetune | 1.0030 | 1.7692 | 0.3544 | 0.1454 | 0.0750 | 3.540 |
| OnlineImagery_Sess01_3class_Base | 1.4091 | 1.2072 | 0.3011 | 0.1284 | 0.0708 | 2.810 |
| OnlineImagery_Sess01_3class_Finetune | 0.9788 | 1.5472 | 0.3318 | 0.1322 | 0.0662 | 3.335 |
| OnlineImagery_Sess02_2class_Base | 1.9013 | 1.8155 | 0.3366 | 0.1483 | 0.0736 | 3.744 |
| OnlineImagery_Sess02_2class_Finetune | 1.9658 | 2.0557 | 0.3319 | 0.1475 | 0.0731 | 4.288 |
| OnlineImagery_Sess02_3class_Base | 2.0717 | 1.2266 | 0.2969 | 0.1257 | 0.0680 | 2.902 |
| OnlineImagery_Sess02_3class_Finetune | 2.3742 | 1.5461 | 0.3298 | 0.1366 | 0.0692 | 3.315 |

## 6. Train/Test Distribution Comparison

- Train trials: 1440, Test trials: 800
- Train global mean: -0.0000 (std=1.3462)
- Test global mean: -0.0000 (std=1.4724)
- **Mean KS statistic**: 0.1143
- Median KS statistic: 0.1142
- Max KS statistic: 0.2203
- Channels where train/test indistinguishable (p>0.05): **15/128**

### Most Similar Channels (lowest KS)

| Ch Idx | Label | KS Stat | p-value |
|--------|-------|---------|---------|
| 64 | C1 | 0.0222 | 0.9561 |
| 16 | A17 | 0.0240 | 0.9203 |
| 53 | B22 | 0.0321 | 0.6521 |
| 122 | D27 | 0.0336 | 0.5938 |
| 49 | B18 | 0.0351 | 0.5368 |

### Most Different Channels (highest KS)

| Ch Idx | Label | KS Stat | p-value |
|--------|-------|---------|---------|
| 62 | B31 | 0.2203 | 0.0000 |
| 23 | A24 | 0.2017 | 0.0000 |
| 24 | A25 | 0.1958 | 0.0000 |
| 20 | A21 | 0.1944 | 0.0000 |
| 118 | D23 | 0.1928 | 0.0000 |

## 7. Salvageability Analysis

**Amplitude threshold**: 500 µV
**Overall**: 2236/2240 trials clean (**99.8%**)

| Session | Total | Clean | Clean% | Amp P50 | Amp P90 | Amp P99 |
|---------|-------|-------|--------|---------|---------|---------|
| OfflineImagery | 640 | 638 | 99.7% | 45.1 | 143.4 | 304.8 |
| OnlineImagery_Sess01_2class_Base | 160 | 160 | 100.0% | 32.3 | 56.5 | 238.0 |
| OnlineImagery_Sess01_2class_Finetune | 160 | 160 | 100.0% | 31.8 | 48.8 | 101.1 |
| OnlineImagery_Sess01_3class_Base | 240 | 239 | 99.6% | 36.2 | 227.1 | 303.1 |
| OnlineImagery_Sess01_3class_Finetune | 240 | 240 | 100.0% | 33.5 | 55.4 | 171.2 |
| OnlineImagery_Sess02_2class_Base | 160 | 160 | 100.0% | 96.0 | 153.2 | 290.9 |
| OnlineImagery_Sess02_2class_Finetune | 160 | 160 | 100.0% | 93.5 | 136.2 | 280.7 |
| OnlineImagery_Sess02_3class_Base | 240 | 239 | 99.6% | 101.1 | 147.3 | 241.3 |
| OnlineImagery_Sess02_3class_Finetune | 240 | 240 | 100.0% | 113.5 | 171.1 | 240.7 |


---

# S21

## 1. Per-Session Overview

| Session | Runs | Trials | Labels | Amp Max | Amp P99.9 | Artifact% | Ch Var Mean | SNR (dB) |
|---------|------|--------|--------|---------|-----------|-----------|-------------|----------|
| OfflineImagery | 32 | 640 | 1:160, 2:160, 3:160, 4:160 | 2861.4 | 268.0 | 31.7% | 835.14 | -22.8 |
| OnlineImagery_Sess01_2class_Base | 8 | 160 | 1:80, 4:80 | 1545.8 | 198.2 | 15.6% | 471.53 | n/a |
| OnlineImagery_Sess01_2class_Finetune | 8 | 160 | 1:80, 4:80 | 2528.0 | 197.1 | 22.5% | 539.93 | n/a |
| OnlineImagery_Sess01_3class_Base | 8 | 240 | 1:80, 2:80, 4:80 | 1860.0 | 201.3 | 16.7% | 523.35 | n/a |
| OnlineImagery_Sess01_3class_Finetune | 8 | 240 | 1:80, 2:80, 4:80 | 1988.9 | 181.8 | 18.3% | 506.01 | n/a |
| OnlineImagery_Sess02_2class_Base | 8 | 160 | 1:80, 4:80 | 2530.5 | 224.5 | 16.2% | 616.64 | n/a |
| OnlineImagery_Sess02_2class_Finetune | 8 | 160 | 1:80, 4:80 | 2880.8 | 212.2 | 8.1% | 806.64 | n/a |
| OnlineImagery_Sess02_3class_Base | 8 | 240 | 1:80, 2:80, 4:80 | 2161.9 | 235.9 | 20.8% | 638.26 | n/a |
| OnlineImagery_Sess02_3class_Finetune | 8 | 240 | 1:80, 2:80, 4:80 | 1352.4 | 204.3 | 7.1% | 473.96 | n/a |

## 2. Run-Level Amplitude Profile

### Top 10 Worst Runs (by max amplitude)

| Session | Run | Trials | Amp Max | Amp Mean | Trial P95 | >500 | >1000 |
|---------|-----|--------|---------|----------|----------|------|-------|
| OnlineImagery_Sess02_2class_Finetune | 7 | 20 | 2880.8 | 15.84 | 1817.0 | 5 | 2 |
| OfflineImagery | 27 | 20 | 2861.4 | 13.40 | 2732.0 | 7 | 6 |
| OfflineImagery | 30 | 20 | 2794.4 | 12.80 | 2714.3 | 6 | 5 |
| OfflineImagery | 29 | 20 | 2611.6 | 13.70 | 1583.7 | 8 | 3 |
| OnlineImagery_Sess02_2class_Base | 2 | 20 | 2530.5 | 12.19 | 1006.4 | 5 | 1 |
| OnlineImagery_Sess01_2class_Finetune | 7 | 20 | 2528.0 | 12.10 | 867.6 | 4 | 1 |
| OfflineImagery | 8 | 20 | 2466.2 | 13.36 | 2279.6 | 6 | 3 |
| OfflineImagery | 26 | 20 | 2239.5 | 13.56 | 1405.1 | 8 | 3 |
| OfflineImagery | 25 | 20 | 2202.7 | 13.33 | 840.5 | 4 | 1 |
| OfflineImagery | 28 | 20 | 2175.4 | 13.51 | 1666.5 | 9 | 4 |

### Per-Session Run Summary

| Session | N Runs | Median AmpMax | Max AmpMax | Total >500 | Total >1000 |
|---------|--------|---------------|------------|-----------|-------------|
| OfflineImagery | 32 | 1496.9 | 2861.4 | 203 | 75 |
| OnlineImagery_Sess01_2class_Base | 8 | 1380.9 | 1545.8 | 25 | 10 |
| OnlineImagery_Sess01_2class_Finetune | 8 | 1537.6 | 2528.0 | 36 | 13 |
| OnlineImagery_Sess01_3class_Base | 8 | 1206.7 | 1860.0 | 40 | 15 |
| OnlineImagery_Sess01_3class_Finetune | 8 | 1480.0 | 1988.9 | 44 | 19 |
| OnlineImagery_Sess02_2class_Base | 8 | 1480.9 | 2530.5 | 26 | 12 |
| OnlineImagery_Sess02_2class_Finetune | 8 | 1096.7 | 2880.8 | 13 | 5 |
| OnlineImagery_Sess02_3class_Base | 8 | 1680.5 | 2161.9 | 50 | 25 |
| OnlineImagery_Sess02_3class_Finetune | 8 | 805.2 | 1352.4 | 17 | 2 |

## 3. Channel Quality Map

Sampled 500 trials.

### Worst 10 Channels by Variance

| Rank | Ch Idx | Label | Mean Var | Region |
|------|--------|-------|----------|--------|
| 1 | 92 | C29 | 9518.5137 | peripheral |
| 2 | 93 | C30 | 8183.9180 | peripheral |
| 3 | 80 | C17 | 7196.3296 | peripheral |
| 4 | 91 | C28 | 6084.0698 | peripheral |
| 5 | 81 | C18 | 3806.4412 | peripheral |
| 6 | 79 | C16 | 3301.5674 | peripheral |
| 7 | 94 | C31 | 3071.7795 | peripheral |
| 8 | 71 | C8 | 2716.3438 | peripheral |
| 9 | 90 | C27 | 2713.2156 | central |
| 10 | 72 | C9 | 2653.6946 | peripheral |

### Worst 10 Channels by Max Amplitude

| Rank | Ch Idx | Label | Mean Max Amp | Region |
|------|--------|-------|-------------|--------|
| 1 | 92 | C29 | 336.84 | peripheral |
| 2 | 93 | C30 | 298.05 | peripheral |
| 3 | 80 | C17 | 289.18 | peripheral |
| 4 | 91 | C28 | 247.23 | peripheral |
| 5 | 81 | C18 | 211.68 | peripheral |
| 6 | 79 | C16 | 194.77 | peripheral |
| 7 | 94 | C31 | 189.09 | peripheral |
| 8 | 71 | C8 | 185.67 | peripheral |
| 9 | 72 | C9 | 183.34 | peripheral |
| 10 | 90 | C27 | 159.58 | central |

**Central vs Peripheral**:
- Variance: central=224.9971, peripheral=770.0861 (ratio=3.42x)
- Max Amp: central=40.59, peripheral=71.80
- Mu/Beta Ratio: central=1.113, peripheral=1.142

## 4. Per-Session Class Discriminability

| Session | Classes | Fisher Mean | AUROC Mean (top5) | Class Counts |
|---------|---------|-----------|-----------------|-------------|
| OfflineImagery | 4 | 0.0088 | 0.5865 | 1:160, 2:160, 3:160, 4:160 |
| OnlineImagery_Sess01_2class_Base | 2 | 0.0068 | 0.5275 | 1:80, 4:80 |
| OnlineImagery_Sess01_2class_Finetune | 2 | 0.0160 | 0.5983 | 1:80, 4:80 |
| OnlineImagery_Sess01_3class_Base | 3 | 0.0094 | 0.5559 | 1:80, 2:80, 4:80 |
| OnlineImagery_Sess01_3class_Finetune | 3 | 0.0118 | 0.6010 | 1:80, 2:80, 4:80 |
| OnlineImagery_Sess02_2class_Base | 2 | 0.0135 | 0.6060 | 1:80, 4:80 |
| OnlineImagery_Sess02_2class_Finetune | 2 | 0.0524 | 0.7057 | 1:80, 4:80 |
| OnlineImagery_Sess02_3class_Base | 3 | 0.0122 | 0.5927 | 1:80, 2:80, 4:80 |
| OnlineImagery_Sess02_3class_Finetune | 3 | 0.0259 | 0.6246 | 1:80, 2:80, 4:80 |

## 5. Per-Session Spectral Profile

| Session | Theta | Mu | Low-β | High-β | γ | Mu/β Ratio |
|---------|-------|-----|-------|--------|---|-----------|
| OfflineImagery | 20.9018 | 2.4818 | 1.2140 | 0.9836 | 0.5799 | 1.129 |
| OnlineImagery_Sess01_2class_Base | 11.0929 | 1.7301 | 1.1165 | 0.9846 | 0.5614 | 0.823 |
| OnlineImagery_Sess01_2class_Finetune | 11.8436 | 2.0722 | 1.1574 | 1.0180 | 0.6357 | 0.953 |
| OnlineImagery_Sess01_3class_Base | 12.0929 | 1.9305 | 1.0169 | 0.7563 | 0.5141 | 1.089 |
| OnlineImagery_Sess01_3class_Finetune | 9.8317 | 1.9440 | 1.0209 | 0.8023 | 0.4446 | 1.066 |
| OnlineImagery_Sess02_2class_Base | 18.6604 | 2.2746 | 1.3610 | 1.1571 | 0.6501 | 0.903 |
| OnlineImagery_Sess02_2class_Finetune | 14.4015 | 2.2729 | 1.3161 | 1.1092 | 0.6346 | 0.937 |
| OnlineImagery_Sess02_3class_Base | 17.1161 | 2.1950 | 1.3365 | 1.2078 | 0.6894 | 0.863 |
| OnlineImagery_Sess02_3class_Finetune | 12.7689 | 2.0364 | 1.2746 | 1.0803 | 0.5992 | 0.865 |

## 6. Train/Test Distribution Comparison

- Train trials: 1440, Test trials: 800
- Train global mean: -0.0000 (std=3.5159)
- Test global mean: 0.0000 (std=5.0748)
- **Mean KS statistic**: 0.1150
- Median KS statistic: 0.1093
- Max KS statistic: 0.2675
- Channels where train/test indistinguishable (p>0.05): **13/128**

### Most Similar Channels (lowest KS)

| Ch Idx | Label | KS Stat | p-value |
|--------|-------|---------|---------|
| 105 | D10 | 0.0328 | 0.6255 |
| 123 | D28 | 0.0360 | 0.5065 |
| 64 | C1 | 0.0383 | 0.4253 |
| 61 | B30 | 0.0403 | 0.3643 |
| 63 | B32 | 0.0426 | 0.2982 |

### Most Different Channels (highest KS)

| Ch Idx | Label | KS Stat | p-value |
|--------|-------|---------|---------|
| 24 | A25 | 0.2675 | 0.0000 |
| 23 | A24 | 0.2387 | 0.0000 |
| 111 | D16 | 0.2274 | 0.0000 |
| 89 | C26 | 0.2158 | 0.0000 |
| 101 | D6 | 0.2096 | 0.0000 |

## 7. Salvageability Analysis

**Amplitude threshold**: 500 µV
**Overall**: 1786/2240 trials clean (**79.7%**)

| Session | Total | Clean | Clean% | Amp P50 | Amp P90 | Amp P99 |
|---------|-------|-------|--------|---------|---------|---------|
| OfflineImagery | 640 | 437 | 68.3% | 333.7 | 1061.6 | 2258.0 |
| OnlineImagery_Sess01_2class_Base | 160 | 135 | 84.4% | 218.2 | 743.8 | 1447.0 |
| OnlineImagery_Sess01_2class_Finetune | 160 | 124 | 77.5% | 209.4 | 825.4 | 1791.1 |
| OnlineImagery_Sess01_3class_Base | 240 | 200 | 83.3% | 251.4 | 686.8 | 1527.9 |
| OnlineImagery_Sess01_3class_Finetune | 240 | 196 | 81.7% | 191.0 | 871.1 | 1632.8 |
| OnlineImagery_Sess02_2class_Base | 160 | 134 | 83.8% | 260.3 | 835.2 | 1578.8 |
| OnlineImagery_Sess02_2class_Finetune | 160 | 147 | 91.9% | 250.3 | 449.2 | 1893.0 |
| OnlineImagery_Sess02_3class_Base | 240 | 190 | 79.2% | 274.4 | 1014.1 | 1797.9 |
| OnlineImagery_Sess02_3class_Finetune | 240 | 223 | 92.9% | 262.7 | 428.0 | 984.0 |
