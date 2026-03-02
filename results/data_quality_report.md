# EEG Data Quality Report

Generated: 2026-03-01 20:52:50
Cache: `caches/preprocessed/.cache_index.json` (EEGNet entries only)
Subjects: 21 (S01-S21)
Analysis time: 25.8s

## Executive Summary

- **Clean subjects**: 10/21
- **Info (minor notes)**: 3/21
- **Minor issues**: 5/21
- **Major issues**: 3/21
- **Critical issues**: 0/21

## Flagged Subjects

| Subject | Severity | Issues |
|---------|----------|--------|
| S03 | minor | moderate_artifacts (7.0% trials) |
| S04 | major | severe_artifacts (max=306796), trial_variance_very_high, unstable_variance (1367714x) |
| S05 | minor | moderate_artifacts (5.8% trials) |
| S09 | minor | moderate_artifacts (6.8% trials) |
| S10 | major | severe_artifacts (max=267904), trial_variance_very_high, unstable_variance (822583x) |
| S12 | info | unstable_variance (20x) |
| S14 | major | severe_artifacts (max=125503), trial_variance_very_high, unstable_variance (68x) |
| S16 | minor | moderate_artifacts (5.7% trials), unstable_variance (15x) |
| S19 | info | unstable_variance (65x) |
| S20 | info | unstable_variance (37x) |
| S21 | minor | moderate_artifacts (9.4% trials) |

## Statistical Outliers (|z| > 2.5 from group mean)

| Subject | Metric | Detail |
|---------|--------|--------|
| S04 | max_amplitude_high | (z=3.2, val=306795.75, group=37838.67±85252.02) |
| S06 | total_trials_low | (z=4.3, val=1900.00, group=2204.95±70.66) |
| S10 | trial_CV_high | (z=2.8, val=4.54, group=1.14±1.22) |
| S10 | max_amplitude_high | (z=2.7, val=267903.78, group=37838.67±85252.02) |

## Per-Subject Overview

| Subject | Trials | Runs | Sessions | Dead Ch | Max|Amp| | SNR (dB) | CV | Dup | Severity |
|---------|--------|------|----------|---------|---------|----------|------|-----|----------|
| S01 | 2200 | 94 | 9 | 0 | 3238.1 | -16.0 | 0.607 | 0 | clean |
| S02 | 2200 | 94 | 9 | 0 | 593.8 | -12.7 | 0.463 | 0 | clean |
| S03 | 2200 | 94 | 9 | 0 | 730.2 | -14.9 | 0.362 | 0 | minor |
| S04 | 2240 | 96 | 9 | 0 | 306795.8 | -21.8 | 3.587 | 0 | major |
| S05 | 2240 | 96 | 9 | 0 | 1088.9 | -15.1 | 0.320 | 0 | minor |
| S06 | 1900 | 79 | 9 | 0 | 2092.2 | -15.9 | 0.567 | 0 | clean |
| S07 | 2200 | 94 | 9 | 0 | 4632.9 | -17.4 | 0.715 | 0 | clean |
| S08 | 2200 | 94 | 9 | 0 | 1097.7 | -18.8 | 0.321 | 0 | clean |
| S09 | 2200 | 94 | 9 | 0 | 5253.7 | -11.2 | 0.607 | 0 | minor |
| S10 | 2200 | 94 | 9 | 0 | 267903.8 | -20.3 | 4.539 | 0 | major |
| S11 | 2200 | 94 | 9 | 0 | 7139.5 | -14.3 | 0.678 | 0 | clean |
| S12 | 2200 | 94 | 9 | 0 | 8164.3 | -16.1 | 1.023 | 0 | info |
| S13 | 2220 | 95 | 9 | 0 | 981.5 | -17.5 | 0.545 | 0 | clean |
| S14 | 2224 | 96 | 9 | 0 | 125503.4 | -19.8 | 3.898 | 0 | major |
| S15 | 2240 | 96 | 9 | 0 | 9127.3 | -13.3 | 0.401 | 0 | clean |
| S16 | 2240 | 96 | 9 | 0 | 2277.2 | -14.7 | 0.562 | 0 | minor |
| S17 | 2240 | 96 | 9 | 0 | 5994.9 | -16.5 | 0.607 | 0 | clean |
| S18 | 2240 | 96 | 9 | 0 | 2180.2 | -16.9 | 0.492 | 0 | clean |
| S19 | 2240 | 96 | 9 | 0 | 22970.4 | -9.5 | 1.524 | 0 | info |
| S20 | 2240 | 96 | 9 | 0 | 13965.6 | -18.4 | 1.537 | 0 | info |
| S21 | 2240 | 96 | 9 | 0 | 2880.8 | -14.5 | 0.546 | 0 | minor |

## Detailed Analysis

### 1. Signal Quality

#### NaN/Inf Analysis

Note: NaN padding at the trailing end of trials is **expected** (variable-length trial padding). Only signal-region NaN/Inf is flagged.

| Subject | Padding NaN Trials | Padding % | Signal NaN Trials | Inf Count |
|---------|-------------------|-----------|-------------------|-----------|
| S01 | 1600 | 72.7% | 0 | 0 |
| S02 | 1600 | 72.7% | 0 | 0 |
| S03 | 1600 | 72.7% | 0 | 0 |
| S04 | 1600 | 71.4% | 0 | 0 |
| S05 | 1600 | 71.4% | 0 | 0 |
| S06 | 1600 | 84.2% | 0 | 0 |
| S07 | 1600 | 72.7% | 0 | 0 |
| S08 | 1600 | 72.7% | 0 | 0 |
| S09 | 1600 | 72.7% | 0 | 0 |
| S10 | 1600 | 72.7% | 0 | 0 |
| S11 | 1600 | 72.7% | 0 | 0 |
| S12 | 1600 | 72.7% | 0 | 0 |
| S13 | 1600 | 72.1% | 0 | 0 |
| S14 | 1584 | 71.2% | 0 | 0 |
| S15 | 1600 | 71.4% | 0 | 0 |
| S16 | 1600 | 71.4% | 0 | 0 |
| S17 | 1600 | 71.4% | 0 | 0 |
| S18 | 1600 | 71.4% | 0 | 0 |
| S19 | 1600 | 71.4% | 0 | 0 |
| S20 | 1600 | 71.4% | 0 | 0 |
| S21 | 1600 | 71.4% | 0 | 0 |

No signal-region NaN or Inf detected. All NaN values are expected padding.

#### Dead/Flat Channels

No dead channels detected (variance threshold: 0.01).

#### Extreme Amplitudes

| Subject | Max|Amp| | P50 | P95 | P99 | P99.9 | Extreme Trials |
|---------|---------|-----|-----|-----|-------|----------------|
| S01 | 3238.1 | 23.0 | 83.0 | 213.1 | 295.9 | 20 |
| S02 | 593.8 | 19.1 | 70.4 | 155.7 | 235.8 | 35 |
| S03 | 730.2 | 14.0 | 32.8 | 58.2 | 151.8 | 154 |
| S04 | 306795.8 | 30.4 | 7314.9 | 116726.2 | 188688.9 | 130 |
| S05 | 1088.9 | 16.7 | 43.9 | 93.0 | 189.5 | 130 |
| S06 | 2092.2 | 16.4 | 52.9 | 116.7 | 194.6 | 51 |
| S07 | 4632.9 | 17.5 | 43.9 | 91.1 | 154.4 | 14 |
| S08 | 1097.7 | 21.7 | 74.9 | 127.5 | 190.6 | 32 |
| S09 | 5253.7 | 38.9 | 167.8 | 294.5 | 668.6 | 149 |
| S10 | 267903.8 | 24.6 | 241.4 | 95833.1 | 182292.5 | 100 |
| S11 | 7139.5 | 26.6 | 92.2 | 183.1 | 319.5 | 64 |
| S12 | 8164.3 | 20.4 | 74.1 | 116.8 | 175.7 | 24 |
| S13 | 981.5 | 25.5 | 94.8 | 157.9 | 315.1 | 57 |
| S14 | 125503.4 | 21.1 | 89.4 | 171.3 | 429.6 | 6 |
| S15 | 9127.3 | 25.7 | 95.1 | 149.7 | 205.7 | 5 |
| S16 | 2277.2 | 31.0 | 113.3 | 160.2 | 249.1 | 127 |
| S17 | 5994.9 | 18.9 | 66.6 | 111.7 | 323.8 | 70 |
| S18 | 2180.2 | 32.1 | 128.8 | 209.4 | 292.2 | 15 |
| S19 | 22970.4 | 31.0 | 128.1 | 218.0 | 576.7 | 83 |
| S20 | 13965.6 | 34.4 | 148.5 | 232.1 | 1001.3 | 7 |
| S21 | 2880.8 | 45.6 | 177.2 | 355.8 | 1130.3 | 210 |

#### Signal-to-Noise Ratio (dB)

| Subject | Mean SNR | Thumb | Index | Middle | Pinky |
|---------|----------|-------|-------|--------|-------|
| S01 | -16.03 | -16.43 | -15.55 | -16.94 | -15.18 |
| S02 | -12.69 | -12.25 | -12.80 | -13.62 | -12.11 |
| S03 | -14.86 | -14.00 | -14.89 | -16.37 | -14.20 |
| S04 | -21.80 | -21.73 | -21.66 | -21.97 | -21.86 |
| S05 | -15.14 | -14.28 | -16.89 | -16.07 | -13.30 |
| S06 | -15.95 | -15.99 | -17.53 | -16.39 | -13.88 |
| S07 | -17.44 | -17.15 | -18.02 | -17.89 | -16.72 |
| S08 | -18.80 | -18.24 | -18.86 | -19.78 | -18.31 |
| S09 | -11.15 | -11.38 | -11.39 | -11.36 | -10.48 |
| S10 | -20.27 | -20.36 | -20.24 | -20.24 | -20.25 |
| S11 | -14.35 | -13.02 | -14.47 | -15.66 | -14.24 |
| S12 | -16.12 | -15.34 | -15.75 | -17.06 | -16.34 |
| S13 | -17.48 | -17.63 | -17.95 | -17.42 | -16.92 |
| S14 | -19.81 | -18.89 | -19.86 | -20.77 | -19.72 |
| S15 | -13.28 | -13.58 | -14.14 | -13.08 | -12.34 |
| S16 | -14.71 | -14.70 | -15.09 | -15.20 | -13.84 |
| S17 | -16.53 | -15.22 | -17.36 | -18.00 | -15.56 |
| S18 | -16.86 | -17.12 | -18.91 | -15.79 | -15.62 |
| S19 | -9.51 | -5.16 | -6.67 | -11.88 | -14.35 |
| S20 | -18.39 | -17.78 | -19.27 | -19.18 | -17.32 |
| S21 | -14.48 | -13.16 | -14.00 | -15.91 | -14.85 |

### 2. Statistical Anomalies

#### Inter-Trial Variance (Coefficient of Variation)

| Subject | Overall CV | Flag |
|---------|-----------|------|
| S01 | 0.6066 | - |
| S02 | 0.4634 | - |
| S03 | 0.3618 | - |
| S04 | 3.5872 | very_high |
| S05 | 0.3199 | - |
| S06 | 0.5675 | - |
| S07 | 0.7145 | - |
| S08 | 0.3213 | - |
| S09 | 0.6067 | - |
| S10 | 4.5391 | very_high |
| S11 | 0.6785 | - |
| S12 | 1.0232 | - |
| S13 | 0.5451 | - |
| S14 | 3.8976 | very_high |
| S15 | 0.4008 | - |
| S16 | 0.5622 | - |
| S17 | 0.6066 | - |
| S18 | 0.4917 | - |
| S19 | 1.5239 | - |
| S20 | 1.5366 | - |
| S21 | 0.5460 | - |

#### Inter-Channel Correlation

| Subject | Mean |r| | Max |r| | High Pairs (>0.9) | Flag |
|---------|---------|---------|-------------------|------|
| S01 | 0.5124 | 0.9994 | 262 | - |
| S02 | 0.4806 | 0.9995 | 246 | - |
| S03 | 0.3291 | 0.9931 | 51 | - |
| S04 | 0.3761 | 0.9983 | 129 | - |
| S05 | 0.3527 | 0.9974 | 58 | - |
| S06 | 0.5391 | 0.9998 | 385 | - |
| S07 | 0.3941 | 0.9975 | 127 | - |
| S08 | 0.2935 | 0.9993 | 9 | - |
| S09 | 0.6218 | 0.9997 | 869 | - |
| S10 | 0.3089 | 0.9988 | 34 | - |
| S11 | 0.3408 | 0.9982 | 45 | - |
| S12 | 0.3899 | 0.9968 | 77 | - |
| S13 | 0.4701 | 0.9997 | 357 | - |
| S14 | 0.4155 | 0.9980 | 112 | - |
| S15 | 0.4688 | 0.9996 | 321 | - |
| S16 | 0.2715 | 0.9985 | 69 | - |
| S17 | 0.3977 | 0.9975 | 146 | - |
| S18 | 0.5209 | 0.9997 | 459 | - |
| S19 | 0.4109 | 0.9992 | 211 | - |
| S20 | 0.4946 | 0.9988 | 271 | - |
| S21 | 0.5767 | 0.9998 | 1051 | - |

#### Label Distribution

Note: Cross-session label imbalance is **expected** (Offline has 4 classes, Online 2class has only classes 1,4). Checking within-session balance.

| Subject | Overall Distribution | Within-Session Imbalance |
|---------|---------------------|--------------------------|
| S01 | 1:790, 2:470, 3:150, 4:790 | - |
| S02 | 1:790, 2:470, 3:150, 4:790 | - |
| S03 | 1:790, 2:470, 3:150, 4:790 | - |
| S04 | 1:800, 2:480, 3:160, 4:800 | - |
| S05 | 1:800, 2:480, 3:160, 4:800 | - |
| S06 | 1:715, 2:395, 3:75, 4:715 | - |
| S07 | 1:790, 2:470, 3:150, 4:790 | - |
| S08 | 1:790, 2:470, 3:150, 4:790 | - |
| S09 | 1:790, 2:470, 3:150, 4:790 | - |
| S10 | 1:790, 2:470, 3:150, 4:790 | - |
| S11 | 1:790, 2:470, 3:150, 4:790 | - |
| S12 | 1:790, 2:470, 3:150, 4:790 | - |
| S13 | 1:795, 2:475, 3:155, 4:795 | - |
| S14 | 1:791, 2:480, 3:160, 4:793 | - |
| S15 | 1:800, 2:480, 3:160, 4:800 | - |
| S16 | 1:800, 2:480, 3:160, 4:800 | - |
| S17 | 1:800, 2:480, 3:160, 4:800 | - |
| S18 | 1:800, 2:480, 3:160, 4:800 | - |
| S19 | 1:800, 2:480, 3:160, 4:800 | - |
| S20 | 1:800, 2:480, 3:160, 4:800 | - |
| S21 | 1:800, 2:480, 3:160, 4:800 | - |

#### Trial Counts per Session

| Subject | Total Trials | Total Runs | Sessions | Anomalous Runs |
|---------|-------------|------------|----------|----------------|
| S01 | 2200 | 94 | OfflineImagery(30), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S02 | 2200 | 94 | OfflineImagery(30), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S03 | 2200 | 94 | OfflineImagery(30), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S04 | 2240 | 96 | OfflineImagery(32), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S05 | 2240 | 96 | OfflineImagery(32), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S06 | 1900 | 79 | OfflineImagery(15), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S07 | 2200 | 94 | OfflineImagery(30), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S08 | 2200 | 94 | OfflineImagery(30), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S09 | 2200 | 94 | OfflineImagery(30), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S10 | 2200 | 94 | OfflineImagery(30), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S11 | 2200 | 94 | OfflineImagery(30), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S12 | 2200 | 94 | OfflineImagery(30), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S13 | 2220 | 95 | OfflineImagery(31), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S14 | 2224 | 96 | OfflineImagery(32), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S15 | 2240 | 96 | OfflineImagery(32), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S16 | 2240 | 96 | OfflineImagery(32), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S17 | 2240 | 96 | OfflineImagery(32), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S18 | 2240 | 96 | OfflineImagery(32), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S19 | 2240 | 96 | OfflineImagery(32), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S20 | 2240 | 96 | OfflineImagery(32), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |
| S21 | 2240 | 96 | OfflineImagery(32), OnlineImagery_Sess01_2class_Base(8), OnlineImagery_Sess01_2class_Finetune(8), OnlineImagery_Sess01_3class_Base(8), OnlineImagery_Sess01_3class_Finetune(8), OnlineImagery_Sess02_2class_Base(8), OnlineImagery_Sess02_2class_Finetune(8), OnlineImagery_Sess02_3class_Base(8), OnlineImagery_Sess02_3class_Finetune(8) | - |

### 3. Cross-Session Consistency

#### Session Amplitude Shift (L2 distance of channel means)

| Subject | Max Shift | Max Shift Pair |
|---------|-----------|---------------|
| S01 | 10.9099 | OfflineImagery vs OnlineImagery_Sess02_3class_Base |
| S02 | 9.2936 | OnlineImagery_Sess01_3class_Base vs OnlineImagery_Sess02_2class_Base |
| S03 | 6.4668 | OnlineImagery_Sess01_3class_Base vs OnlineImagery_Sess02_3class_Base |
| S04 | 148.2239 | OfflineImagery vs OnlineImagery_Sess01_3class_Base |
| S05 | 16.1604 | OnlineImagery_Sess01_3class_Finetune vs OnlineImagery_Sess02_3class_Finetune |
| S06 | 8.5896 | OfflineImagery vs OnlineImagery_Sess01_3class_Finetune |
| S07 | 8.8004 | OnlineImagery_Sess02_2class_Base vs OnlineImagery_Sess02_3class_Base |
| S08 | 6.2667 | OnlineImagery_Sess01_3class_Base vs OnlineImagery_Sess01_3class_Finetune |
| S09 | 56.2485 | OnlineImagery_Sess01_2class_Finetune vs OnlineImagery_Sess01_3class_Base |
| S10 | 32.9055 | OfflineImagery vs OnlineImagery_Sess02_3class_Base |
| S11 | 21.4667 | OfflineImagery vs OnlineImagery_Sess02_2class_Base |
| S12 | 9.8357 | OnlineImagery_Sess02_3class_Base vs OnlineImagery_Sess02_3class_Finetune |
| S13 | 7.1512 | OfflineImagery vs OnlineImagery_Sess01_3class_Base |
| S14 | 10.8283 | OnlineImagery_Sess01_2class_Base vs OnlineImagery_Sess02_2class_Finetune |
| S15 | 12.2272 | OnlineImagery_Sess01_2class_Finetune vs OnlineImagery_Sess02_3class_Finetune |
| S16 | 45.8264 | OnlineImagery_Sess01_3class_Base vs OnlineImagery_Sess02_2class_Base |
| S17 | 8.1813 | OnlineImagery_Sess01_3class_Base vs OnlineImagery_Sess01_3class_Finetune |
| S18 | 5.8260 | OnlineImagery_Sess01_2class_Finetune vs OnlineImagery_Sess02_3class_Base |
| S19 | 32.5933 | OfflineImagery vs OnlineImagery_Sess02_2class_Finetune |
| S20 | 7.7402 | OnlineImagery_Sess01_2class_Finetune vs OnlineImagery_Sess02_3class_Base |
| S21 | 11.7194 | OnlineImagery_Sess01_2class_Base vs OnlineImagery_Sess01_3class_Base |

#### Session Variance Consistency

| Subject | Variance Ratio (max/min) | Flag |
|---------|------------------------|------|
| S01 | 4.57 | - |
| S02 | 3.11 | - |
| S03 | 1.73 | - |
| S04 | 1367713.58 | unstable_variance |
| S05 | 2.65 | - |
| S06 | 4.85 | - |
| S07 | 5.95 | - |
| S08 | 1.85 | - |
| S09 | 5.32 | - |
| S10 | 822582.97 | unstable_variance |
| S11 | 2.71 | - |
| S12 | 20.23 | unstable_variance |
| S13 | 2.44 | - |
| S14 | 68.15 | unstable_variance |
| S15 | 3.43 | - |
| S16 | 15.26 | unstable_variance |
| S17 | 3.32 | - |
| S18 | 2.94 | - |
| S19 | 64.53 | unstable_variance |
| S20 | 37.40 | unstable_variance |
| S21 | 1.77 | - |

### 4. Contamination Checks

#### Duplicate Trials (cosine similarity > 0.999)

No duplicate trials detected.

#### Train/Test Distribution Similarity (KS Test)

| Subject | Mean KS Stat | Similar Ch | Different Ch | Interpretation |
|---------|-------------|------------|--------------|----------------|
| S01 | 0.1154 | 37 | 91 | moderately_similar (normal) |
| S02 | 0.1202 | 26 | 102 | moderately_similar (normal) |
| S03 | 0.1176 | 32 | 96 | moderately_similar (normal) |
| S04 | 0.1325 | 20 | 108 | moderately_similar (normal) |
| S05 | 0.2901 | 6 | 122 | moderately_similar (normal) |
| S06 | 0.1015 | 43 | 85 | moderately_similar (normal) |
| S07 | 0.1369 | 21 | 107 | moderately_similar (normal) |
| S08 | 0.1216 | 26 | 102 | moderately_similar (normal) |
| S09 | 0.1034 | 34 | 94 | moderately_similar (normal) |
| S10 | 0.1157 | 7 | 121 | moderately_similar (normal) |
| S11 | 0.1479 | 24 | 104 | moderately_similar (normal) |
| S12 | 0.2268 | 2 | 126 | moderately_similar (normal) |
| S13 | 0.1064 | 40 | 88 | moderately_similar (normal) |
| S14 | 0.1082 | 39 | 89 | moderately_similar (normal) |
| S15 | 0.1915 | 17 | 111 | moderately_similar (normal) |
| S16 | 0.2460 | 2 | 126 | moderately_similar (normal) |
| S17 | 0.1291 | 25 | 103 | moderately_similar (normal) |
| S18 | 0.0889 | 54 | 74 | very_similar (potential concern) |
| S19 | 0.1139 | 35 | 93 | moderately_similar (normal) |
| S20 | 0.0940 | 45 | 83 | very_similar (potential concern) |
| S21 | 0.1233 | 22 | 106 | moderately_similar (normal) |

## Key Findings and Recommendations

### Major Issues (3 subjects)

**S04** — 最严重的问题被试。最大振幅 306,796 (正常被试 < 10,000)，说明存在严重的电极脱落或运动伪迹。方差比 1,367,714x 意味着某些 session 的数据质量远低于其他 session。CV = 3.59 表示 trial 间变异极大。**建议：在跨被试训练中考虑排除或降权 S04 数据。**

**S10** — 与 S04 类似的严重伪迹问题（max amplitude = 267,904）。方差比 822,583x，CV = 4.54（21 个被试中最高）。SNR = -20.3 dB（倒数第三差）。**建议：同样考虑排除或降权。**

**S14** — 最大振幅 125,503，方差比 68x，CV = 3.90。严重程度低于 S04/S10 但仍显著异常。SNR = -19.8 dB。**建议：谨慎使用，可考虑 artifact rejection。**

### Minor Issues (5 subjects)

**S03, S05, S09, S16, S21** — 存在中等程度的伪迹（5-9% 的 trial 超过 10σ 阈值）。这在 EEG 实验中属于常见范围，但比干净被试（如 S02, S08）高出一个量级。后续 z-score 归一化和滑动窗口分割可以部分缓解。

### Info (3 subjects)

**S12, S19, S20** — 跨 session 方差不稳定（20-65x ratio），但无极端振幅。可能是不同 session 间电极阻抗变化导致。对训练的影响有限。

### Data Contamination Assessment

1. **无 NaN/Inf 信号污染** — 所有 NaN 均为预期的 trial 长度填充
2. **无死通道** — 128 通道全部正常工作
3. **无重复 trial** — 所有 trial 都是唯一的
4. **Train/Test 分布合理** — 大多数被试 train 和 test 数据分布适度不同（KS statistic 0.09-0.29），符合不同 session 采集的预期
5. **S06 数据量偏少** — 仅 1,900 trials（其他被试 2,200+），因 Offline 仅 15 runs（vs 标准 30）
6. **S18, S20 train/test 非常相似** — KS statistic < 0.1，54/45 个通道不可区分。这可能导致验证集上的性能高估

### Impact on Cross-Subject Training

**高风险被试**：S04、S10、S14 的极端伪迹 trial 可能在跨被试训练中引入噪声，影响模型收敛。建议：
- 在 artifact rejection 阶段增加基于振幅阈值的 trial 筛选
- 或在跨被试训练时降低这 3 个被试的采样权重
- 监控这 3 个被试对整体 validation loss 的贡献

**中等风险**：S03、S05、S09、S16、S21 的中等伪迹在 z-score 归一化后可能被缓解，但建议监控。

## Methodology

- **Data source**: HDF5 cache files (post-CAR, post-bandpass 4-40 Hz, downsampled to 100 Hz, pre-z-score)
- **Model filter**: EEGNet entries only (avoid double-counting with CBraMod)
- **Dead channel threshold**: Variance < 0.01 in >50% of runs
- **Extreme amplitude threshold**: |value| > mean + 10*std
- **SNR**: Inter-trial coherence (signal = ERP variance, noise = residual variance)
- **Duplicate detection**: Cosine similarity > 0.999 between same-label trials
- **Train/test similarity**: 2-sample KS test per channel on trial-mean amplitudes
- **Outlier detection**: |z-score| > 2.5 from group mean across subjects
