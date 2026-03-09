# EEG Foundation Models Enable Robust Finger-Level Motor Imagery Decoding with Reduced Channel Configurations

> **草稿说明**：本文为工作草稿。文中大部分图表为脚本自动生成的初步输出，**尚未进行出版级精修**（坐标轴标签、字体大小、配色方案、排版布局等）。标有 `[TODO]` 的章节为尚未完成的实验，其结果将在最终稿中补充。

---

## Abstract

Brain-computer interfaces (BCIs) that decode individual finger movements from electroencephalography (EEG) hold significant promise for fine motor rehabilitation, yet their deployment is hampered by the need for high-density electrode arrays. In this study, we systematically compare a large-scale EEG foundation model, CBraMod (~4M parameters, ICLR 2025), against the widely adopted EEGNet-8,2 (~2.5K parameters) for binary (thumb vs. pinky) and ternary (thumb, middle, pinky) motor imagery classification across 21 healthy participants using a 128-channel BioSemi system. We evaluate three training paradigms—within-subject, cross-subject, and cross-subject-to-individual transfer learning—and conduct a comprehensive channel reduction study spanning 128, 61, 32, 8, and 4 channels using four data-driven selection methods (Fisher Discriminant Ratio, Common Spatial Patterns, gradient-based attention, spectral band power) and two hand-crafted layouts. CBraMod achieves 90.27% cross-subject binary accuracy at 128 channels; critically, a 32-channel configuration selected by Fisher Discriminant Ratio retains 88.10% (only −2.17 percentage points), while EEGNet drops to 67.53% under the same conditions. We further show that transfer learning benefit scales inversely with channel count—negligible at 128 channels but +4.59 pp at 8 channels—and that control experiments with complementary and random channel sets (83.18% and 84.08%) confirm volume conduction–driven information redundancy rather than data leakage. These results demonstrate that combining an EEG foundation model with data-driven channel selection yields a practical, deployable 32-channel BCI system for finger-level motor imagery decoding.

**Keywords:** brain-computer interface, electroencephalography, motor imagery, foundation model, CBraMod, EEGNet, channel reduction, transfer learning, Fisher Discriminant Ratio, volume conduction

---

## 1. Introduction

### 1.1 Background and Motivation

Brain-computer interfaces (BCIs) provide a direct communication pathway between the brain and external devices, offering transformative potential for individuals with severe motor disabilities [1]. Among non-invasive BCI paradigms, motor imagery (MI)—the mental rehearsal of movement without actual execution—has emerged as a leading approach due to its volitional nature and clinical feasibility [2].

While most MI-BCI research targets broad motor categories such as left-hand versus right-hand imagery, recent work has pushed toward finer-grained decoding at the individual finger level [3]. Such fine motor control is essential for practical prosthetic and robotic hand applications, yet it poses substantially greater challenges: finger-specific cortical representations are spatially adjacent and produce weaker, more overlapping EEG signatures compared to gross limb movements.

A persistent barrier to deploying MI-BCI systems in clinical or consumer settings is the reliance on high-density EEG arrays. Research-grade systems with 64–256 channels provide rich spatial information but entail prohibitive setup times, user discomfort, and hardware costs. Conversely, reducing the electrode count risks degrading decoding performance below usable thresholds. Understanding the performance–channel count trade-off is therefore critical to bridging the gap between laboratory results and real-world BCI applications.

### 1.2 Related Work on Finger-Level EEG Decoding

Individual finger movement decoding from scalp EEG has progressed from offline feasibility studies to real-time robotic control. Table 0 summarizes key prior work and positions our contribution.

**Table 0. Comparison with prior finger-level EEG decoding studies.**

| Study | Model | Channels | Evaluation Setting | Binary Acc. | Ternary Acc. | Real-Time |
|-------|-------|----------|--------------------|-------------|-------------|-----------|
| Alazrai et al. 2019 [8] | SVM + CSP | 64 | Offline, within-subject | ~65% | N/A | No |
| Lee et al. 2022 [9] | CNN | 256 | Offline, within-subject | ~70% | N/A | No |
| Ding et al. 2025 [3] | EEGNet | 128 | Online, session-adaptive | 80.56% | 60.61% | **Yes** |
| **This work** | **CBraMod** | **128** | Offline, cross-subject | **90.27%** | **75.42%** | No |
| **This work** | **CBraMod** | **32** (FDR) | Offline, cross-subject | **88.10%** | — | No |

> **Note on comparability**: Direct accuracy comparison across studies is limited by differences in evaluation paradigm (online vs. offline), training protocol (within-session vs. cross-subject), and participant cohorts. Ding et al. [3] report online session-adaptive performance with real-time robotic feedback; our results reflect offline cross-subject generalization evaluated on the same dataset without online adaptation. The comparison highlights the methodological landscape rather than claiming strict superiority.

Two limitations of prior work motivate this study: (1) reliance on high-density electrode setups (64–256 channels) that are impractical for daily use, and (2) use of task-specific models trained from scratch on limited per-subject data, which cannot leverage knowledge from large-scale EEG corpora. We address both by combining a pretrained foundation model with systematic channel reduction.

### 1.3 EEG Foundation Models

The recent emergence of large-scale pretrained models for EEG—analogous to foundation models in natural language processing and computer vision—represents a paradigm shift in neural signal decoding. Rather than training task-specific architectures from scratch on limited per-subject data, these models leverage massive unlabeled EEG corpora to learn general-purpose spatiotemporal representations that can be fine-tuned for downstream tasks.

CBraMod (Criss-Cross Brain Foundation Model) [4], accepted at ICLR 2025, is a Transformer-based model pretrained on the Temple University EEG (TUEG) corpus. Its key architectural innovation, Asymmetric Conditional Positional Encoding (ACPE), enables the model to accept arbitrary numbers of input channels without retraining—a crucial property for channel reduction experiments. With approximately 4 million parameters, CBraMod represents a 1,600× parameter increase over EEGNet-8,2 [5], a compact convolutional neural network (~2,500 parameters) that has become a standard baseline in BCI research.

Other concurrent efforts include LaBraM [6] and broader surveys of EEG foundation models [7], confirming that pretrained approaches consistently outperform task-specific models, particularly in low-data and cross-subject regimes.

### 1.4 Contributions

This paper makes the following contributions:

> 1. **Systematic foundation model evaluation for finger-level MI decoding.** We provide the first comprehensive comparison of an EEG foundation model (CBraMod) against a traditional CNN (EEGNet-8,2) across within-subject, cross-subject, and transfer learning paradigms for individual finger motor imagery, using 21 participants.
>
> 2. **Comprehensive channel reduction analysis.** We evaluate six 32-channel configurations (four data-driven, two hand-crafted), along with 61, 8, and 4-channel setups, establishing that 32 channels selected by Fisher Discriminant Ratio retain **97.6%** of full 128-channel performance with CBraMod.
>
> 3. **Transfer learning–channel count interaction.** We demonstrate a novel inverse relationship: transfer learning benefit is negligible at 128 channels but increases substantially as channel count decreases (**+4.59 pp** at 8 channels), suggesting that individual adaptation compensates for reduced spatial information.
>
> 4. **Control experiments addressing volume conduction.** Through complementary-channel and negative-control experiments, we provide evidence that high decoding accuracy with diverse channel subsets reflects EEG volume conduction redundancy rather than data leakage artifacts.

---

## 2. Materials and Methods

### 2.1 Dataset

We use a publicly available dataset [3] comprising 21 healthy, right-handed participants (S01–S21) who performed finger-level motor imagery and motor execution tasks. EEG was recorded using a 128-channel BioSemi ActiveTwo system at 1024 Hz. The experimental paradigm included:

- **Offline sessions**: 30 training runs with visual cues for individual finger imagery (thumb, index, middle, pinky)
- **Online sessions**: Real-time BCI control sessions across multiple days, each split into a calibration (Base) phase and an adaptation (Finetune) phase

For this study, we focus on the motor imagery paradigm with two classification granularities:

| Task | Classes | Chance Level |
|------|---------|-------------|
| **Binary** | Thumb (class 1) vs. Pinky (class 4) | 50% |
| **Ternary** | Thumb (class 1) vs. Middle (class 3) vs. Pinky (class 4) | 33.3% |

### 2.2 Preprocessing

Given the different input requirements of the two models, we implement two parallel preprocessing pipelines, summarized in Table 1.

**Table 1. Preprocessing pipeline comparison.**

| Step | EEGNet Pipeline | CBraMod Pipeline |
|------|----------------|-----------------|
| Re-referencing | Common Average Reference (trial-level) | Common Average Reference (trial-level) |
| Resampling | 1024 → 100 Hz (`resample_poly`) | 1024 → 200 Hz (`resample_poly`) |
| Bandpass filter | 4–40 Hz, 4th-order Butterworth, causal (`lfilter`) | 0.3–75 Hz, 4th-order Butterworth, causal (`lfilter`) |
| Segmentation | 1 s window, 125 ms step | 1 s window, 500 ms step |
| Normalization | Per-segment Z-score (time axis) | Divide by 100 |
| Artifact rejection | Trials exceeding ±500 µV rejected (training only) | Same |

Both pipelines apply Common Average Reference (CAR) at the trial level (not run level) using `nanmean` to handle NaN-padded variable-length trials (offline trials: 5 s; online trials: 3 s). Resampling uses `scipy.signal.resample_poly` with rational factor computation to avoid FFT-based aliasing artifacts. No data augmentation is applied.

### 2.3 Data Split Protocol

Following the original dataset paper [3], we employ a strict temporal split to prevent data leakage:

**Table 2. Data split protocol.**

| Partition | Source Sessions |
|-----------|----------------|
| **Training** | `OfflineImagery` + `OnlineImagery_Sess01_Base` + `OnlineImagery_Sess01_Finetune` + `OnlineImagery_Sess02_Base` |
| **Test** | `OnlineImagery_Sess02_Finetune` (completely held out) |

Within the training partition, we reserve the last 20% of trials (by temporal order) as a validation set. Crucially, the split operates at the **trial level**—all segments derived from a single trial are assigned to the same partition—preventing information leakage through overlapping sliding windows. The test partition (`Sess02_Finetune`) is never used during model selection or hyperparameter tuning.

### 2.4 Model Architectures

#### 2.4.1 EEGNet-8,2

We employ the EEGNet-8,2 configuration [5], a compact CNN designed for EEG decoding:

- **Block 1 (Temporal + Spatial)**: Temporal convolution (F₁ = 8 filters, kernel size 64 samples ≈ 0.64 s at 100 Hz) → BatchNorm → Depthwise spatial convolution (depth multiplier D = 2, constrained to max L₂ norm = 1.0) → BatchNorm → ELU → AveragePool(1, 4) → Dropout(0.5)
- **Block 2 (Separable)**: Depthwise separable convolution (F₂ = 16 = F₁ × D, kernel 16) → BatchNorm → ELU → AveragePool(1, 8) → Dropout(0.5)
- **Classifier**: Flatten → Linear(features, n_classes)

> Total trainable parameters: **~2,500**

#### 2.4.2 CBraMod

CBraMod [4] is a 12-layer Transformer pretrained on the TUEG corpus with the following configuration:

- **Backbone**: d_model = 200, 12 Transformer layers, 8 attention heads, feedforward dimension = 800
- **Patch embedding**: Each 1-second segment (200 samples at 200 Hz) constitutes one patch
- **ACPE**: Asymmetric Conditional Positional Encoding with convolutional kernel (19, 7), enabling arbitrary channel count input despite being pretrained on 19-channel data
- **Classifier**: Two-layer MLP — Linear(n_channels × n_patches × 200, 200) → ELU → Dropout → Linear(200, n_classes)

> Total trainable parameters: **~4.0M** (backbone) + classifier head — a **1,600×** increase over EEGNet.

### 2.5 Training Procedures

We evaluate three training paradigms with model-specific hyperparameters (Table 3).

**Table 3. Training hyperparameters.**

| Parameter | EEGNet Within | EEGNet Cross | CBraMod Within | CBraMod Cross |
|-----------|--------------|-------------|----------------|--------------|
| Optimizer | Adam | Adam | AdamW (dual group) | AdamW (dual group) |
| Learning rate | 1e-3 | 5e-4 | backbone: 1e-4, classifier: 3e-4 | backbone: 1e-4, classifier: 1.5e-4 |
| Weight decay | 0 | 1e-4 | 0.06 | 0.12 |
| Batch size | 64 | 128 | 128 (exploration: 32) | 256 (exploration: 64) |
| Max epochs | 30 | 50 | 50 | 100 |
| Early stopping patience | 5 | 10 | 10 | 15 |
| Scheduler | ReduceLROnPlateau (factor=0.3) | ReduceLROnPlateau | CAWD (phase=6, decay=0.7) | CAWD (phase=6, decay=0.5) |
| Label smoothing | 0 | 0 | 0.05 | 0.15 |
| Dropout | 0.5 | 0.5 | 0.15 | 0.35 |
| Gradient clipping | — | — | 1.0 | 0.5 |
| Mixed precision | FP16 (AMP) | FP16 (AMP) | FP16 (AMP) | FP16 (AMP) |

**Within-subject training**: Each participant's data is used independently to train and evaluate a personalized model.

**Cross-subject training**: All 21 participants' training data is pooled to train a single unified model. The validation set comprises the last 20% of trials per participant. Test evaluation is conducted per participant on their respective held-out `Sess02_Finetune` data.

**Transfer learning (fine-tuning)**: The best cross-subject pretrained model is fine-tuned for each individual participant using their training data (CBraMod: 15 epochs, lr=1e-4, AdamW).

CBraMod employs a **two-phase batch size strategy**: a smaller batch size during an initial exploration phase (first 6 epochs) followed by the standard batch size, promoting better loss landscape exploration. The **CosineAnnealingWarmupDecay (CAWD)** scheduler divides training into phases of 6 epochs, each with a warmup ramp (10% of phase) followed by cosine decay, with peak learning rate decaying by a factor between phases.

### 2.6 Channel Selection Methods

To investigate the channel count–performance trade-off, we evaluate six 32-channel configurations:

**Data-driven methods** (computed from all 21 participants' training data, 15,663 trials):

1. **Fisher Discriminant Ratio (FDR)**: For each channel, the ratio of between-class variance to within-class variance is computed across the mu (8–13 Hz) and beta (13–30 Hz) bands. Channels are ranked by aggregate FDR score.

2. **Common Spatial Patterns (CSP)**: CSP filters are computed using MNE-Python with Ledoit-Wolf covariance regularization. Channels are ranked by their contribution to the top spatial patterns.

3. **Attention / Gradient**: Combines EEGNet spatial convolution filter magnitudes with CBraMod input gradient magnitudes, aggregated across participants.

4. **Band Power (ANOVA)**: For each channel, spectral power in mu and beta bands is computed, and channels are ranked by ANOVA F-statistic between classes.

**Hand-crafted configurations**:

5. **Motor Cortex**: 32 electrodes densely covering the sensorimotor strip (C3/Cz/C4, supplementary motor area, premotor cortex), selected by nearest-neighbor mapping from standard 10-20 locations to BioSemi 128 positions.

6. **Commercial**: 32 electrodes approximating a standard 10-20 commercial EEG cap layout.

Additionally, we evaluate: **61 channels** (standard 10-10 system), **8 channels** (FDR top-8), and **4 channels** (intersection of FDR top-32 and Attention top-32, yielding 4 common electrodes at BioSemi positions B32, C8, D7, D19).

### 2.7 Evaluation Protocol

**Primary metric**: Trial-level majority voting accuracy. Each trial produces multiple 1-second segments; the model predicts each segment independently, and the final trial prediction is determined by majority vote across segments. Accuracy is computed as the fraction of correctly classified trials.

**Early stopping criterion**: The combined score (mean of segment-level validation accuracy and trial-level majority voting accuracy) is monitored; training halts when no improvement is observed for the patience period.

**Statistical tests**: Paired t-tests and Wilcoxon signed-rank tests (n = 21 participants) are used for pairwise model comparisons.

### 2.8 Data Quality Assessment

We performed a comprehensive 12-metric data quality assessment across all 21 participants, evaluating signal amplitude, SNR, temporal drift, inter-trial variability, and class separability. Participants were categorized as: Clean (10/21), Informational (3/21), Minor artifact (5/21), or Major artifact (3/21). Three participants—S04, S10, and S14—exhibited severe hardware artifacts with maximum amplitudes of 306,796 µV, 267,904 µV, and 125,503 µV, respectively (population mean: 37,839 µV). We retain all 21 participants in all analyses to provide conservative performance estimates; the impact of excluding artifact-heavy participants is discussed in Section 6 (Future Work).

---

## 3. Results

> **Results at a glance** — CBraMod achieves **90.27%** cross-subject binary accuracy at 128 channels. At 32 channels (FDR), it retains **88.10%** (−2.17 pp) while EEGNet drops to 67.53%. Transfer learning gain scales inversely with channel count: negligible at 128ch, **+4.59 pp** at 8ch.

### 3.1 Within-Subject Comparison (128 Channels)

Table 4 summarizes within-subject performance across all 21 participants at 128 channels.

**Table 4. Within-subject binary classification (128 channels, majority voting accuracy).**

| Metric | EEGNet-8,2 | CBraMod |
|--------|-----------|---------|
| Mean ± SD | 78.75 ± 11.56% | 84.64 ± 10.61% |
| Median | 77.50% | 86.88% |
| Min | 55.00% (S20) | 59.38% (S10) |
| Max | 96.88% (S09) | 98.75% (S19) |

CBraMod outperforms EEGNet by **+5.89 percentage points (pp)** on average (paired *t*-test: *t*(20) = 3.36, *p* = 0.003; Wilcoxon: *W* = 24, *p* = 0.004; Cohen's *d* = 0.73). The win/tie/loss breakdown across 21 participants:

| Outcome | Count | Participants |
|---------|-------|-------------|
| CBraMod wins | **16** | S01, S03, S05, S07, S08, S09, S11, S12, S13, S14, S15, S16, S17, S19, S20, S21 |
| Tie | 2 | S02 (95.00%), S18 (91.25%) |
| EEGNet wins | 3 | S04 (90.63% vs. 95.63%), S06 (69.38% vs. 71.25%), S10 (59.38% vs. 70.63%) |

Notably, 2 of the 3 EEGNet-favorable participants (S04, S10) are major-artifact cases.

Per-subject results for both models are provided in Supplementary Table S1.

> **Data sources**: EEGNet within-subject run `20260206_1003`: `results/20260206_1003_comparison_cache_imagery_binary.json`; CBraMod within-subject run `20260210_0435`: `results/20260210_0435_comparison_cache_imagery_binary BLANK-CBRAMOD.json`

**Figure 2.** Within-subject binary classification — per-participant accuracy comparison (128 channels, majority voting). *Purpose: establish baseline model performance before cross-subject pooling.*

**(a)** EEGNet-8,2 within-subject (run `20260206_1003` · 128ch · binary · within-subject · EEGNet · mean 78.75%)
![Fig 2a — EEGNet within-subject 128ch binary](results/20260206_1003_combined_imagery_binary.png)

**(b)** CBraMod within-subject (run `20260210_0435` · 128ch · binary · within-subject · CBraMod · mean 84.64%)
![Fig 2b — CBraMod within-subject 128ch binary](results/20260210_0435_combined_imagery_binary.png)

### 3.2 Cross-Subject Training (128 Channels)

Pooling data across all 21 participants yields substantial gains for CBraMod:

**Table 5. Cross-subject training results (128 channels).**

| Task | CBraMod Mean ± SD |
|------|-------------------|
| Binary (2-class) | 90.27 ± 8.88% |
| Ternary (3-class) | 75.42 ± 12.72% |

Cross-subject binary accuracy of **90.27%** represents a **+5.63 pp** improvement over within-subject training (84.64%; paired *t*-test: *t*(20) = 5.42, *p* < 0.001; Cohen's *d* = 1.18). The gap is especially pronounced for participants who had low within-subject performance:

| Participant | Within-Subject | Cross-Subject | Δ |
|-------------|---------------|---------------|---|
| S15 | 86.88% | 94.38% | +7.50 pp |
| S16 | 81.25% | 92.50% | +11.25 pp |

Ternary cross-subject accuracy of **75.42%** significantly exceeds both chance level (33.3%) and within-subject ternary performance (69.54%, **+5.88 pp**), demonstrating that cross-subject data pooling benefits harder classification tasks as well.

The worst-performing participants across both tasks are S10 (binary: 66.25%) and S20 (binary: 66.88%), consistent with their data quality profiles (S10: major artifact; S20: informational—elevated variance but functional signals).

> **Data sources**: Binary run `20260206_1029`: `results/20260206_1029_cross-subject_cbramod_imagery_binary.json`; Ternary run `20260207_2056`: `results/20260207_2056_cross-subject_cbramod_imagery_ternary.json`

**Figure 3.** Cross-subject training — per-participant accuracy (CBraMod, 128 channels). *Purpose: demonstrate benefit of pooling all 21 participants' data into a single model.*

**(a)** Cross-subject binary (run `20260206_1029` · 128ch · binary · cross-subject · CBraMod · mean 90.27%)
![Fig 3a — CBraMod cross-subject 128ch binary](results/20260206_1029_cross-subject_combined_imagery_binary.png)

**(b)** Cross-subject ternary (run `20260207_2056` · 128ch · ternary · cross-subject · CBraMod · mean 75.42%)
![Fig 3b — CBraMod cross-subject 128ch ternary](results/20260207_2056_cross-subject_combined_imagery_ternary.png)

### 3.3 32-Channel Configuration Comparison

Table 6 presents cross-subject binary accuracy for all six 32-channel configurations.

**Table 6. 32-channel configuration comparison (cross-subject binary, n = 21).**

| Rank | Configuration | Type | CBraMod Mean ± SD | EEGNet Mean ± SD | Δ (CBraMod − EEGNet) |
|------|--------------|------|-------------------|------------------|-----------------------|
| 1 | FDR | Data-driven | **88.10 ± 8.80%** | 67.53 ± 11.12% | +20.57 pp |
| 2 | Attention | Data-driven | 87.02 ± 9.89% | **70.42 ± 12.75%** | +16.60 pp |
| 3 | Commercial | Hand-crafted | 86.31 ± 7.91% | 64.40 ± 9.82% | +21.91 pp |
| 4 | CSP | Data-driven | 85.54 ± 10.34% | 66.52 ± 12.91% | +19.02 pp |
| 5 | Band Power | Data-driven | 85.51 ± 10.11% | 67.17 ± 13.21% | +18.34 pp |
| 6 | Motor Cortex | Hand-crafted | 82.02 ± 9.70% | 63.13 ± 10.48% | +18.89 pp |

The six configurations differ significantly (Friedman test: *χ²*(5) = 27.90, *p* < 0.001). **Key observations:**

> **Finding 1 — Data-driven > hand-crafted.** The top two data-driven methods (FDR, Attention) achieve **87–88%** versus 82–86% for hand-crafted layouts. FDR significantly outperforms Motor Cortex (paired *t*-test: *t*(20) = 4.79, *p* < 0.001, *d* = 1.05).

> **Finding 2 — Model gap widens at reduced channels.** The CBraMod–EEGNet gap expands from +5.89 pp (128ch within-subject) to **+16.60–21.91 pp** at 32 channels, suggesting pretrained representations are especially valuable when spatial information is limited.

> **Finding 3 — Stability vs. accuracy.** Commercial layout shows the lowest standard deviation (**7.91%**) for CBraMod, indicating more stable cross-subject performance, but its mean trails FDR by 1.79 pp.

> **Finding 4 — Non-motor regions matter.** FDR-selected channels predominantly cover *temporal and frontal* regions rather than the traditional motor cortex (C3/Cz/C4), yet achieve the highest CBraMod accuracy. This is discussed further in Section 4.

> **Data sources** (run `20260220_*`, all cross-subject binary):
> - FDR run `20260220_1949`: `results/32_channel/fdr/20260220_1949_cross-subject_cbramod_imagery_binary.json`
> - Attention run `20260220_2159`: `results/32_channel/attention/20260220_2159_cross-subject_cbramod_imagery_binary.json`
> - Commercial run `20260220_1850`: `results/32_channel/commercial/20260220_1850_cross-subject_cbramod_imagery_binary.json`
> - CSP run `20260220_2052`: `results/32_channel/csp/20260220_2052_cross-subject_cbramod_imagery_binary.json`
> - Band Power run `20260220_2301`: `results/32_channel/band_power/20260220_2301_cross-subject_cbramod_imagery_binary.json`
> - Motor Cortex run `20260220_1731`: `results/32_channel/motor_cortex/20260220_1731_cross-subject_cbramod_imagery_binary.json`
> - (EEGNet results from corresponding `*_eegnet_imagery_binary.json` files in same directories)

**Figure 4.** 32-channel 6-configuration comparison — grouped bar chart of CBraMod vs. EEGNet cross-subject binary accuracy for all six 32ch channel selection methods. *Purpose: core contribution — compare data-driven vs. hand-crafted channel layouts and quantify model gap at reduced channels.*

(run `20260222_1324` · 32ch · 6 configs × 2 models · cross-subject binary)
![Fig 4 — 32ch 6-config comparison](results/32_channel/20260222_1324_32ch_config_comparison_imagery_binary.png)

**Figure 5.** Electrode placement maps — 2D scalp projections showing spatial distribution of electrodes for all six 32-channel configurations. *Purpose: visualize that data-driven methods (FDR, Attention) select non-motor-cortex regions.*

![Fig 5 — Electrode placement grid, all 6 configs](results/32_channel/electrode_placements/grid_all_configs_2d.png)

**Figure 6.** Pairwise channel overlap — heatmap showing number of shared electrodes between each pair of 32ch configurations. *Purpose: quantify similarity/independence of channel selection methods.*

![Fig 6 — Channel overlap heatmap](results/32_channel/electrode_placements/overlap_analysis.png)

### 3.4 Channel Scaling Analysis (128 → 61 → 32 → 8 → 4)

Table 7 and Figure 7 illustrate the relationship between channel count and cross-subject binary accuracy.

**Table 7. Channel scaling summary (CBraMod cross-subject binary).**

| Channels | Configuration | Mean ± SD | Δ vs. 128ch | Run ID | Result File |
|----------|--------------|-----------|-------------|--------|-------------|
| 128 | Full array | 90.27 ± 8.88% | — | `20260206_1029` | `results/20260206_1029_cross-subject_cbramod_imagery_binary.json` |
| 61 | Standard 10-10 | 88.72 ± 9.22% | −1.55 pp | `20260227_0049` | `results/61_channel/standard_1010/20260227_0049_cross-subject_cbramod_imagery_binary.json` |
| 32 | FDR (best) | 88.10 ± 8.80% | −2.17 pp | `20260220_1949` | `results/32_channel/fdr/20260220_1949_cross-subject_cbramod_imagery_binary.json` |
| 8 | FDR | 68.33 ± 9.80% | −21.94 pp | `20260221_1218` | `results/8_channel/20260221_1218_cross-subject_cbramod_imagery_binary.json` |
| 4 | FDR ∩ Attention | 82.86 ± 14.55% | −7.41 pp | `20260301_2100` | `results/4_channel/fdr_attention_overlap/20260301_2100_cross-subject_cbramod_imagery_binary.json` |

The performance degradation is markedly **nonlinear**:

| Transition | Electrode Reduction | Accuracy Drop | Interpretation |
|-----------|-------------------|--------------|----------------|
| 128 → 61 | −52% | **−1.55 pp** | High information redundancy |
| 61 → 32 | −48% | **−0.62 pp** | FDR 32ch ≈ standard 10-10 61ch |
| 32 → 8 | −75% | **−19.77 pp** | Below critical spatial sampling threshold |
| 32 → 4 | −88% | −7.41 pp vs. 128ch | See caveat below |

**Important caveat on 4ch vs. 8ch non-monotonicity**: The 4-channel result (82.86%) exceeds the 8-channel result (68.33%), which appears to violate monotonicity. However, these two configurations use **different channel selection methods**: 8ch uses FDR top-8, while 4ch uses the intersection of FDR and Attention top-32 — a fundamentally different and more selective criterion. The two results are therefore not directly comparable on a "channel scaling" axis. The 4ch result demonstrates that a small number of highly informative channels (identified by consensus across methods) can outperform a larger but less optimally selected set, rather than implying that fewer channels are generally better. This result was validated across three independent runs (82.86%, 83.24%, 84.20%; SD between runs < 1 pp), though with considerably higher variance (SD = 14.55%), reflecting extreme sensitivity to individual participants.

> **Key takeaway:** The **32-channel mark** represents the optimal trade-off — retaining **97.6%** of full-array performance with only **25%** of the electrodes. The 128ch → 32ch FDR drop is statistically significant but small (paired *t*-test: *t*(20) = 2.72, *p* = 0.013, *d* = 0.59).

**Figure 7.** Channel scaling — per-participant cross-subject binary accuracy at each channel count (CBraMod). *Purpose: visualize the nonlinear performance degradation curve from 128ch down to 4ch.*

**(a)** 128ch full array (run `20260206_1029` · 128ch · full array · cross-subject · CBraMod · mean 90.27%)
![Fig 7a — 128ch cross-subject](results/20260206_1029_cross-subject_combined_imagery_binary.png)

**(b)** 61ch standard 10-10 (run `20260227_0049` · 61ch · standard 10-10 layout · cross-subject · CBraMod · mean 88.72%)
![Fig 7b — 61ch standard 10-10 cross-subject](results/61_channel/standard_1010/20260227_0049_cross-subject_combined_imagery_binary.png)

**(c)** 32ch FDR (run `20260220_1949` · 32ch · FDR data-driven selection · cross-subject · CBraMod · mean 88.10%)
![Fig 7c — 32ch FDR cross-subject](results/32_channel/fdr/20260220_1949_cross-subject_combined_imagery_binary.png)

**(d)** 8ch FDR (run `20260221_1218` · 8ch · FDR top-8 · cross-subject · CBraMod · mean 68.33%)
![Fig 7d — 8ch FDR cross-subject](results/8_channel/20260221_1218_cross-subject_combined_imagery_binary.png)

**(e)** 4ch FDR∩Attention (run `20260301_2100` · 4ch · intersection of FDR and Attention top-32 · cross-subject · CBraMod · mean 82.86%)
![Fig 7e — 4ch FDR∩Attention cross-subject](results/4_channel/fdr_attention_overlap/20260301_2100_cross-subject_combined_imagery_binary.png)

### 3.5 Transfer Learning Across Channel Configurations

Table 8 reveals a striking interaction between transfer learning benefit and channel count.

**Table 8. Transfer learning effect across channel counts (CBraMod binary).**

| Channels | Configuration | Cross-Subject | After Transfer | Δ Transfer |
|----------|--------------|---------------|----------------|------------|
| 128 | Full array | 90.27% | 90.18% | **−0.09 pp** |
| 32 | FDR | 88.10% | 88.90% | **+0.80 pp** |
| 32 | Attention | 87.02% | 88.69% | **+1.67 pp** |
| 8 | FDR | 68.33% | 72.92% | **+4.59 pp** |

At 128 channels, the cross-subject model has already captured sufficient representational capacity, and individual fine-tuning provides no additional benefit (−0.09 pp; *t*(20) = −0.45, *p* = 0.66, n.s.). However, as channel count decreases, the **transfer learning benefit increases monotonically**: +0.80 pp at 32 channels (FDR), +1.67 pp at 32 channels (Attention), and **+4.59 pp** at 8 channels (*t*(20) = 3.11, *p* = 0.006, *d* = 0.68).

This pattern extends to ternary classification: 32-channel FDR transfer gains +1.89 pp (cross: 70.79% → transfer: 72.68%), and 8-channel FDR gains +5.26 pp (cross: 52.00% → transfer: 57.26%).

**Interpretation**: With fewer channels, the cross-subject model suffers from reduced spatial information diversity, making individual adaptation through transfer learning increasingly valuable for recovering participant-specific neural patterns.

> **Data sources**:
> - 128ch transfer run `20260209_1704`: `results/20260209_1704_transfer_comparison_cache_imagery_binary.json`
> - 32ch FDR transfer run `20260221_0445`: `results/32_channel/fdr/20260221_0445_transfer_comparison_cache_imagery_binary.json`
> - 32ch Attention transfer run `20260228_2218`: `results/32_channel/attention/20260228_2218_transfer_comparison_cache_imagery_binary.json`
> - 8ch FDR transfer run `20260221_1319`: `results/8_channel/20260221_1319_transfer_comparison_cache_imagery_binary.json`

**Figure 8.** Transfer learning — cross-subject vs. individually fine-tuned accuracy at each channel count (CBraMod binary). *Purpose: demonstrate that transfer learning benefit increases as channel count decreases.*

**(a)** 128ch transfer (run `20260209_1704` · 128ch · full array · cross→individual fine-tune · CBraMod · Δ = −0.09 pp)
![Fig 8a — 128ch transfer comparison](results/20260209_1704_transfer_combined_imagery_binary.png)

**(b)** 32ch FDR transfer (run `20260221_0445` · 32ch · FDR selection · cross→individual fine-tune · CBraMod · Δ = +0.80 pp)
![Fig 8b — 32ch FDR transfer comparison](results/32_channel/fdr/20260221_0445_transfer_combined_imagery_binary.png)

**(c)** 32ch Attention transfer (run `20260228_2218` · 32ch · Attention selection · cross→individual fine-tune · CBraMod · Δ = +1.67 pp)
![Fig 8c — 32ch Attention transfer comparison](results/32_channel/attention/20260228_2218_transfer_combined_imagery_binary.png)

**(d)** 8ch FDR transfer (run `20260221_1319` · 8ch · FDR top-8 · cross→individual fine-tune · CBraMod · Δ = +4.59 pp)
![Fig 8d — 8ch FDR transfer comparison](results/8_channel/20260221_1319_transfer_combined_imagery_binary.png)

### 3.6 Control Experiments

To rule out data leakage as an explanation for the consistently high accuracy observed across different channel configurations, we conducted two control experiments.

**Table 9. Control experiment results (32 channels, CBraMod cross-subject binary).**

| Condition | Channels | Mean ± SD | Δ vs. FDR |
|-----------|----------|-----------|-----------|
| FDR (optimal 32ch) | Top-ranked FDR | 88.10 ± 8.80% | — |
| FDR Complement | 32 random from 96 non-FDR channels | 83.18 ± 9.80% | −4.92 pp |
| Negative Control | 32 channels not selected by any method | 84.08 ± 9.36% | −4.02 pp |

**FDR Complement test**: From the 96 channels excluded by the FDR top-32 selection, we randomly sampled 32 channels. Despite using channels explicitly identified as having low discriminative power, the model achieved 83.18% accuracy (replicated across two independent runs: 83.30% and 83.18%).

**Negative Control**: Using 32 channels not selected by any of the four data-driven methods, the model achieved 84.08%.

Both control conditions achieve accuracy far above chance (50%) and only 4–5 pp below the optimally selected configuration. The FDR advantage over its complement is statistically significant (paired *t*-test: *t*(20) = 4.04, *p* < 0.001, *d* = 0.88). This result has two important implications:

> **Implication 1 — No data leakage.** If high accuracy were due to label information leaking into the test set, it should appear *equally* across all channel subsets rather than showing a consistent 4–5 pp gap favoring optimally selected channels.

> **Implication 2 — Volume conduction redundancy.** Even channels with low analytic discriminative power carry sufficient information for a pretrained foundation model to reconstruct a viable classification space, confirming extensive signal redundancy in 128-channel EEG.

> **Data sources**:
> - FDR complement run 1 `20260301_1155`: `results/32_channel/fdr_complement/20260301_1155_cross-subject_cbramod_imagery_binary.json`
> - FDR complement run 2 `20260301_1448`: `results/32_channel/fdr_complement/20260301_1448_cross-subject_cbramod_imagery_binary.json`
> - Negative control run `20260302_0141`: `results/32_channel/negative_control/20260302_0141_cross-subject_cbramod_imagery_binary.json`

**Figure 9.** Control experiments — per-participant accuracy for non-optimal channel subsets (32ch, CBraMod cross-subject binary). *Purpose: rule out data leakage by showing that even "worst" channels achieve >83%, confirming volume conduction redundancy.*

**(a)** FDR complement (run `20260301_1448` · 32ch · 32 random channels from the 96 NOT selected by FDR · cross-subject · CBraMod · mean 83.18%)
![Fig 9a — FDR complement cross-subject](results/32_channel/fdr_complement/20260301_1448_cross-subject_combined_imagery_binary.png)

**(b)** Negative control (run `20260302_0141` · 32ch · 32 channels not selected by ANY data-driven method · cross-subject · CBraMod · mean 84.08%)
![Fig 9b — Negative control cross-subject](results/32_channel/negative_control/20260302_0141_cross-subject_combined_imagery_binary.png)

### 3.7 Data Quality and Subject Heterogeneity

**Table 10. Data quality classification (n = 21).**

| Category | Count | Participants | Criterion |
|----------|-------|-------------|-----------|
| Clean | 10 | S01, S02, S06, S07, S08, S11, S13, S15, S17, S18 | All metrics within normal range |
| Informational | 3 | S12, S19, S20 | Elevated variance (20–65×) but functional signals |
| Minor artifact | 5 | S03, S05, S09, S16, S21 | 5.7–9.4% of trials with artifacts |
| Major artifact | 3 | S04, S10, S14 | Extreme amplitude (126K–307K µV, normal ≤ 38K µV), Fisher Ratio ≈ 0 |

The three major-artifact participants (S04, S10, S14) exhibit amplitudes 3–8× the population maximum, with temporal drift values orders of magnitude above the group mean (S04: 2,717 vs. group mean ~30). Despite this, S04 paradoxically achieves 98.12% cross-subject accuracy (128ch binary), suggesting the model may exploit artifact patterns rather than genuine neural signals for this participant.

> **Data sources**: `results/data_quality_report.md`; `results/data_quality_advanced_report.md`; `results/subject_deep_dive_report.md`

**Figure 10.** Spatial analysis of channel selection patterns. *Purpose: show that FDR favors temporal/frontal regions over motor cortex, and visualize overlap between top-performing methods.*

**(a)** Scalp region distribution — bar chart showing how many channels each 32ch config allocates to each scalp region (frontal, central, temporal, parietal, occipital).
![Fig 10a — Region distribution across configs](results/32_channel/electrode_placements/region_distribution.png)

**(b)** FDR vs. Attention overlap — 2D scalp map showing shared and unique electrode positions between the two best-performing data-driven methods (4 overlapping channels = the 4ch subset).
![Fig 10b — FDR vs Attention channel overlap](results/32_channel/electrode_placements/overlap_fdr_vs_attention_2d.png)

---

## 4. Discussion

### 4.1 Foundation Model Advantage

The consistent superiority of CBraMod over EEGNet across all experimental conditions — within-subject (**+5.89 pp**), cross-subject (**~20 pp** gap at 32 channels), and transfer learning — reflects the value of large-scale pretraining for EEG decoding. The 1,600× parameter difference between the models (4M vs. 2.5K) alone does not explain this gap; rather, CBraMod's pretraining on the TUEG corpus provides general-purpose spatiotemporal EEG representations that transfer effectively to the relatively data-scarce finger-level MI task.

> The performance gap between the two models **widens as channel count decreases** (5.89 pp at 128ch → 16.60–21.91 pp at 32ch), suggesting that pretrained knowledge partially compensates for reduced spatial information — a benefit unavailable to randomly initialized models like EEGNet.

### 4.2 Optimal Channel Configuration for Deployment

The 32-channel FDR configuration emerges as the optimal trade-off for practical BCI deployment:

| Property | Value |
|----------|-------|
| Performance retention | **97.6%** of 128ch (88.10% vs. 90.27%) |
| vs. 61ch standard 10-10 | Equivalent (88.72%) with **nearly half** the channels |
| Hardware compatibility | Standard commercial 32-channel EEG systems |

Notably, FDR-selected channels concentrate in *temporal and frontal regions* rather than the traditional motor cortex. This counterintuitive finding may reflect the fact that the FDR method prioritizes overall information density across the scalp rather than anatomical priors. The foundation model's ability to extract discriminative features from non-traditional electrode locations underscores the distinction between channels that are *individually* discriminative (motor cortex) and channels that *collectively* provide the richest information to a pretrained model.

### 4.3 Volume Conduction and Information Redundancy

The control experiments (Section 3.6) reveal a fundamental property of high-density EEG: due to volume conduction, electrical signals from cortical sources propagate broadly across the scalp, creating substantial information redundancy. With 128 channels, any 32-channel subset (25% of the full array) captures sufficient overlapping information for a powerful pretrained model to reconstruct viable classification features.

This interpretation is further supported by the 4-channel result (82.86%), which demonstrates that even an extreme reduction to 3.1% of the full array achieves above-80% accuracy, albeit with higher variance. The selected 4 channels (BioSemi B32, C8, D7, D19) represent the intersection of the two best-performing data-driven methods, suggesting they capture a core, non-redundant information substrate.

### 4.4 Transfer Learning Saturation

The negligible transfer learning benefit at 128 channels (−0.09 pp) indicates that the cross-subject model has reached a representational ceiling for this task and dataset at full channel density. This saturation is consistent with the original dataset paper's observation of performance plateaus with additional session data [3].

However, the monotonically increasing transfer benefit at lower channel counts (up to +4.59 pp at 8 channels) suggests that the ceiling is channel count–dependent: with fewer spatial samples, the cross-subject model cannot fully generalize, and individual fine-tuning recovers participant-specific patterns lost to spatial undersampling.

### 4.5 Impact of Artifact-Affected Participants

Retaining three major-artifact participants (S04, S10, S14) in all analyses provides conservative performance estimates. These participants exhibit SNR values 4–6 dB below the population mean (−19.8 to −21.8 dB vs. −15.8 dB mean) and Fisher Discriminant Ratios near zero, indicating that genuine class-discriminative neural information is buried beneath artifact noise. The impact of excluding these participants is explored in Section 6.1.

---

## 5. Limitations

Several limitations should be considered when interpreting these results:

| # | Limitation | Impact |
|---|-----------|--------|
| 1 | **Channel selection data scope** — FDR, CSP, Attention, Band Power metrics computed using all sessions (including test session context). Channel indices (not labels) are extracted, but this is an indirect information leak. | May inflate channel selection quality; future work should use strictly held-out data. |
| 2 | **Single dataset** — All experiments use one 21-participant dataset. | Generalization to other populations, paradigms, or hardware unverified. |
| 3 | **Motor imagery only** — Motor execution data not yet evaluated. | Signal characteristics and optimal channels may differ (see Section 6.3). |
| 4 | **No data augmentation** — No temporal shifting, noise injection, or channel dropout applied. | Low-channel regimes may benefit most from augmentation. |
| 5 | **Hyperparameter transferability** — Hyperparameters optimized for 128ch; reduced-channel experiments (especially 4ch) used same defaults. | Suboptimal performance at low channel counts. |
| 6 | **Missing EEGNet cross-subject 128ch baseline** — EEGNet cross-subject results available only for 32ch and 61ch. | Direct 128ch cross-subject model comparison incomplete. |

---

## 6. Ongoing Experiments

The following experiments are planned as part of this study but have not yet been completed. Their results will be incorporated into the final version of this manuscript.

### 6.1 Artifact Subject Exclusion and Re-Evaluation

Three participants (S04, S10, S14) exhibit severe hardware artifacts with maximum amplitudes exceeding 125,000 µV—3–8× the population maximum of 38,000 µV (Section 3.7). Their inclusion in the cross-subject training pool likely suppresses the model's achievable baseline and inflates inter-subject variance.

To quantify this effect, we will remove S04, S10, and S14 from all training and evaluation pipelines and re-run the complete experimental suite (within-subject, cross-subject, transfer learning, channel reduction). We expect that excluding these participants will:
- Raise the cross-subject binary baseline above 92%
- Reduce inter-subject standard deviation by approximately 2–3 pp
- Sharpen the performance gap between data-driven and hand-crafted channel selections

> `[TODO: Insert comparative results table — 21 subjects vs. 18 subjects (excluding S04/S10/S14) across all experimental conditions]`

### 6.2 Plateau-Breaking with Multi-Session Longitudinal Data

The original dataset paper [3] reported that participants' online BCI performance plateaus after 2–3 sessions, with additional session data yielding diminishing returns. Several participants in this dataset have 5 or more longitudinal sessions, providing an opportunity to test whether a foundation model can break through this classical plateau.

We will incrementally add session data to the training pool for participants with ≥5 sessions and track performance evolution, testing whether CBraMod's 4M-parameter capacity can absorb longitudinal variability that overwhelms traditional models.

> `[TODO: Insert learning curves showing accuracy vs. cumulative session count, per participant and aggregated]`

### 6.3 Motor Execution Paradigm Validation

All current results are limited to motor imagery. The same dataset includes motor execution recordings, which involve actual finger movement and typically produce stronger, more localized EEG signatures.

We will replicate the complete experimental pipeline using the movement paradigm (Motor Execution) to address the following questions:
- Does CBraMod's advantage over EEGNet persist for motor execution?
- Does the optimal channel configuration differ between imagery and execution?
- Is the transfer learning–channel count interaction preserved?

> `[TODO: Insert MI vs. ME comparison table across all channel configurations]`

### 6.4 Advanced Visualization and Analysis

To provide deeper neuroscientific insight, we will produce the following additional analyses:

- **Cross-subject feature clustering (UMAP/t-SNE)**: Visualization of learned representations across participants, highlighting whether artifact-affected participants (e.g., S04) form distinct clusters in the embedding space.
- **Electrode topographic maps (topoplots)**: Spatial visualization of volume conduction redundancy, showing that discriminative power is distributed rather than localized to the motor cortex.
- **Session-level learning dynamics**: Epoch-by-epoch training curves for the plateau-breaking experiment (Section 6.2).

> `[TODO: Insert UMAP/t-SNE cluster figures, topographic maps, and learning curve visualizations]`

---

## 7. Conclusion

This study demonstrates that combining an EEG foundation model (CBraMod) with data-driven channel selection provides a viable path toward practical, reduced-channel brain-computer interfaces for finger-level motor imagery decoding. Three key findings emerge:

> **Finding 1 — Foundation model superiority scales with channel reduction.**
> CBraMod's advantage over EEGNet grows from **+5.89 pp** at 128 channels to **+16–22 pp** at 32 channels, establishing pretrained models as particularly valuable when spatial information is constrained.

> **Finding 2 — 32 channels is the optimal deployment target.**
> FDR-selected 32 channels retain **97.6%** of full 128-channel performance (88.10% vs. 90.27%), offering a practical balance between decoding accuracy and hardware requirements.

> **Finding 3 — Transfer learning compensates for channel reduction.**
> Individual fine-tuning benefit increases from negligible at 128 channels to **+4.59 pp** at 8 channels, suggesting a complementary strategy where reduced-channel systems are paired with per-user adaptation.

These results, validated through rigorous control experiments that confirm volume conduction redundancy rather than data leakage, support the deployment of CBraMod-based BCI systems with commercial 32-channel hardware for finger-level motor imagery applications.

---

## References

[1] J. R. Wolpaw, N. Birbaumer, D. J. McFarland, G. Pfurtscheller, and T. M. Vaughan, "Brain-computer interfaces for communication and control," *Clinical Neurophysiology*, vol. 113, no. 6, pp. 767–791, 2002.

[2] G. Pfurtscheller and C. Neuper, "Motor imagery and direct brain-computer communication," *Proceedings of the IEEE*, vol. 89, no. 7, pp. 1123–1134, 2001.

[3] [Dataset paper] "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level." *(Full citation to be completed)*

[4] [CBraMod] W. Wang et al., "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding," in *Proc. International Conference on Learning Representations (ICLR)*, 2025.

[5] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, "EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces," *Journal of Neural Engineering*, vol. 15, no. 5, p. 056013, 2018.

[6] [LaBraM] *(Full citation to be completed from references/2405.18765v1)*

[7] [EEG Foundation Model Survey] *(Full citation to be completed from references/2504.20069v2)*

[8] R. Alazrai, H. Abuhijleh, M. Alwanni, and M. I. Daoud, "EEG-based BCI system for decoding finger movements within the same hand," *Neuroscience Letters*, vol. 698, pp. 113–120, 2019.

[9] H. S. Lee, S. Schreiner, S.-H. Jo, S. Sieghartsleitner, M. Jordan, R. Prettenthaler, D. Geyik, J. Millán, C. Brunner, and G. R. Müller-Putz, "Individual finger movement decoding using a novel ultra-high-density electroencephalography-based brain-computer interface system," *Frontiers in Neuroscience*, vol. 16, p. 1009878, 2022.

---

## Supplementary Material

### Table S1. Per-Subject Results (128 Channels, Binary Classification)

| Subject | EEGNet Within | CBraMod Within | CBraMod Cross | Data Quality |
|---------|--------------|----------------|---------------|--------------|
| S01 | 69.38% | 76.25% | 87.50% | Clean |
| S02 | 95.00% | 95.00% | 96.25% | Clean |
| S03 | 80.00% | 98.13% | 98.75% | Minor |
| S04 | 95.63% | 90.63% | 98.12% | **Major** |
| S05 | 85.00% | 88.13% | 93.12% | Minor |
| S06 | 71.25% | 69.38% | 86.25% | Clean |
| S07 | 76.25% | 81.88% | 90.00% | Clean |
| S08 | 84.38% | 85.63% | 97.50% | Clean |
| S09 | 96.88% | 97.50% | 98.75% | Minor |
| S10 | 70.63% | 59.38% | 66.25% | **Major** |
| S11 | 72.50% | 90.00% | 93.75% | Clean |
| S12 | 81.25% | 88.75% | 88.75% | Info |
| S13 | 87.50% | 91.88% | 91.25% | Clean |
| S14 | 77.50% | 84.38% | 88.12% | **Major** |
| S15 | 71.25% | 86.88% | 94.38% | Clean |
| S16 | 58.75% | 81.25% | 92.50% | Minor |
| S17 | 70.00% | 80.00% | 90.62% | Clean |
| S18 | 91.25% | 91.25% | 95.00% | Clean |
| S19 | 93.13% | 98.75% | 99.38% | Info |
| S20 | 55.00% | 61.88% | 66.88% | Info |
| S21 | 71.25% | 80.63% | 82.50% | Minor |

> **Data sources**: EEGNet within: `results/20260206_1003_comparison_cache_imagery_binary.json`; CBraMod within: `results/20260210_0435_comparison_cache_imagery_binary BLANK-CBRAMOD.json`; CBraMod cross: `results/20260206_1029_cross-subject_cbramod_imagery_binary.json`

### Table S2. 32-Channel Per-Subject Results (CBraMod Cross-Subject Binary)

| Subject | FDR | Attention | CSP | Band Power | Commercial | Motor Cortex |
|---------|-----|-----------|-----|-----------|------------|--------------|
| S01 | 92.50% | 91.25% | 84.38% | 85.63% | 82.50% | 72.50% |
| S02 | 95.63% | 85.63% | 87.50% | 91.88% | 90.00% | 86.88% |
| S03 | 98.13% | 98.13% | 98.13% | 99.38% | 96.25% | 93.13% |
| S04 | 97.50% | 96.88% | 95.00% | 90.63% | 93.13% | 93.75% |
| S05 | 81.25% | 61.88% | 68.75% | 65.00% | 79.38% | 69.38% |
| S06 | 77.50% | 80.00% | 73.75% | 81.25% | 80.00% | 77.50% |
| S07 | 89.38% | 91.25% | 89.38% | 89.38% | 85.63% | 80.63% |
| S08 | 94.38% | 96.88% | 95.63% | 93.75% | 92.50% | 90.63% |
| S09 | 96.88% | 96.88% | 98.13% | 96.25% | 99.38% | 94.38% |
| S10 | 67.50% | 65.63% | 65.63% | 65.00% | 70.63% | 65.63% |
| S11 | 91.88% | 90.00% | 91.88% | 90.63% | 89.38% | 91.88% |
| S12 | 88.13% | 88.13% | 85.00% | 86.25% | 83.75% | 72.50% |
| S13 | 89.38% | 91.25% | 85.00% | 88.75% | 86.25% | 84.38% |
| S14 | 90.63% | 89.38% | 81.88% | 83.75% | 83.13% | 76.88% |
| S15 | 90.00% | 88.75% | 93.75% | 90.63% | 86.25% | 88.75% |
| S16 | 87.50% | 86.25% | 85.63% | 84.38% | 91.88% | 73.75% |
| S17 | 88.13% | 90.63% | 89.38% | 89.38% | 91.25% | 86.25% |
| S18 | 88.75% | 91.25% | 88.13% | 85.63% | 88.75% | 87.50% |
| S19 | 98.13% | 96.25% | 98.13% | 98.75% | 96.88% | 96.25% |
| S20 | 66.25% | 70.63% | 61.88% | 63.75% | 68.75% | 65.00% |
| S21 | 80.63% | 80.63% | 79.38% | 75.63% | 76.88% | 75.00% |

> **Data sources**: `results/32_channel/{fdr,attention,csp,band_power,commercial,motor_cortex}/20260220_*_cross-subject_cbramod_imagery_binary.json`
