# CBraMod Data Preprocessing ML Engineering Report

**Generated**: 2026-01-25 14:21

## Experiment Metadata

- **Paradigm**: imagery
- **N Experiments**: 1
- **Timestamp**: 2026-01-25T14:21:44.455771

## Experiment Summary

### Binary Classification

| Experiment | Group | Description | Mean Acc | Std | vs Baseline | p-value | Effect |
|------------|-------|-------------|----------|-----|-------------|---------|--------|
| A2 | A | Motor band upper limit (0.3-50Hz, 60Hz n... | 65.0% | 0.0% | - | - | - |

### Ternary Classification

| Experiment | Group | Description | Mean Acc | Std | vs Baseline | p-value | Effect |
|------------|-------|-------------|----------|-----|-------------|---------|--------|

## Group Analysis

### Group A: Filtering Parameters

**Binary:**
- A2: 65.0% +/- 0.0% (Motor band upper limit (0.3-50Hz, 60Hz notch))

**Ternary:**

### Group C: Normalization Strategies

**Binary:**

**Ternary:**

### Group D: Window Sliding Strategy

**Binary:**

**Ternary:**

### Group F: Data Quality Control

**Binary:**

**Ternary:**

## Recommendations

### Best Configuration: A2
- **Description**: Motor band upper limit (0.3-50Hz, 60Hz notch)
- **Mean Accuracy**: 65.0%

**Preprocessing Parameters:**
- Bandpass: 0.3-50.0 Hz
- Notch: 60.0 Hz
- Extra normalization: None
- Sliding step: 125.0 ms
- Amplitude threshold: 100.0 uV