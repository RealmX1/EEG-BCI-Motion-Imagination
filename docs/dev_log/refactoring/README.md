# Codebase Refactoring Documentation

This folder contains detailed implementation logs for the 4-phase codebase reorganization completed on 2026-02-05.

## Overview

| Phase | Goal | Status |
|-------|------|--------|
| Phase 1 | Create new modules (`src/config/`, `src/results/`, `src/visualization/`) | Completed |
| Phase 2 | Split large files (`data_loader.py`, `train_within_subject.py`) | Completed |
| Phase 3 | Streamline script layer | Completed |
| Phase 4 | Script directory reorganization | Completed |

## Documents

- [phase2_implementation.md](phase2_implementation.md) - Splitting `data_loader.py` (2420 lines) and `train_within_subject.py` (2462 lines)
- [phase3_implementation.md](phase3_implementation.md) - Streamlining `run_full_comparison.py` and `run_single_model.py`
- [phase4_implementation.md](phase4_implementation.md) - Organizing `scripts/` into subdirectories

## Key Changes

### New Modules Created (Phase 1)
- `src/config/` - Centralized configuration (constants, presets, experiment configs)
- `src/results/` - Result management (dataclasses, serialization, caching, statistics)
- `src/visualization/` - Plotting functions (comparison, single model)

### Files Split (Phase 2)
- `data_loader.py` → `loader.py`, `discovery.py`, `pipeline.py`, `dataset.py`
- `train_within_subject.py` → `schedulers.py`, `evaluation.py`, `trainer.py`

### Scripts Reorganized (Phase 4)
```
scripts/
├── experiments/     # Training scripts
├── preprocessing/   # Data preprocessing
├── tools/          # Utilities
├── analysis/       # Analysis & research
└── internal/       # Internal tools
```

## Backward Compatibility

All original import paths remain functional through re-exports and wrapper scripts.
