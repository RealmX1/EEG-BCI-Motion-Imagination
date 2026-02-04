#!/bin/bash
# Fill missing S01-S07 3-class caches using preprocess_zip.py

echo "==============================================================="
echo "Filling Missing S01-S07 3-class Caches"
echo "==============================================================="
echo ""
echo "Missing configurations:"
echo "  - CBraMod 3-class: S01-S07 (all)"
echo "  - EEGNet 3-class: S05-S07"
echo ""

# Use preprocess_zip.py with specific subjects, models, and tasks
# Assumes data is already extracted to data/S01-S07/

echo "[1/2] Preprocessing CBraMod 3-class for all S01-S07..."
uv run python scripts/preprocess_zip.py \
    --preprocess-only \
    --subjects S01 S02 S03 S04 S05 S06 S07 \
    --models cbramod \
    --tasks ternary \
    --paradigm imagery

echo ""
echo "[2/2] Preprocessing EEGNet 3-class for S05-S07..."
uv run python scripts/preprocess_zip.py \
    --preprocess-only \
    --subjects S05 S06 S07 \
    --models eegnet \
    --tasks ternary \
    --paradigm imagery

echo ""
echo "==============================================================="
echo "Done! Verify with:"
echo "  python scripts/cache_helper.py --stats"
echo "==============================================================="
