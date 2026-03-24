---
name: find-baseline
description: >
  Query ExperimentDB to find the best baseline result for a given experimental setup.
  Use this skill whenever you need to look up historical experiment results, find baselines
  for comparison, or check what the best known accuracy is for a specific configuration.
  Triggers on: "find baseline", "best baseline", "historical results", "what's the best accuracy for",
  "compare with baseline", or any request to look up past experiment performance.
  Always prefer this over manually searching JSON result files.
---

# Find Baseline

Queries the project's ExperimentDB (`results/experiments.db`) to find baseline results for a given experimental setup. Runs marked as baseline (`is_baseline=1`) are highlighted with `**` in the output. Use `--baseline-only` to show only designated baselines. Unified model results are excluded by default; use `--include-unified` to include them. This is the authoritative source for experiment results — always use this instead of searching JSON files manually.

## Usage

```
/find-baseline [options]
```

## Parameters

Parse the user's request to extract these parameters. Use defaults when not specified.

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| model | cbramod, eegnet | cbramod | Model architecture |
| task | binary, ternary, quaternary | binary | Classification task |
| paradigm | imagery, movement | imagery | Experiment paradigm |
| type | within_subject, cross_subject, transfer | within_subject | Training paradigm |
| channels | 4, 8, 32, 61, 128 | 128 | Number of EEG channels |
| subjects | integer | 21 | Minimum number of subjects |
| post-hpo | true, false | true | Only show runs after HPO (2026-03-20) |
| top | integer | 5 | Number of results to show |
| tag | run_tag string | (none) | Show per-subject detail for a specific run |
| baseline-only | true, false | false | Only show runs explicitly marked as baseline |
| include-unified | true, false | false | Include unified model results (excluded by default) |

## How to Execute

Run the query script:

```bash
uv run python .claude/skills/find-baseline/scripts/query_baseline.py \
  --model cbramod --task binary --paradigm imagery \
  --type within_subject --channels 128 --subjects 21 \
  --post-hpo --top 5
```

For per-subject detail of a specific run:

```bash
uv run python .claude/skills/find-baseline/scripts/query_baseline.py --tag 20260321_0343
```

## Output Format

The script outputs formatted tables. Present the results directly to the user. Always include the data source attribution:

```
> **Data source**: ExperimentDB query — `SELECT ... FROM runs JOIN model_summaries WHERE ...`
```

## Examples

**User**: "what's the best binary within-subject cbramod baseline?"
→ Run with defaults (all match this query)

**User**: "find baseline for ternary cross-subject"
→ `--task ternary --type cross_subject`

**User**: "show me the S01-S21 breakdown for run 20260321_0343"
→ `--tag 20260321_0343`

**User**: "compare my result against the best eegnet baseline including pre-HPO runs"
→ `--model eegnet --post-hpo false`

**User**: "show me only the designated baselines"
→ `--baseline-only`

**User**: "include unified model results"
→ `--include-unified`
