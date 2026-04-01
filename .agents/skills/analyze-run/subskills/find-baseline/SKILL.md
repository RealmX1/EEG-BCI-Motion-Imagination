---
name: find-baseline
description: >
  Sub-skill for `analyze-run`: query ExperimentDB to find the best baseline result
  for a given experimental setup. Use this when the task is category-wide historical
  lookup rather than deep inspection of one concrete run. Triggers include:
  "find baseline", "best baseline", "historical results", "what's the best accuracy for",
  "designated baseline", or "compare with baseline" when the user has not already pinned
  one specific run to inspect.
---

# Find Baseline Sub-skill

Use this bundled sub-skill from `analyze-run` when the user wants to search across runs instead of analyzing one concrete run.

Queries the project's ExperimentDB (`results/experiments.db`) to find baseline results for a given experimental setup. Runs marked as baseline (`is_baseline=1`) are highlighted with `**` in the output. Use `--baseline-only` to show only designated baselines. Unified model results are excluded by default; use `--include-unified` to include them.

## Usage

```bash
uv run python .agents/skills/analyze-run/subskills/find-baseline/scripts/query_baseline.py [options]
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
uv run python .agents/skills/analyze-run/subskills/find-baseline/scripts/query_baseline.py \
  --model cbramod --task binary --paradigm imagery \
  --type within_subject --channels 128 --subjects 21 \
  --post-hpo --top 5
```

For per-subject detail of a specific run:

```bash
uv run python .agents/skills/analyze-run/subskills/find-baseline/scripts/query_baseline.py --tag 20260321_0343
```

## Output Format

The script outputs formatted tables. Present the results directly to the user. Always include data source attribution:

```text
> **Data source**: ExperimentDB query — `SELECT ... FROM runs JOIN model_summaries WHERE ...`
```

If the user instead asks about one concrete `run_tag` or substring like `0329_1357`, go back to the parent skill and use `scripts/tools/describe_run.py` first.
