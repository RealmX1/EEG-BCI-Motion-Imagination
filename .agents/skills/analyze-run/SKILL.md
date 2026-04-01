---
name: analyze-run
description: >
  Inspect an ExperimentDB run from a run_tag or run_tag substring and explain what it is.
  Use this skill whenever the user wants to analyze one concrete run, asks about a run_tag like
  "20260329_1357" or a substring like "0329_1357", wants the category / command / test accuracy /
  baseline comparison / per-subject breakdown for a run, or wants a concise historical summary for
  one experiment. For category-wide baseline lookup, delegate to the bundled sub-skill
  `subskills/find-baseline/SKILL.md`.
---

# Analyze Run

Use this skill to inspect one concrete ExperimentDB run deeply.

## Primary entry point

The main component of this skill is:

```bash
uv run python scripts/tools/describe_run.py 0329_1357
```

Use it by default when the user asks to:

- analyze a specific run
- inspect a `run_tag` or substring
- show the run category
- show the original command used to launch the run
- report test accuracy / model summaries / baseline comparison
- show per-subject detail for a concrete run

The script accepts shortened substrings such as `0329_1357`. If multiple runs match, it prints candidates and chooses the latest one by default. Add `--strict` to require an unambiguous match.

## Common commands

Basic analysis:

```bash
uv run python scripts/tools/describe_run.py 0329_1357
```

Disambiguate with filters:

```bash
uv run python scripts/tools/describe_run.py 0330_22 --type cross_subject --channels 4
```

Include per-subject rows:

```bash
uv run python scripts/tools/describe_run.py 20260330_2214 --show-subjects
```

## What to report

Summarize the fields that matter for the user's question, usually:

- resolved `run_tag`
- category: `experiment_type / paradigm / task / channels / channel_config`
- launch command from `runs.command`
- model summary metrics from ExperimentDB
- baseline comparison in the same category when available
- within-run EEGNet vs CBraMod comparison when available

Always cite the data source and resolved `run_tag`.

## When to delegate to the baseline sub-skill

If the task is not "inspect this run", but instead:

- find the best baseline in a category
- find designated baselines
- compare with the best historical baseline across many runs
- search historical best accuracy for a setup

then read and follow:

`subskills/find-baseline/SKILL.md`

That sub-skill owns the "search across runs" workflow. This top-level skill owns the "analyze one run" workflow.
