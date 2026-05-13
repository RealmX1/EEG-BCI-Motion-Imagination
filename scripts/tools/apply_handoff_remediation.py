"""Apply 2026-05-13 handoff-remediation patches to backward-search group JSONs.

Adds the 7 handoff-cited rationale augmentations identified by the remediation
agent (after the initial 3 agents missed 6 of the 13 handoff documents). Idempotent:
re-running is safe — the patch detects already-applied augmentations via the
'[handoff-remediation]' marker and skips them.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("docs/dev_log/backward_search_2026-05-13")
MARKER = "[handoff-remediation 2026-05-13]"

# (bucket_file, group_name, additional_rationale_text, confidence_upgrade)
PATCHES = [
    (
        "bucket_a_128ch_main.json",
        "leave_3_out_sensitivity_may5",
        "docs/handoffs/2026-05-04_overnight_results.md §3.2 confirms this as "
        "'Task B -- leave-S04/S10/S14-out sensitivity' (N=18 robustness check "
        "supporting paper's claim that removing 3 high-artifact subjects does not "
        "collapse cross-subject CBraMod accuracy).",
        "high",
    ),
    (
        "bucket_c_reduced_channel.json",
        "reduced_channel_32ch_fdr_extended_ternary_and_transfer",
        "Run 20260505_0212 is explicitly designated 'Task C -- 32ch FDR Transfer' "
        "(CBraMod, N=21) in docs/handoffs/2026-05-04_overnight_results.md §3.3, "
        "testing paper claim T1.1 that transfer on 32ch FDR provides a small "
        "positive lift over cross-subject alone.",
        "high",
    ),
    (
        "bucket_a_128ch_main.json",
        "dapt_v3_downstream_eval_may5",
        "Both run_tags form the cross-subject half of the V3 4-condition downstream "
        "evaluation matrix per docs/handoffs/2026-05-04_overnight_results.md §9.3 "
        "(Task D continuation). The within-subject pair 20260505_2012/2033 noted in "
        "§9.5 was JSON-only and is not in ExperimentDB.",
        "high",
    ),
    (
        "bucket_c_reduced_channel.json",
        "reduced_channel_64ch_method_agnostic_matrix_closure",
        "Run 20260505_2223 is designated 'Task E-1: 64ch CBraMod cross-subject "
        "(binary)' in docs/handoffs/2026-05-05_paper_review_experiments.md §3.1 -- "
        "the planned key intermediate channel-count point filling paper §3.5 "
        "'channel sweet spot' between 32ch FDR and 128ch baselines.",
        "high",
    ),
    (
        "bucket_c_reduced_channel.json",
        "reduced_channel_4ch_method_sweep_attention_bandpower_csp_fdr",
        "Runs 20260505_2246 (CSP) and 20260505_2308 (band_power) are designated "
        "'Task E-2: 4ch CSP + Band Power cross-subject (binary)' in "
        "docs/handoffs/2026-05-05_paper_review_experiments.md §3.2 -- completing "
        "the §3.5.3 method-comparison matrix at the extreme 4-channel limit.",
        "high",
    ),
    (
        "bucket_a_128ch_main.json",
        "eegnet_transfer_baseline_followups_may5_7",
        "Runs designated 'Task E-3: EEGNet 128ch transfer (binary + ternary)' in "
        "docs/handoffs/2026-05-05_paper_review_experiments.md §3.3, filling the "
        "§3.3 transfer-learning EEGNet column. The May 5 abort retries "
        "(20260505_2318/2321) and May 7 reruns (20260507_1835/1913) are "
        "documented in 2026-05-05_paper_review_results.md §5.1 / §5.4.",
        "high",
    ),
    (
        "bucket_c_reduced_channel.json",
        "reduced_channel_8ch_method_sweep_fdr_csp_bandpower",
        "Run 20260506_2159 is designated 'Task C-2: 8ch Band Power transfer "
        "(binary)' in docs/handoffs/2026-05-05_paper_review_experiments.md §3.4 -- "
        "second data point for paper §3.5.4 testing 'fewer channels => larger "
        "transfer gain' hypothesis. Run 20260507_1958 is the post-Agg-fix "
        "reproducibility rerun per 2026-05-05_paper_review_results.md §5.4.",
        "high",
    ),
]


def patch_one(json_path: Path, group_name: str, addition: str, new_conf: str) -> bool:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for sg in data["subgroups"]:
        if sg["name"] == group_name:
            tagged = f"{MARKER} {addition}"
            if MARKER in (sg.get("rationale") or ""):
                return False  # already applied
            sg["rationale"] = f"{sg['rationale']} {tagged}"
            if new_conf:
                sg["confidence"] = new_conf
            json_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return True
    raise KeyError(f"Group '{group_name}' not found in {json_path}")


def main():
    print(f"Applying {len(PATCHES)} handoff-remediation patches...\n")
    applied = 0
    skipped = 0
    for file_name, group, addition, conf in PATCHES:
        path = ROOT / file_name
        try:
            ok = patch_one(path, group, addition, conf)
            if ok:
                applied += 1
                print(f"  [patched]  {file_name}::{group}  -> confidence={conf}")
            else:
                skipped += 1
                print(f"  [skip]     {file_name}::{group}  (already patched)")
        except KeyError as e:
            print(f"  [ERROR]    {e}")
    print(f"\nDone: {applied} applied, {skipped} skipped (already idempotent)")


if __name__ == "__main__":
    main()
