"""
Milestone checkpoint analysis: stopping strategy comparison.

Uses only history.json data (no .pt files needed).

Three-way comparison of stopping strategies:
  1. combined_score-selected (our system): epoch with best avg(val_acc, val_majority_acc)
  2. val_loss-selected (traditional): epoch with lowest val_loss
  3. test-optimal (oracle): epoch with highest test accuracy (unknowable in practice)

For val_loss-selected, we find the milestone epoch closest to the global val_loss
minimum, since we only have test evaluations at milestone epochs.
"""

import json
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_milestone_data(run_dir: str, task: str = "binary"):
    """Load milestone_test_results + val_loss from all subjects in a run."""
    pattern = f"{run_dir}/{task}/*/history.json"
    files = sorted(glob.glob(pattern))
    subjects = []
    for f in files:
        try:
            h = json.load(open(f))
            mtr = h.get("milestone_test_results", [])
            val_losses = h.get("val_loss", [])
            if not mtr or len(mtr) < 2 or not val_losses:
                continue
            parts = Path(f).parts
            subj = parts[-2]
            subjects.append({
                "subject": subj,
                "milestones": mtr,
                "val_losses": val_losses,
                "val_combined_scores": h.get("val_combined_score", []),
                "n_epochs_trained": len(h.get("train_loss", [])),
            })
        except Exception:
            pass
    return subjects


def find_best_valloss_milestone(milestones, val_losses):
    """Find the milestone epoch with the lowest val_loss.

    Since we only have test evaluations at milestone epochs, we pick the
    milestone whose epoch has the lowest val_loss (not the global minimum,
    which might fall on a non-milestone epoch).
    """
    best_ms = None
    best_loss = float("inf")
    for ms in milestones:
        ep = ms["epoch"]
        if ep <= len(val_losses):
            loss = val_losses[ep - 1]
            if loss < best_loss:
                best_loss = loss
                best_ms = ms
    return best_ms, best_loss


def analyze_three_way(subjects):
    """Three-way comparison: combined_score vs val_loss vs test-optimal."""
    results = []
    for s in subjects:
        ms = s["milestones"]
        val_losses = s["val_losses"]

        # Strategy 1: combined_score-selected (our system) = last milestone
        cs_selected = ms[-1]

        # Strategy 2: val_loss-selected = milestone with lowest val_loss
        vl_selected, vl_loss = find_best_valloss_milestone(ms, val_losses)
        if vl_selected is None:
            continue

        # Strategy 3: test-optimal (oracle)
        test_optimal = max(ms, key=lambda m: m["test_accuracy"])

        # Also find global val_loss minimum epoch (may not be a milestone)
        global_vl_min_epoch = int(np.argmin(val_losses)) + 1

        # val_loss at combined_score-selected epoch
        cs_epoch = cs_selected["epoch"]
        cs_valloss = val_losses[cs_epoch - 1] if cs_epoch <= len(val_losses) else None

        results.append({
            "subject": s["subject"],
            "n_milestones": len(ms),
            "n_epochs_trained": s["n_epochs_trained"],
            # Combined-score selected
            "cs_epoch": cs_selected["epoch"],
            "cs_test": cs_selected["test_accuracy"],
            "cs_val_combined": cs_selected["combined_score"],
            "cs_valloss": cs_valloss,
            # Val-loss selected
            "vl_epoch": vl_selected["epoch"],
            "vl_test": vl_selected["test_accuracy"],
            "vl_loss": vl_loss,
            "vl_val_combined": vl_selected["combined_score"],
            # Test-optimal (oracle)
            "opt_epoch": test_optimal["epoch"],
            "opt_test": test_optimal["test_accuracy"],
            # Global val_loss minimum
            "global_vl_min_epoch": global_vl_min_epoch,
        })
    return results


def print_report(run_name, subjects):
    """Print three-way comparison report."""
    print(f"\n{'='*80}")
    print(f"  Stopping Strategy Comparison: {run_name}")
    print(f"  {len(subjects)} subjects with milestone test data")
    print(f"{'='*80}\n")

    results = analyze_three_way(subjects)
    if not results:
        print("  No valid data.\n")
        return [], results

    # --- Section 1: Summary ---
    cs_tests = [r["cs_test"] for r in results]
    vl_tests = [r["vl_test"] for r in results]
    opt_tests = [r["opt_test"] for r in results]

    cs_mean = np.mean(cs_tests)
    vl_mean = np.mean(vl_tests)
    opt_mean = np.mean(opt_tests)

    # Win counts
    cs_wins_over_vl = sum(1 for r in results if r["cs_test"] > r["vl_test"])
    vl_wins_over_cs = sum(1 for r in results if r["vl_test"] > r["cs_test"])
    ties = sum(1 for r in results if r["cs_test"] == r["vl_test"])
    n = len(results)

    print("1. AGGREGATE COMPARISON")
    print(f"   {'Strategy':<30} {'Mean Test Acc':>14} {'vs Oracle Gap':>14}")
    print(f"   {'-'*60}")
    print(f"   {'Combined Score (our system)':<30} {cs_mean:>14.4f} {cs_mean - opt_mean:>+14.4f}")
    print(f"   {'Val Loss (traditional)':<30} {vl_mean:>14.4f} {vl_mean - opt_mean:>+14.4f}")
    print(f"   {'Test-Optimal (oracle)':<30} {opt_mean:>14.4f} {'—':>14}")
    print()
    print(f"   Head-to-head (combined_score vs val_loss):")
    print(f"     Combined Score wins: {cs_wins_over_vl}/{n}  |  Val Loss wins: {vl_wins_over_cs}/{n}  |  Ties: {ties}/{n}")
    print(f"     Mean diff (CS - VL): {np.mean([r['cs_test'] - r['vl_test'] for r in results]):+.4f}")

    # --- Section 2: Per-subject table ---
    print(f"\n2. PER-SUBJECT BREAKDOWN")
    print(f"   {'Subj':<5} {'CS Ep':>5} {'CS Test':>8} {'VL Ep':>5} {'VL Test':>8} "
          f"{'Opt Ep':>6} {'Opt Test':>8} {'Winner':>10}")
    print(f"   {'-'*62}")
    for r in sorted(results, key=lambda x: x["subject"]):
        if r["cs_test"] > r["vl_test"]:
            winner = "CS"
        elif r["vl_test"] > r["cs_test"]:
            winner = "VL"
        else:
            winner = "tie"
        # Mark if winner is also oracle
        if r["cs_test"] == r["opt_test"] and winner == "CS":
            winner += " (=opt)"
        elif r["vl_test"] == r["opt_test"] and winner == "VL":
            winner += " (=opt)"
        print(f"   {r['subject']:<5} {r['cs_epoch']:>5} {r['cs_test']:>8.4f} "
              f"{r['vl_epoch']:>5} {r['vl_test']:>8.4f} "
              f"{r['opt_epoch']:>6} {r['opt_test']:>8.4f} {winner:>10}")

    # --- Section 3: Epoch divergence ---
    print(f"\n3. EPOCH DIVERGENCE")
    cs_epochs = [r["cs_epoch"] for r in results]
    vl_epochs = [r["vl_epoch"] for r in results]
    opt_epochs = [r["opt_epoch"] for r in results]
    global_vl_epochs = [r["global_vl_min_epoch"] for r in results]
    epoch_diffs = [r["cs_epoch"] - r["vl_epoch"] for r in results]

    print(f"   Mean epochs: CS={np.mean(cs_epochs):.1f}, VL={np.mean(vl_epochs):.1f}, "
          f"Oracle={np.mean(opt_epochs):.1f}")
    print(f"   CS epoch - VL epoch: mean={np.mean(epoch_diffs):+.1f}, "
          f"median={np.median(epoch_diffs):+.1f}")
    print(f"   Global val_loss minimum epoch (may not be milestone): "
          f"mean={np.mean(global_vl_epochs):.1f}")

    # How often does CS select a later epoch than VL?
    cs_later = sum(1 for d in epoch_diffs if d > 0)
    cs_earlier = sum(1 for d in epoch_diffs if d < 0)
    same = sum(1 for d in epoch_diffs if d == 0)
    print(f"   CS selects later epoch: {cs_later}/{n}  |  "
          f"earlier: {cs_earlier}/{n}  |  same: {same}/{n}")

    # --- Section 4: Val loss at CS-selected vs VL-selected ---
    print(f"\n4. VAL LOSS COMPARISON")
    cs_losses = [r["cs_valloss"] for r in results if r["cs_valloss"] is not None]
    vl_losses = [r["vl_loss"] for r in results]
    if cs_losses:
        print(f"   Val loss at CS-selected epoch: mean={np.mean(cs_losses):.4f}")
        print(f"   Val loss at VL-selected epoch: mean={np.mean(vl_losses):.4f}")
        print(f"   Difference: {np.mean(cs_losses) - np.mean(vl_losses):+.4f} "
              f"(positive = CS trains past val_loss minimum)")

    return results


def plot_three_way(results, run_name, output_dir):
    """Generate three-way comparison visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n   [matplotlib not available, skipping plots]")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    results_sorted = sorted(results, key=lambda x: x["subject"])
    subj_labels = [r["subject"] for r in results_sorted]
    x = np.arange(len(subj_labels))
    bar_w = 0.25

    # Plot 1: Three-way test accuracy comparison
    ax = axes[0]
    cs_tests = [r["cs_test"] for r in results_sorted]
    vl_tests = [r["vl_test"] for r in results_sorted]
    opt_tests = [r["opt_test"] for r in results_sorted]
    ax.bar(x - bar_w, cs_tests, bar_w, label="Combined Score (ours)", color="#2E86AB", alpha=0.85)
    ax.bar(x, vl_tests, bar_w, label="Val Loss (traditional)", color="#F6AE2D", alpha=0.85)
    ax.bar(x + bar_w, opt_tests, bar_w, label="Test-Optimal (oracle)", color="#E94F37", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(subj_labels, rotation=45, fontsize=7)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Stopping Strategy Comparison")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: CS vs VL delta (positive = CS better)
    ax = axes[1]
    deltas = [r["cs_test"] - r["vl_test"] for r in results_sorted]
    colors = ["#2E86AB" if d >= 0 else "#F6AE2D" for d in deltas]
    bars = ax.bar(x, deltas, 0.6, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.5)
    mean_d = np.mean(deltas)
    ax.axhline(mean_d, color="gray", linestyle="--", linewidth=1,
               label=f"Mean: {mean_d:+.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(subj_labels, rotation=45, fontsize=7)
    ax.set_ylabel("Test Acc Difference")
    ax.set_title("Combined Score − Val Loss\n(blue=CS better, yellow=VL better)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Plot 3: Epoch comparison scatter
    ax = axes[2]
    cs_epochs = [r["cs_epoch"] for r in results_sorted]
    vl_epochs = [r["vl_epoch"] for r in results_sorted]
    opt_epochs = [r["opt_epoch"] for r in results_sorted]
    max_ep = max(max(cs_epochs), max(vl_epochs), max(opt_epochs)) + 2
    ax.scatter(cs_epochs, vl_epochs, c="#F6AE2D", s=60, alpha=0.8,
               edgecolors="black", linewidth=0.5, label="CS vs VL epoch", zorder=5)
    ax.scatter(cs_epochs, opt_epochs, c="#E94F37", s=40, alpha=0.6,
               edgecolors="black", linewidth=0.5, marker="^", label="CS vs Oracle epoch", zorder=4)
    ax.plot([0, max_ep], [0, max_ep], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("Combined Score Epoch")
    ax.set_ylabel("Other Strategy Epoch")
    ax.set_title("Epoch Selection Comparison")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim(0, max_ep)
    ax.set_ylim(0, max_ep)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    plt.suptitle(f"Stopping Strategy Analysis — {run_name}", fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = Path(output_dir) / "milestone_stopping_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n   Plot saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    baseline_runs = {
        "CBraMod binary (baseline 20260323_2237)": "results/20260323_2237_cbramod_within_subject",
        "CBraMod ternary (baseline 20260323_2320)": "results/20260323_2320_cbramod_within_subject",
        "CBraMod binary (20260321_0343)": "results/20260321_0343_cbramod_within_subject",
        "EEGNet binary (baseline 20260316_1411)": "results/20260316_1411_eegnet_within_subject",
        "CBraMod unified (baseline 20260320_0243)": "results/20260320_0243_cbramod_within_subject",
    }

    for name, run_dir in baseline_runs.items():
        for task in ["binary", "ternary", "unified"]:
            task_dir = Path(run_dir) / task
            if task_dir.exists():
                subjects = load_milestone_data(run_dir, task)
                if len(subjects) >= 10:
                    results = print_report(f"{name} / {task}", subjects)
                    if results:
                        plot_three_way(results, f"{name} / {task}", run_dir)
                    break
        else:
            print(f"\n[SKIP] {name}: no task directory with enough subjects found")
