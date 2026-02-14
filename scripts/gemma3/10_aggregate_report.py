"""Aggregate results across experiments and produce comparison tables/plots.

Usage:
    python scripts/gemma3/10_aggregate_report.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import CONFIGS
from utils import load_results_json


def safe_mean(lst):
    return float(np.mean(lst)) if lst else float('nan')

def safe_median(lst):
    return float(np.median(lst)) if lst else float('nan')

def frac_above(lst, threshold):
    return float(np.mean([x >= threshold for x in lst])) if lst else float('nan')


def load_experiment_results(cfg):
    """Load all available results for an experiment."""
    results = {"name": cfg.name, "model": cfg.model_name, "sae_width": cfg.sae_width}

    # Diagnostics
    diag_path = cfg.path("diagnostics.json")
    if os.path.exists(diag_path):
        results["diagnostics"] = load_results_json(diag_path)

    # ICA results
    ica_path = cfg.path("ica_results.json")
    if os.path.exists(ica_path):
        results["ica"] = load_results_json(ica_path)

    # Similarity
    sim_path = cfg.path("similarity_results.json")
    if os.path.exists(sim_path):
        results["similarity"] = load_results_json(sim_path)

    # Statistics
    stats_path = cfg.path("statistics.json")
    if os.path.exists(stats_path):
        results["statistics"] = load_results_json(stats_path)

    # Random baseline
    rand_path = cfg.path("random_baseline_results.json")
    if os.path.exists(rand_path):
        results["random_baseline"] = load_results_json(rand_path)

    # Autointerp scores
    scores_path = cfg.autointerp_path("scores.json")
    if os.path.exists(scores_path):
        results["scores"] = load_results_json(scores_path)

    # Autointerp explanations
    explain_path = cfg.autointerp_path("explanations.json")
    if os.path.exists(explain_path):
        results["explanations"] = load_results_json(explain_path)

    # Max cosine sim per ICA component
    cos_path = cfg.path("max_cos_sim.npy")
    if os.path.exists(cos_path):
        results["max_cos_sim"] = np.load(cos_path)

    return results


def print_table(headers, rows, col_width=18):
    """Print a formatted table."""
    header_line = "".join(h.ljust(col_width) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("".join(str(v).ljust(col_width) for v in row))


def aggregate():
    print("=" * 80)
    print("  ICA DARK MATTER — CROSS-EXPERIMENT COMPARISON REPORT")
    print("=" * 80)

    # Load all experiments
    all_results = {}
    for name, cfg in CONFIGS.items():
        cfg.ensure_dirs()
        if os.path.exists(cfg.experiment_dir):
            results = load_experiment_results(cfg)
            if len(results) > 3:  # has actual data beyond name/model/sae_width
                all_results[name] = results
                print(f"\nLoaded: {name}")
            else:
                print(f"\nSkipped: {name} (no data)")
        else:
            print(f"\nSkipped: {name} (directory not found)")

    if not all_results:
        print("\nNo experiment data found!")
        return

    # --- TABLE 1: Dark Matter Fraction ---
    print(f"\n{'='*80}")
    print("TABLE 1: Dark Matter (Variance Unexplained by SAE)")
    print(f"{'='*80}")
    headers = ["Experiment", "Model", "SAE Width", "Dark Matter %"]
    rows = []
    for name, r in all_results.items():
        diag = r.get("diagnostics", {})
        var_info = diag.get("variance_analysis", {})
        frac = var_info.get("fraction_unexplained_by_sae")
        frac_str = f"{frac*100:.1f}%" if frac is not None else "N/A"
        rows.append([name, r["model"], f"{r['sae_width']:,}", frac_str])
    print_table(headers, rows)

    # --- TABLE 2: ICA Novelty Distribution ---
    print(f"\n{'='*80}")
    print("TABLE 2: ICA Component Novelty (Cosine Sim to Nearest SAE Feature)")
    print(f"{'='*80}")
    headers = ["Experiment", "Novel (<0.3)", "Medium", "High (>0.8)", "Mean Cos"]
    rows = []
    for name, r in all_results.items():
        sim = r.get("similarity", {})
        if sim:
            rows.append([
                name,
                sim.get("low_similarity_lt_0.3", "N/A"),
                sim.get("medium_similarity_0.3_to_0.8", "N/A"),
                sim.get("high_similarity_gt_0.8", "N/A"),
                f"{sim.get('mean_max_cosine', 0):.3f}",
            ])
    if rows:
        print_table(headers, rows)
    else:
        print("  No similarity data available.")

    # --- TABLE 3: Detection Accuracy ---
    print(f"\n{'='*80}")
    print("TABLE 3: Autointerp Detection Accuracy (Balanced)")
    print(f"{'='*80}")
    headers = ["Experiment", "ICA Mean", "SAE Mean", "Random Mean", "ICA Novel"]
    rows = []
    for name, r in all_results.items():
        scores = r.get("scores", {})
        max_cos = r.get("max_cos_sim")

        ica_accs = [v["balanced_accuracy"] for v in scores.get("ica", {}).values()
                    if v.get("balanced_accuracy") is not None]
        sae_accs = [v["balanced_accuracy"] for v in scores.get("sae", {}).values()
                    if v.get("balanced_accuracy") is not None]
        rand_accs = [v["balanced_accuracy"] for v in scores.get("random", {}).values()
                     if v.get("balanced_accuracy") is not None]

        # Novel ICA accuracy
        novel_accs = []
        if max_cos is not None and scores.get("ica"):
            for key, sc in scores["ica"].items():
                idx = int(key)
                if idx < len(max_cos) and max_cos[idx] < 0.3 and sc.get("balanced_accuracy") is not None:
                    novel_accs.append(sc["balanced_accuracy"])

        def fmt(lst):
            return f"{safe_mean(lst):.3f}" if lst else "N/A"

        rows.append([name, fmt(ica_accs), fmt(sae_accs), fmt(rand_accs), fmt(novel_accs)])

    if rows:
        print_table(headers, rows)
    else:
        print("  No scoring data available.")

    # --- TABLE 4: Kurtosis Comparison ---
    print(f"\n{'='*80}")
    print("TABLE 4: Kurtosis (Sparsity Proxy)")
    print(f"{'='*80}")
    headers = ["Experiment", "ICA Mean", "SAE Mean", "Random Mean", "Residual Mean"]
    rows = []
    for name, r in all_results.items():
        stats = r.get("statistics", {})
        diag = r.get("diagnostics", {})
        rand = r.get("random_baseline", {})

        ica_k = stats.get("ica_kurtosis", {}).get("mean")
        sae_k = stats.get("sae_kurtosis", {}).get("mean")
        rand_k = rand.get("kurtosis", {}).get("mean")
        res_k = diag.get("residual_kurtosis", {}).get("mean")

        def fmt(v):
            return f"{v:.2f}" if v is not None else "N/A"

        rows.append([name, fmt(ica_k), fmt(sae_k), fmt(rand_k), fmt(res_k)])

    if rows:
        print_table(headers, rows)

    # --- Summary comparison plot ---
    experiments_with_scores = [
        (name, r) for name, r in all_results.items()
        if r.get("scores")
    ]
    if len(experiments_with_scores) >= 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(experiments_with_scores))
        width = 0.25

        for i, (name, r) in enumerate(experiments_with_scores):
            scores = r.get("scores", {})
            ica_accs = [v["balanced_accuracy"] for v in scores.get("ica", {}).values()
                        if v.get("balanced_accuracy") is not None]
            sae_accs = [v["balanced_accuracy"] for v in scores.get("sae", {}).values()
                        if v.get("balanced_accuracy") is not None]
            rand_accs = [v["balanced_accuracy"] for v in scores.get("random", {}).values()
                         if v.get("balanced_accuracy") is not None]

            ica_mean = safe_mean(ica_accs) if ica_accs else 0
            sae_mean = safe_mean(sae_accs) if sae_accs else 0
            rand_mean = safe_mean(rand_accs) if rand_accs else 0

            ax.bar(i - width, ica_mean, width, label='ICA' if i == 0 else '', color='steelblue', alpha=0.8)
            ax.bar(i, sae_mean, width, label='SAE' if i == 0 else '', color='orange', alpha=0.8)
            ax.bar(i + width, rand_mean, width, label='Random' if i == 0 else '', color='gray', alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([name for name, _ in experiments_with_scores], rotation=15)
        ax.set_ylabel("Mean Detection Accuracy")
        ax.set_title("Autointerp Detection Accuracy: ICA vs SAE vs Random")
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Chance')
        ax.legend()
        ax.set_ylim([0, 1])
        plt.tight_layout()

        plot_path = os.path.join("experiments", "comparison_detection_accuracy.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved {plot_path}")

    # --- Save aggregate JSON ---
    aggregate_data = {}
    for name, r in all_results.items():
        entry = {
            "model": r["model"],
            "sae_width": r["sae_width"],
        }
        diag = r.get("diagnostics", {})
        if diag:
            entry["dark_matter_frac"] = diag.get("variance_analysis", {}).get("fraction_unexplained_by_sae")
        sim = r.get("similarity", {})
        if sim:
            entry["novel_components"] = sim.get("low_similarity_lt_0.3")
            entry["mean_max_cosine"] = sim.get("mean_max_cosine")

        scores = r.get("scores", {})
        for dtype in ["ica", "sae", "random"]:
            accs = [v["balanced_accuracy"] for v in scores.get(dtype, {}).values()
                    if v.get("balanced_accuracy") is not None]
            if accs:
                entry[f"{dtype}_mean_acc"] = safe_mean(accs)
                entry[f"{dtype}_median_acc"] = safe_median(accs)

        aggregate_data[name] = entry

    agg_path = os.path.join("experiments", "aggregate_results.json")
    with open(agg_path, "w") as f:
        json.dump(aggregate_data, f, indent=2)
    print(f"\nSaved {agg_path}")

    # --- Key findings ---
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    for name, r in all_results.items():
        scores = r.get("scores", {})
        rand_accs = [v["balanced_accuracy"] for v in scores.get("random", {}).values()
                     if v.get("balanced_accuracy") is not None]
        ica_accs = [v["balanced_accuracy"] for v in scores.get("ica", {}).values()
                    if v.get("balanced_accuracy") is not None]
        sae_accs = [v["balanced_accuracy"] for v in scores.get("sae", {}).values()
                    if v.get("balanced_accuracy") is not None]

        if rand_accs and ica_accs:
            ica_m = safe_mean(ica_accs)
            rand_m = safe_mean(rand_accs)
            print(f"\n  {name}:")
            print(f"    ICA acc ({ica_m:.3f}) vs Random acc ({rand_m:.3f}): "
                  f"{'ICA significantly above random' if ica_m > rand_m + 0.1 else 'ICA near random level'}")
            if sae_accs:
                sae_m = safe_mean(sae_accs)
                print(f"    SAE acc ({sae_m:.3f}) for comparison")


if __name__ == "__main__":
    aggregate()
