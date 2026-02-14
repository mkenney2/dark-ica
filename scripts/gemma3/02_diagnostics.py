"""Diagnostics: variance analysis, kurtosis, PCA on residuals.

Usage:
    python scripts/gemma3/02_diagnostics.py --experiment gemma3_1b_262k
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from config import get_config
from utils import save_results_json, load_results_json, set_seeds


def run_diagnostics(cfg):
    cfg.ensure_dirs()
    set_seeds(cfg.random_seed)

    print("Loading data...")
    residuals = np.load(cfg.path("residuals.npy"))
    activations = np.load(cfg.path("activations.npy"))
    print(f"  residuals: {residuals.shape}")
    print(f"  activations: {activations.shape}")

    # --- Variance analysis ---
    total_var = np.var(activations, axis=0).sum()
    residual_var = np.var(residuals, axis=0).sum()
    frac_unexplained = residual_var / total_var

    print(f"\nVariance Analysis:")
    print(f"  Total activation variance: {total_var:.2f}")
    print(f"  Residual variance: {residual_var:.2f}")
    print(f"  Fraction unexplained by SAE: {frac_unexplained:.3f} ({frac_unexplained*100:.1f}%)")

    assert 0 < frac_unexplained < 1, f"Dark matter fraction {frac_unexplained} out of range!"

    # --- Per-dimension kurtosis ---
    kurt = kurtosis(residuals, axis=0)

    print(f"\nResidual Kurtosis (excess):")
    print(f"  Mean: {kurt.mean():.3f}")
    print(f"  Std: {kurt.std():.3f}")
    print(f"  Min: {kurt.min():.3f}")
    print(f"  Max: {kurt.max():.3f}")
    print(f"  Dims with kurtosis > 1: {(kurt > 1).sum()} / {len(kurt)}")
    print(f"  Dims with kurtosis > 3: {(kurt > 3).sum()} / {len(kurt)}")

    # Decision point
    print(f"\n--- DECISION POINT ---")
    if abs(kurt.mean()) < 0.5:
        print("Mean kurtosis near zero — residual is approximately Gaussian.")
        print("ICA may not find much structure.")
    else:
        print(f"Mean kurtosis = {kurt.mean():.3f} — non-Gaussianity detected!")
        print("Proceeding with ICA is well-motivated.")

    # Plot kurtosis
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(kurt, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Gaussian (kurtosis=0)')
    ax.axvline(x=kurt.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean={kurt.mean():.2f}')
    ax.set_xlabel("Excess Kurtosis", fontsize=12)
    ax.set_ylabel("Count (dimensions)", fontsize=12)
    ax.set_title(f"Per-Dimension Kurtosis of SAE Residual — {cfg.name}", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(cfg.plot_path("residual_kurtosis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved residual_kurtosis.png")

    # --- PCA ---
    n_pca = min(cfg.pca_max_components, residuals.shape[1])
    print(f"\nRunning PCA with up to {n_pca} components...")
    pca_diagnostic = PCA(n_components=n_pca)
    pca_diagnostic.fit(residuals)

    cumvar = np.cumsum(pca_diagnostic.explained_variance_ratio_)
    n_components_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    n_components_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    n_components_99 = int(np.searchsorted(cumvar, 0.99) + 1)

    print(f"PCA on Residuals:")
    print(f"  Components for 90% variance: {n_components_90}")
    print(f"  Components for 95% variance: {n_components_95}")
    print(f"  Components for 99% variance: {n_components_99}")
    if len(cumvar) >= 10:
        print(f"  Variance in first 10 components: {cumvar[9]*100:.1f}%")
    if len(cumvar) >= 50:
        print(f"  Variance in first 50 components: {cumvar[49]*100:.1f}%")

    # Plot PCA
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, 'b-', linewidth=2)
    ax.axhline(y=0.90, color='r', linestyle='--', alpha=0.7, label=f'90% ({n_components_90} components)')
    ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label=f'95% ({n_components_95} components)')
    ax.axvline(x=n_components_90, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel("Number of PCA Components", fontsize=12)
    ax.set_ylabel("Cumulative Variance Explained", fontsize=12)
    ax.set_title(f"PCA of SAE Residual — {cfg.name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(cfg.plot_path("residual_pca.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved residual_pca.png")

    # Recommended ICA components
    recommended = min(n_components_90, cfg.n_ica_components)
    print(f"\nRecommended ICA component count: {recommended}")
    print(f"  (Config setting: {cfg.n_ica_components})")

    # Save diagnostics
    diagnostics = {
        "experiment": cfg.name,
        "tokens_analyzed": int(residuals.shape[0]),
        "d_model": int(residuals.shape[1]),
        "variance_analysis": {
            "total_activation_variance": float(total_var),
            "residual_variance": float(residual_var),
            "fraction_unexplained_by_sae": float(frac_unexplained),
        },
        "residual_kurtosis": {
            "mean": float(kurt.mean()),
            "std": float(kurt.std()),
            "min": float(kurt.min()),
            "max": float(kurt.max()),
            "dims_gt_1": int((kurt > 1).sum()),
            "dims_gt_3": int((kurt > 3).sum()),
        },
        "pca": {
            "n_components_90pct": n_components_90,
            "n_components_95pct": n_components_95,
            "n_components_99pct": n_components_99,
            "recommended_n_ica": recommended,
        },
    }
    save_results_json(diagnostics, cfg.path("diagnostics.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    cfg = get_config(args.experiment)
    run_diagnostics(cfg)
