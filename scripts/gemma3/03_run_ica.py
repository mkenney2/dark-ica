"""PCA whitening + FastICA on residuals, with robustness check.

Usage:
    python scripts/gemma3/03_run_ica.py --experiment gemma3_1b_262k
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import get_config
from utils import save_results_json, set_seeds

from sklearn.decomposition import PCA, FastICA


def run_ica(cfg):
    cfg.ensure_dirs()
    set_seeds(cfg.random_seed)

    print("Loading residuals...")
    residuals = np.load(cfg.path("residuals.npy"))
    print(f"  residuals: {residuals.shape}")

    n_components = cfg.n_ica_components
    print(f"\nRunning ICA with {n_components} components...")

    # PCA pre-whitening
    print("PCA whitening...")
    pca_pre = PCA(n_components=n_components, whiten=True, random_state=cfg.random_seed)
    residuals_whitened = pca_pre.fit_transform(residuals)
    pca_var_explained = pca_pre.explained_variance_ratio_.sum()
    print(f"  Whitened shape: {residuals_whitened.shape}")
    print(f"  PCA variance captured: {pca_var_explained*100:.1f}%")

    # FastICA
    print("Running FastICA...")
    ica = FastICA(
        n_components=n_components,
        algorithm='parallel',
        whiten=False,
        max_iter=cfg.ica_max_iter,
        tol=cfg.ica_tol,
        random_state=cfg.random_seed,
    )
    ica_sources = ica.fit_transform(residuals_whitened)
    print(f"  ICA converged in {ica.n_iter_} iterations")
    assert ica.n_iter_ < cfg.ica_max_iter, f"ICA did not converge! ({ica.n_iter_} >= {cfg.ica_max_iter})"

    # Compute ICA directions in original d_model-dim space
    ica_directions = ica.mixing_.T @ pca_pre.components_  # (n_components, d_model)
    ica_directions = ica_directions / np.linalg.norm(ica_directions, axis=1, keepdims=True)
    print(f"  ICA directions shape: {ica_directions.shape}")

    # Verify unit normalization
    norms = np.linalg.norm(ica_directions, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Directions not unit-normalized! norms: {norms}"

    # --- Robustness check ---
    print(f"\nRobustness check with seeds: {cfg.robustness_seeds}")
    all_directions = []
    for seed in cfg.robustness_seeds:
        ica_check = FastICA(
            n_components=n_components,
            algorithm='parallel',
            whiten=False,
            max_iter=cfg.ica_max_iter,
            tol=cfg.ica_tol,
            random_state=seed,
        )
        ica_check.fit(residuals_whitened)
        dirs = ica_check.mixing_.T @ pca_pre.components_
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        all_directions.append(dirs)
        print(f"  Seed {seed}: converged in {ica_check.n_iter_} iterations")

    robustness_results = []
    print(f"\nCross-seed robustness:")
    for i in range(len(cfg.robustness_seeds)):
        for j in range(i + 1, len(cfg.robustness_seeds)):
            cos_matrix = np.abs(all_directions[i] @ all_directions[j].T)
            best_matches = cos_matrix.max(axis=1)
            mean_match = best_matches.mean()
            min_match = best_matches.min()
            n_above_09 = (best_matches > 0.9).sum()
            print(f"  Seeds {cfg.robustness_seeds[i]} vs {cfg.robustness_seeds[j]}: "
                  f"mean={mean_match:.3f}, min={min_match:.3f}, "
                  f">0.9: {n_above_09}/{n_components}")
            robustness_results.append({
                "seed_a": cfg.robustness_seeds[i],
                "seed_b": cfg.robustness_seeds[j],
                "mean_best_match": float(mean_match),
                "min_best_match": float(min_match),
                "n_above_0.9": int(n_above_09),
            })

    del all_directions, residuals_whitened

    # Compute ICA activations
    print("\nComputing ICA activations...")
    activations = np.load(cfg.path("activations.npy"))
    ica_activations = activations @ ica_directions.T  # (N, n_components)
    print(f"  ICA activations shape: {ica_activations.shape}")
    del activations

    # Compute ICA reconstruction quality
    residual_var = np.var(residuals, axis=0).sum()
    ica_reconstruction = ica_sources @ ica.mixing_.T @ pca_pre.components_
    ica_recon_var = np.var(residuals - ica_reconstruction, axis=0).sum()
    ica_frac_explained = 1.0 - ica_recon_var / residual_var
    print(f"  ICA explains {ica_frac_explained*100:.1f}% of residual variance")
    del ica_reconstruction, ica_sources, residuals

    # Save
    np.save(cfg.path("ica_directions.npy"), ica_directions)
    np.save(cfg.path("ica_activations.npy"), ica_activations)
    print(f"\nSaved ica_directions.npy, ica_activations.npy")

    # Save ICA results
    ica_results = {
        "experiment": cfg.name,
        "n_components": n_components,
        "pca_variance_captured": float(pca_var_explained),
        "ica_iterations": int(ica.n_iter_),
        "ica_frac_residual_variance_explained": float(ica_frac_explained),
        "robustness": robustness_results,
    }
    save_results_json(ica_results, cfg.path("ica_results.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    cfg = get_config(args.experiment)
    run_ica(cfg)
