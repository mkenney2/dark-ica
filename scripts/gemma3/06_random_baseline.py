"""Generate random direction baseline for control comparison.

Random unit vectors in d_model space should have:
- Low cosine similarity to SAE features (~0.03-0.10 for high-dim)
- Kurtosis near 0 (CLT: projection onto random direction -> Gaussian)
- Detection accuracy near 0.5 (random = uninterpretable)

Usage:
    python scripts/gemma3/06_random_baseline.py --experiment gemma3_1b_262k
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

sys.path.insert(0, os.path.dirname(__file__))
from config import get_config
from utils import DEVICE, load_sae, save_results_json, set_seeds


def run_random_baseline(cfg):
    cfg.ensure_dirs()
    set_seeds(cfg.random_seed)

    print("Loading activations...")
    activations = np.load(cfg.path("activations.npy"))
    print(f"  activations: {activations.shape}")

    # Generate random unit vectors
    n_dirs = cfg.n_random_directions
    d = cfg.d_model
    print(f"\nGenerating {n_dirs} random unit vectors in R^{d}...")
    random_dirs = np.random.randn(n_dirs, d).astype(np.float32)
    random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
    print(f"  random_directions shape: {random_dirs.shape}")

    # Project activations onto random directions
    print("Computing random activations...")
    random_activations = activations @ random_dirs.T  # (N, n_dirs)
    print(f"  random_activations shape: {random_activations.shape}")

    # Kurtosis of random projections
    random_kurt = kurtosis(random_activations, axis=0)
    print(f"\nRandom direction kurtosis:")
    print(f"  Mean: {random_kurt.mean():.3f}")
    print(f"  Std: {random_kurt.std():.3f}")
    print(f"  Min: {random_kurt.min():.3f}, Max: {random_kurt.max():.3f}")
    print(f"  Expected: ~0 (Gaussian by CLT)")

    # Cosine similarity to SAE decoder
    print(f"\nComputing cosine similarity to SAE decoder...")
    sae = load_sae(cfg)
    sae_decoder = sae.W_dec.detach().cpu().float().numpy()
    sae_decoder_norms = np.maximum(np.linalg.norm(sae_decoder, axis=1, keepdims=True), 1e-8)
    sae_decoder_normed = sae_decoder / sae_decoder_norms
    del sae_decoder, sae

    cos_sim = random_dirs @ sae_decoder_normed.T
    max_cos_sim_random = np.max(np.abs(cos_sim), axis=1)
    del cos_sim, sae_decoder_normed

    print(f"\nRandom-to-SAE cosine similarity:")
    print(f"  Mean max cosine: {max_cos_sim_random.mean():.4f}")
    print(f"  Std: {max_cos_sim_random.std():.4f}")
    print(f"  Max: {max_cos_sim_random.max():.4f}")
    expected_cos = 1.0 / np.sqrt(d)  # expected for random vectors
    print(f"  Expected scale: ~{expected_cos:.4f} (1/sqrt(d_model))")

    # Compare with ICA if available
    ica_cos_sim_path = cfg.path("max_cos_sim.npy")
    if os.path.exists(ica_cos_sim_path):
        ica_max_cos = np.load(ica_cos_sim_path)
        print(f"\n  ICA mean max cosine: {ica_max_cos.mean():.4f} (for comparison)")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Kurtosis comparison
    ax = axes[0]
    if os.path.exists(cfg.path("ica_activations.npy")):
        ica_acts = np.load(cfg.path("ica_activations.npy"))
        ica_kurt = kurtosis(ica_acts, axis=0)
        ax.hist(np.clip(ica_kurt, -10, 200), bins=30, alpha=0.5,
                label=f'ICA (n={len(ica_kurt)})', density=True, color='steelblue')
        del ica_acts
    ax.hist(np.clip(random_kurt, -5, 10), bins=30, alpha=0.5,
            label=f'Random (n={len(random_kurt)})', density=True, color='gray')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Gaussian (k=0)')
    ax.set_xlabel("Excess Kurtosis")
    ax.set_ylabel("Density")
    ax.set_title("Kurtosis: ICA vs Random Directions")
    ax.legend()

    # Cosine similarity comparison
    ax = axes[1]
    if os.path.exists(ica_cos_sim_path):
        ica_max_cos = np.load(ica_cos_sim_path)
        ax.hist(ica_max_cos, bins=30, alpha=0.5,
                label=f'ICA (n={len(ica_max_cos)})', density=True, color='steelblue')
    ax.hist(max_cos_sim_random, bins=30, alpha=0.5,
            label=f'Random (n={len(max_cos_sim_random)})', density=True, color='gray')
    ax.set_xlabel("Max Cosine Sim to SAE")
    ax.set_ylabel("Density")
    ax.set_title("SAE Similarity: ICA vs Random")
    ax.legend()

    plt.tight_layout()
    plt.savefig(cfg.plot_path("random_baseline.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved random_baseline.png")

    # Save
    np.save(cfg.path("random_directions.npy"), random_dirs)
    np.save(cfg.path("random_activations.npy"), random_activations)

    baseline_results = {
        "experiment": cfg.name,
        "n_random_directions": n_dirs,
        "d_model": d,
        "kurtosis": {
            "mean": float(random_kurt.mean()),
            "std": float(random_kurt.std()),
            "min": float(random_kurt.min()),
            "max": float(random_kurt.max()),
        },
        "sae_cosine_sim": {
            "mean_max": float(max_cos_sim_random.mean()),
            "std": float(max_cos_sim_random.std()),
            "max": float(max_cos_sim_random.max()),
        },
    }
    save_results_json(baseline_results, cfg.path("random_baseline_results.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    cfg = get_config(args.experiment)
    run_random_baseline(cfg)
