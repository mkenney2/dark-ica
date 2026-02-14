"""Kurtosis comparison: ICA components vs SAE features (streaming).

Usage:
    python scripts/gemma3/05_statistics.py --experiment gemma3_1b_262k
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


def run_statistics(cfg):
    cfg.ensure_dirs()
    set_seeds(cfg.random_seed)

    print("Loading data...")
    ica_activations = np.load(cfg.path("ica_activations.npy"))
    activations = np.load(cfg.path("activations.npy"))
    print(f"  ica_activations: {ica_activations.shape}")
    print(f"  activations: {activations.shape}")

    # --- ICA kurtosis ---
    ica_kurtosis = kurtosis(ica_activations, axis=0)
    print(f"\nICA component kurtosis:")
    print(f"  Mean: {ica_kurtosis.mean():.2f}")
    print(f"  Median: {np.median(ica_kurtosis):.2f}")
    print(f"  Min: {ica_kurtosis.min():.2f}, Max: {ica_kurtosis.max():.2f}")

    # --- SAE feature kurtosis (streaming) ---
    print(f"\nComputing SAE feature kurtosis (streaming)...")
    sae = load_sae(cfg)
    n_features = sae.W_dec.shape[0]
    n_samples = activations.shape[0]
    chunk_size = cfg.sae_chunk_size
    n_chunks = (n_samples + chunk_size - 1) // chunk_size

    # Single-pass moment accumulation
    fire_counts = np.zeros(n_features, dtype=np.float64)
    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_x2 = np.zeros(n_features, dtype=np.float64)
    sum_x3 = np.zeros(n_features, dtype=np.float64)
    sum_x4 = np.zeros(n_features, dtype=np.float64)

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        chunk = torch.tensor(activations[start:end], device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            sae_acts = sae.encode(chunk).cpu().numpy().astype(np.float64)
        fire_counts += (sae_acts > 0).sum(axis=0)
        sum_x += sae_acts.sum(axis=0)
        sum_x2 += (sae_acts ** 2).sum(axis=0)
        sum_x3 += (sae_acts ** 3).sum(axis=0)
        sum_x4 += (sae_acts ** 4).sum(axis=0)
        del sae_acts, chunk
        if (i + 1) % 10 == 0:
            print(f"  Chunk {i+1}/{n_chunks}")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    del sae

    # Active features
    fire_rate = fire_counts / n_samples
    active_mask = fire_rate > 0.001
    n_active = int(active_mask.sum())
    print(f"\n  Active SAE features (fire >0.1%): {n_active} / {n_features}")

    # Compute kurtosis from moments
    mean = sum_x[active_mask] / n_samples
    ex2 = sum_x2[active_mask] / n_samples
    ex3 = sum_x3[active_mask] / n_samples
    ex4 = sum_x4[active_mask] / n_samples

    var = ex2 - mean**2
    m4 = ex4 - 4*mean*ex3 + 6*(mean**2)*ex2 - 3*mean**4

    valid = var > 1e-12
    sae_kurtosis = np.full(n_active, np.nan)
    sae_kurtosis[valid] = (m4[valid] / var[valid]**2) - 3.0
    sae_kurtosis_clean = sae_kurtosis[~np.isnan(sae_kurtosis)]

    print(f"\nSAE feature kurtosis ({len(sae_kurtosis_clean)} valid features):")
    print(f"  Mean: {sae_kurtosis_clean.mean():.2f}")
    print(f"  Median: {np.median(sae_kurtosis_clean):.2f}")
    print(f"  Min: {sae_kurtosis_clean.min():.2f}, Max: {sae_kurtosis_clean.max():.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ica_kurt_clipped = np.clip(ica_kurtosis, -10, 200)
    sae_kurt_clipped = np.clip(sae_kurtosis_clean, -10, 200)
    ax.hist(ica_kurt_clipped, bins=50, alpha=0.5,
            label=f'ICA components (n={len(ica_kurtosis)})', density=True, color='steelblue')
    ax.hist(sae_kurt_clipped, bins=50, alpha=0.5,
            label=f'SAE features (n={len(sae_kurtosis_clean)})', density=True, color='orange')
    ax.set_xlabel("Excess Kurtosis", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Kurtosis: ICA vs SAE Features — {cfg.name}", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(cfg.plot_path("ica_vs_sae_kurtosis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved ica_vs_sae_kurtosis.png")

    # Save fire rates for later use (feature selection in 07)
    np.save(cfg.path("sae_fire_rates.npy"), fire_rate)

    stats = {
        "experiment": cfg.name,
        "ica_kurtosis": {
            "mean": float(ica_kurtosis.mean()),
            "median": float(np.median(ica_kurtosis)),
            "min": float(ica_kurtosis.min()),
            "max": float(ica_kurtosis.max()),
        },
        "sae_kurtosis": {
            "n_active": n_active,
            "n_valid": len(sae_kurtosis_clean),
            "mean": float(sae_kurtosis_clean.mean()),
            "median": float(np.median(sae_kurtosis_clean)),
            "min": float(sae_kurtosis_clean.min()),
            "max": float(sae_kurtosis_clean.max()),
        },
    }
    save_results_json(stats, cfg.path("statistics.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    cfg = get_config(args.experiment)
    run_statistics(cfg)
