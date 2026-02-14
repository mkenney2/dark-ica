"""Compare ICA directions to SAE decoder via cosine similarity.

Usage:
    python scripts/gemma3/04_compare_sae.py --experiment gemma3_1b_262k
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import get_config
from utils import DEVICE, load_sae, save_results_json, set_seeds


def compare_sae(cfg):
    cfg.ensure_dirs()
    set_seeds(cfg.random_seed)

    print("Loading ICA directions...")
    ica_directions = np.load(cfg.path("ica_directions.npy"))
    print(f"  ica_directions: {ica_directions.shape}")

    print(f"Loading SAE: {cfg.sae_release} / {cfg.sae_id}")
    sae = load_sae(cfg)

    # Extract and normalize SAE decoder
    sae_decoder = sae.W_dec.detach().cpu().float().numpy()
    print(f"  SAE decoder shape: {sae_decoder.shape}")
    sae_decoder_norms = np.maximum(np.linalg.norm(sae_decoder, axis=1, keepdims=True), 1e-8)
    sae_decoder_normed = sae_decoder / sae_decoder_norms
    del sae_decoder, sae

    # Cosine similarity: (n_ica, d_model) @ (d_model, n_sae) -> (n_ica, n_sae)
    # For 262k SAE: 200 x 262k x 4B = ~210 MB — fits in memory
    print(f"\nComputing cosine similarity matrix ({ica_directions.shape[0]} x {sae_decoder_normed.shape[0]})...")
    cos_sim = ica_directions @ sae_decoder_normed.T
    print(f"  cos_sim shape: {cos_sim.shape} ({cos_sim.nbytes/1e6:.0f} MB)")

    max_cos_sim = np.max(np.abs(cos_sim), axis=1)
    best_sae_match = np.argmax(np.abs(cos_sim), axis=1)
    del cos_sim, sae_decoder_normed

    # Categorize
    n_high = int((max_cos_sim > 0.8).sum())
    n_medium = int(((max_cos_sim > 0.3) & (max_cos_sim <= 0.8)).sum())
    n_low = int((max_cos_sim <= 0.3).sum())
    n_total = len(max_cos_sim)

    print(f"\nICA-to-SAE Similarity:")
    print(f"  Mean max cosine: {max_cos_sim.mean():.3f}")
    print(f"  Median: {np.median(max_cos_sim):.3f}")
    print(f"  High (>0.8):    {n_high:3d} ({n_high/n_total*100:.0f}%) — SAE already knows")
    print(f"  Medium (0.3-0.8): {n_medium:3d} ({n_medium/n_total*100:.0f}%) — partial overlap")
    print(f"  Low (<0.3):     {n_low:3d} ({n_low/n_total*100:.0f}%) — genuinely novel!")

    # Verify values in valid range
    assert np.all((max_cos_sim >= 0) & (max_cos_sim <= 1)), "max_cos_sim out of [0,1]!"

    # Print most novel / most similar
    sorted_by_sim = np.argsort(max_cos_sim)
    print(f"\nMost novel (lowest SAE similarity):")
    for idx in sorted_by_sim[:5]:
        print(f"  Component {idx}: cos={max_cos_sim[idx]:.3f} (SAE feature {best_sae_match[idx]})")
    print(f"\nMost SAE-similar:")
    for idx in sorted_by_sim[-5:][::-1]:
        print(f"  Component {idx}: cos={max_cos_sim[idx]:.3f} (SAE feature {best_sae_match[idx]})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(max_cos_sim, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=0.3, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Novel/Medium (0.3)')
    ax.axvline(x=0.8, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Medium/High (0.8)')
    ax.set_xlabel("Max Cosine Similarity with Any SAE Feature", fontsize=12)
    ax.set_ylabel("Count (ICA Components)", fontsize=12)
    ax.set_title(f"ICA-SAE Similarity — {cfg.name}", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(cfg.plot_path("ica_sae_similarity.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved ica_sae_similarity.png")

    # Save
    np.save(cfg.path("max_cos_sim.npy"), max_cos_sim)
    np.save(cfg.path("best_sae_match.npy"), best_sae_match)

    similarity_results = {
        "experiment": cfg.name,
        "sae_width": cfg.sae_width,
        "n_ica_components": int(n_total),
        "mean_max_cosine": float(max_cos_sim.mean()),
        "median_max_cosine": float(np.median(max_cos_sim)),
        "high_similarity_gt_0.8": n_high,
        "medium_similarity_0.3_to_0.8": n_medium,
        "low_similarity_lt_0.3": n_low,
    }
    save_results_json(similarity_results, cfg.path("similarity_results.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    cfg = get_config(args.experiment)
    compare_sae(cfg)
