"""Collect top-activating examples for ICA, random, and SAE baseline directions.

Usage:
    python scripts/gemma3/07_collect_examples.py --experiment gemma3_1b_262k
    python scripts/gemma3/07_collect_examples.py --experiment gpt2_random_baseline --direction-types random
"""

import argparse
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import get_config
from utils import DEVICE, load_model, load_sae, get_context, save_results_json, set_seeds


def collect_examples_for_activations(act_values, tokens, tokenizer, cfg, top_k=None, min_examples=20):
    """Collect top activating examples with context, with deduplication."""
    if top_k is None:
        top_k = cfg.autointerp_top_k
    n_candidates = min(top_k * 2, len(act_values))
    top_indices = np.argsort(np.abs(act_values))[-n_candidates:][::-1]

    examples = []
    seen_contexts = set()
    for idx in top_indices:
        if len(examples) >= top_k:
            break
        context, token_text = get_context(
            idx, tokens, tokenizer,
            ctx_before=cfg.autointerp_context_before,
            ctx_after=cfg.autointerp_context_after,
        )
        context_key = context[:80]
        if context_key in seen_contexts:
            continue
        seen_contexts.add(context_key)
        examples.append({
            "context": context,
            "token": token_text,
            "activation": float(act_values[idx]),
            "position": int(idx),
        })
    return examples


def collect_negative_examples(act_values, tokens, tokenizer, cfg, n=None):
    """Collect random non-activating examples."""
    if n is None:
        n = cfg.autointerp_n_negatives
    abs_acts = np.abs(act_values)
    threshold = np.percentile(abs_acts, 25)

    if threshold <= 0:
        low_indices = np.where(abs_acts == 0)[0]
        if len(low_indices) == 0:
            low_indices = np.argsort(abs_acts)[:len(abs_acts) // 4]
    else:
        low_indices = np.where(abs_acts <= threshold)[0]

    if len(low_indices) == 0:
        return []

    chosen = np.random.choice(low_indices, size=min(n, len(low_indices)), replace=False)
    examples = []
    for idx in chosen:
        context, token_text = get_context(
            idx, tokens, tokenizer,
            ctx_before=cfg.autointerp_context_before,
            ctx_after=cfg.autointerp_context_after,
        )
        examples.append({
            "context": context,
            "token": token_text,
            "activation": float(act_values[idx]),
            "position": int(idx),
        })
    return examples


def collect_examples(cfg, direction_types):
    cfg.ensure_dirs()
    set_seeds(cfg.random_seed)

    print("Loading data...")
    tokens_flat = np.load(cfg.path("tokens.npy"))
    activations = np.load(cfg.path("activations.npy"))
    print(f"  tokens: {tokens_flat.shape}")
    print(f"  activations: {activations.shape}")

    # Load model for tokenizer
    model = load_model(cfg)
    tokenizer = model.tokenizer
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    n_tokens = activations.shape[0]

    # --- ICA examples ---
    if "ica" in direction_types:
        print(f"\n--- Collecting ICA examples ---")
        ica_activations = np.load(cfg.path("ica_activations.npy"))
        n_ica = ica_activations.shape[1]
        print(f"  ica_activations: {ica_activations.shape}")

        ica_examples = {}
        for i in range(n_ica):
            acts = ica_activations[:, i]
            positives = collect_examples_for_activations(acts, tokens_flat, tokenizer, cfg)
            negatives = collect_negative_examples(acts, tokens_flat, tokenizer, cfg)
            ica_examples[str(i)] = {"positives": positives, "negatives": negatives}
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{n_ica} ICA components done")
        del ica_activations

        save_results_json(ica_examples, cfg.autointerp_path("ica_examples.json"))
        total_pos = sum(len(v["positives"]) for v in ica_examples.values())
        print(f"  Saved: {len(ica_examples)} components, {total_pos} positive examples")

    # --- Random examples ---
    if "random" in direction_types:
        print(f"\n--- Collecting Random examples ---")
        random_activations = np.load(cfg.path("random_activations.npy"))
        n_random = random_activations.shape[1]
        print(f"  random_activations: {random_activations.shape}")

        random_examples = {}
        for i in range(n_random):
            acts = random_activations[:, i]
            positives = collect_examples_for_activations(acts, tokens_flat, tokenizer, cfg)
            negatives = collect_negative_examples(acts, tokens_flat, tokenizer, cfg)
            random_examples[str(i)] = {"positives": positives, "negatives": negatives}
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{n_random} random directions done")
        del random_activations

        save_results_json(random_examples, cfg.autointerp_path("random_examples.json"))
        total_pos = sum(len(v["positives"]) for v in random_examples.values())
        print(f"  Saved: {len(random_examples)} directions, {total_pos} positive examples")

    # --- SAE baseline examples ---
    if "sae" in direction_types:
        print(f"\n--- Collecting SAE baseline examples ---")
        sae = load_sae(cfg)
        n_sae_features = sae.W_dec.shape[0]

        # Load or compute fire rates
        fire_rate_path = cfg.path("sae_fire_rates.npy")
        if os.path.exists(fire_rate_path):
            fire_rate = np.load(fire_rate_path)
            print(f"  Loaded fire rates from {fire_rate_path}")
        else:
            print(f"  Computing SAE fire rates...")
            chunk_size = cfg.sae_chunk_size
            n_chunks = (n_tokens + chunk_size - 1) // chunk_size
            fire_counts = np.zeros(n_sae_features, dtype=np.int64)
            for c in range(n_chunks):
                s, e = c * chunk_size, min((c + 1) * chunk_size, n_tokens)
                chunk = torch.tensor(activations[s:e], device=DEVICE, dtype=torch.float32)
                with torch.no_grad():
                    sae_acts = sae.encode(chunk)
                    fire_counts += (sae_acts > 0).sum(dim=0).cpu().numpy()
                del sae_acts, chunk
            fire_rate = fire_counts / n_tokens
            np.save(fire_rate_path, fire_rate)

        # Select baseline SAE features
        active_indices = np.where(fire_rate > 0.001)[0]
        n_baseline = min(cfg.n_baseline_sae_features, len(active_indices))
        baseline_indices = np.sort(np.random.choice(active_indices, size=n_baseline, replace=False))
        print(f"  Selected {n_baseline} SAE features from {len(active_indices)} active features")

        # Extract activations for selected features in chunks
        print(f"  Extracting SAE feature activations...")
        chunk_size = cfg.sae_chunk_size
        n_chunks = (n_tokens + chunk_size - 1) // chunk_size
        sae_feature_acts = np.zeros((n_tokens, n_baseline), dtype=np.float32)
        for c in range(n_chunks):
            s, e = c * chunk_size, min((c + 1) * chunk_size, n_tokens)
            chunk = torch.tensor(activations[s:e], device=DEVICE, dtype=torch.float32)
            with torch.no_grad():
                sae_acts = sae.encode(chunk)
                sae_feature_acts[s:e] = sae_acts[:, baseline_indices].cpu().numpy()
            del sae_acts, chunk
            if (c + 1) % 10 == 0:
                print(f"    Chunk {c+1}/{n_chunks}")
        del sae
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Collect examples
        sae_examples = {}
        for j, feat_idx in enumerate(baseline_indices):
            acts = sae_feature_acts[:, j]
            positives = collect_examples_for_activations(acts, tokens_flat, tokenizer, cfg)
            negatives = collect_negative_examples(acts, tokens_flat, tokenizer, cfg)
            sae_examples[str(int(feat_idx))] = {"positives": positives, "negatives": negatives}
            if (j + 1) % 20 == 0:
                print(f"  {j+1}/{n_baseline} SAE features done")
        del sae_feature_acts

        save_results_json(sae_examples, cfg.autointerp_path("sae_examples.json"))
        save_results_json(baseline_indices.tolist(), cfg.autointerp_path("baseline_sae_indices.json"))
        total_pos = sum(len(v["positives"]) for v in sae_examples.values())
        print(f"  Saved: {len(sae_examples)} features, {total_pos} positive examples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--direction-types", nargs="+", default=["ica", "random", "sae"],
                        choices=["ica", "random", "sae"],
                        help="Which direction types to collect examples for")
    args = parser.parse_args()

    cfg = get_config(args.experiment)
    collect_examples(cfg, args.direction_types)
