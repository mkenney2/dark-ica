"""Collect model activations, run SAE encode/decode, compute residuals.

Usage:
    python scripts/gemma3/01_collect_activations.py --experiment gemma3_1b_262k
    python scripts/gemma3/01_collect_activations.py --experiment gemma3_1b_16k --reuse-activations-from gemma3_1b_262k
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import get_config
from utils import DEVICE, print_device_info, load_model, load_sae, get_exclude_mask, set_seeds


def collect_activations(cfg, reuse_from=None):
    cfg.ensure_dirs()
    set_seeds(cfg.random_seed)
    print_device_info()

    # --- If reusing activations from another experiment, skip model forward pass ---
    if reuse_from is not None:
        reuse_cfg = get_config(reuse_from)
        act_path = reuse_cfg.path("activations.npy")
        tok_path = reuse_cfg.path("tokens.npy")
        print(f"Reusing activations from {reuse_from}: {act_path}")

        activations = np.load(act_path)
        tokens_flat = np.load(tok_path)
        print(f"  activations: {activations.shape}")
        print(f"  tokens: {tokens_flat.shape}")

        # Just re-run SAE encode/decode with the new SAE
        print(f"\nLoading SAE: {cfg.sae_release} / {cfg.sae_id}")
        sae = load_sae(cfg)

        n_samples = activations.shape[0]
        chunk_size = cfg.sae_chunk_size
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        all_residuals = []

        print(f"Computing residuals with new SAE in {n_chunks} chunks...")
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n_samples)
            chunk = torch.tensor(activations[start:end], device=DEVICE, dtype=torch.float32)
            with torch.no_grad():
                feature_acts = sae.encode(chunk)
                reconstructed = sae.decode(feature_acts)
                residual = chunk - reconstructed
            all_residuals.append(residual.cpu().float().numpy())
            del feature_acts, reconstructed, residual, chunk
            if (i + 1) % 10 == 0:
                print(f"  Chunk {i+1}/{n_chunks}")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        residuals = np.concatenate(all_residuals, axis=0)
        del all_residuals

        # Save (copy activations and tokens, save new residuals)
        np.save(cfg.path("activations.npy"), activations)
        np.save(cfg.path("tokens.npy"), tokens_flat)
        np.save(cfg.path("residuals.npy"), residuals)
        print(f"\nSaved activations, tokens, residuals to {cfg.experiment_dir}/")
        print(f"  residuals: {residuals.shape} ({residuals.nbytes/1e9:.2f} GB)")
        return

    # --- Full collection: model forward pass + SAE ---
    print(f"\nLoading model: {cfg.model_name}")
    model = load_model(cfg)

    print(f"\nLoading SAE: {cfg.sae_release} / {cfg.sae_id}")
    sae = load_sae(cfg)

    print(f"\nModel info:")
    print(f"  d_model: {model.cfg.d_model}")
    print(f"  n_layers: {model.cfg.n_layers}")

    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_residuals = []
    all_activations = []
    all_tokens = []
    tokens_collected = 0
    batch_texts = []

    print(f"\nCollecting ~{cfg.target_tokens:,} tokens...")
    print(f"  Hook: {cfg.hook_name}")
    print(f"  Exclude token IDs: {cfg.exclude_token_ids}")
    print(f"  Auto-prepends BOS: {cfg.auto_prepends_bos}")

    for i, example in enumerate(dataset):
        text = example["text"]
        if len(text.strip()) < 20:
            continue
        batch_texts.append(text)

        if len(batch_texts) < cfg.batch_size:
            continue

        # Tokenize
        if cfg.auto_prepends_bos:
            # Gemma tokenizer auto-prepends BOS; just truncate
            tokenized = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=cfg.context_len,
            )
            tokens = tokenized["input_ids"].to(DEVICE)
        else:
            # GPT-2: manually prepend BOS
            tokenized = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=cfg.context_len - 1,
            )
            input_ids = tokenized["input_ids"].to(DEVICE)
            bos_id = tokenizer.bos_token_id
            bos = torch.full((input_ids.shape[0], 1), bos_id, dtype=input_ids.dtype, device=DEVICE)
            tokens = torch.cat([bos, input_ids], dim=1)[:, :cfg.context_len]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[cfg.hook_name],
            )
            acts = cache[cfg.hook_name]
            acts_flat = acts.reshape(-1, acts.shape[-1])
            tokens_flat_batch = tokens.reshape(-1)

            keep_mask = get_exclude_mask(tokens_flat_batch, cfg.exclude_token_ids)
            acts_kept = acts_flat[keep_mask]

            # SAE encode/decode in sub-chunks if needed
            n_kept = acts_kept.shape[0]
            chunk_size = cfg.sae_chunk_size
            residual_chunks = []
            for cs in range(0, n_kept, chunk_size):
                ce = min(cs + chunk_size, n_kept)
                chunk = acts_kept[cs:ce]
                feature_acts = sae.encode(chunk)
                reconstructed = sae.decode(feature_acts)
                residual_chunks.append((chunk - reconstructed).cpu().float().numpy())
                del feature_acts, reconstructed

            residual = np.concatenate(residual_chunks, axis=0)
            all_residuals.append(residual)
            all_activations.append(acts_kept.cpu().float().numpy())
            all_tokens.append(tokens_flat_batch[keep_mask].cpu().numpy())

        tokens_collected += int(keep_mask.sum())
        batch_texts = []

        if (i // cfg.batch_size) % 25 == 0:
            mem_gb = sum(r.nbytes for r in all_residuals) / 1e9
            print(f"  {tokens_collected:,} / {cfg.target_tokens:,} tokens | {mem_gb:.1f} GB stored")

        if tokens_collected >= cfg.target_tokens:
            break

        if i % 500 == 0 and DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Concatenate
    residuals = np.concatenate(all_residuals, axis=0)
    activations = np.concatenate(all_activations, axis=0)
    tokens_flat = np.concatenate(all_tokens, axis=0)
    del all_residuals, all_activations, all_tokens
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"\nCollection complete!")
    print(f"  Tokens kept: {residuals.shape[0]:,}")
    print(f"  Residuals shape: {residuals.shape} ({residuals.nbytes/1e9:.2f} GB)")
    print(f"  Activations shape: {activations.shape} ({activations.nbytes/1e9:.2f} GB)")

    # Verify
    assert residuals.shape[1] == cfg.d_model, f"Expected d_model={cfg.d_model}, got {residuals.shape[1]}"
    assert not np.any(np.isnan(residuals)), "NaN in residuals!"
    assert not np.any(np.isinf(residuals)), "Inf in residuals!"
    res_norm = np.linalg.norm(residuals, axis=1).mean()
    act_norm = np.linalg.norm(activations, axis=1).mean()
    print(f"  Mean residual norm: {res_norm:.4f}")
    print(f"  Mean activation norm: {act_norm:.4f}")
    assert res_norm < act_norm, "Residual norm should be < activation norm"

    # Save
    np.save(cfg.path("residuals.npy"), residuals)
    np.save(cfg.path("activations.npy"), activations)
    np.save(cfg.path("tokens.npy"), tokens_flat)
    print(f"\nSaved residuals.npy, activations.npy, tokens.npy to {cfg.experiment_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment config name")
    parser.add_argument("--reuse-activations-from", default=None,
                        help="Reuse activations from another experiment (skip model forward pass)")
    args = parser.parse_args()

    cfg = get_config(args.experiment)
    collect_activations(cfg, reuse_from=args.reuse_activations_from)
