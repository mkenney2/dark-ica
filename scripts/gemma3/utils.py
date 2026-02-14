"""Shared utilities for ICA dark matter analysis."""

import os
import json
import numpy as np
import torch
from typing import Optional

# Point HuggingFace cache to the volume so it persists and doesn't fill container disk
if os.path.isdir("/workspace") and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

from config import ExperimentConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def print_device_info():
    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def load_model(cfg: ExperimentConfig):
    """Load a HookedTransformer model."""
    try:
        from sae_lens import HookedSAETransformer
        model = HookedSAETransformer.from_pretrained(cfg.model_name, device=DEVICE)
        print(f"Loaded {cfg.model_name} via HookedSAETransformer")
    except (ImportError, Exception) as e:
        print(f"HookedSAETransformer not available ({e}), falling back to HookedTransformer")
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained(cfg.model_name, device=DEVICE)
        print(f"Loaded {cfg.model_name} via HookedTransformer")
    return model


def load_sae(cfg: ExperimentConfig):
    """Load an SAE, handling sae-lens v6+ return format."""
    from sae_lens import SAE
    result = SAE.from_pretrained(
        release=cfg.sae_release,
        sae_id=cfg.sae_id,
        device=DEVICE,
    )
    sae = result[0] if isinstance(result, tuple) else result
    print(f"SAE loaded: {cfg.sae_release} / {cfg.sae_id}")
    print(f"  W_dec shape: {sae.W_dec.shape}")
    return sae


def get_exclude_mask(token_ids: torch.Tensor, exclude_ids: list) -> torch.Tensor:
    """Return a boolean mask where True = keep (token NOT in exclude list)."""
    mask = torch.ones(token_ids.shape, dtype=torch.bool, device=token_ids.device)
    for tid in exclude_ids:
        mask &= (token_ids != tid)
    return mask


def get_context(idx: int, tokens: np.ndarray, tokenizer,
                ctx_before: int = 50, ctx_after: int = 20) -> str:
    """Get surrounding context for a token at position idx, with [brackets] around target."""
    start = max(0, idx - ctx_before)
    end = min(len(tokens), idx + ctx_after + 1)

    before_ids = tokens[start:idx].tolist()
    target_id = int(tokens[idx])
    after_ids = tokens[idx + 1:end].tolist()

    before_text = tokenizer.decode(before_ids)
    target_text = tokenizer.decode([target_id])
    after_text = tokenizer.decode(after_ids)

    return f"{before_text}[{target_text}]{after_text}", target_text


def save_results_json(data: dict, path: str):
    """Save a dict as JSON with indent=2."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {path}")


def load_results_json(path: str) -> dict:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
