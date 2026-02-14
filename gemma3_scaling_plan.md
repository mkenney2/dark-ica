# Scaling ICA Dark Matter Analysis to Gemma 3 (Gemma Scope 2)

## Why This Matters

Running on GPT-2 small with an older SAE (ReLU, 24K features) invites the objection 
"just train a bigger SAE." Gemma Scope 2 provides state-of-the-art JumpReLU SAEs with 
Matryoshka training, widths up to 1M features, trained by Google DeepMind on 110 PB of 
data. If ICA STILL finds interpretable features in the dark matter of these SAEs, the 
"just scale the SAE" objection is dead.

## Model Ladder

Run the full analysis at 3 scale points to see how dark matter changes:

| Model | d_model | SAE Width | GPU Needed | Feasibility |
|-------|---------|-----------|------------|-------------|
| Gemma 3 1B PT | 1152 | 16k / 262k | ~8GB (A100 ideal, T4 possible) | Easy |
| Gemma 3 4B PT | 2560 | 16k / 262k | ~24GB (A100) | Medium |
| Gemma 3 12B PT | 3840 | 16k / 262k | ~48GB (A100 80GB) | Hard |

Start with Gemma 3 1B — it's small enough to run on consumer hardware but large 
enough to be interesting. If results are positive, scale to 4B.

## SAE Selection

Gemma Scope 2 provides SAEs at 3 sites per layer:
- Residual stream (res) — most comparable to GPT-2 analysis
- MLP output (mlp)
- Attention output (att)

For each model, pick:
- **Layer**: Middle layer (~50% depth), same strategy as GPT-2 analysis
  - Gemma 3 1B: layer 13 (of 26)
  - Gemma 3 4B: layer 18 (of 36)
- **Width**: 262k features (large dictionary — harder to argue "just add more features")
- **L0**: "medium" (30-60 active features per token — standard operating point)
- **Site**: Residual stream (for comparability with GPT-2)

Also run on a 16k-width SAE at the same layer for comparison — if the dark matter 
shrinks dramatically from 16k→262k, that tells us something about SAE capacity vs 
inductive bias.

## HuggingFace Paths

```python
# Gemma 3 1B pretrained, residual stream SAEs
# https://huggingface.co/google/gemma-scope-2-1b-pt-res

# Load SAE weights
from huggingface_hub import hf_hub_download
import torch

# Example: layer 13, width 262k, medium L0
path = hf_hub_download(
    repo_id="google/gemma-scope-2-1b-pt-res",
    filename="layer_13/width_262k/average_l0_medium/params.npz",
    # exact path format TBD - check repo structure
)
```

Note: Exact file paths may differ. Check the HuggingFace repo and Gemma Scope 2 
Colab notebook for correct loading patterns. SAELens may also support these directly.

## Pipeline (Same as GPT-2, adapted for scale)

### Step 1: Cache Activations
```python
# Use TransformerLens or Mishax (Google's tool) to cache activations
# Target: 500K-1M tokens from a standard dataset
# For Gemma 3: use the same data distribution SAE was trained on if possible
# (check Gemma Scope 2 tech report for training data details)

# Memory consideration: 1M tokens × 1152 dims × 4 bytes = ~4.6 GB for 1B model
# Manageable, but cache in chunks if needed
```

### Step 2: Compute SAE Residual
```python
# Forward pass through SAE, compute residual
# residual = activation - sae.decode(sae.encode(activation))

# IMPORTANT: Exclude special tokens (<bos>, <eos>, padding)
# Report both with and without special tokens (as we did for GPT-2)
```

### Step 3: Variance Analysis
```python
# Compute total activation variance (content tokens only)
# Compute residual variance
# Report fraction unexplained

# KEY COMPARISON POINT:
# GPT-2 small (24k features, ReLU): 12.7% dark matter
# Gemma 3 1B (262k features, JumpReLU): ???
# Gemma 3 1B (16k features, JumpReLU): ???
# 
# If 262k JumpReLU still has substantial dark matter, that's significant.
# JumpReLU is specifically designed to reduce reconstruction error vs ReLU.
```

### Step 4: Kurtosis Check
```python
# Compute excess kurtosis of residual dimensions
# If mean kurtosis > 0 (non-Gaussian), proceed with ICA
# If ~0, the dark matter is Gaussian noise and ICA won't help
# 
# Prediction: kurtosis will still be positive but possibly lower than GPT-2
# as JumpReLU SAEs capture more of the non-Gaussian structure
```

### Step 5: ICA Decomposition
```python
# PCA whitening → FastICA
# Scale number of components with d_model:
#   Gemma 3 1B (d_model=1152): try 200 components
#   Gemma 3 4B (d_model=2560): try 300 components
# 
# Compare ICA directions to SAE decoder vectors (262k of them!)
# Cosine similarity threshold: still 0.3 for "novel"
```

### Step 6: Autointerp
```python
# Same pipeline: explain pass + detection scoring
# Use Claude Sonnet 4 for consistency
# 
# Run on:
#   - All ICA components
#   - 200 random SAE features (density-matched)
#   - 200 RANDOM DIRECTIONS (the baseline we didn't do for GPT-2!)
# 
# The random directions baseline is critical for this paper.
```

## Key Questions This Answers

1. **Does dark matter persist with state-of-the-art SAEs?**
   If Gemma Scope 2's 262k JumpReLU Matryoshka SAEs still leave substantial 
   interpretable dark matter, the problem is fundamental to the sparsity assumption, 
   not just SAE capacity or architecture.

2. **Does the kurtosis gap persist?**
   GPT-2 showed a 1,900x gap. If Gemma 3 shows a similar gap, the sparsity-vs-density 
   distinction is a universal property of how LMs represent information, not a GPT-2 quirk.

3. **Does the dark matter encode the same *kind* of features?**
   Are the ICA components still topic-level/semantic in larger models? Or does the 
   character of the dark matter change with scale? Larger models might have more 
   complex features hiding in the residual.

4. **How does dark matter fraction scale?**
   Olah reported ~35% for Claude 3 Sonnet. Our GPT-2 was 12.7%. Where does Gemma 3 1B 
   fall? Does 16k→262k width reduce it substantially, and if so, what's left?

5. **Does the random directions baseline hold?**
   If random directions in the Gemma 3 residual score 0.5 on detection while ICA 
   scores 0.7+, that's airtight evidence across model families.

## Compute Budget

| Step | Gemma 3 1B | Gemma 3 4B |
|------|-----------|-----------|
| Load model + cache 1M tokens | ~20 min, 8GB VRAM | ~30 min, 24GB VRAM |
| Compute SAE residual (262k) | ~15 min | ~30 min |
| PCA + ICA (200 components) | ~10 min | ~20 min |
| Autointerp (400 calls) | ~30 min, ~$10-15 | ~30 min, ~$10-15 |
| **Total** | **~1.5 hours, ~$10** | **~2 hours, ~$10** |

Both are very doable. The GPU time is the bottleneck, not API cost.

## Hardware Options

- **Colab Pro**: A100 40GB ($10/month) — runs 1B easily, 4B tight
- **Lambda/RunPod**: A100 80GB (~$1.50/hr) — runs everything up to 12B
- **Personal GPU**: RTX 3090/4090 24GB — runs 1B, maybe 4B with offloading

## What Makes This a Paper vs a Blog Post

Running across multiple model scales with state-of-the-art SAEs, including the random 
directions baseline, moves this from "interesting blog post" to "workshop paper at 
ICML/NeurIPS." The story becomes:

**Title**: "The Dark Matter of Sparse Autoencoders: ICA Reveals Dense Interpretable 
Features Across Model Scales"

**Contribution**: 
1. ICA on SAE residuals finds interpretable features SAEs miss (novel method)
2. The kurtosis gap quantifies WHY SAEs miss them (mechanistic explanation)
3. The effect persists across model scales and SAE architectures (generality)
4. Dark matter features occupy high-centrality positions in attribution graphs (functional role)

If results hold on Gemma 3 with Gemma Scope 2, this is a solid contribution to the 
"alternatives to SAEs" literature alongside Shafran et al. and Engels et al.

## Execution Order

1. Gemma 3 1B + 262k SAE (primary result)
2. Gemma 3 1B + 16k SAE (width comparison)
3. Random directions baseline on BOTH GPT-2 and Gemma 3 1B
4. If positive: Gemma 3 4B + 262k SAE (scaling result)
5. If still positive: consider Gemma 3 12B (stretch goal)
