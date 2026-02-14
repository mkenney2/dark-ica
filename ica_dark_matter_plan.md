# ICA Analysis of SAE Residuals: Implementation Plan

## Goal
Decompose the "dark matter" of neural network activations — the ~35% reconstruction error left over after SAE decomposition — using Independent Component Analysis (ICA). Determine whether the SAE residual contains interpretable structure that SAEs miss.

## Hardware
- Local machine with 8GB NVIDIA GPU (sufficient for GPT-2 small)
- If needed: cheap RunPod instance for larger models

## Model & SAE Selection

### Start with: GPT-2 small
- **Why**: 768-dim residual stream, fits easily in 8GB VRAM, extensive SAE ecosystem
- **SAE source**: Use SAELens to load pre-trained SAEs from Neuronpedia
- **Specific SAE**: `gpt2-small-res-jb` from Joseph Bloom's SAELens release
  - These are residual stream SAEs trained at multiple layers
  - Start with **layer 6** (middle of network — neither too early/syntactic nor too late/output-y)
  - The SAE at this layer has expansion factor 32x, so ~24,576 features

### Dataset
- Use `Skylion007/openwebtext` from HuggingFace (same distribution the SAEs were trained on)
- Collect activations on **1M tokens** (enough for robust ICA, runs in minutes)
- Use context length 128 for speed

## Implementation Steps

### Step 1: Environment Setup

```
pip install sae-lens transformer-lens torch scikit-learn datasets matplotlib numpy pandas
```

### Step 2: Load Model and SAE

```python
import torch
from sae_lens import SAE, HookedSAETransformer

# Load GPT-2 small with TransformerLens hooks
model = HookedSAETransformer.from_pretrained("gpt2", device="cuda")

# Load the pre-trained SAE for layer 6 residual stream
# Check SAELens docs for exact release ID — it should be something like:
sae_id = "blocks.6.hook_resid_post"
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id=sae_id,
    device="cuda"
)
```

**Important**: The exact release name and sae_id may have changed. Check:
- https://github.com/jbloomAus/SAELens
- https://www.neuronpedia.org/gpt2-small
- Run `SAE.from_pretrained?` to see available releases

If `HookedSAETransformer` doesn't exist or the API has changed, fall back to:
```python
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2", device="cuda")
```
And load the SAE separately. The key thing is getting the SAE's encode/decode methods.

### Step 3: Collect Activations and Compute Residuals

```python
from datasets import load_dataset
import numpy as np

dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

# Collect activations
all_residuals = []
all_activations = []
tokens_collected = 0
target_tokens = 1_000_000
batch_size = 32
context_len = 128

# Tokenize and batch
for batch in dataset:
    tokens = model.to_tokens(batch["text"], prepend_bos=True)
    # Truncate/pad to context_len
    tokens = tokens[:, :context_len]
    
    if tokens.shape[0] < 1:
        continue
    
    with torch.no_grad():
        # Run model, cache activations at the SAE hook point
        _, cache = model.run_with_cache(
            tokens,
            names_filter=["blocks.6.hook_resid_post"]
        )
        
        # Get activations: shape (batch, seq_len, d_model)
        acts = cache["blocks.6.hook_resid_post"]
        
        # Reshape to (batch * seq_len, d_model)
        acts_flat = acts.reshape(-1, acts.shape[-1])
        
        # Run through SAE
        feature_acts = sae.encode(acts_flat)
        reconstructed = sae.decode(feature_acts)
        
        # Compute residual
        residual = acts_flat - reconstructed
        
        all_residuals.append(residual.cpu().numpy())
        all_activations.append(acts_flat.cpu().numpy())
    
    tokens_collected += tokens.numel()
    if tokens_collected >= target_tokens:
        break

residuals = np.concatenate(all_residuals, axis=0)  # shape: (N, 768)
activations = np.concatenate(all_activations, axis=0)
print(f"Collected {residuals.shape[0]} token activations")
print(f"Residual shape: {residuals.shape}")
```

**Note**: The exact API for running the SAE may differ. Some SAEs use:
- `sae.encode(x)` returns feature activations
- `sae.decode(feature_acts)` returns reconstruction
- Or `sae(x)` returns a tuple/dict with reconstruction and feature activations

Check the SAE object's methods. The key equation is:
```
residual = original_activation - sae_reconstruction
```

**Memory management**: If 1M tokens * 768 dims doesn't fit in RAM as float32 (~3GB), either:
- Collect in chunks and concatenate
- Use float16
- Reduce to 500k tokens

### Step 4: Basic Diagnostics on the Residual

Before running ICA, characterize what we're working with.

```python
# 4a. Variance explained by SAE
total_var = np.var(activations, axis=0).sum()
residual_var = np.var(residuals, axis=0).sum()
frac_unexplained = residual_var / total_var
print(f"Fraction of variance unexplained by SAE: {frac_unexplained:.3f}")
# Expect something like 0.15-0.35

# 4b. Is the residual Gaussian?
# If yes, ICA won't find much (ICA relies on non-Gaussianity)
from scipy.stats import kurtosis, shapiro
kurt = kurtosis(residuals, axis=0)  # Per-dimension kurtosis
print(f"Mean kurtosis across dimensions: {kurt.mean():.3f}")
print(f"Std kurtosis: {kurt.std():.3f}")
# Gaussian has kurtosis = 0 (excess). Values >> 0 mean heavy tails = good for ICA.
# Values near 0 mean the residual is approximately Gaussian = ICA won't help much.

# 4c. Plot kurtosis distribution
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.hist(kurt, bins=50)
plt.xlabel("Excess Kurtosis")
plt.ylabel("Count (dimensions)")
plt.title("Kurtosis of SAE Residual Across Dimensions")
plt.axvline(x=0, color='r', linestyle='--', label='Gaussian')
plt.legend()
plt.savefig("residual_kurtosis.png", dpi=150, bbox_inches='tight')
plt.show()

# 4d. PCA on residuals — how many dimensions carry signal?
from sklearn.decomposition import PCA
pca = PCA(n_components=100)
pca.fit(residuals)
cumvar = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 5))
plt.plot(cumvar)
plt.xlabel("Number of PCA components")
plt.ylabel("Cumulative variance explained")
plt.title("PCA of SAE Residual")
plt.savefig("residual_pca.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Components needed for 90% of residual variance: {np.searchsorted(cumvar, 0.9) + 1}")
```

**Decision point after Step 4:**
- If kurtosis is near zero across most dimensions → residual is ~Gaussian → ICA won't find structure → this is still a result! It means SAEs are capturing most of the non-Gaussian (i.e., meaningful) structure and the dark matter is noise.
- If kurtosis is substantially positive → there's non-Gaussian structure → ICA should find components → proceed to Step 5.

### Step 5: Run ICA on the Residual

```python
from sklearn.decomposition import FastICA

# First reduce dimensionality with PCA to top-k components
# This speeds up ICA and removes noise dimensions
n_components = 100  # Adjust based on Step 4 PCA plot
# If 90% of residual variance is in <50 dims, use fewer

# Whiten first (PCA), then run ICA
pca_pre = PCA(n_components=n_components, whiten=True)
residuals_whitened = pca_pre.fit_transform(residuals)

ica = FastICA(
    n_components=n_components,
    algorithm='parallel',
    whiten=False,  # Already whitened
    max_iter=1000,
    tol=1e-4,
    random_state=42
)
ica_sources = ica.fit_transform(residuals_whitened)  # shape: (N, n_components)

# The ICA components in the original activation space:
# ica.mixing_ gives the mixing matrix in whitened space
# To get back to original 768-dim space:
ica_directions = ica.mixing_.T @ pca_pre.components_  # shape: (n_components, 768)

# Normalize each direction
ica_directions = ica_directions / np.linalg.norm(ica_directions, axis=1, keepdims=True)

print(f"ICA converged: {ica.n_iter_} iterations")
print(f"ICA directions shape: {ica_directions.shape}")
```

### Step 6: Compare ICA Directions to SAE Decoder Directions

This is the key comparison. Do ICA components of the residual look like SAE features?

```python
# Get SAE decoder weights: shape (n_features, d_model)
sae_decoder = sae.W_dec.detach().cpu().numpy()  # Check actual attribute name
# Might be sae.W_dec, sae.decoder.weight, etc.
# Normalize
sae_decoder_normed = sae_decoder / np.linalg.norm(sae_decoder, axis=1, keepdims=True)

# Compute cosine similarity between each ICA direction and all SAE features
# shape: (n_ica_components, n_sae_features)
cos_sim = ica_directions @ sae_decoder_normed.T

# For each ICA component, find the most similar SAE feature
max_cos_sim = np.max(np.abs(cos_sim), axis=1)
best_sae_match = np.argmax(np.abs(cos_sim), axis=1)

plt.figure(figsize=(10, 5))
plt.hist(max_cos_sim, bins=50)
plt.xlabel("Max cosine similarity with any SAE feature")
plt.ylabel("Count (ICA components)")
plt.title("ICA Component Similarity to SAE Features")
plt.savefig("ica_sae_similarity.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Mean max cosine sim: {max_cos_sim.mean():.3f}")
print(f"ICA components with >0.8 cosine sim to an SAE feature: {(max_cos_sim > 0.8).sum()}")
print(f"ICA components with <0.3 cosine sim to any SAE feature: {(max_cos_sim < 0.3).sum()}")
```

**Interpretation:**
- **High similarity (>0.8)**: ICA found the same directions as SAEs, just in the residual. This means the SAE knows about these features but doesn't reconstruct them well — possibly a training issue.
- **Medium similarity (0.3-0.8)**: Partially overlapping. Could be rotated/mixed versions of SAE features.
- **Low similarity (<0.3)**: ICA found genuinely new directions the SAE missed entirely. These are the most interesting — they might be the "missing features" from Olah's five hurdles.

### Step 7: Interpret the ICA Components

For each ICA component, find the tokens that activate it most strongly and look for patterns.

```python
# Project all original activations onto ICA directions to get "ICA activations"
ica_activations = activations @ ica_directions.T  # shape: (N, n_components)

# We need the tokens to look up what they are
# Collect tokens alongside activations in Step 3 — add this:
# all_tokens.append(tokens.reshape(-1).cpu().numpy())
# tokens_flat = np.concatenate(all_tokens, axis=0)

# For each ICA component, find top activating token positions
def get_top_activating_examples(component_idx, k=20):
    acts = ica_activations[:, component_idx]
    top_indices = np.argsort(np.abs(acts))[-k:][::-1]
    
    results = []
    for idx in top_indices:
        token_id = tokens_flat[idx]
        token_str = model.to_string([token_id])
        # Get surrounding context (±5 tokens)
        start = max(0, idx - 5)
        end = min(len(tokens_flat), idx + 6)
        context = model.to_string(tokens_flat[start:end])
        results.append({
            'token': token_str,
            'context': context,
            'activation': acts[idx],
            'position': idx
        })
    return results

# Print top activations for the first 10 ICA components
for comp in range(min(10, n_components)):
    cos = max_cos_sim[comp]
    examples = get_top_activating_examples(comp)
    print(f"\n{'='*60}")
    print(f"ICA Component {comp} | Max SAE cosine sim: {cos:.3f}")
    print(f"{'='*60}")
    for ex in examples[:10]:
        print(f"  [{ex['activation']:+.3f}] ...{ex['context']}...")
```

**What to look for:**
- Components with low SAE similarity but clear interpretable patterns → these are dark matter features
- Components where top tokens share a clear theme (same topic, same syntactic role, same position-in-sequence) → structure the SAE missed
- Components that look random → noise dimensions, expected for some

### Step 8: Statistical Comparison

Quantify whether ICA components are more or less interpretable than SAE features.

```python
# 8a. Sparsity comparison
# Compute kurtosis of each ICA component's activations
ica_kurtosis = kurtosis(ica_activations, axis=0)
# Compare to SAE feature activations
with torch.no_grad():
    all_sae_acts = []
    # Re-run SAE encoding on stored activations
    for chunk in np.array_split(activations, 100):
        sae_acts = sae.encode(torch.tensor(chunk, device="cuda", dtype=torch.float32))
        all_sae_acts.append(sae_acts.cpu().numpy())
    sae_activations = np.concatenate(all_sae_acts, axis=0)

# Only look at SAE features that actually fire
active_sae_mask = (sae_activations > 0).mean(axis=0) > 0.001  # fire on >0.1% of tokens
sae_kurtosis = kurtosis(sae_activations[:, active_sae_mask], axis=0)

plt.figure(figsize=(10, 5))
plt.hist(ica_kurtosis, bins=50, alpha=0.5, label='ICA components', density=True)
plt.hist(sae_kurtosis, bins=50, alpha=0.5, label='SAE features (active)', density=True)
plt.xlabel("Excess Kurtosis")
plt.ylabel("Density")
plt.title("Sparsity of ICA Components vs SAE Features")
plt.legend()
plt.savefig("ica_vs_sae_kurtosis.png", dpi=150, bbox_inches='tight')
plt.show()

# 8b. How much of the residual does ICA explain?
ica_reconstruction = ica_sources @ ica.mixing_.T @ pca_pre.components_
ica_recon_err = np.var(residuals - ica_reconstruction, axis=0).sum()
print(f"ICA explains {1 - ica_recon_err/residual_var:.1%} of residual variance")
# (This should be ~100% if n_components is high enough, since ICA is a linear transform)
# The question is whether the components are INTERPRETABLE, not whether they explain variance
```

### Step 9: Save Everything and Generate Report

```python
import json

results = {
    "model": "gpt2-small",
    "sae": "gpt2-small-res-jb, layer 6",
    "tokens_analyzed": int(residuals.shape[0]),
    "fraction_variance_unexplained_by_sae": float(frac_unexplained),
    "residual_mean_kurtosis": float(kurt.mean()),
    "n_ica_components": n_components,
    "ica_sae_similarity": {
        "mean_max_cosine": float(max_cos_sim.mean()),
        "high_similarity_count": int((max_cos_sim > 0.8).sum()),
        "low_similarity_count": int((max_cos_sim < 0.3).sum()),
    },
    "ica_component_kurtosis_mean": float(ica_kurtosis.mean()),
    "sae_feature_kurtosis_mean": float(sae_kurtosis.mean()),
}

with open("ica_dark_matter_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save the ICA directions for further analysis
np.save("ica_directions.npy", ica_directions)
np.save("ica_activations.npy", ica_activations)
np.save("residuals.npy", residuals)
```

## Expected Runtime

| Step | Time (8GB GPU) |
|------|----------------|
| Load model + SAE | ~30 seconds |
| Collect 1M token activations | ~10-20 minutes |
| Diagnostics (PCA, kurtosis) | ~1 minute |
| FastICA (100 components, 1M samples) | ~2-5 minutes |
| SAE comparison | ~1 minute |
| Interpretation (manual) | ~30 minutes |
| **Total** | **~1 hour including interpretation** |

## Key Gotchas for Claude Code

1. **SAELens API changes frequently.** If the exact loading code doesn't work, check `pip show sae-lens` for version and consult the current docs. The core concept is the same — load model, load SAE, get encode/decode methods.

2. **Token collection needs care.** You need to keep track of which tokens correspond to which activation vectors so you can interpret the ICA components. Store the token IDs alongside activations.

3. **Memory.** 1M tokens × 768 dims × float32 = ~3GB for activations alone. Plus residuals = 6GB. Plus SAE features (~1M × 24k × float32 is too big). Don't store all SAE feature activations at once for the kurtosis comparison — sample or compute in chunks.

4. **The SAE decoder weight matrix might be transposed.** Check the shape — it should be (n_features, d_model) or (d_model, n_features). Normalize along the d_model axis.

5. **ICA can be sensitive to initialization.** Run with 3 different random seeds and check that the top components are consistent (high cosine similarity across runs). If not, the components aren't robust.

6. **The PCA pre-whitening step matters.** Don't skip it. ICA on high-dimensional data without whitening is slow and unstable. The number of PCA components to keep is a judgment call — start with 100, but if the PCA plot from Step 4 shows 90% variance in 30 components, use fewer.

## What to Do with Results

### If kurtosis of residual is ~0 (Gaussian):
→ Write up: "SAEs capture essentially all non-Gaussian structure. The dark matter is noise."
→ This is still publishable and interesting. It validates SAEs.

### If ICA finds interpretable components with low SAE similarity:
→ This is the exciting outcome. You've found "dark matter features."
→ Write up with examples, compare to SAE features, discuss why SAEs miss them.
→ Potential follow-up: are these features causally important? (Ablate ICA directions, measure output change)

### If ICA components are high-similarity with SAE features:
→ The SAE knows about these directions but doesn't reconstruct them well.
→ This points to SAE training issues (capacity, sparsity penalty) rather than fundamental limitations.
→ Still useful — suggests the dark matter is reducible with better SAE training, not new methods.

## Stretch Goals (if initial results are interesting)

1. **Run across multiple layers** (0, 3, 6, 9, 11) — does the dark matter structure change with depth?
2. **Compare ICA to RICA (Reconstruction ICA)** — a variant that's better for overcomplete representations
3. **Run on Gemma-2-2B** with Anthropic's SAEs for direct comparison to their published dark matter numbers
4. **Causal validation** — for the most interesting ICA components, ablate that direction and measure effect on model output. If ablation changes behavior, the dark matter feature is causally relevant.
5. **Autointerp** — feed the top-activating examples for ICA components to Claude and ask for descriptions, same way people interpret SAE features.
