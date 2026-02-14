# ICA Dark Matter Project

## Overview
Decompose the "dark matter" of neural network activations — the ~35% reconstruction error left over after SAE (Sparse Autoencoder) decomposition — using Independent Component Analysis (ICA). Determine whether the SAE residual contains interpretable structure that SAEs miss.

## Tech Stack
- **Language**: Python 3
- **ML Framework**: PyTorch (CUDA)
- **Key Libraries**: sae-lens, transformer-lens, scikit-learn (FastICA, PCA), datasets, matplotlib, numpy, pandas, scipy
- **Model**: GPT-2 small (768-dim residual stream)
- **SAE**: `gpt2-small-res-jb` from Joseph Bloom's SAELens release (layer 6, ~24,576 features)
- **Dataset**: `Skylion007/openwebtext` (1M tokens, context length 128)

## Hardware Constraints
- Local machine with 8GB NVIDIA GPU
- Memory budget: ~6GB for activations + residuals (1M tokens x 768 dims x float32)
- SAE feature activations must be computed in chunks (too large to store all at once)

## Project Structure
```
ica-dark-matter-project/
├── CLAUDE.md                          # This file
├── ica_dark_matter_plan.md            # Original research plan
├── scripts/
│   ├── 01_setup_and_load.py           # Environment verification, model + SAE loading
│   ├── 02_collect_activations.py      # Collect activations and compute residuals
│   ├── 03_diagnostics.py              # Kurtosis, PCA, Gaussianity checks
│   ├── 04_run_ica.py                  # FastICA on residuals
│   ├── 05_compare_sae.py              # Compare ICA directions to SAE decoder
│   ├── 06_interpret.py                # Top activating examples per ICA component
│   ├── 07_statistics.py               # Sparsity comparison, variance explained
│   └── 08_save_report.py              # Save results JSON + numpy arrays
├── outputs/                           # Generated plots, numpy arrays, JSON results
│   ├── residual_kurtosis.png
│   ├── residual_pca.png
│   ├── ica_sae_similarity.png
│   ├── ica_vs_sae_kurtosis.png
│   ├── ica_directions.npy
│   ├── ica_activations.npy
│   ├── residuals.npy
│   └── ica_dark_matter_results.json
└── notebooks/                         # Optional: Jupyter notebooks for exploration
```

## Key API Notes
- SAELens API changes frequently — verify loading code against current version
- The SAE decoder weight matrix might be transposed — check shape is (n_features, d_model)
- `HookedSAETransformer` may not exist; fall back to `transformer_lens.HookedTransformer`
- SAE encode/decode methods may vary: check for `sae.encode()`, `sae.decode()`, or `sae()` returning a dict

## Key Decision Points
1. **After diagnostics (Step 3)**: If residual kurtosis ~0, the dark matter is Gaussian noise — ICA won't find structure (still a valid result)
2. **PCA component count**: Use PCA explained variance plot to choose n_components for ICA (start with 100, reduce if 90% variance captured in fewer)
3. **ICA robustness**: Run with 3 random seeds; if components aren't consistent, they aren't robust

## Conventions
- Save all plots to `outputs/` directory at 150 DPI
- Use `random_state=42` as default seed
- Process SAE activations in chunks to avoid OOM
- Store token IDs alongside activations for interpretation
- All scripts should be runnable independently but share data via saved numpy files in `outputs/`
