"""Experiment configuration for ICA dark matter analysis."""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class ExperimentConfig:
    # Experiment identity
    name: str

    # Model
    model_name: str
    d_model: int
    n_layers: int
    layer: int
    hook_name: str

    # SAE
    sae_release: str
    sae_id: str
    sae_width: int

    # Tokenizer
    exclude_token_ids: List[int] = field(default_factory=list)
    auto_prepends_bos: bool = False

    # Data collection
    target_tokens: int = 1_000_000
    context_len: int = 128
    batch_size: int = 32
    sae_chunk_size: int = 5_000

    # ICA
    n_ica_components: int = 200
    ica_max_iter: int = 1000
    ica_tol: float = 1e-4
    pca_max_components: int = 300
    robustness_seeds: List[int] = field(default_factory=lambda: [42, 123, 999])

    # Random baseline
    n_random_directions: int = 200

    # Autointerp
    autointerp_model: str = "claude-sonnet-4-20250514"
    autointerp_top_k: int = 40
    autointerp_n_explain: int = 10
    autointerp_n_negatives: int = 10
    autointerp_context_before: int = 50
    autointerp_context_after: int = 20
    n_baseline_sae_features: int = 200
    autointerp_rate_limit: float = 0.5

    # Paths
    random_seed: int = 42

    @property
    def experiment_dir(self) -> str:
        return os.path.join("experiments", self.name)

    @property
    def plots_dir(self) -> str:
        return os.path.join(self.experiment_dir, "plots")

    @property
    def autointerp_dir(self) -> str:
        return os.path.join(self.experiment_dir, "autointerp")

    def ensure_dirs(self):
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.autointerp_dir, exist_ok=True)

    def path(self, filename: str) -> str:
        return os.path.join(self.experiment_dir, filename)

    def plot_path(self, filename: str) -> str:
        return os.path.join(self.plots_dir, filename)

    def autointerp_path(self, filename: str) -> str:
        return os.path.join(self.autointerp_dir, filename)


# --- Preset Configurations ---

CONFIGS = {
    "gemma3_1b_262k": ExperimentConfig(
        name="gemma3_1b_262k",
        model_name="google/gemma-3-1b-pt",
        d_model=1152,
        n_layers=26,
        layer=13,
        hook_name="blocks.13.hook_resid_post",
        sae_release="gemma-scope-2-1b-pt-res",
        sae_id="layer_13_width_262k_l0_medium",
        sae_width=262144,
        exclude_token_ids=[0, 1, 2],  # pad, eos, bos
        auto_prepends_bos=True,
        n_ica_components=200,
        pca_max_components=300,
        sae_chunk_size=5_000,
    ),
    "gemma3_1b_16k": ExperimentConfig(
        name="gemma3_1b_16k",
        model_name="google/gemma-3-1b-pt",
        d_model=1152,
        n_layers=26,
        layer=13,
        hook_name="blocks.13.hook_resid_post",
        sae_release="gemma-scope-2-1b-pt-res",
        sae_id="layer_13_width_16k_l0_medium",
        sae_width=16384,
        exclude_token_ids=[0, 1, 2],
        auto_prepends_bos=True,
        n_ica_components=200,
        pca_max_components=300,
        sae_chunk_size=50_000,
    ),
    "gpt2_random_baseline": ExperimentConfig(
        name="gpt2_random_baseline",
        model_name="gpt2",
        d_model=768,
        n_layers=12,
        layer=6,
        hook_name="blocks.6.hook_resid_pre",
        sae_release="gpt2-small-res-jb",
        sae_id="blocks.6.hook_resid_pre",
        sae_width=24576,
        exclude_token_ids=[50256],  # bos/eos/pad all same token
        auto_prepends_bos=False,
        n_ica_components=100,
        pca_max_components=100,
        sae_chunk_size=50_000,
    ),
}


def get_config(experiment_name: str) -> ExperimentConfig:
    if experiment_name not in CONFIGS:
        raise ValueError(
            f"Unknown experiment '{experiment_name}'. "
            f"Available: {list(CONFIGS.keys())}"
        )
    return CONFIGS[experiment_name]
