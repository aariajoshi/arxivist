"""
hopann.utils.config — Configuration dataclass and YAML loader.

Provides a typed Config dataclass and a load_config() function that reads
a YAML file, validates all required fields, and raises helpful errors for
missing or invalid values.

All hyperparameters are stored here with their SIR confidence annotations.
No paths are hardcoded — all file paths are controlled via config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    hidden_dim: int = 16            # ASSUMED: Q=16 (conf 0.52); searched
    variance_type: str = "ann"      # ASSUMED: 'ann' (conf 0.62); swappable
    dropout_rate: float = 0.0       # ASSUMED: not mentioned in paper


@dataclass
class DataConfig:
    """Configuration for data loading and splitting."""
    data_path: str = ""             # Required: path to preprocessed CSV
    train_frac: float = 0.70        # ASSUMED: conf 0.52
    val_frac: float = 0.15          # ASSUMED: conf 0.52
    date_col: str = "reviewTime"    # ASSUMED: Ni et al. 2019 field name
    rating_col: str = "overall"     # ASSUMED: Ni et al. 2019 field name
    experiment_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4])


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""
    batch_size: int = 64            # ASSUMED: conf 0.52
    lr: float = 1e-3                # ASSUMED: conf 0.52
    max_epochs: int = 200           # ASSUMED: reasonable upper bound
    patience: int = 15              # ASSUMED: conf 0.52
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 10
    monitor_metric: str = "val_nll"
    # Grid search settings
    search_hidden_dims: list[int] = field(default_factory=lambda: [8, 16, 32, 64])
    search_lrs: list[float] = field(default_factory=lambda: [1e-2, 1e-3, 1e-4])
    search_max_epochs: int = 50     # Epochs per config during search
    run_hyperparameter_search: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    metrics: list[str] = field(default_factory=lambda: [
        "f1_macro", "accuracy", "pr_auc", "mse", "mae", "cohen_kappa", "roc_auc"
    ])
    output_dir: str = "outputs"


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility."""
    seed: int = 42
    deterministic: bool = False     # Set True for full determinism (may slow training)
    cudnn_benchmark: bool = True    # Set False when deterministic=True


@dataclass
class Config:
    """
    Top-level configuration object for the HOPANN training pipeline.

    All paths and hyperparameters are configurable here. Nothing is hardcoded
    in the rest of the codebase.

    Fields correspond to config.yaml sections.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    device: str = "auto"            # 'auto' = CUDA if available, else CPU
    run_name: str = "hopann_run"

    def resolve_device(self) -> str:
        """Return the actual device string, resolving 'auto'."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def validate(self) -> None:
        """
        Validate config values and raise helpful errors for invalid settings.

        Raises:
            ValueError: For missing or invalid fields.
        """
        # Data path must be set
        if not self.data.data_path:
            raise ValueError(
                "Config error: data.data_path is not set. "
                "Provide the path to the preprocessed Amazon review CSV."
            )

        # Train + val fractions must sum < 1
        total = self.data.train_frac + self.data.val_frac
        if total >= 1.0:
            raise ValueError(
                f"Config error: data.train_frac + data.val_frac = {total:.3f} >= 1.0. "
                "Must leave room for a test set."
            )

        # Batch size must be positive
        if self.training.batch_size < 1:
            raise ValueError(
                f"Config error: training.batch_size={self.training.batch_size} must be >= 1."
            )

        # Patience must be positive
        if self.training.patience < 1:
            raise ValueError(
                f"Config error: training.patience={self.training.patience} must be >= 1."
            )

        # Variance type must be known
        from hopann.models.variance_network import VARIANCE_NETWORK_REGISTRY
        if self.model.variance_type not in VARIANCE_NETWORK_REGISTRY:
            raise ValueError(
                f"Config error: model.variance_type='{self.model.variance_type}' "
                f"is not registered. Choose from: {list(VARIANCE_NETWORK_REGISTRY.keys())}"
            )

        # Experiment IDs must be in {1, 2, 3, 4}
        for eid in self.data.experiment_ids:
            if eid not in {1, 2, 3, 4}:
                raise ValueError(
                    f"Config error: experiment_id={eid} not in {{1, 2, 3, 4}}."
                )

        # Warn about deterministic + cudnn_benchmark conflict
        if self.reproducibility.deterministic and self.reproducibility.cudnn_benchmark:
            import warnings
            warnings.warn(
                "Config: deterministic=True and cudnn_benchmark=True may conflict. "
                "Consider setting cudnn_benchmark=False for full reproducibility.",
                UserWarning,
                stacklevel=2,
            )

    def __repr__(self) -> str:
        return (
            f"Config(run_name='{self.run_name}', device='{self.device}', "
            f"seed={self.reproducibility.seed})"
        )


def load_config(config_path: str, overrides: dict[str, Any] | None = None) -> Config:
    """
    Load configuration from a YAML file, with optional runtime overrides.

    Args:
        config_path: Path to the YAML configuration file.
        overrides:   Dict of dot-notation key→value overrides applied after
                     loading YAML (e.g. {"training.lr": 5e-4}).

    Returns:
        Validated Config object.

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValueError:        If YAML structure is invalid or required fields missing.

    Example:
        cfg = load_config("configs/config.yaml", {"training.lr": 5e-4})
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required: pip install pyyaml") from exc

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Apply overrides (dot-notation: "training.lr" → raw["training"]["lr"])
    if overrides:
        for key, val in overrides.items():
            parts = key.split(".")
            d = raw
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = val

    # Build typed config
    cfg = Config(
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        training=TrainingConfig(**raw.get("training", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        reproducibility=ReproducibilityConfig(**raw.get("reproducibility", {})),
        device=raw.get("device", "auto"),
        run_name=raw.get("run_name", "hopann_run"),
    )
    cfg.validate()
    return cfg
