"""Typed model and layout records shared by multi-task policy modules."""

from __future__ import annotations
from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass
class ModelConfig:
    hidden_dim: int = 96
    initial_log_std: float = -2.0
    window_sigma: float = 2.0


@dataclass
class GeometryPolicyConfig:
    point_count: int = 512
    geometry_hidden_dims: tuple[int, ...] = (64, 128, 128)
    geometry_embed_dim: int = 128
    program_embed_dim: int = 128
    program_layers: int = 2
    program_heads: int = 4
    max_program_phases: int = 12
    hidden_dim: int = 256
    initial_log_std: float = -2.0
    window_sigma: float = 2.0


@dataclass
class EnvConfig:
    num_envs_per_variant: int = 2
    device: str = "cuda:0"


@dataclass
class PPOConfig:
    control_steps: int = 32
    rollout_steps: int = 8
    epochs: int = 2
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.001
    max_grad_norm: float = 1.0


@dataclass
class PilotConfig:
    variant_manifest: str = ""
    run_dir: str = ""
    persistent_root: str = str(Path("outputs/multitask").resolve())
    seed: int = 42
    wandb_entity: str = os.environ.get("WANDB_ENTITY", "")
    wandb_project: str = "dexlink"
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
