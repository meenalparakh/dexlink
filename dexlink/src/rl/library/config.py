"""Validated full-method preset for native task-library references."""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from dexlink.src.rl.config.base import (
    apply_overrides_from_dict,
    apply_overrides,
    load_experiment_config,
)
from dexlink.src.rl.config.seq_oakink_presets import SEQ_BIMANUAL_REQUIRED_SETTINGS


@dataclass
class TrainingConfig:
    library: Path = Path("tasks/library40")
    instance: str = "t001_a"
    num_envs: int = 4096
    timesteps: int = 50000
    seed: int = 42
    collision_pairs: int = 256
    group: str = "library40"


def task_config(config: TrainingConfig, overrides: list[str] | None = None):
    if config.num_envs < 2 or config.timesteps <= 0 or config.timesteps % 8:
        raise ValueError(
            "Expected num_envs>=2 and positive timesteps divisible by rollout length 8."
        )
    directory = config.library.resolve() / config.instance
    row = json.loads((directory / "task.json").read_text())
    settings = {
        "env.env_class": "library_rigid",
        "env.task_cfg.demo_dataset": "random_traj",
        "env.task_cfg.oakink_exported_path": str(directory / "reference_motion.npz"),
        "env.task_cfg.metric_cfg.metric_id": None,
        "env.task_cfg.save_reference_video": False,
        "env.sim_cfg.num_envs": config.num_envs,
        "env.sim_cfg.show_viewer": False,
        "env.sim_cfg.enable_self_collision": True,
        "env.sim_cfg.max_collision_pairs": config.collision_pairs,
        "env.sim_cfg.dt": 1 / 240,
        "env.seed": config.seed,
        "env.rewards_cfg.log_essential_metrics_only": True,
        "agent.log_essential_metrics_only": True,
        "agent.timesteps": config.timesteps,
        "agent.check_overheat": False,
        "agent.resume": False,
        "agent.resume_latest_fixed_budget": False,
        "agent.eval_metrics_freq": 0,
        "agent.eval_video_freq": 0,
        "logger.project": "dexlink",
        "logger.group": config.group,
        "logger.runname": f"{config.instance}_s{config.seed}",
        "logger.timestamp": False,
        "logger.seed": False,
        "logger.disable": False,
        "logger.log_freq": 25,
        "logger.model_save_freq": 250,
    }
    if row["active_sides"] == ["left", "right"]:
        settings.update(SEQ_BIMANUAL_REQUIRED_SETTINGS)
    elif row["active_sides"] not in (["left"], ["right"]):
        raise ValueError(f"Unexpected active_sides: {row['active_sides']}")
    preset = load_experiment_config(
        "seq_oakink_residual_target_trajectory_weighted_policy_v3"
    )
    if row["active_sides"] == ["left"]:
        substitutions = {"right": "left", "ref_right": "ref_left"}
        settings["agent.policy_inputs"] = tuple(
            substitutions.get(name, name) for name in preset.agent.policy_inputs
        )
        settings["agent.critic_inputs"] = tuple(
            substitutions.get(name, name) for name in preset.agent.critic_inputs
        )
    cfg = apply_overrides_from_dict(preset, settings)
    return apply_overrides(cfg, overrides) if overrides else cfg
