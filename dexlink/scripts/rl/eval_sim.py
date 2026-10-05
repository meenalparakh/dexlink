#!/usr/bin/env python
"""Run a trained DexLink policy continuously in simulation.

Examples:
    python -m dexlink.scripts.rl.eval_sim --rl_run_dir outputs/example_run --ckpt_timestep 50000
    python -m dexlink.scripts.rl.eval_sim --rl_run_dir outputs/example_run --ckpt_timestep 50000 --sample_actions

Override syntax:
    --env.<nested_field> VALUE
    --agent.<field> VALUE
    --logger.<field> VALUE
    --eval.<field> VALUE
    --real_eval.<field> VALUE
Supports bool casting: pass True/False or 1/0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import setuptools  # noqa: F401

os.environ["WANDB_MODE"] = "offline"


def _should_use_egl(script_path: str) -> bool:
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return not has_display


file_path = os.path.abspath(__file__)
if _should_use_egl(file_path):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

from dexlink.src.rl.core_agent.wandb_logger import WandbLogger
from dexlink.utils.retargeting.path_utils import (
    get_logs_dir,
    get_src,
)
from dexlink.src.rl.agent.ppo_eval import PPOEval
from dexlink.src.rl.config.base import (
    PRESETS,
    apply_overrides,
    apply_overrides_from_dict,
    extract_overrides,
    load_experiment_config_from_dict,
)
from dexlink.src.rl.multi_object.env_factory import create_env


def _resolve_checkpoint_path(checkpoint_path: str, logger_logdir: str) -> Path:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(logger_logdir) / ckpt_path
    if ckpt_path.name != "policy.pt":
        raise ValueError(
            "agent.resume_from_ckpt must point to a checkpoint file ending in 'policy.pt', "
            f"got {ckpt_path}."
        )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def _resolve_policy_checkpoint(
    *,
    rl_run_dir: str | None,
    ckpt_timestep: int | None,
    resume_from_ckpt: str | None,
    logger_logdir: str,
) -> Path:
    if rl_run_dir is not None:
        if ckpt_timestep is None:
            raise ValueError(
                "--ckpt_timestep is required when --rl_run_dir is provided."
            )
        if ckpt_timestep <= 0:
            raise ValueError(
                "Evaluation requires a strictly positive checkpoint timestep; "
                f"got {ckpt_timestep}. Timestep 0 is an untrained policy."
            )
        policy_ckpt = (
            Path(rl_run_dir) / "models" / str(int(ckpt_timestep)) / "policy.pt"
        )
        if not policy_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {policy_ckpt}")
        return policy_ckpt

    if resume_from_ckpt is None:
        raise ValueError(
            "agent.resume_from_ckpt must be set when --rl_run_dir is not provided."
        )
    return _resolve_checkpoint_path(resume_from_ckpt, logger_logdir)


def _infer_run_dir_from_policy_checkpoint(policy_ckpt: Path) -> Path:
    ckpt_dir = policy_ckpt.parent
    if ckpt_dir.parent.name != "models":
        raise ValueError(
            "Expected checkpoint layout '<run_dir>/models/<timestep>/policy.pt', "
            f"got {policy_ckpt}."
        )
    try:
        timestep = int(ckpt_dir.name)
    except ValueError as error:
        raise ValueError(
            "Expected the checkpoint parent directory to be an integer timestep, "
            f"got {ckpt_dir.name!r}."
        ) from error
    if timestep <= 0:
        raise ValueError(
            "Evaluation requires a strictly positive checkpoint timestep; "
            f"got {timestep}. Timestep 0 is an untrained policy."
        )
    return ckpt_dir.parent.parent


def _load_training_run_config(run_dir: Path):
    exp_cfg_path = run_dir / "exp_cfg.json"
    if not exp_cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {exp_cfg_path}")
    with exp_cfg_path.open("r") as f:
        cfg_dict = json.load(f)
    return load_experiment_config_from_dict(cfg_dict)


def _resolve_runtime_path(path_str: str | None) -> str | None:
    """Resolve a stored project-relative path without guessing old layouts."""

    if path_str is None:
        return None
    path = Path(path_str)
    return str(path if path.is_absolute() else get_src() / path)


def convert_to_local_paths(exp_cfg) -> None:
    """Resolve paths from a portable training configuration for this checkout."""

    exp_cfg.logger.logdir = str(get_logs_dir())
    exp_cfg.eval.logdir = str(get_logs_dir() / "eval")
    exp_cfg.real_eval.logdir = str(get_logs_dir() / "real_eval")

    task_cfg = exp_cfg.env.task_cfg
    task_cfg.oakink_exported_path = _resolve_runtime_path(task_cfg.oakink_exported_path)
    task_cfg.original_demo_path = _resolve_runtime_path(task_cfg.original_demo_path)
    task_cfg.reference_object_urdf = _resolve_runtime_path(
        task_cfg.reference_object_urdf
    )
    task_cfg.target_object_urdf = _resolve_runtime_path(task_cfg.target_object_urdf)
    task_cfg.retargeted_results = _resolve_runtime_path(task_cfg.retargeted_results)

    exp_cfg.env.sim_cfg.video_save_dir = "videos"
    exp_cfg.env.real_cfg.video_save_dir = "real_videos"
    exp_cfg.env.real_cfg.world_calibration_cache_path = _resolve_runtime_path(
        exp_cfg.env.real_cfg.world_calibration_cache_path
    )
    exp_cfg.env.real_cfg.robot_control_config_path = _resolve_runtime_path(
        exp_cfg.env.real_cfg.robot_control_config_path
    )


def _get_override_value(overrides: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for override in overrides:
        if override.startswith(prefix):
            return override[len(prefix) :]
    return None


def _get_eval_preset_overrides(preset_name: str) -> dict[str, object]:
    preset = PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(f"Unknown preset '{preset_name}'.")
    ckpt_keys = {"agent.resume", "agent.resume_from_ckpt"}
    return {key: value for key, value in preset.items() if key not in ckpt_keys}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a trained policy continuously in simulation"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="oakink_eval_sim",
        help="Preset name from dexlink.src.rl.config.base.PRESETS.",
    )
    parser.add_argument(
        "--rl_run_dir",
        type=str,
        default=None,
        help="Optional explicit RL run directory containing models/<timestep>/policy.pt.",
    )
    parser.add_argument(
        "--ckpt_timestep",
        type=int,
        default=None,
        help="Optional explicit checkpoint timestep under --rl_run_dir/models/.",
    )
    parser.add_argument(
        "--compute_gt",
        action="store_true",
        help="Compute ground-truth extras on each sim step during evaluation.",
    )
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--greedy_actions",
        action="store_true",
        dest="greedy_actions",
        default=True,
        help="Use the policy distribution mean action during sim evaluation.",
    )
    action_group.add_argument(
        "--sample_actions",
        action="store_false",
        dest="greedy_actions",
        help="Sample actions from the policy distribution during sim evaluation.",
    )

    args, unknown = parser.parse_known_args(argv)
    overrides = extract_overrides(args, unknown)
    preset_overrides = _get_eval_preset_overrides(args.preset)
    resume_from_ckpt_override = _get_override_value(overrides, "agent.resume_from_ckpt")
    preset_resume_from_ckpt = PRESETS[args.preset].get("agent.resume_from_ckpt")
    policy_ckpt = _resolve_policy_checkpoint(
        rl_run_dir=args.rl_run_dir,
        ckpt_timestep=args.ckpt_timestep,
        resume_from_ckpt=resume_from_ckpt_override or preset_resume_from_ckpt,
        logger_logdir=str(get_logs_dir()),
    )
    run_dir = _infer_run_dir_from_policy_checkpoint(policy_ckpt)
    exp_cfg = _load_training_run_config(run_dir)
    convert_to_local_paths(exp_cfg)
    ckpt_timestep = int(policy_ckpt.parent.name)
    print(
        "Resolved sim eval checkpoint:\n"
        f"  policy checkpoint: {policy_ckpt}\n"
        f"  training run dir: {run_dir}\n"
        f"  checkpoint timestep: {ckpt_timestep}"
    )
    exp_cfg = apply_overrides_from_dict(exp_cfg, preset_overrides)
    if overrides:
        exp_cfg = apply_overrides(exp_cfg, overrides)
    exp_cfg.agent.resume_from_ckpt = str(policy_ckpt)

    env_cfg = exp_cfg.env
    logger_cfg = exp_cfg.logger
    agent_cfg = exp_cfg.agent
    eval_cfg = exp_cfg.eval

    if env_cfg.backend != "sim":
        raise ValueError(
            f"eval_sim requires env.backend='sim', got {env_cfg.backend!r}."
        )
    # if env_cfg.sim_cfg.num_envs != 1:
    #     raise ValueError(
    #         "eval_sim requires env.sim_cfg.num_envs == 1, "
    #         f"got {env_cfg.sim_cfg.num_envs}."
    #     )

    agent_cfg.resume = False
    logger = WandbLogger(logger_cfg, exp_seed=env_cfg.seed)

    env_cfg.sim_cfg.video_save_dir = str(
        logger.experiment_dir / env_cfg.sim_cfg.video_save_dir
    )
    eval_cfg.logdir = str(logger.experiment_dir)
    agent_cfg.seed = env_cfg.seed
    agent_cfg.device = env_cfg.device

    env = create_env(env_cfg)
    env.task_manager.verify_target_object_motion(mode=env_cfg.backend)

    agent_cfg.use_discriminator = env.use_discriminator
    logger.add_config({"experiment": asdict(exp_cfg)})
    logger.save_config(asdict(exp_cfg), name="sim_eval_exp_cfg.json")
    agent = PPOEval(env, logger=logger, agent_cfg=agent_cfg, eval_cfg=eval_cfg)

    try:
        agent.eval_policy_sim(
            policy_ckpt=policy_ckpt,
            seed=int(env_cfg.seed),
            compute_gt=bool(args.compute_gt),
            greedy_actions=bool(args.greedy_actions),
        )
    except KeyboardInterrupt:
        print("\nSim evaluation interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
