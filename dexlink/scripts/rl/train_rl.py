#!/usr/bin/env python
"""Train the DexLink RL agent with dot-notation config overrides.

Examples:
    python -m dexlink.scripts.rl.train_rl --preset oakink_default
    python -m dexlink.scripts.rl.train_rl --preset oakink_default --env.sim_cfg.num_envs 24

Override syntax:
    --env.<nested_field> VALUE
    --agent.<field> VALUE
    --logger.<field> VALUE
    --eval.<field> VALUE
    --real_eval.<field> VALUE
Supports bool casting: pass True/False or 1/0.
"""
import argparse
import os
import sys
import traceback
import warnings
from dataclasses import asdict

import setuptools  # noqa: F401


def _should_use_egl(script_path: str) -> bool:
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return not has_display


# Prefer EGL whenever no desktop display is available.
file_path = os.path.abspath(__file__)
if _should_use_egl(file_path):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

from dexlink.src.rl.core_agent.wandb_logger import WandbLogger
from dexlink.src.rl.agent.ppo import PPO
from dexlink.src.rl.config.base import extract_overrides, load_experiment_config
from dexlink.src.rl.multi_object.env_factory import create_env

# os.environ['PYOPENGL_PLATFORM'] = 'egl'


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch Motion tracking RL agent")
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        help="Preset name from dexlink.src.rl.config.base.PRESETS",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally with viewer enabled and 32 simulation environments.",
    )

    # parse known/unknown to allow dynamic overrides
    args, unknown = parser.parse_known_args(argv)
    overrides = extract_overrides(args, unknown)
    exp_cfg = load_experiment_config(preset=args.preset, overrides=overrides)
    if args.local:
        exp_cfg.env.sim_cfg.show_viewer = True
        exp_cfg.env.sim_cfg.num_envs = 32

    # Instantiate env
    # Convert hierarchical EnvConfig to legacy EnvCfg-compatible structure if needed
    env_cfg = exp_cfg.env
    logger_cfg = exp_cfg.logger
    agent_cfg = exp_cfg.agent
    eval_cfg = exp_cfg.eval

    logger = WandbLogger(logger_cfg, exp_seed=env_cfg.seed)

    env_cfg.sim_cfg.video_save_dir = str(
        logger.experiment_dir / env_cfg.sim_cfg.video_save_dir
    )
    eval_cfg.logdir = str(logger.experiment_dir)
    agent_cfg.seed = env_cfg.seed
    agent_cfg.device = env_cfg.device

    # print_cfg_summary(exp_cfg)

    env = create_env(env_cfg)
    env.task_manager.verify_target_object_motion(mode=env_cfg.backend)

    # override agent args based on env properties
    agent_cfg.use_discriminator = env.use_discriminator
    logger.add_config({"experiment": asdict(exp_cfg)})
    logger.save_config(asdict(exp_cfg))

    # print_cfg_summary(exp_cfg)

    agent = PPO(env, logger=logger, agent_cfg=agent_cfg, eval_cfg=eval_cfg)

    agent.train()


if __name__ == "__main__":
    main(sys.argv[1:])
