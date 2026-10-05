from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE, get_clip
from dexlink.src.reference.arctic.runtime import materialize_runtime_sequence
from dexlink.scripts.rl.train_rl import main as train_rl
from dexlink.src.rl.config.base import PRESETS
from dexlink.src.rl.config.seq_oakink_final_experiment_presets import (
    NO_TEMPLATE_REWARD_OVERRIDES,
)
from dexlink.src.rl.config.seq_oakink_presets import (
    SEQ_BIMANUAL_REQUIRED_SETTINGS,
    SEQ_RESIDUAL_TARGET_TRAJECTORY_WEIGHTED_POLICY_V3_PRESET_SIM,
)

ARCTIC_EVAL_PRESET = "arctic_evals_our_method"
ARCTIC_NO_TEMPLATE_REWARD_PRESET = "arctic_evals_no_template_reward"
RewardVariant = Literal["our_method", "no_template_reward"]


@dataclass(frozen=True)
class TrainArgs:
    clip: str
    workspace: Path
    seed: int
    runname: str | None
    local: bool
    reward_variant: RewardVariant
    timesteps: int


def parse_args(argv: list[str] | None = None) -> tuple[TrainArgs, list[str]]:
    parser = argparse.ArgumentParser(
        description="Train our existing residual RL method on a dense ARCTIC clip."
    )
    parser.add_argument("--clip", required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runname", default=None)
    parser.add_argument("--local", action="store_true")
    parser.add_argument(
        "--reward-variant",
        choices=("our_method", "no_template_reward"),
        default="our_method",
    )
    parser.add_argument("--timesteps", type=int, default=50_000)
    raw, unknown = parser.parse_known_args(argv)
    if raw.timesteps <= 0:
        raise ValueError(f"Expected --timesteps > 0, got {raw.timesteps}.")
    return (
        TrainArgs(
            clip=str(raw.clip),
            workspace=raw.workspace.resolve(),
            seed=int(raw.seed),
            runname=raw.runname,
            local=bool(raw.local),
            reward_variant=raw.reward_variant,
            timesteps=int(raw.timesteps),
        ),
        unknown,
    )


def register_training_preset(
    *,
    exported_path: Path,
    seed: int,
    runname: str,
    reward_variant: RewardVariant = "our_method",
    timesteps: int = 50_000,
) -> str:
    if not exported_path.is_file():
        raise FileNotFoundError(
            f"ARCTIC RL sequence is missing: {exported_path}. Run assembly first."
        )
    if timesteps <= 0:
        raise ValueError(f"Expected timesteps > 0, got {timesteps}.")
    preset_name = {
        "our_method": ARCTIC_EVAL_PRESET,
        "no_template_reward": ARCTIC_NO_TEMPLATE_REWARD_PRESET,
    }[reward_variant]
    preset = SEQ_RESIDUAL_TARGET_TRAJECTORY_WEIGHTED_POLICY_V3_PRESET_SIM.copy()
    preset.update(SEQ_BIMANUAL_REQUIRED_SETTINGS)
    if reward_variant == "no_template_reward":
        preset.update(NO_TEMPLATE_REWARD_OVERRIDES)
    preset.update(
        {
            "preset": preset_name,
            "env.env_class": "multi_object",
            "env.task_cfg.demo_dataset": "oakink",
            "env.task_cfg.oakink_exported_path": str(exported_path),
            "env.task_cfg.start_timestep": 0,
            "env.task_cfg.end_timestep": -1,
            "env.task_cfg.skip": 1,
            "env.task_cfg.save_reference_video": True,
            "env.seed": seed,
            "env.key_task_protocol_version": "arctic_evals_v1",
            "env.sim_cfg.num_envs": 1024,
            "agent.timesteps": timesteps,
            "agent.normalize_advantages": True,
            "agent.eval_freq": 0,
            "agent.eval_metrics_freq": 0,
            "agent.eval_video_freq": 0,
            "logger.project": "dexlink_arctic_evals",
            "logger.group": reward_variant,
            "logger.runname": runname,
        }
    )
    PRESETS[preset_name] = preset
    return preset_name


def main(argv: list[str] | None = None) -> int:
    args, overrides = parse_args(argv)
    clip = get_clip(args.clip)
    runname = args.runname or (
        f"{clip.clip_id}__{args.reward_variant}__{args.timesteps}__seed{args.seed}"
    )
    runtime_sequence = materialize_runtime_sequence(clip, args.workspace)
    preset_name = register_training_preset(
        exported_path=runtime_sequence,
        seed=args.seed,
        runname=runname,
        reward_variant=args.reward_variant,
        timesteps=args.timesteps,
    )
    command = ["--preset", preset_name]
    if args.local:
        command.append("--local")
    train_rl([*command, *overrides])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
