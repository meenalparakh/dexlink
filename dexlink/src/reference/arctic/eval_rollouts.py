from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE, get_clip
from dexlink.src.reference.arctic.runtime import (
    materialize_runtime_sequence,
    materialize_runtime_transfer_sequence,
)
from dexlink.src.reference.arctic.transfer_config import get_transfer
from dexlink.scripts.rl.eval_final_metrics import main as eval_final_metrics
from dexlink.src.rl.config.base import PRESETS

ARCTIC_ROLLOUT_EVAL_PRESET = "arctic_evals_rollout_eval"


@dataclass(frozen=True)
class EvalArgs:
    clip: str | None
    transfer: str | None
    workspace: Path


def parse_args(argv: list[str] | None = None) -> tuple[EvalArgs, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run fixed-horizon ARCTIC rollouts for ADD-AUC evaluation."
    )
    benchmark = parser.add_mutually_exclusive_group(required=True)
    benchmark.add_argument("--clip")
    benchmark.add_argument("--transfer")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    raw, remaining = parser.parse_known_args(argv)
    return (
        EvalArgs(
            None if raw.clip is None else str(raw.clip),
            None if raw.transfer is None else str(raw.transfer),
            raw.workspace.resolve(),
        ),
        remaining,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the standard 50-episode evaluator and retain raw rollout chunks."""
    args, remaining = parse_args(argv)
    if args.clip is not None:
        clip = get_clip(args.clip)
        sequence_path = materialize_runtime_sequence(clip, args.workspace)
        frame_count = clip.frame_count
    else:
        if args.transfer is None:
            raise ValueError("Expected exactly one ARCTIC clip or transfer.")
        transfer = get_transfer(args.transfer)
        sequence_path = materialize_runtime_transfer_sequence(transfer, args.workspace)
        frame_count = transfer.frame_count
    if "--preset" in remaining:
        raise ValueError(
            "ARCTIC rollout evaluation owns --preset; do not pass a second preset."
        )
    PRESETS[ARCTIC_ROLLOUT_EVAL_PRESET] = {
        "preset": ARCTIC_ROLLOUT_EVAL_PRESET,
    }
    command = ["--preset", ARCTIC_ROLLOUT_EVAL_PRESET, *remaining]
    if "--num_episodes" not in command:
        command.extend(["--num_episodes", "50"])
    if "--dtw_window_fraction" not in command:
        command.extend(["--dtw_window_fraction", "0.1"])
    if "--max_steps" not in command:
        command.extend(["--max_steps", str(frame_count)])
    if (
        "--compute_template_metrics" not in command
        and "--no-compute_template_metrics" not in command
    ):
        command.append("--no-compute_template_metrics")
    command.extend(
        [
            "--env.task_cfg.oakink_exported_path",
            str(sequence_path),
            "--env.task_cfg.save_reference_video",
            "False",
            "--env.reset_cfg.eval_start_timestep",
            "0",
            "--env.reset_cfg.obj_noise",
            "0.0",
            "--env.reset_cfg.hand_noise_reach_phase",
            "0.0",
            "--env.reset_cfg.hand_noise_lift_phase",
            "0.0",
            "--env.reset_cfg.dr",
            "False",
            "--env.sim_cfg.apply_random_force",
            "False",
            "--env.rewards_cfg.early_terminate",
            "False",
        ]
    )
    return eval_final_metrics(command)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
