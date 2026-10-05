"""Evaluate one ARCTIC checkpoint and compute DexMachina-compatible ADD-AUC."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from dexlink.src.reference.arctic.add_auc import main as add_auc_main
from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE
from dexlink.src.reference.arctic.eval_rollouts import main as eval_rollouts_main


@dataclass(frozen=True)
class EvaluateArcticCheckpointArgs:
    clip: str | None
    transfer: str | None
    workspace: Path
    run_dir: Path
    checkpoint_timestep: int
    output_dir: Path
    wandb_entity: str
    wandb_project: str
    wandb_group: str
    wandb_run_name: str
    wandb_mode: str


def parse_args() -> tuple[EvaluateArcticCheckpointArgs, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    benchmark = parser.add_mutually_exclusive_group(required=True)
    benchmark.add_argument("--clip")
    benchmark.add_argument("--transfer")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-timestep", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    raw, remaining = parser.parse_known_args()
    if raw.checkpoint_timestep <= 0:
        raise ValueError(
            "Expected --checkpoint-timestep > 0, got " f"{raw.checkpoint_timestep}."
        )
    return (
        EvaluateArcticCheckpointArgs(
            clip=None if raw.clip is None else str(raw.clip),
            transfer=None if raw.transfer is None else str(raw.transfer),
            workspace=raw.workspace.resolve(),
            run_dir=raw.run_dir,
            checkpoint_timestep=int(raw.checkpoint_timestep),
            output_dir=raw.output_dir,
            wandb_entity=str(raw.wandb_entity),
            wandb_project=str(raw.wandb_project),
            wandb_group=str(raw.wandb_group),
            wandb_run_name=str(raw.wandb_run_name),
            wandb_mode=str(raw.wandb_mode),
        ),
        remaining,
    )


def main() -> int:
    args, overrides = parse_args()
    benchmark_args = (
        ["--clip", args.clip]
        if args.clip is not None
        else ["--transfer", args.transfer]
    )
    benchmark_args.extend(["--workspace", str(args.workspace)])
    rollout_args = [
        *benchmark_args,
        "--rl_run_dir",
        str(args.run_dir),
        "--ckpt_timestep",
        str(args.checkpoint_timestep),
        "--output_dir",
        str(args.output_dir),
        "--num_episodes",
        "50",
        "--max_parallel_envs",
        "50",
        "--dtw_window_fraction",
        "0.1",
        "--greedy_actions",
        "--wandb_mode",
        args.wandb_mode,
        "--wandb_entity",
        args.wandb_entity,
        "--wandb_project",
        args.wandb_project,
        "--wandb_group",
        args.wandb_group,
        "--wandb_run_name",
        args.wandb_run_name,
        *overrides,
    ]
    summary_path = args.output_dir / "summary_metrics.json"
    episode_path = args.output_dir / "episode_metrics.csv"
    rollout_error_path = args.output_dir / "rollout_metric_error.json"
    complete_rollout_metrics = summary_path.is_file() and episode_path.is_file()
    if not complete_rollout_metrics and not rollout_error_path.is_file():
        try:
            eval_rollouts_main(rollout_args)
        except ValueError as error:
            chunks = tuple((args.output_dir / "rollouts").glob("chunk_*.npz"))
            expected_error = (
                "No warping path found compatible with the local constraints"
            )
            if str(error) != expected_error or not chunks:
                raise
            # ADD-AUC uses the preserved rollout chunks and does not depend on DTW.
            rollout_error_path.write_text(
                json.dumps(
                    {
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "add_auc_still_valid": True,
                        "rollout_chunk_count": len(chunks),
                    },
                    indent=2,
                )
                + "\n"
            )
    add_auc_summary = args.output_dir / "dexmachina_add_auc/summary.json"
    if not add_auc_summary.is_file():
        try:
            add_auc_main(
                [
                    *benchmark_args,
                    "--rollout-dir",
                    str(args.output_dir),
                ]
            )
        except ValueError as error:
            (args.output_dir / "add_auc_error.json").write_text(
                json.dumps(
                    {
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "checkpoint_timestep": args.checkpoint_timestep,
                        "clip": args.clip,
                        "transfer": args.transfer,
                    },
                    indent=2,
                )
                + "\n"
            )
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
