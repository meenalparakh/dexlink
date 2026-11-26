"""List, validate, generate, train and roll out DexLink tasks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

from dexlink.utils.retargeting.path_utils import get_logs_dir

from .catalog import ROOT, load_catalog, sha256, task_by_id, validate_task

FULL_METHOD_PRESET = "seq_oakink_residual_target_trajectory_weighted_policy_v3"


@dataclass(frozen=True)
class TrainArguments:
    task: str
    run_directory: Path
    control_steps: int = 50000
    num_envs: int = 1024
    seed: int = 42
    project: str = "dexlink"
    entity: str | None = None
    numerical_audit: bool = False
    max_seconds: int | None = None


@dataclass(frozen=True)
class RolloutArguments:
    task: str
    episodes: int = 5
    parallel_envs: int = 5
    output: Path | None = None
    video: bool = False
    seed: int = 42


def run_module(
    module: str, arguments: list[str], *, environment: dict[str, str] | None = None
) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=environment)


def train(arguments: TrainArguments, overrides: list[str]) -> None:
    from .training import train_task

    if Path(arguments.task).suffix == ".npy":
        import numpy as np

        path = Path(arguments.task).resolve(strict=True)
        loaded = np.load(path, allow_pickle=True)
        if loaded.shape != () or not isinstance(loaded.item(), dict):
            raise ValueError(
                f"Expected a scalar-dictionary OakInk sequence NPY, received {path}."
            )
        task = {
            "id": path.stem,
            "reference_format": "oakink_sequence",
            "reference": str(path),
            "reference_sha256": sha256(path),
            "training_overrides": [],
        }
    else:
        task = task_by_id(arguments.task)
        validate_task(task, require_checkpoint=False)
    if (
        arguments.control_steps <= 0
        or arguments.control_steps % 8
        or arguments.num_envs < 2
    ):
        raise ValueError(
            "Expected positive control steps divisible by 8 and num_envs >= 2."
        )
    if arguments.max_seconds is not None and arguments.max_seconds < 1:
        raise ValueError("Expected max-seconds >= 1 when a time budget is provided.")
    destination = arguments.run_directory.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"Fresh training requires a new run directory, received {destination}."
        )
    train_task(task, arguments, overrides)


def rollout(
    args: RolloutArguments, overrides: list[str], *, record_policy_inputs: bool = False
) -> None:
    task = task_by_id(args.task)
    validate_task(task)
    checkpoint = task["checkpoint"]
    if args.episodes < 1 or args.parallel_envs < 1:
        raise ValueError("Expected positive episodes and parallel environments.")
    if args.video and args.episodes > args.parallel_envs:
        raise ValueError("Video recording requires episodes <= parallel-envs.")
    output = (
        args.output
        or get_logs_dir()
        / "rollouts"
        / args.task
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if output.exists():
        raise FileExistsError(
            f"Expected a new rollout output directory, received {output}."
        )
    command = [
        "--preset",
        "seq_oakink_eval_sim",
        "--rl_run_dir",
        checkpoint["run_directory"],
        "--ckpt_timestep",
        str(checkpoint["timestep"]),
        "--num_episodes",
        str(args.episodes),
        "--max_parallel_envs",
        str(args.parallel_envs),
        "--greedy_actions",
        "--seed",
        str(args.seed),
        "--no-compute_template_metrics",
        "--output_dir",
        str(output),
        "--env.sim_cfg.show_viewer",
        "False",
        "--env.task_cfg.save_reference_video",
        "False",
        "--env.sim_cfg.video_save_dir",
        str(output / "videos"),
        *overrides,
    ]
    if args.video:
        command.append("--record_video")
    if record_policy_inputs:
        command.append("--record_policy_inputs")
    run_module("dexlink.scripts.rl.eval_final_metrics", command)


def reference(args: argparse.Namespace) -> None:
    from .reference import ReferenceArguments, generate_reference

    generate_reference(
        ReferenceArguments(
            args.spec,
            args.output,
            args.density,
            args.max_workers,
            args.skip_retargeting,
            args.refine_collisions,
        )
    )


def configure_runtime() -> None:
    """Keep child processes and headless simulator caches in project storage."""
    os.chdir(ROOT)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    for name, directory in {
        "GS_CACHE_FILE_PATH": "genesis",
        "TI_OFFLINE_CACHE_FILE_PATH": "taichi",
        "NUMBA_CACHE_DIR": "numba",
        "MPLCONFIGDIR": "matplotlib",
    }.items():
        cache = get_logs_dir() / "cache" / directory
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(cache))
    os.environ.setdefault("OMP_NUM_THREADS", "4")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser(
        "list", help="List task descriptions and packaged checkpoint steps."
    )
    listing.add_argument("--group", choices=("library40", "tutorials"))
    check = commands.add_parser(
        "check", help="Check task references, assets and checkpoint checksums."
    )
    check.add_argument("task", help="Task ID or 'all'.")
    check.add_argument("--references-only", action="store_true")
    generate = commands.add_parser(
        "reference", help="Generate a reference from a YAML task specification."
    )
    generate.add_argument("spec", type=Path)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument(
        "--density",
        type=float,
        required=True,
        help="Explicit object density in kg/m^3.",
    )
    generate.add_argument("--max-workers", type=int, default=4)
    generate.add_argument("--skip-retargeting", action="store_true")
    generate.add_argument("--refine-collisions", action="store_true")
    learning = commands.add_parser(
        "train", help="Start fresh full-method PPO training."
    )
    learning.add_argument("task")
    learning.add_argument("--run-directory", type=Path, required=True)
    learning.add_argument("--control-steps", type=int, default=50000)
    learning.add_argument("--num-envs", type=int, default=1024)
    learning.add_argument("--seed", type=int, default=42)
    learning.add_argument("--project", default="dexlink")
    learning.add_argument("--entity")
    learning.add_argument(
        "--max-seconds",
        type=int,
        help="Stop after a complete PPO update and save its positive checkpoint.",
    )
    evaluation = commands.add_parser(
        "rollout", help="Evaluate or record an included trained policy."
    )
    evaluation.add_argument("task")
    evaluation.add_argument("--episodes", type=int, default=5)
    evaluation.add_argument("--parallel-envs", type=int, default=5)
    evaluation.add_argument("--output", type=Path)
    evaluation.add_argument("--video", action="store_true")
    evaluation.add_argument("--seed", type=int, default=42)
    collection = commands.add_parser(
        "collect",
        help="Save policy observations/actions, state trajectories and evaluation metrics without videos.",
    )
    collection.add_argument("tasks", nargs="*")
    collection.add_argument("--group", choices=("library40", "tutorials"))
    collection.add_argument("--output", type=Path, required=True)
    collection.add_argument("--episodes", type=int, default=50)
    collection.add_argument("--parallel-envs", type=int, default=10)
    collection.add_argument("--seed", type=int, default=42)
    rendering = commands.add_parser(
        "render",
        help="Render a reference or saved policy rollout without running a policy.",
    )
    source = rendering.add_mutually_exclusive_group(required=True)
    source.add_argument("--task")
    source.add_argument("--reference", type=Path)
    source.add_argument("--eval-directory", type=Path)
    rendering.add_argument("--reference-override", type=Path)
    rendering.add_argument("--task-id", default="custom")
    rendering.add_argument("--output", type=Path, required=True)
    rendering.add_argument("--episode", type=int, default=0)
    rendering.add_argument("--width", type=int, default=640)
    rendering.add_argument("--height", type=int, default=480)
    rendering.add_argument("--frame-stride", type=int, default=2)
    rendering.add_argument("--fps", type=float, default=15.0)
    rendering.add_argument(
        "--camera-mode", choices=("tracked", "fixed"), default="tracked"
    )
    smoke = commands.add_parser(
        "smoke", help="Run real PPO updates with strict finite-value auditing."
    )
    smoke.add_argument("task")
    smoke.add_argument("--run-directory", type=Path, required=True)
    smoke.add_argument("--control-steps", type=int, default=64)
    smoke.add_argument("--num-envs", type=int, default=32)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--project", default="dexlink")
    smoke.add_argument("--entity")
    smoke.add_argument("--max-seconds", type=int)
    args, overrides = parser.parse_known_args(argv)
    if overrides and args.command not in ("train", "rollout", "smoke"):
        parser.error(f"Unexpected arguments: {overrides}")
    configure_runtime()
    if args.command == "list":
        for task in load_catalog().values():
            if args.group is None or task["group"] == args.group:
                checkpoint = task["checkpoint"]
                step = str(checkpoint["timestep"]) if checkpoint else "train first"
                print(f"{task['id']:20} {step:>12}  {task['name']}")
    elif args.command == "check":
        tasks = (
            list(load_catalog().values())
            if args.task == "all"
            else [task_by_id(args.task)]
        )
        for task in tasks:
            print(
                json.dumps(
                    validate_task(
                        task,
                        require_checkpoint=not args.references_only
                        and task["checkpoint"] is not None,
                    )
                )
            )
    elif args.command == "reference":
        reference(args)
    elif args.command in ("train", "smoke"):
        train(
            TrainArguments(
                args.task,
                args.run_directory,
                args.control_steps,
                args.num_envs,
                args.seed,
                args.project,
                args.entity,
                numerical_audit=args.command == "smoke",
                max_seconds=args.max_seconds,
            ),
            overrides,
        )
    elif args.command == "rollout":
        rollout(
            RolloutArguments(
                task=args.task,
                episodes=args.episodes,
                parallel_envs=args.parallel_envs,
                output=args.output,
                video=args.video,
                seed=args.seed,
            ),
            overrides,
        )
    elif args.command == "collect":
        from .data import CollectionArguments, collect

        collect(
            CollectionArguments(
                args.tasks,
                args.output,
                args.group,
                args.episodes,
                args.parallel_envs,
                args.seed,
            )
        )
    elif args.command == "render":
        from .data import RenderArguments, render

        values = vars(args).copy()
        values.pop("command")
        render(RenderArguments(**values))
