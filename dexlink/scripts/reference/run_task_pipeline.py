"""Run the public DexLink reference-generation pipeline for one task YAML."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dexlink.src.reference.task_io import load_task_spec
from dexlink.utils.retargeting.path_utils import get_src, resolve_project_path


@dataclass(frozen=True)
class PipelineArgs:
    task: Path
    max_workers: int
    retarget_only_frames: int
    density: float | None
    skip_retargeting: bool
    skip_reference_build: bool
    refine_collisions: bool
    allow_unresolved_collisions: bool
    retarget_hand_type: str
    refined_output: Path | None


def _parse_args(argv: list[str] | None = None) -> PipelineArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Retarget a DexLink task, build its reference trajectory, export the "
            "RL payload, and optionally run Drake collision refinement."
        )
    )
    parser.add_argument("task", type=Path, help="Path to a task YAML file.")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--retarget-only-frames", type=int, default=20)
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help="Object density override. By default the task category range is used.",
    )
    parser.add_argument("--skip-retargeting", action="store_true")
    parser.add_argument("--skip-reference-build", action="store_true")
    parser.add_argument("--refine-collisions", action="store_true")
    parser.add_argument("--allow-unresolved-collisions", action="store_true")
    parser.add_argument(
        "--retarget-hand-type",
        choices=("xhand", "inspire_hand"),
        default="xhand",
    )
    parser.add_argument("--refined-output", type=Path, default=None)
    parsed = parser.parse_args(argv)
    return PipelineArgs(
        task=parsed.task,
        max_workers=int(parsed.max_workers),
        retarget_only_frames=int(parsed.retarget_only_frames),
        density=None if parsed.density is None else float(parsed.density),
        skip_retargeting=bool(parsed.skip_retargeting),
        skip_reference_build=bool(parsed.skip_reference_build),
        refine_collisions=bool(parsed.refine_collisions),
        allow_unresolved_collisions=bool(parsed.allow_unresolved_collisions),
        retarget_hand_type=str(parsed.retarget_hand_type),
        refined_output=parsed.refined_output,
    )


def _run(module: str, arguments: list[str], *, root: Path) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("[dexlink]", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def _exported_rl_path(task: dict) -> Path:
    trajectory = task["trajectory"]
    if not isinstance(trajectory, dict):
        raise TypeError(
            f"Expected task['trajectory'] to be a mapping, got "
            f"{type(trajectory).__name__}."
        )
    path = trajectory.get("exported_rl_path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "Task trajectory must define a non-empty 'exported_rl_path'."
        )
    return resolve_project_path(path)


def run_pipeline(args: PipelineArgs) -> Path:
    if args.max_workers <= 0:
        raise ValueError(f"Expected max_workers > 0, got {args.max_workers}.")
    if args.retarget_only_frames <= 0:
        raise ValueError(
            "Expected retarget_only_frames > 0, got "
            f"{args.retarget_only_frames}."
        )
    if args.density is not None and args.density <= 0.0:
        raise ValueError(f"Expected a positive density, got {args.density}.")

    root = get_src()
    task_path = resolve_project_path(args.task)
    task = load_task_spec(task_path)
    exported_path = _exported_rl_path(task)

    # Independent explicit and category retargeting jobs run on local CPU workers.
    if not args.skip_retargeting:
        if task.get("retarget_jobs"):
            _run(
                "dexlink.scripts.retargeting.run_sequence_retargetings",
                [
                    "--spec",
                    str(task_path),
                    "--max-workers",
                    str(args.max_workers),
                    "--retarget-only-frames",
                    str(args.retarget_only_frames),
                ],
                root=root,
            )
        _run(
            "dexlink.scripts.retargeting.run_task_category_retargetings",
            [
                "--spec",
                str(task_path),
                "--max-workers",
                str(args.max_workers),
                "--retarget-only-frames",
                str(args.retarget_only_frames),
                "--hand-type",
                args.retarget_hand_type,
                "--skip-existing",
            ],
            root=root,
        )

    # Reference construction also writes the RL-readable sequence payload.
    if not args.skip_reference_build:
        build_arguments = [
            "--spec",
            str(task_path),
            "--write-manifest",
            "--retarget-hand-type",
            args.retarget_hand_type,
            "--export-oakink",
        ]
        if args.density is not None:
            build_arguments.extend(["--density", str(args.density)])
        _run(
            "dexlink.scripts.retargeting.build_sequence_from_spec",
            build_arguments,
            root=root,
        )
    if not exported_path.is_file():
        raise FileNotFoundError(
            f"Expected the RL reference payload at {exported_path} after building."
        )

    if not args.refine_collisions:
        return exported_path

    refined_output = (
        resolve_project_path(args.refined_output)
        if args.refined_output is not None
        else exported_path.with_name(f"{exported_path.stem}_refined.npy")
    )
    refinement_arguments = [
        "--input",
        str(exported_path),
        "--output",
        str(refined_output),
    ]
    if args.allow_unresolved_collisions:
        refinement_arguments.append("--allow-unresolved")
    _run(
        "dexlink.scripts.retargeting.generate_non_colliding_trajectory_drake",
        refinement_arguments,
        root=root,
    )
    if not refined_output.is_file():
        raise FileNotFoundError(
            f"Collision refinement did not write its expected output: "
            f"{refined_output}."
        )
    return refined_output


def main(argv: list[str] | None = None) -> None:
    output = run_pipeline(_parse_args(argv))
    print(f"[dexlink] RL reference: {output}")


if __name__ == "__main__":
    main()
