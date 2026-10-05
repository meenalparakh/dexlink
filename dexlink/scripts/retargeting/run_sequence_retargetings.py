#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dexlink.src.reference.task_io import load_task_spec
from dexlink.utils.retargeting.path_utils import get_logs_dir, get_src


def _load_spec(path: Path) -> dict:
    return load_task_spec(path)


def _default_run_dir(root: Path, experiment_id: str, job_id: str) -> Path:
    return get_logs_dir() / "sequence_retargetings" / experiment_id / job_id


def _job_command(
    root: Path, spec: dict, spec_path: Path, job: dict, run_dir: Path
) -> list[str]:
    preset = str(job.get("preset", spec.get("preset", "oakink_default")))
    hand_side = str(job.get("hand_side", spec.get("hand_side", "right")))
    robot_model = str(job.get("robot_model", spec.get("robot_model", "xhand")))
    cmd = [
        sys.executable,
        "-m",
        "dexlink.scripts.retargeting.run_trajopt_oakink",
        "--preset",
        preset,
        "--cfg.oakink.demo_path",
        str(job["demo_path"]),
        "--cfg.oakink.hand",
        hand_side,
        "--cfg.oakink.oakink_hand_side",
        hand_side,
        "--cfg.oakink.robot_model",
        robot_model,
        "--cfg.run_dir",
        str(run_dir),
        "--cfg.trajopt.visualize",
        "False",
        "--cfg.trajopt.pause_after_visualize",
        "False",
        "--cfg.trajopt_seq.visualize",
        "False",
        "--cfg.trajopt_seq.pause_after_visualize",
        "False",
    ]
    if "target_obj_path" in job and str(job["target_obj_path"]).strip():
        cmd.extend(["--cfg.oakink.target_obj_path", str(job["target_obj_path"])])
    if "target_demo_path" in job and str(job["target_demo_path"]).strip():
        cmd.extend(["--cfg.oakink.target_demo_path", str(job["target_demo_path"])])
    if "target_transform_info" in job and str(
        job["target_transform_info"]
    ).strip() not in {"", "TBD", "None"}:
        cmd.extend(
            ["--cfg.oakink.target_transform_info", str(job["target_transform_info"])]
        )
    if "target_scale" in job:
        cmd.extend(["--cfg.oakink.target_scale", str(job["target_scale"])])
    if "start_timestep" in job:
        cmd.extend(["--cfg.oakink.start_timestep", str(job["start_timestep"])])
    if "end_timestep" in job:
        cmd.extend(["--cfg.oakink.end_timestep", str(job["end_timestep"])])
    if "skip" in job:
        cmd.extend(["--cfg.oakink.skip", str(job["skip"])])
    return cmd


def _job_manifest(spec_path: Path, spec: dict, job: dict, run_dir: Path) -> dict:
    return {
        "spec_path": str(spec_path),
        "experiment_id": spec["experiment_id"],
        "job_id": job["job_id"],
        "run_dir": str(run_dir),
        "demo_path": job.get("demo_path"),
        "target_obj_path": job.get("target_obj_path"),
        "target_demo_path": job.get("target_demo_path"),
        "target_transform_info": job.get("target_transform_info"),
        "target_scale": job.get("target_scale", 1.0),
        "hand_side": job.get("hand_side", spec.get("hand_side", "right")),
        "robot_model": job.get("robot_model", spec.get("robot_model", "xhand")),
    }


def _validate_job(job: dict, spec_path: Path) -> None:
    if "job_id" not in job:
        raise ValueError(f"{spec_path}: retarget job is missing 'job_id'.")
    if "demo_path" not in job or str(job["demo_path"]).strip() in {"", "TBD"}:
        raise ValueError(
            f"{spec_path}: retarget job {job.get('job_id')} has unresolved demo_path."
        )
    has_target_obj = "target_obj_path" in job and str(
        job["target_obj_path"]
    ).strip() not in {"", "TBD"}
    has_target_demo = "target_demo_path" in job and str(
        job["target_demo_path"]
    ).strip() not in {"", "TBD"}
    if not (has_target_obj or has_target_demo):
        raise ValueError(
            f"{spec_path}: retarget job {job.get('job_id')} must define target_obj_path or target_demo_path."
        )


def _run_one(cmd: list[str], root: Path, run_dir: Path, dry_run: bool) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return 0
    completed = subprocess.run(cmd, cwd=root, check=False)
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source-target retargeting jobs declared by DexLink task YAML files."
    )
    parser.add_argument("--spec", type=Path, action="append", default=[])
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument("--glob", type=str, default="*.yaml")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--retarget-only-frames",
        type=int,
        default=20,
        help="Number of repeated frames to export for the retarget-only template motion.",
    )
    return parser.parse_args()


def _gather_specs(args: argparse.Namespace) -> list[Path]:
    spec_paths = [path.resolve() for path in args.spec]
    if args.spec_dir is not None:
        spec_paths.extend(sorted(args.spec_dir.resolve().glob(args.glob)))
    unique = []
    seen = set()
    for path in spec_paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    if not unique:
        raise ValueError("No spec files were provided.")
    return unique


def main() -> None:
    args = parse_args()
    root = get_src()
    spec_paths = _gather_specs(args)

    scheduled_jobs: list[tuple[list[str], Path]] = []
    for spec_path in spec_paths:
        spec = _load_spec(spec_path)
        experiment_id = str(spec["experiment_id"])
        jobs = spec.get("retarget_jobs", [])
        if not jobs:
            continue
        for job in jobs:
            _validate_job(job, spec_path)
            run_dir = Path(
                job.get(
                    "run_dir", _default_run_dir(root, experiment_id, str(job["job_id"]))
                )
            )
            manifest = _job_manifest(spec_path, spec, job, run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "retarget_job_manifest.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(manifest, f, indent=2)
            cmd = _job_command(root, spec, spec_path, job, run_dir)
            cmd.extend(
                [
                    "--retarget-only",
                    "--retarget-only-frames",
                    str(args.retarget_only_frames),
                ]
            )
            scheduled_jobs.append((cmd, run_dir))

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_map = {
            executor.submit(_run_one, cmd, root, run_dir, args.dry_run): (cmd, run_dir)
            for cmd, run_dir in scheduled_jobs
        }
        for future in as_completed(future_map):
            cmd, run_dir = future_map[future]
            return_code = future.result()
            if return_code != 0:
                failures += 1
                print(f"[failed] run_dir={run_dir} code={return_code}: {' '.join(cmd)}")
            else:
                print(f"[ok] run_dir={run_dir}")

    if failures > 0:
        raise SystemExit(failures)


if __name__ == "__main__":
    main()
