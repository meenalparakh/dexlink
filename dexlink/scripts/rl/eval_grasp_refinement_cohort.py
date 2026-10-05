#!/usr/bin/env python
"""Evaluate completed grasp-refinement runs and render their terminal states."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


RUN_INDEX_PATTERN = re.compile(r"^grasp_refine_gravity_terminal__(\d{3})__")
REFERENCE_ROOT_NAME = "grasp_refinement_sequences"


@dataclass(frozen=True)
class CohortEvalArgs:
    runs_root: Path
    references_root: Path
    output_root: Path
    checkpoint: str
    num_episodes: int
    max_parallel_envs: int
    camera_radius: float
    camera_height: float
    rerun: bool


def parse_args() -> CohortEvalArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--references-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("best", "5000"), default="best")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--max-parallel-envs", type=int, default=50)
    parser.add_argument("--camera-radius", type=float, default=0.55)
    parser.add_argument("--camera-height", type=float, default=1.8)
    parser.add_argument("--rerun", action="store_true")
    parsed = parser.parse_args()
    if parsed.num_episodes <= 0:
        raise ValueError(f"Expected num_episodes > 0, got {parsed.num_episodes}.")
    if parsed.max_parallel_envs < parsed.num_episodes:
        raise ValueError(
            "Video recording requires max_parallel_envs >= num_episodes, got "
            f"{parsed.max_parallel_envs} and {parsed.num_episodes}."
        )
    return CohortEvalArgs(
        runs_root=parsed.runs_root.resolve(),
        references_root=parsed.references_root.resolve(),
        output_root=parsed.output_root.resolve(),
        checkpoint=parsed.checkpoint,
        num_episodes=parsed.num_episodes,
        max_parallel_envs=parsed.max_parallel_envs,
        camera_radius=parsed.camera_radius,
        camera_height=parsed.camera_height,
        rerun=parsed.rerun,
    )


def _load_manifest_records(references_root: Path) -> dict[int, dict[str, object]]:
    manifest_path = references_root / "grasp_refinement_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Expected records list in {manifest_path}.")
    by_index: dict[int, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("index"), int):
            raise ValueError(f"Malformed manifest record: {record!r}.")
        index = int(record["index"])
        if index in by_index:
            raise ValueError(f"Duplicate manifest index {index}.")
        by_index[index] = record
    return by_index


def _run_index(run_dir: Path) -> int:
    match = RUN_INDEX_PATTERN.match(run_dir.name)
    if match is None:
        raise ValueError(f"Unexpected grasp-refinement run name: {run_dir.name!r}.")
    return int(match.group(1))


def _local_reference_path(record: dict[str, object], references_root: Path) -> Path:
    reference = record.get("reference_path")
    if not isinstance(reference, str):
        raise ValueError(f"Manifest record lacks string reference_path: {record!r}.")
    parts = Path(reference).parts
    try:
        root_index = parts.index(REFERENCE_ROOT_NAME)
    except ValueError as exc:
        raise ValueError(
            f"Expected reference path to contain {REFERENCE_ROOT_NAME!r}, got {reference!r}."
        ) from exc
    local_path = references_root.joinpath(*parts[root_index + 1 :])
    if not local_path.is_file():
        raise FileNotFoundError(f"Downloaded reference not found: {local_path}")
    return local_path


def _completed_runs(runs_root: Path) -> list[Path]:
    runs = [
        run_dir
        for run_dir in runs_root.iterdir()
        if run_dir.is_dir() and (run_dir / "models" / "5000" / "policy.pt").is_file()
    ]
    return sorted(runs, key=_run_index)


def _eval_command(
    args: CohortEvalArgs,
    run_dir: Path,
    reference_path: Path,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "dexlink.scripts.rl.eval_final_metrics",
        "--preset",
        "seq_oakink_grasp_stability_refinement_gravity_terminal",
        "--rl_run_dir",
        str(run_dir),
        "--num_episodes",
        str(args.num_episodes),
        "--max_parallel_envs",
        str(args.max_parallel_envs),
        "--greedy_actions",
        "--record_video",
        "--output_dir",
        str(output_dir),
        "--env.task_cfg.oakink_exported_path",
        str(reference_path),
        "--env.sim_cfg.camera_radius",
        str(args.camera_radius),
        "--env.sim_cfg.camera_ht",
        str(args.camera_height),
    ]
    if args.checkpoint == "best":
        command.append("--best_checkpoint")
    else:
        command.extend(("--ckpt_timestep", args.checkpoint))
    return command


def _extract_final_frame(video_path: Path, frame_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-sseof",
            "-0.1",
            "-i",
            str(video_path),
            "-update",
            "1",
            "-frames:v",
            "1",
            str(frame_path),
        ],
        check=True,
    )
    if not frame_path.is_file() or frame_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to extract a final frame from {video_path}.")


def _result_row(
    run_dir: Path,
    record: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    summary = json.loads((output_dir / "summary_metrics.json").read_text())
    metadata = json.loads((output_dir / "metadata.json").read_text())
    selection = metadata.get("checkpoint_selection") or {}
    return {
        "index": int(record["index"]),
        "category": str(record["category"]),
        "template_id": str(record["template_id"]),
        "run_name": run_dir.name,
        "checkpoint": str(metadata["policy_checkpoint"]),
        "best_policy_timestep": selection.get("policy_timestep", ""),
        "num_episodes": int(summary["num_episodes"]),
        "grasp_stability_success": summary["grasp_stability_success_mean"],
        "terminal_drift_m": summary["grasp_current_object_drift_m_mean"],
        "max_drift_m": summary["grasp_max_object_drift_m_mean"],
        "final_template_angle_mse_rad2": summary[
            "final_template_angle_mse_rad2_mean"
        ],
        "video_path": str(output_dir / "rollout.mp4"),
        "final_frame_path": str(output_dir / "final_frame.png"),
    }


def _write_aggregate(output_root: Path, rows: list[dict[str, object]]) -> None:
    csv_path = output_root / "cohort_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    successes = [float(row["grasp_stability_success"]) for row in rows]
    terminal_drifts = [float(row["terminal_drift_m"]) for row in rows]
    summary = {
        "completed_runs": len(rows),
        "successful_runs": sum(value >= 1.0 for value in successes),
        "macro_success_rate": sum(successes) / len(successes),
        "mean_terminal_drift_m": sum(terminal_drifts) / len(terminal_drifts),
        "max_terminal_drift_m": max(terminal_drifts),
        "categories": sorted({str(row["category"]) for row in rows}),
    }
    (output_root / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def main() -> int:
    args = parse_args()
    records = _load_manifest_records(args.references_root)
    runs = _completed_runs(args.runs_root)
    if not runs:
        raise ValueError(f"No completed 5k runs found below {args.runs_root}.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for position, run_dir in enumerate(runs, start=1):
        index = _run_index(run_dir)
        if index not in records:
            raise KeyError(f"Run index {index} is absent from the refinement manifest.")
        record = records[index]
        reference_path = _local_reference_path(record, args.references_root)
        output_dir = args.output_root / run_dir.name
        summary_path = output_dir / "summary_metrics.json"
        video_path = output_dir / "rollout.mp4"
        if args.rerun or not summary_path.is_file() or not video_path.is_file():
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "eval.log"
            print(
                f"[cohort-eval] {position}/{len(runs)} index={index:03d} "
                f"category={record['category']} template={record['template_id']}",
                flush=True,
            )
            with log_path.open("w", encoding="utf-8") as log_handle:
                subprocess.run(
                    _eval_command(args, run_dir, reference_path, output_dir),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        frame_path = output_dir / "final_frame.png"
        if args.rerun or not frame_path.is_file():
            _extract_final_frame(video_path, frame_path)
        rows.append(_result_row(run_dir, record, output_dir))
        _write_aggregate(args.output_root, rows)
    print(f"[cohort-eval] complete: {len(rows)} runs -> {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
