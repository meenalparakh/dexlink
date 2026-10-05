from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from dexlink.src.reference.arctic.assemble_dense_motion import frame_motion_path
from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE, ArcticClipSpec, get_clip


@dataclass(frozen=True)
class RetargetClipArgs:
    clip: str
    workspace: Path
    max_workers: int
    python_executable: str
    preset: str
    overwrite: bool
    start_timestep: int | None
    end_timestep: int | None


@dataclass(frozen=True)
class RetargetJob:
    timestep: int
    side: str
    output_path: Path
    log_path: Path
    command: tuple[str, ...]
    overwrite: bool


def parse_args(
    argv: list[str] | None = None,
) -> tuple[RetargetClipArgs, list[str]]:
    parser = argparse.ArgumentParser(
        description="Retarget every frame and both hands with bounded local workers."
    )
    parser.add_argument("--clip", required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--preset", default="arctic_default")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-timestep", type=int, default=None)
    parser.add_argument("--end-timestep", type=int, default=None)
    raw, trajopt_overrides = parser.parse_known_args(argv)
    if raw.max_workers <= 0:
        raise ValueError(f"Expected --max-workers > 0, got {raw.max_workers}.")
    return (
        RetargetClipArgs(
            clip=str(raw.clip),
            workspace=raw.workspace.resolve(),
            max_workers=int(raw.max_workers),
            python_executable=str(Path(raw.python_executable).resolve()),
            preset=str(raw.preset),
            overwrite=bool(raw.overwrite),
            start_timestep=raw.start_timestep,
            end_timestep=raw.end_timestep,
        ),
        trajopt_overrides,
    )


def _jobs(
    clip: ArcticClipSpec,
    args: RetargetClipArgs,
    trajopt_overrides: list[str],
) -> list[RetargetJob]:
    jobs: list[RetargetJob] = []
    start = clip.start_frame if args.start_timestep is None else args.start_timestep
    end = clip.end_frame if args.end_timestep is None else args.end_timestep
    if start < clip.start_frame or end > clip.end_frame or end <= start:
        raise ValueError(
            f"Requested retarget interval [{start}, {end}) must be a non-empty "
            f"subset of {clip.clip_id}=[{clip.start_frame}, {clip.end_frame})."
        )
    for timestep in range(start, end):
        for side in ("left", "right"):
            output = frame_motion_path(args.workspace, clip, timestep, side)
            if output.is_file() and not args.overwrite:
                continue
            command = [
                args.python_executable,
                "-m",
                "dexlink.src.reference.arctic.retarget_frame",
                "--clip",
                clip.clip_id,
                "--side",
                side,
                "--timestep",
                str(timestep),
                "--workspace",
                str(args.workspace),
                "--preset",
                args.preset,
                *trajopt_overrides,
            ]
            jobs.append(
                RetargetJob(
                    timestep=timestep,
                    side=side,
                    output_path=output,
                    log_path=output.parent / "retarget.log",
                    command=tuple(command),
                    overwrite=args.overwrite,
                )
            )
    return jobs


def _run_job(job: RetargetJob) -> tuple[int, str]:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    if job.overwrite and job.output_path.is_file():
        job.output_path.unlink()
    with job.log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            job.command,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        output_tail = "\n".join(
            job.log_path.read_text(encoding="utf-8").splitlines()[-80:]
        )
        raise RuntimeError(
            f"Retarget failed for timestep={job.timestep}, side={job.side}, "
            f"returncode={result.returncode}; log={job.log_path}:\n{output_tail}"
        )
    if not job.output_path.is_file():
        raise RuntimeError(
            f"Retarget command returned success but did not create {job.output_path}."
        )
    return job.timestep, job.side


def main(argv: list[str] | None = None) -> int:
    args, trajopt_overrides = parse_args(argv)
    clip = get_clip(args.clip)
    jobs = _jobs(clip, args, trajopt_overrides)
    start = clip.start_frame if args.start_timestep is None else args.start_timestep
    end = clip.end_frame if args.end_timestep is None else args.end_timestep
    expected_jobs = 2 * (end - start)
    print(
        f"Running {len(jobs)}/{expected_jobs} missing retarget jobs for "
        f"{clip.clip_id} with {args.max_workers} workers."
    )
    failures = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_job = {executor.submit(_run_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                future.result()
                print(f"[retarget] complete timestep={job.timestep} side={job.side}")
            except Exception as exc:
                failures.append(str(exc))
                print(
                    f"[retarget] FAILED timestep={job.timestep} side={job.side}: "
                    f"{exc}"
                )
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(jobs)} retarget jobs failed.\n"
            + "\n\n".join(failures)
        )
    print(
        f"All {expected_jobs} outputs are available for {clip.clip_id} "
        f"interval [{start}, {end})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
