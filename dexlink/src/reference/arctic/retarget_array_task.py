from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from dexlink.src.reference.arctic.assemble_dense_motion import frame_motion_path
from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE, ArcticClipSpec, get_clip
from dexlink.src.reference.arctic.retarget_frame import main as retarget_frame_main


@dataclass(frozen=True)
class RetargetArrayTaskArgs:
    clip: str
    workspace: Path
    array_index: int
    preset: str


def parse_args(argv: list[str] | None = None) -> RetargetArrayTaskArgs:
    parser = argparse.ArgumentParser(
        description="Run one hand/frame ARCTIC retarget from a Slurm array index."
    )
    parser.add_argument("--clip", required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--array-index", type=int, required=True)
    parser.add_argument("--preset", default="arctic_default")
    raw = parser.parse_args(argv)
    return RetargetArrayTaskArgs(
        clip=str(raw.clip),
        workspace=raw.workspace.resolve(),
        array_index=int(raw.array_index),
        preset=str(raw.preset),
    )


def decode_array_index(clip: ArcticClipSpec, array_index: int) -> tuple[int, str]:
    task_count = 2 * clip.frame_count
    if not 0 <= array_index < task_count:
        raise ValueError(
            f"Array index {array_index} is outside [0, {task_count}) for "
            f"{clip.clip_id!r}."
        )
    timestep = clip.start_frame + array_index // 2
    side = ("left", "right")[array_index % 2]
    return timestep, side


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    clip = get_clip(args.clip)
    timestep, side = decode_array_index(clip, args.array_index)
    output = frame_motion_path(args.workspace, clip, timestep, side)
    if output.is_file():
        print(
            f"Skipping existing retarget: clip={clip.clip_id}, "
            f"timestep={timestep}, side={side}, output={output}"
        )
        return 0

    retarget_frame_main(
        [
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
        ]
    )
    if not output.is_file():
        raise FileNotFoundError(
            f"Retarget task returned without creating expected output: {output}."
        )
    print(
        f"Completed retarget: clip={clip.clip_id}, timestep={timestep}, "
        f"side={side}, output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
