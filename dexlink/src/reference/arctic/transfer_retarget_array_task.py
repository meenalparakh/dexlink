from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE
from dexlink.src.reference.arctic.transfer_config import ArcticTransferSpec, get_transfer
from dexlink.src.reference.arctic.transfer_retarget_frame import main as retarget_frame_main


@dataclass(frozen=True)
class TransferArrayTaskArgs:
    transfer: str
    workspace: Path
    array_index: int
    preset: str


def parse_args(argv: list[str] | None = None) -> TransferArrayTaskArgs:
    parser = argparse.ArgumentParser(
        description="Run one ARCTIC geometry-transfer hand/frame Slurm task."
    )
    parser.add_argument("--transfer", required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--array-index", type=int, required=True)
    parser.add_argument("--preset", default="arctic_default")
    raw = parser.parse_args(argv)
    return TransferArrayTaskArgs(
        transfer=str(raw.transfer),
        workspace=raw.workspace.resolve(),
        array_index=int(raw.array_index),
        preset=str(raw.preset),
    )


def decode_array_index(
    transfer: ArcticTransferSpec, array_index: int
) -> tuple[int, str]:
    task_count = 2 * transfer.frame_count
    if not 0 <= array_index < task_count:
        raise ValueError(
            f"Array index {array_index} is outside [0, {task_count}) for "
            f"{transfer.transfer_id!r}."
        )
    timestep = transfer.source_clip.start_frame + array_index // 2
    side = ("left", "right")[array_index % 2]
    return timestep, side


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    transfer = get_transfer(args.transfer)
    timestep, side = decode_array_index(transfer, args.array_index)
    output = transfer.frame_motion_path(args.workspace, timestep, side)
    if output.is_file():
        print(
            f"Skipping existing retarget: transfer={transfer.transfer_id}, "
            f"timestep={timestep}, side={side}, output={output}"
        )
        return 0
    retarget_frame_main(
        [
            "--transfer",
            transfer.transfer_id,
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
        f"Completed retarget: transfer={transfer.transfer_id}, "
        f"timestep={timestep}, side={side}, output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
