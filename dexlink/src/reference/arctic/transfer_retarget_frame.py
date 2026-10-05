from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from dexlink.scripts.retargeting.run_trajopt_arctic import main as run_arctic_trajopt
from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE
from dexlink.src.reference.arctic.transfer_config import ArcticTransferSpec, get_transfer


@dataclass(frozen=True)
class TransferRetargetFrameArgs:
    transfer: str
    side: str
    timestep: int
    workspace: Path
    preset: str
    robot_model: str
    save_retarget_probes: bool


def parse_args(
    argv: list[str] | None = None,
) -> tuple[TransferRetargetFrameArgs, list[str]]:
    parser = argparse.ArgumentParser(
        description="Retarget one hand/frame of an ARCTIC geometry-transfer clip."
    )
    parser.add_argument("--transfer", required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--timestep", type=int, required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--preset", default="arctic_default")
    parser.add_argument("--robot-model", default="xhand", choices=("xhand",))
    parser.add_argument("--save-retarget-probes", action="store_true")
    raw, trajopt_overrides = parser.parse_known_args(argv)
    invalid_overrides = [
        token
        for token in trajopt_overrides
        if token.startswith("--") and not token.startswith("--cfg.")
    ]
    if invalid_overrides:
        raise ValueError(
            "Trajectory optimizer overrides must use the --cfg.<field> form; "
            f"got {invalid_overrides}."
        )
    return (
        TransferRetargetFrameArgs(
            transfer=str(raw.transfer),
            side=str(raw.side),
            timestep=int(raw.timestep),
            workspace=raw.workspace.resolve(),
            preset=str(raw.preset),
            robot_model=str(raw.robot_model),
            save_retarget_probes=bool(raw.save_retarget_probes),
        ),
        trajopt_overrides,
    )


def build_trajopt_args(
    args: TransferRetargetFrameArgs,
    transfer: ArcticTransferSpec,
    trajopt_overrides: list[str],
) -> list[str]:
    clip = transfer.source_clip
    output_dir = transfer.frame_output_dir(
        args.workspace, args.timestep, args.side
    )
    command = [
        "--preset",
        args.preset,
        "--retarget-only",
        "--retarget-only-frames",
        "1",
        "--cfg.arctic.demo_path",
        str(clip.processed_demo_path()),
        "--cfg.arctic.target_object",
        transfer.target_object,
        "--cfg.arctic.timestep",
        str(args.timestep),
        "--cfg.arctic.start_timestep",
        str(clip.start_frame),
        "--cfg.arctic.end_timestep",
        str(clip.end_frame),
        "--cfg.arctic.skip",
        "1",
        "--cfg.arctic.hand",
        args.side,
        "--cfg.arctic.robot_model",
        args.robot_model,
        "--cfg.run_dir",
        str(output_dir),
        "--cfg.trajopt.visualize",
        "False",
        "--cfg.trajopt.pause_after_visualize",
        "False",
    ]
    if args.save_retarget_probes:
        command.append("--save-retarget-probes")
    return [*command, *trajopt_overrides]


def main(argv: list[str] | None = None) -> int:
    args, trajopt_overrides = parse_args(argv)
    transfer = get_transfer(args.transfer)
    clip = transfer.source_clip
    if not clip.start_frame <= args.timestep < clip.end_frame:
        raise ValueError(
            f"Timestep {args.timestep} is outside {clip.clip_id!r} interval "
            f"[{clip.start_frame}, {clip.end_frame})."
        )
    demo_path = clip.processed_demo_path()
    if not demo_path.is_file():
        raise FileNotFoundError(
            f"Prepared demo does not exist: {demo_path}. Run ARCTIC prepare first."
        )
    run_arctic_trajopt(build_trajopt_args(args, transfer, trajopt_overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
