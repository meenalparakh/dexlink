from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from dexlink.scripts.retargeting.run_trajopt_arctic import main as run_arctic_trajopt
from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE, get_clip
from dexlink.src.reference.arctic.runtime import register_processor


@dataclass(frozen=True)
class RetargetFrameArgs:
    clip: str
    side: str
    timestep: int
    workspace: Path
    preset: str
    robot_model: str
    save_retarget_probes: bool
    run_dir: Path | None


def parse_args(
    argv: list[str] | None = None,
) -> tuple[RetargetFrameArgs, list[str]]:
    parser = argparse.ArgumentParser(
        description="Densely retarget one hand at one frame of an ARCTIC eval clip."
    )
    parser.add_argument("--clip", required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--timestep", type=int, required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--preset", default="arctic_default")
    parser.add_argument("--robot-model", default="xhand", choices=("xhand",))
    parser.add_argument("--save-retarget-probes", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    raw, unknown = parser.parse_known_args(argv)
    return (
        RetargetFrameArgs(
            clip=str(raw.clip),
            side=str(raw.side),
            timestep=int(raw.timestep),
            workspace=raw.workspace.resolve(),
            preset=str(raw.preset),
            robot_model=str(raw.robot_model),
            save_retarget_probes=bool(raw.save_retarget_probes),
            run_dir=None if raw.run_dir is None else raw.run_dir.resolve(),
        ),
        unknown,
    )


def output_dir(args: RetargetFrameArgs) -> Path:
    if args.run_dir is not None:
        return args.run_dir
    return (
        args.workspace / "retargeted" / args.clip / f"t{args.timestep:04d}" / args.side
    )


def main(argv: list[str] | None = None) -> int:
    args, trajopt_overrides = parse_args(argv)
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
    clip = get_clip(args.clip)
    if not clip.start_frame <= args.timestep < clip.end_frame:
        raise ValueError(
            f"Timestep {args.timestep} is outside {clip.clip_id!r} interval "
            f"[{clip.start_frame}, {clip.end_frame})."
        )
    demo_path = clip.processed_demo_path()
    if not demo_path.is_file():
        raise FileNotFoundError(
            f"Prepared demo does not exist: {demo_path}. Run prepare first."
        )
    target_object = register_processor(clip, args.workspace)
    command = [
        "--preset",
        args.preset,
        "--retarget-only",
        "--retarget-only-frames",
        "1",
        "--cfg.arctic.demo_path",
        str(demo_path),
        "--cfg.arctic.target_object",
        target_object,
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
        str(output_dir(args)),
        "--cfg.trajopt.visualize",
        "False",
        "--cfg.trajopt.pause_after_visualize",
        "False",
    ]
    if args.save_retarget_probes:
        command.append("--save-retarget-probes")
    run_arctic_trajopt([*command, *trajopt_overrides])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
