#!/usr/bin/env python
"""Compose synchronized human-reference and robot-rollout videos side by side."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComposeVideoArgs:
    human_reference: Path
    robot_rollout: Path
    output: Path
    left_label: str
    right_label: str
    panel_width: int
    panel_height: int


def parse_args(argv: list[str] | None = None) -> ComposeVideoArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-reference", type=Path, required=True)
    parser.add_argument("--robot-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-label", default="ARCTIC human reference")
    parser.add_argument("--right-label", default="Learned robot rollout")
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=480)
    raw = parser.parse_args(argv)
    if raw.panel_width <= 0 or raw.panel_height <= 0:
        raise ValueError(
            "Expected positive panel dimensions, got "
            f"{raw.panel_width}x{raw.panel_height}."
        )
    return ComposeVideoArgs(
        human_reference=raw.human_reference.resolve(),
        robot_rollout=raw.robot_rollout.resolve(),
        output=raw.output.resolve(),
        left_label=str(raw.left_label),
        right_label=str(raw.right_label),
        panel_width=int(raw.panel_width),
        panel_height=int(raw.panel_height),
    )


def _escape_drawtext(value: str) -> str:
    return value.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")


def compose(args: ComposeVideoArgs) -> Path:
    for path in (args.human_reference, args.robot_rollout):
        if not path.is_file():
            raise FileNotFoundError(f"Expected input video at {path}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    left = _escape_drawtext(args.left_label)
    right = _escape_drawtext(args.right_label)
    filter_graph = (
        f"[0:v]scale={args.panel_width}:{args.panel_height}:force_original_aspect_ratio=decrease,"
        f"pad={args.panel_width}:{args.panel_height}:(ow-iw)/2:(oh-ih)/2:white,setsar=1,"
        f"drawtext=text='{left}':x=(w-text_w)/2:y=18:fontsize=24:"
        "fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=8[left];"
        f"[1:v]scale={args.panel_width}:{args.panel_height}:force_original_aspect_ratio=decrease,"
        f"pad={args.panel_width}:{args.panel_height}:(ow-iw)/2:(oh-ih)/2:white,setsar=1,"
        f"drawtext=text='{right}':x=(w-text_w)/2:y=18:fontsize=24:"
        "fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=8[right];"
        "[left][right]hstack=inputs=2[out]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(args.human_reference),
            "-i",
            str(args.robot_rollout),
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-shortest",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(args.output),
        ],
        check=True,
    )
    if not args.output.is_file():
        raise FileNotFoundError(f"ffmpeg did not write {args.output}.")
    print(f"[arctic-side-by-side] {args.output}")
    return args.output


def main(argv: list[str] | None = None) -> int:
    compose(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
