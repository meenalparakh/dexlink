#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from export_oakink_lift_data import build_payload


def _default_output_path(motion_npy_path: Path) -> Path:
    motion_npy_path = motion_npy_path.resolve()
    stem = motion_npy_path.stem
    if stem.endswith(".npz"):
        stem = Path(stem).stem
    return motion_npy_path.parent / f"{stem}_franka_oakink_lift_exported.npy"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export a Franka-substituted OakInk motion trajectory into a payload "
            "compatible with OakInkLiftProcessor."
        )
    )
    parser.add_argument("--motion-npy-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--density", type=float, required=True)
    parser.add_argument(
        "--robot-model",
        type=str,
        default="franka_xhand",
        help="Robot model for the exported motion. Defaults to 'franka_xhand'.",
    )
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = _default_output_path(args.motion_npy_path)
    return args


def main():
    args = parse_args()
    payload = build_payload(args)
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, payload, allow_pickle=True)
    print(f"Saved Franka OakInk lift payload to {output_path}")


if __name__ == "__main__":
    main()
