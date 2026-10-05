#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from dexlink.src.retargeting.optimization.franka_substitution import (
    FrankaSubstitution,
    FrankaSubstitutionConfig,
)
from dexlink.utils.retargeting.path_utils import get_assets_dir


@dataclass
class RunFrankaExportConfig:
    input_trajectory: str
    export_path: str | None = None
    franka_urdf: str = str(
        get_assets_dir() / "robots" / "frankaX_hand" / "panda_xhand_obj.urdf"
    )
    visualize: bool = True
    pause_after_visualize: bool = True
    visualize_time_step: float = 0.05


def _parse_args(argv=None) -> RunFrankaExportConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run FrankaSubstitution on an exported floating-xhand trajectory and "
            "optionally visualize the source and substituted motions in Meshcat."
        )
    )
    parser.add_argument(
        "--input-trajectory",
        required=True,
        type=str,
        help="Path to the source .npz trajectory exported from the xhand sequence stage.",
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default=None,
        help="Optional output .npz path. Defaults next to the input as *_franka_substitution.npz.",
    )
    parser.add_argument(
        "--franka-urdf",
        type=str,
        default=str(
            get_assets_dir() / "robots" / "frankaX_hand" / "panda_xhand_obj.urdf"
        ),
        help="Path to the Franka+xhand URDF.",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Visualize the source xhand and optimized Franka+xhand trajectories in Meshcat.",
    )
    parser.add_argument(
        "--pause-after-visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pause after publishing the Meshcat recording.",
    )
    parser.add_argument(
        "--visualize-time-step",
        type=float,
        default=0.05,
        help="Playback timestep used when publishing the comparison trajectory.",
    )
    args = parser.parse_args(argv)
    return RunFrankaExportConfig(
        input_trajectory=args.input_trajectory,
        export_path=args.export_path,
        franka_urdf=args.franka_urdf,
        visualize=args.visualize,
        pause_after_visualize=args.pause_after_visualize,
        visualize_time_step=args.visualize_time_step,
    )


def _default_export_path(input_trajectory: str) -> Path:
    input_path = Path(input_trajectory)
    if input_path.suffix != ".npz":
        raise ValueError(
            f"Expected --input-trajectory to point to a .npz file, got {input_path}."
        )
    return input_path.with_name(f"{input_path.stem}_franka_substitution.npz")


def main(argv=None) -> None:
    cfg = _parse_args(argv)
    export_path = (
        Path(cfg.export_path)
        if cfg.export_path is not None
        else _default_export_path(cfg.input_trajectory)
    )

    substitution = FrankaSubstitution(
        provided_traj_info=cfg.input_trajectory,
        franka_xhand_urdf=cfg.franka_urdf,
        cfg=FrankaSubstitutionConfig(),
    )
    result = substitution.solve()
    print(f"Franka substitution success: {result.is_success()}")

    saved_payload = substitution.export_info(export_path=export_path)
    print(f"Saved Franka substitution result to {export_path}")
    print("Exported arrays: " f"{sorted(saved_payload.keys())}")

    if cfg.visualize:
        substitution.visualize_trajectory(
            pause=cfg.pause_after_visualize,
            time_step=cfg.visualize_time_step,
        )


if __name__ == "__main__":
    main()
