"""Render the two pilot references with exact robot FK and hash-bound receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np

from dexlink.visualizations.rl.render_clean_policy_rollout_video import (
    _fixed_camera_sequence,
    _frame_meshes,
    _load_hand_models,
    _load_scene_objects,
    _mesh_bounds,
)
from dexlink.visualizations.rl.camera import CameraSpec
from dexlink.visualizations.rl.rendering import _render_panel
from dexlink.src.rl.config.base import load_experiment_config


def render(manifest_path: Path, output: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    config = load_experiment_config(
        "seq_oakink_residual_target_trajectory_weighted_policy_v3"
    )
    control_period = config.env.sim_cfg.dt * config.env.decimation
    output.mkdir(parents=True, exist_ok=False)
    for variant in manifest["variants"]:
        path = Path(variant["native_reference"])
        expected = variant["native_reference_sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Reference hash changed before rendering.")
        with np.load(path, allow_pickle=True) as archive:
            motion = dict(archive)
        folder = output / variant["id"]
        folder.mkdir()
        hands, base = _load_hand_models(motion=motion, cache_dir=folder / "cache")
        objects = _load_scene_objects(
            reference_motion=motion,
            included_object_ids=("cup", "table"),
            cache_dir=folder / "cache",
        )
        indices = np.arange(variant["frames"])
        bounds = [
            _mesh_bounds(
                _frame_meshes(
                    motion=motion,
                    frame_index=int(frame),
                    hand_models=hands,
                    robot_base_pose=base,
                    scene_objects=tuple(
                        obj for obj in objects if obj.object_id != "table"
                    ),
                    support_plane=None,
                )
            )
            for frame in indices
        ]
        for view, direction in (
            ("front", (-1.0, -1.0, 0.65)),
            ("side", (1.0, 1.0, 0.65)),
        ):
            cameras, xmags, ymags, _ = _fixed_camera_sequence(
                mesh_bounds=bounds,
                camera=CameraSpec(direction=direction, margin=1.12),
                width=640,
                height=480,
            )
            destination = folder / (view + ".mp4")
            with imageio.get_writer(
                destination,
                fps=1 / control_period,
                codec="libx264",
                quality=7,
                macro_block_size=None,
            ) as writer:
                for index, frame in enumerate(indices):
                    meshes = _frame_meshes(
                        motion=motion,
                        frame_index=int(frame),
                        hand_models=hands,
                        robot_base_pose=base,
                        scene_objects=objects,
                        support_plane=None,
                    )
                    panel = _render_panel(
                        meshes,
                        camera_pose=cameras[index],
                        xmag=float(xmags[index]),
                        ymag=float(ymags[index]),
                        width=640,
                        height=480,
                    )
                    writer.append_data(np.asarray(panel, dtype=np.uint8))
                    if index in {0, len(indices) - 1}:
                        imageio.imwrite(folder / f"{view}_{index:03d}.png", panel)
            receipt = {
                "kind": "reference",
                "variant": variant["id"],
                "view": view,
                "native_reference_sha256": expected,
                "video_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "frames": len(indices),
                "fps": 1 / control_period,
                "control_period_s": control_period,
                "author_review": "pending",
                "physics_execution": False,
            }
            destination.with_suffix(".json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.manifest, args.output)
