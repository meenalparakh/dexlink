#!/usr/bin/env python3
"""Validate generated cabinets: kinematic sweep clearance (urdfpy) and a Genesis joint sweep.

    conda run -n genesis python assets/procedural_kitchen_cabinet/validate.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from urdfpy import URDF

ASSET_DIR = Path(__file__).resolve().parent
CABINETS = ("kitchen_cabinet_single_door", "kitchen_cabinet_double_door_drawer")


@dataclass(frozen=True)
class ValidateConfig:
    sweep_samples: int = 25
    point_spacing: float = 0.004
    genesis_steps_per_leg: int = 300
    dt: float = 0.01
    device: str = "cpu"
    run_genesis: bool = True


def link_collision_boxes(urdf: URDF) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Per link: (4x4 box pose in link frame, box size) for every collision box."""

    boxes: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for link in urdf.links:
        entries = []
        for collision in link.collisions:
            if collision.geometry.box is None:
                raise ValueError(f"Expected only box collisions, link {link.name!r} has {collision.geometry}.")
            entries.append((np.asarray(collision.origin, dtype=np.float64), np.asarray(collision.geometry.box.size, dtype=np.float64)))
        boxes[link.name] = entries
    return boxes


def box_surface_points(size: np.ndarray, spacing: float) -> np.ndarray:
    """Grid points on the surface of a centered box."""

    axes = [np.linspace(-0.5 * s, 0.5 * s, max(2, int(np.ceil(s / spacing)) + 1)) for s in size]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    on_surface = np.any(np.isclose(np.abs(grid), 0.5 * size[None], atol=1e-9), axis=1)
    return grid[on_surface]


def box_signed_distance(points: np.ndarray, pose: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Exact signed distance (negative inside) from world points to an oriented box."""

    local = (points - pose[:3, 3]) @ pose[:3, :3]
    q = np.abs(local) - 0.5 * size
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def kinematic_sweep(urdf_path: Path, cfg: ValidateConfig) -> dict[str, float]:
    """Max penetration (m) between each moving link and every other link over the joint sweep."""

    urdf = URDF.load(str(urdf_path))
    boxes = link_collision_boxes(urdf)
    moving = [joint for joint in urdf.actuated_joints]
    result: dict[str, float] = {}
    for joint in moving:
        values = np.linspace(joint.limit.lower, joint.limit.upper, cfg.sweep_samples)
        # The joint child and all of its descendants move together.
        moving_links = {joint.child}
        while True:
            grown = moving_links | {j.child for j in urdf.joints if j.parent in moving_links}
            if grown == moving_links:
                break
            moving_links = grown
        worst = 0.0
        for value in values:
            fk = urdf.link_fk(cfg={joint.name: float(value)}, use_names=True)
            moving_points = np.concatenate(
                [
                    box_surface_points(size, cfg.point_spacing) @ (fk[name] @ pose)[:3, :3].T + (fk[name] @ pose)[:3, 3]
                    for name in moving_links
                    for pose, size in boxes[name]
                ]
            )
            for name, entries in boxes.items():
                if name in moving_links:
                    continue
                for pose, size in entries:
                    depth = -box_signed_distance(moving_points, fk[name] @ pose, size).min()
                    worst = max(worst, float(depth))
        result[joint.name] = worst
    return result


def genesis_sweep(urdf_path: Path, cfg: ValidateConfig) -> dict[str, dict[str, float]]:
    """Drive each joint 0 -> upper -> 0 with PD control under gravity; record reached values."""

    import genesis as gs

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=cfg.dt, substeps=4),
        rigid_options=gs.options.RigidOptions(enable_collision=True, enable_self_collision=True, enable_joint_limit=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.URDF(file=str(ASSET_DIR / "support/counter.urdf"), fixed=True))
    # Place the carcass bottom on the counter top (z=0); the URDF root is the grip center.
    layout = json.loads((urdf_path.parent / "layout.json").read_text(encoding="utf-8"))
    root_position = tuple(float(v) for v in layout["root_origin_in_cabinet_frame"])
    cabinet = scene.add_entity(
        gs.morphs.URDF(file=str(urdf_path), fixed=True, pos=root_position, convexify=True, recompute_inertia=False)
    )
    scene.build()
    urdf = URDF.load(str(urdf_path))
    report: dict[str, dict[str, float]] = {}
    for joint in urdf.actuated_joints:
        dof = [cabinet.get_joint(joint.name).dofs_idx_local[0]]
        cabinet.set_dofs_kp(np.array([400.0]), dofs_idx_local=dof)
        cabinet.set_dofs_kv(np.array([40.0]), dofs_idx_local=dof)
        trace = []
        for target in (joint.limit.upper, 0.0):
            cabinet.control_dofs_position(np.array([target]), dofs_idx_local=dof)
            for _ in range(cfg.genesis_steps_per_leg):
                scene.step()
                trace.append(float(cabinet.get_dofs_position(dof).cpu().numpy()[0]))
        trace_array = np.asarray(trace)
        half = cfg.genesis_steps_per_leg
        report[joint.name] = {
            "upper_limit": float(joint.limit.upper),
            "max_reached": float(trace_array[:half].max()),
            "final_after_return": float(trace_array[-1]),
            "min_reached": float(trace_array.min()),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip_genesis", action="store_true")
    args = parser.parse_args()
    cfg = ValidateConfig(run_genesis=not args.skip_genesis)
    report: dict[str, object] = {}
    if cfg.run_genesis:
        import genesis as gs

        gs.init(backend=gs.cpu if cfg.device == "cpu" else gs.cuda, logging_level="warning")
    for name in CABINETS:
        urdf_path = ASSET_DIR / name / f"{name}.urdf"
        entry: dict[str, object] = {"kinematic_sweep_max_penetration_m": kinematic_sweep(urdf_path, cfg)}
        if cfg.run_genesis:
            entry["genesis_sweep"] = genesis_sweep(urdf_path, cfg)
        report[name] = entry
        print(name, json.dumps(entry, indent=2))
    (ASSET_DIR / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
