#!/usr/bin/env python3
"""Validate URDF topology, inertia, joint limits, and sampled box collisions.

Uses the 15-axis separating-axis test on the actual URDF collision boxes.
NO pairs of different links are disabled, including neighboring hinges and
all tab/keeper/wall pairs. This is sampled geometry validation, not dynamics
or a continuous-time collision proof.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from kinematics import Model
from motion import STAGES, Motion

ROOT = Path(__file__).resolve().parent


def sat_overlap(
    Ta: np.ndarray, ha: np.ndarray, Tb: np.ndarray, hb: np.ndarray
) -> float:
    """Minimum interval overlap on separating axes; >0 means penetration."""
    A, B = Ta[:3, :3], Tb[:3, :3]
    axes = np.concatenate(
        (A.T, B.T, np.cross(A.T[:, None, :], B.T[None, :, :]).reshape(-1, 3))
    )
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > 1e-10] / lengths[lengths > 1e-10, None]
    r = np.abs(axes @ A) @ ha + np.abs(axes @ B) @ hb
    overlaps = r - np.abs(axes @ (Tb[:3, 3] - Ta[:3, 3]))
    return float(np.min(overlaps))


def collisions_at(model: Model, cfg: dict[str, float], tolerance: float = 1e-8):
    poses = model.box_poses(cfg, collision=True)
    centers = np.array([T[:3, 3] for _, T in poses])
    half = np.array([b.size / 2 for b, _ in poses])
    radii = np.array([np.abs(T[:3, :3]) @ h for (_, T), h in zip(poses, half)])
    broad = np.all(
        np.abs(centers[:, None, :] - centers[None, :, :])
        < radii[:, None, :] + radii[None, :, :] - tolerance,
        axis=2,
    )
    result = []
    for i, j in np.argwhere(np.triu(broad, 1)):
        bi, Ti = poses[i]
        bj, Tj = poses[j]
        if bi.link == bj.link:
            continue
        penetration = sat_overlap(Ti, half[i], Tj, half[j])
        if penetration > tolerance:
            result.append((bi.name, bj.name, penetration))
    return result


def insertion_metrics(
    model: Model, motion: Motion, q: dict[str, float], end: str
) -> dict:
    fk = model.fk(q)
    base_joint = next(j for j in model.joints if j.name == f"{end}_socket_base")
    socket_frame = fk[base_joint.parent] @ base_joint.pose
    tab = next(b for b in model.collisions if b.link == f"{end}_tuck_tab")
    T = np.linalg.inv(socket_frame) @ fk[tab.link] @ tab.pose
    verts = tab.vertices() @ T[:3, :3].T + T[:3, 3]
    zmin, zmax = float(verts[:, 2].min()), float(verts[:, 2].max())
    t, s = motion.p["thickness"], motion.d["socket_axis_spacing"]
    return {
        "wall_face_clearance_m": zmin - t / 2,
        "keeper_face_clearance_m": s - t / 2 - zmax,
        "mouth_lateral_clearance_m": motion.d["mouth_width"] / 2
        - float(np.max(np.abs(verts[:, 0]))),
        "tip_insertion_depth_m": max(0.0, -float(verts[:, 1].min())),
        "tab_centering_error_m": abs((zmin + zmax) / 2 - s / 2),
    }


def validate(model: Model, motion: Motion, samples: int = 2001) -> dict:
    if samples < 2:
        raise ValueError("At least two samples are required")
    errors = []
    for link in model.links:
        if link not in model.inertias:
            errors.append(f"No inertia for {link}")
            continue
        mass, I = model.inertias[link]
        eig = np.linalg.eigvalsh(I)
        if mass <= 0 or np.any(eig <= 0) or not np.all(np.isfinite(eig)):
            errors.append(f"Invalid mass/inertia in {link}")
        if eig[-1] > eig[0] + eig[1] + 1e-12:
            errors.append(f"Inertia triangle inequality in {link}")
    if any(np.any(b.size <= 0) for b in model.collisions):
        errors.append("Nonpositive collision box")
    # At both ends, the receiver must have exactly two separated ear boxes.
    for end in ("top", "bottom"):
        ears = [b for b in model.collisions if b.link == f"{end}_socket_margin"]
        if len(ears) != 2 or not all(abs(b.pose[0, 3]) > b.size[0] / 2 for b in ears):
            errors.append(f"{end} receiver mouth is not structurally open")
    # Endpoints and stage boundaries are checked in addition to uniform samples.
    times = np.unique(np.r_[np.linspace(0, motion.total, samples), motion.bounds])
    counts, examples, worst = Counter(), [], 0.0
    limit_errors = []
    measured = {end: [] for end in ("top", "bottom")}
    angle_history = []
    max_fk_error = None
    for idx, t in enumerate(times):
        q, label, stage_idx = motion.evaluate(float(t))
        angle_history.append([q[j.name] for j in model.joints])
        for j in model.joints:
            if q[j.name] < j.lower - 1e-9 or q[j.name] > j.upper + 1e-9:
                limit_errors.append([float(t), j.name, q[j.name]])
        found = collisions_at(model, q)
        for a, b, penetration in found:
            counts[(a, b)] += 1
            worst = max(worst, penetration)
            if len(examples) < 20:
                examples.append(
                    {
                        "time_s": float(t),
                        "stage": label,
                        "a": a,
                        "b": b,
                        "sat_overlap_m": penetration,
                    }
                )
        stage = STAGES[stage_idx]
        if stage.kind == "insert":
            measured[stage.end].append(insertion_metrics(model, motion, q, stage.end))
        if idx in (0, len(times) // 2, len(times) - 1):
            error = model.crosscheck_urdfpy(q)
            if error is not None:
                max_fk_error = max(max_fk_error or 0, error)
    qs = np.asarray(angle_history)
    velocities = np.abs(np.diff(qs, axis=0) / np.diff(times)[:, None])
    max_vel = np.max(velocities, axis=0)
    vel_errors = [
        j.name for j, v in zip(model.joints, max_vel) if v > j.velocity + 1e-5
    ]
    summary = {}
    for end in ("top", "bottom"):
        metrics = measured[end]
        final = insertion_metrics(model, motion, motion.keyframes[-1], end)
        summary[end] = {
            "samples_in_insertion": len(metrics),
            "min_wall_clearance_m": min(
                [v["wall_face_clearance_m"] for v in metrics]
                + [final["wall_face_clearance_m"]]
            ),
            "min_keeper_clearance_m": min(
                [v["keeper_face_clearance_m"] for v in metrics]
                + [final["keeper_face_clearance_m"]]
            ),
            "max_centering_error_m": max(
                [v["tab_centering_error_m"] for v in metrics]
                + [final["tab_centering_error_m"]]
            ),
            "seated": final,
        }
        if (
            min(
                summary[end]["min_wall_clearance_m"],
                summary[end]["min_keeper_clearance_m"],
            )
            < -1e-9
        ):
            errors.append(f"Negative {end} insertion clearance")
    if max_fk_error is not None and max_fk_error > 1e-9:
        errors.append("urdfpy/reference FK disagreement")
    vertices = []
    for box, pose in model.box_poses(motion.keyframes[-1], collision=True):
        vertices.append(box.vertices() @ pose[:3, :3].T + pose[:3, 3])
    points = np.concatenate(vertices)
    lo, hi = points.min(axis=0), points.max(axis=0)
    return {
        "passed": not (errors or counts or limit_errors or vel_errors),
        "assembled_aabb_root_xyz_min_m": lo.tolist(),
        "assembled_aabb_root_xyz_max_m": hi.tolist(),
        "assembled_exterior_width_depth_height_m": (hi - lo)[[0, 2, 1]].tolist(),
        "backend": model.backend,
        "urdfpy_executed": model.robot is not None,
        "num_links": len(model.links),
        "num_joints": len(model.joints),
        "num_collision_boxes": len(model.collisions),
        "num_configurations": len(times),
        "duration_s": motion.total,
        "all_interlink_collision_pairs_enabled": True,
        "collision_tolerance_m": 1e-8,
        "static_errors": errors,
        "joint_limit_errors": limit_errors[:20],
        "velocity_limit_errors": vel_errors,
        "max_sampled_joint_speed_rad_s": float(max_vel.max()),
        "collision_pair_sample_count": sum(counts.values()),
        "collision_pairs": [
            {"a": a, "b": b, "samples": n} for (a, b), n in counts.items()
        ],
        "collision_examples": examples,
        "max_sat_overlap_m": worst,
        "urdfpy_reference_max_transform_error": max_fk_error,
        "insertion": summary,
        "limitations": "Sampled, rigid-geometry tests only. No contact dynamics, friction, glue, continuous collision proof, or cardboard elasticity validation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=ROOT / "carton.urdf")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--backend", choices=("urdfpy", "reference"), default="urdfpy")
    parser.add_argument("--samples", type=int, default=2001)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "output" / "validation.json"
    )
    args = parser.parse_args()
    try:
        model = Model(args.urdf, args.backend)
        motion = Motion(args.metadata or args.urdf.with_suffix(".json"))
        report = validate(model, motion, args.samples)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
