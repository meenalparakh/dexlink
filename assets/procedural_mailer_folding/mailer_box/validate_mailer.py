#!/usr/bin/env python3
"""Sampled OBB collision, inertia, velocity, clearance and insertion validation.
All inter-link pairs are enabled, INCLUDING neighboring hinge links.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from collision import collisions_at, sat_overlap
from kinematics import Model
from motion import Motion

ROOT = Path(__file__).resolve().parent


def vertices(model, q, link):
    values = []
    for b, T in model.box_poses(q, collision=True):
        if b.link == link:
            values.append(b.vertices() @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(values)


def metrics(model, motion, q):
    p, d = motion.p, motion.d
    W, D, H, t = p["width"], p["depth"], p["height"], p["thickness"]
    s = d["rollover_spacing"]
    output = {}
    for side, sgn in [("left", -1), ("right", 1)]:
        for k, y in enumerate(d["lock_y_positions"]):
            key = f"{side}_lock_{k}"
            pts = vertices(model, q, key)
            # Reflect left X so all clearances use the same sign convention.
            xx = sgn * pts[:, 0]
            xc = d["slot_x"]
            yc = y
            output[key] = {
                "min_face_clearance_m": min(
                    float(xx.min()) - (xc - d["slot_clear_gap"] / 2),
                    xc + d["slot_clear_gap"] / 2 - float(xx.max()),
                ),
                "lateral_clearance_m": p["lock_width"] / 2
                + p["lateral_clearance"]
                - float(np.abs(pts[:, 1] - yc).max()),
                "centering_error_m": abs(float(xx.mean()) - xc),
                "tip_z_m": float(pts[:, 2].min()),
                "protrusion_below_base_underside_m": max(
                    0, -t / 2 - float(pts[:, 2].min())
                ),
            }
    pts = vertices(model, q, "lid_front_tab")
    yy = pts[:, 1]
    output["front"] = {
        "min_face_clearance_m": min(
            float(yy.min()) - (D - s + t / 2), D - t / 2 - float(yy.max())
        ),
        "lateral_clearance_m": d["front_mouth_width"] / 2
        - float(np.abs(pts[:, 0]).max()),
        "centering_error_m": abs(float(yy.mean()) - (D - s / 2)),
        "tip_depth_m": H - float(pts[:, 2].min()),
    }
    for side, sgn in [("left", -1), ("right", 1)]:
        pts = vertices(model, q, f"lid_{side}_wing")
        xx = sgn * pts[:, 0]
        output[f"{side}_wing"] = {
            "min_face_clearance_m": min(
                float(xx.min()) - (W / 2 - s + t / 2), W / 2 - t / 2 - float(xx.max())
            ),
            "tip_depth_m": H - float(pts[:, 2].min()),
        }
    return output


def validate(model, motion, samples=2001):
    if samples < 2:
        raise ValueError("samples must be at least 2")
    errors = []
    for link in model.links:
        if link not in model.inertias:
            errors.append(f"Missing inertia {link}")
            continue
        m, I = model.inertias[link]
        eig = np.linalg.eigvalsh(I)
        if m <= 0 or np.any(eig <= 0) or not np.all(np.isfinite(eig)):
            errors.append(f"Invalid inertia {link}")
        if eig[-1] > sum(eig[:2]) + 1e-12:
            errors.append(f"Inertia triangle inequality {link}")
    for name in ("left_roll_margin", "right_roll_margin", "front_socket_margin"):
        boxes = [b for b in model.collisions if b.link == name]
        if len(boxes) != 2 or not all(abs(b.pose[0, 3]) > b.size[0] / 2 for b in boxes):
            errors.append(f"Blocked socket {name}")
    ts = np.unique(np.r_[np.linspace(0, motion.total, samples), motion.bounds])
    hist = []
    counts = Counter()
    examples = []
    max_overlap = 0
    limits = []
    insertion = []
    fk_err = None
    for i, t in enumerate(ts):
        q, label, k = motion.evaluate(float(t))
        hist.append([q[j.name] for j in model.joints])
        for j in model.joints:
            if not j.lower - 1e-9 <= q[j.name] <= j.upper + 1e-9:
                limits.append([float(t), j.name, q[j.name]])
        for a, b, pen in collisions_at(model, q):
            counts[(a, b)] += 1
            max_overlap = max(max_overlap, pen)
            if len(examples) < 25:
                examples.append(
                    dict(time_s=float(t), stage=label, a=a, b=b, sat_overlap_m=pen)
                )
        if k == motion.lock_insert_index:
            v = metrics(model, motion, q)
            insertion.append(
                min(
                    v[f"{s}_lock_{j}"]["min_face_clearance_m"]
                    for s in ("left", "right")
                    for j in range(2)
                )
            )
        if k == motion.front_insert_index:
            insertion.append(metrics(model, motion, q)["front"]["min_face_clearance_m"])
        if i in (0, len(ts) // 2, len(ts) - 1):
            er = model.crosscheck_urdfpy(q)
            if er is not None:
                fk_err = max(fk_err or 0, er)
    speed = np.abs(np.diff(hist, axis=0) / np.diff(ts)[:, None]).max(axis=0)
    velocity_errors = [
        j.name for j, v in zip(model.joints, speed) if v > j.velocity + 1e-5
    ]
    seated = metrics(model, motion, motion.keyframes[-1])
    if min(insertion) < -1e-9:
        errors.append("Negative insertion face clearance")
    if any(v["min_face_clearance_m"] < -1e-9 for v in seated.values()):
        errors.append("Negative seated face clearance")
    if any(v.get("centering_error_m", 0) > 1e-9 for v in seated.values()):
        errors.append("Tabs not centered")
    # Check that each nominal floor hole is truly empty of base collision boxes.
    hole_vol = 0
    for h in motion.meta["base_holes"]:
        x0, x1, y0, y1 = h
        T = np.eye(4)
        T[:3, 3] = [(x0 + x1) / 2, (y0 + y1) / 2, 0]
        half = np.array([x1 - x0, y1 - y0, motion.p["thickness"]]) / 2
        hole_vol += np.prod(2 * half)
        for b in model.collisions:
            if b.link == "base" and sat_overlap(T, half, b.pose, b.size / 2) > 1e-9:
                errors.append("Base collision geometry fills a slot")
    base_vol = sum(float(np.prod(b.size)) for b in model.collisions if b.link == "base")
    p, d = motion.p, motion.d
    desired = (p["width"] - 2 * d["hinge_relief"]) * (
        p["depth"] - 2 * d["hinge_relief"]
    ) * p["thickness"] - hole_vol
    if abs(base_vol - desired) > 1e-12:
        errors.append("Incorrect perforated base volume")
    if fk_err is not None and fk_err > 1e-9:
        errors.append("urdfpy/reference FK mismatch")
    cloud = np.concatenate(
        [
            b.vertices() @ T[:3, :3].T + T[:3, 3]
            for b, T in model.box_poses(motion.keyframes[-1], True)
        ]
    )
    lo, hi = cloud.min(0), cloud.max(0)
    return dict(
        passed=not (errors or counts or limits or velocity_errors),
        backend=model.backend,
        urdfpy_executed=model.robot is not None,
        num_links=len(model.links),
        num_joints=len(model.joints),
        collision_boxes=len(model.collisions),
        sampled_configurations=len(ts),
        duration_s=motion.total,
        all_interlink_collision_pairs_enabled=True,
        collision_tolerance_m=1e-8,
        collision_pair_sample_count=sum(counts.values()),
        collision_pairs=[dict(a=a, b=b, samples=n) for (a, b), n in counts.items()],
        collision_examples=examples,
        maximum_sat_overlap_m=max_overlap,
        static_errors=errors,
        joint_limit_errors=limits[:20],
        velocity_limit_errors=velocity_errors,
        max_sampled_joint_speed_rad_s=float(speed.max()),
        minimum_sampled_insertion_face_clearance_m=min(insertion),
        seated=seated,
        assembled_aabb_min_m=lo.tolist(),
        assembled_aabb_max_m=hi.tolist(),
        assembled_exterior_width_depth_height_m=(hi - lo).tolist(),
        urdfpy_reference_max_transform_error=fk_err,
        limitations="Sampled rigid geometry only; no continuous collision proof, contact solver, elastic cardboard, frictional retention or crease-strength model.",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", type=Path, default=ROOT / "mailer.urdf")
    ap.add_argument("--backend", choices=["urdfpy", "reference"], default="urdfpy")
    ap.add_argument("--samples", type=int, default=2001)
    ap.add_argument("--report", type=Path, default=ROOT / "output" / "validation.json")
    args = ap.parse_args()
    try:
        report = validate(
            Model(args.urdf, args.backend),
            Motion(args.urdf.with_suffix(".json")),
            args.samples,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        ap.exit(2, f"Error: {exc}\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
