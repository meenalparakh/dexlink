from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from urdfpy import URDF


@dataclass(frozen=True)
class RetargetSolveTiming:
    main_solve_time_sec: float | None
    solve_attempts: int
    eval_count: int | None


def _load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh from {path}, got {type(mesh).__name__}.")
    if mesh.is_empty or mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError(f"Expected non-empty triangle mesh at {path}.")
    return mesh


def _export_plain_obj(mesh: trimesh.Trimesh, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plain_mesh = mesh.copy()
    plain_mesh.visual = trimesh.visual.ColorVisuals(mesh=plain_mesh)
    plain_mesh.export(output_path)


def _pose_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose matrix, got {pose.shape}.")
    return pose


def export_retarget_probe_objs(
    *,
    output_dir: str | Path,
    target_obj_path: str | Path,
    target_pose: np.ndarray,
    robot_urdf_path: str | Path,
    dof_names: list[str],
    q_solution: np.ndarray,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    q_solution = np.asarray(q_solution, dtype=float).reshape(-1)
    if q_solution.shape != (len(dof_names),):
        raise ValueError(
            f"Expected q_solution shape ({len(dof_names)},), got {q_solution.shape}."
        )

    object_mesh = _load_mesh(target_obj_path)
    object_mesh.apply_transform(_pose_matrix(target_pose))
    object_obj_path = output_dir / "retgt_target_object.obj"
    _export_plain_obj(object_mesh, object_obj_path)

    robot_urdf = URDF.load(str(robot_urdf_path))
    available_joint_names = {joint.name for joint in robot_urdf.actuated_joints}
    q_by_name = {
        name: float(value) for name, value in zip(dof_names, q_solution.tolist())
    }
    missing_joint_names = sorted(available_joint_names - set(q_by_name))
    if missing_joint_names:
        raise ValueError(
            f"Robot URDF expects joints missing from q_solution: {missing_joint_names}."
        )
    joint_cfg = {
        name: value
        for name, value in q_by_name.items()
        if name in available_joint_names
    }
    fk = robot_urdf.visual_trimesh_fk(cfg=joint_cfg)
    hand_pieces = []
    for mesh, transform in fk.items():
        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(np.asarray(transform, dtype=float))
        hand_pieces.append(transformed_mesh)
    if not hand_pieces:
        raise ValueError(f"No visual meshes found in robot URDF: {robot_urdf_path}")
    hand_mesh = trimesh.util.concatenate(hand_pieces)
    hand_obj_path = output_dir / "retgt_solved_hand.obj"
    _export_plain_obj(hand_mesh, hand_obj_path)

    overlay_mesh = trimesh.util.concatenate([object_mesh, hand_mesh])
    overlay_obj_path = output_dir / "retgt_solved_overlay.obj"
    _export_plain_obj(overlay_mesh, overlay_obj_path)

    return {
        "target_object_obj": str(object_obj_path),
        "solved_hand_obj": str(hand_obj_path),
        "solved_overlay_obj": str(overlay_obj_path),
    }


def write_retarget_timing_json(
    *,
    output_dir: str | Path,
    timing: RetargetSolveTiming,
    extra: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_dir) / "retgt_timing.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = asdict(timing)
    if extra is not None:
        payload.update(extra)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
