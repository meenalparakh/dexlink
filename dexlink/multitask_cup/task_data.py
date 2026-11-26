"""Immutable approved trajectories, physical geometry and ordered task programs."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from urdfpy import URDF

from .geometry_policy import PHASE_KINDS
from .task_catalog import file_sha256, geometry_groups, validate_training_catalog

PHASE_KIND = {
    "reach": "reach",
    "grasp": "grasp",
    "lift": "lift",
    "hold_lift": "hold",
    "tilt": "rotate",
    "hold_tilt": "hold",
    "return_upright": "rotate",
    "translate_to_goal": "translate",
    "lower": "lower",
    "release": "release",
    "retreat": "retreat",
    "visit_1": "rotate",
    "visit_2": "rotate",
    "visit_3": "rotate",
    "visit_4": "rotate",
    "hold_visit_1": "hold",
    "hold_visit_2": "hold",
    "hold_visit_3": "hold",
    "hold_visit_4": "hold",
}
ROOT_JOINTS = tuple(
    f"R_forearm_{axis}_link_joint"
    for axis in ("tx", "ty", "tz", "roll", "pitch", "yaw")
)


def pose_from_matrix(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrix[..., :3, :3]).as_quat()
    return np.concatenate((matrix[..., :3, 3], xyzw[..., [3, 0, 1, 2]]), axis=-1)


def geometry_arrays(path: Path, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    robot = URDF.load(str(path))
    if len(robot.links) != 1 or robot.joints:
        raise ValueError(f"Expected a single rigid cup link with no joints: {path}.")
    triangles = []
    for mesh, transform in robot.visual_trimesh_fk().items():
        vertices = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
        triangles.append(vertices[mesh.faces])
    triangles = np.concatenate(triangles)
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = np.linalg.norm(cross, axis=-1) / 2
    positive = areas > 0
    triangles, cross, areas = triangles[positive], cross[positive], areas[positive]
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(areas), size=count, p=areas / areas.sum())
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, 0:1])
    weights = np.concatenate(
        (1 - root, root * (1 - uv[:, 1:2]), root * uv[:, 1:2]), axis=1
    )
    xyz = (triangles[selected] * weights[..., None]).sum(1)
    normals = cross[selected] / (2 * areas[selected, None])
    vertices = triangles.reshape(-1, 3)
    lower, upper = vertices.min(0), vertices.max(0)
    inertial = robot.links[0].inertial
    if inertial is None:
        raise ValueError(
            f"Expected explicit approved cup mass, COM and inertia in {path}."
        )
    rotation = inertial.origin[:3, :3]
    inertia = rotation @ inertial.inertia @ rotation.T
    metadata = np.concatenate(
        (
            upper - lower,
            (upper + lower) / 2,
            [inertial.mass],
            inertial.origin[:3, 3],
            inertia[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)],
        )
    )
    return np.concatenate((xyz, normals), axis=1), metadata


@dataclass(frozen=True)
class TaskBank:
    wrist: torch.Tensor
    fingers: torch.Tensor
    joints: torch.Tensor
    cup: torch.Tensor
    lengths: torch.Tensor
    phase_lengths: torch.Tensor
    phase_ends: torch.Tensor
    program_tokens: torch.Tensor
    program_mask: torch.Tensor
    geometry_points: torch.Tensor
    geometry_metadata: torch.Tensor
    geometry_index: torch.Tensor
    finger_lower: torch.Tensor
    finger_upper: torch.Tensor

    def to(self, device: torch.device | str) -> TaskBank:
        return TaskBank(
            **{item.name: getattr(self, item.name).to(device) for item in fields(self)}
        )

    @classmethod
    def load(cls, path: Path, device: torch.device | str = "cpu") -> TaskBank:
        return cls(**torch.load(path, map_location=device, weights_only=True))


def build_bank(manifest_path: Path, output: Path, point_count: int = 512) -> dict:
    manifest = json.loads(manifest_path.read_text())
    variants = manifest["variants"]
    task_count = validate_training_catalog(manifest)
    robot_path = Path(manifest["robot_urdf"])
    if file_sha256(robot_path) != manifest["robot_urdf_sha256"]:
        raise ValueError("Robot URDF differs from the approved source manifest.")
    robot = URDF.load(str(robot_path))
    # The exporter and live Hand use URDF declaration order, not urdfpy's topological order.
    joints = [joint for joint in robot.joints if joint.joint_type != "fixed"]
    names = [joint.name for joint in joints]
    if len(names) != 18 or not set(ROOT_JOINTS).issubset(names):
        raise ValueError(
            f"Expected the declared 18-DOF floating right XHand, received {names}."
        )
    finger_indices = [i for i, name in enumerate(names) if name not in ROOT_JOINTS]
    lower = np.array([joints[i].limit.lower for i in finger_indices])
    upper = np.array([joints[i].limit.upper for i in finger_indices])
    if np.any(upper <= lower):
        raise ValueError("Expected finite increasing limits for all 12 finger joints.")
    maximum = max(row["frames"] for row in variants)
    shape = (task_count, maximum)
    data = {
        "wrist": np.zeros((*shape, 7)),
        "fingers": np.zeros((*shape, 12)),
        "joints": np.zeros((*shape, 18)),
        "cup": np.zeros((*shape, 7)),
        "lengths": np.array([row["frames"] for row in variants], dtype=np.int64),
        "phase_lengths": np.zeros((task_count, 12), dtype=np.int64),
        "phase_ends": np.zeros((task_count, 12), dtype=np.int64),
        "program_tokens": np.zeros((task_count, 12, 32)),
        "program_mask": np.zeros((task_count, 12), dtype=bool),
        "finger_lower": lower,
        "finger_upper": upper,
    }
    # FK uses the actual robot tree, avoiding assumptions about the six floating joint axes.
    for index, row in enumerate(variants):
        path = Path(row["reference"])
        if file_sha256(path) != row["reference_sha256"]:
            raise ValueError(f"Reference hash differs for {row['id']}.")
        payload = np.load(path, allow_pickle=True).item()
        if list(payload["right_jnames"]) != names or payload["right_jvals"].shape != (
            row["frames"],
            18,
        ):
            raise ValueError(f"Unexpected robot joint order/shape for {row['id']}.")
        if payload["control_period_seconds"] != 1 / 15:
            raise ValueError("Expected the approved 15 Hz reference clock.")
        q = payload["right_jvals"]
        transforms = robot.link_fk_batch(
            cfgs={name: q[:, i] for i, name in enumerate(names)}
        )
        wrist = pose_from_matrix(transforms[robot.link_map["right_hand_link"]])
        cup = np.concatenate(
            (payload["reference_obj_pos"], payload["reference_obj_quat"]), axis=-1
        )
        T = row["frames"]
        for key, value in (
            ("wrist", wrist),
            ("fingers", q[:, finger_indices]),
            ("joints", q),
            ("cup", cup),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite {key} in {row['id']}.")
            data[key][index, :T] = value
            data[key][index, T:] = value[-1]
        lengths = np.array(row["phase_frames"])
        P = len(lengths)
        if P > 12 or np.any(lengths <= 0) or lengths.sum() != T:
            raise ValueError(f"Invalid approved phase lengths for {row['id']}.")
        ends = lengths.cumsum(0)
        data["phase_lengths"][index, :P] = lengths
        data["phase_ends"][index] = T
        data["phase_ends"][index, :P] = ends
        data["program_mask"][index, :P] = True
        rotation0 = Rotation.from_quat(cup[0, [4, 5, 6, 3]]).as_matrix()
        signed_angle = 0.0
        for phase, (name, end) in enumerate(zip(row["phase_names"], ends)):
            if name not in PHASE_KIND:
                raise ValueError(f"Unknown phase name {name!r} in {row['id']}.")
            if name == "tilt":
                signed_angle = np.deg2rad(row["tilt_degrees"])
            elif name in {"visit_1", "visit_2", "visit_3", "visit_4"}:
                signed_angle = np.deg2rad(row["sequence_degrees"][int(name[-1]) - 1])
            elif name == "return_upright":
                signed_angle = 0.0
            endpoint = cup[end - 1]
            rotation = (
                rotation0.T @ Rotation.from_quat(endpoint[[4, 5, 6, 3]]).as_matrix()
            )
            token = data["program_tokens"][index, phase]
            token[PHASE_KINDS.index(PHASE_KIND[name])] = 1
            token[9] = lengths[phase] / 150
            token[10:13] = rotation0.T @ (endpoint[:3] - cup[0, :3]) / 0.1
            token[13:19] = rotation[:, :2].T.reshape(6)
            token[19:31] = (
                2 * (q[end - 1, finger_indices] - lower) / (upper - lower) - 1
            )
            token[31] = signed_angle
    groups = geometry_groups(variants)
    points, metadata = zip(
        *(
            geometry_arrays(
                Path(variants[group.task_indices[0]]["scene_urdfs"][0]), point_count, 42
            )
            for group in groups
        )
    )
    data["geometry_points"], data["geometry_metadata"] = (
        np.stack(points),
        np.stack(metadata),
    )
    data["geometry_index"] = np.empty(task_count, dtype=np.int64)
    for group in groups:
        data["geometry_index"][list(group.task_indices)] = group.geometry_index
        expected = variants[group.task_indices[0]]["geometry_extent_xyz"]
        if not np.allclose(
            metadata[group.geometry_index][:3], expected, atol=2e-6, rtol=0
        ):
            raise ValueError(
                f"Mesh extents differ from source manifest for geometry {group.geometry_index}."
            )
    tensors = {
        key: torch.from_numpy(value).to(torch.float32)
        if value.dtype.kind == "f"
        else torch.from_numpy(value)
        for key, value in data.items()
    }
    torch.save(tensors, output)
    return {
        "tasks": task_count,
        "geometries": len(groups),
        "maximum_frames": maximum,
        "phase_counts": sorted(set(len(v["phase_frames"]) for v in variants)),
        "finger_indices": finger_indices,
        "joint_names": names,
        "manifest_sha256": file_sha256(manifest_path),
        "bank_sha256": file_sha256(output),
        "bank_path": str(output.resolve()),
    }
