from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R


def parse_transform_scalar(value: float | int | str, field_name: str) -> float:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if isinstance(value, str):
        expr = value.strip()
        if not expr:
            raise ValueError(f"`{field_name}` contains an empty string expression.")
        try:
            return float(eval(expr, {"__builtins__": {}}, {"pi": np.pi}))
        except Exception as exc:
            raise ValueError(
                f"Failed to parse `{field_name}` expression {value!r}."
            ) from exc
    raise TypeError(
        f"`{field_name}` entries must be numeric or string expressions, got {type(value)}."
    )


def parse_transform_vector(
    values: object, expected_len: int, field_name: str
) -> np.ndarray:
    if not isinstance(values, (list, tuple)):
        raise TypeError(
            f"`{field_name}` must be a length-{expected_len} list or tuple, got {type(values)}."
        )
    if len(values) != expected_len:
        raise ValueError(
            f"`{field_name}` must have length {expected_len}, got {len(values)}."
        )
    parsed = [parse_transform_scalar(v, field_name) for v in values]
    return np.asarray(parsed, dtype=np.float64)


def parse_transform_matrix(
    values: object, shape: tuple[int, int], field_name: str
) -> np.ndarray:
    if not isinstance(values, (list, tuple)):
        raise TypeError(
            f"`{field_name}` must be a {shape[0]}x{shape[1]} nested list, got {type(values)}."
        )
    if len(values) != shape[0]:
        raise ValueError(
            f"`{field_name}` must have {shape[0]} rows, got {len(values)}."
        )
    rows = [parse_transform_vector(row, shape[1], field_name) for row in values]
    return np.stack(rows, axis=0)


def load_target_transform(
    target_transform_info: str | Path | None,
) -> np.ndarray | None:
    if target_transform_info is None:
        return None

    transform_path = Path(target_transform_info).resolve()
    if not transform_path.exists():
        raise FileNotFoundError(f"Target transform file not found: {transform_path}")

    raw_text = transform_path.read_text(encoding="utf-8")
    sanitized_text = re.sub(r",(\s*[}\]])", r"\1", raw_text)
    try:
        payload = json.loads(sanitized_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse target transform JSON at {transform_path}."
        ) from exc

    if not isinstance(payload, dict):
        raise TypeError(
            f"Target transform file must contain a JSON object, got {type(payload)}."
        )

    if "matrix" in payload:
        if len(payload) != 1:
            raise ValueError(
                "Target transform JSON with `matrix` must not define any other keys."
            )
        return parse_transform_matrix(payload["matrix"], (3, 3), "matrix").astype(
            np.float64, copy=False
        )

    rotation_keys = [key for key in ("euler_xyz", "quat_wxyz") if key in payload]
    if len(rotation_keys) > 1:
        raise ValueError(
            "Target transform JSON must define only one rotation representation, got "
            f"{rotation_keys}."
        )
    if not rotation_keys:
        raise ValueError(
            "Target transform JSON must define either `matrix` or one of "
            "`euler_xyz`, `quat_wxyz`."
        )

    if "euler_xyz" in payload:
        euler_xyz = parse_transform_vector(payload["euler_xyz"], 3, "euler_xyz")
        rotation = R.from_euler("xyz", euler_xyz, degrees=False).as_matrix()
    else:
        quat_wxyz = parse_transform_vector(payload["quat_wxyz"], 4, "quat_wxyz")
        rotation = R.from_quat(np.roll(quat_wxyz, -1)).as_matrix()

    allowed_keys = set(rotation_keys)
    extra_keys = set(payload) - allowed_keys
    if extra_keys:
        raise ValueError(
            "Unsupported keys in target transform JSON: "
            f"{sorted(extra_keys)}. Expected exactly one of `euler_xyz` or `quat_wxyz`."
        )
    return rotation.astype(np.float64, copy=False)


def rotation_matrix_from_euler_xyz(euler_xyz: np.ndarray) -> np.ndarray:
    euler_xyz = np.asarray(euler_xyz, dtype=np.float64)
    if euler_xyz.shape != (3,):
        raise ValueError(f"`euler_xyz` must have shape (3,), got {euler_xyz.shape}.")
    return R.from_euler("xyz", euler_xyz, degrees=False).as_matrix()


def euler_xyz_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"`rotation` must have shape (3, 3), got {rotation.shape}.")
    return R.from_matrix(rotation).as_euler("xyz", degrees=False)


def apply_target_mesh_transform(
    mesh: trimesh.Trimesh,
    transform: np.ndarray | None = None,
    scale: float = 1.0,
    center: np.ndarray | None = None,
) -> trimesh.Trimesh:
    if scale <= 0.0:
        raise ValueError(f"`scale` must be positive, got {scale}.")
    transformed_mesh = mesh.copy()
    if (transform is None) and (scale == 1.0):
        return transformed_mesh

    if transform is None:
        rotation = np.eye(3, dtype=np.float64)
    else:
        rotation = np.asarray(transform, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(
                f"`transform` must have shape (3, 3), got {rotation.shape}."
            )

    if center is None:
        center = transformed_mesh.vertices.mean(axis=0)
    else:
        center = np.asarray(center, dtype=np.float64)
        if center.shape != (3,):
            raise ValueError(f"`center` must have shape (3,), got {center.shape}.")

    transformed_mesh.vertices = transformed_mesh.vertices - center.reshape(1, 3)
    transformed_mesh.vertices = transformed_mesh.vertices * float(scale)
    transformed_mesh.vertices = transformed_mesh.vertices @ rotation.T
    return transformed_mesh
