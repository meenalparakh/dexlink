from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

from dexlink.utils.retargeting.palm_vertex_indices import (
    LEFT_PALM_VERTEX_INDICES,
    RIGHT_PALM_VERTEX_INDICES,
)
from dexlink.utils.retargeting.path_utils import get_external_data_dir

HUMAN_HAND_KPT_SPECS = (
    ("thumb_distal_kpt1", "thumb_proximal", "thumb_tip", 1.0 / 3.0),
    ("thumb_distal_kpt2", "thumb_proximal", "thumb_tip", 2.0 / 3.0),
    ("index_distal_kpt1", "index_proximal", "index_tip", 1.0 / 3.0),
    ("index_distal_kpt2", "index_proximal", "index_tip", 2.0 / 3.0),
    ("middle_distal_kpt1", "middle_proximal", "middle_tip", 1.0 / 3.0),
    ("middle_distal_kpt2", "middle_proximal", "middle_tip", 2.0 / 3.0),
    ("ring_distal_kpt1", "ring_proximal", "ring_tip", 1.0 / 3.0),
    ("ring_distal_kpt2", "ring_proximal", "ring_tip", 2.0 / 3.0),
    ("little_distal_kpt1", "little_proximal", "little_tip", 1.0 / 3.0),
    ("little_distal_kpt2", "little_proximal", "little_tip", 2.0 / 3.0),
    ("thumb_proximal_kpt1", "thumb_base", "thumb_proximal", 1.0 / 3.0),
    ("thumb_proximal_kpt2", "thumb_base", "thumb_proximal", 2.0 / 3.0),
    ("index_proximal_kpt1", "index_base", "index_proximal", 1.0 / 3.0),
    ("index_proximal_kpt2", "index_base", "index_proximal", 2.0 / 3.0),
    ("middle_proximal_kpt1", "middle_base", "middle_proximal", 1.0 / 3.0),
    ("middle_proximal_kpt2", "middle_base", "middle_proximal", 2.0 / 3.0),
    ("ring_proximal_kpt1", "ring_base", "ring_proximal", 1.0 / 3.0),
    ("ring_proximal_kpt2", "ring_base", "ring_proximal", 2.0 / 3.0),
    ("little_proximal_kpt1", "little_base", "little_proximal", 1.0 / 3.0),
    ("little_proximal_kpt2", "little_base", "little_proximal", 2.0 / 3.0),
    ("index_palm_kpt1", "wrist", "index_base", 0.5),
    ("middle_palm_kpt1", "wrist", "middle_base", 0.5),
    ("little_palm_kpt1", "wrist", "little_base", 0.5),
)

MANO_ORIGINAL_JOINT_MAP = {
    "wrist": 0,
    "index_base": 1,
    "index_proximal": 2,
    "middle_base": 4,
    "middle_proximal": 5,
    "little_base": 7,
    "little_proximal": 8,
    "ring_base": 10,
    "ring_proximal": 11,
    "thumb_base": 13,
    "thumb_proximal": 14,
}

MANO_FINGERTIP_VERTEX_IDS = {
    "thumb_tip": 745,
    "index_tip": 317,
    "middle_tip": 444,
    "ring_tip": 556,
    "little_tip": 673,
}

HUMAN_HAND_KPT_FACE_CACHE_VERSION = 2


@lru_cache(maxsize=2)
def load_mano_model_data(side: str) -> dict:
    if side not in ("left", "right"):
        raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")
    filename = "MANO_LEFT.pkl" if side == "left" else "MANO_RIGHT.pkl"
    path = get_external_data_dir() / "mano" / "mano_v1_2" / "models" / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing licensed MANO model at {path}. Download MANO v1.2 from "
            "its official provider and place it under "
            "external_data/mano/mano_v1_2/models/."
        )
    with Path(path).open("rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


@lru_cache(maxsize=2)
def load_mano_faces(side: str) -> np.ndarray:
    data = load_mano_model_data(side)
    faces = np.asarray(data["f"], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected MANO faces shape (F, 3), got {faces.shape}.")
    return faces


def human_hand_kpt_face_cache_path() -> Path:
    return get_external_data_dir() / "mano" / "human_hand_kpt_face_indices.json"


def _palm_vertex_indices(side: str) -> np.ndarray:
    if side == "left":
        return LEFT_PALM_VERTEX_INDICES
    if side == "right":
        return RIGHT_PALM_VERTEX_INDICES
    raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")


def _ray_first_hit_face(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    name: str,
) -> int:
    triangles = vertices[faces]
    v0 = triangles[:, 0]
    edge1 = triangles[:, 1] - v0
    edge2 = triangles[:, 2] - v0
    pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    det = np.sum(edge1 * pvec, axis=1)
    nonparallel = np.abs(det) > 1e-10

    inv_det = np.zeros_like(det)
    inv_det[nonparallel] = 1.0 / det[nonparallel]
    tvec = origin[None, :] - v0
    u = inv_det * np.sum(tvec * pvec, axis=1)
    qvec = np.cross(tvec, edge1)
    v = inv_det * np.sum(direction[None, :] * qvec, axis=1)
    t = inv_det * np.sum(edge2 * qvec, axis=1)

    hit_mask = (
        nonparallel
        & (u >= -1e-9)
        & (v >= -1e-9)
        & ((u + v) <= 1.0 + 1e-9)
        & (t >= 1e-9)
    )
    if not np.any(hit_mask):
        raise ValueError(
            f"No MANO inner-side face hit for {name!r} from origin "
            f"{origin.tolist()} along {direction.tolist()}."
        )

    hit_face_indices = np.nonzero(hit_mask)[0]
    return int(hit_face_indices[np.argmin(t[hit_mask])])


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-12
    normalized = np.zeros_like(normals)
    normalized[valid] = normals[valid] / norms[valid, None]
    return normalized


def _inner_side_face(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    centerline: np.ndarray,
    direction: np.ndarray,
    name: str,
) -> int:
    try:
        return _ray_first_hit_face(
            vertices=vertices,
            faces=faces,
            origin=centerline,
            direction=direction,
            name=name,
        )
    except ValueError:
        pass

    centroids = np.mean(vertices[faces], axis=1)
    offsets = centroids - centerline[None, :]
    projected = offsets @ direction
    candidate_mask = projected > 0.0
    if not np.any(candidate_mask):
        raise ValueError(
            f"No MANO inner-side face candidate for {name!r} from centerline "
            f"{centerline.tolist()} along {direction.tolist()}."
        )

    candidate_indices = np.nonzero(candidate_mask)[0]
    candidate_offsets = offsets[candidate_indices]
    face_normals = _face_normals(vertices, faces)[candidate_indices]
    normal_penalty = np.maximum(face_normals @ direction, 0.0)
    distance_sq = np.sum(candidate_offsets * candidate_offsets, axis=1)
    score = distance_sq + 1e-4 * normal_penalty
    return int(candidate_indices[np.argmin(score)])


def _face_indices_from_semantic_joints(
    *,
    side: str,
    vertices: np.ndarray,
    joint_positions: dict[str, np.ndarray],
) -> dict[str, int]:
    if side not in ("left", "right"):
        raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices shape (V, 3), got {vertices.shape}.")

    faces = load_mano_faces(side)
    if int(np.max(faces)) >= vertices.shape[0]:
        raise ValueError(
            f"MANO faces reference vertex {int(np.max(faces))}, but vertices have "
            f"length {vertices.shape[0]}."
        )

    palm_indices = _palm_vertex_indices(side)
    if int(np.max(palm_indices)) >= vertices.shape[0]:
        raise ValueError(
            f"Palm vertex indices reference vertex {int(np.max(palm_indices))}, "
            f"but vertices have length {vertices.shape[0]}."
        )
    inner_palm_center = np.mean(vertices[palm_indices], axis=0)

    face_indices = {}
    for name, start_joint, end_joint, fraction in HUMAN_HAND_KPT_SPECS:
        missing = [j for j in (start_joint, end_joint) if j not in joint_positions]
        if missing:
            raise ValueError(f"Missing joint positions for {name}: {missing}.")
        start = joint_positions[start_joint]
        end = joint_positions[end_joint]
        centerline = start + float(fraction) * (end - start)
        direction = inner_palm_center - centerline
        direction_norm = np.linalg.norm(direction)
        if direction_norm <= 1e-10:
            raise ValueError(
                f"Cannot compute inner-side direction for {name}; centerline point "
                "equals inner palm center."
            )
        direction = direction / direction_norm
        face_indices[name] = _inner_side_face(
            vertices=vertices,
            faces=faces,
            centerline=centerline,
            direction=direction,
            name=name,
        )

    return face_indices


def compute_human_hand_kpt_face_indices(
    *,
    side: str,
    vertices: np.ndarray,
    joints: np.ndarray,
    hand_joint_map: dict[str, int],
) -> dict[str, int]:
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError(f"Expected joints shape (J, 3), got {joints.shape}.")
    joint_positions = {}
    for name, joint_idx in hand_joint_map.items():
        joint_positions[name] = joints[int(joint_idx)]
    return _face_indices_from_semantic_joints(
        side=side,
        vertices=vertices,
        joint_positions=joint_positions,
    )


def compute_canonical_human_hand_kpt_face_indices(side: str) -> dict[str, int]:
    if side not in ("left", "right"):
        raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")
    data = load_mano_model_data(side)
    vertices = np.asarray(data["v_template"], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"Expected MANO v_template shape (V, 3), got {vertices.shape}."
        )

    j_regressor = data["J_regressor"]
    if hasattr(j_regressor, "toarray"):
        j_regressor = j_regressor.toarray()
    mano_joints = np.asarray(j_regressor, dtype=np.float64) @ vertices
    if mano_joints.ndim != 2 or mano_joints.shape[1] != 3:
        raise ValueError(
            f"Expected canonical MANO joint positions shape (J, 3), got "
            f"{mano_joints.shape}."
        )

    joint_positions = {
        name: mano_joints[joint_idx]
        for name, joint_idx in MANO_ORIGINAL_JOINT_MAP.items()
    }
    for name, vertex_idx in MANO_FINGERTIP_VERTEX_IDS.items():
        if vertex_idx >= vertices.shape[0]:
            raise ValueError(
                f"MANO fingertip vertex {vertex_idx} for {name!r} is out of range "
                f"for vertex count {vertices.shape[0]}."
            )
        joint_positions[name] = vertices[vertex_idx]

    return _face_indices_from_semantic_joints(
        side=side,
        vertices=vertices,
        joint_positions=joint_positions,
    )


def write_precomputed_human_hand_kpt_face_indices(
    path: str | Path | None = None,
) -> Path:
    output_path = human_hand_kpt_face_cache_path() if path is None else Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    side_face_indices = {
        side: compute_canonical_human_hand_kpt_face_indices(side)
        for side in ("left", "right")
    }
    side_face_vertex_indices = {}
    for side, face_indices in side_face_indices.items():
        faces = load_mano_faces(side)
        side_face_vertex_indices[side] = {
            name: faces[int(face_idx)].astype(int).tolist()
            for name, face_idx in face_indices.items()
        }

    payload = {
        "version": HUMAN_HAND_KPT_FACE_CACHE_VERSION,
        "source": "MANO v1.2 v_template canonical hand",
        "sides": side_face_indices,
        "face_vertex_indices": side_face_vertex_indices,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def _load_human_hand_kpt_cache_payload() -> dict:
    path = human_hand_kpt_face_cache_path()
    if not path.exists():
        write_precomputed_human_hand_kpt_face_indices(path)

    payload = json.loads(path.read_text())
    version = int(payload.get("version", -1))
    if version != HUMAN_HAND_KPT_FACE_CACHE_VERSION:
        raise ValueError(
            f"Expected human hand kpt face cache version "
            f"{HUMAN_HAND_KPT_FACE_CACHE_VERSION}, got {version} in {path}. "
            "Regenerate this persistent asset cache in an environment with MANO "
            "assets available by running: python -m dexlink.src.retargeting.data.human_hand_kpts"
        )
    return payload


@lru_cache(maxsize=2)
def load_precomputed_human_hand_kpt_face_indices(side: str) -> dict[str, int]:
    if side not in ("left", "right"):
        raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")
    payload = _load_human_hand_kpt_cache_payload()
    sides = payload.get("sides")
    path = human_hand_kpt_face_cache_path()
    if not isinstance(sides, dict) or side not in sides:
        raise ValueError(f"Missing side {side!r} in human hand kpt face cache {path}.")
    face_indices = {name: int(face_idx) for name, face_idx in sides[side].items()}
    expected_names = {spec[0] for spec in HUMAN_HAND_KPT_SPECS}
    missing_names = sorted(expected_names - set(face_indices))
    if missing_names:
        raise ValueError(
            f"Human hand kpt face cache {path} is missing names: {missing_names}."
        )
    return face_indices


@lru_cache(maxsize=2)
def load_precomputed_human_hand_kpt_face_vertex_indices(
    side: str,
) -> dict[str, np.ndarray]:
    if side not in ("left", "right"):
        raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")
    payload = _load_human_hand_kpt_cache_payload()
    all_face_vertex_indices = payload.get("face_vertex_indices")
    path = human_hand_kpt_face_cache_path()
    if (
        not isinstance(all_face_vertex_indices, dict)
        or side not in all_face_vertex_indices
    ):
        raise ValueError(
            f"Missing side {side!r} in human hand keypoint face-vertex cache {path}."
        )
    face_vertex_indices = {
        name: np.asarray(value, dtype=np.int64)
        for name, value in all_face_vertex_indices[side].items()
    }
    expected_names = {spec[0] for spec in HUMAN_HAND_KPT_SPECS}
    missing_names = sorted(expected_names - set(face_vertex_indices))
    if missing_names:
        raise ValueError(
            f"Human hand kpt face-vertex cache {path} is missing names: {missing_names}."
        )
    for name, value in face_vertex_indices.items():
        if value.shape != (3,):
            raise ValueError(
                f"Expected cached face vertex indices for {side}/{name} to have "
                f"shape (3,), got {value.shape}."
            )
    return face_vertex_indices


def human_hand_kpt_positions_from_faces(
    *,
    side: str,
    vertices: np.ndarray,
    face_indices: dict[str, int],
    names: list[str] | None = None,
    frame_idx: int | None = None,
) -> dict[str, np.ndarray]:
    vertices = np.asarray(vertices)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"Expected vertices shape (T, V, 3), got {vertices.shape}.")
    faces = load_mano_faces(side)
    query_names = list(face_indices.keys()) if names is None else list(names)

    result = {}
    for name in query_names:
        if name not in face_indices:
            raise KeyError(f"Unknown human hand kpt name {name!r}.")
        face = faces[int(face_indices[name])]
        positions = np.mean(vertices[:, face, :], axis=1)
        result[name] = positions if frame_idx is None else positions[int(frame_idx)]
    return result


def human_hand_kpt_positions_from_face_vertices(
    *,
    vertices: np.ndarray,
    face_vertex_indices: dict[str, np.ndarray],
    names: list[str] | None = None,
    frame_idx: int | None = None,
) -> dict[str, np.ndarray]:
    vertices = np.asarray(vertices)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"Expected vertices shape (T, V, 3), got {vertices.shape}.")
    query_names = list(face_vertex_indices.keys()) if names is None else list(names)

    result = {}
    for name in query_names:
        if name not in face_vertex_indices:
            raise KeyError(f"Unknown human hand kpt name {name!r}.")
        face = np.asarray(face_vertex_indices[name], dtype=np.int64)
        if face.shape != (3,):
            raise ValueError(
                f"Expected face vertex indices for {name!r} to have shape (3,), "
                f"got {face.shape}."
            )
        if int(np.max(face)) >= vertices.shape[1]:
            raise ValueError(
                f"Face vertex indices for {name!r} reference vertex "
                f"{int(np.max(face))}, but vertices have length {vertices.shape[1]}."
            )
        positions = np.mean(vertices[:, face, :], axis=1)
        result[name] = positions if frame_idx is None else positions[int(frame_idx)]
    return result


if __name__ == "__main__":
    output_path = write_precomputed_human_hand_kpt_face_indices()
    print(f"Wrote human hand keypoint face cache: {output_path}")
