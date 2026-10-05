from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from urdfpy import URDF

from dexlink.src.retargeting.data.hand_cfgs.franka_xhand import XHAND_CFGs as FRANKA_XHAND_CFGs
from dexlink.src.retargeting.data.hand_cfgs.inspire import INSPIRE_CFGs
from dexlink.src.retargeting.data.hand_cfgs.xhand import XHAND_CFGs
from dexlink.utils.retargeting.path_utils import process_urdf


@dataclass(frozen=True)
class TemplateState:
    template_id: str
    active_side: str
    robot_model: str
    left_joint_names: list[str]
    left_joint_vals: np.ndarray
    right_joint_names: list[str]
    right_joint_vals: np.ndarray
    object_name: str
    object_urdf_path: str
    object_pos: np.ndarray
    object_quat_wxyz: np.ndarray
    object_arti_vals: np.ndarray
    object_arti_names: list[str]
    metadata: dict
    human_skeleton_keypoint_names: list[str] | None = None
    human_bps_keypoint_names: list[str] | None = None
    left_human_skeleton_keypoint_pos: np.ndarray | None = None
    right_human_skeleton_keypoint_pos: np.ndarray | None = None
    left_human_bps_keypoint_pos: np.ndarray | None = None
    right_human_bps_keypoint_pos: np.ndarray | None = None
    left_human_skeleton_keypoint_valid: bool = False
    right_human_skeleton_keypoint_valid: bool = False
    left_human_bps_keypoint_valid: bool = False
    right_human_bps_keypoint_valid: bool = False

    @property
    def active_joint_names(self) -> list[str]:
        return (
            self.right_joint_names
            if self.active_side == "right"
            else self.left_joint_names
        )

    @property
    def active_joint_vals(self) -> np.ndarray:
        return (
            self.right_joint_vals
            if self.active_side == "right"
            else self.left_joint_vals
        )

    @property
    def active_sides(self) -> tuple[str, ...]:
        active = self.metadata.get("active_sides")
        if active is None:
            return (self.active_side,)
        sides = tuple(str(side) for side in active)
        if len(sides) == 0 or any(side not in ("left", "right") for side in sides):
            raise ValueError(
                f"Template {self.template_id!r} has invalid active_sides {active!r}."
            )
        return sides


def _normalize_robot_model(robot_model: str) -> str:
    aliases = {
        "inspire": "inspire",
        "inspire_hand": "inspire",
        "xhand": "xhand",
        "franka": "franka_xhand",
        "franka_xhand": "franka_xhand",
        "franka-xhand": "franka_xhand",
        "panda_xhand": "franka_xhand",
    }
    value = str(robot_model).lower()
    if value not in aliases:
        raise ValueError(
            f"Unsupported robot_model {robot_model!r}; expected one of {sorted(aliases)}."
        )
    return aliases[value]


def _robot_hand_cfg(robot_model: str, side: str) -> dict:
    model = _normalize_robot_model(robot_model)
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side {side!r}; use 'left' or 'right'.")
    if model == "inspire":
        cfgs = INSPIRE_CFGs
    elif model == "xhand":
        cfgs = XHAND_CFGs
    else:
        cfgs = FRANKA_XHAND_CFGs
    return cfgs[side]


def _hand_config_side(state: TemplateState, side: str) -> str:
    source_by_side = state.metadata.get("hand_config_source_by_side", {})
    if not isinstance(source_by_side, dict):
        raise ValueError(
            "Expected hand_config_source_by_side metadata to be a dict, "
            f"got {type(source_by_side).__name__}."
        )
    source_side = str(source_by_side.get(side, side))
    if source_side not in ("left", "right"):
        raise ValueError(
            "Expected hand config source side to be 'left' or 'right', "
            f"got {source_side!r} for role side {side!r}."
        )
    return source_side


def _as_array_dict(npz_path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _to_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _frame_index(num_frames: int, frame_spec: str | int) -> int:
    if frame_spec == "last":
        return num_frames - 1
    idx = int(frame_spec)
    if idx < 0 or idx >= num_frames:
        raise ValueError(f"Frame index {idx} out of bounds for {num_frames} frames.")
    return idx


def _find_index(joint_names: list[str], suffix: str) -> int:
    matches = [idx for idx, name in enumerate(joint_names) if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one joint ending with {suffix!r}, found {matches}."
        )
    return matches[0]


def _root_indices(joint_names: list[str]) -> dict[str, int]:
    return {
        "tx": _find_index(joint_names, "forearm_tx_link_joint"),
        "ty": _find_index(joint_names, "forearm_ty_link_joint"),
        "tz": _find_index(joint_names, "forearm_tz_link_joint"),
        "roll": _find_index(joint_names, "forearm_roll_link_joint"),
        "pitch": _find_index(joint_names, "forearm_pitch_link_joint"),
        "yaw": _find_index(joint_names, "forearm_yaw_link_joint"),
    }


def _quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    return np.roll(quat_wxyz, -1)


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    return np.roll(quat_xyzw, 1)


@lru_cache(maxsize=None)
def _load_robot_urdf(robot_model: str, side: str) -> URDF:
    cfg = _robot_hand_cfg(robot_model, side)
    return URDF.load(str(process_urdf(cfg["urdf_path"])))


@lru_cache(maxsize=None)
def _load_object_urdf(urdf_path: str) -> URDF:
    if not urdf_path:
        raise ValueError("Object URDF path is empty; cannot compute object link FK.")
    return URDF.load(str(process_urdf(urdf_path)))


def _articulation_joint_names_from_urdf(urdf_path: str) -> list[str]:
    urdf = _load_object_urdf(urdf_path)
    return [str(joint.name) for joint in urdf.joints if joint.joint_type != "fixed"]


@lru_cache(maxsize=None)
def _robot_wrist_link(robot_model: str, side: str):
    cfg = _robot_hand_cfg(robot_model, side)
    urdf = _load_robot_urdf(robot_model, side)
    link_name = str(cfg["wrist_link_name"])
    matches = [link for link in urdf.links if link.name == link_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one wrist link {link_name!r} in {cfg['urdf_path']}, "
            f"found {len(matches)}."
        )
    return matches[0]


def _joint_cfg(joint_names: list[str], joint_vals: np.ndarray) -> dict[str, float]:
    vals = np.asarray(joint_vals, dtype=np.float64).reshape(-1)
    if vals.shape[0] != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} joint values, got {vals.shape[0]}."
        )
    return {name: float(val) for name, val in zip(joint_names, vals.tolist())}


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected keypoint positions with shape (K, 3), got {points.shape}."
        )
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4, 4), got {transform.shape}.")
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def _load_keypoint_names(motion: dict[str, np.ndarray], key: str) -> list[str] | None:
    if key not in motion:
        return None
    return [str(name) for name in np.asarray(motion[key]).reshape(-1).tolist()]


def _load_frame_keypoints(
    motion: dict[str, np.ndarray],
    key: str,
    frame_idx: int,
) -> np.ndarray | None:
    if key not in motion:
        return None
    values = np.asarray(motion[key], dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"Expected {key} to have shape (T, K, 3), got {values.shape}.")
    return values[frame_idx].copy()


def _load_frame_valid(
    motion: dict[str, np.ndarray],
    key: str,
    frame_idx: int,
    has_keypoints: bool,
) -> bool:
    if key not in motion:
        return bool(has_keypoints)
    values = np.asarray(motion[key], dtype=bool)
    if values.ndim == 1:
        return bool(values[frame_idx])
    if values.ndim == 2:
        return bool(np.all(values[frame_idx]))
    raise ValueError(
        f"Expected {key} to have shape (T,) or (T, K), got {values.shape}."
    )


def _state_keypoint_piece(state: TemplateState, frames: int) -> dict[str, np.ndarray]:
    piece: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        for kind in ("skeleton", "bps"):
            pos = getattr(state, f"{side}_human_{kind}_keypoint_pos")
            valid = bool(getattr(state, f"{side}_human_{kind}_keypoint_valid"))
            if pos is None:
                continue
            piece[f"{side}_human_{kind}_keypoint_pos"] = np.repeat(
                np.asarray(pos, dtype=np.float32)[None, :, :],
                frames,
                axis=0,
            )
            piece[f"{side}_human_{kind}_keypoint_valid"] = np.full(
                (frames,),
                valid,
                dtype=bool,
            )
    return piece


def _interpolate_keypoint_piece(
    start_state: TemplateState,
    end_state: TemplateState,
    *,
    frames: int,
) -> dict[str, np.ndarray]:
    piece: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        for kind in ("skeleton", "bps"):
            start_pos = getattr(start_state, f"{side}_human_{kind}_keypoint_pos")
            end_pos = getattr(end_state, f"{side}_human_{kind}_keypoint_pos")
            if start_pos is None and end_pos is None:
                continue
            if start_pos is None or end_pos is None:
                raise ValueError(
                    f"Cannot interpolate {side} human {kind} keypoints when only one "
                    "template defines them."
                )
            if np.asarray(start_pos).shape != np.asarray(end_pos).shape:
                raise ValueError(
                    f"Cannot interpolate {side} human {kind} keypoints with shapes "
                    f"{np.asarray(start_pos).shape} and {np.asarray(end_pos).shape}."
                )
            piece[f"{side}_human_{kind}_keypoint_pos"] = _interpolate_joint_vectors(
                np.asarray(start_pos, dtype=np.float32),
                np.asarray(end_pos, dtype=np.float32),
                frames,
            )
            valid = bool(
                getattr(start_state, f"{side}_human_{kind}_keypoint_valid")
            ) and bool(getattr(end_state, f"{side}_human_{kind}_keypoint_valid"))
            piece[f"{side}_human_{kind}_keypoint_valid"] = np.full(
                (frames,),
                valid,
                dtype=bool,
            )
    return piece


def _active_wrist_pose(state: TemplateState, active_q: np.ndarray) -> np.ndarray:
    config_side = _hand_config_side(state, state.active_side)
    urdf = _load_robot_urdf(state.robot_model, config_side)
    wrist_link = _robot_wrist_link(state.robot_model, config_side)
    fk = urdf.link_fk(cfg=_joint_cfg(state.active_joint_names, active_q))
    if wrist_link not in fk:
        raise ValueError(f"Wrist link {wrist_link.name!r} was not returned by URDF FK.")
    return np.asarray(fk[wrist_link], dtype=np.float64)


def active_wrist_pose(
    state: TemplateState, active_joint_vals: np.ndarray | None = None
) -> np.ndarray:
    """Return the active robot wrist pose from the canonical robot URDF FK."""
    joint_vals = (
        state.active_joint_vals
        if active_joint_vals is None
        else np.asarray(active_joint_vals, dtype=np.float32)
    )
    return _active_wrist_pose(state, joint_vals)


def set_active_wrist_pose(
    state: TemplateState,
    *,
    template_id: str,
    wrist_position: np.ndarray,
    wrist_quat_wxyz: np.ndarray,
) -> TemplateState:
    """Set the floating wrist joints while preserving every finger joint value."""
    position = np.asarray(wrist_position, dtype=np.float64)
    quat_wxyz = np.asarray(wrist_quat_wxyz, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"Expected wrist_position shape (3,), got {position.shape}.")
    if quat_wxyz.shape != (4,):
        raise ValueError(
            f"Expected wrist_quat_wxyz shape (4,), got {quat_wxyz.shape}."
        )
    quat_norm = float(np.linalg.norm(quat_wxyz))
    if quat_norm <= 1e-8:
        raise ValueError("Expected a non-zero wrist quaternion.")
    target_pose = np.eye(4, dtype=np.float64)
    target_pose[:3, :3] = R.from_quat(
        _quat_wxyz_to_xyzw(quat_wxyz / quat_norm)
    ).as_matrix()
    target_pose[:3, 3] = position

    joint_names = state.active_joint_names
    start_q = np.asarray(state.active_joint_vals, dtype=np.float64)
    initial_root = _root_vector(start_q, joint_names)
    initial_root[:3] = position

    def residual(root_values: np.ndarray) -> np.ndarray:
        candidate_q = _set_root_vector(start_q, joint_names, root_values)
        pose = _active_wrist_pose(state, candidate_q)
        return np.concatenate(
            [
                pose[:3, 3] - target_pose[:3, 3],
                _rotation_error(target_pose[:3, :3], pose[:3, :3]),
            ],
            axis=0,
        )

    result = least_squares(
        residual,
        initial_root,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=200,
    )
    solved_q = _set_root_vector(start_q, joint_names, result.x).astype(np.float32)
    solved_pose = _active_wrist_pose(state, solved_q)
    translation_error = float(np.linalg.norm(solved_pose[:3, 3] - position))
    rotation_error = float(
        np.linalg.norm(_rotation_error(target_pose[:3, :3], solved_pose[:3, :3]))
    )
    if (
        not result.success
        or translation_error > 1e-5
        or rotation_error > 1e-5
    ):
        raise ValueError(
            "Could not solve the floating robot wrist joints for the requested pose. "
            f"translation_error={translation_error:.6g}, "
            f"rotation_error={rotation_error:.6g}, "
            f"optimizer_success={result.success}."
        )
    return replace(
        _replace_active_q(state, solved_q),
        template_id=template_id,
    )


def _object_base_pose(state: TemplateState) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R.from_quat(_quat_wxyz_to_xyzw(state.object_quat_wxyz)).as_matrix()
    pose[:3, 3] = np.asarray(state.object_pos, dtype=np.float64).reshape(3)
    return pose


def _object_joint_cfg(state: TemplateState, arti_vals: np.ndarray) -> dict[str, float]:
    vals = np.asarray(arti_vals, dtype=np.float64).reshape(-1)
    if vals.shape[0] != len(state.object_arti_names):
        raise ValueError(
            f"Expected {len(state.object_arti_names)} object articulation values, "
            f"got {vals.shape[0]}."
        )
    return {
        name: float(val) for name, val in zip(state.object_arti_names, vals.tolist())
    }


def _object_link_pose(
    state: TemplateState,
    *,
    link_name: str,
    arti_vals: np.ndarray,
) -> np.ndarray:
    urdf = _load_object_urdf(state.object_urdf_path)
    matches = [link for link in urdf.links if link.name == link_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one object link {link_name!r} in "
            f"{state.object_urdf_path}, found {len(matches)}."
        )
    fk = urdf.link_fk(cfg=_object_joint_cfg(state, arti_vals))
    if matches[0] not in fk:
        raise ValueError(f"Object link {link_name!r} was not returned by URDF FK.")
    return _object_base_pose(state) @ np.asarray(fk[matches[0]], dtype=np.float64)


def _transform_matrix(rotation: R, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _rotation_error(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    target_rot = R.from_matrix(np.asarray(target, dtype=np.float64))
    actual_rot = R.from_matrix(np.asarray(actual, dtype=np.float64))
    return (target_rot.inv() * actual_rot).as_rotvec()


def _root_vector(q: np.ndarray, joint_names: list[str]) -> np.ndarray:
    root = _root_indices(joint_names)
    return np.asarray(
        [
            q[root["tx"]],
            q[root["ty"]],
            q[root["tz"]],
            q[root["roll"]],
            q[root["pitch"]],
            q[root["yaw"]],
        ],
        dtype=np.float64,
    )


def _set_root_vector(
    q: np.ndarray, joint_names: list[str], root_values: np.ndarray
) -> np.ndarray:
    root_values = np.asarray(root_values, dtype=np.float64).reshape(6)
    root = _root_indices(joint_names)
    new_q = np.asarray(q, dtype=np.float64).copy()
    new_q[root["tx"]] = root_values[0]
    new_q[root["ty"]] = root_values[1]
    new_q[root["tz"]] = root_values[2]
    new_q[root["roll"]] = root_values[3]
    new_q[root["pitch"]] = root_values[4]
    new_q[root["yaw"]] = root_values[5]
    return new_q


def _metadata_link_names(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _metadata_link_names(value.item())
        return [str(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    raise ValueError(
        "Expected articulated link metadata to be a string or sequence of strings, "
        f"got {type(value).__name__}."
    )


def _infer_articulated_link_from_joint(state: TemplateState) -> str:
    if len(state.object_arti_names) != 1:
        raise ValueError(
            "Cannot infer articulated link from object joints because expected exactly "
            f"one articulation joint, got {state.object_arti_names}."
        )
    urdf = _load_object_urdf(state.object_urdf_path)
    joint_name = state.object_arti_names[0]
    matches = [joint for joint in urdf.joints if joint.name == joint_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one object joint {joint_name!r} in "
            f"{state.object_urdf_path}, found {len(matches)}."
        )
    return str(matches[0].child)


def resolve_articulated_link_name(
    state: TemplateState,
    explicit_link_name: str | None = None,
) -> str:
    if explicit_link_name is not None:
        return str(explicit_link_name)

    metadata_link_names = _metadata_link_names(
        state.metadata.get("target_bps_sampling_link_names")
    )
    if len(metadata_link_names) == 1:
        return metadata_link_names[0]
    if len(metadata_link_names) > 1:
        raise ValueError(
            "Multiple target_bps_sampling_link_names are available; specify "
            f"`articulation_link_name` explicitly. Got {metadata_link_names}."
        )
    return _infer_articulated_link_from_joint(state)


def _apply_rigid_transform_to_active_hand(
    state: TemplateState,
    transform: np.ndarray,
    *,
    translation_tol: float = 1e-4,
    rotation_tol: float = 1e-4,
) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4, 4), got {transform.shape}.")

    joint_names = state.active_joint_names
    start_q = np.asarray(state.active_joint_vals, dtype=np.float64)
    before_pose = _active_wrist_pose(state, start_q)
    target_pose = transform @ before_pose
    initial_root = _root_vector(start_q, joint_names)
    initial_root[:3] += transform[:3, 3]

    def residual(root_values: np.ndarray) -> np.ndarray:
        candidate_q = _set_root_vector(start_q, joint_names, root_values)
        pose = _active_wrist_pose(state, candidate_q)
        return np.concatenate(
            [
                pose[:3, 3] - target_pose[:3, 3],
                _rotation_error(target_pose[:3, :3], pose[:3, :3]),
            ],
            axis=0,
        )

    result = least_squares(
        residual,
        initial_root,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=200,
    )
    solved_q = _set_root_vector(start_q, joint_names, result.x).astype(np.float32)
    after_pose = _active_wrist_pose(state, solved_q)
    translation_error = float(np.linalg.norm(after_pose[:3, 3] - target_pose[:3, 3]))
    rotation_error = float(
        np.linalg.norm(_rotation_error(target_pose[:3, :3], after_pose[:3, :3]))
    )
    if (
        not result.success
        or translation_error > translation_tol
        or rotation_error > rotation_tol
    ):
        raise ValueError(
            "Could not apply rigid transform through forearm joints. "
            f"translation_error={translation_error:.6g}, "
            f"rotation_error={rotation_error:.6g}, optimizer_success={result.success}."
        )
    return solved_q


def apply_pair_rigid_transform(
    state: TemplateState,
    *,
    template_id: str,
    transform: np.ndarray,
) -> TemplateState:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4, 4), got {transform.shape}.")

    delta_rot = R.from_matrix(transform[:3, :3])
    object_rot = R.from_quat(_quat_wxyz_to_xyzw(state.object_quat_wxyz))
    new_object_rot = delta_rot * object_rot
    new_object_pos = (
        transform[:3, :3] @ np.asarray(state.object_pos, dtype=np.float64)
        + transform[:3, 3]
    )
    new_q = _apply_rigid_transform_to_active_hand(state, transform)
    keypoint_updates = {}
    for kind in ("skeleton", "bps"):
        attr = f"{state.active_side}_human_{kind}_keypoint_pos"
        points = getattr(state, attr)
        if points is not None:
            keypoint_updates[attr] = _transform_points(points, transform)
    return replace(
        _replace_active_q(state, new_q),
        template_id=template_id,
        object_pos=new_object_pos.astype(np.float32),
        object_quat_wxyz=_quat_xyzw_to_wxyz(new_object_rot.as_quat()).astype(
            np.float32
        ),
        **keypoint_updates,
    )


def set_object_articulation_model(
    state: TemplateState,
    *,
    object_urdf_path: str,
) -> TemplateState:
    joint_names = _articulation_joint_names_from_urdf(object_urdf_path)
    if state.object_arti_names:
        if state.object_arti_names != joint_names:
            raise ValueError(
                "Template articulation joints do not match target URDF joints: "
                f"template={state.object_arti_names}, target={joint_names}."
            )
        object_arti_vals = state.object_arti_vals
    else:
        object_arti_vals = np.zeros(len(joint_names), dtype=np.float32)
    if object_arti_vals.shape != (len(joint_names),):
        raise ValueError(
            "Template articulation value shape does not match target URDF joints: "
            f"values={object_arti_vals.shape}, joints={joint_names}."
        )
    return replace(
        state,
        object_urdf_path=str(object_urdf_path),
        object_arti_names=joint_names,
        object_arti_vals=object_arti_vals.astype(np.float32, copy=False),
    )


def load_template_state_from_motion(
    motion_path: str | Path,
    template_id: str,
    frame: str | int = "last",
) -> TemplateState:
    motion = _as_array_dict(motion_path)
    n_frames = int(np.asarray(motion["asset_pos"]).shape[0])
    frame_idx = _frame_index(n_frames, frame)

    active_side = str(_to_scalar(motion["traj_hand_side"]))
    robot_model = str(
        _to_scalar(motion.get("traj_robot_model", motion["traj_hand_name"]))
    )
    left_joint_names = [str(name) for name in motion["left_joint_names"].tolist()]
    right_joint_names = [str(name) for name in motion["right_joint_names"].tolist()]
    left_joint_vals = np.asarray(motion["left_joint_vals"][frame_idx], dtype=np.float32)
    right_joint_vals = np.asarray(
        motion["right_joint_vals"][frame_idx], dtype=np.float32
    )
    object_name = str(_to_scalar(motion["traj_object_name"]))
    object_urdf_path = str(_to_scalar(motion["traj_object_urdf_path"]))
    object_pos = np.asarray(motion["asset_pos"][frame_idx], dtype=np.float32)
    object_quat_wxyz = np.asarray(motion["asset_quat"][frame_idx], dtype=np.float32)
    object_arti_vals = np.asarray(
        motion["asset_arti_vals"][frame_idx], dtype=np.float32
    )
    object_arti_names = [str(name) for name in motion["asset_arti_names"].tolist()]
    metadata = {key: _to_scalar(value) for key, value in motion.items()}
    left_human_skeleton = _load_frame_keypoints(
        motion, "left_human_skeleton_keypoint_pos", frame_idx
    )
    right_human_skeleton = _load_frame_keypoints(
        motion, "right_human_skeleton_keypoint_pos", frame_idx
    )
    left_human_bps = _load_frame_keypoints(
        motion, "left_human_bps_keypoint_pos", frame_idx
    )
    right_human_bps = _load_frame_keypoints(
        motion, "right_human_bps_keypoint_pos", frame_idx
    )

    state = TemplateState(
        template_id=template_id,
        active_side=active_side,
        robot_model=robot_model,
        left_joint_names=left_joint_names,
        left_joint_vals=left_joint_vals,
        right_joint_names=right_joint_names,
        right_joint_vals=right_joint_vals,
        object_name=object_name,
        object_urdf_path=object_urdf_path,
        object_pos=object_pos,
        object_quat_wxyz=object_quat_wxyz,
        object_arti_vals=object_arti_vals,
        object_arti_names=object_arti_names,
        metadata=metadata,
        human_skeleton_keypoint_names=_load_keypoint_names(
            motion, "human_skeleton_keypoint_names"
        ),
        human_bps_keypoint_names=_load_keypoint_names(
            motion, "human_bps_keypoint_names"
        ),
        left_human_skeleton_keypoint_pos=left_human_skeleton,
        right_human_skeleton_keypoint_pos=right_human_skeleton,
        left_human_bps_keypoint_pos=left_human_bps,
        right_human_bps_keypoint_pos=right_human_bps,
        left_human_skeleton_keypoint_valid=_load_frame_valid(
            motion,
            "left_human_skeleton_keypoint_valid",
            frame_idx,
            left_human_skeleton is not None,
        ),
        right_human_skeleton_keypoint_valid=_load_frame_valid(
            motion,
            "right_human_skeleton_keypoint_valid",
            frame_idx,
            right_human_skeleton is not None,
        ),
        left_human_bps_keypoint_valid=_load_frame_valid(
            motion,
            "left_human_bps_keypoint_valid",
            frame_idx,
            left_human_bps is not None,
        ),
        right_human_bps_keypoint_valid=_load_frame_valid(
            motion,
            "right_human_bps_keypoint_valid",
            frame_idx,
            right_human_bps is not None,
        ),
    )
    return state


def combine_bimanual_template_states(
    *,
    template_id: str,
    left_state: TemplateState,
    right_state: TemplateState,
) -> TemplateState:
    if left_state.active_side != "left":
        raise ValueError(
            f"Expected left bimanual state to have active_side='left', got {left_state.active_side!r}."
        )
    if right_state.active_side != "right":
        raise ValueError(
            f"Expected right bimanual state to have active_side='right', got {right_state.active_side!r}."
        )
    if left_state.robot_model != right_state.robot_model:
        raise ValueError(
            "Cannot combine bimanual templates with different robot models: "
            f"{left_state.robot_model!r} vs {right_state.robot_model!r}."
        )
    if left_state.left_joint_names != right_state.left_joint_names:
        raise ValueError("Left joint names do not match between bimanual templates.")
    if left_state.right_joint_names != right_state.right_joint_names:
        raise ValueError("Right joint names do not match between bimanual templates.")
    if left_state.object_name != right_state.object_name:
        raise ValueError(
            "Cannot combine bimanual templates with different objects: "
            f"{left_state.object_name!r} vs {right_state.object_name!r}."
        )
    if left_state.object_urdf_path != right_state.object_urdf_path:
        raise ValueError(
            "Cannot combine bimanual templates with different object URDF paths: "
            f"{left_state.object_urdf_path!r} vs {right_state.object_urdf_path!r}."
        )
    if left_state.object_arti_names != right_state.object_arti_names:
        raise ValueError(
            "Object articulation names do not match between bimanual templates."
        )
    if not np.allclose(left_state.object_pos, right_state.object_pos, atol=1e-5):
        raise ValueError(
            "Object positions do not match between bimanual templates: "
            f"left={left_state.object_pos}, right={right_state.object_pos}."
        )
    if not np.allclose(
        left_state.object_quat_wxyz, right_state.object_quat_wxyz, atol=1e-5
    ):
        raise ValueError(
            "Object quaternions do not match between bimanual templates: "
            f"left={left_state.object_quat_wxyz}, right={right_state.object_quat_wxyz}."
        )
    if not np.allclose(
        left_state.object_arti_vals, right_state.object_arti_vals, atol=1e-5
    ):
        raise ValueError(
            "Object articulation values do not match between bimanual templates: "
            f"left={left_state.object_arti_vals}, right={right_state.object_arti_vals}."
        )

    metadata = dict(right_state.metadata)
    metadata.update(
        {
            "bimanual_template": True,
            "active_sides": ("left", "right"),
            "left_template_id": left_state.template_id,
            "right_template_id": right_state.template_id,
            "left_template_metadata": dict(left_state.metadata),
            "right_template_metadata": dict(right_state.metadata),
        }
    )

    return TemplateState(
        template_id=template_id,
        active_side="right",
        robot_model=right_state.robot_model,
        left_joint_names=left_state.left_joint_names,
        left_joint_vals=left_state.left_joint_vals,
        right_joint_names=right_state.right_joint_names,
        right_joint_vals=right_state.right_joint_vals,
        object_name=right_state.object_name,
        object_urdf_path=right_state.object_urdf_path,
        object_pos=right_state.object_pos,
        object_quat_wxyz=right_state.object_quat_wxyz,
        object_arti_vals=right_state.object_arti_vals,
        object_arti_names=right_state.object_arti_names,
        metadata=metadata,
        human_skeleton_keypoint_names=right_state.human_skeleton_keypoint_names,
        human_bps_keypoint_names=right_state.human_bps_keypoint_names,
        left_human_skeleton_keypoint_pos=left_state.left_human_skeleton_keypoint_pos,
        right_human_skeleton_keypoint_pos=right_state.right_human_skeleton_keypoint_pos,
        left_human_bps_keypoint_pos=left_state.left_human_bps_keypoint_pos,
        right_human_bps_keypoint_pos=right_state.right_human_bps_keypoint_pos,
        left_human_skeleton_keypoint_valid=left_state.left_human_skeleton_keypoint_valid,
        right_human_skeleton_keypoint_valid=right_state.right_human_skeleton_keypoint_valid,
        left_human_bps_keypoint_valid=left_state.left_human_bps_keypoint_valid,
        right_human_bps_keypoint_valid=right_state.right_human_bps_keypoint_valid,
    )


def _replace_active_q(state: TemplateState, new_q: np.ndarray) -> TemplateState:
    new_q = np.asarray(new_q, dtype=np.float32)
    if state.active_side == "right":
        return replace(state, right_joint_vals=new_q)
    return replace(state, left_joint_vals=new_q)


def derive_open_reach_template(
    state: TemplateState,
    *,
    template_id: str,
    backoff_translation: np.ndarray,
    backoff_distance: float | None = None,
    height_offset: float = 0.06,
) -> TemplateState:
    cfg = _robot_hand_cfg(
        state.robot_model,
        _hand_config_side(state, state.active_side),
    )
    joint_names = state.active_joint_names
    q = state.active_joint_vals.copy()
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    for joint_name, value in cfg["open_qpos"].items():
        if joint_name in name_to_idx:
            q[name_to_idx[joint_name]] = float(value)

    backoff_translation = np.asarray(backoff_translation, dtype=np.float32).reshape(3)
    if backoff_distance is None:
        backoff_distance = float(np.linalg.norm(backoff_translation))
    if backoff_distance <= 0.0:
        raise ValueError(f"Expected positive backoff_distance, got {backoff_distance}.")
    wrist_pose = _active_wrist_pose(state, state.active_joint_vals)
    wrist_y_axis = np.asarray(wrist_pose[:3, 1], dtype=np.float32)
    wrist_y_norm = float(np.linalg.norm(wrist_y_axis))
    if wrist_y_norm <= 1e-8:
        raise ValueError("Wrist y-axis has near-zero norm in URDF FK pose.")
    pullback_delta = -float(backoff_distance) * wrist_y_axis / wrist_y_norm

    root = _root_indices(joint_names)
    q[root["tx"]] += float(pullback_delta[0])
    q[root["ty"]] += float(pullback_delta[1])
    q[root["tz"]] += float(pullback_delta[2]) + float(height_offset)
    return replace(
        _replace_active_q(state, q),
        template_id=template_id,
        **{
            f"{state.active_side}_human_skeleton_keypoint_valid": False,
            f"{state.active_side}_human_bps_keypoint_valid": False,
        },
    )


def derive_open_hand_template(
    state: TemplateState,
    *,
    template_id: str,
) -> TemplateState:
    cfg = _robot_hand_cfg(
        state.robot_model,
        _hand_config_side(state, state.active_side),
    )
    joint_names = state.active_joint_names
    q = state.active_joint_vals.copy()
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    for joint_name, value in cfg["open_qpos"].items():
        if joint_name in name_to_idx:
            q[name_to_idx[joint_name]] = float(value)
    return replace(
        _replace_active_q(state, q),
        template_id=template_id,
        **{
            f"{state.active_side}_human_skeleton_keypoint_valid": False,
            f"{state.active_side}_human_bps_keypoint_valid": False,
        },
    )


def derive_partial_open_clearance_template(
    state: TemplateState,
    *,
    template_id: str,
    clearance_translation: np.ndarray,
    open_fraction: float,
) -> TemplateState:
    if open_fraction < 0.0 or open_fraction > 1.0:
        raise ValueError(f"Expected open_fraction in [0, 1], got {open_fraction}.")
    clearance_translation = np.asarray(clearance_translation, dtype=np.float32).reshape(
        3
    )
    cfg = _robot_hand_cfg(
        state.robot_model,
        _hand_config_side(state, state.active_side),
    )
    joint_names = state.active_joint_names
    q = state.active_joint_vals.copy()
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    for joint_name, open_value in cfg["open_qpos"].items():
        if joint_name in name_to_idx:
            idx = name_to_idx[joint_name]
            q[idx] = (1.0 - open_fraction) * q[idx] + open_fraction * float(open_value)
    root = _root_indices(joint_names)
    q[root["tx"]] += clearance_translation[0]
    q[root["ty"]] += clearance_translation[1]
    q[root["tz"]] += clearance_translation[2]
    return replace(_replace_active_q(state, q), template_id=template_id)


def apply_pair_translation(
    state: TemplateState,
    *,
    template_id: str,
    delta: np.ndarray,
) -> TemplateState:
    delta = np.asarray(delta, dtype=np.float32).reshape(3)
    return apply_pair_rigid_transform(
        state,
        template_id=template_id,
        transform=_transform_matrix(R.identity(), delta),
    )


def apply_pair_rotation(
    state: TemplateState,
    *,
    template_id: str,
    axis: np.ndarray,
    degrees: float,
    pivot: str = "object_origin",
) -> TemplateState:
    if pivot != "object_origin":
        raise ValueError(
            f"Unsupported pivot {pivot!r}; only 'object_origin' is supported."
        )

    axis = np.asarray(axis, dtype=np.float32).reshape(3)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        raise ValueError("Rotation axis must have non-zero norm.")
    axis = axis / axis_norm
    delta_rot = R.from_rotvec(np.deg2rad(float(degrees)) * axis)
    pivot_pos = np.asarray(state.object_pos, dtype=np.float64)
    translation = pivot_pos - delta_rot.as_matrix() @ pivot_pos
    return apply_pair_rigid_transform(
        state,
        template_id=template_id,
        transform=_transform_matrix(delta_rot, translation),
    )


def apply_pair_pose(
    state: TemplateState,
    *,
    template_id: str,
    object_pos: np.ndarray,
    object_quat_wxyz: np.ndarray | None = None,
) -> TemplateState:
    object_pos = np.asarray(object_pos, dtype=np.float32).reshape(3)
    if object_quat_wxyz is None:
        return apply_pair_translation(
            state,
            template_id=template_id,
            delta=object_pos - state.object_pos,
        )

    object_quat_wxyz = np.asarray(object_quat_wxyz, dtype=np.float32).reshape(4)
    current_object_rot = R.from_quat(_quat_wxyz_to_xyzw(state.object_quat_wxyz))
    target_object_rot = R.from_quat(_quat_wxyz_to_xyzw(object_quat_wxyz))
    delta_rot = target_object_rot * current_object_rot.inv()
    translation = object_pos.astype(np.float64) - (
        delta_rot.as_matrix() @ np.asarray(state.object_pos, dtype=np.float64)
    )
    return apply_pair_rigid_transform(
        state,
        template_id=template_id,
        transform=_transform_matrix(delta_rot, translation),
    )


def apply_object_pose_only(
    state: TemplateState,
    *,
    template_id: str,
    object_pos: np.ndarray | None = None,
    object_quat_wxyz: np.ndarray | None = None,
) -> TemplateState:
    if object_pos is None and object_quat_wxyz is None:
        raise ValueError("set_object_pose must define object_pos or object_quat_wxyz.")
    updates = {}
    if object_pos is not None:
        updates["object_pos"] = np.asarray(object_pos, dtype=np.float32).reshape(3)
    if object_quat_wxyz is not None:
        updates["object_quat_wxyz"] = np.asarray(
            object_quat_wxyz, dtype=np.float32
        ).reshape(4)
    return replace(state, template_id=template_id, **updates)


def _object_transform_matrix(
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R.from_quat(_quat_wxyz_to_xyzw(quat_wxyz)).as_matrix()
    pose[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return pose


def _object_z_extent(state: TemplateState) -> tuple[float, float]:
    urdf_inst = URDF.load(str(process_urdf(state.object_urdf_path)))
    joint_dict = dict(
        zip(
            state.object_arti_names,
            np.asarray(state.object_arti_vals, dtype=np.float32).reshape(-1),
        )
    )
    fk = urdf_inst.visual_trimesh_fk(joint_dict)
    if len(fk) == 0:
        fk = urdf_inst.collision_trimesh_fk(joint_dict)
    if len(fk) == 0:
        raise ValueError(
            f"Object {state.object_name!r} has no visual or collision meshes."
        )

    object_pose = _object_transform_matrix(state.object_pos, state.object_quat_wxyz)
    lowest_z = np.inf
    highest_z = -np.inf
    for mesh, link_pose in fk.items():
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            raise ValueError(
                f"Expected mesh vertices shape (V, 3), got {vertices.shape}."
            )
        vertices_h = np.concatenate(
            [vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        link_pose = np.asarray(link_pose, dtype=np.float64)
        world_vertices = (object_pose @ link_pose @ vertices_h.T).T[:, :3]
        lowest_z = min(lowest_z, float(np.min(world_vertices[:, 2])))
        highest_z = max(highest_z, float(np.max(world_vertices[:, 2])))
    if not np.isfinite(lowest_z) or not np.isfinite(highest_z):
        raise ValueError(
            f"Could not compute z extent for object {state.object_name!r}."
        )
    return float(lowest_z), float(highest_z)


def object_lowest_z(state: TemplateState) -> float:
    return _object_z_extent(state)[0]


def object_highest_z(state: TemplateState) -> float:
    return _object_z_extent(state)[1]


def align_pair_lowest_z(
    state: TemplateState,
    *,
    template_id: str,
    target_lowest_z: float,
) -> TemplateState:
    dz = float(target_lowest_z) - object_lowest_z(state)
    return apply_pair_translation(
        state,
        template_id=template_id,
        delta=np.asarray([0.0, 0.0, dz], dtype=np.float32),
    )


def apply_object_articulation(
    state: TemplateState,
    *,
    template_id: str,
    values: np.ndarray | None = None,
    joint_values: dict[str, float] | None = None,
) -> TemplateState:
    if (values is None) == (joint_values is None):
        raise ValueError("Define exactly one of `values` or `joint_values`.")
    if values is not None:
        new_values = np.asarray(values, dtype=np.float32).reshape(-1)
        if new_values.shape != state.object_arti_vals.shape:
            raise ValueError(
                "Articulation values shape does not match template state: "
                f"expected {state.object_arti_vals.shape}, got {new_values.shape}."
            )
    else:
        assert joint_values is not None
        name_to_idx = {name: idx for idx, name in enumerate(state.object_arti_names)}
        unknown = sorted(set(joint_values) - set(name_to_idx))
        if unknown:
            raise ValueError(
                f"Unknown articulation joints {unknown}; expected names from {state.object_arti_names}."
            )
        new_values = state.object_arti_vals.copy()
        for joint_name, value in joint_values.items():
            new_values[name_to_idx[joint_name]] = float(value)
    return replace(
        state,
        template_id=template_id,
        object_arti_vals=new_values.astype(np.float32, copy=False),
    )


def apply_object_articulation_with_link_motion(
    state: TemplateState,
    *,
    template_id: str,
    values: np.ndarray | None = None,
    joint_values: dict[str, float] | None = None,
    link_name: str | None = None,
) -> TemplateState:
    end_state = apply_object_articulation(
        state,
        template_id=template_id,
        values=values,
        joint_values=joint_values,
    )
    link_name = resolve_articulated_link_name(state, link_name)
    start_link_pose = _object_link_pose(
        state,
        link_name=link_name,
        arti_vals=state.object_arti_vals,
    )
    end_link_pose = _object_link_pose(
        end_state,
        link_name=link_name,
        arti_vals=end_state.object_arti_vals,
    )
    link_delta = end_link_pose @ np.linalg.inv(start_link_pose)
    active_q = _apply_rigid_transform_to_active_hand(state, link_delta)
    return _replace_active_q(end_state, active_q)


def derive_template_from_transforms(
    base_state: TemplateState,
    *,
    template_id: str,
    transforms: list[dict],
) -> TemplateState:
    state = replace(base_state, template_id=template_id)
    for transform in transforms:
        transform_type = str(transform["type"])
        if transform_type == "translate_pair":
            state = apply_pair_translation(
                state,
                template_id=template_id,
                delta=np.asarray(transform["delta"], dtype=np.float32),
            )
        elif transform_type == "rotate_pair":
            state = apply_pair_rotation(
                state,
                template_id=template_id,
                axis=np.asarray(transform["axis"], dtype=np.float32),
                degrees=float(transform["degrees"]),
                pivot=str(transform.get("pivot", "object_origin")),
            )
        elif transform_type == "set_pair_pose":
            state = apply_pair_pose(
                state,
                template_id=template_id,
                object_pos=np.asarray(transform["object_pos"], dtype=np.float32),
                object_quat_wxyz=(
                    None
                    if "object_quat_wxyz" not in transform
                    else np.asarray(transform["object_quat_wxyz"], dtype=np.float32)
                ),
            )
        elif transform_type == "set_object_pose":
            state = apply_object_pose_only(
                state,
                template_id=template_id,
                object_pos=(
                    None
                    if "object_pos" not in transform
                    else np.asarray(transform["object_pos"], dtype=np.float32)
                ),
                object_quat_wxyz=(
                    None
                    if "object_quat_wxyz" not in transform
                    else np.asarray(transform["object_quat_wxyz"], dtype=np.float32)
                ),
            )
        elif transform_type == "align_pair_lowest_z":
            state = align_pair_lowest_z(
                state,
                template_id=template_id,
                target_lowest_z=float(transform["target_lowest_z"]),
            )
        elif transform_type == "open_hand":
            state = derive_open_hand_template(state, template_id=template_id)
        elif transform_type == "set_articulation":
            state = apply_object_articulation_with_link_motion(
                state,
                template_id=template_id,
                values=(
                    None
                    if "values" not in transform
                    else np.asarray(transform["values"], dtype=np.float32)
                ),
                joint_values=transform.get("joint_values"),
                link_name=transform.get("articulation_link_name"),
            )
        else:
            raise ValueError(f"Unsupported transform type {transform_type!r}.")
    return state


def _interpolate_joint_vectors(
    q_start: np.ndarray, q_end: np.ndarray, frames: int
) -> np.ndarray:
    if frames < 1:
        raise ValueError(f"Expected frames >= 1, got {frames}.")
    alphas = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    return np.stack([(1.0 - a) * q_start + a * q_end for a in alphas], axis=0)


def _nearest_root_angle_equivalent(
    q_start: np.ndarray, q_end: np.ndarray, joint_names: list[str]
) -> np.ndarray:
    q_end = np.asarray(q_end, dtype=np.float64).copy()
    if len(joint_names) == 0:
        return q_end.astype(np.float32)
    root = _root_indices(joint_names)
    angle_keys = ("roll", "pitch", "yaw")
    angle_indices = [root[key] for key in angle_keys]
    start_angles = np.asarray([q_start[idx] for idx in angle_indices], dtype=np.float64)
    end_angles = np.asarray([q_end[idx] for idx in angle_indices], dtype=np.float64)
    base_candidates = [
        end_angles,
        np.asarray(
            [
                end_angles[0] + np.pi,
                np.pi - end_angles[1],
                end_angles[2] + np.pi,
            ],
            dtype=np.float64,
        ),
    ]
    candidates: list[np.ndarray] = []
    for base in base_candidates:
        for roll_shift in (-1, 0, 1):
            for pitch_shift in (-1, 0, 1):
                for yaw_shift in (-1, 0, 1):
                    candidates.append(
                        base
                        + 2.0 * np.pi * np.asarray([roll_shift, pitch_shift, yaw_shift])
                    )
    nearest_angles = min(
        candidates,
        key=lambda candidate: float(np.linalg.norm(candidate - start_angles)),
    )
    for idx, angle in zip(angle_indices, nearest_angles.tolist()):
        q_end[idx] = angle
    return q_end.astype(np.float32)


def _interpolate_joint_vectors_with_nearest_root_angles(
    q_start: np.ndarray, q_end: np.ndarray, joint_names: list[str], frames: int
) -> np.ndarray:
    return _interpolate_joint_vectors(
        q_start,
        _nearest_root_angle_equivalent(q_start, q_end, joint_names),
        frames,
    )


def _nearest_active_wrist_pose_equivalent(
    start_state: TemplateState, end_state: TemplateState
) -> TemplateState:
    joint_names = start_state.active_joint_names
    if joint_names != end_state.active_joint_names:
        raise ValueError("Active joint names do not match between templates.")

    q_start = np.asarray(start_state.active_joint_vals, dtype=np.float64)
    q_end = np.asarray(end_state.active_joint_vals, dtype=np.float64)
    target_pose = _active_wrist_pose(end_state, q_end)
    initial_q = _nearest_root_angle_equivalent(q_start, q_end, joint_names)
    initial_root = _root_vector(initial_q, joint_names)

    def residual(root_values: np.ndarray) -> np.ndarray:
        candidate_q = _set_root_vector(q_end, joint_names, root_values)
        pose = _active_wrist_pose(end_state, candidate_q)
        return np.concatenate(
            [
                pose[:3, 3] - target_pose[:3, 3],
                _rotation_error(target_pose[:3, :3], pose[:3, :3]),
            ],
            axis=0,
        )

    result = least_squares(
        residual,
        initial_root,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=200,
    )
    solved_q = _set_root_vector(q_end, joint_names, result.x).astype(np.float32)
    solved_pose = _active_wrist_pose(end_state, solved_q)
    translation_error = float(np.linalg.norm(solved_pose[:3, 3] - target_pose[:3, 3]))
    rotation_error = float(
        np.linalg.norm(_rotation_error(target_pose[:3, :3], solved_pose[:3, :3]))
    )
    if not result.success or translation_error > 1e-5 or rotation_error > 1e-5:
        raise ValueError(
            "Could not find a nearby active wrist root equivalent. "
            f"translation_error={translation_error:.6g}, "
            f"rotation_error={rotation_error:.6g}, optimizer_success={result.success}."
        )

    if start_state.active_side == "right":
        return replace(end_state, right_joint_vals=solved_q)
    return replace(end_state, left_joint_vals=solved_q)


def _interpolate_object_quats(
    quat_start_wxyz: np.ndarray, quat_end_wxyz: np.ndarray, frames: int
) -> np.ndarray:
    rotations = R.from_quat(
        np.stack(
            [
                _quat_wxyz_to_xyzw(quat_start_wxyz),
                _quat_wxyz_to_xyzw(quat_end_wxyz),
            ],
            axis=0,
        )
    )
    slerp = Slerp([0.0, 1.0], rotations)
    interp = slerp(np.linspace(0.0, 1.0, frames, dtype=np.float32)).as_quat()
    return np.stack([_quat_xyzw_to_wxyz(q) for q in interp], axis=0).astype(np.float32)


def hold_template(state: TemplateState, *, frames: int) -> dict[str, np.ndarray]:
    if frames < 1:
        raise ValueError(f"Expected frames >= 1, got {frames}.")
    piece = {
        "left_joint_vals": np.repeat(state.left_joint_vals[None, :], frames, axis=0),
        "right_joint_vals": np.repeat(state.right_joint_vals[None, :], frames, axis=0),
        "asset_pos": np.repeat(state.object_pos[None, :], frames, axis=0),
        "asset_quat": np.repeat(state.object_quat_wxyz[None, :], frames, axis=0),
        "asset_arti_vals": np.repeat(state.object_arti_vals[None, :], frames, axis=0),
    }
    piece.update(_state_keypoint_piece(state, frames))
    return piece


def interpolate_templates(
    start_state: TemplateState,
    end_state: TemplateState,
    *,
    frames: int,
) -> dict[str, np.ndarray]:
    if start_state.active_side != end_state.active_side:
        raise ValueError("Cannot interpolate templates with different active sides.")
    if start_state.left_joint_names != end_state.left_joint_names:
        raise ValueError("Left joint names do not match between templates.")
    if start_state.right_joint_names != end_state.right_joint_names:
        raise ValueError("Right joint names do not match between templates.")
    if start_state.object_arti_names != end_state.object_arti_names:
        raise ValueError(
            "Object articulation joint names do not match between templates."
        )

    end_state = _nearest_active_wrist_pose_equivalent(start_state, end_state)
    piece = {
        "left_joint_vals": _interpolate_joint_vectors_with_nearest_root_angles(
            start_state.left_joint_vals,
            end_state.left_joint_vals,
            start_state.left_joint_names,
            frames,
        ),
        "right_joint_vals": _interpolate_joint_vectors_with_nearest_root_angles(
            start_state.right_joint_vals,
            end_state.right_joint_vals,
            start_state.right_joint_names,
            frames,
        ),
        "asset_pos": _interpolate_joint_vectors(
            start_state.object_pos,
            end_state.object_pos,
            frames,
        ),
        "asset_quat": _interpolate_object_quats(
            start_state.object_quat_wxyz,
            end_state.object_quat_wxyz,
            frames,
        ),
        "asset_arti_vals": _interpolate_joint_vectors(
            start_state.object_arti_vals,
            end_state.object_arti_vals,
            frames,
        ),
    }
    piece.update(_interpolate_keypoint_piece(start_state, end_state, frames=frames))
    return piece


def interpolate_pair_templates(
    start_state: TemplateState,
    end_state: TemplateState,
    *,
    frames: int,
) -> dict[str, np.ndarray]:
    if start_state.active_side != end_state.active_side:
        raise ValueError("Cannot interpolate templates with different active sides.")
    if start_state.left_joint_names != end_state.left_joint_names:
        raise ValueError("Left joint names do not match between templates.")
    if start_state.right_joint_names != end_state.right_joint_names:
        raise ValueError("Right joint names do not match between templates.")
    if start_state.object_arti_names != end_state.object_arti_names:
        raise ValueError(
            "Object articulation joint names do not match between templates."
        )
    if frames < 1:
        raise ValueError(f"Expected frames >= 1, got {frames}.")

    asset_pos = _interpolate_joint_vectors(
        start_state.object_pos,
        end_state.object_pos,
        frames,
    )
    asset_quat = _interpolate_object_quats(
        start_state.object_quat_wxyz,
        end_state.object_quat_wxyz,
        frames,
    )
    asset_arti_vals = _interpolate_joint_vectors(
        start_state.object_arti_vals,
        end_state.object_arti_vals,
        frames,
    )

    left_joint_vals = []
    right_joint_vals = []
    for frame_idx in range(frames):
        frame_state = apply_pair_pose(
            start_state,
            template_id=f"{start_state.template_id}_pair_interp_{frame_idx}",
            object_pos=asset_pos[frame_idx],
            object_quat_wxyz=asset_quat[frame_idx],
        )
        frame_state = replace(
            frame_state,
            object_arti_vals=asset_arti_vals[frame_idx].astype(np.float32),
        )
        left_joint_vals.append(frame_state.left_joint_vals)
        right_joint_vals.append(frame_state.right_joint_vals)

    piece = {
        "left_joint_vals": np.stack(left_joint_vals, axis=0).astype(np.float32),
        "right_joint_vals": np.stack(right_joint_vals, axis=0).astype(np.float32),
        "asset_pos": asset_pos.astype(np.float32),
        "asset_quat": asset_quat.astype(np.float32),
        "asset_arti_vals": asset_arti_vals.astype(np.float32),
    }
    piece.update(_interpolate_keypoint_piece(start_state, end_state, frames=frames))
    return piece


def _rebase_articulated_state_to_reference_base(
    state: TemplateState,
    *,
    reference_state: TemplateState,
    link_name: str,
) -> TemplateState:
    old_link_pose = _object_link_pose(
        state,
        link_name=link_name,
        arti_vals=state.object_arti_vals,
    )
    rebased_state = replace(
        state,
        object_pos=reference_state.object_pos.copy(),
        object_quat_wxyz=reference_state.object_quat_wxyz.copy(),
    )
    new_link_pose = _object_link_pose(
        rebased_state,
        link_name=link_name,
        arti_vals=rebased_state.object_arti_vals,
    )
    link_delta = new_link_pose @ np.linalg.inv(old_link_pose)
    active_q = _apply_rigid_transform_to_active_hand(state, link_delta)
    return _replace_active_q(rebased_state, active_q)


def interpolate_articulation_link_motion(
    start_state: TemplateState,
    end_state: TemplateState,
    *,
    frames: int,
    link_name: str | None = None,
) -> dict[str, np.ndarray]:
    if start_state.active_side != end_state.active_side:
        raise ValueError("Cannot interpolate templates with different active sides.")
    if start_state.left_joint_names != end_state.left_joint_names:
        raise ValueError("Left joint names do not match between templates.")
    if start_state.right_joint_names != end_state.right_joint_names:
        raise ValueError("Right joint names do not match between templates.")
    if start_state.object_arti_names != end_state.object_arti_names:
        raise ValueError(
            "Object articulation joint names do not match between templates."
        )
    link_name = resolve_articulated_link_name(start_state, link_name)
    if not np.allclose(start_state.object_pos, end_state.object_pos, atol=1e-6):
        end_state = _rebase_articulated_state_to_reference_base(
            end_state,
            reference_state=start_state,
            link_name=link_name,
        )
    if not np.allclose(
        start_state.object_quat_wxyz,
        end_state.object_quat_wxyz,
        atol=1e-6,
    ):
        end_state = _rebase_articulated_state_to_reference_base(
            end_state,
            reference_state=start_state,
            link_name=link_name,
        )
    if frames < 1:
        raise ValueError(f"Expected frames >= 1, got {frames}.")

    asset_arti_vals = _interpolate_joint_vectors(
        start_state.object_arti_vals,
        end_state.object_arti_vals,
        frames,
    )
    start_link_pose = _object_link_pose(
        start_state,
        link_name=link_name,
        arti_vals=start_state.object_arti_vals,
    )

    left_joint_vals = []
    right_joint_vals = []
    for frame_idx in range(frames):
        frame_link_pose = _object_link_pose(
            start_state,
            link_name=link_name,
            arti_vals=asset_arti_vals[frame_idx],
        )
        link_delta = frame_link_pose @ np.linalg.inv(start_link_pose)
        active_q = _apply_rigid_transform_to_active_hand(start_state, link_delta)
        frame_state = _replace_active_q(start_state, active_q)
        left_joint_vals.append(frame_state.left_joint_vals)
        right_joint_vals.append(frame_state.right_joint_vals)

    piece = {
        "left_joint_vals": np.stack(left_joint_vals, axis=0).astype(np.float32),
        "right_joint_vals": np.stack(right_joint_vals, axis=0).astype(np.float32),
        "asset_pos": np.repeat(start_state.object_pos[None, :], frames, axis=0),
        "asset_quat": np.repeat(start_state.object_quat_wxyz[None, :], frames, axis=0),
        "asset_arti_vals": asset_arti_vals.astype(np.float32),
    }
    end_state = apply_object_articulation_with_link_motion(
        start_state,
        template_id=f"{start_state.template_id}_articulated_keypoints",
        values=asset_arti_vals[-1],
        link_name=link_name,
    )
    piece.update(_interpolate_keypoint_piece(start_state, end_state, frames=frames))
    return piece


def interpolate_hand_templates(
    start_state: TemplateState,
    end_state: TemplateState,
    *,
    frames: int,
    object_state: TemplateState | None = None,
) -> dict[str, np.ndarray]:
    if start_state.active_side != end_state.active_side:
        raise ValueError("Cannot interpolate templates with different active sides.")
    if start_state.left_joint_names != end_state.left_joint_names:
        raise ValueError("Left joint names do not match between templates.")
    if start_state.right_joint_names != end_state.right_joint_names:
        raise ValueError("Right joint names do not match between templates.")
    if frames < 1:
        raise ValueError(f"Expected frames >= 1, got {frames}.")

    end_state = _nearest_active_wrist_pose_equivalent(start_state, end_state)
    static_object = start_state if object_state is None else object_state
    piece = {
        "left_joint_vals": _interpolate_joint_vectors_with_nearest_root_angles(
            start_state.left_joint_vals,
            end_state.left_joint_vals,
            start_state.left_joint_names,
            frames,
        ),
        "right_joint_vals": _interpolate_joint_vectors_with_nearest_root_angles(
            start_state.right_joint_vals,
            end_state.right_joint_vals,
            start_state.right_joint_names,
            frames,
        ),
        "asset_pos": np.repeat(static_object.object_pos[None, :], frames, axis=0),
        "asset_quat": np.repeat(
            static_object.object_quat_wxyz[None, :], frames, axis=0
        ),
        "asset_arti_vals": np.repeat(
            static_object.object_arti_vals[None, :], frames, axis=0
        ),
    }
    piece.update(_interpolate_keypoint_piece(start_state, end_state, frames=frames))
    return piece


def hand_transition_via_clearance(
    start_state: TemplateState,
    end_state: TemplateState,
    *,
    frames: int,
    clearance_translation: np.ndarray,
    open_fraction: float,
    object_state: TemplateState | None = None,
) -> dict[str, np.ndarray]:
    if frames < 1:
        raise ValueError(f"Expected frames >= 1, got {frames}.")
    if frames == 1:
        return interpolate_hand_templates(
            start_state,
            end_state,
            frames=1,
            object_state=object_state,
        )

    mid_state = derive_partial_open_clearance_template(
        start_state,
        template_id=f"{start_state.template_id}_clearance",
        clearance_translation=clearance_translation,
        open_fraction=open_fraction,
    )
    first_frames = max(1, frames // 2)
    second_frames = frames - first_frames
    first_piece = interpolate_hand_templates(
        start_state,
        mid_state,
        frames=first_frames,
        object_state=object_state,
    )
    if second_frames == 0:
        return first_piece

    second_piece = interpolate_hand_templates(
        mid_state,
        end_state,
        frames=second_frames + 1,
        object_state=object_state,
    )
    return {
        key: np.concatenate([first_piece[key], second_piece[key][1:]], axis=0)
        for key in first_piece
    }


def move_pair_to_pose(
    start_state: TemplateState,
    *,
    frames: int,
    object_pos: np.ndarray,
    object_quat_wxyz: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    end_state = apply_pair_pose(
        start_state,
        template_id=f"{start_state.template_id}_moved",
        object_pos=object_pos,
        object_quat_wxyz=object_quat_wxyz,
    )
    return interpolate_pair_templates(start_state, end_state, frames=frames)


def place_object(
    start_state: TemplateState,
    *,
    frames: int,
    object_pos: np.ndarray,
    object_quat_wxyz: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    target_state = apply_pair_pose(
        start_state,
        template_id=f"{start_state.template_id}_place_target",
        object_pos=object_pos,
        object_quat_wxyz=object_quat_wxyz,
    )
    return interpolate_pair_templates(start_state, target_state, frames=frames)


def release_or_open(
    start_state: TemplateState,
    *,
    frames: int,
    clearance_translation: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if clearance_translation is None:
        clearance_translation = np.asarray([0.0, 0.0, 0.08], dtype=np.float32)
    else:
        clearance_translation = np.asarray(clearance_translation, dtype=np.float32)

    open_state = derive_open_hand_template(
        start_state,
        template_id=f"{start_state.template_id}_open",
    )
    if frames == 1:
        return interpolate_hand_templates(
            start_state,
            open_state,
            frames=1,
            object_state=start_state,
        )

    clear_state = derive_partial_open_clearance_template(
        open_state,
        template_id=f"{start_state.template_id}_open_clear",
        clearance_translation=clearance_translation,
        open_fraction=1.0,
    )
    open_frames = max(1, frames // 2)
    clear_frames = frames - open_frames
    open_piece = interpolate_hand_templates(
        start_state,
        open_state,
        frames=open_frames,
        object_state=start_state,
    )
    if clear_frames == 0:
        return open_piece

    clear_piece = interpolate_hand_templates(
        open_state,
        clear_state,
        frames=clear_frames + 1,
        object_state=start_state,
    )
    return {
        key: np.concatenate([open_piece[key], clear_piece[key][1:]], axis=0)
        for key in open_piece
    }


def articulate_object(
    start_state: TemplateState,
    *,
    frames: int,
    values: np.ndarray | None = None,
    joint_values: dict[str, float] | None = None,
    link_name: str | None = None,
) -> dict[str, np.ndarray]:
    end_state = apply_object_articulation(
        start_state,
        template_id=f"{start_state.template_id}_articulated",
        values=values,
        joint_values=joint_values,
    )
    return interpolate_articulation_link_motion(
        start_state,
        end_state,
        frames=frames,
        link_name=link_name,
    )


def build_motion_payload(
    *,
    active_side: str,
    active_sides: tuple[str, ...] | None = None,
    robot_model: str,
    left_joint_names: list[str],
    right_joint_names: list[str],
    object_name: str,
    object_urdf_path: str,
    object_arti_names: list[str],
    left_joint_vals: np.ndarray,
    right_joint_vals: np.ndarray,
    asset_pos: np.ndarray,
    asset_quat: np.ndarray,
    asset_arti_vals: np.ndarray,
    phase_names: list[str],
    phase_lengths: list[int],
    metadata: dict[str, object] | None = None,
    scene_payload: dict[str, np.ndarray] | None = None,
    extra_payload: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    if active_sides is None:
        active_sides = (active_side,)
    active_sides = tuple(str(side) for side in active_sides)
    if len(active_sides) == 0 or any(
        side not in ("left", "right") for side in active_sides
    ):
        raise ValueError(
            f"Expected active_sides to contain left/right, got {active_sides}."
        )
    asset_quat = np.asarray(asset_quat, dtype=np.float32)
    asset_rot = (
        R.from_quat(np.roll(asset_quat, -1, axis=-1)).as_rotvec().astype(np.float32)
    )
    payload = {
        "demo_kind": np.array("sequence", dtype=object),
        "asset_pos": np.asarray(asset_pos, dtype=np.float32),
        "asset_quat": asset_quat,
        "asset_rot": asset_rot,
        "asset_arti_vals": np.asarray(asset_arti_vals, dtype=np.float32),
        "asset_arti_names": np.asarray(object_arti_names, dtype=object),
        "traj_hand_side": np.array(active_side, dtype=object),
        "traj_active_sides": np.asarray(active_sides, dtype=object),
        "traj_hand_name": np.array(robot_model, dtype=object),
        "traj_robot_model": np.array(robot_model, dtype=object),
        "traj_object_name": np.array(object_name, dtype=object),
        "traj_object_urdf_path": np.array(str(object_urdf_path), dtype=object),
        "left_joint_names": np.asarray(left_joint_names, dtype=object),
        "left_joint_vals": np.asarray(left_joint_vals, dtype=np.float32),
        "right_joint_names": np.asarray(right_joint_names, dtype=object),
        "right_joint_vals": np.asarray(right_joint_vals, dtype=np.float32),
        # Compatibility aliases used by vis_robot_retgt_motion.py and older motion loaders.
        "obj_pos": np.asarray(asset_pos, dtype=np.float32),
        "obj_quat": asset_quat,
        "obj_rot": asset_rot,
        "obj_arti": np.asarray(asset_arti_vals, dtype=np.float32),
        "joint_names": np.asarray(object_arti_names, dtype=object),
        "left_jnames": np.asarray(left_joint_names, dtype=object),
        "left_jvals": np.asarray(left_joint_vals, dtype=np.float32),
        "right_jnames": np.asarray(right_joint_names, dtype=object),
        "right_jvals": np.asarray(right_joint_vals, dtype=np.float32),
        "exact_timesteps": np.arange(np.asarray(asset_pos).shape[0], dtype=np.int64),
        "phase_names": np.asarray(phase_names, dtype=object),
        "phase_lengths": np.asarray(phase_lengths, dtype=np.int64),
        "n_reach_frames": np.array([0], dtype=np.int64),
        "n_lift_frames": np.array([0], dtype=np.int64),
        "is_active_left": np.array(["left" in active_sides], dtype=bool),
        "is_active_right": np.array(["right" in active_sides], dtype=bool),
    }
    if metadata is not None:
        for key, value in metadata.items():
            payload[key] = np.array(value, dtype=object)
    if scene_payload is not None:
        payload.update(scene_payload)
    if extra_payload is not None:
        payload.update(extra_payload)
    return payload
