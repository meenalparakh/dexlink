#!/usr/bin/env python

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from pydrake.all import RigidTransform, RobotDiagramBuilder
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from dexlink.src.retargeting.data.hand_cfgs.franka_xhand import XHAND_CFGs as FRANKA_XHAND_CFGs
from dexlink.src.retargeting.data.hand_cfgs.inspire import INSPIRE_CFGs
from dexlink.src.retargeting.data.hand_cfgs.xhand import XHAND_CFGs
from dexlink.utils.retargeting.path_utils import process_urdf

DEXLINK_ROOT = Path(__file__).resolve().parents[3]
ROOT_JOINT_TOKENS = (
    "forearm_tx_link_joint",
    "forearm_ty_link_joint",
    "forearm_tz_link_joint",
    "forearm_roll_link_joint",
    "forearm_pitch_link_joint",
    "forearm_yaw_link_joint",
)
ROOT_ROTATION_JOINT_TOKENS = (
    "forearm_roll_link_joint",
    "forearm_pitch_link_joint",
    "forearm_yaw_link_joint",
)
TWO_PI = 2.0 * np.pi
WRIST_ROTATION_REMAP_LOWER = -np.pi
WRIST_ROTATION_REMAP_UPPER = np.pi


@dataclass(frozen=True)
class DrakeLineSearchConfig:
    input_path: Path
    output_path: Path
    hand: str
    start_frame: int
    end_frame: int | None
    line_search_samples: int
    line_search_passes: int
    joint_radius: float
    root_translation_radius: float
    root_rotation_radius: float
    refinement_collision_distance: float
    valid_collision_distance: float
    refine_root_joints: bool
    fail_on_unresolved: bool
    prefer_open_qpos_hull: bool
    required_collision_free_start_frames: int
    backward_max_future_frame_gap: int
    backward_joint_radius: float
    backward_root_translation_radius: float
    backward_root_rotation_radius: float


@dataclass(frozen=True)
class DrakeObjectFrameInput:
    urdf_path: Path
    pose: np.ndarray
    joint_names: tuple[str, ...]
    joint_values: np.ndarray


@dataclass(frozen=True)
class DrakeObjectModel:
    model: object
    joint_names: tuple[str, ...]


@dataclass(frozen=True)
class DrakeFramePlant:
    diagram: object
    plant: object
    root_context: object
    context: object
    hand_model: object
    object_models: tuple[DrakeObjectModel, ...]
    hand_scene_geom_pairs: tuple[tuple[object, object], ...]
    hand_joint_names: tuple[str, ...]
    lower_limits: np.ndarray
    upper_limits: np.ndarray


@dataclass(frozen=True)
class DrakeLineSearchStats:
    total_frames: int
    frames: int
    successfully_refined_frames: int
    changed: int
    initially_colliding: int
    unresolved: int
    mean_l1_delta: float
    max_l1_delta: float
    max_radius_bound_violation: float
    wrist_rotation_remapped_values: int
    wrist_rotation_max_abs_delta: float


def _as_payload(raw: object, input_path: Path) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, np.ndarray) and raw.shape == ():
        payload = raw.item()
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"Expected {input_path} to contain an exported payload dict.")


def _load_payload(input_path: Path) -> dict:
    raw = np.load(input_path, allow_pickle=True)
    payload = _as_payload(raw, input_path)
    if str(payload.get("demo_kind", "sequence")).lower() != "sequence":
        raise ValueError(
            f"Expected exported sequence payload, got demo_kind={payload.get('demo_kind')!r}."
        )
    return dict(payload)


def _unwrap_scalar(value: object, field_name: str) -> object:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"Expected scalar {field_name!r}, got {value.shape}.")
        return value.item()
    return value


def _payload_bool(payload: dict, key: str) -> bool:
    if key not in payload:
        raise ValueError(f"Expected exported payload to contain {key!r}.")
    value = payload[key]
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(
                f"Expected {key!r} to contain one bool, got {value.shape}."
            )
        value = value.reshape(-1)[0]
    return bool(value)


def _resolve_stored_path(path: Path) -> Path:
    return path if path.is_absolute() else DEXLINK_ROOT / path


def _normalize_robot_model(robot_model: object) -> str:
    robot_model = _unwrap_scalar(robot_model, "robot_model")
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


def _robot_urdf_path(robot_model: str, side: str) -> Path:
    if side not in ("left", "right"):
        raise ValueError(f"Expected hand side 'left' or 'right', got {side!r}.")
    if robot_model == "inspire":
        return Path(INSPIRE_CFGs[side]["urdf_path"])
    if robot_model == "xhand":
        return Path(XHAND_CFGs[side]["urdf_path"])
    if robot_model == "franka_xhand":
        return Path(FRANKA_XHAND_CFGs[side]["urdf_path"])
    raise ValueError(f"Unsupported robot_model {robot_model!r}.")


def _robot_open_qpos(robot_model: str, side: str) -> dict[str, float]:
    if side not in ("left", "right"):
        raise ValueError(f"Expected hand side 'left' or 'right', got {side!r}.")
    if robot_model == "inspire":
        robot_cfg = INSPIRE_CFGs[side]
        qpos_key = "open_qpos"
    elif robot_model == "xhand":
        robot_cfg = XHAND_CFGs[side]
        qpos_key = "open_qpos_hull"
    elif robot_model == "franka_xhand":
        robot_cfg = FRANKA_XHAND_CFGs[side]
        qpos_key = "open_qpos"
    else:
        raise ValueError(f"Unsupported robot_model {robot_model!r}.")
    return {str(name): float(value) for name, value in robot_cfg[qpos_key].items()}


def _selected_hand(payload: dict, requested_hand: str) -> str:
    if requested_hand != "auto":
        if requested_hand not in ("left", "right"):
            raise ValueError(
                f"Expected --hand to be 'auto', 'left', or 'right', got {requested_hand!r}."
            )
        return requested_hand

    active_left = _payload_bool(payload, "is_active_left")
    active_right = _payload_bool(payload, "is_active_right")
    if active_left == active_right:
        side = str(payload.get("side", "")).lower()
        if side not in ("left", "right"):
            raise ValueError(
                "Expected exactly one active hand or side metadata 'left'/'right'; "
                f"is_active_left={active_left}, is_active_right={active_right}, side={side!r}."
            )
        return side
    return "left" if active_left else "right"


def _joint_names_from_payload(payload: dict, side: str) -> list[str]:
    key = f"{side}_jnames"
    if key not in payload:
        raise ValueError(f"Expected exported payload to contain {key!r}.")
    return [str(name) for name in np.asarray(payload[key], dtype=object).tolist()]


def _joint_values_from_payload(payload: dict, side: str) -> np.ndarray:
    key = f"{side}_jvals"
    if key not in payload:
        raise ValueError(f"Expected exported payload to contain {key!r}.")
    values = np.asarray(payload[key], dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected {key!r} to have shape (T, J), got {values.shape}.")
    return values


def _payload_num_frames(payload: dict) -> int:
    if "retargeted_obj_pos" in payload:
        return int(np.asarray(payload["retargeted_obj_pos"]).shape[0])
    raise ValueError("Expected retargeted_obj_pos to determine payload frame count.")


def _frame_range(n_frames: int, cfg: DrakeLineSearchConfig) -> tuple[int, int]:
    start = int(cfg.start_frame)
    end = n_frames if cfg.end_frame is None else int(cfg.end_frame)
    if start < 0 or start >= n_frames:
        raise ValueError(f"Expected 0 <= start_frame < {n_frames}, got {start}.")
    if end <= start or end > n_frames:
        raise ValueError(f"Expected start_frame < end_frame <= {n_frames}, got {end}.")
    return start, end


def _target_object_urdf_path(payload: dict) -> Path:
    articulated = _unwrap_scalar(
        payload.get("target_articulated_urdf_path", None),
        "target_articulated_urdf_path",
    )
    if articulated not in (None, "", "None"):
        return _resolve_stored_path(Path(str(articulated)))
    if "target_urdf_path" not in payload:
        raise ValueError("Expected payload to contain target_urdf_path.")
    return _resolve_stored_path(
        Path(str(_unwrap_scalar(payload["target_urdf_path"], "target_urdf_path")))
    )


def _table_urdf_path(table_height: float) -> Path:
    table_thickness = 0.04
    table_extent = 4.0
    table_center_z = float(table_height) - 0.5 * table_thickness
    table_dir = DEXLINK_ROOT / "outputs" / "generated_refinement_urdfs"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / f"dexlink_refinement_table_{table_height:.6f}.urdf"
    if not table_path.exists():
        table_path.write_text(
            f"""<?xml version="1.0"?>
<robot name="dexlink_refinement_table">
  <link name="table">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 {table_center_z:.9f}" rpy="0 0 0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {table_center_z:.9f}" rpy="0 0 0"/>
      <geometry><box size="{table_extent} {table_extent} {table_thickness}"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 {table_center_z:.9f}" rpy="0 0 0"/>
      <geometry><box size="{table_extent} {table_extent} {table_thickness}"/></geometry>
    </collision>
  </link>
</robot>
""",
            encoding="utf-8",
        )
    return table_path


def _pose_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_quat(
        np.roll(np.asarray(quat_wxyz, dtype=np.float64).reshape(4), -1)
    ).as_matrix()
    transform[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return transform


def _object_joint_names_from_scene(payload: dict, object_idx: int) -> tuple[str, ...]:
    names_raw = np.asarray(payload["scene_object_arti_names"], dtype=object)[object_idx]
    valid_mask = np.asarray(payload["scene_object_arti_valid_mask"], dtype=bool)[
        object_idx
    ]
    names = np.asarray(names_raw, dtype=object).reshape(-1)
    valid_count = int(np.count_nonzero(valid_mask))
    if names.shape != (valid_count,):
        raise ValueError(
            "Expected scene_object_arti_names length to match valid articulation "
            f"count for object {object_idx}, got {names.shape} and {valid_count}."
        )
    return tuple(str(name) for name in names.tolist())


def _object_joint_values_from_scene(
    payload: dict, frame_idx: int, object_idx: int
) -> np.ndarray:
    values = np.asarray(payload["scene_object_arti_vals"], dtype=float)
    valid_mask = np.asarray(payload["scene_object_arti_valid_mask"], dtype=bool)
    if values.ndim != 3:
        raise ValueError(
            f"Expected scene_object_arti_vals shape (T, N, Amax), got {values.shape}."
        )
    if valid_mask.shape != values.shape[1:]:
        raise ValueError(
            f"Expected scene_object_arti_valid_mask shape {values.shape[1:]}, "
            f"got {valid_mask.shape}."
        )
    return values[frame_idx, object_idx, valid_mask[object_idx]].reshape(-1)


def _scene_object_inputs_for_frame(
    payload: dict,
    frame_idx: int,
    *,
    include_table: bool = True,
) -> tuple[DrakeObjectFrameInput, ...]:
    if "scene_object_urdf_paths" not in payload:
        object_joint_names = tuple(
            str(name)
            for name in np.asarray(
                payload.get("retargeted_joint_names", []), dtype=object
            ).tolist()
        )
        return (
            DrakeObjectFrameInput(
                urdf_path=_target_object_urdf_path(payload),
                pose=_pose_matrix(
                    np.asarray(payload["retargeted_obj_pos"], dtype=float)[frame_idx],
                    np.asarray(payload["retargeted_obj_quat"], dtype=float)[frame_idx],
                ),
                joint_names=object_joint_names,
                joint_values=_object_joint_values_for_frame(payload, frame_idx),
            ),
        )

    urdf_paths = np.asarray(payload["scene_object_urdf_paths"], dtype=object)
    object_pos = np.asarray(payload["scene_object_pos"], dtype=float)
    object_quat = np.asarray(payload["scene_object_quat"], dtype=float)
    if object_pos.ndim != 3 or object_pos.shape[2] != 3:
        raise ValueError(
            f"Expected scene_object_pos shape (T, N, 3), got {object_pos.shape}."
        )
    if object_quat.shape != (object_pos.shape[0], object_pos.shape[1], 4):
        raise ValueError(
            f"Expected scene_object_quat shape {(object_pos.shape[0], object_pos.shape[1], 4)}, "
            f"got {object_quat.shape}."
        )
    if urdf_paths.shape != (object_pos.shape[1],):
        raise ValueError(
            f"Expected scene_object_urdf_paths shape {(object_pos.shape[1],)}, got "
            f"{urdf_paths.shape}."
        )

    inputs: list[DrakeObjectFrameInput] = []
    for object_idx, urdf_path in enumerate(urdf_paths.tolist()):
        inputs.append(
            DrakeObjectFrameInput(
                urdf_path=_resolve_stored_path(Path(str(urdf_path))),
                pose=_pose_matrix(
                    object_pos[frame_idx, object_idx],
                    object_quat[frame_idx, object_idx],
                ),
                joint_names=_object_joint_names_from_scene(payload, object_idx),
                joint_values=_object_joint_values_from_scene(
                    payload, frame_idx, object_idx
                ),
            )
        )
    if include_table and "scene_table_height" in payload:
        table_height = float(
            _unwrap_scalar(payload["scene_table_height"], "scene_table_height")
        )
        inputs.append(
            DrakeObjectFrameInput(
                urdf_path=_table_urdf_path(table_height),
                pose=np.eye(4, dtype=np.float64),
                joint_names=(),
                joint_values=np.zeros(0, dtype=float),
            )
        )
    return tuple(inputs)


def _build_frame_plant(
    *,
    robot_urdf_path: Path,
    object_inputs: tuple[DrakeObjectFrameInput, ...],
) -> DrakeFramePlant:
    if len(object_inputs) == 0:
        raise ValueError("Expected at least one scene object for Drake refinement.")
    builder = RobotDiagramBuilder(time_step=0.001)
    plant = builder.plant()
    parser = builder.parser()
    if hasattr(parser, "SetAutoRenaming"):
        parser.SetAutoRenaming(True)
    hand_model = parser.AddModels(
        str(process_urdf(robot_urdf_path, append_jobname=True))
    )[0]
    object_models = [
        parser.AddModels(str(process_urdf(object_input.urdf_path)))[0]
        for object_input in object_inputs
    ]

    world_frame = plant.world_frame()
    plant.WeldFrames(world_frame, plant.GetFrameByName("root", hand_model))
    for object_model, object_input in zip(object_models, object_inputs, strict=True):
        object_body_indices = plant.GetBodyIndices(object_model)
        if len(object_body_indices) == 0:
            raise ValueError(f"Object URDF has no bodies: {object_input.urdf_path}")
        object_frame = plant.get_body(object_body_indices[0]).body_frame()
        plant.WeldFrames(world_frame, object_frame, RigidTransform(object_input.pose))

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    hand_geom_ids = []
    for body_idx in plant.GetBodyIndices(hand_model):
        hand_geom_ids.extend(
            plant.GetCollisionGeometriesForBody(plant.get_body(body_idx))
        )
    scene_geom_ids = []
    for object_model in object_models:
        for body_idx in plant.GetBodyIndices(object_model):
            scene_geom_ids.extend(
                plant.GetCollisionGeometriesForBody(plant.get_body(body_idx))
            )
    hand_scene_geom_pairs = tuple(
        (hand_geom_id, scene_geom_id)
        for hand_geom_id in hand_geom_ids
        for scene_geom_id in scene_geom_ids
    )
    if not hand_scene_geom_pairs:
        raise ValueError("Expected at least one hand-scene collision geometry pair.")
    drake_object_models = tuple(
        DrakeObjectModel(
            model=object_model,
            joint_names=tuple(
                plant.GetPositionNames(
                    object_model,
                    add_model_instance_prefix=False,
                    always_add_suffix=False,
                )
            ),
        )
        for object_model in object_models
    )
    lower_limits, upper_limits = _model_position_limits(plant, hand_model)

    return DrakeFramePlant(
        diagram=diagram,
        plant=plant,
        root_context=context,
        context=plant_context,
        hand_model=hand_model,
        object_models=drake_object_models,
        hand_scene_geom_pairs=hand_scene_geom_pairs,
        hand_joint_names=tuple(
            plant.GetPositionNames(
                hand_model,
                add_model_instance_prefix=False,
                always_add_suffix=False,
            )
        ),
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )


def _model_position_limits(
    plant: object, model_instance: object
) -> tuple[np.ndarray, np.ndarray]:
    full_lower = np.asarray(plant.GetPositionLowerLimits(), dtype=float)
    full_upper = np.asarray(plant.GetPositionUpperLimits(), dtype=float)
    position_indices: list[int] = []
    for joint_idx in plant.GetJointIndices(model_instance):
        joint = plant.get_joint(joint_idx)
        start = int(joint.position_start())
        count = int(joint.num_positions())
        position_indices.extend(range(start, start + count))
    expected_count = int(plant.num_positions(model_instance))
    if len(position_indices) != expected_count:
        raise ValueError(
            f"Expected {expected_count} model positions, got {len(position_indices)} "
            f"from model joint indices."
        )
    return full_lower[position_indices], full_upper[position_indices]


def _object_joint_values_for_frame(payload: dict, frame_idx: int) -> np.ndarray:
    values = np.asarray(
        payload.get("retargeted_obj_arti", np.zeros((1, 0))), dtype=float
    )
    if values.ndim != 2:
        raise ValueError(
            f"Expected retargeted_obj_arti shape (T, A), got {values.shape}."
        )
    if values.shape[1] == 0:
        return np.zeros(0, dtype=float)
    return values[frame_idx].reshape(-1)


def _validate_frame_plant(
    frame_plant: DrakeFramePlant,
    hand_joint_names: list[str],
    object_inputs: tuple[DrakeObjectFrameInput, ...],
) -> None:
    if list(frame_plant.hand_joint_names) != hand_joint_names:
        raise ValueError(
            "Expected payload hand joint names to exactly match Drake plant positions.\n"
            f"payload={hand_joint_names}\n"
            f"plant={list(frame_plant.hand_joint_names)}"
        )
    if len(frame_plant.object_models) != len(object_inputs):
        raise ValueError(
            "Expected Drake object model count to match scene input count, got "
            f"{len(frame_plant.object_models)} and {len(object_inputs)}."
        )
    for object_idx, (drake_object, object_input) in enumerate(
        zip(frame_plant.object_models, object_inputs, strict=True)
    ):
        if drake_object.joint_names != object_input.joint_names:
            raise ValueError(
                "Expected payload object joint names to exactly match Drake plant "
                f"positions for scene object {object_idx}.\n"
                f"payload={list(object_input.joint_names)}\n"
                f"plant={list(drake_object.joint_names)}"
            )
        if len(object_input.joint_names) != object_input.joint_values.shape[0]:
            raise ValueError(
                "Expected object articulation values to match Drake object "
                f"positions for scene object {object_idx}. "
                f"payload_shape={object_input.joint_values.shape}, "
                f"plant_object_positions={list(drake_object.joint_names)}"
            )


def _root_joint_mask(joint_names: list[str]) -> np.ndarray:
    return np.asarray(
        [
            any(str(name).endswith(token) for token in ROOT_JOINT_TOKENS)
            for name in joint_names
        ],
        dtype=bool,
    )


def _root_rotation_joint_indices(joint_names: list[str]) -> np.ndarray:
    return np.asarray(
        [
            idx
            for idx, name in enumerate(joint_names)
            if any(str(name).endswith(token) for token in ROOT_ROTATION_JOINT_TOKENS)
        ],
        dtype=np.int64,
    )


def _periodic_values_in_limits(value: float, lower: float, upper: float) -> np.ndarray:
    if not np.isfinite(value) or not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(
            f"Expected finite periodic remap inputs, got value={value}, "
            f"lower={lower}, upper={upper}."
        )
    if lower > upper:
        raise ValueError(f"Expected lower <= upper, got {lower} > {upper}.")
    k_min = int(np.ceil((lower - value) / TWO_PI))
    k_max = int(np.floor((upper - value) / TWO_PI))
    if k_min > k_max:
        raise ValueError(
            "Could not map wrist rotation value into joint limits by adding "
            f"2*pi*k: value={value}, limits=({lower}, {upper})."
        )
    ks = np.arange(k_min, k_max + 1, dtype=np.int64)
    candidates = value + TWO_PI * ks.astype(np.float64)
    valid = (candidates >= lower - 1.0e-9) & (candidates <= upper + 1.0e-9)
    candidates = candidates[valid]
    if candidates.size == 0:
        raise ValueError(
            "No numerically valid wrist rotation candidate after periodic remap: "
            f"value={value}, limits=({lower}, {upper}), k_range=({k_min}, {k_max})."
        )
    return np.clip(candidates, lower, upper)


def _remap_periodic_sequence_to_limits(
    values: np.ndarray,
    *,
    joint_idx: int,
    joint_name: str,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, int, float]:
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    lower = max(float(lower), float(WRIST_ROTATION_REMAP_LOWER))
    upper = min(float(upper), float(WRIST_ROTATION_REMAP_UPPER))
    if lower > upper:
        raise ValueError(
            f"Wrist rotation joint {joint_name!r} at index {joint_idx} has no "
            "overlap between URDF limits and the requested principal branch "
            f"[{-np.pi}, {np.pi}]."
        )
    mapped = np.empty_like(source)
    changed = 0
    max_abs_delta = 0.0
    previous: float | None = None
    for frame_idx, raw_value in enumerate(source):
        candidates = _periodic_values_in_limits(float(raw_value), lower, upper)
        target = float(raw_value) if previous is None else previous
        candidate_idx = int(np.argmin(np.abs(candidates - target)))
        mapped_value = float(candidates[candidate_idx])
        delta = abs(mapped_value - float(raw_value))
        if delta > 1.0e-6:
            changed += 1
            max_abs_delta = max(max_abs_delta, delta)
        mapped[frame_idx] = mapped_value
        previous = mapped_value

    if np.any(mapped < lower - 1.0e-6) or np.any(mapped > upper + 1.0e-6):
        raise ValueError(
            f"Remapped wrist rotation joint {joint_name!r} at index {joint_idx} "
            f"outside limits ({lower}, {upper})."
        )
    return mapped, changed, max_abs_delta


def _remap_wrist_rotations_to_limits(
    q_values: np.ndarray,
    *,
    joint_names: list[str],
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    remapped = np.asarray(q_values, dtype=np.float64).copy()
    if remapped.ndim != 2:
        raise ValueError(f"Expected q_values shape (T, J), got {remapped.shape}.")
    if remapped.shape[1] != len(joint_names):
        raise ValueError(
            f"Expected q_values joint dim {remapped.shape[1]} to match "
            f"{len(joint_names)} joint names."
        )
    changed_total = 0
    max_abs_delta = 0.0
    for joint_idx in _root_rotation_joint_indices(joint_names).tolist():
        mapped, changed, joint_max_delta = _remap_periodic_sequence_to_limits(
            remapped[:, joint_idx],
            joint_idx=joint_idx,
            joint_name=joint_names[joint_idx],
            lower=float(lower_limits[joint_idx]),
            upper=float(upper_limits[joint_idx]),
        )
        remapped[:, joint_idx] = mapped
        changed_total += changed
        max_abs_delta = max(max_abs_delta, joint_max_delta)
    return remapped, changed_total, max_abs_delta


def _movable_joint_indices(
    joint_names: list[str], refine_root_joints: bool
) -> np.ndarray:
    mask = np.ones(len(joint_names), dtype=bool)
    if not refine_root_joints:
        mask &= ~_root_joint_mask(joint_names)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(
            "No movable joints selected; pass --refine-root-joints to include root joints."
        )
    return indices


def _set_frame_positions(
    frame_plant: DrakeFramePlant,
    q_hand: np.ndarray,
    object_inputs: tuple[DrakeObjectFrameInput, ...],
) -> None:
    q_hand = np.asarray(q_hand, dtype=float).reshape(-1)
    frame_plant.plant.SetPositions(
        frame_plant.context,
        frame_plant.hand_model,
        q_hand,
    )
    for drake_object, object_input in zip(
        frame_plant.object_models, object_inputs, strict=True
    ):
        q_object = np.asarray(object_input.joint_values, dtype=float).reshape(-1)
        if not q_object.size:
            continue
        frame_plant.plant.SetPositions(
            frame_plant.context,
            drake_object.model,
            q_object,
        )


def _signed_distances(
    frame_plant: DrakeFramePlant,
    q_hand: np.ndarray,
    object_inputs: tuple[DrakeObjectFrameInput, ...],
) -> np.ndarray:
    _set_frame_positions(frame_plant, q_hand, object_inputs)
    query_port = frame_plant.plant.get_geometry_query_input_port()
    if not query_port.HasValue(frame_plant.context):
        raise RuntimeError("Drake geometry query input port has no value.")
    query_object = query_port.Eval(frame_plant.context)
    distances = np.empty(len(frame_plant.hand_scene_geom_pairs), dtype=float)
    for pair_idx, (hand_geom_id, scene_geom_id) in enumerate(
        frame_plant.hand_scene_geom_pairs
    ):
        distances[pair_idx] = query_object.ComputeSignedDistancePairClosestPoints(
            hand_geom_id,
            scene_geom_id,
        ).distance
    return distances


def _collision_violation(
    frame_plant: DrakeFramePlant,
    q_hand: np.ndarray,
    object_inputs: tuple[DrakeObjectFrameInput, ...],
    collision_distance: float,
) -> float:
    distances = _signed_distances(frame_plant, q_hand, object_inputs)
    return float(max(0.0, float(collision_distance) - float(np.min(distances))))


def _joint_radius(name: str, cfg: DrakeLineSearchConfig) -> float:
    if name.endswith(
        ("forearm_tx_link_joint", "forearm_ty_link_joint", "forearm_tz_link_joint")
    ):
        return float(cfg.root_translation_radius)
    if name.endswith(
        (
            "forearm_roll_link_joint",
            "forearm_pitch_link_joint",
            "forearm_yaw_link_joint",
        )
    ):
        return float(cfg.root_rotation_radius)
    return float(cfg.joint_radius)


def _candidate_values(
    *,
    q_reference: np.ndarray,
    q_current: np.ndarray,
    joint_idx: int,
    joint_names: list[str],
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    cfg: DrakeLineSearchConfig,
) -> np.ndarray:
    radius = _joint_radius(joint_names[joint_idx], cfg)
    if radius <= 0.0:
        return np.asarray([q_reference[joint_idx], q_current[joint_idx]], dtype=float)
    lower = max(float(lower_limits[joint_idx]), float(q_reference[joint_idx]) - radius)
    upper = min(float(upper_limits[joint_idx]), float(q_reference[joint_idx]) + radius)
    if lower > upper:
        raise ValueError(
            f"Line-search interval is empty for joint {joint_names[joint_idx]!r}: "
            f"[{lower}, {upper}]."
        )
    return np.unique(
        np.concatenate(
            [
                np.linspace(lower, upper, int(cfg.line_search_samples), dtype=float),
                np.asarray(
                    [q_reference[joint_idx], q_current[joint_idx]],
                    dtype=float,
                ),
            ]
        )
    )


def _radius_bound_violation(
    q_candidate: np.ndarray,
    q_reference: np.ndarray,
    joint_names: list[str],
    cfg: DrakeLineSearchConfig,
) -> float:
    q_candidate = np.asarray(q_candidate, dtype=float).reshape(-1)
    q_reference = np.asarray(q_reference, dtype=float).reshape(-1)
    if q_candidate.shape != q_reference.shape:
        raise ValueError(
            "Expected q_candidate and q_reference to have matching shapes, got "
            f"{q_candidate.shape} and {q_reference.shape}."
        )
    if q_candidate.shape[0] != len(joint_names):
        raise ValueError(
            f"Expected {q_candidate.shape[0]} joint values to match "
            f"{len(joint_names)} joint names."
        )
    max_violation = 0.0
    for joint_idx, joint_name in enumerate(joint_names):
        radius = _joint_radius(joint_name, cfg)
        if radius < 0.0:
            raise ValueError(
                f"Expected non-negative radius for joint {joint_name!r}, got {radius}."
            )
        delta = abs(float(q_candidate[joint_idx] - q_reference[joint_idx]))
        max_violation = max(max_violation, delta - radius)
    return float(max(0.0, max_violation))


def _open_hand_distance(
    q_candidate: np.ndarray,
    joint_names: list[str],
    open_qpos: dict[str, float],
) -> float:
    q_candidate = np.asarray(q_candidate, dtype=float).reshape(-1)
    if q_candidate.shape != (len(joint_names),):
        raise ValueError(
            f"Expected q_candidate shape {(len(joint_names),)}, got "
            f"{q_candidate.shape}."
        )
    open_joint_indices = [
        idx
        for idx, joint_name in enumerate(joint_names)
        if joint_name in open_qpos
        and not any(str(joint_name).endswith(token) for token in ROOT_JOINT_TOKENS)
    ]
    if not open_joint_indices:
        raise ValueError(
            "Expected at least one refinement joint to appear in robot open_qpos."
        )
    return float(
        np.sum(
            [
                abs(float(q_candidate[idx]) - float(open_qpos[joint_names[idx]]))
                for idx in open_joint_indices
            ]
        )
    )


def _refine_frame(
    *,
    frame_plant: DrakeFramePlant,
    q_reference: np.ndarray,
    q_seeds: tuple[np.ndarray, ...],
    object_inputs: tuple[DrakeObjectFrameInput, ...],
    joint_names: list[str],
    open_qpos: dict[str, float],
    movable_indices: np.ndarray,
    cfg: DrakeLineSearchConfig,
) -> tuple[np.ndarray, float, float]:
    q_reference = np.asarray(q_reference, dtype=float).reshape(-1)
    q_reference = np.clip(
        q_reference, frame_plant.lower_limits, frame_plant.upper_limits
    )
    if len(q_seeds) == 0:
        raise ValueError("Expected at least one seed for Drake refinement.")
    reference_violation = _collision_violation(
        frame_plant,
        q_reference,
        object_inputs,
        cfg.refinement_collision_distance,
    )
    if reference_violation <= 1e-12:
        return q_reference, reference_violation, 0.0

    best_q = q_reference.copy()
    reference_open_hand_distance = (
        _open_hand_distance(q_reference, joint_names, open_qpos)
        if cfg.prefer_open_qpos_hull
        else 0.0
    )
    best_key = (
        reference_violation > 1e-12,
        reference_violation,
        reference_open_hand_distance,
        0.0,
    )

    for raw_seed in q_seeds:
        q_current = np.asarray(raw_seed, dtype=float).reshape(-1)
        if q_current.shape != q_reference.shape:
            raise ValueError(
                "Expected Drake refinement seed shape "
                f"{q_reference.shape}, got {q_current.shape}."
            )
        q_current = np.clip(
            q_current, frame_plant.lower_limits, frame_plant.upper_limits
        )
        initial_violation = _collision_violation(
            frame_plant,
            q_current,
            object_inputs,
            cfg.refinement_collision_distance,
        )
        seed_best_q = q_current.copy()
        seed_open_hand_distance = (
            _open_hand_distance(q_current, joint_names, open_qpos)
            if cfg.prefer_open_qpos_hull
            else 0.0
        )
        seed_best_key = (
            initial_violation > 1e-12,
            initial_violation,
            seed_open_hand_distance,
            float(np.sum(np.abs(q_current - q_reference))),
        )
        if seed_best_key < best_key:
            best_key = seed_best_key
            best_q = seed_best_q.copy()

        for _ in range(int(cfg.line_search_passes)):
            for joint_idx in movable_indices.tolist():
                for joint_value in _candidate_values(
                    q_reference=q_reference,
                    q_current=seed_best_q,
                    joint_idx=joint_idx,
                    joint_names=joint_names,
                    lower_limits=frame_plant.lower_limits,
                    upper_limits=frame_plant.upper_limits,
                    cfg=cfg,
                ):
                    q_candidate = seed_best_q.copy()
                    q_candidate[joint_idx] = joint_value
                    violation = _collision_violation(
                        frame_plant,
                        q_candidate,
                        object_inputs,
                        cfg.refinement_collision_distance,
                    )
                    l1_delta = float(np.sum(np.abs(q_candidate - q_reference)))
                    open_hand_distance = (
                        _open_hand_distance(q_candidate, joint_names, open_qpos)
                        if cfg.prefer_open_qpos_hull
                        else 0.0
                    )
                    key = (
                        violation > 1e-12,
                        violation,
                        open_hand_distance,
                        l1_delta,
                    )
                    if key < seed_best_key:
                        seed_best_key = key
                        seed_best_q = q_candidate
                    if key < best_key:
                        best_key = key
                        best_q = q_candidate

    return best_q, reference_violation, float(best_key[1])


def _backward_refinement_cfg(cfg: DrakeLineSearchConfig) -> DrakeLineSearchConfig:
    return replace(
        cfg,
        joint_radius=float(cfg.backward_joint_radius),
        root_translation_radius=float(cfg.backward_root_translation_radius),
        root_rotation_radius=float(cfg.backward_root_rotation_radius),
    )


def _run_backward_refinement_pass(
    *,
    payload: dict,
    start: int,
    end: int,
    robot_urdf_path: Path,
    joint_names: list[str],
    open_qpos: dict[str, float],
    movable_indices: np.ndarray,
    q_values_for_refinement: np.ndarray,
    refined_values: np.ndarray,
    valid_frame_mask: np.ndarray,
    anchor_valid_frame_mask: np.ndarray,
    unresolved_frame_mask: np.ndarray,
    final_collision_violation: np.ndarray,
    acceptance_collision_violation: np.ndarray,
    radius_bound_violation: np.ndarray,
    l1_deltas: np.ndarray,
    cfg: DrakeLineSearchConfig,
) -> None:
    for frame_idx in range(end - 1, start - 1, -1):
        if valid_frame_mask[frame_idx]:
            continue
        future_end = min(end, frame_idx + 1 + int(cfg.backward_max_future_frame_gap))
        future_valid_indices = np.flatnonzero(
            anchor_valid_frame_mask[frame_idx + 1 : future_end]
        )
        if future_valid_indices.size == 0:
            continue
        nearest_future_idx = frame_idx + 1 + int(future_valid_indices[0])
        object_inputs = _scene_object_inputs_for_frame(payload, frame_idx)
        frame_plant = _build_frame_plant(
            robot_urdf_path=robot_urdf_path,
            object_inputs=object_inputs,
        )
        _validate_frame_plant(frame_plant, joint_names, object_inputs)
        q_future = np.asarray(refined_values[nearest_future_idx], dtype=float)
        q_new, _, final_violation = _refine_frame(
            frame_plant=frame_plant,
            q_reference=q_future,
            q_seeds=(q_future,),
            object_inputs=object_inputs,
            joint_names=joint_names,
            open_qpos=open_qpos,
            movable_indices=movable_indices,
            cfg=cfg,
        )
        final_acceptance_violation = _collision_violation(
            frame_plant,
            q_new,
            object_inputs,
            cfg.valid_collision_distance,
        )
        final_radius_violation = _radius_bound_violation(
            q_new,
            q_future,
            joint_names,
            cfg,
        )
        if final_acceptance_violation > 1e-12 or final_radius_violation > 1e-12:
            continue
        refined_values[frame_idx] = q_new.astype(np.float32)
        valid_frame_mask[frame_idx] = True
        unresolved_frame_mask[frame_idx] = False
        final_collision_violation[frame_idx] = np.float32(final_violation)
        acceptance_collision_violation[frame_idx] = np.float32(
            final_acceptance_violation
        )
        radius_bound_violation[frame_idx] = np.float32(final_radius_violation)
        l1_deltas[frame_idx] = float(
            np.sum(np.abs(q_new - q_values_for_refinement[frame_idx]))
        )


def refine_payload_with_drake(
    payload: dict,
    cfg: DrakeLineSearchConfig,
) -> tuple[dict, DrakeLineSearchStats]:
    if cfg.required_collision_free_start_frames < 0:
        raise ValueError(
            "Expected required_collision_free_start_frames to be non-negative, got "
            f"{cfg.required_collision_free_start_frames}."
        )
    if cfg.backward_max_future_frame_gap < 1:
        raise ValueError(
            "Expected backward_max_future_frame_gap >= 1, got "
            f"{cfg.backward_max_future_frame_gap}."
        )
    n_frames = _payload_num_frames(payload)
    start, end = _frame_range(n_frames, cfg)
    required_start_end = min(
        end,
        start + int(cfg.required_collision_free_start_frames),
    )
    side = _selected_hand(payload, cfg.hand)
    q_values = _joint_values_from_payload(payload, side)
    joint_names = _joint_names_from_payload(payload, side)
    if q_values.shape[0] != n_frames:
        raise ValueError(
            f"Expected {side}_jvals frame count {q_values.shape[0]} to match {n_frames}."
        )

    robot_model = _normalize_robot_model(payload.get("robot_model", "xhand"))
    robot_urdf_path = _robot_urdf_path(robot_model, side)
    open_qpos = _robot_open_qpos(robot_model, side)

    movable_indices = _movable_joint_indices(joint_names, cfg.refine_root_joints)
    probe_object_inputs = _scene_object_inputs_for_frame(payload, start)
    wrist_remap_probe_plant = _build_frame_plant(
        robot_urdf_path=robot_urdf_path,
        object_inputs=probe_object_inputs,
    )
    _validate_frame_plant(
        wrist_remap_probe_plant,
        joint_names,
        probe_object_inputs,
    )
    q_values_for_refinement, wrist_remapped_count, wrist_remap_max_abs_delta = (
        _remap_wrist_rotations_to_limits(
            q_values,
            joint_names=joint_names,
            lower_limits=wrist_remap_probe_plant.lower_limits,
            upper_limits=wrist_remap_probe_plant.upper_limits,
        )
    )
    refined_values = q_values_for_refinement.copy()
    previous_refined_q: np.ndarray | None = None
    previous_refinement_valid = False
    initially_colliding = 0
    unresolved_frame_mask = np.zeros(n_frames, dtype=bool)
    valid_frame_mask = np.zeros(n_frames, dtype=bool)
    final_collision_violation = np.zeros(n_frames, dtype=np.float32)
    acceptance_collision_violation = np.zeros(n_frames, dtype=np.float32)
    radius_bound_violation = np.zeros(n_frames, dtype=np.float32)
    l1_deltas = np.zeros(n_frames, dtype=np.float64)

    for frame_idx in tqdm(range(start, end), total=(end - start)):
        object_inputs = _scene_object_inputs_for_frame(payload, frame_idx)
        frame_plant = _build_frame_plant(
            robot_urdf_path=robot_urdf_path,
            object_inputs=object_inputs,
        )
        _validate_frame_plant(
            frame_plant,
            joint_names,
            object_inputs,
        )
        q_reference = q_values_for_refinement[frame_idx]
        q_seeds = (q_reference,)
        if previous_refinement_valid:
            if previous_refined_q is None:
                raise ValueError(
                    "Previous refinement is marked valid but no previous refined "
                    "joint values are available."
                )
            q_seeds = (q_reference, previous_refined_q)

        q_new, initial_violation, final_violation = _refine_frame(
            frame_plant=frame_plant,
            q_reference=q_reference,
            q_seeds=q_seeds,
            object_inputs=object_inputs,
            joint_names=joint_names,
            open_qpos=open_qpos,
            movable_indices=movable_indices,
            cfg=cfg,
        )
        if initial_violation > 1e-12:
            initially_colliding += 1
        final_acceptance_violation = _collision_violation(
            frame_plant,
            q_new,
            object_inputs,
            cfg.valid_collision_distance,
        )
        final_radius_violation = _radius_bound_violation(
            q_new,
            q_values_for_refinement[frame_idx],
            joint_names,
            cfg,
        )
        is_valid = (
            final_acceptance_violation <= 1e-12 and final_radius_violation <= 1e-12
        )
        if not is_valid:
            unresolved_frame_mask[frame_idx] = True
            refined_values[frame_idx] = q_values_for_refinement[frame_idx].astype(
                np.float32
            )
            final_collision_violation[frame_idx] = np.float32(final_violation)
            acceptance_collision_violation[frame_idx] = np.float32(
                final_acceptance_violation
            )
            radius_bound_violation[frame_idx] = np.float32(final_radius_violation)
            previous_refined_q = None
            previous_refinement_valid = False
            continue

        refined_values[frame_idx] = q_new.astype(np.float32)
        valid_frame_mask[frame_idx] = True
        final_collision_violation[frame_idx] = np.float32(final_violation)
        acceptance_collision_violation[frame_idx] = np.float32(
            final_acceptance_violation
        )
        radius_bound_violation[frame_idx] = np.float32(final_radius_violation)
        previous_refined_q = q_new
        previous_refinement_valid = True
        l1_deltas[frame_idx] = float(
            np.sum(np.abs(q_new - q_values_for_refinement[frame_idx]))
        )

    backward_anchor_valid_frame_mask = valid_frame_mask.copy()
    _run_backward_refinement_pass(
        payload=payload,
        start=start,
        end=end,
        robot_urdf_path=robot_urdf_path,
        joint_names=joint_names,
        open_qpos=open_qpos,
        movable_indices=movable_indices,
        q_values_for_refinement=q_values_for_refinement,
        refined_values=refined_values,
        valid_frame_mask=valid_frame_mask,
        anchor_valid_frame_mask=backward_anchor_valid_frame_mask,
        unresolved_frame_mask=unresolved_frame_mask,
        final_collision_violation=final_collision_violation,
        acceptance_collision_violation=acceptance_collision_violation,
        radius_bound_violation=radius_bound_violation,
        l1_deltas=l1_deltas,
        cfg=cfg,
    )
    _run_backward_refinement_pass(
        payload=payload,
        start=start,
        end=end,
        robot_urdf_path=robot_urdf_path,
        joint_names=joint_names,
        open_qpos=open_qpos,
        movable_indices=movable_indices,
        q_values_for_refinement=q_values_for_refinement,
        refined_values=refined_values,
        valid_frame_mask=valid_frame_mask,
        anchor_valid_frame_mask=backward_anchor_valid_frame_mask,
        unresolved_frame_mask=unresolved_frame_mask,
        final_collision_violation=final_collision_violation,
        acceptance_collision_violation=acceptance_collision_violation,
        radius_bound_violation=radius_bound_violation,
        l1_deltas=l1_deltas,
        cfg=_backward_refinement_cfg(cfg),
    )

    for frame_idx in range(start, required_start_end):
        if valid_frame_mask[frame_idx]:
            continue
        if not cfg.fail_on_unresolved:
            continue
        future_valid_indices = np.flatnonzero(valid_frame_mask[frame_idx + 1 : end])
        if future_valid_indices.size == 0:
            raise RuntimeError(
                "Could not repair required collision-free start frame "
                f"{frame_idx}: no future valid refined frame exists in "
                f"[{frame_idx + 1}, {end})."
            )
        nearest_future_idx = frame_idx + 1 + int(future_valid_indices[0])
        raise RuntimeError(
            "Could not repair required collision-free start frame "
            f"{frame_idx} from future valid frame {nearest_future_idx} "
            "after normal and wide backward searches."
        )

    if cfg.fail_on_unresolved and np.any(unresolved_frame_mask[start:end]):
        unresolved_indices = np.flatnonzero(unresolved_frame_mask[start:end]) + start
        raise RuntimeError(
            "Could not find valid refined frames after forward and backward "
            f"passes: {unresolved_indices.astype(int).tolist()}."
        )

    refined_payload = dict(payload)
    refined_payload[f"drake_collision_refinement_{side}_jvals"] = refined_values.astype(
        np.float32
    )
    refined_payload["drake_collision_refinement_jvals"] = refined_values.astype(
        np.float32
    )
    refined_payload["drake_collision_refinement_jnames"] = np.asarray(
        joint_names, dtype=object
    )
    refined_payload["drake_collision_refinement_input_path"] = str(cfg.input_path)
    refined_payload["drake_collision_refinement_hand"] = side
    refined_payload["drake_collision_refinement_start_frame"] = np.array(
        start, dtype=np.int64
    )
    refined_payload["drake_collision_refinement_end_frame"] = np.array(
        end, dtype=np.int64
    )
    refined_payload["drake_collision_refinement_unresolved"] = np.array(
        np.count_nonzero(unresolved_frame_mask), dtype=np.int64
    )
    refined_payload["drake_collision_refinement_unresolved_frame_mask"] = (
        unresolved_frame_mask
    )
    refined_payload["drake_collision_refinement_unresolved_frame_indices"] = (
        np.flatnonzero(unresolved_frame_mask).astype(np.int64)
    )
    invalid_frame_mask = ~valid_frame_mask
    refined_payload["drake_collision_refinement_valid_frame_mask"] = valid_frame_mask
    refined_payload["drake_collision_refinement_invalid_frame_mask"] = (
        invalid_frame_mask
    )
    refined_payload["drake_collision_refinement_valid_frame_indices"] = np.flatnonzero(
        valid_frame_mask
    ).astype(np.int64)
    refined_payload["drake_collision_refinement_invalid_frame_indices"] = (
        np.flatnonzero(invalid_frame_mask).astype(np.int64)
    )
    refined_payload["drake_collision_refinement_final_collision_violation"] = (
        final_collision_violation
    )
    refined_payload["drake_collision_refinement_acceptance_collision_violation"] = (
        acceptance_collision_violation
    )
    refined_payload["drake_collision_refinement_radius_bound_violation"] = (
        radius_bound_violation
    )
    refined_payload["drake_collision_refinement_collision_distance"] = np.array(
        cfg.refinement_collision_distance, dtype=np.float32
    )
    refined_payload["drake_collision_refinement_refinement_collision_distance"] = (
        np.array(cfg.refinement_collision_distance, dtype=np.float32)
    )
    refined_payload["drake_collision_refinement_valid_collision_distance"] = np.array(
        cfg.valid_collision_distance, dtype=np.float32
    )
    refined_payload[
        "drake_collision_refinement_required_collision_free_start_frames"
    ] = np.array(cfg.required_collision_free_start_frames, dtype=np.int64)
    refined_payload["drake_collision_refinement_backward_max_future_frame_gap"] = (
        np.array(cfg.backward_max_future_frame_gap, dtype=np.int64)
    )
    refined_payload["drake_collision_refinement_backward_joint_radius"] = np.array(
        cfg.backward_joint_radius, dtype=np.float32
    )
    refined_payload["drake_collision_refinement_backward_root_translation_radius"] = (
        np.array(cfg.backward_root_translation_radius, dtype=np.float32)
    )
    refined_payload["drake_collision_refinement_backward_root_rotation_radius"] = (
        np.array(cfg.backward_root_rotation_radius, dtype=np.float32)
    )
    refined_payload["wrist_rotation_periodic_remap_hand"] = side
    refined_payload["wrist_rotation_periodic_remap_jnames"] = np.asarray(
        [joint_names[idx] for idx in _root_rotation_joint_indices(joint_names)],
        dtype=object,
    )
    refined_payload["wrist_rotation_periodic_remap_changed_values"] = np.array(
        wrist_remapped_count,
        dtype=np.int64,
    )
    refined_payload["wrist_rotation_periodic_remap_range"] = np.asarray(
        [WRIST_ROTATION_REMAP_LOWER, WRIST_ROTATION_REMAP_UPPER],
        dtype=np.float32,
    )
    refined_payload["wrist_rotation_periodic_remap_max_abs_delta"] = np.array(
        wrist_remap_max_abs_delta,
        dtype=np.float32,
    )

    return refined_payload, DrakeLineSearchStats(
        total_frames=n_frames,
        frames=end - start,
        successfully_refined_frames=int(np.count_nonzero(valid_frame_mask)),
        changed=int(np.count_nonzero(l1_deltas > 1e-12)),
        initially_colliding=initially_colliding,
        unresolved=int(np.count_nonzero(unresolved_frame_mask)),
        mean_l1_delta=float(np.mean(l1_deltas[start:end])),
        max_l1_delta=float(np.max(l1_deltas[start:end])),
        max_radius_bound_violation=(
            float(np.max(radius_bound_violation))
            if radius_bound_violation.size
            else 0.0
        ),
        wrist_rotation_remapped_values=wrist_remapped_count,
        wrist_rotation_max_abs_delta=wrist_remap_max_abs_delta,
    )


def parse_args() -> DrakeLineSearchConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially refine an exported OakInk sequence payload using Drake "
            "per-frame plants and deterministic joint line search."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hand", choices=["auto", "left", "right"], default="auto")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Exclusive end frame. Defaults to the payload frame count.",
    )
    parser.add_argument("--line-search-samples", type=int, default=48)
    parser.add_argument("--line-search-passes", type=int, default=3)
    parser.add_argument("--joint-radius", type=float, default=0.2)
    parser.add_argument("--root-translation-radius", type=float, default=0.005)
    parser.add_argument("--root-rotation-radius", type=float, default=0.2)
    parser.add_argument(
        "--refinement-collision-distance",
        dest="refinement_collision_distance",
        type=float,
        default=0.001,
        help=(
            "Signed-distance clearance target used by the line search. "
            "Default requires hand links to be at least 1 mm from the object."
        ),
    )
    parser.add_argument(
        "--valid-collision-distance",
        type=float,
        default=-0.001,
        help=(
            "Signed-distance threshold used to decide whether to keep the refined "
            "solution. Default allows up to 1 mm penetration."
        ),
    )
    parser.add_argument(
        "--refine-root-joints",
        action="store_true",
        help="Allow the 6-DoF forearm/root joints to move during line search.",
    )
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Save best-effort frames even if some remain colliding.",
    )
    parser.add_argument(
        "--prefer-open-qpos-hull",
        action="store_true",
        help=(
            "Prefer otherwise-equivalent line-search candidates that are closer "
            "to the robot hull-open finger configuration."
        ),
    )
    parser.add_argument(
        "--required-collision-free-start-frames",
        type=int,
        default=2,
        help=(
            "Number of frames from --start-frame that must be collision-free. "
            "Invalid required start frames are repaired backward from the nearest "
            "future valid refined frame and still raise if repair fails."
        ),
    )
    parser.add_argument(
        "--backward-max-future-frame-gap",
        type=int,
        default=2,
        help=(
            "Maximum future-frame gap allowed when using backward refinement. "
            "Frames without a forward-valid future anchor within this many "
            "timesteps remain unresolved."
        ),
    )
    parser.add_argument("--backward-joint-radius", type=float, default=1.0)
    parser.add_argument("--backward-root-translation-radius", type=float, default=0.05)
    parser.add_argument("--backward-root-rotation-radius", type=float, default=1.0)
    args = parser.parse_args()
    return DrakeLineSearchConfig(
        input_path=args.input.resolve(),
        output_path=args.output.resolve(),
        hand=str(args.hand),
        start_frame=int(args.start_frame),
        end_frame=None if args.end_frame is None else int(args.end_frame),
        line_search_samples=int(args.line_search_samples),
        line_search_passes=int(args.line_search_passes),
        joint_radius=float(args.joint_radius),
        root_translation_radius=float(args.root_translation_radius),
        root_rotation_radius=float(args.root_rotation_radius),
        refinement_collision_distance=float(args.refinement_collision_distance),
        valid_collision_distance=float(args.valid_collision_distance),
        refine_root_joints=bool(args.refine_root_joints),
        fail_on_unresolved=not bool(args.allow_unresolved),
        prefer_open_qpos_hull=bool(args.prefer_open_qpos_hull),
        required_collision_free_start_frames=int(
            args.required_collision_free_start_frames
        ),
        backward_max_future_frame_gap=int(args.backward_max_future_frame_gap),
        backward_joint_radius=float(args.backward_joint_radius),
        backward_root_translation_radius=float(args.backward_root_translation_radius),
        backward_root_rotation_radius=float(args.backward_root_rotation_radius),
    )


def main() -> None:
    cfg = parse_args()
    payload = _load_payload(cfg.input_path)
    refined_payload, stats = refine_payload_with_drake(payload, cfg)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cfg.output_path, refined_payload, allow_pickle=True)
    print(f"Saved Drake-refined OakInk payload to {cfg.output_path}")
    print(
        f"frames={stats.frames} "
        f"successfully_refined_frames={stats.successfully_refined_frames}/{stats.total_frames} "
        f"changed={stats.changed} "
        f"initially_colliding={stats.initially_colliding} "
        f"unresolved={stats.unresolved} "
        f"mean_l1_delta={stats.mean_l1_delta:.6e} "
        f"max_l1_delta={stats.max_l1_delta:.6e} "
        f"max_radius_bound_violation={stats.max_radius_bound_violation:.6e} "
        f"wrist_rotation_remapped_values={stats.wrist_rotation_remapped_values} "
        f"wrist_rotation_max_abs_delta={stats.wrist_rotation_max_abs_delta:.6e}"
    )


if __name__ == "__main__":
    main()
