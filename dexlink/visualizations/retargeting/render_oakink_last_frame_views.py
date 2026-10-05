from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from dexlink.src.retargeting.data.exported_oakink_processor import ExportedOakInkProcessor
from dexlink.src.retargeting.data.hand_cfgs.inspire import INSPIRE_CFGs
from dexlink.utils.retargeting.path_utils import get_assets_dir, process_urdf
from dexlink.utils.retargeting.render_utils import TrimeshInstance, render_centered_four_views

# NumPy 2.x compatibility for urdfpy.
if not hasattr(np, "product"):
    np.product = np.prod

ROBOT_URDFS = {
    "inspire": {
        "left": INSPIRE_CFGs["left"]["urdf_path"],
        "right": INSPIRE_CFGs["right"]["urdf_path"],
    },
    "xhand": {
        "left": "assets/robots/xhand/xhand_left_6dof_abs_obj.urdf",
        "right": "assets/robots/xhand/xhand_right_6dof_abs_obj.urdf",
    },
    "franka_xhand": {
        "left": None,
        "right": None,
    },
}


def _normalize_robot_model(robot_model: str | None) -> str:
    if robot_model is None:
        return "auto"
    aliases = {
        "auto": "auto",
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
            "Unsupported robot_model "
            f"{robot_model!r}; use 'auto', 'inspire', 'xhand', or 'franka_xhand'."
        )
    return aliases[value]


def _resolve_robot_model(processor, motion: dict, robot_model_arg: str) -> str:
    normalized = _normalize_robot_model(robot_model_arg)
    if normalized != "auto":
        return normalized

    candidates = [
        motion.get("robot_model", None),
        motion.get("traj_robot_model", None),
        getattr(processor, "robot_model", None),
        getattr(processor, "traj_hand_name", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, np.ndarray) and candidate.shape == ():
            candidate = candidate.item()
        return _normalize_robot_model(str(candidate))

    return "xhand"


def _resolve_active_hands(processor, motion: dict) -> dict[str, bool]:
    active = {"left": True, "right": True}
    for side in ("left", "right"):
        key = f"is_active_{side}"
        if key in motion:
            value = motion[key]
            if isinstance(value, np.ndarray):
                value = value.reshape(-1)[0]
            active[side] = bool(value)

    hand_side_available = getattr(processor, "hand_side_available", None)
    if isinstance(hand_side_available, dict):
        for side in ("left", "right"):
            if side in hand_side_available:
                active[side] = bool(hand_side_available[side])
    return active


def _joint_dict_for_frame(
    joint_names: list[str] | np.ndarray,
    joint_vals_frame: np.ndarray | None,
) -> dict[str, float]:
    if joint_vals_frame is None:
        return {}
    names = [str(name) for name in list(joint_names)]
    vals = np.asarray(joint_vals_frame, dtype=np.float32).reshape(-1)
    if len(names) != vals.shape[0]:
        raise ValueError(
            "Joint names / values mismatch: "
            f"{len(names)} names vs {vals.shape[0]} values."
        )
    return {name: float(val) for name, val in zip(names, vals.tolist())}


def _resolve_robot_base_pose(processor, motion: dict) -> np.ndarray:
    candidate = motion.get("robot_base_pose", None)
    if candidate is None:
        candidate = getattr(processor, "robot_base_pose", None)
    if candidate is None:
        return np.eye(4, dtype=np.float32)

    base_pose = np.asarray(candidate, dtype=np.float32)
    if base_pose.shape != (4, 4):
        raise ValueError(
            f"Expected robot_base_pose to have shape (4, 4), got {base_pose.shape}."
        )
    return base_pose


def _pose_from_pos_rot(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.asarray(rot, dtype=np.float32)
    pose[:3, 3] = np.asarray(pos, dtype=np.float32)
    return pose


def _object_pose_dict_for_frame(
    motion: dict, frame_idx: int
) -> tuple[np.ndarray, dict[str, float]]:
    pos = np.asarray(motion["obj_pos"][frame_idx], dtype=np.float32)
    if "obj_quat" in motion and motion["obj_quat"] is not None:
        quat_wxyz = np.asarray(motion["obj_quat"][frame_idx], dtype=np.float32)
        rot = R.from_quat(np.roll(quat_wxyz, -1)).as_matrix().astype(np.float32)
    elif "obj_rot" in motion and motion["obj_rot"] is not None:
        rot = (
            R.from_rotvec(np.asarray(motion["obj_rot"][frame_idx], dtype=np.float32))
            .as_matrix()
            .astype(np.float32)
        )
    else:
        raise ValueError("Motion is missing both obj_quat and obj_rot.")

    obj_arti = motion.get("obj_arti", None)
    joint_vals = None if obj_arti is None else np.asarray(obj_arti[frame_idx])
    joint_dict = _joint_dict_for_frame(motion.get("joint_names", []), joint_vals)
    return _pose_from_pos_rot(pos, rot), joint_dict


def _copy_with_color(mesh, color_rgba: np.ndarray):
    colored = mesh.copy()
    color = np.asarray(color_rgba, dtype=np.uint8).reshape(4)
    colored.visual.face_colors = np.tile(color[None, :], (len(colored.faces), 1))
    return colored


def _fk_mesh_instances(
    *,
    urdf_inst: URDF,
    joint_dict: dict[str, float],
    base_pose: np.ndarray,
    mesh_type: str,
    color_rgba: np.ndarray,
) -> list[TrimeshInstance]:
    if mesh_type == "visual":
        fk = (
            urdf_inst.visual_trimesh_fk()
            if len(joint_dict) == 0
            else urdf_inst.visual_trimesh_fk(joint_dict)
        )
    elif mesh_type == "collision":
        fk = (
            urdf_inst.collision_trimesh_fk()
            if len(joint_dict) == 0
            else urdf_inst.collision_trimesh_fk(cfg=joint_dict)
        )
    else:
        raise ValueError(f"Unsupported mesh_type {mesh_type!r}.")

    mesh_instances: list[TrimeshInstance] = []
    for mesh, local_pose in fk.items():
        mesh_instances.append(
            TrimeshInstance(
                mesh=_copy_with_color(mesh, color_rgba),
                pose=base_pose @ np.asarray(local_pose, dtype=np.float32),
            )
        )
    return mesh_instances


def _resolve_robot_urdf_path(robot_model: str, side: str) -> Path:
    from dexlink.src.retargeting.data.hand_cfgs.franka_xhand import XHAND_CFGs as FRANKA_XHAND_CFGs

    if robot_model == "franka_xhand":
        return Path(process_urdf(FRANKA_XHAND_CFGs[side]["urdf_path"]))
    rel_path = ROBOT_URDFS[robot_model][side]
    if rel_path is None:
        raise ValueError(f"Missing URDF entry for {robot_model=} {side=}.")
    urdf_path = Path(rel_path)
    if not urdf_path.is_absolute():
        urdf_path = get_assets_dir().parent / urdf_path
    return Path(process_urdf(urdf_path))


def _default_output_path(exported_path: Path) -> Path:
    return exported_path.with_name(f"{exported_path.stem}_last_frame_four_views.png")


def _build_scene_for_export(
    *,
    exported_path: Path,
    mesh_type: str,
    robot_model_arg: str,
) -> tuple[list[TrimeshInstance], str]:
    processor = ExportedOakInkProcessor(exported_npy_path=exported_path)
    motion = processor.retargeted_motion
    if motion is None:
        raise ValueError(f"No retargeted motion found in {exported_path}.")
    if "obj_pos" not in motion:
        raise ValueError(f"Expected obj_pos in retargeted motion from {exported_path}.")

    n_frames = int(np.asarray(motion["obj_pos"]).shape[0])
    if n_frames <= 0:
        raise ValueError(f"Expected at least one frame in {exported_path}.")
    frame_idx = n_frames - 1

    active_hands = _resolve_active_hands(processor, motion)
    robot_model = _resolve_robot_model(processor, motion, robot_model_arg)
    robot_base_pose = _resolve_robot_base_pose(processor, motion)
    object_pose, object_joint_dict = _object_pose_dict_for_frame(motion, frame_idx)

    mesh_instances: list[TrimeshInstance] = []
    hand_colors = {
        "left": np.array([60, 120, 220, 255], dtype=np.uint8),
        "right": np.array([30, 160, 200, 255], dtype=np.uint8),
    }
    for side in ("left", "right"):
        if not active_hands[side]:
            continue
        joint_key = f"{side}_jvals"
        joint_names_key = f"{side}_jnames"
        if joint_key not in motion or joint_names_key not in motion:
            raise ValueError(
                f"Expected both {joint_names_key!r} and {joint_key!r} in motion."
            )
        joint_dict = _joint_dict_for_frame(
            motion[joint_names_key],
            np.asarray(motion[joint_key])[frame_idx],
        )
        robot_urdf = URDF.load(str(_resolve_robot_urdf_path(robot_model, side)))
        mesh_instances.extend(
            _fk_mesh_instances(
                urdf_inst=robot_urdf,
                joint_dict=joint_dict,
                base_pose=robot_base_pose,
                mesh_type=mesh_type,
                color_rgba=hand_colors[side],
            )
        )

    object_urdf = URDF.load(str(process_urdf(processor.target_obj_urdf_path)))
    mesh_instances.extend(
        _fk_mesh_instances(
            urdf_inst=object_urdf,
            joint_dict=object_joint_dict,
            base_pose=object_pose,
            mesh_type=mesh_type,
            color_rgba=np.array([235, 140, 40, 255], dtype=np.uint8),
        )
    )
    if len(mesh_instances) == 0:
        raise ValueError(f"No meshes were generated for {exported_path}.")

    label = (
        f"{exported_path.name}: frame={frame_idx} "
        f"robot_model={robot_model} mesh_type={mesh_type}"
    )
    return mesh_instances, label


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Render the last timestep from one or more exported OakInk payloads into "
            "four-view pyrender images."
        )
    )
    parser.add_argument(
        "--oakink-exported-path",
        action="append",
        required=True,
        help="Path to an exported OakInk `.npy` payload. Repeat for multiple renders.",
    )
    parser.add_argument(
        "--out",
        action="append",
        default=None,
        help=(
            "Optional output path for each render. Repeat once per "
            "`--oakink-exported-path` in matching order."
        ),
    )
    parser.add_argument(
        "--mesh-type",
        type=str,
        default="visual",
        choices=["visual", "collision"],
        help="URDF mesh type used for rendering.",
    )
    parser.add_argument(
        "--robot-model",
        type=str,
        default="auto",
        choices=["auto", "inspire", "xhand", "franka_xhand"],
        help=(
            "Robot URDF family to render. 'auto' uses exported motion metadata when "
            "available."
        ),
    )
    parser.add_argument("--width", type=int, default=640, help="View width in pixels.")
    parser.add_argument(
        "--height", type=int, default=480, help="View height in pixels."
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=0.5,
        help="Camera distance used for each rendered view in meters.",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=0.25,
        help="Camera elevation used for each rendered view in meters.",
    )
    args = parser.parse_args()

    exported_paths = [Path(path).resolve() for path in args.oakink_exported_path]
    outputs = [] if args.out is None else [Path(path).resolve() for path in args.out]
    if len(outputs) not in {0, len(exported_paths)}:
        raise ValueError(
            "Expected either zero `--out` values or exactly one per exported path, "
            f"got {len(outputs)} outputs for {len(exported_paths)} payloads."
        )

    for idx, exported_path in enumerate(exported_paths):
        out_path = (
            _default_output_path(exported_path) if len(outputs) == 0 else outputs[idx]
        )
        mesh_instances, label = _build_scene_for_export(
            exported_path=exported_path,
            mesh_type=args.mesh_type,
            robot_model_arg=args.robot_model,
        )
        image = render_centered_four_views(
            mesh_instances=mesh_instances,
            image_width=args.width,
            image_height=args.height,
            camera_distance_m=args.camera_distance,
            camera_elevation_m=args.camera_elevation,
            grid_columns=4,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        print(f"Saved {label} to {out_path}")


if __name__ == "__main__":
    main()
