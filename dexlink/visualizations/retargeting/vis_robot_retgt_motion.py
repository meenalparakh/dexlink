import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from urdfpy import URDF

from dexlink.src.retargeting.data.exported_oakink_processor import ExportedOakInkProcessor
from dexlink.src.retargeting.data.process_arctic import RandomTrajProcessor, get_processor_cfg
from dexlink.visualizations.retargeting.render_oakink_last_frame_views import (
    _resolve_robot_urdf_path,
)
from dexlink.utils.retargeting.path_utils import get_logs_dir, process_urdf
from dexlink.utils.retargeting.urdf_utils import get_non_colliding_jpos_near
from dexlink.utils.retargeting.vis_server import VisAnimator

def _normalize_robot_model(robot_model: str | None) -> str:
    if robot_model is None:
        return "xhand"
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


def _resolve_robot_model(args_robot_model, processor, motion) -> str:
    # return "franka_xhand"
    # return args_robot_model
    robot_model = _normalize_robot_model(args_robot_model)
    if robot_model != "auto":
        return robot_model

    candidates = [
        motion.get("robot_model", None) if isinstance(motion, dict) else None,
        motion.get("traj_robot_model", None) if isinstance(motion, dict) else None,
        getattr(processor, "robot_model", None),
        getattr(processor, "traj_hand_name", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, np.ndarray) and candidate.shape == ():
            candidate = candidate.item()
        candidate_str = str(candidate)
        if candidate_str:
            return _normalize_robot_model(candidate_str)

    return "xhand"


def _joint_dict_for_frame(joint_names, joint_vals_frame):
    if joint_vals_frame is None or len(joint_names) == 0:
        return {}
    vals = np.asarray(joint_vals_frame).reshape(-1)
    return {name: float(val) for name, val in zip(joint_names, vals.tolist())}


def _motion_scalar(value, field_name: str):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
        raise ValueError(f"Expected scalar {field_name!r}, got shape {value.shape}.")
    return value


def _hand_config_source_by_side(motion: dict) -> dict[str, str]:
    raw = motion.get("hand_config_source_by_side", {})
    if isinstance(raw, np.ndarray):
        if raw.shape != ():
            raise ValueError(
                "Expected hand_config_source_by_side to be a scalar object array, "
                f"got shape {raw.shape}."
            )
        raw = raw.item()
    if raw in (None, "", "None"):
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            "Expected hand_config_source_by_side to be a dict, "
            f"got {type(raw).__name__}."
        )
    source_by_side: dict[str, str] = {}
    for role_side, source_side in raw.items():
        role_side = str(role_side)
        source_side = str(source_side)
        if role_side not in ("left", "right") or source_side not in ("left", "right"):
            raise ValueError(
                "Expected hand_config_source_by_side to map left/right role sides "
                f"to left/right config sides, got {raw!r}."
            )
        source_by_side[role_side] = source_side
    return source_by_side


def _motion_joint_names(value, field_name: str) -> list[str]:
    arr = np.asarray(value, dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"Expected {field_name!r} shape (J,), got {arr.shape}.")
    return [str(name) for name in arr.tolist()]


def _resolve_visualized_joint_motion(
    motion: dict,
    *,
    side: str,
    use_refined: bool,
) -> tuple[list[str], np.ndarray | None, str]:
    if side not in ("left", "right"):
        raise ValueError(f"Expected side 'left' or 'right', got {side!r}.")

    base_jnames = _motion_joint_names(
        motion.get(f"{side}_jnames", []), f"{side}_jnames"
    )
    base_jvals = motion.get(f"{side}_jvals", None)
    if not use_refined or "drake_collision_refinement_jvals" not in motion:
        return base_jnames, base_jvals, "original"

    required_keys = (
        "drake_collision_refinement_jvals",
        "drake_collision_refinement_jnames",
        "drake_collision_refinement_hand",
    )
    missing = [key for key in required_keys if key not in motion]
    if missing:
        raise ValueError(
            "Refined visualization requested, but Drake refinement fields are "
            f"incomplete; missing {missing}."
        )

    refined_hand = str(
        _motion_scalar(
            motion["drake_collision_refinement_hand"],
            "drake_collision_refinement_hand",
        )
    )
    if refined_hand != side:
        return base_jnames, base_jvals, "original"

    refined_jnames = _motion_joint_names(
        motion["drake_collision_refinement_jnames"],
        "drake_collision_refinement_jnames",
    )
    refined_key = f"drake_collision_refinement_{side}_jvals"
    if refined_key not in motion:
        refined_key = "drake_collision_refinement_jvals"
    refined_jvals = np.asarray(motion[refined_key], dtype=np.float32)
    if refined_jvals.ndim != 2:
        raise ValueError(
            f"Expected {refined_key!r} shape (T, J), got {refined_jvals.shape}."
        )
    if refined_jvals.shape[1] != len(refined_jnames):
        raise ValueError(
            f"Expected {refined_key!r} joint dim to match "
            f"drake_collision_refinement_jnames length, got "
            f"{refined_jvals.shape[1]} and {len(refined_jnames)}."
        )
    return refined_jnames, refined_jvals, "drake_refined"


def _pose_dict_for_object_motion_frame(motion, frame_idx):
    pos = np.asarray(motion["obj_pos"][frame_idx], dtype=float)

    if "obj_quat" in motion and motion["obj_quat"] is not None:
        quat_wxyz = np.asarray(motion["obj_quat"][frame_idx], dtype=float)
        quat_xyzw = np.roll(quat_wxyz, -1)
        rot = R.from_quat(quat_xyzw).as_matrix()
    else:
        rot = R.from_rotvec(
            np.asarray(motion["obj_rot"][frame_idx], dtype=float)
        ).as_matrix()

    joints = _joint_dict_for_frame(
        motion.get("joint_names", []),
        None if motion.get("obj_arti", None) is None else motion["obj_arti"][frame_idx],
    )
    return {"pos": pos, "rot": rot, "joints": joints}


def _has_scene_object_motion(motion: dict) -> bool:
    return "scene_object_pos" in motion


def _load_npz_hand_config_source_by_side(path: str | Path) -> dict[str, np.ndarray]:
    """Load the role-to-config hand mapping an exported payload records.

    A task can drive a role side with the other side's retargeting and robot
    config: the lid and throw-catch tasks label one hand left for the RL
    interface but pose it with the right hand's URDF and right-hand joint names.
    The processor's motion does not carry that mapping, so without reading it
    here the visualizer falls back to left->left, loads the left URDF and is then
    handed `R_`-prefixed joint names, which fails as a missing joint.
    """

    with np.load(path, allow_pickle=True) as payload:
        if "hand_config_source_by_side" not in payload.files:
            return {}
        return {"hand_config_source_by_side": payload["hand_config_source_by_side"]}


def _load_npz_scene_object_motion(
    path: str | Path,
    *,
    start_timestep: int,
    end_timestep: int,
    skip: int,
) -> dict[str, np.ndarray]:
    """Load multi-object fields discarded by the legacy random trajectory loader."""
    required_keys = {
        "scene_object_ids",
        "scene_object_urdf_paths",
        "scene_object_pos",
        "scene_object_quat",
        "scene_object_arti_vals",
        "scene_object_arti_valid_mask",
        "scene_object_arti_names",
    }
    with np.load(path, allow_pickle=True) as payload:
        if "scene_object_pos" not in payload:
            return {}
        missing_keys = sorted(required_keys - set(payload.files))
        if missing_keys:
            raise ValueError(
                f"Multi-object trajectory {path} is missing fields: {missing_keys}."
            )
        if skip <= 0:
            raise ValueError(f"Expected positive trajectory skip, got {skip}.")
        scene_pos = np.asarray(payload["scene_object_pos"])
        if scene_pos.ndim != 3 or scene_pos.shape[-1] != 3:
            raise ValueError(
                "Expected scene_object_pos shape (frames, objects, 3), got "
                f"{scene_pos.shape} in {path}."
            )
        stop = scene_pos.shape[0] if end_timestep == -1 else end_timestep
        time_slice = slice(start_timestep, stop, skip)
        scene_motion = {
            "scene_object_ids": np.asarray(payload["scene_object_ids"]).copy(),
            "scene_object_urdf_paths": np.asarray(
                payload["scene_object_urdf_paths"]
            ).copy(),
            "scene_object_pos": scene_pos[time_slice].copy(),
            "scene_object_quat": np.asarray(payload["scene_object_quat"])[
                time_slice
            ].copy(),
            "scene_object_arti_vals": np.asarray(payload["scene_object_arti_vals"])[
                time_slice
            ].copy(),
            "scene_object_arti_valid_mask": np.asarray(
                payload["scene_object_arti_valid_mask"]
            ).copy(),
            "scene_object_arti_names": np.asarray(
                payload["scene_object_arti_names"], dtype=object
            ).copy(),
        }
    object_count = len(scene_motion["scene_object_ids"])
    for key in ("scene_object_urdf_paths", "scene_object_arti_valid_mask"):
        if scene_motion[key].shape[0] != object_count:
            raise ValueError(
                f"Expected {key} first dimension {object_count}, got "
                f"{scene_motion[key].shape}."
            )
    for key in ("scene_object_pos", "scene_object_quat", "scene_object_arti_vals"):
        if scene_motion[key].shape[1] != object_count:
            raise ValueError(
                f"Expected {key} object dimension {object_count}, got "
                f"{scene_motion[key].shape}."
            )
    return scene_motion


def _scene_object_ids(motion: dict) -> list[str]:
    if "scene_object_ids" not in motion:
        raise ValueError("Multi-object motion is missing scene_object_ids.")
    return [str(object_id) for object_id in list(motion["scene_object_ids"])]


def _scene_object_joint_dict_for_frame(
    motion: dict, frame_idx: int, object_idx: int
) -> dict[str, float]:
    if "scene_object_arti_names" not in motion:
        raise ValueError("Multi-object motion is missing scene_object_arti_names.")
    if "scene_object_arti_vals" not in motion:
        raise ValueError("Multi-object motion is missing scene_object_arti_vals.")
    if "scene_object_arti_valid_mask" not in motion:
        raise ValueError("Multi-object motion is missing scene_object_arti_valid_mask.")

    raw_names = motion["scene_object_arti_names"][object_idx]
    names = [str(name) for name in list(raw_names)]
    valid_mask = np.asarray(
        motion["scene_object_arti_valid_mask"][object_idx], dtype=bool
    )
    vals = np.asarray(
        motion["scene_object_arti_vals"][frame_idx, object_idx], dtype=float
    )
    if int(valid_mask.sum()) != len(names):
        raise ValueError(
            "scene_object_arti_valid_mask true count must match "
            f"scene_object_arti_names length for object index {object_idx}; "
            f"got {int(valid_mask.sum())} and {len(names)}."
        )
    return {
        name: float(value)
        for name, value in zip(names, vals[valid_mask].reshape(-1).tolist())
    }


def _pose_dict_for_scene_object_motion_frame(
    motion: dict, frame_idx: int, object_idx: int
) -> dict:
    if "scene_object_pos" not in motion:
        raise ValueError("Multi-object motion is missing scene_object_pos.")
    if "scene_object_quat" not in motion:
        raise ValueError("Multi-object motion is missing scene_object_quat.")

    pos = np.asarray(motion["scene_object_pos"][frame_idx, object_idx], dtype=float)
    quat_wxyz = np.asarray(
        motion["scene_object_quat"][frame_idx, object_idx], dtype=float
    )
    rot = R.from_quat(np.roll(quat_wxyz, -1)).as_matrix()
    return {
        "pos": pos,
        "rot": rot,
        "joints": _scene_object_joint_dict_for_frame(motion, frame_idx, object_idx),
    }


def _resolve_robot_base_pose(processor, motion: dict) -> np.ndarray:
    candidate = None
    if isinstance(motion, dict):
        candidate = motion.get("robot_base_pose", None)
    if candidate is None:
        candidate = getattr(processor, "robot_base_pose", None)

    if candidate is not None:
        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape == (4, 4):
            return candidate
        raise ValueError(
            f"Expected robot_base_pose to have shape (4, 4), got {candidate.shape}."
        )
    return np.eye(4, dtype=float)


def _pose_dict_for_robot_motion_frame(joint_names, joint_vals_frame, base_pose=None):
    if base_pose is None:
        pos = np.zeros(3, dtype=float)
        rot = np.eye(3, dtype=float)
    else:
        base_pose = np.asarray(base_pose, dtype=float)
        if base_pose.shape != (4, 4):
            raise ValueError(
                f"Expected robot base pose to have shape (4, 4), got {base_pose.shape}."
            )
        pos = base_pose[:3, 3]
        rot = base_pose[:3, :3]
    return {
        "pos": pos,
        "rot": rot,
        "joints": _joint_dict_for_frame(joint_names, joint_vals_frame),
    }


def _pose_dict_to_matrix(pose_dict: dict) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(pose_dict["rot"], dtype=float)
    transform[:3, 3] = np.asarray(pose_dict["pos"], dtype=float)
    return transform


def _collision_mesh_obstacles_for_pose(
    urdf_inst: URDF, pose_dict: dict, prefix: str
) -> list[tuple[str, trimesh.Trimesh, np.ndarray]]:
    joint_dict = dict(pose_dict["joints"])
    base_pose = _pose_dict_to_matrix(pose_dict)
    if len(joint_dict) == 0:
        fk = urdf_inst.collision_trimesh_fk()
    else:
        fk = urdf_inst.collision_trimesh_fk(cfg=joint_dict)

    obstacle_entries = []
    for mesh_idx, (mesh, local_pose) in enumerate(fk.items()):
        obstacle_entries.append(
            (
                f"{prefix}_{mesh_idx}",
                mesh,
                base_pose @ np.asarray(local_pose, dtype=float),
            )
        )
    return obstacle_entries


def _root_fixed_joint_names(joint_names) -> list[str]:
    return [name for name in joint_names if "forearm" in str(name)]


def _ordered_reference_points(processor, side: str) -> np.ndarray | None:
    ref_pos_dict = getattr(processor, f"relevant_{side}_joint_pos", None)
    if not ref_pos_dict:
        return None
    if not hasattr(processor, "hand_joint_map"):
        raise ValueError(
            "Processor is missing hand_joint_map required for reference visualization."
        )
    ordered_names = list(processor.hand_joint_map.keys())
    missing_names = [name for name in ordered_names if name not in ref_pos_dict]
    if missing_names:
        raise ValueError(
            f"Reference {side} joint positions are missing required keys: {missing_names}"
        )
    return np.stack(
        [np.asarray(ref_pos_dict[name], dtype=np.float32) for name in ordered_names],
        axis=1,
    )


def _describe_reference_points(points: np.ndarray | None, side: str) -> str:
    if points is None:
        return f"{side}: missing"
    mins = np.min(points.reshape(-1, 3), axis=0)
    maxs = np.max(points.reshape(-1, 3), axis=0)
    return (
        f"{side}: shape={points.shape}, "
        f"min={np.array2string(mins, precision=3)}, "
        f"max={np.array2string(maxs, precision=3)}"
    )


def _frame_indices_for_visualization(
    motion: dict,
    start_timestep: int,
    end_timestep: int,
) -> np.ndarray:
    exact_timesteps = motion.get("exact_timesteps", None)
    if exact_timesteps is not None:
        exact_timesteps = np.asarray(exact_timesteps, dtype=np.int64).reshape(-1)
        if end_timestep == -1:
            mask = exact_timesteps >= start_timestep
        else:
            mask = (exact_timesteps >= start_timestep) & (
                exact_timesteps < end_timestep
            )
        return np.flatnonzero(mask)

    n_frames = None
    for value in motion.values():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            n_frames = value.shape[0]
            break
    if n_frames is None:
        raise ValueError("Could not infer the number of frames from motion.")

    end_idx = n_frames if end_timestep == -1 else min(int(end_timestep), n_frames)
    start_idx = min(max(0, int(start_timestep)), end_idx)
    return np.arange(start_idx, end_idx, dtype=np.int64)


def _resolve_active_hands(processor, motion: dict) -> dict[str, bool]:
    active = {"left": True, "right": True}
    if isinstance(motion, dict):
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


def _safe_load_motion_processor(
    demo_path: str | None,
    target_object: str | None,
    retargeted_results: str | None,
    use_oakink_demo: bool,
    oakink_exported_path: str | None,
    start_timestep: int,
    end_timestep: int,
    skip: int,
):
    if use_oakink_demo:
        if oakink_exported_path is None:
            raise ValueError(
                "--oakink_exported_path is required when --use_oakink_demo is set."
            )
        return ExportedOakInkProcessor(
            exported_npy_path=oakink_exported_path,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            skip=skip,
        )

    if demo_path is not None and target_object is not None:
        processor_cfg = get_processor_cfg(target_object)
        processor_cls = processor_cfg["processor_cls"]
        return processor_cls(
            original_demo_path=demo_path,
            target_object=target_object,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            skip=skip,
            reference_urdf=processor_cfg["reference_urdf"],
            target_urdf=processor_cfg["target_urdf_path"],
            retargeted_results=retargeted_results,
            retargeted_results_upsample=True,
        )

    if retargeted_results is None:
        raise ValueError(
            "retargeted results required when not using --use_oakink_demo. "
            "Pass --retargeted_results, or both --demo_path and --target_object."
        )

    return RandomTrajProcessor(
        original_demo_path=None,
        target_object=target_object,
        start_timestep=start_timestep,
        end_timestep=end_timestep,
        skip=skip,
        reference_urdf=None,
        target_urdf=None,
        retargeted_results=retargeted_results,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize robot retargeted motion (left/right robot URDF + target object) "
            "and compute per-frame penetration/intersection metrics."
        )
    )
    # parser.add_argument(
    parser.add_argument(
        "--retargeted_results",
        type=str,
        default=None,
        help=(
            "Path to retargeted trajopt .npz results. Optional when --demo_path and "
            "--target_object are provided (path auto-derived), and unused in OakInk mode."
        ),
    )
    parser.add_argument(
        "--demo_path",
        type=str,
        default=None,
        help="Optional ARCTIC demo path. If set with --target_object, uses ArcticProcessor.",
    )
    parser.add_argument(
        "--target_object",
        type=str,
        default=None,
        help="Optional target object key for get_processor_cfg (e.g. arctic_laptop).",
    )
    parser.add_argument(
        "--use_oakink_demo",
        action="store_true",
        help="Use ExportedOakInkProcessor instead of Arctic/random processors.",
    )
    parser.add_argument(
        "--oakink_exported_path",
        type=str,
        default=None,
        help="Path to exported OakInk .npy payload consumed by ExportedOakInkProcessor.",
    )
    parser.add_argument("--start_timestep", type=int, default=0)
    parser.add_argument("--end_timestep", type=int, default=-1)
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--port", type=int, default=6000, help="MeshCat ZMQ port.")
    parser.add_argument(
        "--animation-repetitions",
        type=int,
        default=1,
        help="Number of times MeshCat repeats the loaded animation.",
    )
    parser.add_argument(
        "--mesh_type",
        type=str,
        default="collision",
        choices=["visual", "collision"],
        help="URDF mesh type used for MeshCat rendering.",
    )
    parser.add_argument(
        "--robot-model",
        type=str,
        default="auto",
        choices=["auto", "inspire", "xhand", "franka_xhand"],
        help=(
            "Robot URDF family to render. 'auto' uses motion metadata when available, "
            "else defaults to xhand."
        ),
    )
    parser.add_argument(
        "--show-nearby-non-colliding",
        action="store_true",
        help="Replace each frame's robot joint values with a random nearby non-colliding sample.",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help=(
            "Disable Drake-refined joint fields in exported .npy payloads and "
            "visualize the original retargeted joint values instead."
        ),
    )
    parser.add_argument(
        "--nearby-std",
        type=float,
        default=0.1,
        help="Gaussian std used by nearby non-colliding sampling.",
    )
    parser.add_argument(
        "--nearby-max-tries",
        type=int,
        default=200,
        help="Maximum sampling attempts per frame for nearby non-colliding visualization.",
    )
    args = parser.parse_args()
    if args.animation_repetitions <= 0:
        raise ValueError(
            "Expected --animation-repetitions to be positive, got "
            f"{args.animation_repetitions}."
        )

    def get_retgt_results_path(demo_name, target_asset):
        return str(
            get_logs_dir()
            / f"trajopt_results_good/single_step_chunks_v1__{target_asset}__s01__{demo_name}__stacked/single_step_trajectory_processed_periodic_aligned.npz"
        )

    retargeted_results = args.retargeted_results
    if (
        not args.use_oakink_demo
        and retargeted_results is None
        and args.demo_path is not None
        and args.target_object is not None
    ):
        retargeted_results = get_retgt_results_path(
            Path(args.demo_path).stem, args.target_object
        )

    processor = _safe_load_motion_processor(
        demo_path=args.demo_path,
        target_object=args.target_object,
        retargeted_results=retargeted_results,
        use_oakink_demo=args.use_oakink_demo,
        oakink_exported_path=args.oakink_exported_path,
        start_timestep=args.start_timestep,
        end_timestep=args.end_timestep,
        skip=args.skip,
    )

    motion = processor.retargeted_motion
    if motion is None:
        raise ValueError("No retargeted motion found in processor.")
    if not args.use_oakink_demo and retargeted_results is not None:
        motion.update(
            _load_npz_scene_object_motion(
                retargeted_results,
                start_timestep=args.start_timestep,
                end_timestep=args.end_timestep,
                skip=args.skip,
            )
        )
        # Separate from the scene-object fields above, which a payload without
        # scene objects skips entirely; a two-hand task still needs its mapping.
        motion.update(_load_npz_hand_config_source_by_side(retargeted_results))

    use_refined = not bool(args.no_refine)
    left_jnames, left_jvals, left_motion_source = _resolve_visualized_joint_motion(
        motion,
        side="left",
        use_refined=use_refined,
    )
    right_jnames, right_jvals, right_motion_source = _resolve_visualized_joint_motion(
        motion,
        side="right",
        use_refined=use_refined,
    )
    robot_base_pose = _resolve_robot_base_pose(processor, motion)
    active_hands = _resolve_active_hands(processor, motion)
    left_ref_points = _ordered_reference_points(processor, "left")
    right_ref_points = _ordered_reference_points(processor, "right")
    left_fixed_joints = _root_fixed_joint_names(left_jnames)
    right_fixed_joints = _root_fixed_joint_names(right_jnames)
    print("Reference points summary:")
    print(_describe_reference_points(left_ref_points, "left"))
    print(_describe_reference_points(right_ref_points, "right"))
    print(
        "Robot base pose for visualization:\n"
        f"{np.array2string(robot_base_pose, precision=3)}"
    )
    print(f"Active hands for visualization: {active_hands}")
    print(
        "Robot joint source for visualization: "
        f"left={left_motion_source}, right={right_motion_source}"
    )
    if args.show_nearby_non_colliding:
        print(f"Left fixed joints for nearby sampling: {left_fixed_joints}")
        print(f"Right fixed joints for nearby sampling: {right_fixed_joints}")

    target_obj_motion = (
        motion
        if motion.get("obj_pos", None) is not None
        else processor.object_reference_motion
    )
    frame_indices = _frame_indices_for_visualization(
        motion=target_obj_motion,
        start_timestep=args.start_timestep,
        end_timestep=args.end_timestep,
    )
    if frame_indices.size == 0:
        raise ValueError(
            "No frames selected for visualization with "
            f"start_timestep={args.start_timestep} and end_timestep={args.end_timestep}."
        )
    print(
        "Visualizing frames: "
        f"count={frame_indices.size}, first_index={int(frame_indices[0])}, "
        f"last_index={int(frame_indices[-1])}"
    )
    robot_model = _resolve_robot_model(args.robot_model, processor, motion)
    print(f"Using robot model for visualization: {robot_model}")
    hand_config_source_by_side = _hand_config_source_by_side(motion)
    left_config_side = hand_config_source_by_side.get("left", "left")
    right_config_side = hand_config_source_by_side.get("right", "right")
    print(
        "Robot config source for visualization: "
        f"left={left_config_side}, right={right_config_side}"
    )
    if args.show_nearby_non_colliding and args.mesh_type != "collision":
        print(
            "Warning: nearby non-colliding sampling uses collision meshes, but the "
            f"visualizer is rendering {args.mesh_type!r} meshes. Use --mesh_type collision "
            "for a faithful collision-check view."
        )

    left_urdf = (
        URDF.load(str(_resolve_robot_urdf_path(robot_model, left_config_side)))
        if active_hands["left"]
        else None
    )
    right_urdf = (
        URDF.load(str(_resolve_robot_urdf_path(robot_model, right_config_side)))
        if active_hands["right"]
        else None
    )
    scene_object_urdfs = {}
    scene_object_ids = []
    if _has_scene_object_motion(motion):
        if "scene_object_urdf_paths" not in motion:
            raise ValueError("Multi-object motion is missing scene_object_urdf_paths.")
        scene_object_ids = _scene_object_ids(motion)
        if len(set(scene_object_ids)) != len(scene_object_ids):
            raise ValueError(
                f"Expected unique scene_object_ids, got {scene_object_ids}."
            )
        scene_object_urdf_paths = list(motion["scene_object_urdf_paths"])
        if len(scene_object_urdf_paths) != len(scene_object_ids):
            raise ValueError(
                "scene_object_urdf_paths length must match scene_object_ids length; "
                f"got {len(scene_object_urdf_paths)} and {len(scene_object_ids)}."
            )
        for object_id, object_urdf_path in zip(
            scene_object_ids, scene_object_urdf_paths
        ):
            scene_object_urdfs[object_id] = URDF.load(
                str(process_urdf(Path(str(object_urdf_path))))
            )
        print("Loaded scene objects for visualization: " + ", ".join(scene_object_ids))
    else:
        object_urdf = URDF.load(str(process_urdf(processor.target_obj_urdf_path)))

    vis = VisAnimator(port_vis=args.port)
    if left_urdf is not None:
        vis.initialize_urdf(
            urdf_inst=left_urdf,
            name="robot/left_hand",
            mesh_type=args.mesh_type,
            color="cool",
            alpha=0.8,
        )
    if right_urdf is not None:
        vis.initialize_urdf(
            urdf_inst=right_urdf,
            name="robot/right_hand",
            mesh_type=args.mesh_type,
            color="cool",
            alpha=0.8,
        )
    if scene_object_urdfs:
        for object_id, object_urdf in scene_object_urdfs.items():
            vis.initialize_urdf(
                urdf_inst=object_urdf,
                name=f"target/{object_id}",
                mesh_type=args.mesh_type,
                color="orange",
                alpha=0.8,
            )
    else:
        vis.initialize_urdf(
            urdf_inst=object_urdf,
            name="target/object",
            mesh_type=args.mesh_type,
            color="orange",
            alpha=0.8,
        )
    if active_hands["left"] and left_ref_points is not None:
        vis.initialize_points(
            mesh_vertices=left_ref_points[0],
            mesh_colors=np.tile(
                np.array([[1.0, 0.35, 0.2]]), (left_ref_points.shape[1], 1)
            ),
            posed=False,
            num_points=left_ref_points.shape[1],
            name="reference/left_joint_points",
            radius=0.01,
            opacity=0.95,
        )
    else:
        print(
            "No left reference points found; skipping left joint point visualization."
        )
    if active_hands["right"] and right_ref_points is not None:
        vis.initialize_points(
            mesh_vertices=right_ref_points[0],
            mesh_colors=np.tile(
                np.array([[0.2, 0.8, 0.35]]), (right_ref_points.shape[1], 1)
            ),
            posed=False,
            num_points=right_ref_points.shape[1],
            name="reference/right_joint_points",
            radius=0.01,
            opacity=0.95,
        )
    else:
        print(
            "No right reference points found; skipping right joint point visualization."
        )
    vis.initialize()

    n_left_perturbed = 0
    n_right_perturbed = 0
    n_left_failed = 0
    n_right_failed = 0
    initial_urdf_dict = None

    for frame_idx in tqdm(
        frame_indices.tolist(), desc="Preparing visualization frames"
    ):
        object_obstacles = []
        if scene_object_urdfs:
            urdf_dict = {}
            for object_idx, object_id in enumerate(scene_object_ids):
                object_pose_dict = _pose_dict_for_scene_object_motion_frame(
                    motion, frame_idx, object_idx
                )
                urdf_dict[f"target/{object_id}"] = object_pose_dict
                object_obstacles.extend(
                    _collision_mesh_obstacles_for_pose(
                        scene_object_urdfs[object_id],
                        object_pose_dict,
                        prefix=f"object_{object_id}_{frame_idx}",
                    )
                )
        else:
            object_pose_dict = _pose_dict_for_object_motion_frame(
                target_obj_motion, frame_idx
            )
            urdf_dict = {"target/object": object_pose_dict}
            object_obstacles = _collision_mesh_obstacles_for_pose(
                object_urdf,
                object_pose_dict,
                prefix=f"object_{frame_idx}",
            )

        left_joint_dict = {}
        if active_hands["left"] and left_jvals is not None and len(left_jnames) > 0:
            left_joint_dict = _joint_dict_for_frame(left_jnames, left_jvals[frame_idx])

        right_joint_dict = {}
        if active_hands["right"] and right_jvals is not None and len(right_jnames) > 0:
            right_joint_dict = _joint_dict_for_frame(
                right_jnames, right_jvals[frame_idx]
            )

        if args.show_nearby_non_colliding and left_joint_dict:
            left_obstacles = list(object_obstacles)
            if right_joint_dict:
                right_pose_dict = _pose_dict_for_robot_motion_frame(
                    right_jnames, right_jvals[frame_idx], base_pose=robot_base_pose
                )
                left_obstacles.extend(
                    _collision_mesh_obstacles_for_pose(
                        right_urdf,
                        right_pose_dict,
                        prefix=f"right_hand_{frame_idx}",
                    )
                )
            sampled_left = get_non_colliding_jpos_near(
                left_urdf,
                left_joint_dict,
                left_obstacles,
                fixed_joints=left_fixed_joints,
                std=args.nearby_std,
                max_tries=args.nearby_max_tries,
            )
            if sampled_left is None:
                n_left_failed += 1
            else:
                left_joint_dict = sampled_left
                n_left_perturbed += 1

        if left_joint_dict:
            urdf_dict["robot/left_hand"] = _pose_dict_for_robot_motion_frame(
                left_jnames,
                np.asarray(
                    [left_joint_dict[name] for name in left_jnames], dtype=float
                ),
                base_pose=robot_base_pose,
            )

        if args.show_nearby_non_colliding and right_joint_dict:
            right_obstacles = list(object_obstacles)
            if left_joint_dict:
                left_pose_dict = _pose_dict_for_robot_motion_frame(
                    left_jnames,
                    np.asarray(
                        [left_joint_dict[name] for name in left_jnames], dtype=float
                    ),
                    base_pose=robot_base_pose,
                )
                right_obstacles.extend(
                    _collision_mesh_obstacles_for_pose(
                        left_urdf,
                        left_pose_dict,
                        prefix=f"left_hand_{frame_idx}",
                    )
                )
            sampled_right = get_non_colliding_jpos_near(
                right_urdf,
                right_joint_dict,
                right_obstacles,
                fixed_joints=right_fixed_joints,
                std=args.nearby_std,
                max_tries=args.nearby_max_tries,
                visualize=True,
                collision_threshold=0.01,
            )
            if sampled_right is None:
                n_right_failed += 1
            else:
                right_joint_dict = sampled_right
                n_right_perturbed += 1

        if right_joint_dict:
            urdf_dict["robot/right_hand"] = _pose_dict_for_robot_motion_frame(
                right_jnames,
                np.asarray(
                    [right_joint_dict[name] for name in right_jnames], dtype=float
                ),
                base_pose=robot_base_pose,
            )

        pts_dict = {}
        if active_hands["left"] and left_ref_points is not None:
            pts_dict["reference/left_joint_points"] = left_ref_points[frame_idx]
        if active_hands["right"] and right_ref_points is not None:
            pts_dict["reference/right_joint_points"] = right_ref_points[frame_idx]

        if initial_urdf_dict is None:
            initial_urdf_dict = urdf_dict
        vis.add_frame(pts_dict=pts_dict, urdf_dict=urdf_dict)

    if initial_urdf_dict is None:
        raise RuntimeError("Expected at least one visualization frame.")
    # MeshCat URDF geometry is initialized at the URDF-local origin. Set the
    # first world-space frame explicitly so every scene object is visible even
    # before playback begins or after a browser reconnect.
    vis.set_urdf_poses(initial_urdf_dict)
    vis.play(repetitions=args.animation_repetitions)

    if args.show_nearby_non_colliding:
        print(
            "Nearby non-colliding sampling summary: "
            f"left_success={n_left_perturbed}, left_failed={n_left_failed}, "
            f"right_success={n_right_perturbed}, right_failed={n_right_failed}"
        )
    print("Animation loaded in MeshCat. Keep this process running while viewing.")


if __name__ == "__main__":
    main()
