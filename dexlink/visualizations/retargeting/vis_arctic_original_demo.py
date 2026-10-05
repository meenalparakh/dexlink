import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from dexlink.src.retargeting.data.process_arctic import get_processor_cfg
from dexlink.utils.retargeting.path_utils import process_urdf
from dexlink.utils.retargeting.vis_server import VisAnimator


def _joint_dict_for_frame(joint_names, joint_vals_frame):
    if joint_vals_frame is None or len(joint_names) == 0:
        return {}
    vals = np.asarray(joint_vals_frame).reshape(-1)
    return dict(zip(joint_names, vals.tolist()))


def _pose_dict_for_motion_frame(motion, frame_idx):
    pos = np.asarray(motion["obj_pos"][frame_idx], dtype=float)

    if "obj_quat" in motion and motion["obj_quat"] is not None:
        quat_wxyz = np.asarray(motion["obj_quat"][frame_idx], dtype=float)
        quat_xyzw = np.roll(quat_wxyz, -1)
        rot = R.from_quat(quat_xyzw).as_matrix()
    else:
        rot = R.from_rotvec(
            np.asarray(motion["obj_rot"][frame_idx], dtype=float)
        ).as_matrix()

    joint_names = motion.get("joint_names", [])
    joint_vals = motion.get("obj_arti", None)
    joints = _joint_dict_for_frame(
        joint_names, None if joint_vals is None else joint_vals[frame_idx]
    )
    return {"pos": pos, "rot": rot, "joints": joints}


def _infer_target_object_from_demo_path(demo_path: str) -> str:
    stem = Path(demo_path).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            "Could not infer object from demo filename. "
            "Expected a stem like '<object>_use_01'. Please pass --target-object."
        )
    return f"arctic_{'_'.join(parts[:-2])}"


def _save_object_trajectory_plots(motion: dict, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_pos = np.asarray(motion["obj_pos"], dtype=float)
    frames = np.arange(obj_pos.shape[0])

    fig, ax = plt.subplots(figsize=(10, 4))
    for dim_idx, axis_name in enumerate(["x", "y", "z"]):
        ax.plot(frames, obj_pos[:, dim_idx], label=axis_name)
    ax.set_title("Object Position Trajectory")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Position")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_object_position_trajectory.png", dpi=160)
    plt.close(fig)

    obj_arti = motion.get("obj_arti")
    if obj_arti is None:
        return

    obj_arti = np.asarray(obj_arti, dtype=float)
    if obj_arti.ndim == 1:
        obj_arti = obj_arti[:, None]
    if obj_arti.shape[1] == 0:
        return

    joint_names = motion.get("joint_names", [])
    fig, ax = plt.subplots(figsize=(10, 4))
    for joint_idx in range(obj_arti.shape[1]):
        label = (
            joint_names[joint_idx]
            if joint_idx < len(joint_names)
            else f"joint_{joint_idx}"
        )
        ax.plot(frames, obj_arti[:, joint_idx], label=label)
    ax.set_title("Object Articulation Trajectory")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Joint Value")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_object_articulation_trajectory.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the original ARCTIC demo (hands + original object motion) in MeshCat."
    )
    parser.add_argument(
        "--demo_path",
        type=str,
        required=True,
        help="Path to original ARCTIC demo .npy/.npz file.",
    )
    parser.add_argument(
        "--target-object",
        type=str,
        default=None,
        help="Target object key (default: inferred from demo filename, e.g. arctic_laptop).",
    )
    parser.add_argument(
        "--timesteps",
        type=str,
        default="0,-1",
        help="Start and end timestep as 'a,b' (end is exclusive, -1 means full length).",
    )
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--port", type=int, default=6000, help="MeshCat ZMQ port.")
    parser.add_argument(
        "--num_points",
        type=int,
        default=1200,
        help="Max points sampled per hand point cloud.",
    )
    parser.add_argument(
        "--mesh_type",
        type=str,
        default="visual",
        choices=["visual", "collision"],
        help="URDF mesh type for object rendering.",
    )
    args = parser.parse_args()
    try:
        start_str, end_str = [x.strip() for x in args.timesteps.split(",", maxsplit=1)]
        start_timestep = int(start_str)
        end_timestep = int(end_str)
    except Exception as exc:
        raise ValueError(
            f"Invalid --timesteps value '{args.timesteps}'. Expected format 'a,b'."
        ) from exc

    target_object = (
        args.target_object
        if args.target_object is not None
        else _infer_target_object_from_demo_path(args.demo_path)
    )

    processor_cfg = get_processor_cfg(target_object)
    processor_cls = processor_cfg["processor_cls"]
    reference_urdf = processor_cfg["reference_urdf"]
    target_urdf = processor_cfg["target_urdf_path"]

    processor = processor_cls(
        args.demo_path,
        target_object,
        start_timestep,
        end_timestep,
        skip=args.skip,
        reference_urdf=reference_urdf,
        target_urdf=target_urdf,
        retargeted_results=None,
    )

    left_hand = processor.left_hand_verts
    right_hand = processor.right_hand_verts
    n_frames = processor.n_frames
    original_motion = processor.original_object_motion()
    plot_dir = Path("tmp")
    demo_stem = Path(args.demo_path).stem
    _save_object_trajectory_plots(original_motion, plot_dir, demo_stem)
    print(f"Saved trajectory plots to: {plot_dir.resolve()}")

    obj_urdf = URDF.load(str(process_urdf(processor.obj_urdf_path)))
    vis = VisAnimator(port_vis=args.port)

    left_color = np.array([0.10, 0.45, 0.95], dtype=float)
    right_color = np.array([0.95, 0.35, 0.10], dtype=float)

    vis.initialize_points(
        mesh_vertices=left_hand[0],
        mesh_colors=np.tile(left_color[None, :], (left_hand.shape[1], 1)),
        posed=False,
        num_points=args.num_points,
        name="original/left_hand",
    )
    vis.initialize_points(
        mesh_vertices=right_hand[0],
        mesh_colors=np.tile(right_color[None, :], (right_hand.shape[1], 1)),
        posed=False,
        num_points=args.num_points,
        name="original/right_hand",
    )
    vis.initialize_urdf(
        urdf_inst=obj_urdf,
        name="original/object",
        mesh_type=args.mesh_type,
        color="cool",
    )
    vis.initialize()

    for frame_idx in range(n_frames):
        vis.add_frame(
            pts_dict={
                "original/left_hand": left_hand[frame_idx],
                "original/right_hand": right_hand[frame_idx],
            },
            urdf_dict={
                "original/object": _pose_dict_for_motion_frame(
                    original_motion, frame_idx
                )
            },
        )

    vis.play()
    print("Original demo loaded in MeshCat. Keep this process running while viewing.")


if __name__ == "__main__":
    main()
