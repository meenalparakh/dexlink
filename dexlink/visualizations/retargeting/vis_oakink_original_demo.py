import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dexlink.utils.retargeting.vis_server import VisAnimator


def _parse_timesteps(timesteps: str) -> tuple[int, int]:
    try:
        start_str, end_str = [x.strip() for x in timesteps.split(",", maxsplit=1)]
        return int(start_str), int(end_str)
    except Exception as exc:
        raise ValueError(
            f"Invalid --timesteps value '{timesteps}'. Expected format 'a,b'."
        ) from exc


def _slice_frames(arr: np.ndarray, start: int, end: int, skip: int) -> np.ndarray:
    end_idx = None if end == -1 else end
    return arr[start:end_idx:skip]


def _save_joint_trajectory_plots(
    joints_left: np.ndarray,
    joints_right: np.ndarray,
    out_dir: Path,
    stem: str,
    joint_indices: list[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(joints_left.shape[0])

    for side, joints in [("left", joints_left), ("right", joints_right)]:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for j_idx in joint_indices:
            if not (0 <= j_idx < joints.shape[1]):
                continue
            axes[0].plot(frames, joints[:, j_idx, 0], label=f"j{j_idx}")
            axes[1].plot(frames, joints[:, j_idx, 1], label=f"j{j_idx}")
            axes[2].plot(frames, joints[:, j_idx, 2], label=f"j{j_idx}")

        axes[0].set_ylabel("x")
        axes[1].set_ylabel("y")
        axes[2].set_ylabel("z")
        axes[2].set_xlabel("frame")
        axes[0].set_title(f"{side.capitalize()} Hand Joint Trajectories")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}_{side}_joint_trajectory.png", dpi=160)
        plt.close(fig)


def _load_oakink_demo(
    demo_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(demo_path, "rb") as f:
        data = pickle.load(f)

    world = data["world_coord"]
    verts_left = np.asarray(world["verts.left"], dtype=np.float32)
    verts_right = np.asarray(world["verts.right"], dtype=np.float32)
    joints_left = np.asarray(world["joints.left"], dtype=np.float32)
    joints_right = np.asarray(world["joints.right"], dtype=np.float32)
    return verts_left, verts_right, joints_left, joints_right


def main():
    parser = argparse.ArgumentParser(
        description="Visualize OakInk hand vertices and joint trajectories in MeshCat."
    )
    parser.add_argument(
        "--demo_path",
        type=str,
        required=True,
        help="Path to OakInk converted .pkl file.",
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
        "--joint_num_points",
        type=int,
        default=64,
        help="Max points sampled for static trajectory cloud per hand.",
    )
    parser.add_argument(
        "--joint_indices",
        type=str,
        default="0,4,8,12,16,20",
        help="Comma-separated joint indices to plot trajectories for.",
    )
    args = parser.parse_args()

    start_timestep, end_timestep = _parse_timesteps(args.timesteps)
    joint_indices = [int(x.strip()) for x in args.joint_indices.split(",") if x.strip()]

    verts_left, verts_right, joints_left, joints_right = _load_oakink_demo(
        args.demo_path
    )
    verts_left = _slice_frames(verts_left, start_timestep, end_timestep, args.skip)
    verts_right = _slice_frames(verts_right, start_timestep, end_timestep, args.skip)
    joints_left = _slice_frames(joints_left, start_timestep, end_timestep, args.skip)
    joints_right = _slice_frames(joints_right, start_timestep, end_timestep, args.skip)
    n_frames = verts_left.shape[0]
    joint_indices = [j for j in joint_indices if 0 <= j < joints_left.shape[1]]
    if len(joint_indices) == 0:
        joint_indices = [0]

    plot_dir = Path("tmp")
    demo_stem = Path(args.demo_path).stem
    _save_joint_trajectory_plots(
        joints_left=joints_left,
        joints_right=joints_right,
        out_dir=plot_dir,
        stem=demo_stem,
        joint_indices=joint_indices,
    )
    print(f"Saved trajectory plots to: {plot_dir.resolve()}")

    vis = VisAnimator(port_vis=args.port)
    left_vert_color = np.array([0.10, 0.45, 0.95], dtype=float)
    right_vert_color = np.array([0.95, 0.35, 0.10], dtype=float)
    left_joint_color = np.array([0.00, 0.70, 0.25], dtype=float)
    right_joint_color = np.array([0.90, 0.05, 0.55], dtype=float)

    vis.initialize_points(
        mesh_vertices=verts_left[0],
        mesh_colors=np.tile(left_vert_color[None, :], (verts_left.shape[1], 1)),
        posed=False,
        num_points=args.num_points,
        name="oakink/left_verts",
    )
    vis.initialize_points(
        mesh_vertices=verts_right[0],
        mesh_colors=np.tile(right_vert_color[None, :], (verts_right.shape[1], 1)),
        posed=False,
        num_points=args.num_points,
        name="oakink/right_verts",
    )
    vis.initialize_points(
        mesh_vertices=joints_left[0],
        mesh_colors=np.tile(left_joint_color[None, :], (joints_left.shape[1], 1)),
        posed=False,
        num_points=joints_left.shape[1],
        name="oakink/left_joints",
    )
    vis.initialize_points(
        mesh_vertices=joints_right[0],
        mesh_colors=np.tile(right_joint_color[None, :], (joints_right.shape[1], 1)),
        posed=False,
        num_points=joints_right.shape[1],
        name="oakink/right_joints",
    )

    # Static trajectory clouds for quick spatial understanding.
    vis.initialize_points(
        mesh_vertices=joints_left[:, joint_indices, :].reshape(-1, 3),
        mesh_colors=np.tile(
            left_joint_color[None, :], (n_frames * len(joint_indices), 1)
        ),
        posed=False,
        num_points=args.joint_num_points,
        name="oakink/left_joint_traj",
    )
    vis.initialize_points(
        mesh_vertices=joints_right[:, joint_indices, :].reshape(-1, 3),
        mesh_colors=np.tile(
            right_joint_color[None, :], (n_frames * len(joint_indices), 1)
        ),
        posed=False,
        num_points=args.joint_num_points,
        name="oakink/right_joint_traj",
    )

    vis.initialize()

    for frame_idx in range(n_frames):
        vis.add_frame(
            pts_dict={
                "oakink/left_verts": verts_left[frame_idx],
                "oakink/right_verts": verts_right[frame_idx],
                "oakink/left_joints": joints_left[frame_idx],
                "oakink/right_joints": joints_right[frame_idx],
            }
        )

    vis.play()
    input("OakInk demo loaded in MeshCat. Keep this process running while viewing.")


if __name__ == "__main__":
    main()
