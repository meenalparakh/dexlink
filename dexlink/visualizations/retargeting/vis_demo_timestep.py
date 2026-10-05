import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from dexlink.src.retargeting.data.process_arctic import get_processor_cfg
from dexlink.utils.retargeting.path_utils import process_urdf
from dexlink.utils.retargeting.vis_server import VisAnimator, VizServer


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


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ARCTIC reference hands + reference object motion + retargeted target object motion."
    )
    parser.add_argument(
        "--demo_path",
        type=str,
        required=True,
        help="Path to original ARCTIC demo .npy/.npz file used by ArcticProcessor.",
    )
    parser.add_argument(
        "--target_object",
        type=str,
        required=True,
        help="Target object key for process_arctic.get_processor_cfg (e.g. arctic_laptop, proc_box_1).",
    )
    parser.add_argument("--start_timestep", type=int, default=0)
    parser.add_argument("--end_timestep", type=int, default=-1)
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
    parser.add_argument(
        "--retargeted_results",
        type=str,
        default=None,
        help="Optional trajopt result filename under logs/trajopt_results to override target object motion in processor.",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=None,
        help="If provided, render only this processed frame index as a static view.",
    )
    args = parser.parse_args()

    processor_cfg = get_processor_cfg(args.target_object)
    processor_cls = processor_cfg["processor_cls"]

    processor = processor_cls(
        original_demo_path=args.demo_path,
        target_object=args.target_object,
        start_timestep=args.start_timestep,
        end_timestep=args.end_timestep,
        skip=args.skip,
        reference_urdf=processor_cfg["reference_urdf"],
        target_urdf=processor_cfg["target_urdf_path"],
        retargeted_results=args.retargeted_results,
        retargeted_results_upsample=True,
    )

    # Hand vertices from the reference (original ARCTIC) motion
    left_hand = processor.left_hand_verts
    right_hand = processor.right_hand_verts
    n_frames = processor.n_frames

    left_color = np.array([0.10, 0.45, 0.95], dtype=float)
    right_color = np.array([0.95, 0.35, 0.10], dtype=float)

    # Object motions
    ref_obj_motion = processor.original_object_motion()
    tgt_obj_motion = processor.object_reference_motion

    ref_obj_urdf = URDF.load(str(process_urdf(processor.obj_urdf_path)))
    tgt_obj_urdf = URDF.load(str(process_urdf(processor.target_obj_urdf_path)))

    vis = VisAnimator(port_vis=args.port)

    vis.initialize_points(
        mesh_vertices=left_hand[0],
        mesh_colors=np.tile(left_color[None, :], (left_hand.shape[1], 1)),
        posed=False,
        num_points=args.num_points,
        name="reference/left_hand",
    )
    vis.initialize_points(
        mesh_vertices=right_hand[0],
        mesh_colors=np.tile(right_color[None, :], (right_hand.shape[1], 1)),
        posed=False,
        num_points=args.num_points,
        name="reference/right_hand",
    )

    vis.initialize_urdf(
        urdf_inst=ref_obj_urdf,
        name="reference/object",
        mesh_type=args.mesh_type,
        color="cool",
    )
    vis.initialize_urdf(
        urdf_inst=tgt_obj_urdf,
        name="target/object",
        mesh_type=args.mesh_type,
        color="warm",
    )
    vis.initialize()

    if args.timestep is not None:
        if args.timestep < 0 or args.timestep >= n_frames:
            raise ValueError(
                f"--timestep must be in [0, {n_frames - 1}], got {args.timestep}."
            )

        frame_idx = args.timestep
        vis_static = VizServer(mc_vis=vis.mc_vis)

        ref_mesh, ref_pose = processor.get_trimesh_at_timestep(
            timestep=frame_idx,
            obj_type="original",
            mesh_type=args.mesh_type,
        )
        tgt_mesh, tgt_pose = processor.get_trimesh_at_timestep(
            timestep=frame_idx,
            obj_type="target",
            mesh_type=args.mesh_type,
        )
        vis_static.view_trimesh(
            name="reference/object_mesh",
            mesh=ref_mesh,
            pose=ref_pose,
            color=0x2E86C1,
            opacity=0.65,
        )
        vis_static.view_trimesh(
            name="target/object_mesh",
            mesh=tgt_mesh,
            pose=tgt_pose,
            color=0xD35400,
            opacity=0.65,
        )

        # also visualize the urdfpy for the reference and target object.

        print(
            f"Rendered frame {frame_idx} in MeshCat with reference + target trimesh objects."
        )

    for frame_idx in range(n_frames):
        vis.add_frame(
            pts_dict={
                "reference/left_hand": left_hand[frame_idx],
                "reference/right_hand": right_hand[frame_idx],
            },
            urdf_dict={
                "reference/object": _pose_dict_for_motion_frame(
                    ref_obj_motion, frame_idx
                ),
                "target/object": _pose_dict_for_motion_frame(tgt_obj_motion, frame_idx),
            },
        )

    vis.play()
    print("Animation loaded in MeshCat. Keep this process running while viewing.")


if __name__ == "__main__":
    main()
