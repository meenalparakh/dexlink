from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from urdfpy import URDF

from dexlink.src.retargeting.data.human_hand_kpts import (
    human_hand_kpt_positions_from_faces,
    load_precomputed_human_hand_kpt_face_indices,
)
from dexlink.src.retargeting.optimization.target_transform_utils import (
    apply_target_mesh_transform,
    load_target_transform,
)
from dexlink.utils.retargeting.math_utils import get_X
from dexlink.utils.retargeting.palm_vertex_indices import (
    LEFT_PALM_VERTEX_INDICES,
    RIGHT_PALM_VERTEX_INDICES,
)
from dexlink.utils.retargeting.path_utils import get_assets_dir, get_logs_dir, process_urdf
from dexlink.utils.retargeting.train_utils import add_noise_to_jpos
from dexlink.utils.retargeting.urdf_utils import create_urdf_from_obj, scale_urdf

# my wrapper to extract useful information from the original arctic demo file.
# ideally should just be a good queryable class, without doing complex processing on the demos.


if TYPE_CHECKING:
    from dexlink.src.rl.sim.robot import Hand


def np_array_to_list(arr):
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    arr = [
        n.decode("utf-8") if isinstance(n, (bytes, np.bytes_)) else str(n) for n in arr
    ]
    return arr


def normalize_retargeted_joint_name(name: str) -> str:
    name = str(name)

    # Drop robot instance prefixes like `handleft01151_` / `handrightXXXX_`.
    for side_prefix in ("L_", "R_"):
        marker = f"_{side_prefix}"
        if marker in name:
            name = name[name.index(marker) + 1 :]
            break

    # Retargeted trajectories may append an axis/value suffix for scalar joints.
    for suffix in ("_q", "_x", "_y", "_z"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    return name


def normalize_retargeted_joint_names(names):
    return [normalize_retargeted_joint_name(n) for n in names]


class RandomTrajProcessor:
    def __init__(
        self,
        original_demo_path: str | None = None,
        target_object: str | None = None,
        start_timestep=0,
        end_timestep=-1,
        skip=1,
        reference_urdf=None,
        target_urdf=None,
        retargeted_results: str | None = None,
    ):
        # A lightweight processor for hand-authored robot+object trajectories.
        # It mirrors ArcticProcessor where possible, but its primary payload is
        # `retargeted_motion` (there is no human mocap counterpart).
        # self.original_demo_path = original_demo_path
        self.robot_obj_traj_path = retargeted_results
        # self.target_object_name = target_object
        # self.provided_reference_urdf = reference_urdf
        # self.provided_target_urdf = target_urdf
        # self.retargeted_results = retargeted_results
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.skip = skip
        self.hand_joint_map = {}

        # original_demo_path is a npz file obtained from:
        """
        save_dict = {
            "asset_pos": asset_pos,
            "asset_quat": asset_quat,  # wxyz
            "asset_arti_vals": asset_arti_vals,
            "asset_arti_names": asset_arti_names,
            "left_joint_vals": left_joint_vals,
            "left_joint_names": np.array(left_joint_names, dtype=object),
            "right_joint_vals": right_joint_vals,
            "right_joint_names": np.array(right_joint_names, dtype=object),
            "traj_hand_side": np.array(hand_side, dtype=object),
            "traj_hand_name": np.array(hand_name, dtype=object),
            "traj_object_name": np.array(object_name, dtype=object),
            "traj_object_urdf_path": np.array(str(object_urdf_path), dtype=object),
        }

        out_path = get_logs_dir() / "trajopt_results" / save_relpath
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **save_dict)
        """
        self.process()

    def process(self):
        # Minimal placeholders for ArcticProcessor-like attributes.
        self._data = {}
        self.world_coord = {}
        self.left_hand_verts = None
        self.right_hand_verts = None
        self.left_inner_hand_verts = None
        self.right_inner_hand_verts = None
        self.relevant_left_joint_pos = {}
        self.relevant_right_joint_pos = {}

        # Load the trajectory once and use it as both object reference motion and
        # retargeted motion (this processor is already in robot/object space).
        motion = self.load_trajectory()
        self.retargeted_motion = motion
        self.object_reference_motion = {
            k: (
                v.copy()
                if isinstance(v, np.ndarray)
                else list(v) if isinstance(v, list) else v
            )
            for k, v in motion.items()
            if k in {"obj_pos", "obj_quat", "obj_rot", "obj_arti", "joint_names"}
        }

        self._n_frames = motion["obj_pos"].shape[0]
        self._obj_name = self._traj_object_name
        self.target_object_name = self._obj_name

        obj_urdf = self._traj_object_urdf_path
        self._target_obj_urdf_path = process_urdf(obj_urdf)
        self._obj_urdf_path = self._target_obj_urdf_path

        # Maintain ArcticProcessor init behavior: precompute retargeted FK data
        # if a caller asks for it later with concrete hand instances.
        self.process_retargeted_motion()

    def _slice_time(self, arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 0:
            return arr
        end_timestep = arr.shape[0] if self.end_timestep == -1 else self.end_timestep
        return arr[self.start_timestep : end_timestep : self.skip]

    def _unwrap_npz_scalar(self, value):
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8")
        return value

    def load_trajectory(self):
        # if self.retargeted_results is None:
        #     return None

        # # it is assumed that the retargeted results belong to target object
        # results_path = get_logs_dir() / "trajopt_results" / self.retargeted_results
        # the results is a npz files, containing the following dict
        # save_dict = {
        #     "asset_pos": asset_positions,
        #     "asset_quat": asset_quats,
        #     "asset_arti_vals": asset_arti_vals,
        #     "asset_arti_names": asset_arti_names,
        #     "left_joint_vals": left_joint_vals,
        #     'left_joint_names': left_joint_names,
        #     "right_joint_vals": right_joint_vals,
        #     'right_joint_names': right_joint_names,
        # }
        results = np.load(self.robot_obj_traj_path, allow_pickle=True)

        self._traj_object_name = (
            self._unwrap_npz_scalar(results["traj_object_name"])
            if "traj_object_name" in results
            else None
        )
        self._traj_object_urdf_path = (
            self._unwrap_npz_scalar(results["traj_object_urdf_path"])
            if "traj_object_urdf_path" in results
            else None
        )
        self.traj_hand_side = (
            self._unwrap_npz_scalar(results["traj_hand_side"])
            if "traj_hand_side" in results
            else None
        )
        self.traj_hand_name = (
            self._unwrap_npz_scalar(results["traj_hand_name"])
            if "traj_hand_name" in results
            else None
        )
        active_left = (
            bool(self._unwrap_npz_scalar(results["is_active_left"]))
            if "is_active_left" in results
            else self.traj_hand_side == "left"
        )
        active_right = (
            bool(self._unwrap_npz_scalar(results["is_active_right"]))
            if "is_active_right" in results
            else self.traj_hand_side == "right"
        )
        self.hand_side_available = {"left": active_left, "right": active_right}

        asset_pos = self._slice_time(results["asset_pos"]).astype(
            np.float32, copy=False
        )
        asset_quat = self._slice_time(results["asset_quat"]).astype(
            np.float32, copy=False
        )

        asset_arti = None
        if "asset_arti_vals" in results:
            _asset_arti = results["asset_arti_vals"]
            _asset_arti = self._unwrap_npz_scalar(_asset_arti)
            if _asset_arti is not None:
                asset_arti = self._slice_time(_asset_arti).astype(
                    np.float32, copy=False
                )
                if asset_arti.ndim == 1:
                    asset_arti = asset_arti[:, None]

        asset_arti_names = (
            results["asset_arti_names"] if "asset_arti_names" in results else []
        )
        asset_arti_names = self._unwrap_npz_scalar(asset_arti_names)
        if asset_arti_names is None:
            joint_names = []
        elif isinstance(asset_arti_names, (list, tuple, np.ndarray)):
            joint_names = np_array_to_list(np.asarray(asset_arti_names, dtype=object))
        else:
            joint_names = [str(asset_arti_names)]

        if asset_arti is None and len(joint_names) > 0:
            raise ValueError(
                "Trajectory file provides `asset_arti_names` but `asset_arti_vals` is missing/None."
            )
        if asset_arti is not None and asset_arti.shape[-1] != len(joint_names):
            raise ValueError(
                f"asset_arti_vals shape {asset_arti.shape} is incompatible with "
                f"{len(joint_names)} asset_arti_names."
            )

        # asset_quat is in wxyz, convert to xyzw
        asset_quat_xyzw = np.roll(asset_quat, -1, axis=-1)
        asset_rot = R.from_quat(asset_quat_xyzw).as_rotvec()

        left_joint_vals = self._slice_time(results["left_joint_vals"]).astype(
            np.float32, copy=False
        )
        right_joint_vals = self._slice_time(results["right_joint_vals"]).astype(
            np.float32, copy=False
        )
        n_frames = asset_pos.shape[0]
        for name, arr in [
            ("asset_quat", asset_quat),
            ("left_joint_vals", left_joint_vals),
            ("right_joint_vals", right_joint_vals),
        ]:
            if arr.shape[0] != n_frames:
                raise ValueError(
                    f"{name} has {arr.shape[0]} frames, expected {n_frames} to match asset_pos."
                )
        if asset_arti is not None and asset_arti.shape[0] != n_frames:
            raise ValueError(
                f"asset_arti_vals has {asset_arti.shape[0]} frames, expected {n_frames}."
            )

        # this is the new reference motion for the target object
        retargeted_motion = {
            "obj_pos": asset_pos,
            "obj_quat": asset_quat,
            "obj_rot": asset_rot,
            "obj_arti": asset_arti,
            "joint_names": joint_names,
            "left_jnames": normalize_retargeted_joint_names(
                np_array_to_list(results["left_joint_names"])
            ),
            "left_jvals": left_joint_vals,
            "right_jnames": normalize_retargeted_joint_names(
                np_array_to_list(results["right_joint_names"])
            ),
            "right_jvals": right_joint_vals,
        }

        # self.object_reference_motion = retargeted_motion
        return retargeted_motion

    def process_retargeted_motion(
        self, left_hand: Hand | None = None, right_hand: Hand | None = None
    ):
        # additional quantities derived from self.retargeted_motion. specifically
        # 1. the link origin positions keyed by hand.human_mapped_link_names
        #    {link_name: (n_frames, 3)}
        # use urdfpy from the Hand class to compute fk for the retargeted motion
        # some utils might already exist to compute fk. if so, use them.
        # add this as a new attribute "link_pos" (or joint pos works too)
        if self.retargeted_motion is None:
            return None

        motion = self.retargeted_motion

        def _process_side(side: str, hand: Hand):
            retgt_jnames = motion[f"{side}_jnames"]
            retgt_jvals = motion[f"{side}_jvals"]

            missing_joints = [jn for jn in hand.joint_names if jn not in retgt_jnames]
            if missing_joints:
                raise ValueError(
                    f"Retargeted motion missing {side} hand joints required by URDF FK: {missing_joints}"
                )

            # Reorder retargeted joints to the FK URDF joint order.
            jidx_map = [retgt_jnames.index(jn) for jn in hand.joint_names]
            jvals_ordered = retgt_jvals[:, jidx_map]

            jmap = {j.name: jvals_ordered[:, i] for i, j in enumerate(hand.joints)}
            link_fk = hand.urdf_inst.link_fk_batch(cfgs=jmap)
            link_fk = {link.name: poses for link, poses in link_fk.items()}

            link_names = hand.human_mapped_link_names
            missing_links = [ln for ln in link_names if ln not in link_fk]
            if missing_links:
                raise ValueError(
                    f"URDF FK missing mapped links for {side} hand: {missing_links}"
                )

            link_pos = {
                link_name: link_fk[link_name][:, :3, 3].astype(np.float32, copy=False)
                for link_name in link_names
            }

            motion[f"{side}_link_names"] = list(link_names)
            motion[f"{side}_link_pos"] = link_pos
            return link_pos

        results = {}
        if left_hand is not None:
            results["left_link_pos"] = _process_side("left", left_hand)
        if right_hand is not None:
            results["right_link_pos"] = _process_side("right", right_hand)

        # Convenience alias only when exactly one hand is provided.
        if left_hand is not None and right_hand is None:
            motion["link_names"] = motion["left_link_names"]
            motion["link_pos"] = motion["left_link_pos"]
        elif right_hand is not None and left_hand is None:
            motion["link_names"] = motion["right_link_names"]
            motion["link_pos"] = motion["right_link_pos"]

        return results if results else None

    def original_object_motion(self):
        return {
            k: (
                v.copy()
                if isinstance(v, np.ndarray)
                else list(v) if isinstance(v, list) else v
            )
            for k, v in self.object_reference_motion.items()
        }

    def is_identical(self):
        return True

    @property
    def data(self):
        return self._data

    @property
    def n_frames(self):
        return self._n_frames

    @property
    def obj_urdf_path(self):
        return self._obj_urdf_path

    @property
    def obj_name(self):
        return self._obj_name

    @property
    def target_obj_name(self):
        return self.target_object_name

    @property
    def target_obj_urdf_path(self):
        return self._target_obj_urdf_path


def upsample(motion, n_frames, upsample_factor=2.0):
    # get the number of frames in motion, and compute the true upsampling required
    # if it is not near give upsample_factor, raise WARNINg
    # otherwise, do uniform interpolation to match the n_frames
    if not motion:
        return motion
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")

    src_n_frames = None
    for value in motion.values():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            src_n_frames = value.shape[0]
            break
    if src_n_frames is None:
        raise ValueError("Could not infer source frame count for retargeted motion.")

    if src_n_frames == n_frames:
        return {
            k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in motion.items()
        }

    if src_n_frames < 2:
        raise ValueError(
            f"Cannot interpolate retargeted motion with {src_n_frames} source frame(s)."
        )

    true_factor = n_frames / float(src_n_frames)
    if not np.isclose(true_factor, upsample_factor, rtol=0.15, atol=0.05):
        warnings.warn(
            "Retargeted upsample factor mismatch: "
            f"expected ~{upsample_factor:.3f}, actual {true_factor:.3f} "
            f"({src_n_frames} -> {n_frames} frames).",
            stacklevel=2,
        )

    src_t = np.linspace(0.0, 1.0, src_n_frames, dtype=np.float64)
    dst_t = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    out = {}

    for key, value in motion.items():
        if not isinstance(value, np.ndarray) or value.ndim == 0:
            out[key] = value
            continue
        if value.shape[0] != src_n_frames:
            out[key] = value.copy()
            continue

        # Handle quaternion channels with spherical interpolation.
        if "quat" in key and value.shape[-1] == 4:
            quat_xyzw = np.roll(value, -1, axis=-1)
            slerp = Slerp(src_t, R.from_quat(quat_xyzw))
            quat_interp_xyzw = slerp(dst_t).as_quat()
            out[key] = np.roll(
                quat_interp_xyzw.astype(value.dtype, copy=False), 1, axis=-1
            )
            continue

        flat = value.reshape(src_n_frames, -1)
        interp_flat = np.empty((n_frames, flat.shape[1]), dtype=np.float64)
        for i in range(flat.shape[1]):
            interp_flat[:, i] = np.interp(dst_t, src_t, flat[:, i])
        out[key] = interp_flat.reshape((n_frames, *value.shape[1:])).astype(
            value.dtype, copy=False
        )

    return out


def clip_motion(retargeted_motion, start_timestep, end_timestep):
    # retargeted motion has field exact_timesteps: for each frame in the other values, the corersponding time step in the original demo.
    # use that to clip the retargeted motion.
    # assume: the start and end timestep are even numbers, and retargeted motion also has frames for even timesteps. that is exact_timesteps has even numbers.
    if retargeted_motion is None:
        return None
    if not isinstance(retargeted_motion, dict):
        raise TypeError("retargeted_motion must be a dict.")
    if "exact_timesteps" not in retargeted_motion:
        raise KeyError("retargeted_motion must contain `exact_timesteps`.")

    exact_timesteps = np.asarray(retargeted_motion["exact_timesteps"])
    if exact_timesteps.ndim != 1:
        raise ValueError(
            f"`exact_timesteps` must be a 1D array, got shape {exact_timesteps.shape}."
        )
    n_frames = exact_timesteps.shape[0]

    if end_timestep == -1:
        frame_mask = exact_timesteps >= start_timestep
    else:
        frame_mask = (exact_timesteps >= start_timestep) & (
            exact_timesteps < end_timestep
        )

    def _clip_value(value):
        if isinstance(value, np.ndarray):
            if value.ndim > 0 and value.shape[0] == n_frames:
                return value[frame_mask].copy()
            return value.copy()
        if isinstance(value, dict):
            return {k: _clip_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return tuple(value)
        return value

    return {k: _clip_value(v) for k, v in retargeted_motion.items()}


class ArcticProcessor:
    def __init__(
        self,
        original_demo_path: str,
        target_object: str,
        start_timestep=0,
        end_timestep=-1,
        skip=1,  # the step size. 1 means use every frame
        reference_urdf=None,
        target_urdf=None,
        target_obj_path: str | None = None,
        target_transform_info: str | None = None,
        target_scale: float = 1.0,
        run_dir: str | Path | None = None,
        retargeted_results: str | None = None,
        retargeted_results_upsample: bool = False,
        ref_motion_noise_sig: tuple[float, float] = (0.0, 0.0),
    ):
        """
        start_timestep: int, starting timestep to consider from the original demo (inclusive)
        end_timestep: int, ending timestep (exclusive)
        """

        self.original_demo_path = original_demo_path
        self.target_object_name = target_object
        self.provided_reference_urdf = reference_urdf
        self.provided_target_urdf = target_urdf
        self.target_obj_path = target_obj_path
        self.target_transform_info = target_transform_info
        self.target_transform = load_target_transform(target_transform_info)
        self.target_scale = float(target_scale)
        if self.target_scale <= 0.0:
            raise ValueError(
                f"`target_scale` must be positive, got {self.target_scale}."
            )
        self.run_dir = (
            Path(run_dir)
            if run_dir is not None
            else get_logs_dir() / "arctic_processor"
        )
        self.export_dir = self.run_dir / "arctic_processor_exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.retargeted_results = retargeted_results
        self.retargeted_results_upsample = retargeted_results_upsample
        self.ref_motion_noise_sig = ref_motion_noise_sig
        self._human_hand_kpt_face_indices = {}
        # maps important joints in the hand to their indices in the
        # joints array (n_frames, 21, 3)
        # the robots are assumed to have similar mapping provided
        # where the keys map to link names in the urdf
        self.hand_joint_map = {
            "wrist": 0,
            "thumb_tip": 16,
            "index_tip": 17,
            "middle_tip": 18,
            "ring_tip": 19,
            "little_tip": 20,
            "thumb_base": 14,
            "index_base": 1,
            "middle_base": 4,
            "ring_base": 10,
            "little_base": 7,
            "thumb_proximal": 15,
            "index_proximal": 2,
            "middle_proximal": 5,
            "ring_proximal": 11,
            "little_proximal": 8,
            "thumb_distal": 13,
        }
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.skip = skip
        self.process()
        # the following will override the object_reference_motion if retargeted results are provided
        self.retargeted_motion = self.load_retargeted_motion_v2()
        # self.retargeted_motion = self.load_retargeted_motion()
        self.process_retargeted_motion()

    def load_retargeted_motion_v2(self):
        # only load the hand motion from retargeted results.
        # the object motion should be copied from reference self.object_reference_motion
        # the hand motion should be from retargeted results (appropriately sampled)
        if self.retargeted_results is None:
            return None

        # results_path = get_logs_dir() / "trajopt_results" / self.retargeted_results
        results_path = self.retargeted_results
        print("Reading from ", results_path)
        results = np.load(results_path, allow_pickle=True)

        retargeted_motion = {
            "left_jnames": normalize_retargeted_joint_names(
                np_array_to_list(results["left_joint_names"])
            ),
            "left_jvals": results["left_joint_vals"],
            "right_jnames": normalize_retargeted_joint_names(
                np_array_to_list(results["right_joint_names"])
            ),
            "right_jvals": results["right_joint_vals"],
            "exact_timesteps": (
                np.asarray(results["exact_timesteps"], dtype=np.int64)
                if "exact_timesteps" in results
                else np.arange(results["left_joint_vals"].shape[0], dtype=np.int64) * 2
            ),
        }

        retargeted_motion = clip_motion(
            retargeted_motion, self.start_timestep, self.end_timestep
        )

        n_frames = retargeted_motion["left_jvals"].shape[0]
        if retargeted_motion["right_jvals"].shape[0] != n_frames:
            raise AssertionError(
                "Retargeted left/right hand motions have mismatched frame counts: "
                f"{n_frames} vs {retargeted_motion['right_jvals'].shape[0]}."
            )

        if n_frames != self.n_frames:
            if not self.retargeted_results_upsample:
                raise AssertionError(
                    f"Retargeted hand motion has {n_frames} frames, but expected {self.n_frames} frames."
                )
            hand_motion = {
                "left_jvals": retargeted_motion["left_jvals"],
                "right_jvals": retargeted_motion["right_jvals"],
            }
            hand_motion = upsample(hand_motion, self.n_frames)
            retargeted_motion["left_jvals"] = hand_motion["left_jvals"]
            retargeted_motion["right_jvals"] = hand_motion["right_jvals"]

        # Keep object motion from ARCTIC/reference processing.
        ref_motion = self.object_reference_motion
        retargeted_motion.update(
            {
                "obj_pos": ref_motion["obj_pos"].copy(),
                "obj_quat": ref_motion["obj_quat"].copy(),
                "obj_rot": ref_motion["obj_rot"].copy(),
                "obj_arti": (
                    None
                    if ref_motion["obj_arti"] is None
                    else ref_motion["obj_arti"].copy()
                ),
                "joint_names": list(ref_motion["joint_names"]),
            }
        )

        return retargeted_motion

    def load_retargeted_motion(self):
        if self.retargeted_results is None:
            return None

        # it is assumed that the retargeted results belong to target object
        results_path = get_logs_dir() / "trajopt_results" / self.retargeted_results
        # the results is a npz files, containing the following dict
        # save_dict = {
        #     "asset_pos": asset_positions,
        #     "asset_quat": asset_quats,
        #     "asset_arti_vals": asset_arti_vals,
        #     "asset_arti_names": asset_arti_names,
        #     "left_joint_vals": left_joint_vals,
        #     'left_joint_names': left_joint_names,
        #     "right_joint_vals": right_joint_vals,
        #     'right_joint_names': right_joint_names,
        # }
        results = np.load(results_path, allow_pickle=True)

        asset_arti_names = results["asset_arti_names"]
        retargeted_motion = {
            "obj_pos": results["asset_pos"],
            "obj_quat": results["asset_quat"],
            "obj_arti": results["asset_arti_vals"],
            "joint_names": np_array_to_list(asset_arti_names),
            "left_jnames": normalize_retargeted_joint_names(
                np_array_to_list(results["left_joint_names"])
            ),
            "left_jvals": results["left_joint_vals"],
            "right_jnames": normalize_retargeted_joint_names(
                np_array_to_list(results["right_joint_names"])
            ),
            "right_jvals": results["right_joint_vals"],
        }

        # assert that the number of time steps are same as self.n_frames
        # we need one-to-one mapping, because we use retargeted motion for
        # object commands, and initialization, while we use original reference
        # motion for computing rewards (to address the retargeting artifacts)
        # the next commands could be 1 or 2 step in the future.
        n_frames = retargeted_motion["obj_pos"].shape[0]
        if n_frames != self.n_frames:
            if not self.retargeted_results_upsample:
                raise AssertionError(
                    f"Retargeted results has {n_frames} frames, but expected {self.n_frames} frames."
                )
            retargeted_motion = upsample(retargeted_motion, self.n_frames)
            n_frames = retargeted_motion["obj_pos"].shape[0]
            if n_frames != self.n_frames:
                raise AssertionError(
                    f"Upsampled retargeted results has {n_frames} frames, but expected {self.n_frames} frames."
                )

        # asset_quat is in wxyz, convert to xyzw
        asset_quat_xyzw = np.roll(retargeted_motion["obj_quat"], -1, axis=-1)
        asset_rot = R.from_quat(asset_quat_xyzw).as_rotvec()
        retargeted_motion["obj_rot"] = asset_rot

        # self.object_reference_motion = retargeted_motion
        return retargeted_motion

    def process_retargeted_motion(
        self, left_hand: Hand | None = None, right_hand: Hand | None = None
    ):
        # additional quantities derived from self.retargeted_motion. specifically
        # 1. the link origin positions keyed by hand.human_mapped_link_names
        #    {link_name: (n_frames, 3)}
        # use urdfpy from the Hand class to compute fk for the retargeted motion
        # some utils might already exist to compute fk. if so, use them.
        # add this as a new attribute "link_pos" (or joint pos works too)
        if self.retargeted_motion is None:
            return None

        motion = self.retargeted_motion

        def _process_side(side: str, hand: Hand):
            retgt_jnames = normalize_retargeted_joint_names(motion[f"{side}_jnames"])
            motion[f"{side}_jnames"] = retgt_jnames
            retgt_jvals = motion[f"{side}_jvals"]

            missing_joints = [jn for jn in hand.joint_names if jn not in retgt_jnames]
            if missing_joints:
                raise ValueError(
                    f"Retargeted motion missing {side} hand joints required by URDF FK: {missing_joints}"
                )

            # Reorder retargeted joints to the FK URDF joint order.
            jidx_map = [retgt_jnames.index(jn) for jn in hand.joint_names]
            jvals_ordered = retgt_jvals[:, jidx_map]

            jmap = {j.name: jvals_ordered[:, i] for i, j in enumerate(hand.joints)}
            link_fk = hand.urdf_inst.link_fk_batch(cfgs=jmap)
            link_fk = {link.name: poses for link, poses in link_fk.items()}

            link_names = hand.human_mapped_link_names
            missing_links = [ln for ln in link_names if ln not in link_fk]
            if missing_links:
                raise ValueError(
                    f"URDF FK missing mapped links for {side} hand: {missing_links}"
                )

            link_pos = {
                link_name: link_fk[link_name][:, :3, 3].astype(np.float32, copy=False)
                for link_name in link_names
            }

            motion[f"{side}_link_names"] = list(link_names)
            motion[f"{side}_link_pos"] = link_pos
            return link_pos

        results = {}
        if left_hand is not None:
            results["left_link_pos"] = _process_side("left", left_hand)
        if right_hand is not None:
            results["right_link_pos"] = _process_side("right", right_hand)

        # Convenience alias only when exactly one hand is provided.
        if left_hand is not None and right_hand is None:
            motion["link_names"] = motion["left_link_names"]
            motion["link_pos"] = motion["left_link_pos"]
        elif right_hand is not None and left_hand is None:
            motion["link_names"] = motion["right_link_names"]
            motion["link_pos"] = motion["right_link_pos"]

        return results if results else None

    def get_joint_names(self, urdf_path):
        urdf_inst = URDF.load(urdf_path)
        joint_names = [j.name for j in urdf_inst.joints if j.joint_type != "fixed"]
        return joint_names

    def process_seq_len(self, data):

        # get the initial height from the first frame.
        self.initial_height = data["params"]["obj_trans"][0, 2] / 1000.0 - 0.03

        n_frames = data["world_coord"]["joints.left"].shape[0]
        if self.end_timestep == -1:
            self.end_timestep = n_frames

        # this ignores the skip
        self.gt_n_frames = self.end_timestep - self.start_timestep

        # for all the fields in data, slice them accordingly
        for field in ["world_coord", "params"]:
            for key in data[field]:
                val = data[field][key]
                if isinstance(val, np.ndarray):
                    assert (
                        val.shape[0] == n_frames
                    ), f"Expected the first dimension of {field}.{key} to be {n_frames}, got {val.shape[0]}"
                    data[field][key] = val[
                        self.start_timestep : self.end_timestep : self.skip
                    ]

        return data

    def add_noise(self, data):
        # for the joints, add gauss noise
        trans_noise = self.ref_motion_noise_sig[0]
        rot_noise = self.ref_motion_noise_sig[1]

        # if any is > 0.0, call add_noise from train_utils.
        # and set the jpos for left and right hands for all frames
        if trans_noise <= 0.0 and rot_noise <= 0.0:
            return data

        world_coord = data["world_coord"]
        for side in ("left", "right"):
            key = f"joints.{side}"
            if key not in world_coord:
                continue

            joints = np.asarray(world_coord[key])
            if joints.ndim != 3 or joints.shape[-2:] != (21, 3):
                raise ValueError(
                    f"Expected {key} to have shape (T, 21, 3), got {joints.shape}."
                )

            noisy_joints = add_noise_to_jpos(
                jpos=joints,
                noise_scale=(trans_noise, rot_noise),
                wrist_idx=self.hand_joint_map["wrist"],
            )
            world_coord[key] = noisy_joints.astype(joints.dtype, copy=False)

        return data

    def process(self):
        # add to the data:
        # correct urdf path for the object
        self._data = np.load(self.original_demo_path, allow_pickle=True).item()

        self._data = self.process_seq_len(self._data)

        self._data = self.add_noise(self._data)

        self.world_coord = self._data["world_coord"]
        self._obj_name = "_".join(Path(self.original_demo_path).stem.split("_")[:-2])
        # assets here are prefixed with 'arctic'
        self._obj_name = "arctic_" + self._obj_name
        self._obj_urdf_path = process_urdf(
            # self.get_urdf_path(self._obj_name, scale=1.0)
            self.get_urdf_path(
                self._obj_name, scale=1.0, provided_urdf=self.provided_reference_urdf
            ),
            append_jobname=True,
        )
        print(f"Original object urdf path: {self._obj_urdf_path}")
        self._n_frames = self._data["params"]["obj_trans"].shape[0]
        print(f"Number of demo frames: {self._n_frames}")

        self.object_reference_motion = self.replicated_object_motion()

        self.left_hand_verts = self.world_coord[
            "verts.left"
        ]  # (num_frames, num_points, 3)
        self.right_hand_verts = self.world_coord[
            "verts.right"
        ]  # (num_frames, num_points, 3)

        self.relevant_left_joint_pos = {
            k: self.world_coord["joints.left"][:, j_idx]
            for k, j_idx in self.hand_joint_map.items()
        }
        self.relevant_right_joint_pos = {
            k: self.world_coord["joints.right"][:, j_idx]
            for k, j_idx in self.hand_joint_map.items()
        }

        self.left_inner_hand_verts = self.left_hand_verts[
            :, LEFT_PALM_VERTEX_INDICES, :
        ]
        self.right_inner_hand_verts = self.right_hand_verts[
            :, RIGHT_PALM_VERTEX_INDICES, :
        ]
        # self.disc_reference_obs = self._disc_reference_obs()

    def get_human_hand_kpt_face_indices(self, side: str) -> dict[str, int]:
        if side not in ("left", "right"):
            raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")
        if side not in self._human_hand_kpt_face_indices:
            self._human_hand_kpt_face_indices[side] = (
                load_precomputed_human_hand_kpt_face_indices(side)
            )
        return dict(self._human_hand_kpt_face_indices[side])

    def get_human_hand_kpt_positions(
        self,
        side: str,
        names: list[str] | None = None,
        frame_idx: int | None = None,
    ) -> dict[str, np.ndarray]:
        face_indices = self.get_human_hand_kpt_face_indices(side)
        return human_hand_kpt_positions_from_faces(
            side=side,
            vertices=self.world_coord[f"verts.{side}"],
            face_indices=face_indices,
            names=names,
            frame_idx=frame_idx,
        )

    def is_identical(self):
        return self.target_object_name == self._obj_name

    def replicated_object_motion(self):
        if self.target_object_name == self._obj_name:
            print(
                "Target object is the same as original demo object. Using the original motion as it is."
            )
            obj_quat = R.from_rotvec(self._data["params"]["obj_rot"]).as_quat()
            # make it scalar first, np.roll
            obj_quat = np.roll(obj_quat, 1, axis=-1)
            self._target_obj_urdf_path = self._obj_urdf_path

            return {
                "obj_pos": self._data["params"]["obj_trans"]
                / 1000.0,  # convert to meters
                "obj_rot": self._data["params"]["obj_rot"],
                "obj_arti": self._data["params"]["obj_arti"][
                    ..., None
                ],  # add the joint dim (arctic has only 1 joint objects)
                "obj_quat": obj_quat,
                "joint_names": self.get_joint_names(self._obj_urdf_path),
            }
        else:
            print(
                f"Target object is different from original demo object: {self.target_object_name} != {self._obj_name}"
            )
            # self.object_reference_motion = self.replicated_object_motion()
            raise NotImplementedError("override it in subclass")

    def original_object_motion(self):
        obj_quat = R.from_rotvec(self._data["params"]["obj_rot"]).as_quat()
        obj_quat = np.roll(obj_quat, 1, axis=-1)
        return {
            "obj_pos": self._data["params"]["obj_trans"] / 1000.0,  # convert to meters
            "obj_rot": self._data["params"]["obj_rot"],
            "obj_arti": self._data["params"]["obj_arti"][
                ..., None
            ],  # add the joint dim (arctic has only 1 joint objects)
            "obj_quat": obj_quat,
            "joint_names": self.get_joint_names(self._obj_urdf_path),
        }

    def get_trimesh_at_timestep(
        self, timestep, obj_type="original", mesh_type="visual"
    ):
        if obj_type == "original":
            print("Using original object mesh.")
            urdf_path = self._obj_urdf_path
            motion = self.original_object_motion()

        else:
            print("Using target object mesh.")
            urdf_path = self._target_obj_urdf_path
            motion = self.object_reference_motion

        urdf_inst = URDF.load(urdf_path)
        jdict = dict(zip(motion["joint_names"], motion["obj_arti"][timestep]))

        if mesh_type == "collision":
            fk = urdf_inst.collision_trimesh_fk(jdict)
        elif mesh_type == "visual":
            fk = urdf_inst.visual_trimesh_fk(jdict)
        else:
            raise ValueError(
                f"`mesh_type` must be one of ['visual', 'collision'], got {mesh_type!r}."
            )

        mesh_pieces = []
        for mesh, pose in fk.items():
            # transform the trimesh by base_pose @ pose
            transformed_mesh = mesh.copy()
            transformed_mesh.apply_transform(pose)
            mesh_pieces.append(transformed_mesh)

        pos = motion["obj_pos"][timestep]
        quat = motion["obj_quat"][timestep]

        base_pose = get_X(pos, quat)

        # return the concatenated mesh
        return trimesh.util.concatenate(mesh_pieces), base_pose

    def get_urdf_path(self, obj_name, scale=None, provided_urdf=None):
        obj_urdf_dir = get_assets_dir() / obj_name
        obj_urdf_dir.mkdir(parents=True, exist_ok=True)

        if provided_urdf is not None:
            object_urdf_path = provided_urdf
        else:
            # search for urdf file in the directory
            urdf_files = list(obj_urdf_dir.glob("*.urdf"))

            # remove scaled urdf files, and for cluster, remove cluster urdf files
            urdf_files = [f for f in urdf_files if "scaled" not in f.stem]
            urdf_files = [f for f in urdf_files if "cluster" not in f.stem]
            _urdf_files = [f for f in urdf_files if "_abs" in f.stem]
            urdf_files = _urdf_files if len(_urdf_files) > 0 else urdf_files

            decomp_urdf_files = list(obj_urdf_dir.glob("decomp/*_decomp.urdf"))
            # similar filtering for decomp
            decomp_urdf_files = [f for f in decomp_urdf_files if "scaled" not in f.stem]
            decomp_urdf_files = [
                f for f in decomp_urdf_files if "cluster" not in f.stem
            ]
            _decomp_urdf_files = [f for f in decomp_urdf_files if "_abs" in f.stem]
            decomp_urdf_files = (
                _decomp_urdf_files if len(_decomp_urdf_files) > 0 else decomp_urdf_files
            )

            if len(decomp_urdf_files) > 0:
                object_urdf_path = decomp_urdf_files[0]
            elif len(urdf_files) > 0:
                object_urdf_path = urdf_files[0]
            else:
                raise FileNotFoundError(f"No urdf file found in {obj_urdf_dir}")

        if (scale is not None) and (scale != 1.0):
            # create a scaled urdf

            scaled_urdf_path = obj_urdf_dir / f"{obj_name}_scaled.urdf"

            scale_urdf(object_urdf_path, scaled_urdf_path, scale)
            object_urdf_path = scaled_urdf_path

        return object_urdf_path

    # def _disc_reference_obs(self):
    #     # used for discriminator training
    #     trans = self.object_reference_motion["obj_pos"]
    #     quat = self.object_reference_motion["obj_quat"]
    #     if self.disc_rot_format == "quat":
    #         pass
    #     elif self.disc_rot_format == "6d":
    #         rot_mats = R.from_quat(quat).as_matrix()
    #         quat = rot_mats[:, :, :2].reshape(self.n_frames, 6)

    #     arti = self.object_reference_motion["obj_arti"]
    #     obs = np.concatenate([trans, quat, arti], axis=-1)
    #     return torch.from_numpy(obs).float()

    def _command_reference_motion_tensor(self):
        """
        the reference motion to provide to the policy as command input
        for the target object
        """
        # return the object reference motion as torch tensors
        trans = (
            torch.from_numpy(self.object_reference_motion["obj_pos"]).float().to("cpu")
        )
        quat = (
            torch.from_numpy(self.object_reference_motion["obj_quat"]).float().to("cpu")
        )
        arti = (
            torch.from_numpy(self.object_reference_motion["obj_arti"]).float().to("cpu")
        )
        obs = torch.cat([trans, quat, arti], dim=-1)
        assert obs.shape == (
            self.n_frames,
            7 + len(self.object_reference_motion["joint_names"]),
        )
        return obs

    @property
    def command_dim(self):
        return self._command_dim

    @property
    def data(self):
        # return processed data
        return self._data

    @property
    def n_frames(self):
        # return number of frames in the demo
        return self._n_frames

    @property
    def obj_urdf_path(self):
        return self._obj_urdf_path

    @property
    def obj_name(self):
        return self._obj_name

    @property
    def target_obj_name(self):
        return self.target_object_name

    @property
    def target_obj_urdf_path(self):
        return self._target_obj_urdf_path


class LaptopArcticProcessor(ArcticProcessor):

    @property
    def target_obj_urdf_path(self):
        return self._target_obj_urdf_path

    def replicated_object_motion(self):
        print("Hardcoded function to replicate object motion for laptop demo.")

        # if provided urdf, means it was already scaled appropriately
        scale = 1.0 if self.provided_target_urdf else 0.3

        self._target_obj_urdf_path = process_urdf(
            # self.get_urdf_path(self.target_object_name, scale=0.3)
            self.get_urdf_path(
                self.target_object_name,
                scale=scale,
                provided_urdf=self.provided_target_urdf,
            )
        )

        print(f"Target object urdf path: {self._target_obj_urdf_path}")
        urdf_inst = URDF.load(self._target_obj_urdf_path)
        joint_names = [j.name for j in urdf_inst.joints if j.joint_type != "fixed"]

        # enforce the object motion (with appropriate convention transfer
        print("Hardcoded function to replicate object motion for laptop demo.")

        # batch process the original demo to get object motion
        # it is in millimeters, convert to meters
        target_obj_pos = self._data["params"]["obj_trans"] / 1000.0

        # update the rot for the new object
        original_obj_rot_matrix = R.from_rotvec(
            self._data["params"]["obj_rot"]
        ).as_matrix()
        relative_rot = R.from_euler(
            "xyz", [0, math.pi / 2, np.pi]
        ).as_matrix()  # 90 deg about Y
        relative_rot = relative_rot[None, :, :]  # (1, 3, 3)
        target_obj_rot_matrix = relative_rot @ original_obj_rot_matrix  # (n, 3, 3)
        target_obj_rot_aa = R.from_matrix(target_obj_rot_matrix).as_rotvec()
        target_obj_quat = R.from_matrix(target_obj_rot_matrix).as_quat()
        target_obj_quat = np.roll(target_obj_quat, 1, axis=-1)  # make scalar first

        # update the joint articulation
        original_obj_arti = self._data["params"]["obj_arti"]
        target_obj_arti = original_obj_arti - np.pi / 2 - np.pi / 15  # hinge joint only
        target_obj_arti = target_obj_arti[..., None]  # add joint dim

        joint_names = self.get_joint_names(self._target_obj_urdf_path)
        # the new data has form similar to object_demo_data
        target_motion = {
            "obj_pos": target_obj_pos,
            "obj_quat": target_obj_quat,
            "obj_rot": target_obj_rot_aa,
            "obj_arti": target_obj_arti,
            "joint_names": joint_names,
        }

        return target_motion


# [box-align] using T_arctic_to_proc estimate. proc=1.urdf, theta_proc=120.00 deg, theta_arctic=60.00 deg
# [box-align] T_arctic_to_proc =
# [[-0.747751 -0.663946  0.006641  0.125985]
#  [-0.006494 -0.002689 -0.999975  0.04803 ]
#  [ 0.663947 -0.747776 -0.0023    0.084554]
#  [ 0.        0.        0.        1.      ]]
# [laptop-align] using T_arctic_to_proc estimate. proc=1.urdf, theta_proc=120.00 deg, theta_arctic=120.00 deg
# [laptop-align] T_arctic_to_proc =
# [[-0.98494  -0.172473 -0.012054  0.092692]
#  [ 0.011505  0.004186 -0.999925  0.014505]
#  [ 0.17251  -0.985005 -0.002138  0.009996]
#  [ 0.        0.        0.        1.      ]]
# [notebook-align] using T_arctic_to_proc estimate. proc=1.urdf, theta_proc=120.00 deg, theta_arctic=120.00 deg
# [notebook-align] T_arctic_to_proc =
# [[ 0.988397  0.151391  0.01232   0.065728]
#  [ 0.010782  0.010975 -0.999882 -0.029847]
#  [-0.151509  0.988413  0.009215  0.013836]
#  [ 0.        0.        0.        1.      ]]
# [waffleiron-align] using T_arctic_to_proc estimate. proc=1.urdf, theta_proc=120.00 deg, theta_arctic=120.00 deg
# [waffleiron-align] T_arctic_to_proc =
# [[-0.986549 -0.162293 -0.019553  0.116477]
#  [ 0.018525  0.007842 -0.999798  0.027468]
#  [ 0.162414 -0.986711 -0.00473   0.01799 ]
#  [ 0.        0.        0.        1.      ]]
# [ketchup-align] using T_arctic_to_proc estimate. proc=1.urdf, theta_proc=120.00 deg, theta_arctic=120.00 deg
# [ketchup-align] T_arctic_to_proc =
# [[-0.116285  0.993188 -0.007403 -0.024609]
#  [-0.004896  0.00688   0.999964  0.005563]
#  [ 0.993204  0.116317  0.004062  0.158898]
#  [ 0.        0.        0.        1.      ]]

PROC_ARCTIC_TO_PROC_T = {
    "box": np.array(
        [
            [-0.747751, -0.663946, 0.006641, 0.125985],
            [-0.006494, -0.002689, -0.999975, 0.04803],
            [0.663947, -0.747776, -0.0023, 0.084554],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
    "laptop": np.array(
        [
            [-0.98494, -0.172473, -0.012054, 0.092692],
            [0.011505, 0.004186, -0.999925, 0.014505],
            [0.17251, -0.985005, -0.002138, 0.009996],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
    "notebook": np.array(
        [
            [0.988397, 0.151391, 0.01232, 0.065728],
            [0.010782, 0.010975, -0.999882, -0.029847],
            [-0.151509, 0.988413, 0.009215, 0.013836],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
    "waffleiron": np.array(
        [
            [-0.986549, -0.162293, -0.019553, 0.116477],
            [0.018525, 0.007842, -0.999798, 0.027468],
            [0.162414, -0.986711, -0.00473, 0.01799],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
    "ketchup": np.array(
        [
            [-0.116285, 0.993188, -0.007403, -0.024609],
            [-0.004896, 0.00688, 0.999964, 0.005563],
            [0.993204, 0.116317, 0.004062, 0.158898],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
}


class _ProcAlignedArcticProcessor(ArcticProcessor):
    PROC_ASSET_NAME = None
    ARTI_RELATION = "identity"  # "identity" or "pi_minus"

    def _map_arctic_to_proc_articulation(self, arctic_theta):
        arctic_theta = np.asarray(arctic_theta)
        if self.ARTI_RELATION == "pi_minus":
            return np.pi - arctic_theta
        if self.ARTI_RELATION == "identity":
            return arctic_theta
        raise ValueError(f"Unknown articulation relation: {self.ARTI_RELATION}")

    def _estimate_alignment_T_arctic_to_proc(
        self, proc_urdf_path, theta_arctic_ref, theta_proc_ref
    ):
        T_hardcoded = PROC_ARCTIC_TO_PROC_T.get(self.PROC_ASSET_NAME)
        if T_hardcoded is not None:
            print(
                f"{self.PROC_ASSET_NAME}: using hardcoded T_arctic_to_proc "
                f"(verified visually; theta_arctic={theta_arctic_ref:.4f}, theta_proc={theta_proc_ref:.4f})"
            )
            return T_hardcoded.copy()

        from dexlink.src.retargeting.data.arctic_proc_alignment import (
            estimate_box_arctic_to_proc_alignment,
            estimate_ketchup_arctic_to_proc_alignment,
            estimate_laptop_arctic_to_proc_alignment,
            estimate_notebook_arctic_to_proc_alignment,
            estimate_waffleiron_arctic_to_proc_alignment,
        )

        fn_map = {
            "box": estimate_box_arctic_to_proc_alignment,
            "laptop": estimate_laptop_arctic_to_proc_alignment,
            "notebook": estimate_notebook_arctic_to_proc_alignment,
            "waffleiron": estimate_waffleiron_arctic_to_proc_alignment,
            "ketchup": estimate_ketchup_arctic_to_proc_alignment,
        }
        fn = fn_map[self.PROC_ASSET_NAME]
        res = fn(
            proc_urdf_path=proc_urdf_path,
            theta_proc=float(theta_proc_ref),
            theta_arctic=float(theta_arctic_ref),
        )
        print(
            f"{self.PROC_ASSET_NAME}: estimated T_arctic_to_proc with "
            f"theta_arctic={theta_arctic_ref:.4f}, theta_proc={theta_proc_ref:.4f}"
        )
        print(
            np.array2string(res["T_arctic_to_proc"], precision=6, suppress_small=True)
        )
        return res["T_arctic_to_proc"]

    def replicated_object_motion(self):
        if self.PROC_ASSET_NAME is None:
            raise NotImplementedError("Set PROC_ASSET_NAME in subclass")
        print(f"Retargeting ARCTIC demo motion to procedural {self.PROC_ASSET_NAME}.")

        # procedural URDFs are authored in target scale; do not rescale unless caller provides one
        scale = 1.0
        self._target_obj_urdf_path = process_urdf(
            self.get_urdf_path(
                self.target_object_name,
                scale=scale,
                provided_urdf=self.provided_target_urdf,
            )
        )
        print(f"Target object urdf path: {self._target_obj_urdf_path}")

        target_obj_pos_arctic = self._data["params"]["obj_trans"] / 1000.0
        original_obj_rot_matrix = R.from_rotvec(
            self._data["params"]["obj_rot"]
        ).as_matrix()
        original_obj_arti = self._data["params"]["obj_arti"]
        target_obj_arti_vals = self._map_arctic_to_proc_articulation(original_obj_arti)

        # Estimate local-frame alignment once at a representative articulation.
        theta_arctic_ref = float(original_obj_arti[0])
        theta_proc_ref = float(target_obj_arti_vals[0])
        T_arctic_to_proc = self._estimate_alignment_T_arctic_to_proc(
            proc_urdf_path=self._target_obj_urdf_path,
            theta_arctic_ref=theta_arctic_ref,
            theta_proc_ref=theta_proc_ref,
        )
        T_proc_to_arctic = np.linalg.inv(T_arctic_to_proc)

        n_frames = target_obj_pos_arctic.shape[0]
        X_WA = np.tile(np.eye(4), (n_frames, 1, 1))
        X_WA[:, :3, :3] = original_obj_rot_matrix
        X_WA[:, :3, 3] = target_obj_pos_arctic

        X_WP = X_WA @ T_proc_to_arctic[None, :, :]
        target_obj_pos = X_WP[:, :3, 3]
        target_obj_rot_matrix = X_WP[:, :3, :3]
        target_obj_rot_aa = R.from_matrix(target_obj_rot_matrix).as_rotvec()
        target_obj_quat = R.from_matrix(target_obj_rot_matrix).as_quat()
        target_obj_quat = np.roll(target_obj_quat, 1, axis=-1)  # scalar first

        target_obj_arti = target_obj_arti_vals[..., None]  # add joint dim
        joint_names = self.get_joint_names(self._target_obj_urdf_path)
        return {
            "obj_pos": target_obj_pos,
            "obj_quat": target_obj_quat,
            "obj_rot": target_obj_rot_aa,
            "obj_arti": target_obj_arti,
            "joint_names": joint_names,
        }


class ProcBoxArcticProcessor(_ProcAlignedArcticProcessor):
    PROC_ASSET_NAME = "box"
    ARTI_RELATION = "pi_minus"


class ProcLaptopArcticProcessor(_ProcAlignedArcticProcessor):
    PROC_ASSET_NAME = "laptop"
    ARTI_RELATION = "identity"


class ProcNotebookArcticProcessor(_ProcAlignedArcticProcessor):
    PROC_ASSET_NAME = "notebook"
    ARTI_RELATION = "identity"


class ProcWaffleironArcticProcessor(_ProcAlignedArcticProcessor):
    PROC_ASSET_NAME = "waffleiron"
    ARTI_RELATION = "identity"


class CrossWaffleironToProcNotebookArcticProcessor(_ProcAlignedArcticProcessor):
    """Transfer an ARCTIC waffleiron motion into a procedural notebook frame."""

    PROC_ASSET_NAME = "notebook"
    ARTI_RELATION = "identity"

    # Direct waffleiron-to-notebook fit at 120 degrees.  The hinge origins,
    # axes, and top/bottom plate normals define the local-frame correspondence.
    WAFFLEIRON_TO_NOTEBOOK_T = np.array(
        [
            [-0.99220751, -0.12360389, -0.01569505, 0.07673289],
            [0.01471925, 0.00880388, -0.99985291, 0.02792579],
            [0.12372389, -0.99229258, -0.00691592, 0.00980301],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    def _estimate_alignment_T_arctic_to_proc(
        self, proc_urdf_path, theta_arctic_ref, theta_proc_ref
    ):
        print(
            "waffleiron-to-notebook: using direct, visually pending "
            "T_arctic_to_proc fit"
        )
        return self.WAFFLEIRON_TO_NOTEBOOK_T.copy()


class ProcKetchupArcticProcessor(_ProcAlignedArcticProcessor):
    PROC_ASSET_NAME = "ketchup"
    ARTI_RELATION = "identity"


class ExternalObjArcticProcessor(ArcticProcessor):
    def _load_target_mesh(self) -> trimesh.Trimesh:
        if self.target_obj_path is None:
            raise ValueError(
                f"{type(self).__name__} requires `target_obj_path` for target "
                f"{self.target_object_name!r}."
            )
        mesh_path = Path(self.target_obj_path).resolve()
        if not mesh_path.exists():
            raise FileNotFoundError(f"Target OBJ path does not exist: {mesh_path}")
        if mesh_path.suffix.lower() != ".obj":
            raise ValueError(
                f"Expected target OBJ path to end in .obj, got {mesh_path}."
            )

        loaded = trimesh.load(mesh_path, process=False)
        if isinstance(loaded, trimesh.Scene):
            geometries = list(loaded.geometry.values())
            if not geometries:
                raise ValueError(f"Target OBJ contains no geometry: {mesh_path}")
            return trimesh.util.concatenate(geometries)
        if isinstance(loaded, trimesh.Trimesh):
            return loaded
        raise TypeError(
            f"Expected trimesh to load target OBJ as Trimesh or Scene, got {type(loaded)}."
        )

    def _materialize_target_urdf(self) -> Path:
        if self.target_obj_path is None:
            raise ValueError(
                f"{type(self).__name__} requires `target_obj_path` for target "
                f"{self.target_object_name!r}."
            )
        mesh_path = Path(self.target_obj_path).resolve()
        output_dir = self.export_dir / f"{self.target_object_name}_target_mesh"
        output_dir.mkdir(parents=True, exist_ok=True)

        mesh = apply_target_mesh_transform(
            self._load_target_mesh(),
            transform=self.target_transform,
            scale=self.target_scale,
        )
        processed_obj_path = output_dir / f"{mesh_path.stem}_processed.obj"
        mesh.export(processed_obj_path)
        urdf_path = Path(
            create_urdf_from_obj(
                str(processed_obj_path.resolve()),
                str(output_dir),
                scale=1.0,
            )
        )
        return urdf_path

    def replicated_object_motion(self):
        print(f"Retargeting ARCTIC demo motion to external OBJ {self.target_obj_path}.")
        self._target_obj_urdf_path = process_urdf(self._materialize_target_urdf())
        print(f"Target object urdf path: {self._target_obj_urdf_path}")

        obj_rot = self._data["params"]["obj_rot"]
        obj_quat = R.from_rotvec(obj_rot).as_quat()
        obj_quat = np.roll(obj_quat, 1, axis=-1)
        n_frames = self._data["params"]["obj_trans"].shape[0]
        return {
            "obj_pos": self._data["params"]["obj_trans"] / 1000.0,
            "obj_quat": obj_quat,
            "obj_rot": obj_rot,
            "obj_arti": np.zeros((n_frames, 0), dtype=np.float32),
            "joint_names": [],
        }


ARCTIC_ABS_URDF_PATHS = {
    "arctic_box": get_assets_dir() / "arctic_box" / "decomp" / "box_decomp_abs.urdf",
    "arctic_waffleiron": get_assets_dir()
    / "arctic_waffleiron"
    / "decomp"
    / "waffleiron_decomp_abs.urdf",
    "arctic_notebook": get_assets_dir()
    / "arctic_notebook"
    / "decomp"
    / "notebook_decomp_abs.urdf",
    "arctic_ketchup": get_assets_dir()
    / "arctic_ketchup"
    / "decomp"
    / "ketchup_decomp_abs_obj.urdf",
    "arctic_laptop": get_assets_dir()
    / "arctic_laptop"
    / "decomp"
    / "laptop_decomp_abs_obj.urdf",
}

class_dict = {
    "laptop": {
        "processor_cls": LaptopArcticProcessor,
        "reference_urdf": None,
        "target_urdf_path": None,
    },
    "ipad": {
        "processor_cls": LaptopArcticProcessor,
        "reference_urdf": None,
        "target_urdf_path": None,
    },
    "arctic_box": {
        "processor_cls": ArcticProcessor,
        "reference_urdf": ARCTIC_ABS_URDF_PATHS["arctic_box"],
        "target_urdf_path": ARCTIC_ABS_URDF_PATHS["arctic_box"],
    },
    "arctic_laptop": {
        "processor_cls": ArcticProcessor,
        "reference_urdf": ARCTIC_ABS_URDF_PATHS["arctic_laptop"],
        "target_urdf_path": ARCTIC_ABS_URDF_PATHS["arctic_laptop"],
    },
    "arctic_ketchup": {
        "processor_cls": ArcticProcessor,
        "reference_urdf": ARCTIC_ABS_URDF_PATHS["arctic_ketchup"],
        "target_urdf_path": ARCTIC_ABS_URDF_PATHS["arctic_ketchup"],
    },
    "arctic_waffleiron": {
        "processor_cls": ArcticProcessor,
        "reference_urdf": ARCTIC_ABS_URDF_PATHS["arctic_waffleiron"],
        "target_urdf_path": ARCTIC_ABS_URDF_PATHS["arctic_waffleiron"],
    },
    "arctic_notebook": {
        "processor_cls": ArcticProcessor,
        "reference_urdf": ARCTIC_ABS_URDF_PATHS["arctic_notebook"],
        "target_urdf_path": ARCTIC_ABS_URDF_PATHS["arctic_notebook"],
    },
    "banana_handover": {
        "processor_cls": ExternalObjArcticProcessor,
        "reference_urdf": ARCTIC_ABS_URDF_PATHS["arctic_ketchup"],
        "target_urdf_path": None,
    },
}


def _register_proc_variation_entries():
    proc_specs = {
        "box": (ProcBoxArcticProcessor, ARCTIC_ABS_URDF_PATHS["arctic_box"]),
        "laptop": (ProcLaptopArcticProcessor, ARCTIC_ABS_URDF_PATHS["arctic_laptop"]),
        "notebook": (
            ProcNotebookArcticProcessor,
            ARCTIC_ABS_URDF_PATHS["arctic_notebook"],
        ),
        "waffleiron": (
            ProcWaffleironArcticProcessor,
            ARCTIC_ABS_URDF_PATHS["arctic_waffleiron"],
        ),
        "ketchup": (
            ProcKetchupArcticProcessor,
            ARCTIC_ABS_URDF_PATHS["arctic_ketchup"],
        ),
    }

    assets_dir = get_assets_dir()

    def _iter_proc_variants(asset_name: str):
        proc_dir = assets_dir / f"proc_{asset_name}"
        if not proc_dir.exists():
            return

        sample_dirs = sorted(
            [p for p in proc_dir.iterdir() if p.is_dir()],
            key=lambda p: (not p.name.isdigit(), p.name),
        )
        for sample_dir in sample_dirs:
            var_idx = sample_dir.name
            base_urdf = sample_dir / f"{var_idx}.urdf"
            if not base_urdf.exists():
                # Fallback: first non-6dof urdf in the sample dir.
                candidates = sorted(
                    p
                    for p in sample_dir.glob("*.urdf")
                    if not p.name.endswith("_6dof.urdf")
                )
                if not candidates:
                    continue
                base_urdf = candidates[0]

            yield var_idx, base_urdf

    for asset_name, (processor_cls, ref_urdf) in proc_specs.items():
        for var_idx, base_urdf in _iter_proc_variants(asset_name):
            class_dict[f"proc_{asset_name}_{var_idx}"] = {
                "processor_cls": processor_cls,
                "reference_urdf": ref_urdf,
                "target_urdf_path": base_urdf,
            }

    # Cross-category: use ARCTIC waffleiron demonstrations with laptop targets.
    # class_dict["cross_waffleiron_to_arctic_laptop"] = {
    #     "processor_cls": ProcWaffleironArcticProcessor,
    #     "reference_urdf": ARCTIC_ABS_URDF_PATHS["arctic_waffleiron"],
    #     "target_urdf_path": ARCTIC_ABS_URDF_PATHS["arctic_laptop"],
    # }
    for var_idx, base_urdf in _iter_proc_variants("notebook"):
        class_dict[f"cross_waffleiron_to_proc_notebook_{var_idx}"] = {
            "processor_cls": CrossWaffleironToProcNotebookArcticProcessor,
            "reference_urdf": ARCTIC_ABS_URDF_PATHS["arctic_waffleiron"],
            "target_urdf_path": base_urdf,
        }


_register_proc_variation_entries()


def get_processor_cfg(target_object: str):
    try:
        entry = class_dict[target_object]
    except KeyError:
        [print(k) for k in class_dict.keys()]
        raise ValueError(f"Unknown target object: {target_object}.")
    if isinstance(entry, type):
        return {
            "processor_cls": entry,
            "reference_urdf": None,
            "target_urdf_path": None,
        }
    return entry


def _urdf_single_joint_cfg(urdf_inst: URDF, theta: float):
    cfg = {}
    for joint in urdf_inst.joints:
        if joint.joint_type == "fixed":
            continue
        cfg[joint] = float(theta)
    return cfg


def _visual_mesh_entries_with_link(urdf_inst: URDF, joint_cfg: dict):
    link_fk = urdf_inst.link_fk(cfg=joint_cfg, links=None)
    for link, link_pose in link_fk.items():
        visuals = getattr(link, "visuals", None) or []
        for visual in visuals:
            geom = getattr(visual, "geometry", None)
            if geom is None:
                continue
            origin = visual.origin if visual.origin is not None else np.eye(4)
            meshes = getattr(geom, "meshes", None) or []
            for mesh in meshes:
                pose = link_pose @ origin
                mesh_field = getattr(geom, "mesh", None)
                scale = (
                    getattr(mesh_field, "scale", None)
                    if mesh_field is not None
                    else None
                )
                if scale is not None:
                    S = np.eye(4, dtype=np.float64)
                    S[:3, :3] = np.diag(scale)
                    pose = pose @ S
                yield link.name, mesh, pose


def _link_points_world(urdf_inst: URDF, theta: float, link_name: str) -> np.ndarray:
    pts = []
    joint_cfg = _urdf_single_joint_cfg(urdf_inst, theta)
    for cur_link_name, mesh, pose in _visual_mesh_entries_with_link(
        urdf_inst, joint_cfg
    ):
        if cur_link_name != link_name:
            continue
        verts = np.asarray(mesh.vertices)
        if verts.size == 0:
            continue
        verts_h = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)
        pts_world = (pose @ verts_h.T).T[:, :3]
        pts.append(pts_world)

    if not pts:
        raise ValueError(f"No visual mesh vertices found for link '{link_name}'.")
    return np.concatenate(pts, axis=0)


def _obb_center_and_normal(points: np.ndarray):
    mean = points.mean(axis=0)
    centered = points - mean
    cov = centered.T @ centered / max(points.shape[0], 1)
    evals, evecs = np.linalg.eigh(cov)
    # Columns of evecs correspond to ascending eigenvalues.
    local = centered @ evecs
    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    extents = maxs - mins
    center_local = 0.5 * (mins + maxs)
    center_world = mean + evecs @ center_local

    normal_idx = int(np.argmin(extents))
    normal = evecs[:, normal_idx]
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    return center_world, normal, extents, evecs


def _fit_rigid_transform(src_pts: np.ndarray, dst_pts: np.ndarray):
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)
    src_c = src_pts - src_mean
    dst_c = dst_pts - dst_mean
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R_fit = Vt.T @ U.T
    if np.linalg.det(R_fit) < 0:
        Vt[-1, :] *= -1
        R_fit = Vt.T @ U.T
    t_fit = dst_mean - R_fit @ src_mean
    T = np.eye(4)
    T[:3, :3] = R_fit
    T[:3, 3] = t_fit
    return T


def _apply_T(T: np.ndarray, p: np.ndarray):
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        return T[:3, :3] @ p + T[:3, 3]
    return (T[:3, :3] @ p.T).T + T[:3, 3]


def _box_link_frame_summary(urdf_path: str | Path, theta: float):
    urdf_inst = URDF.load(str(urdf_path))
    out = {}
    for link_name in ("bottom", "top"):
        pts = _link_points_world(urdf_inst, theta=theta, link_name=link_name)
        center, normal, extents, basis = _obb_center_and_normal(pts)
        out[link_name] = {
            "center": center,
            "normal": normal,
            "extents_pca": extents,
            "basis_pca": basis,
            "n_points": int(pts.shape[0]),
        }
    return out


def estimate_box_arctic_to_proc_alignment(
    proc_urdf_path: str | Path,
    arctic_urdf_path: str | Path | None = None,
    theta_proc: float = math.radians(60.0),
    theta_arctic: float | None = None,
):
    """
    Estimate rigid transform T such that p_proc ~= T @ p_arctic (homogeneous convention).

    Uses link-wise geometric summaries (OBB center + minor-axis normal) for the `bottom`
    and `top` links at matching articulation. By default, theta_arctic is inferred from:
    theta_arctic = pi - theta_proc
    """
    if arctic_urdf_path is None:
        arctic_urdf_path = ARCTIC_ABS_URDF_PATHS["arctic_box"]
    if theta_arctic is None:
        theta_arctic = math.pi - theta_proc

    arctic = _box_link_frame_summary(arctic_urdf_path, theta=float(theta_arctic))
    proc = _box_link_frame_summary(proc_urdf_path, theta=float(theta_proc))

    # Sign of PCA normals is ambiguous. Search sign flips and choose the best residual.
    best = None
    for s_bottom in (-1.0, 1.0):
        for s_top in (-1.0, 1.0):
            a_bottom_n = s_bottom * arctic["bottom"]["normal"]
            a_top_n = s_top * arctic["top"]["normal"]

            src_pts = np.stack(
                [
                    arctic["bottom"]["center"],
                    arctic["bottom"]["center"] + a_bottom_n,
                    arctic["top"]["center"],
                    arctic["top"]["center"] + a_top_n,
                ],
                axis=0,
            )
            dst_pts = np.stack(
                [
                    proc["bottom"]["center"],
                    proc["bottom"]["center"] + proc["bottom"]["normal"],
                    proc["top"]["center"],
                    proc["top"]["center"] + proc["top"]["normal"],
                ],
                axis=0,
            )
            T = _fit_rigid_transform(src_pts, dst_pts)

            a_bottom_n_aligned = _apply_T(T, a_bottom_n) - _apply_T(T, np.zeros(3))
            a_top_n_aligned = _apply_T(T, a_top_n) - _apply_T(T, np.zeros(3))
            a_bottom_n_aligned /= max(np.linalg.norm(a_bottom_n_aligned), 1e-12)
            a_top_n_aligned /= max(np.linalg.norm(a_top_n_aligned), 1e-12)

            pos_err = np.linalg.norm(_apply_T(T, src_pts) - dst_pts, axis=1).mean()
            ang_err = math.acos(
                np.clip(np.dot(a_bottom_n_aligned, proc["bottom"]["normal"]), -1.0, 1.0)
            ) + math.acos(
                np.clip(np.dot(a_top_n_aligned, proc["top"]["normal"]), -1.0, 1.0)
            )
            score = pos_err + 0.05 * ang_err
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "sign_flips": {"bottom": s_bottom, "top": s_top},
                    "T_arctic_to_proc": T,
                    "pos_err_mean": pos_err,
                    "ang_err_sum_rad": ang_err,
                }

    result = {
        "theta_proc": float(theta_proc),
        "theta_arctic": float(theta_arctic),
        "articulation_relation": "theta_arctic = pi - theta_proc",
        "arctic": arctic,
        "proc": proc,
        **best,
    }
    return result


def print_box_alignment_report(
    proc_urdf_path: str | Path,
    arctic_urdf_path: str | Path | None = None,
    theta_proc: float = math.radians(60.0),
    theta_arctic: float | None = None,
):
    res = estimate_box_arctic_to_proc_alignment(
        proc_urdf_path=proc_urdf_path,
        arctic_urdf_path=arctic_urdf_path,
        theta_proc=theta_proc,
        theta_arctic=theta_arctic,
    )

    print("=== Box ARCTIC -> Procedural Alignment ===")
    print(f"proc urdf:   {proc_urdf_path}")
    print(f"arctic urdf: {arctic_urdf_path or ARCTIC_ABS_URDF_PATHS['arctic_box']}")
    print(
        f"theta_proc={res['theta_proc']:.6f} rad ({math.degrees(res['theta_proc']):.2f} deg), "
        f"theta_arctic={res['theta_arctic']:.6f} rad ({math.degrees(res['theta_arctic']):.2f} deg)"
    )
    for family in ("arctic", "proc"):
        for link_name in ("bottom", "top"):
            info = res[family][link_name]
            c = np.array2string(info["center"], precision=6, suppress_small=True)
            n = np.array2string(info["normal"], precision=6, suppress_small=True)
            e = np.array2string(info["extents_pca"], precision=6, suppress_small=True)
            print(f"{family}.{link_name}.center = {c}")
            print(f"{family}.{link_name}.normal = {n}")
            print(f"{family}.{link_name}.extents_pca = {e}")
    print(f"chosen normal sign flips (ARCTIC): {res['sign_flips']}")
    print(f"mean anchor position error: {res['pos_err_mean']:.6e} m")
    print(f"normal angular error sum:   {res['ang_err_sum_rad']:.6e} rad")
    print("T_arctic_to_proc =")
    print(np.array2string(res["T_arctic_to_proc"], precision=6, suppress_small=True))
    return res


if __name__ == "__main__":
    import math

    print_box_alignment_report(
        proc_urdf_path="assets/proc_box/box_0.urdf",
        theta_proc=math.radians(180.0),  # or your chosen proc articulation
        # theta_arctic defaults to math.pi - theta_proc
    )
