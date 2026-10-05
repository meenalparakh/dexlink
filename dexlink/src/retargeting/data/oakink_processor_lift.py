from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh
from urdfpy import URDF

from dexlink.src.retargeting.data.human_hand_kpts import (
    human_hand_kpt_positions_from_faces,
    load_precomputed_human_hand_kpt_face_indices,
)
from dexlink.src.retargeting.data.oakink_processor_grasp import (
    normalize_retargeted_joint_names,
    np_array_to_list,
)
from dexlink.utils.retargeting.math_utils import get_X
from dexlink.utils.retargeting.path_utils import process_urdf

if TYPE_CHECKING:
    from dexlink.src.rl.sim.robot import Hand


class OakinkProcessorLift:
    def __init__(
        self,
        exported_npy_path: str | Path,
        start_timestep: int = 0,
        end_timestep: int = -1,
        skip: int = 1,
        support_height: float = 0.75,
        support_margin: float = 0.002,
    ):
        self.exported_npy_path = str(exported_npy_path)
        self.start_timestep = int(start_timestep)
        self.end_timestep = int(end_timestep)
        self.skip = int(skip)
        self.support_height = float(support_height)
        self.support_margin = float(support_margin)
        self.demo_ht = 0.0
        if self.skip <= 0:
            raise ValueError("skip must be positive.")

        self._human_hand_kpt_face_indices = {}
        self.hand_joint_map = {
            "wrist": 0,
            "thumb_tip": 4,
            "index_tip": 8,
            "middle_tip": 12,
            "ring_tip": 16,
            "little_tip": 20,
            "thumb_base": 2,
            "index_base": 5,
            "middle_base": 9,
            "ring_base": 13,
            "little_base": 17,
            "thumb_proximal": 3,
            "index_proximal": 6,
            "middle_proximal": 10,
            "ring_proximal": 14,
            "little_proximal": 18,
        }

        self._load()
        self._build_retargeted_motion()

    def compute_demo_base_ht(
        self,
        obj_pos: np.ndarray,
        obj_quat: np.ndarray,
        obj_arti: np.ndarray,
        joint_names: list[str],
    ) -> float:
        if self.robot_base_pose is not None:
            robot_base_z = float(
                np.asarray(self.robot_base_pose, dtype=np.float32)[2, 3]
            )
            return self.support_height - robot_base_z

        urdf_inst = URDF.load(str(self._target_obj_urdf_path))
        first_obj_pos = np.asarray(obj_pos[0], dtype=np.float32)
        first_obj_quat = np.asarray(obj_quat[0], dtype=np.float32)
        first_obj_arti = np.asarray(obj_arti[0], dtype=np.float32).reshape(-1)
        joint_dict = dict(zip(joint_names, first_obj_arti))

        fk = urdf_inst.visual_trimesh_fk(joint_dict)
        if len(fk) == 0:
            fk = urdf_inst.collision_trimesh_fk(joint_dict)
        mesh_pieces = []
        for mesh, pose in fk.items():
            transformed_mesh = mesh.copy()
            transformed_mesh.apply_transform(pose)
            mesh_pieces.append(transformed_mesh)
        mesh = trimesh.util.concatenate(mesh_pieces)
        mesh.apply_transform(get_X(first_obj_pos, first_obj_quat))
        lowest_z = float(mesh.bounds[0, 2])
        return self.support_height + self.support_margin - lowest_z

    def _apply_demo_height(self, arr: np.ndarray) -> np.ndarray:
        """Shift xyz positions upward by demo_ht along z."""
        out = np.asarray(arr).astype(np.float32, copy=True)
        out[..., 2] += float(self.demo_ht)
        return out

    def _slice_time(self, arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 0:
            return arr
        end = arr.shape[0] if self.end_timestep == -1 else self.end_timestep
        return arr[self.start_timestep : end : self.skip]

    def _apply_demo_height_to_joint_values(
        self, jvals: np.ndarray, jnames: list[str]
    ) -> np.ndarray:
        out = np.asarray(jvals, dtype=np.float32).copy()
        for tz_joint_name in ("L_forearm_tz_link_joint", "R_forearm_tz_link_joint"):
            if tz_joint_name in jnames:
                out[..., jnames.index(tz_joint_name)] += float(self.demo_ht)
        return out

    def _load_drake_collision_refinement(self, payload: dict) -> dict:
        if "drake_collision_refinement_jvals" not in payload:
            return {}
        required_keys = (
            "drake_collision_refinement_jvals",
            "drake_collision_refinement_jnames",
            "drake_collision_refinement_hand",
            "drake_collision_refinement_unresolved_frame_mask",
            "drake_collision_refinement_final_collision_violation",
            "drake_collision_refinement_collision_distance",
        )
        missing = [key for key in required_keys if key not in payload]
        if missing:
            raise ValueError(
                "Drake collision refinement payload is incomplete; missing "
                f"{missing}."
            )

        jnames = np_array_to_list(
            np.asarray(payload["drake_collision_refinement_jnames"], dtype=object)
        )
        jvals = self._slice_time(
            np.asarray(payload["drake_collision_refinement_jvals"], dtype=np.float32)
        )
        if jvals.ndim != 2:
            raise ValueError(
                "Expected drake_collision_refinement_jvals shape (T, J), got "
                f"{jvals.shape}."
            )
        jvals = self._apply_demo_height_to_joint_values(jvals, jnames)
        mask = self._slice_time(
            np.asarray(
                payload["drake_collision_refinement_unresolved_frame_mask"],
                dtype=bool,
            )
        )
        if mask.ndim != 1 or mask.shape[0] != jvals.shape[0]:
            raise ValueError(
                "Expected drake_collision_refinement_unresolved_frame_mask shape "
                f"({jvals.shape[0]},), got {mask.shape}."
            )
        final_violation = self._slice_time(
            np.asarray(
                payload["drake_collision_refinement_final_collision_violation"],
                dtype=np.float32,
            )
        )
        if final_violation.ndim != 1 or final_violation.shape[0] != jvals.shape[0]:
            raise ValueError(
                "Expected drake_collision_refinement_final_collision_violation shape "
                f"({jvals.shape[0]},), got {final_violation.shape}."
            )
        hand = str(payload["drake_collision_refinement_hand"])
        refinement = {
            "drake_collision_refinement_jvals": jvals,
            "drake_collision_refinement_jnames": list(jnames),
            "drake_collision_refinement_hand": hand,
            "drake_collision_refinement_unresolved_frame_mask": mask,
            "drake_collision_refinement_unresolved_frame_indices": np.flatnonzero(
                mask
            ).astype(np.int64),
            "drake_collision_refinement_final_collision_violation": final_violation,
            "drake_collision_refinement_collision_distance": np.asarray(
                payload["drake_collision_refinement_collision_distance"],
                dtype=np.float32,
            ),
        }
        side_key = f"drake_collision_refinement_{hand}_jvals"
        if side_key in payload:
            side_jvals = self._slice_time(
                np.asarray(payload[side_key], dtype=np.float32)
            )
            if side_jvals.shape != jvals.shape:
                raise ValueError(
                    f"Expected {side_key} shape {jvals.shape}, got {side_jvals.shape}."
                )
            refinement[side_key] = self._apply_demo_height_to_joint_values(
                side_jvals, jnames
            )
        return refinement

    def _as_payload(self, obj):
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, np.ndarray) and obj.shape == ():
            obj = obj.item()
        if isinstance(obj, dict):
            return obj
        raise ValueError(
            f"Expected exported payload dict in {self.exported_npy_path}, got {type(obj)}."
        )

    def _payload_bool(self, payload: dict, key: str, default: bool) -> bool:
        value = payload.get(key, default)
        if isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
            elif value.size == 1:
                value = value.reshape(-1)[0]
        return bool(value)

    def _derive_obj_name(self, urdf_path: str) -> str:
        stem = Path(urdf_path).stem
        if stem.lower() in ("original", "target", "simple_object"):
            return Path(urdf_path).parent.name
        return stem

    def _load(self):
        raw = np.load(self.exported_npy_path, allow_pickle=True)
        payload = self._as_payload(raw)

        if str(payload.get("demo_kind", "lift")).lower() != "lift":
            raise ValueError(
                f"Expected a lift payload, got demo_kind={payload.get('demo_kind')!r}."
            )

        self.side = str(payload.get("side", "right")).lower()
        self.hand_side_available = {
            "left": self._payload_bool(payload, "is_active_left", self.side == "left"),
            "right": self._payload_bool(
                payload, "is_active_right", self.side == "right"
            ),
        }
        self.robot_model = str(payload.get("robot_model", "xhand"))
        self.robot_base_pose = (
            np.asarray(payload["traj_robot_base_pose"], dtype=np.float32)
            if "traj_robot_base_pose" in payload
            else None
        )
        self.object_raised_by = (
            float(
                np.asarray(payload["traj_object_raised_by"], dtype=np.float32).reshape(
                    -1
                )[0]
            )
            if "traj_object_raised_by" in payload
            # else 0.1
            else 0.0
        )
        raw_obj_urdf_path = str(Path(payload["original_urdf_path"]).resolve())
        raw_target_obj_urdf_path = str(Path(payload["target_urdf_path"]).resolve())

        self._obj_name = str(
            payload.get("original_obj_name", self._derive_obj_name(raw_obj_urdf_path))
        )
        self.target_object_name = str(
            payload.get(
                "target_obj_name", self._derive_obj_name(raw_target_obj_urdf_path)
            )
        )
        self._obj_urdf_path = str(process_urdf(raw_obj_urdf_path))
        self._target_obj_urdf_path = str(process_urdf(raw_target_obj_urdf_path))

        left_joints_raw = self._slice_time(
            np.asarray(payload["left_mano_joints"], dtype=np.float32)
        )
        right_joints_raw = self._slice_time(
            np.asarray(payload["right_mano_joints"], dtype=np.float32)
        )
        left_verts_raw = self._slice_time(
            np.asarray(payload["left_mano_verts"], dtype=np.float32)
        )
        right_verts_raw = self._slice_time(
            np.asarray(payload["right_mano_verts"], dtype=np.float32)
        )

        self._n_frames = int(left_joints_raw.shape[0])

        reference_joint_names = np_array_to_list(
            np.asarray(payload["reference_joint_names"], dtype=object)
        )
        reference_obj_pos_raw = self._slice_time(
            np.asarray(payload["reference_obj_pos"], dtype=np.float32)
        )
        reference_obj_quat = self._slice_time(
            np.asarray(payload["reference_obj_quat"], dtype=np.float32)
        )
        reference_obj_rot = self._slice_time(
            np.asarray(payload["reference_obj_rot"], dtype=np.float32)
        )
        reference_obj_arti = self._slice_time(
            np.asarray(payload["reference_obj_arti"], dtype=np.float32)
        )

        self._left_jnames = normalize_retargeted_joint_names(
            np_array_to_list(np.asarray(payload["left_jnames"], dtype=object))
        )
        self._right_jnames = normalize_retargeted_joint_names(
            np_array_to_list(np.asarray(payload["right_jnames"], dtype=object))
        )
        self._left_jvals = self._slice_time(
            np.asarray(payload["left_jvals"], dtype=np.float32)
        )
        self._right_jvals = self._slice_time(
            np.asarray(payload["right_jvals"], dtype=np.float32)
        )
        self.n_reach_frames = int(
            np.asarray(payload["n_reach_frames"], dtype=np.int64).reshape(-1)[0]
        )
        self.n_lift_frames = int(
            np.asarray(payload["n_lift_frames"], dtype=np.int64).reshape(-1)[0]
        )
        retargeted_obj_pos_raw = self._slice_time(
            np.asarray(payload["retargeted_obj_pos"], dtype=np.float32)
        )
        self._retargeted_obj_quat = self._slice_time(
            np.asarray(payload["retargeted_obj_quat"], dtype=np.float32)
        )
        self._retargeted_obj_rot = self._slice_time(
            np.asarray(payload["retargeted_obj_rot"], dtype=np.float32)
        )
        self._retargeted_obj_arti = self._slice_time(
            np.asarray(payload["retargeted_obj_arti"], dtype=np.float32)
        )
        self._retargeted_joint_names = np_array_to_list(
            np.asarray(payload["retargeted_joint_names"], dtype=object)
        )
        self.demo_ht = self.compute_demo_base_ht(
            obj_pos=retargeted_obj_pos_raw,
            obj_quat=self._retargeted_obj_quat,
            obj_arti=self._retargeted_obj_arti,
            joint_names=self._retargeted_joint_names,
        )
        if self.robot_base_pose is not None:
            self.robot_base_pose = np.asarray(
                self.robot_base_pose,
                dtype=np.float32,
                copy=True,
            )
            self.robot_base_pose[2, 3] += float(self.demo_ht)
        left_joints = self._apply_demo_height(left_joints_raw)
        right_joints = self._apply_demo_height(right_joints_raw)
        left_verts = self._apply_demo_height(left_verts_raw)
        right_verts = self._apply_demo_height(right_verts_raw)
        reference_obj_pos = self._apply_demo_height(reference_obj_pos_raw)
        self._retargeted_obj_pos = self._apply_demo_height(retargeted_obj_pos_raw)
        self.left_hand_verts = left_verts
        self.right_hand_verts = right_verts
        self.left_inner_hand_verts = left_verts
        self.right_inner_hand_verts = right_verts
        self.relevant_left_joint_pos = {
            k: left_joints[:, j_idx] for k, j_idx in self.hand_joint_map.items()
        }
        self.relevant_right_joint_pos = {
            k: right_joints[:, j_idx] for k, j_idx in self.hand_joint_map.items()
        }
        self.world_coord = {
            "joints.left": left_joints,
            "joints.right": right_joints,
            "verts.left": left_verts,
            "verts.right": right_verts,
        }
        self.object_reference_motion = {
            "obj_pos": reference_obj_pos,
            "obj_quat": reference_obj_quat,
            "obj_rot": reference_obj_rot,
            "obj_arti": reference_obj_arti,
            "joint_names": reference_joint_names,
        }

        if (
            "L_forearm_tz_link_joint" in self._left_jnames
            and self._left_jvals.shape[1] > 0
        ):
            left_tz_joint_idx = self._left_jnames.index("L_forearm_tz_link_joint")
            self._left_jvals[..., left_tz_joint_idx] += self.demo_ht
        if (
            "R_forearm_tz_link_joint" in self._right_jnames
            and self._right_jvals.shape[1] > 0
        ):
            right_tz_joint_idx = self._right_jnames.index("R_forearm_tz_link_joint")
            self._right_jvals[..., right_tz_joint_idx] += self.demo_ht
        self._drake_collision_refinement = self._load_drake_collision_refinement(
            payload
        )

        self._data = {
            "world_coord": self.world_coord,
            "params": {
                "obj_trans": reference_obj_pos,
                "obj_rot": reference_obj_rot,
                "obj_arti": reference_obj_arti,
            },
        }

    def _build_retargeted_motion(self):
        self.retargeted_motion = {
            "obj_pos": self._retargeted_obj_pos.copy(),
            "obj_quat": self._retargeted_obj_quat.copy(),
            "obj_rot": self._retargeted_obj_rot.copy(),
            "obj_arti": self._retargeted_obj_arti.copy(),
            "joint_names": list(self._retargeted_joint_names),
            "left_jnames": list(self._left_jnames),
            "left_jvals": self._left_jvals.copy(),
            "right_jnames": list(self._right_jnames),
            "right_jvals": self._right_jvals.copy(),
            "robot_model": self.robot_model,
            "robot_base_pose": (
                None if self.robot_base_pose is None else self.robot_base_pose.copy()
            ),
            "object_raised_by": float(self.object_raised_by),
            "n_reach_frames": self.n_reach_frames,
            "n_lift_frames": self.n_lift_frames,
        }
        if self._drake_collision_refinement:
            self.retargeted_motion.update(
                {
                    key: (
                        value.copy()
                        if isinstance(value, np.ndarray)
                        else list(value) if isinstance(value, list) else value
                    )
                    for key, value in self._drake_collision_refinement.items()
                }
            )

    def get_joint_names(self, urdf_path: str | Path):
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
        joint_names = []
        for joint in root.findall("joint"):
            if joint.get("type", "fixed") == "fixed":
                continue
            name = joint.get("name")
            if name:
                joint_names.append(name)
        return joint_names

    def process_retargeted_motion(
        self, left_hand: Hand | None = None, right_hand: Hand | None = None
    ):
        if self.retargeted_motion is None:
            return None

        motion = self.retargeted_motion

        def _process_side(side: str, hand: Hand):
            retgt_jnames = motion[f"{side}_jnames"]
            retgt_jvals = motion[f"{side}_jvals"]
            if len(retgt_jnames) == 0 or retgt_jvals.shape[1] == 0:
                return None

            missing_joints = [jn for jn in hand.joint_names if jn not in retgt_jnames]
            if missing_joints:
                raise ValueError(
                    f"Retargeted motion missing {side} hand joints required by URDF FK: {missing_joints}"
                )

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
                link_name: self._apply_demo_height(
                    link_fk[link_name][:, :3, 3].astype(np.float32, copy=False)
                )
                for link_name in link_names
            }
            motion[f"{side}_link_names"] = list(link_names)
            motion[f"{side}_link_pos"] = link_pos
            return link_pos

        results = {}
        if left_hand is not None and self._left_jvals.shape[1] > 0:
            left_link_pos = _process_side("left", left_hand)
            if left_link_pos is not None:
                results["left_link_pos"] = left_link_pos
        if right_hand is not None and self._right_jvals.shape[1] > 0:
            right_link_pos = _process_side("right", right_hand)
            if right_link_pos is not None:
                results["right_link_pos"] = right_link_pos

        if self.side == "left" and "left_link_pos" in results:
            motion["link_names"] = motion["left_link_names"]
            motion["link_pos"] = motion["left_link_pos"]
        elif self.side == "right" and "right_link_pos" in results:
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

    def get_human_hand_kpt_face_indices(self, side: str) -> dict[str, int]:
        if side not in ("left", "right"):
            raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")
        if not self.hand_side_available[side]:
            raise ValueError(f"OakinkProcessorLift has no active {side} hand data.")
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
        return self._obj_urdf_path == self._target_obj_urdf_path

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

    @property
    def data(self):
        return self._data
