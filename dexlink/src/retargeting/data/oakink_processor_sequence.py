from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
from urdfpy import URDF

from dexlink.src.retargeting.data.human_hand_kpts import (
    human_hand_kpt_positions_from_face_vertices,
    load_precomputed_human_hand_kpt_face_vertex_indices,
)
from dexlink.src.retargeting.data.oakink_processor_grasp import (
    normalize_retargeted_joint_names,
    np_array_to_list,
)
from dexlink.utils.retargeting.math_utils import get_X, quat_from_matrix
from dexlink.utils.retargeting.path_utils import process_urdf

if TYPE_CHECKING:
    from dexlink.src.rl.sim.robot import Hand


class OakinkProcessorSequence:
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
        self.requires_human_reference_joint_pos_for_eval = True
        if self.skip <= 0:
            raise ValueError("skip must be positive.")

        self._human_hand_kpt_face_vertex_indices = {}
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

    @staticmethod
    def _forearm_tz_joint_index(joint_names: list[str], *, side: str) -> int:
        matches = OakinkProcessorSequence._forearm_tz_joint_indices(joint_names)
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one forearm_tz_link_joint for {side} hand "
                f"reference joints, got {matches} from {joint_names}."
            )
        return int(matches[0])

    @staticmethod
    def _forearm_tz_joint_indices(joint_names: list[str]) -> list[int]:
        return [
            idx
            for idx, joint_name in enumerate(joint_names)
            if str(joint_name).endswith("forearm_tz_link_joint")
        ]

    def _apply_demo_height_to_hand_jvals(
        self,
        jvals: np.ndarray,
        jnames: list[str],
        *,
        side: str,
    ) -> None:
        if jvals.shape[1] == 0:
            return
        matches = [int(idx) for idx in self._forearm_tz_joint_indices(jnames)]
        if len(matches) == 1:
            jvals[..., matches[0]] += self.demo_ht
            return
        if self.robot_base_pose is not None and len(matches) == 0:
            return
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one forearm_tz_link_joint for {side} hand "
                f"reference joints, got {matches} from {jnames}."
            )

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
        support_top = 0.0
        if self._scene_objects is not None:
            block_pos = np.asarray(
                self._scene_objects.get(
                    "scene_support_block_pos",
                    np.zeros((0, 3), dtype=np.float32),
                ),
                dtype=np.float32,
            )
            block_size = np.asarray(
                self._scene_objects.get(
                    "scene_support_block_size",
                    np.zeros((0, 3), dtype=np.float32),
                ),
                dtype=np.float32,
            )
            if block_pos.shape[0] != block_size.shape[0]:
                raise ValueError(
                    "Expected scene_support_block_pos and scene_support_block_size "
                    f"to have matching block counts, got {block_pos.shape} and "
                    f"{block_size.shape}."
                )
            if block_pos.shape[0] > 0:
                support_top = float(np.max(block_pos[:, 2] + 0.5 * block_size[:, 2]))
        return self.support_height + support_top + self.support_margin - lowest_z

    def _apply_demo_height(self, arr: np.ndarray) -> np.ndarray:
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
        if "drake_collision_refinement_valid_frame_mask" in payload:
            valid_mask = self._slice_time(
                np.asarray(
                    payload["drake_collision_refinement_valid_frame_mask"],
                    dtype=bool,
                )
            )
            if valid_mask.shape != mask.shape:
                raise ValueError(
                    "Expected drake_collision_refinement_valid_frame_mask shape "
                    f"{mask.shape}, got {valid_mask.shape}."
                )
        else:
            valid_mask = ~mask
        if "drake_collision_refinement_invalid_frame_mask" in payload:
            invalid_mask = self._slice_time(
                np.asarray(
                    payload["drake_collision_refinement_invalid_frame_mask"],
                    dtype=bool,
                )
            )
            if invalid_mask.shape != mask.shape:
                raise ValueError(
                    "Expected drake_collision_refinement_invalid_frame_mask shape "
                    f"{mask.shape}, got {invalid_mask.shape}."
                )
        else:
            invalid_mask = ~valid_mask
        if np.any(valid_mask & invalid_mask):
            raise ValueError(
                "Drake collision refinement valid and invalid frame masks overlap."
            )
        if not np.all(valid_mask | invalid_mask):
            raise ValueError(
                "Drake collision refinement valid and invalid frame masks do not "
                "cover every frame."
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
            "drake_collision_refinement_valid_frame_mask": valid_mask,
            "drake_collision_refinement_invalid_frame_mask": invalid_mask,
            "drake_collision_refinement_valid_frame_indices": np.flatnonzero(
                valid_mask
            ).astype(np.int64),
            "drake_collision_refinement_invalid_frame_indices": np.flatnonzero(
                invalid_mask
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

    def _payload_float_range(
        self,
        payload: dict,
        key: str,
    ) -> tuple[float, float]:
        if key not in payload:
            raise ValueError(f"Expected exported sequence payload to define {key!r}.")
        return self._parse_float_range(payload[key], key)

    def _payload_optional_float_range(
        self,
        payload: dict,
        key: str,
    ) -> tuple[float, float] | None:
        if key not in payload:
            return None
        return self._parse_float_range(payload[key], key)

    def _parse_float_range(self, value: object, key: str) -> tuple[float, float]:
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        values = np.asarray(value, dtype=np.float32).reshape(-1)
        if values.shape != (2,):
            raise ValueError(
                f"Expected {key!r} to have shape (2,), got {values.shape}."
            )
        min_value = float(values[0])
        max_value = float(values[1])
        if min_value <= 0.0 or max_value <= 0.0 or min_value > max_value:
            raise ValueError(
                f"Expected {key!r} to be positive and ordered, got {values}."
            )
        return (min_value, max_value)

    def _payload_optional_positive_float(
        self,
        payload: dict,
        key: str,
    ) -> float | None:
        if key not in payload:
            return None
        value = np.asarray(payload[key], dtype=np.float32).reshape(-1)
        if value.shape != (1,):
            raise ValueError(f"Expected {key!r} to have one value, got {value.shape}.")
        scalar = float(value[0])
        if scalar <= 0.0:
            raise ValueError(f"Expected {key!r} to be positive, got {scalar}.")
        return scalar

    def _payload_optional_path(self, payload: dict, key: str) -> str | None:
        value = payload.get(key)
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        if value in (None, "", "None"):
            return None
        return str(Path(value).resolve())

    def _load_human_reference_joint_pos(
        self,
        payload: dict,
        side: str,
    ) -> dict[str, np.ndarray] | None:
        if side not in ("left", "right"):
            raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")
        key = f"{side}_human_reference_keypoint_pos"
        if key not in payload:
            return None
        names = np_array_to_list(
            np.asarray(payload["reference_keypoint_names"], dtype=object)
        )
        points = self._apply_demo_height(
            self._slice_time(np.asarray(payload[key], dtype=np.float32))
        )
        expected_shape = (self._n_frames, len(names), 3)
        if points.shape != expected_shape:
            raise ValueError(
                f"Expected {key!r} to have shape {expected_shape}, got {points.shape}."
            )
        return {name: points[:, idx] for idx, name in enumerate(names)}

    def _load_human_reference_joint_valid(
        self,
        payload: dict,
        side: str,
    ) -> np.ndarray | None:
        if side not in ("left", "right"):
            raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")
        key = f"{side}_human_reference_keypoint_valid"
        if key not in payload:
            return None
        valid = self._slice_time(np.asarray(payload[key], dtype=bool)).reshape(-1)
        if valid.shape != (self._n_frames,):
            raise ValueError(
                f"Expected {key!r} shape {(self._n_frames,)}, got {valid.shape}."
            )
        return valid

    def _apply_explicit_human_reference_to_joints(
        self,
        joints: np.ndarray,
        human_reference_joint_pos: dict[str, np.ndarray] | None,
    ) -> np.ndarray:
        if human_reference_joint_pos is None:
            return joints
        out = np.asarray(joints, dtype=np.float32).copy()
        missing_names = [
            name
            for name in self.hand_joint_map
            if name not in human_reference_joint_pos
        ]
        if missing_names:
            raise ValueError(
                "Explicit human reference keypoints are missing MANO skeleton names "
                f"required by RL non-retargeted references: {missing_names}."
            )
        for name, joint_idx in self.hand_joint_map.items():
            values = np.asarray(human_reference_joint_pos[name], dtype=np.float32)
            expected_shape = (self._n_frames, 3)
            if values.shape != expected_shape:
                raise ValueError(
                    f"Expected human reference keypoint {name!r} shape "
                    f"{expected_shape}, got {values.shape}."
                )
            out[:, joint_idx, :] = values
        return out

    def _load_scene_objects(self, payload: dict) -> dict | None:
        if "scene_object_pos" not in payload:
            return None

        required_keys = (
            "scene_object_ids",
            "scene_object_names",
            "scene_object_urdf_paths",
            "scene_object_pos",
            "scene_object_quat",
            "scene_object_rot",
            "scene_object_arti_vals",
            "scene_object_arti_valid_mask",
            "scene_object_arti_names",
            "scene_object_active_mask",
            "scene_object_static",
        )
        missing = [key for key in required_keys if key not in payload]
        if missing:
            raise ValueError(
                "Exported sequence payload contains scene_object_pos, so expected "
                f"full multi-object keys {required_keys}; missing {missing}."
            )

        scene_pos = self._slice_time(
            np.asarray(payload["scene_object_pos"], dtype=np.float32)
        )
        scene_quat = self._slice_time(
            np.asarray(payload["scene_object_quat"], dtype=np.float32)
        )
        scene_rot = self._slice_time(
            np.asarray(payload["scene_object_rot"], dtype=np.float32)
        )
        scene_arti_vals = self._slice_time(
            np.asarray(payload["scene_object_arti_vals"], dtype=np.float32)
        )
        scene_active_mask = self._slice_time(
            np.asarray(payload["scene_object_active_mask"], dtype=bool)
        )
        scene_static = np.asarray(payload["scene_object_static"], dtype=bool)
        scene_friction = np.asarray(
            payload.get(
                "scene_object_friction",
                np.ones(scene_pos.shape[1], dtype=np.float32),
            ),
            dtype=np.float32,
        )
        scene_arti_valid_mask = np.asarray(
            payload["scene_object_arti_valid_mask"], dtype=bool
        )
        scene_ids = np_array_to_list(
            np.asarray(payload["scene_object_ids"], dtype=object)
        )
        scene_names = np_array_to_list(
            np.asarray(payload["scene_object_names"], dtype=object)
        )
        scene_urdf_paths = np_array_to_list(
            np.asarray(payload["scene_object_urdf_paths"], dtype=object)
        )
        scene_arti_names = np.asarray(payload["scene_object_arti_names"], dtype=object)

        if scene_pos.ndim != 3 or scene_pos.shape[2] != 3:
            raise ValueError(
                "Expected scene_object_pos to have shape (T, N, 3), got "
                f"{scene_pos.shape}."
            )
        if scene_pos.shape[0] != self._n_frames:
            raise ValueError(
                "Expected sliced scene_object_pos frame count to match sequence "
                f"frame count {self._n_frames}, got {scene_pos.shape[0]}."
            )
        if scene_quat.shape != (scene_pos.shape[0], scene_pos.shape[1], 4):
            raise ValueError(
                "Expected scene_object_quat shape (T, N, 4) matching scene_object_pos, "
                f"got {scene_quat.shape}."
            )
        if scene_rot.shape != (scene_pos.shape[0], scene_pos.shape[1], 3):
            raise ValueError(
                "Expected scene_object_rot shape (T, N, 3) matching scene_object_pos, "
                f"got {scene_rot.shape}."
            )
        if (
            scene_arti_vals.ndim != 3
            or scene_arti_vals.shape[:2] != scene_pos.shape[:2]
        ):
            raise ValueError(
                "Expected scene_object_arti_vals shape (T, N, Amax) matching "
                f"scene_object_pos, got {scene_arti_vals.shape} and {scene_pos.shape}."
            )
        if scene_arti_valid_mask.shape != scene_arti_vals.shape[1:]:
            raise ValueError(
                "Expected scene_object_arti_valid_mask shape (N, Amax), got "
                f"{scene_arti_valid_mask.shape} for {scene_arti_vals.shape}."
            )
        if scene_active_mask.shape != scene_pos.shape[:2]:
            raise ValueError(
                "Expected scene_object_active_mask shape (T, N), got "
                f"{scene_active_mask.shape} for {scene_pos.shape}."
            )
        n_objects = int(scene_pos.shape[1])
        if scene_static.shape != (n_objects,):
            raise ValueError(
                f"Expected scene_object_static shape {(n_objects,)}, got "
                f"{scene_static.shape}."
            )
        if scene_friction.shape != (n_objects,) or np.any(scene_friction <= 0.0):
            raise ValueError(
                f"Expected positive scene_object_friction shape {(n_objects,)}, "
                f"got {scene_friction}."
            )
        for key, value in (
            ("scene_object_ids", scene_ids),
            ("scene_object_names", scene_names),
            ("scene_object_urdf_paths", scene_urdf_paths),
            ("scene_object_arti_names", scene_arti_names),
        ):
            if len(value) != n_objects:
                raise ValueError(
                    f"Expected {key} length to match scene object count {n_objects}, "
                    f"got {len(value)}."
                )

        scene_payload = {
            "scene_object_ids": list(scene_ids),
            "scene_object_names": list(scene_names),
            "scene_object_urdf_paths": [str(path) for path in scene_urdf_paths],
            "scene_object_pos": scene_pos,
            "scene_object_quat": scene_quat,
            "scene_object_rot": scene_rot,
            "scene_object_arti_vals": scene_arti_vals,
            "scene_object_arti_valid_mask": scene_arti_valid_mask,
            "scene_object_arti_names": scene_arti_names,
            "scene_object_active_mask": scene_active_mask,
            "scene_object_static": scene_static,
            "scene_object_friction": scene_friction,
        }
        for key in (
            "scene_active_object_index",
            "scene_canonical_object_id",
            "scene_table_height",
            "scene_table_margin",
            "scene_table_size",
            "scene_uses_block",
            "scene_object_support_z_offsets",
            "scene_support_block_ids",
            "scene_support_block_pos",
            "scene_support_block_size",
        ):
            if key in payload:
                scene_payload[key] = np.asarray(payload[key]).copy()
        return scene_payload

    def _derive_obj_name(self, urdf_path: str) -> str:
        stem = Path(urdf_path).stem
        if stem.lower() in ("original", "target", "simple_object"):
            return Path(urdf_path).parent.name
        return stem

    def _load(self):
        raw = np.load(self.exported_npy_path, allow_pickle=True)
        payload = self._as_payload(raw)

        if str(payload.get("demo_kind", "sequence")).lower() != "sequence":
            raise ValueError(
                f"Expected a sequence payload, got demo_kind={payload.get('demo_kind')!r}."
            )

        self.side = str(payload.get("side", "right")).lower()
        self.hand_side_available = {
            "left": self._payload_bool(payload, "is_active_left", self.side == "left"),
            "right": self._payload_bool(
                payload, "is_active_right", self.side == "right"
            ),
        }
        self.hand_config_source_by_side = self._load_hand_config_source_by_side(payload)
        self.robot_model = str(payload.get("robot_model", "xhand"))
        self.robot_base_pose = (
            np.asarray(payload["traj_robot_base_pose"], dtype=np.float32)
            if "traj_robot_base_pose" in payload
            else None
        )
        raw_obj_urdf_path = str(Path(payload["original_urdf_path"]).resolve())
        raw_target_obj_urdf_path = str(Path(payload["target_urdf_path"]).resolve())
        raw_target_articulated_urdf_path = self._payload_optional_path(
            payload, "target_articulated_urdf_path"
        )
        self._obj_name = str(
            payload.get("original_obj_name", self._derive_obj_name(raw_obj_urdf_path))
        )
        self.target_object_name = str(
            payload.get(
                "target_obj_name", self._derive_obj_name(raw_target_obj_urdf_path)
            )
        )
        self._obj_urdf_path = str(process_urdf(raw_obj_urdf_path, append_jobname=True))
        self._target_visual_obj_urdf_path = str(
            process_urdf(raw_target_obj_urdf_path, append_jobname=True)
        )
        self._target_obj_urdf_path = str(
            process_urdf(
                (
                    raw_target_articulated_urdf_path
                    if raw_target_articulated_urdf_path is not None
                    else raw_target_obj_urdf_path
                ),
                append_jobname=True,
            )
        )
        self.target_articulation_link_names = np_array_to_list(
            np.asarray(payload.get("target_articulation_link_names", []), dtype=object)
        )
        self.object_static = bool(
            np.asarray(payload.get("object_static", False), dtype=bool).reshape(-1)[0]
        )
        self.object_density_range = self._payload_optional_float_range(
            payload, "object_density_range"
        )
        self.target_object_volume = self._payload_optional_positive_float(
            payload,
            "target_object_volume",
        )
        self.object_mass_range = self._payload_optional_float_range(
            payload,
            "object_mass_range",
        )
        if (
            self.object_mass_range is None
            and self.object_density_range is not None
            and self.target_object_volume is not None
        ):
            self.object_mass_range = (
                self.object_density_range[0] * self.target_object_volume,
                self.object_density_range[1] * self.target_object_volume,
            )

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
        self._scene_objects = self._load_scene_objects(payload)

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
        self.phase_names = np_array_to_list(
            np.asarray(payload.get("phase_names", []), dtype=object)
        )
        self.phase_lengths = np.asarray(
            payload.get("phase_lengths", np.asarray([], dtype=np.int64)),
            dtype=np.int64,
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
        self.raw_relevant_left_joint_pos = {
            k: left_joints[:, j_idx] for k, j_idx in self.hand_joint_map.items()
        }
        self.raw_relevant_right_joint_pos = {
            k: right_joints[:, j_idx] for k, j_idx in self.hand_joint_map.items()
        }
        reference_obj_pos = self._apply_demo_height(reference_obj_pos_raw)
        self._retargeted_obj_pos = self._apply_demo_height(retargeted_obj_pos_raw)
        if self._scene_objects is not None:
            self._scene_objects["scene_object_pos"] = self._apply_demo_height(
                self._scene_objects["scene_object_pos"]
            )
        self.human_reference_left_joint_pos = self._load_human_reference_joint_pos(
            payload=payload,
            side="left",
        )
        self.human_reference_right_joint_pos = self._load_human_reference_joint_pos(
            payload=payload,
            side="right",
        )
        self.human_reference_left_joint_valid = self._load_human_reference_joint_valid(
            payload=payload,
            side="left",
        )
        self.human_reference_right_joint_valid = self._load_human_reference_joint_valid(
            payload=payload,
            side="right",
        )
        left_human_joints = self._apply_explicit_human_reference_to_joints(
            left_joints,
            self.human_reference_left_joint_pos,
        )
        right_human_joints = self._apply_explicit_human_reference_to_joints(
            right_joints,
            self.human_reference_right_joint_pos,
        )
        self.left_hand_verts = left_verts
        self.right_hand_verts = right_verts
        self.left_inner_hand_verts = left_verts
        self.right_inner_hand_verts = right_verts
        self.relevant_left_joint_pos = {
            k: left_human_joints[:, j_idx] for k, j_idx in self.hand_joint_map.items()
        }
        self.relevant_right_joint_pos = {
            k: right_human_joints[:, j_idx] for k, j_idx in self.hand_joint_map.items()
        }
        self.world_coord = {
            "joints.left": left_human_joints,
            "joints.right": right_human_joints,
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

        self._apply_demo_height_to_hand_jvals(
            self._left_jvals,
            self._left_jnames,
            side="left",
        )
        self._apply_demo_height_to_hand_jvals(
            self._right_jvals,
            self._right_jnames,
            side="right",
        )
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
            "phase_names": list(self.phase_names),
            "phase_lengths": self.phase_lengths.copy(),
            "target_articulation_link_names": list(self.target_articulation_link_names),
        }
        self.retargeted_motion["object_static"] = bool(self.object_static)
        if self.object_density_range is not None:
            self.retargeted_motion["object_density_range"] = tuple(
                self.object_density_range
            )
        if self.object_mass_range is not None:
            self.retargeted_motion["object_mass_range"] = tuple(self.object_mass_range)
        if self.target_object_volume is not None:
            self.retargeted_motion["target_object_volume"] = self.target_object_volume
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
        if self.hand_config_source_by_side:
            self.retargeted_motion["hand_config_source_by_side"] = dict(
                self.hand_config_source_by_side
            )
        if self._scene_objects is not None:
            self.retargeted_motion.update(
                {
                    "scene_object_ids": list(self._scene_objects["scene_object_ids"]),
                    "scene_object_names": list(
                        self._scene_objects["scene_object_names"]
                    ),
                    "scene_object_urdf_paths": list(
                        self._scene_objects["scene_object_urdf_paths"]
                    ),
                    "scene_object_pos": self._scene_objects["scene_object_pos"].copy(),
                    "scene_object_quat": self._scene_objects[
                        "scene_object_quat"
                    ].copy(),
                    "scene_object_rot": self._scene_objects["scene_object_rot"].copy(),
                    "scene_object_arti_vals": self._scene_objects[
                        "scene_object_arti_vals"
                    ].copy(),
                    "scene_object_arti_valid_mask": self._scene_objects[
                        "scene_object_arti_valid_mask"
                    ].copy(),
                    "scene_object_arti_names": self._scene_objects[
                        "scene_object_arti_names"
                    ].copy(),
                    "scene_object_active_mask": self._scene_objects[
                        "scene_object_active_mask"
                    ].copy(),
                    "scene_object_static": self._scene_objects[
                        "scene_object_static"
                    ].copy(),
                    "scene_object_friction": self._scene_objects[
                        "scene_object_friction"
                    ].copy(),
                }
            )
            for key in (
                "scene_active_object_index",
                "scene_canonical_object_id",
                "scene_table_height",
                "scene_table_margin",
                "scene_table_size",
                "scene_uses_block",
                "scene_object_support_z_offsets",
                "scene_support_block_ids",
                "scene_support_block_pos",
                "scene_support_block_size",
            ):
                if key in self._scene_objects:
                    self.retargeted_motion[key] = np.asarray(
                        self._scene_objects[key]
                    ).copy()

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

        def _is_missing_only_panda_joints(
            missing_joints: list[str], hand: Hand
        ) -> bool:
            panda_joints = set(getattr(hand, "panda_joints", ()))
            if len(panda_joints) == 0:
                return False
            return set(missing_joints).issubset(panda_joints)

        def _xhand_wrist_pose_from_forearm_motion(
            *,
            side: str,
            retgt_jnames: list[str],
            retgt_jvals: np.ndarray,
            target_hand: Hand,
        ) -> tuple[np.ndarray, np.ndarray]:
            from dexlink.src.rl.world.robot import get_hand_specific_cfg

            xhand_cfg = get_hand_specific_cfg("xhand")[side]
            xhand_urdf = URDF.load(str(process_urdf(xhand_cfg["urdf_path"])))
            xhand_joints = [
                joint for joint in xhand_urdf.joints if joint.joint_type != "fixed"
            ]
            xhand_joint_names = [joint.name for joint in xhand_joints]
            missing_xhand_joints = [
                joint_name
                for joint_name in xhand_joint_names
                if joint_name not in retgt_jnames
            ]
            if missing_xhand_joints:
                raise ValueError(
                    "Cannot synthesize Franka Panda joints from 6-DoF forearm "
                    f"retargeting because xhand FK is missing joints for {side}: "
                    f"{missing_xhand_joints}."
                )
            xhand_joint_indices = [
                retgt_jnames.index(name) for name in xhand_joint_names
            ]
            xhand_jvals = retgt_jvals[:, xhand_joint_indices]
            xhand_joint_dict = {
                joint.name: xhand_jvals[:, idx]
                for idx, joint in enumerate(xhand_joints)
            }
            xhand_fk = xhand_urdf.link_fk_batch(cfgs=xhand_joint_dict)
            xhand_fk = {link.name: pose for link, pose in xhand_fk.items()}
            wrist_link_name = str(xhand_cfg["wrist_link_name"])
            if wrist_link_name not in xhand_fk:
                raise ValueError(
                    f"xhand FK did not produce wrist link {wrist_link_name!r} "
                    f"for {side}."
                )
            base_pose = np.asarray(target_hand.base_pose, dtype=np.float32)
            if base_pose.shape != (4, 4):
                raise ValueError(
                    f"Expected target hand base_pose shape (4, 4), got {base_pose.shape}."
                )
            local_wrist_pose = np.asarray(xhand_fk[wrist_link_name], dtype=np.float32)
            world_wrist_pose = np.einsum("ij,tjk->tik", base_pose, local_wrist_pose)
            wrist_pos = world_wrist_pose[:, :3, 3].astype(np.float32, copy=False)
            wrist_quat = (
                quat_from_matrix(
                    torch.as_tensor(world_wrist_pose[:, :3, :3], dtype=torch.float32)
                )
                .cpu()
                .numpy()
            )
            return wrist_pos, wrist_quat.astype(np.float32, copy=False)

        def _synthesize_franka_jvals_from_xhand_forearm(
            *,
            side: str,
            hand: Hand,
            retgt_jnames: list[str],
            retgt_jvals: np.ndarray,
        ) -> np.ndarray:
            finger_missing = [
                joint_name
                for joint_name in hand.finger_joint_names
                if joint_name not in retgt_jnames
            ]
            if finger_missing:
                raise ValueError(
                    "Cannot synthesize Franka Panda joints because retargeted "
                    f"motion is missing {side} finger joints: {finger_missing}."
                )
            wrist_pos, wrist_quat = _xhand_wrist_pose_from_forearm_motion(
                side=side,
                retgt_jnames=retgt_jnames,
                retgt_jvals=retgt_jvals,
                target_hand=hand,
            )
            jvals = np.zeros(
                (retgt_jvals.shape[0], len(hand.joint_names)), dtype=np.float32
            )
            for joint_col, joint_name in enumerate(hand.joint_names):
                if joint_name in retgt_jnames:
                    jvals[:, joint_col] = retgt_jvals[:, retgt_jnames.index(joint_name)]

            device = torch.device(hand.env.device)
            finger_indices = [
                hand.joint_names.index(name) for name in hand.finger_joint_names
            ]
            arm_ik_dof_indices = np.asarray(
                [hand.dof_map[idx] for idx in hand.wrist_control_joint_idx],
                dtype=np.int64,
            )
            if arm_ik_dof_indices.shape[0] == 0:
                raise ValueError(
                    f"Expected franka_xhand {side} hand to expose Panda IK joints."
                )
            q_init = torch.as_tensor(jvals[0:1], dtype=torch.float32, device=device)
            num_envs = int(hand.env.num_envs)
            for timestep in range(jvals.shape[0]):
                q_init[:, finger_indices] = torch.as_tensor(
                    jvals[timestep : timestep + 1, finger_indices],
                    dtype=torch.float32,
                    device=device,
                )
                target_pos = torch.as_tensor(
                    wrist_pos[timestep : timestep + 1],
                    dtype=torch.float32,
                    device=device,
                ).repeat(num_envs, 1)
                target_quat = torch.as_tensor(
                    wrist_quat[timestep : timestep + 1],
                    dtype=torch.float32,
                    device=device,
                ).repeat(num_envs, 1)
                ik_qpos = hand.physics.solve_entity_ik(
                    hand.robot,
                    link_name=hand.wrist_link_name,
                    target_pos=target_pos,
                    target_quat=target_quat,
                    dofs_idx_local=arm_ik_dof_indices.tolist(),
                    init_qpos=q_init.repeat(num_envs, 1),
                    init_qpos_dofs_idx_local=hand.dof_map,
                )
                jvals[timestep] = ik_qpos[0].detach().cpu().numpy()
                q_init = ik_qpos[0:1].detach().clone()
            return jvals

        def _process_side(side: str, hand: Hand):
            retgt_jnames = motion[f"{side}_jnames"]
            retgt_jvals = motion[f"{side}_jvals"]
            if len(retgt_jnames) == 0 or retgt_jvals.shape[1] == 0:
                return None

            missing_joints = [jn for jn in hand.joint_names if jn not in retgt_jnames]
            if missing_joints:
                if not _is_missing_only_panda_joints(missing_joints, hand):
                    raise ValueError(
                        "Retargeted motion missing "
                        f"{side} hand joints required by URDF FK: {missing_joints}"
                    )
                jvals_ordered = _synthesize_franka_jvals_from_xhand_forearm(
                    side=side,
                    hand=hand,
                    retgt_jnames=retgt_jnames,
                    retgt_jvals=retgt_jvals,
                )
                motion[f"{side}_jnames"] = list(hand.joint_names)
                motion[f"{side}_jvals"] = jvals_ordered.copy()
                if (
                    str(motion.get("drake_collision_refinement_hand", "")) == side
                    and "drake_collision_refinement_jnames" in motion
                ):
                    drake_jnames = [
                        str(name)
                        for name in motion["drake_collision_refinement_jnames"]
                    ]
                    drake_missing = [
                        joint_name
                        for joint_name in hand.joint_names
                        if joint_name not in drake_jnames
                    ]
                    if _is_missing_only_panda_joints(drake_missing, hand):
                        motion["drake_collision_refinement_jnames"] = list(
                            hand.joint_names
                        )
                        motion["drake_collision_refinement_jvals"] = (
                            jvals_ordered.copy()
                        )
                        motion[f"drake_collision_refinement_{side}_jvals"] = (
                            jvals_ordered.copy()
                        )
            else:
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

        if left_hand is not None and right_hand is None:
            motion["link_names"] = motion["left_link_names"]
            motion["link_pos"] = motion["left_link_pos"]
        elif right_hand is not None and left_hand is None:
            motion["link_names"] = motion["right_link_names"]
            motion["link_pos"] = motion["right_link_pos"]

        return results if results else None

    def _load_hand_config_source_by_side(self, payload: dict) -> dict[str, str]:
        raw = payload.get("hand_config_source_by_side", {})
        if isinstance(raw, np.ndarray):
            if raw.shape != ():
                raise ValueError(
                    "Expected hand_config_source_by_side payload to be a scalar "
                    f"object array, got shape {raw.shape}."
                )
            raw = raw.item()
        if raw in (None, "", "None"):
            return {}
        if not isinstance(raw, dict):
            raise ValueError(
                "Expected hand_config_source_by_side payload to be a dict, "
                f"got {type(raw).__name__}."
            )
        source_by_side: dict[str, str] = {}
        for role_side, source_side in raw.items():
            role_side = str(role_side)
            source_side = str(source_side)
            if role_side not in ("left", "right") or source_side not in (
                "left",
                "right",
            ):
                raise ValueError(
                    "Expected hand_config_source_by_side to map left/right role "
                    f"sides to left/right config sides, got {raw!r}."
                )
            source_by_side[role_side] = source_side
        return source_by_side

    def get_human_hand_kpt_face_vertex_indices(
        self,
        side: str,
    ) -> dict[str, np.ndarray]:
        if side not in ("left", "right"):
            raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")
        if not self.hand_side_available[side]:
            raise ValueError(f"OakinkProcessorSequence has no active {side} hand data.")
        if side not in self._human_hand_kpt_face_vertex_indices:
            self._human_hand_kpt_face_vertex_indices[side] = (
                load_precomputed_human_hand_kpt_face_vertex_indices(side)
            )
        return dict(self._human_hand_kpt_face_vertex_indices[side])

    def get_human_hand_kpt_positions(
        self,
        side: str,
        names: list[str] | None = None,
        frame_idx: int | None = None,
    ) -> dict[str, np.ndarray]:
        face_vertex_indices = self.get_human_hand_kpt_face_vertex_indices(side)
        return human_hand_kpt_positions_from_face_vertices(
            vertices=self.world_coord[f"verts.{side}"],
            face_vertex_indices=face_vertex_indices,
            names=names,
            frame_idx=frame_idx,
        )

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
    def target_visual_obj_urdf_path(self):
        return self._target_visual_obj_urdf_path

    @property
    def data(self):
        return self._data
