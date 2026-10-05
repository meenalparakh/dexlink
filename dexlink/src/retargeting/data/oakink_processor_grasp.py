from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from dexlink.src.retargeting.data.human_hand_kpts import (
    human_hand_kpt_positions_from_faces,
    load_precomputed_human_hand_kpt_face_indices,
)
from dexlink.utils.retargeting.path_utils import process_urdf

if TYPE_CHECKING:
    from dexlink.src.rl.sim.robot import Hand


def np_array_to_list(arr):
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    return [
        n.decode("utf-8") if isinstance(n, (bytes, np.bytes_)) else str(n) for n in arr
    ]


def normalize_retargeted_joint_name(name: str) -> str:
    name = str(name)
    for side_prefix in ("L_", "R_"):
        marker = f"_{side_prefix}"
        if marker in name:
            name = name[name.index(marker) + 1 :]
            break
    for suffix in ("_q", "_x", "_y", "_z"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def normalize_retargeted_joint_names(names):
    return [normalize_retargeted_joint_name(n) for n in names]


class OakinkProcessorGrasp:
    """
    ARCTIC-like sequence processor wrapper for exported OakInk pair data.
    Expected input file is the dict saved by `export_oakink_pair_data.py`.
    """

    def __init__(
        self,
        exported_npy_path: str | Path,
        n_frames: int | None = None,
        start_timestep: int = 0,
        end_timestep: int = -1,
        skip: int = 1,
    ):
        self.side = "right"
        self.hand_side_available = {"left": False, "right": True}
        self.exported_npy_path = str(exported_npy_path)
        self.start_timestep = int(start_timestep)
        self.end_timestep = int(end_timestep)
        self.skip = int(skip)
        if self.skip <= 0:
            raise ValueError("skip must be positive.")
        self._n_frames = int(
            n_frames
            if n_frames is not None
            else self._resolve_n_frames(
                start_timestep=self.start_timestep,
                end_timestep=self.end_timestep,
                skip=self.skip,
            )
        )
        self.demo_ht = 1.0

        if self._n_frames <= 0:
            raise ValueError("n_frames must be positive.")

        # 21-joint indexing used by downstream code.
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
        self.object_reference_motion = self.original_object_motion()
        self._build_retargeted_motion()

    @staticmethod
    def _resolve_n_frames(start_timestep: int, end_timestep: int, skip: int) -> int:
        if end_timestep == -1:
            return 30
        if end_timestep <= start_timestep:
            raise ValueError("end_timestep must be greater than start_timestep, or -1.")
        return len(range(start_timestep, end_timestep, skip))

    def _repeat_frames(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 0:
            arr = arr[None]
        if arr.shape[0] == self._n_frames:
            return arr
        if arr.shape[0] == 1:
            return np.repeat(arr, self._n_frames, axis=0)
        # fallback: tile first frame
        first = arr[:1]
        return np.repeat(first, self._n_frames, axis=0)

    def _apply_demo_height(self, arr: np.ndarray) -> np.ndarray:
        """Shift xyz positions upward by demo_ht along z."""
        out = np.asarray(arr).astype(np.float32, copy=True)
        out[..., 2] += float(self.demo_ht)
        return out

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

    def _derive_obj_name(self, urdf_path: str) -> str:
        p = Path(urdf_path)
        stem = p.stem
        if stem.lower() in ("original", "target", "simple_object"):
            return p.parent.name
        return stem

    def _load(self):
        raw = np.load(self.exported_npy_path, allow_pickle=True)
        payload = self._as_payload(raw)

        raw_obj_urdf_path = str(Path(payload["original_urdf_path"]).resolve())
        raw_target_obj_urdf_path = str(Path(payload["target_urdf_path"]).resolve())

        self._obj_name = self._derive_obj_name(raw_obj_urdf_path)
        self.target_object_name = self._derive_obj_name(raw_target_obj_urdf_path)

        # Rewrite URDF paths for cluster execution before any downstream parsing/use.
        self._obj_urdf_path = str(process_urdf(raw_obj_urdf_path))
        self._target_obj_urdf_path = str(process_urdf(raw_target_obj_urdf_path))

        right_verts = np.asarray(payload["right_mano_verts"], dtype=np.float32)
        right_joints = np.asarray(payload["right_mano_joints"], dtype=np.float32)
        if right_verts.ndim != 2 or right_verts.shape[-1] != 3:
            raise ValueError(
                f"Expected right_mano_verts (N,3), got {right_verts.shape}."
            )
        if right_joints.shape != (21, 3):
            raise ValueError(
                f"Expected right_mano_joints (21,3), got {right_joints.shape}."
            )

        left_jvals = np.asarray(payload["left_jvals"], dtype=np.float32).reshape(-1)
        right_jvals = np.asarray(payload["right_jvals"], dtype=np.float32).reshape(-1)
        left_jnames = normalize_retargeted_joint_names(
            np_array_to_list(np.asarray(payload["left_jnames"], dtype=object))
        )
        right_jnames = normalize_retargeted_joint_names(
            np_array_to_list(np.asarray(payload["right_jnames"], dtype=object))
        )

        self.left_hand_verts = self._apply_demo_height(
            self._repeat_frames(np.zeros_like(right_verts)[None, ...])
        )
        self.right_hand_verts = self._apply_demo_height(
            self._repeat_frames(right_verts[None, ...])
        )
        left_joints = self._apply_demo_height(
            self._repeat_frames(np.zeros_like(right_joints)[None, ...])
        )
        right_joints = self._apply_demo_height(
            self._repeat_frames(right_joints[None, ...])
        )

        self.relevant_left_joint_pos = {
            k: left_joints[:, j_idx] for k, j_idx in self.hand_joint_map.items()
        }
        self.relevant_right_joint_pos = {
            k: right_joints[:, j_idx] for k, j_idx in self.hand_joint_map.items()
        }

        left_tz_joint_idx = left_jnames.index("L_forearm_tz_link_joint")
        right_tz_joint_idx = right_jnames.index("R_forearm_tz_link_joint")

        self._left_jnames = left_jnames
        self._right_jnames = right_jnames
        self._left_jvals = self._repeat_frames(left_jvals[None, ...]).astype(np.float32)
        self._right_jvals = self._repeat_frames(right_jvals[None, ...]).astype(
            np.float32
        )

        self._left_jvals[..., left_tz_joint_idx] += self.demo_ht
        self._right_jvals[..., right_tz_joint_idx] += self.demo_ht

        # Keep a small compatibility payload with ARCTIC-like keys.
        self.world_coord = {
            "joints.left": left_joints,
            "joints.right": right_joints,
            "verts.left": self.left_hand_verts,
            "verts.right": self.right_hand_verts,
        }
        self._data = {
            "world_coord": self.world_coord,
            "params": {
                "obj_trans": np.zeros((self._n_frames, 3), dtype=np.float32),
                "obj_rot": np.zeros((self._n_frames, 3), dtype=np.float32),
                "obj_arti": np.zeros(
                    (self._n_frames, len(self.get_joint_names(self._obj_urdf_path))),
                    dtype=np.float32,
                ),
            },
        }
        self._data["params"]["obj_trans"][:, 2] = float(self.demo_ht)

    def get_human_hand_kpt_face_indices(self, side: str) -> dict[str, int]:
        if side != "right":
            raise ValueError(
                f"OakinkProcessorGrasp only provides right MANO data, got {side!r}."
            )
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

    def original_object_motion(self):
        # Exported pair data is static single-step; expose as repeated sequence.
        obj_pos = np.zeros((self._n_frames, 3), dtype=np.float32)
        obj_pos[:, 2] = float(self.demo_ht)
        obj_rot = np.zeros((self._n_frames, 3), dtype=np.float32)
        obj_quat_xyzw = R.from_rotvec(obj_rot).as_quat()
        obj_quat = np.roll(obj_quat_xyzw, 1, axis=-1)  # xyzw -> wxyz

        joint_names = self.get_joint_names(self._obj_urdf_path)
        if len(joint_names) == 0:
            obj_arti = np.zeros((self._n_frames, 0), dtype=np.float32)
        else:
            # No articulated state in exported payload; keep zeros.
            obj_arti = np.zeros((self._n_frames, len(joint_names)), dtype=np.float32)

        return {
            "obj_pos": obj_pos,  # meters
            "obj_rot": obj_rot,
            "obj_arti": obj_arti,
            "obj_quat": obj_quat,
            "joint_names": joint_names,
        }

    def _build_retargeted_motion(self):
        ref = self.object_reference_motion
        self.retargeted_motion = {
            "obj_pos": ref["obj_pos"].copy(),
            "obj_quat": ref["obj_quat"].copy(),
            "obj_rot": ref["obj_rot"].copy(),
            "obj_arti": ref["obj_arti"].copy(),
            "joint_names": list(ref["joint_names"]),
            "left_jnames": list(self._left_jnames),
            "left_jvals": self._left_jvals.copy(),
            "right_jnames": list(self._right_jnames),
            "right_jvals": self._right_jvals.copy(),
        }

    def is_identical(self):
        return self._obj_urdf_path == self._target_obj_urdf_path

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
                link_name: self._apply_demo_height(
                    link_fk[link_name][:, :3, 3].astype(np.float32, copy=False)
                )
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

    @property
    def data(self):
        return self._data
