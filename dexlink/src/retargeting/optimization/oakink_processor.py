from __future__ import annotations

import json
import pickle
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from dexlink.src.retargeting.data.human_hand_kpts import (
    human_hand_kpt_positions_from_faces,
    load_precomputed_human_hand_kpt_face_indices,
)
from dexlink.src.retargeting.optimization.oakink_data_format import create_urdf_from_obj
from dexlink.utils.retargeting.path_utils import get_external_data_dir

OAKINK_DIR = get_external_data_dir() / "oakink"

# NumPy 2.x compatibility for urdfpy.
if not hasattr(np, "product"):
    np.product = np.prod


def _ensure_ptp_compat():
    # NumPy 2 removed ndarray.ptp(); some trimesh code still calls it.
    try:
        _ = np.asarray([0.0]).ptp()
        return
    except Exception:
        pass

    # Easiest fix: restore ndarray.ptp -> np.ptp (may fail on immutable ndarray type).
    try:
        setattr(
            np.ndarray,
            "ptp",
            lambda self, *args, **kwargs: np.ptp(np.asarray(self), *args, **kwargs),
        )
        _ = np.asarray([0.0]).ptp()
        return
    except Exception:
        pass

    # Fallback for older trimesh: patch only the failing setter path.
    try:
        from trimesh.constants import log, tol

        vertex_prop = trimesh.Trimesh.vertex_normals
        if isinstance(vertex_prop, property) and vertex_prop.fset is not None:

            def _vertex_normals_setter_np2(self, values):
                if values is not None:
                    values = np.asanyarray(values, order="C", dtype=np.float64)
                    if values.shape == self.vertices.shape:
                        if np.ptp(values) < tol.merge:
                            log.warning("vertex_normals are all set to zero!")
                        self._cache["vertex_normals"] = values

            trimesh.Trimesh.vertex_normals = property(
                vertex_prop.fget,
                _vertex_normals_setter_np2,
                vertex_prop.fdel,
                vertex_prop.__doc__,
            )

        face_prop = trimesh.Trimesh.face_normals
        if isinstance(face_prop, property) and face_prop.fset is not None:

            def _face_normals_setter_np2(self, values):
                if values is not None:
                    values = np.asanyarray(values, order="C", dtype=np.float64)
                    if values.shape == self.faces.shape:
                        if np.ptp(values) < tol.merge:
                            log.warning("face_normals are all set to zero!")
                        self._cache["face_normals"] = values

            trimesh.Trimesh.face_normals = property(
                face_prop.fget,
                _face_normals_setter_np2,
                face_prop.fdel,
                face_prop.__doc__,
            )
    except Exception:
        pass


_ensure_ptp_compat()


from dexlink.src.retargeting.data.oakink.oakink_loader import OakInkSingleHandLoader
from dexlink.src.retargeting.data.process_arctic import (
    clip_motion,
    normalize_retargeted_joint_names,
    np_array_to_list,
    upsample,
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
from dexlink.utils.retargeting.urdf_utils import scale_urdf

# my wrapper to extract useful information from the original arctic demo file.
# ideally should just be a good queryable class, without doing complex processing on the demos.


if TYPE_CHECKING:
    from dexlink.src.rl.sim.robot import Hand


class OakInkProcessor:
    def __init__(
        self,
        original_demo_path: str,
        run_dir: str | Path | None = None,
        target_object: str | None = None,
        target_demo_path: str | None = None,
        target_obj_path: str | None = None,
        start_timestep=0,
        end_timestep=-1,
        skip=1,  # the step size. 1 means use every frame
        reference_urdf=None,
        target_urdf=None,
        retargeted_results: str | None = None,
        retargeted_results_upsample: bool = False,
        oakink_hand_side: str = "right",
        oakink_smplx_model_path: str | None = None,
        oakink_mano_assets_root: str | None = None,
        target_transform_info: str | None = None,
        target_scale: float = 1.0,
    ):
        """
        start_timestep: int, starting timestep to consider from the original demo (inclusive)
        end_timestep: int, ending timestep (exclusive)
        """

        self.original_demo_path = original_demo_path
        self.run_dir = (
            Path(run_dir)
            if run_dir is not None
            else get_logs_dir() / "oakink_processor"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir = self.run_dir / "oakink_processor_exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.target_object_name = (
            str(target_object)
            if target_object is not None
            else f"oakink_{Path(original_demo_path).stem}"
        )
        self.provided_reference_urdf = reference_urdf
        self.provided_target_urdf = target_urdf
        self.target_demo_path = target_demo_path
        self.target_obj_path = target_obj_path
        self.retargeted_results = retargeted_results
        self.retargeted_results_upsample = retargeted_results_upsample
        self.oakink_hand_side = str(oakink_hand_side).lower()
        if self.oakink_hand_side not in ("left", "right"):
            raise ValueError(
                f"`oakink_hand_side` must be 'left' or 'right', got {oakink_hand_side!r}."
            )
        self.oakink_smplx_model_path = oakink_smplx_model_path
        self.oakink_mano_assets_root = oakink_mano_assets_root
        self.target_transform_info = target_transform_info
        self.target_transform = self._load_target_transform(target_transform_info)
        self.target_scale = float(target_scale)
        if self.target_scale <= 0.0:
            raise ValueError(
                f"`target_scale` must be positive, got {self.target_scale}."
            )
        self._target_mesh_transform_center = None
        self._skip_applied_in_loader = False
        # maps important joints in the hand to their indices in the
        # joints array (n_frames, 21, 3)
        # the robots are assumed to have similar mapping provided
        # where the keys map to link names in the urdf

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

        retgt_in = Path(self.retargeted_results)
        if retgt_in.exists():
            results_path = retgt_in
        else:
            candidate1 = get_logs_dir() / "trajopt_results" / self.retargeted_results
            candidate2 = get_logs_dir() / self.retargeted_results
            if candidate1.exists():
                results_path = candidate1
            elif candidate2.exists():
                results_path = candidate2
            else:
                raise FileNotFoundError(
                    f"Could not find retargeted results: {self.retargeted_results}"
                )
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
        if urdf_path is None:
            return []
        # Parse XML directly to avoid mesh loading side effects (e.g., textured PLY parsing).
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

    def _to_numpy(self, value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _to_axis_angle(self, value):
        arr = self._to_numpy(value)
        if arr is None:
            return None
        if arr.ndim == 3 and arr.shape[-2:] == (3, 3):
            return R.from_matrix(arr).as_rotvec()
        if arr.shape[-1] == 4:
            # if quaternion, infer layout (xyzw or wxyz)
            quat = arr
            avg_abs_w_first = np.mean(np.abs(quat[..., 0]))
            avg_abs_w_last = np.mean(np.abs(quat[..., -1]))
            if avg_abs_w_first >= avg_abs_w_last:
                quat_xyzw = np.roll(quat, -1, axis=-1)
            else:
                quat_xyzw = quat
            return R.from_quat(quat_xyzw).as_rotvec()
        if arr.shape[-1] == 3:
            return arr
        raise ValueError(f"Unsupported rotation representation with shape {arr.shape}.")

    def _ensure_mm_translation(self, trans):
        trans = self._to_numpy(trans).astype(np.float32, copy=False)
        # ARCTIC stores object translation in millimeters.
        # OakInk payloads are commonly in meters, so detect and convert.
        median_norm = float(np.median(np.linalg.norm(trans, axis=-1)))
        if median_norm < 20.0:
            trans = trans * 1000.0
        return trans

    def __build_mano_21joints(self, mano_joints, wrist_pos):
        if not isinstance(mano_joints, dict):
            raise ValueError("OakInk data requires `mano_joints` as a dict.")

        def _get_any(keys):
            for key in keys:
                if key in mano_joints:
                    return self._to_numpy(mano_joints[key]).astype(
                        np.float32, copy=False
                    )
            raise KeyError(f"Missing MANO joint entry, expected one of: {keys}")

        # MANO-like 21-joint order used in ARCTIC processing.
        joints = [
            self._to_numpy(wrist_pos).astype(np.float32, copy=False),
            _get_any(["index_proximal", "index_base"]),
            _get_any(["index_intermediate"]),
            _get_any(["index_distal"]),
            _get_any(["middle_proximal", "middle_base"]),
            _get_any(["middle_intermediate"]),
            _get_any(["middle_distal"]),
            _get_any(["pinky_proximal", "little_proximal", "little_base"]),
            _get_any(["pinky_intermediate", "little_intermediate"]),
            _get_any(["pinky_distal", "little_distal"]),
            _get_any(["ring_proximal", "ring_base"]),
            _get_any(["ring_intermediate"]),
            _get_any(["ring_distal"]),
            _get_any(["thumb_proximal", "thumb_base"]),
            _get_any(["thumb_intermediate"]),
            _get_any(["thumb_distal"]),
            _get_any(["thumb_tip"]),
            _get_any(["index_tip"]),
            _get_any(["middle_tip"]),
            _get_any(["ring_tip"]),
            _get_any(["pinky_tip", "little_tip"]),
        ]
        return np.stack(joints, axis=1).astype(np.float32, copy=False)

    def _resolve_oakink_obj_mesh_path(self, obj_id: str) -> str | None:
        meta_root = OAKINK_DIR / "shape" / "metaV2"
        object_meta_path = meta_root / "object_id.json"
        virtual_meta_path = meta_root / "virtual_object_id.json"

        obj_id = str(obj_id)
        obj_name = None
        obj_root = None

        if object_meta_path.exists():
            with open(object_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if obj_id in meta:
                obj_name = meta[obj_id]["name"]
                obj_root = OAKINK_DIR / "shape" / "OakInkObjectsV2"

        if obj_name is None and virtual_meta_path.exists():
            with open(virtual_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if obj_id in meta:
                obj_name = meta[obj_id]["name"]
                obj_root = OAKINK_DIR / "shape" / "OakInkVirtualObjectsV2"

        if obj_name is None or obj_root is None:
            return None

        obj_dir = obj_root / obj_name
        if not obj_dir.exists():
            return None

        mesh_path = None
        for mesh_dir in ("align_ds", "align"):
            mesh_dir_path = obj_dir / mesh_dir
            if not mesh_dir_path.exists():
                continue
            candidates = sorted(mesh_dir_path.glob("*.obj")) + sorted(
                mesh_dir_path.glob("*.ply")
            )
            if candidates:
                mesh_path = candidates[0]
                break
        if mesh_path is None:
            return None

        return str(mesh_path.resolve())

    def _resolve_oakink_obj_urdf(self, obj_id: str) -> str | None:
        mesh_path = self._resolve_oakink_obj_mesh_path(obj_id)
        if mesh_path is None:
            return None

        urdf_out_dir = self.export_dir / f"oakink_{obj_id}"
        return self._build_urdf_from_mesh_path(
            mesh_path=mesh_path,
            output_dir=urdf_out_dir,
            create_urdf_fn=create_urdf_from_obj,
        )

    def _build_urdf_from_mesh_path(
        self,
        mesh_path: str | Path,
        output_dir: str | Path,
        create_urdf_fn,
    ) -> str | None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        urdf_path = output_dir / "simple_object.urdf"
        mesh_for_urdf = self._export_processed_mesh_obj(
            mesh_path=mesh_path,
            output_dir=output_dir,
            output_stem=Path(mesh_path).stem,
        )
        create_urdf_fn(str(mesh_for_urdf), str(output_dir), scale=1.0)
        return str(urdf_path)

    def _load_ply_mesh_without_trimesh_parser(
        self, mesh_path: str | Path
    ) -> trimesh.Trimesh:
        try:
            from plyfile import PlyData
        except ImportError as exc:
            raise ImportError(
                "Loading .ply meshes requires the optional `plyfile` package. "
                f"Install it before using PLY target meshes: {mesh_path}"
            ) from exc

        ply = PlyData.read(str(mesh_path))
        if "vertex" not in ply:
            raise ValueError(f"PLY file is missing vertex data: {mesh_path}")

        vertex_data = ply["vertex"].data
        vertices = np.column_stack(
            [vertex_data["x"], vertex_data["y"], vertex_data["z"]]
        ).astype(np.float64, copy=False)

        if "face" not in ply:
            warnings.warn(
                f"PLY file has no face element; using convex hull from vertices: {mesh_path}",
                RuntimeWarning,
            )
            return self._build_convex_hull_mesh(vertices, mesh_path)

        face_rows = ply["face"].data
        face_field = (
            "vertex_indices"
            if "vertex_indices" in face_rows.dtype.names
            else face_rows.dtype.names[0]
        )
        triangles: list[list[int]] = []
        for row in face_rows:
            indices = np.asarray(row[face_field], dtype=np.int64).reshape(-1)
            if indices.size < 3:
                continue
            if indices.size == 3:
                triangles.append(indices.tolist())
                continue
            for i in range(1, indices.size - 1):
                triangles.append(
                    [int(indices[0]), int(indices[i]), int(indices[i + 1])]
                )

        if not triangles:
            warnings.warn(
                f"PLY file has no valid triangle faces; using convex hull from vertices: {mesh_path}",
                RuntimeWarning,
            )
            return self._build_convex_hull_mesh(vertices, mesh_path)

        return trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(triangles, dtype=np.int64),
            process=False,
        )

    def _build_convex_hull_mesh(
        self, vertices: np.ndarray, mesh_path: str | Path
    ) -> trimesh.Trimesh:
        unique_vertices = np.unique(np.asarray(vertices, dtype=np.float64), axis=0)
        if unique_vertices.shape[0] < 4:
            raise ValueError(
                "Cannot build a mesh from vertex-only PLY: need at least 4 unique "
                f"vertices, got {unique_vertices.shape[0]} in {mesh_path}."
            )
        try:
            hull = trimesh.convex.convex_hull(unique_vertices)
        except Exception as exc:
            raise ValueError(
                f"Failed to construct a convex hull mesh from vertex-only PLY: {mesh_path}."
            ) from exc

        if not isinstance(hull, trimesh.Trimesh) or len(hull.faces) == 0:
            raise ValueError(
                f"Convex hull construction produced an invalid mesh for {mesh_path}."
            )
        return hull

    def _load_mesh(self, mesh_path: str | Path) -> trimesh.Trimesh:
        mesh_path = Path(mesh_path).resolve()
        try:
            loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
        except TypeError as exc:
            if mesh_path.suffix.lower() != ".ply":
                raise
            if "structured or void" not in str(exc):
                raise
            loaded = self._load_ply_mesh_without_trimesh_parser(mesh_path)

        if isinstance(loaded, trimesh.Scene):
            geometries = [
                g
                for g in loaded.geometry.values()
                if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0
            ]
            if not geometries:
                raise ValueError(f"No triangle mesh geometry found in {mesh_path}.")
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = loaded

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Unsupported mesh type {type(mesh)} for {mesh_path}.")
        if len(mesh.faces) == 0:
            raise ValueError(f"Mesh has no faces: {mesh_path}")
        return mesh.copy()

    def _mesh_transform_center(self, mesh_path: str | Path) -> np.ndarray:
        mesh = self._load_mesh(mesh_path)
        return np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)

    def _parse_transform_scalar(
        self, value: float | int | str, field_name: str
    ) -> float:
        if isinstance(value, (float, int, np.floating, np.integer)):
            return float(value)
        if isinstance(value, str):
            expr = value.strip()
            if not expr:
                raise ValueError(f"`{field_name}` contains an empty string expression.")
            try:
                return float(eval(expr, {"__builtins__": {}}, {"pi": np.pi}))
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse `{field_name}` expression {value!r}."
                ) from exc
        raise TypeError(
            f"`{field_name}` entries must be numeric or string expressions, got {type(value)}."
        )

    def _parse_transform_vector(
        self, values: object, expected_len: int, field_name: str
    ) -> np.ndarray:
        if not isinstance(values, (list, tuple)):
            raise TypeError(
                f"`{field_name}` must be a length-{expected_len} list or tuple, got {type(values)}."
            )
        if len(values) != expected_len:
            raise ValueError(
                f"`{field_name}` must have length {expected_len}, got {len(values)}."
            )
        parsed = [self._parse_transform_scalar(v, field_name) for v in values]
        return np.asarray(parsed, dtype=np.float64)

    def _parse_transform_matrix(
        self, values: object, shape: tuple[int, int], field_name: str
    ) -> np.ndarray:
        if not isinstance(values, (list, tuple)):
            raise TypeError(
                f"`{field_name}` must be a {shape[0]}x{shape[1]} nested list, got {type(values)}."
            )
        if len(values) != shape[0]:
            raise ValueError(
                f"`{field_name}` must have {shape[0]} rows, got {len(values)}."
            )
        rows = [
            self._parse_transform_vector(row, shape[1], field_name) for row in values
        ]
        return np.stack(rows, axis=0)

    def _load_target_transform(
        self, target_transform_info: str | None
    ) -> np.ndarray | None:
        return load_target_transform(target_transform_info)

    def _repair_mesh_for_drake_pytorch3d(
        self, mesh: trimesh.Trimesh
    ) -> trimesh.Trimesh:
        try:
            from pytorch3d.ops import cubify
        except Exception as exc:
            raise ImportError(
                "PyTorch3D cubify is not available in this environment."
            ) from exc

        mesh = mesh.copy()
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()

        bounds = mesh.bounds
        extent = float(np.max(bounds[1] - bounds[0]))
        if extent <= 0.0:
            raise ValueError("Cannot repair degenerate mesh with zero bounding extent.")

        target_resolution = 64
        pitch = max(extent / target_resolution, 1e-4)
        vox = mesh.voxelized(pitch=pitch)
        occ = np.asarray(vox.matrix, dtype=np.float32)
        if occ.ndim != 3 or occ.size == 0:
            raise ValueError("Voxelization failed to produce a valid occupancy grid.")

        occ_t = torch.from_numpy(occ[None]).to(dtype=torch.float32)
        mesh_p3d = cubify(occ_t, thresh=0.5)
        verts = mesh_p3d.verts_list()[0].detach().cpu().numpy()
        faces = mesh_p3d.faces_list()[0].detach().cpu().numpy()
        if verts.size == 0 or faces.size == 0:
            raise ValueError("PyTorch3D cubify produced an empty mesh.")

        # Map cubify vertices from voxel index space back into world coordinates.
        transform_raw = vox.transform
        if hasattr(transform_raw, "matrix"):
            transform = np.asarray(transform_raw.matrix, dtype=np.float64)
        else:
            transform = np.asarray(transform_raw, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(
                f"Unexpected voxel transform shape {transform.shape}; expected (4, 4)."
            )
        verts_h = np.concatenate(
            [verts.astype(np.float64, copy=False), np.ones((verts.shape[0], 1))], axis=1
        )
        verts_world = (transform @ verts_h.T).T[:, :3]
        repaired = trimesh.Trimesh(
            vertices=verts_world,
            faces=faces.astype(np.int64, copy=False),
            process=False,
        )
        repaired.remove_unreferenced_vertices()
        repaired.merge_vertices()
        try:
            trimesh.repair.fix_winding(repaired)
        except Exception:
            pass
        try:
            trimesh.repair.fix_normals(repaired)
        except Exception:
            pass
        return repaired

    def _repair_mesh_for_drake(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        return mesh.copy()
        mesh = mesh.copy()

        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)

        # Keep the repair path within trimesh only; voxel marching-cubes
        # pulls in scikit-image, which is ABI-incompatible in this env.
        mesh.remove_unreferenced_vertices()
        mesh.rezero()
        mesh.remove_infinite_values()
        try:
            mesh.fill_holes()
        except Exception:
            pass

        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            components = [c for c in components if len(c.faces) > 0]
            if components:
                mesh = max(components, key=lambda m: float(m.area))

        volume = abs(float(mesh.volume)) if mesh.is_volume else 0.0
        if (not mesh.is_watertight) or volume < 1e-14:
            mesh = self._repair_mesh_for_drake_pytorch3d(mesh)
            mesh.remove_unreferenced_vertices()
            mesh.merge_vertices()

            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fix_normals(mesh)

        if float(mesh.volume) < 0.0:
            mesh.invert()

        final_volume = abs(float(mesh.volume)) if mesh.is_volume else 0.0
        if final_volume < 1e-14:
            raise ValueError(
                "Processed mesh still has near-zero volume after repair; Drake will reject it."
            )
        return mesh

    def _export_processed_mesh_obj(
        self,
        mesh_path: str | Path,
        output_dir: str | Path,
        output_stem: str,
        transform: np.ndarray | None = None,
        scale: float = 1.0,
        center: np.ndarray | None = None,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        obj_out = output_dir / f"{output_stem}.obj"

        mesh = apply_target_mesh_transform(
            self._load_mesh(mesh_path),
            transform=transform,
            scale=scale,
            center=center,
        )

        # print mesh dimensions (3d bounding box extent)
        print(
            f"Processed mesh {mesh_path} has bounding box extent: "
            f"{mesh.bounds[1] - mesh.bounds[0]}"
        )

        mesh = self._repair_mesh_for_drake(mesh)
        mesh.export(obj_out)
        return obj_out

    def _materialize_target_mesh(
        self,
        mesh_path: str | Path,
        obj_name: str,
        center: np.ndarray | None = None,
    ) -> str:
        mesh_path = Path(mesh_path).resolve()
        if (self.target_transform is None) and (self.target_scale == 1.0):
            return str(mesh_path)

        output_dir = self.export_dir / f"{obj_name}_target_mesh"
        transformed_obj = self._export_processed_mesh_obj(
            mesh_path=mesh_path,
            output_dir=output_dir,
            output_stem=f"{Path(mesh_path).stem}_processed",
            transform=self.target_transform,
            scale=self.target_scale,
            center=center,
        )
        return str(transformed_obj.resolve())

    def materialize_target_bps_mesh(self, mesh_path: str | Path) -> str:
        center = None
        if (self.target_transform is not None) or (self.target_scale != 1.0):
            if self._target_mesh_transform_center is None:
                raise ValueError(
                    "Target BPS mesh preprocessing requires the full target mesh "
                    "transform center, but it has not been initialized."
                )
            center = self._target_mesh_transform_center
        return self._materialize_target_mesh(
            mesh_path=mesh_path,
            obj_name=f"{self.target_object_name}_bps_sampling",
            center=center,
        )

    def _export_mesh_as_obj(
        self,
        mesh_path: str | Path,
        output_dir: str | Path,
        output_stem: str,
    ) -> str:
        return str(
            self._export_processed_mesh_obj(
                mesh_path=mesh_path,
                output_dir=output_dir,
                output_stem=output_stem,
            )
        )

    def _resolve_target_from_obj_path(self, obj_path: str | Path):
        mesh_path = Path(obj_path).resolve()
        if not mesh_path.exists():
            raise FileNotFoundError(f"Target object mesh not found: {mesh_path}")
        if mesh_path.suffix.lower() not in (".obj", ".ply"):
            raise ValueError(
                f"Unsupported target object mesh: {mesh_path}. Expected .obj or .ply."
            )

        stem = mesh_path.stem
        obj_name = stem if stem.startswith("oakink_") else f"oakink_{stem}"
        urdf_out_dir = self.export_dir / obj_name
        obj_urdf = self._build_urdf_from_mesh_path(
            mesh_path=mesh_path,
            output_dir=urdf_out_dir,
            create_urdf_fn=create_urdf_from_obj,
        )
        return obj_name, obj_urdf, str(mesh_path)

    def _resolve_mano_assets_root(self) -> str:
        root = get_external_data_dir() / "mano" / "mano_v1_2"
        if not (root / "models" / "MANO_RIGHT.pkl").is_file():
            raise FileNotFoundError(
                f"Missing licensed MANO v1.2 assets at {root}. See "
                "external_data/README.md for the required external-data layout."
            )
        return str(root)

    def _mano_pose_to_axis_angle(self, pose: np.ndarray) -> np.ndarray:
        pose = self._to_numpy(pose).astype(np.float32, copy=False)
        if pose.ndim == 1:
            pose = pose[None, :]
        if pose.ndim == 2 and pose.shape[-1] == 48:
            return pose
        if pose.ndim == 3 and pose.shape[-2:] == (16, 3):
            return pose.reshape(pose.shape[0], 48)
        if pose.ndim == 3 and pose.shape[-2:] == (16, 4):
            rotvec = R.from_quat(pose.reshape(-1, 4)).as_rotvec()
            return rotvec.reshape(pose.shape[0], 48).astype(np.float32, copy=False)
        raise ValueError(
            f"Unsupported MANO pose shape {pose.shape}. Expected (48,), (N,48), (N,16,3), or (N,16,4)."
        )

    def _convert_shape_hand_param_payload(self, raw_data):
        # OakInk-Shape static grasp format: {"pose": (16,4), "shape": (10,), "tsl": (3,)}
        from manotorch.manolayer import ManoLayer, MANOOutput

        if "pose" not in raw_data or "shape" not in raw_data or "tsl" not in raw_data:
            raise ValueError(
                "OakInk-Shape payload requires `pose`, `shape`, and `tsl`."
            )

        mano_pose = self._mano_pose_to_axis_angle(raw_data["pose"])
        mano_shape = self._to_numpy(raw_data["shape"]).astype(np.float32, copy=False)
        if mano_shape.ndim == 1:
            mano_shape = mano_shape[None, :]

        tsl = self._to_numpy(raw_data["tsl"]).astype(np.float32, copy=False)
        if tsl.ndim == 1:
            tsl = tsl[None, :]
        if tsl.shape[-1] != 3:
            raise ValueError(f"Expected `tsl` with final dim 3, got {tsl.shape}.")

        n_frames = max(mano_pose.shape[0], mano_shape.shape[0], tsl.shape[0])
        if mano_pose.shape[0] == 1 and n_frames > 1:
            mano_pose = np.repeat(mano_pose, n_frames, axis=0)
        if mano_shape.shape[0] == 1 and n_frames > 1:
            mano_shape = np.repeat(mano_shape, n_frames, axis=0)
        if tsl.shape[0] == 1 and n_frames > 1:
            tsl = np.repeat(tsl, n_frames, axis=0)

        if (
            mano_pose.shape[0] != n_frames
            or mano_shape.shape[0] != n_frames
            or tsl.shape[0] != n_frames
        ):
            raise ValueError(
                "Inconsistent frame counts among pose/shape/tsl: "
                f"{mano_pose.shape[0]}, {mano_shape.shape[0]}, {tsl.shape[0]}"
            )

        mano_root = self._resolve_mano_assets_root()
        mano_layer = ManoLayer(center_idx=0, mano_assets_root=mano_root)
        mano_output: MANOOutput = mano_layer(
            torch.from_numpy(mano_pose),
            torch.from_numpy(mano_shape),
        )
        joints_active = (
            mano_output.joints.detach().cpu().numpy().astype(np.float32, copy=False)
            + tsl[:, None, :]
        )
        verts_active = (
            mano_output.verts.detach().cpu().numpy().astype(np.float32, copy=False)
            + tsl[:, None, :]
        )

        wrist_rot_aa = mano_pose[:, :3].astype(np.float32, copy=False)
        wrist_pos = joints_active[:, 0].astype(np.float32, copy=False)

        zeros_joints = np.zeros_like(joints_active, dtype=np.float32)
        zeros_verts = np.zeros_like(verts_active, dtype=np.float32)
        zeros_pos = np.zeros_like(wrist_pos, dtype=np.float32)
        zeros_rot = np.zeros_like(wrist_rot_aa, dtype=np.float32)

        hand_side = self.oakink_hand_side
        if hand_side == "left":
            joints_left, joints_right = joints_active, zeros_joints
            verts_left, verts_right = verts_active, zeros_verts
            trans_l, rot_l = wrist_pos, wrist_rot_aa
            trans_r, rot_r = zeros_pos, zeros_rot
        else:
            joints_left, joints_right = zeros_joints, joints_active
            verts_left, verts_right = zeros_verts, verts_active
            trans_l, rot_l = zeros_pos, zeros_rot
            trans_r, rot_r = wrist_pos, wrist_rot_aa

        demo_path = Path(self.original_demo_path).resolve()
        obj_id = demo_path.stem
        parents = demo_path.parents
        if len(parents) >= 2 and parents[1].name:
            obj_id = parents[1].name
        obj_mesh_path = self._resolve_oakink_obj_mesh_path(obj_id)
        obj_urdf_path = self._resolve_oakink_obj_urdf(obj_id)

        return {
            "left_joints": joints_left,
            "right_joints": joints_right,
            "left_verts": verts_left,
            "right_verts": verts_right,
            "trans_l": trans_l,
            "rot_l": rot_l,
            "trans_r": trans_r,
            "rot_r": rot_r,
            "obj_trans": np.zeros((n_frames, 3), dtype=np.float32),
            "obj_rot": np.zeros((n_frames, 3), dtype=np.float32),
            "obj_arti": np.zeros((n_frames,), dtype=np.float32),
            "_meta": {
                "obj_name": f"oakink_{obj_id}",
                "obj_mesh_path": obj_mesh_path,
                "obj_urdf_path": obj_urdf_path,
            },
        }

    def _convert_sequence_payload_with_oakink_loader(self):
        if not self.oakink_smplx_model_path:
            raise ValueError(
                "OakInk sequence annotation requires `oakink_smplx_model_path` to build joints/verts."
            )
        loader = OakInkSingleHandLoader(
            pkl_path=self.original_demo_path,
            smplx_model_path=self.oakink_smplx_model_path,
        )
        payload = loader.load(hand_side=self.oakink_hand_side, skip=max(1, self.skip))
        self._skip_applied_in_loader = True
        return self._convert_oakink_payload(payload)

    def _convert_oakink_payload(self, raw_data):
        if not isinstance(raw_data, dict):
            raise ValueError("OakInk demo file must contain a dictionary payload.")

        hand_side = str(raw_data.get("hand_side", "left")).lower()
        if hand_side not in ("left", "right"):
            raise ValueError(
                f"`hand_side` must be 'left' or 'right', got {hand_side!r}."
            )

        wrist_pos = raw_data.get("wrist_pos", None)
        mano_joints = raw_data.get("mano_joints", None)
        if wrist_pos is None or mano_joints is None:
            raise ValueError(
                "OakInk payload requires both `wrist_pos` and `mano_joints` keys."
            )

        wrist_pos = self._to_numpy(wrist_pos).astype(np.float32, copy=False)

        # Build 21 MANO joints once (works for both hands)
        joints_active = self.__build_mano_21joints(mano_joints, wrist_pos)
        n_frames = joints_active.shape[0]

        wrist_rot = raw_data.get("wrist_rot", None)
        if wrist_rot is None:
            wrist_rot_aa = np.zeros((n_frames, 3), dtype=np.float32)
        else:
            wrist_rot_aa = self._to_axis_angle(wrist_rot).astype(np.float32, copy=False)

        # Object trajectory (unchanged)
        obj_trajectory = raw_data.get("obj_trajectory", None)
        if obj_trajectory is None:
            raise ValueError("OakInk payload requires `obj_trajectory`.")
        obj_trajectory = self._to_numpy(obj_trajectory)
        if obj_trajectory.ndim == 3 and obj_trajectory.shape[-2:] == (4, 4):
            obj_trans = obj_trajectory[:, :3, 3]
            obj_rot_aa = R.from_matrix(obj_trajectory[:, :3, :3]).as_rotvec()
        else:
            raise ValueError(
                "`obj_trajectory` must be an array of shape (N, 4, 4) in OakInk payload."
            )

        obj_trans_mm = self._ensure_mm_translation(obj_trans)
        obj_rot_aa = obj_rot_aa.astype(np.float32, copy=False)

        obj_arti = raw_data.get("obj_arti", None)
        if obj_arti is None:
            obj_arti = np.zeros((n_frames,), dtype=np.float32)
        else:
            obj_arti = self._to_numpy(obj_arti).astype(np.float32, copy=False)
            if obj_arti.ndim > 1 and obj_arti.shape[-1] == 1:
                obj_arti = obj_arti[:, 0]

        # Hand verts (optional)
        active_verts = raw_data.get("hand_verts", None)
        if active_verts is None:
            active_verts = raw_data.get(
                "left_hand_verts", None
            )  # legacy name; ok for either
        if active_verts is None:
            active_verts = joints_active.copy()
        else:
            active_verts = self._to_numpy(active_verts).astype(np.float32, copy=False)

        zeros_joints = np.zeros_like(joints_active, dtype=np.float32)
        zeros_verts = np.zeros_like(active_verts, dtype=np.float32)
        zeros_pos = np.zeros_like(wrist_pos, dtype=np.float32)
        zeros_rot = np.zeros_like(wrist_rot_aa, dtype=np.float32)

        if hand_side == "left":
            joints_left, joints_right = joints_active, zeros_joints
            verts_left, verts_right = active_verts, zeros_verts
            trans_l, rot_l = wrist_pos, wrist_rot_aa
            trans_r, rot_r = zeros_pos, zeros_rot
        else:
            joints_left, joints_right = zeros_joints, joints_active
            verts_left, verts_right = zeros_verts, active_verts
            trans_l, rot_l = zeros_pos, zeros_rot
            trans_r, rot_r = wrist_pos, wrist_rot_aa

        obj_name = raw_data.get("obj_id", None) or raw_data.get("obj_name", None)
        if obj_name is None:
            obj_name = Path(self.original_demo_path).stem
        obj_name = str(obj_name)
        if not obj_name.startswith("oakink_"):
            obj_name = f"oakink_{obj_name}"

        obj_urdf_path = raw_data.get("obj_urdf_path", None)
        obj_mesh_path = raw_data.get("obj_mesh_path", None)

        converted = {
            "left_joints": joints_left,
            "right_joints": joints_right,
            "left_verts": verts_left,
            "right_verts": verts_right,
            "trans_l": trans_l.astype(np.float32, copy=False),
            "rot_l": rot_l.astype(np.float32, copy=False),
            "trans_r": trans_r.astype(np.float32, copy=False),
            "rot_r": rot_r.astype(np.float32, copy=False),
            "obj_trans": obj_trans_mm,
            "obj_rot": obj_rot_aa,
            "obj_arti": obj_arti,
            "_meta": {
                "obj_name": obj_name,
                "obj_mesh_path": obj_mesh_path,
                "obj_urdf_path": obj_urdf_path,
            },
        }
        return converted

    def _load_demo_payload(self, demo_path: str | Path | None = None):
        demo_path = Path(self.original_demo_path if demo_path is None else demo_path)
        suffix = demo_path.suffix.lower()
        if suffix == ".pkl":
            with open(demo_path, "rb") as f:
                data = pickle.load(f)
        elif suffix == ".npz":
            npz = np.load(demo_path, allow_pickle=True)
            if "arr_0" in npz.files:
                data = npz["arr_0"].item()
            else:
                data = {k: npz[k] for k in npz.files}
        else:
            data = np.load(demo_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.shape == ():
                data = data.item()
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict payload in {demo_path}, got {type(data)}.")
        return data

    def _resolve_target_from_demo_path(self, demo_path: str | Path):
        demo_path = Path(demo_path).resolve()
        raw = self._load_demo_payload(demo_path)

        meta = raw.get("_meta", {}) if isinstance(raw, dict) else {}
        obj_name_raw = (
            meta.get("obj_name")
            or raw.get("obj_name")
            or raw.get("obj_id")
            or demo_path.stem
        )
        obj_name = str(obj_name_raw)
        if not obj_name.startswith("oakink_"):
            obj_name = f"oakink_{obj_name}"

        obj_urdf_path = meta.get("obj_urdf_path") or raw.get("obj_urdf_path")
        obj_mesh_path = meta.get("obj_mesh_path") or raw.get("obj_mesh_path")

        if obj_urdf_path is None and all(k in raw for k in ("pose", "shape", "tsl")):
            obj_id = (
                demo_path.parent.parent.name
                if len(demo_path.parents) >= 2
                else demo_path.stem
            )
            obj_mesh_path = self._resolve_oakink_obj_mesh_path(obj_id)
            obj_urdf_path = self._resolve_oakink_obj_urdf(obj_id)
            obj_name = f"oakink_{obj_id}"

        if obj_urdf_path is None:
            obj_id = raw.get("obj_id", None)
            if obj_id is not None:
                obj_mesh_path = obj_mesh_path or self._resolve_oakink_obj_mesh_path(
                    str(obj_id)
                )
                obj_urdf_path = self._resolve_oakink_obj_urdf(str(obj_id))

        if obj_urdf_path is None:
            try:
                obj_urdf_path = self.get_urdf_path(obj_name, scale=1.0)
            except Exception:
                obj_urdf_path = None

        return obj_name, obj_urdf_path, obj_mesh_path

    def _load_and_standardize_demo_data(self):
        data = self._load_demo_payload()
        if ("world_coord" in data) and ("params" in data):
            return {
                "left_joints": data["world_coord"]["joints.left"],
                "right_joints": data["world_coord"]["joints.right"],
                "left_verts": data["world_coord"]["verts.left"],
                "right_verts": data["world_coord"]["verts.right"],
                "trans_l": data["params"]["trans_l"],
                "rot_l": data["params"]["rot_l"],
                "trans_r": data["params"]["trans_r"],
                "rot_r": data["params"]["rot_r"],
                "obj_trans": data["params"]["obj_trans"],
                "obj_rot": data["params"]["obj_rot"],
                "obj_arti": data["params"]["obj_arti"],
                "_meta": data.get("_meta", {}),
            }
        if ("mocap_frame_id_list" in data) and ("raw_smplx" in data):
            return self._convert_sequence_payload_with_oakink_loader()
        if ("pose" in data) and ("shape" in data) and ("tsl" in data):
            return self._convert_shape_hand_param_payload(data)
        return self._convert_oakink_payload(data)

    def process_seq_len(self, data):

        # get the initial height from the first frame.
        self.initial_height = data["obj_trans"][0, 2] / 1000.0 - 0.015

        n_frames = data["left_joints"].shape[0]
        if self.end_timestep == -1:
            self.end_timestep = n_frames

        # this ignores the skip
        self.gt_n_frames = self.end_timestep - self.start_timestep

        # slice all time-dependent array fields
        step = 1 if self._skip_applied_in_loader else self.skip
        for key, val in list(data.items()):
            if key == "_meta" or not isinstance(val, np.ndarray):
                continue
            assert (
                val.shape[0] == n_frames
            ), f"Expected the first dimension of {key} to be {n_frames}, got {val.shape[0]}"
            data[key] = val[self.start_timestep : self.end_timestep : step]

        return data

    def process(self):
        # add to the data:
        # correct urdf path for the object
        self._data = self._load_and_standardize_demo_data()

        self._data = self.process_seq_len(self._data)

        meta = self._data.get("_meta", {})
        if "obj_name" in meta:
            self._obj_name = str(meta["obj_name"])
        else:
            self._obj_name = f"oakink_{Path(self.original_demo_path).stem}"

        self._target_obj_mesh_path = None
        if self.provided_target_urdf is None:
            if self.target_obj_path is not None:
                target_name, target_urdf, target_mesh_path = (
                    self._resolve_target_from_obj_path(self.target_obj_path)
                )
                if target_name:
                    self.target_object_name = str(target_name)
                if target_urdf is not None:
                    self.provided_target_urdf = str(target_urdf)
                self._target_obj_mesh_path = target_mesh_path
            elif self.target_demo_path is not None:
                target_name, target_urdf, target_mesh_path = (
                    self._resolve_target_from_demo_path(self.target_demo_path)
                )
                if target_name:
                    self.target_object_name = str(target_name)
                if target_urdf is not None:
                    self.provided_target_urdf = str(target_urdf)
                self._target_obj_mesh_path = target_mesh_path

        self._obj_mesh_path = meta.get("obj_mesh_path", None)
        if self._obj_mesh_path is None and self._obj_name.startswith("oakink_"):
            self._obj_mesh_path = self._resolve_oakink_obj_mesh_path(
                self._obj_name[len("oakink_") :]
            )
        if self._target_obj_mesh_path is None:
            self._target_obj_mesh_path = self._obj_mesh_path

        if (self.target_transform is not None) or (self.target_scale != 1.0):
            if self._target_obj_mesh_path is None:
                raise FileNotFoundError(
                    "Target mesh preprocessing was requested, but no target object mesh path could be resolved."
                )
            self._target_mesh_transform_center = self._mesh_transform_center(
                self._target_obj_mesh_path
            )
            self._target_obj_mesh_path = self._materialize_target_mesh(
                self._target_obj_mesh_path,
                self.target_object_name,
                center=self._target_mesh_transform_center,
            )
            self.provided_target_urdf = self._build_urdf_from_mesh_path(
                mesh_path=self._target_obj_mesh_path,
                output_dir=self.export_dir / f"{self.target_object_name}_target_urdf",
                create_urdf_fn=create_urdf_from_obj,
            )
            print(f"Applied target scale: {self.target_scale}")
            if self.target_transform is not None:
                print("Applied target rotation matrix:")
                print(
                    np.array2string(
                        self.target_transform, precision=6, suppress_small=True
                    )
                )

        raw_urdf = self.provided_reference_urdf or meta.get("obj_urdf_path", None)
        if raw_urdf is None:
            try:
                raw_urdf = self.get_urdf_path(self._obj_name, scale=1.0)
            except FileNotFoundError:
                raw_urdf = None
        self._obj_urdf_path = None if raw_urdf is None else process_urdf(raw_urdf)
        print(f"Original object urdf path: {self._obj_urdf_path}")
        self._n_frames = self._data["obj_trans"].shape[0]
        print(f"Number of demo frames: {self._n_frames}")

        self.object_reference_motion = self.replicated_object_motion()

        self.left_hand_verts = self._data["left_verts"]  # (num_frames, num_points, 3)
        self.right_hand_verts = self._data["right_verts"]  # (num_frames, num_points, 3)
        self._human_hand_kpt_face_indices = {}

        self.relevant_left_joint_pos = {
            k: self._data["left_joints"][:, j_idx]
            for k, j_idx in self.hand_joint_map.items()
        }
        self.relevant_right_joint_pos = {
            k: self._data["right_joints"][:, j_idx]
            for k, j_idx in self.hand_joint_map.items()
        }

        left_max_idx = int(np.max(LEFT_PALM_VERTEX_INDICES))
        right_max_idx = int(np.max(RIGHT_PALM_VERTEX_INDICES))
        if self.left_hand_verts.shape[1] > left_max_idx:
            self.left_inner_hand_verts = self.left_hand_verts[
                :, LEFT_PALM_VERTEX_INDICES, :
            ]
        else:
            self.left_inner_hand_verts = self.left_hand_verts
        if self.right_hand_verts.shape[1] > right_max_idx:
            self.right_inner_hand_verts = self.right_hand_verts[
                :, RIGHT_PALM_VERTEX_INDICES, :
            ]
        else:
            self.right_inner_hand_verts = self.right_hand_verts

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
        if side not in ("left", "right"):
            raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")
        face_indices = self.get_human_hand_kpt_face_indices(side)
        return human_hand_kpt_positions_from_faces(
            side=side,
            vertices=getattr(self, f"{side}_hand_verts"),
            face_indices=face_indices,
            names=names,
            frame_idx=frame_idx,
        )
        # self.disc_reference_obs = self._disc_reference_obs()

    def is_identical(self):
        return self.target_object_name == self._obj_name

    def _build_object_motion(self, urdf_path: str | Path | None = None):
        obj_rot = self._data["obj_rot"]
        obj_trans = self._data["obj_trans"]
        n_frames = obj_trans.shape[0]
        obj_quat = R.from_rotvec(obj_rot).as_quat()
        obj_quat = np.roll(obj_quat, 1, axis=-1)  # xyzw -> wxyz

        urdf_for_joints = self._obj_urdf_path if urdf_path is None else urdf_path
        joint_names = self.get_joint_names(urdf_for_joints)
        n_joints = len(joint_names)
        if n_joints == 0:
            obj_arti = np.zeros((n_frames, 0), dtype=np.float32)
        else:
            raw_arti = self._data["obj_arti"]
            if raw_arti.ndim == 1:
                raw_arti = raw_arti[:, None]
            if raw_arti.shape[1] == n_joints:
                obj_arti = raw_arti
            elif raw_arti.shape[1] == 1:
                obj_arti = np.repeat(raw_arti, n_joints, axis=1)
            else:
                obj_arti = raw_arti[:, :n_joints]
            obj_arti = obj_arti.astype(np.float32, copy=False)

        return {
            "obj_pos": obj_trans / 1000.0,  # convert to meters
            "obj_rot": obj_rot,
            "obj_arti": obj_arti,
            "obj_quat": obj_quat,
            "joint_names": joint_names,
        }

    def replicated_object_motion(self):
        # If a target mesh/URDF is explicitly provided, always use it for target object
        # schema (joint names / articulation dimensionality), regardless of object-name match.
        if self.provided_target_urdf is not None:
            self._target_obj_urdf_path = process_urdf(self.provided_target_urdf)
            if self.target_object_name == self._obj_name:
                print(
                    "Target object name matches demo object, but using provided target URDF."
                )
            else:
                warnings.warn(
                    "Target object differs from demo object; motion replication is approximate "
                    "(demo pose/rotation reused, articulation resized to target URDF joints)."
                )
            return self._build_object_motion(self._target_obj_urdf_path)

        if self.target_object_name == self._obj_name:
            print(
                "Target object is the same as original demo object. Using the original motion as it is."
            )
            self._target_obj_urdf_path = self._obj_urdf_path
            return self._build_object_motion(self._target_obj_urdf_path)
        else:
            warnings.warn(
                "Target object differs from demo object, but OakInk object-motion replication "
                "is not implemented. Falling back to original object motion."
            )
            self._target_obj_urdf_path = self._obj_urdf_path
            return self.original_object_motion()

    def original_object_motion(self):
        return self._build_object_motion(self._obj_urdf_path)

    def get_object_base_pose(self, obj_type="original"):
        if obj_type == "original":
            motion = self.original_object_motion()
        elif obj_type == "target":
            motion = self.object_reference_motion
        else:
            raise ValueError(
                f"`obj_type` must be one of ['original', 'target'], got {obj_type!r}."
            )
        return get_X(motion["obj_pos"][0], motion["obj_quat"][0])

    def export_object_obj(
        self, output_dir: str | Path | None = None, obj_type="original"
    ) -> str:
        if obj_type == "original":
            mesh_path = self._obj_mesh_path
            obj_name = self._obj_name
        elif obj_type == "target":
            mesh_path = self._target_obj_mesh_path or self._obj_mesh_path
            obj_name = self.target_object_name
        else:
            raise ValueError(
                f"`obj_type` must be one of ['original', 'target'], got {obj_type!r}."
            )

        if mesh_path is None:
            raise FileNotFoundError(f"No mesh path found for {obj_type} OakInk object.")
        stem = f"{obj_name}_{obj_type}"
        export_root = self.export_dir if output_dir is None else Path(output_dir)
        return self._export_mesh_as_obj(
            mesh_path, output_dir=export_root, output_stem=stem
        )

    def get_trimesh_at_timestep(
        self, timestep, obj_type="original", mesh_type="visual"
    ):
        if obj_type == "original":
            urdf_path = self._obj_urdf_path
            motion = self.original_object_motion()
        elif obj_type == "target":
            urdf_path = self._target_obj_urdf_path
            motion = self.object_reference_motion
        else:
            raise ValueError(
                f"`obj_type` must be one of ['original', 'target'], got {obj_type!r}."
            )

        urdf_inst = URDF.load(str(urdf_path))
        joint_names = motion["joint_names"]
        obj_arti = motion["obj_arti"]
        if obj_arti.ndim == 1:
            obj_arti = obj_arti[:, None]
        jdict = dict(zip(joint_names, obj_arti[timestep]))

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
            transformed_mesh = mesh.copy()
            transformed_mesh.apply_transform(pose)
            mesh_pieces.append(transformed_mesh)

        pos = motion["obj_pos"][timestep]
        quat = motion["obj_quat"][timestep]
        base_pose = get_X(pos, quat)
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

    @property
    def obj_mesh_path(self):
        return self._obj_mesh_path

    @property
    def target_obj_mesh_path(self):
        return self._target_obj_mesh_path


# Backward compatibility for existing imports.
ArcticProcessor = OakInkProcessor

# Backward compatibility for existing imports.
ArcticProcessor = OakInkProcessor
