from __future__ import annotations

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from dtw import dtw
from einops import repeat
from urdfpy import URDF

from dexlink.src.retargeting.data.demo_loader import load_demo_processor
from dexlink.src.retargeting.data.exported_oakink_processor import ExportedOakInkProcessor
from dexlink.src.retargeting.data.process_arctic import RandomTrajProcessor
from dexlink.utils.retargeting.math_utils import (
    matrix_from_quat,
    quat_error_magnitude,
    quat_from_matrix,
)
from dexlink.utils.retargeting.mesh_utils import (
    compute_kpts_traj,
    construct_pytorch_mesh,
    sample_kpts,
)
from pytorch3d.loss import chamfer_distance

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import EnvConfig, EvalConfig
    from dexlink.src.retargeting.data.process_arctic import ArcticProcessor

SHORT_NAMES = {
    "demo": "demo",
    "original_reference": "orig",
    "target_reference": "targ",
    "command_reference": "cmd",
    "euclidean": "ed",
    "angular": "ad",
    "kpts": "kp",
    "hand_jvals_left": "hl",
    "hand_jvals_right": "hr",
    "DTW Distance": "dtw",
    "Corresp. Distance": "corr",
    "DTW Distance (Unnormalized)": "dtw_unnorm",
}


class TrajInfo:
    def __init__(self, name):
        self.name = name
        self.traj = {}
        self.current_step = 0

    def flatten_dict(self, info):
        # for nested info dicts, we flatten them into a single level
        flat = {}
        for k, v in info.items():
            if isinstance(v, dict):
                # recursively flatten the nested dict
                nested = self.flatten_dict(v)
                for nk, nv in nested.items():
                    flat[f"{k}.{nk}"] = nv
            else:
                flat[k] = v
        return flat

    def append(self, info: dict):
        flat_info = self.flatten_dict(info)
        for k, v in flat_info.items():
            if k not in self.traj:
                self.traj[k] = []
                # fill with None for past steps
                self.traj[k].extend([None] * self.current_step)

            self.traj[k].append(v)

        if len(info) > 0:
            self.current_step += 1

    def save(self, save_dir: Path | str):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.current_step == 0:
            return

        # assume no entries are None

        # values are either lists of single values or list of tensors
        # the length of each should be self.current_step
        # convert them to (self.current_step, ...)

        # we convert everything to numpy arrays for saving

        save_dict = {}
        for k, v in self.traj.items():
            if isinstance(v[0], np.ndarray):
                save_dict[k] = np.stack(v, axis=0)
                save_dict[k] = save_dict[k][:, 0, ...]  # remove env dim if exists
            elif isinstance(v[0], torch.Tensor):
                save_dict[k] = torch.stack(v, dim=0).detach().cpu().numpy()
                save_dict[k] = save_dict[k][:, 0, ...]  # remove env dim if exists
            elif isinstance(v[0], (list, int, float)):
                save_dict[k] = np.array(v)

            assert save_dict[k].shape[0] == self.current_step

        # save as npz
        np.savez_compressed(save_dir / f"{self.name}.npz", **save_dict)


class TrajInfoBatched:
    def __init__(self, name: str, num_envs: int):
        self.name = name
        self.num_envs = num_envs
        self.traj = {}
        self.current_step = 0

    def _flatten_dict(self, info: dict) -> dict:
        flat = {}
        for k, v in info.items():
            if isinstance(v, dict):
                nested = self._flatten_dict(v)
                for nk, nv in nested.items():
                    flat[f"{k}.{nk}"] = nv
            else:
                flat[k] = v
        return flat

    def append(self, info: dict):
        flat_info = self._flatten_dict(info)
        for k, v in flat_info.items():
            if k not in self.traj:
                self.traj[k] = []
                self.traj[k].extend([None] * self.current_step)
            self.traj[k].append(v)
        if len(flat_info) > 0:
            self.current_step += 1

    def _stack_series(self, series):
        if isinstance(series[0], torch.Tensor):
            return torch.stack(series, dim=0).detach().cpu().numpy()
        if isinstance(series[0], np.ndarray):
            return np.stack(series, axis=0)
        if isinstance(series[0], (list, int, float)):
            return np.array(series)
        raise ValueError(f"Unsupported type in trajectory: {type(series[0])}")

    def save(self, save_path: Path | str):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if self.current_step == 0:
            return
        save_dict = {}
        for k, v in self.traj.items():
            stacked = self._stack_series(v)
            save_dict[k] = stacked
        np.savez_compressed(save_path, **save_dict)

    def save_per_env(
        self, save_dir: Path | str, max_envs: Optional[int] = None
    ) -> list[Path]:
        """Save per-env trajectories as separate npz files. Returns list of paths."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        if self.current_step == 0:
            return paths
        stacked = {}
        for k, v in self.traj.items():
            stacked[k] = self._stack_series(v)

        n_envs = self.num_envs if max_envs is None else min(self.num_envs, max_envs)

        for env_idx in range(n_envs):
            env_dict = {}

            # batched traj info also stores done masks
            # because diff envs may finish at different times
            done_mask = stacked["done"]
            assert np.any(done_mask[:, env_idx]), "No done signal found for this env."
            done_idx = np.where(done_mask[:, env_idx])[0][0] + 1
            for k, arr in stacked.items():
                if arr.ndim >= 2 and arr.shape[1] == self.num_envs:
                    env_dict[k] = arr[:done_idx, env_idx, ...]
                else:
                    env_dict[k] = arr

            path = save_dir / f"{self.name}_env{env_idx}.npz"
            np.savez_compressed(path, **env_dict)
            paths.append(path)
        return paths


class EvalMetrics:
    def __init__(self, cfg: EvalConfig, env_cfg: EnvConfig):

        self.save_dir = Path(cfg.logdir) / "eval_metrics"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cfg = cfg
        self.relative_traj = cfg.relative_traj

        self.env_cfg = env_cfg
        scene_cfg = env_cfg.scene_cfg
        start_timestep = (
            env_cfg.scene_cfg.start_timestep + env_cfg.reset_cfg.eval_start_timestep
        )
        self.original_demo: (
            ArcticProcessor | ExportedOakInkProcessor | RandomTrajProcessor
        )

        self.original_demo = load_demo_processor(
            scene_cfg,
            start_timestep=start_timestep,
            end_timestep=env_cfg.scene_cfg.end_timestep,
            ref_motion_noise_sig=(0.0, 0.0),
        )

        self.original_obj_inst = URDF.load(self.original_demo.obj_urdf_path)
        self.original_obj_mesh = construct_pytorch_mesh(
            urdf_inst=self.original_obj_inst
        )
        self.original_kpts = sample_kpts(
            self.original_obj_mesh, num_additional_pts=cfg.num_additional_kpts
        )
        self.n_original_joints = len(self.original_obj_inst.actuated_joints)

        if self.original_demo.is_identical():
            self.target_obj_inst = self.original_obj_inst
            self.target_obj_mesh = self.original_obj_mesh
            self.target_kpts = self.original_kpts
            self.n_target_joints = self.n_original_joints
        else:
            self.target_obj_inst = URDF.load(self.original_demo.target_obj_urdf_path)
            self.target_obj_mesh = construct_pytorch_mesh(
                urdf_inst=self.target_obj_inst
            )
            self.n_target_joints = len(self.target_obj_inst.actuated_joints)
            self.target_kpts = sample_kpts(
                self.target_obj_mesh, num_additional_pts=cfg.num_additional_kpts
            )

    def set_demo(self, demo_path):
        self.demo_path = demo_path
        # the demo data contains the n_env dimension at dim=1
        # choosing the first environment
        with np.load(demo_path, allow_pickle=True) as data:
            # self.traj = {k: data[k][:, 0, ...] for k in data.files}
            self.traj = {k: data[k] for k in data.files}

        print(f"Loaded demo from {demo_path}.")
        print(f"Trajectory keys: {list(self.traj.keys())}")

    def construct_relative_gt(self, pos, quat):
        ff_pose = torch.zeros(4, 4)
        ff_pose[3, 3] = 1.0
        ff_pose[:3, 3] = pos[0]
        ff_pose[:3, :3] = matrix_from_quat(quat[0])
        ff_pose_inv = ff_pose.inverse()

        n = len(pos)
        poses = torch.zeros(n, 4, 4)
        poses[:, 3, 3] = 1.0
        poses[:, :3, 3] = pos
        poses[:, :3, :3] = matrix_from_quat(quat)

        relative_poses = ff_pose_inv[None, ...] @ poses
        pos = relative_poses[:, :3, 3]
        quat = quat_from_matrix(relative_poses[:, :3, :3])
        return pos.to(self.device), quat.to(self.device)

    def compute_kpts_traj(self, obj="demo"):
        pos = self.get_gt(obj, "pos")
        quat = self.get_gt(obj, "quat")
        if self.relative_traj:
            pos, quat = self.construct_relative_gt(pos, quat)
        arti = self.get_gt(obj, "arti")

        obj_inst = (
            self.original_obj_inst
            if obj == "original_reference"
            else self.target_obj_inst
        )
        kpts = self.original_kpts if obj == "original_reference" else self.target_kpts
        jnames = [j.name for j in obj_inst.joints if j.joint_type != "fixed"]

        pcd_time_series = compute_kpts_traj(
            kpts, pos, quat, arti, obj_inst=obj_inst, jnames=jnames
        )
        return pcd_time_series

    def get_gt(self, obj: str = "demo", attr: str = "pos"):
        # obj is in ['demo', 'original_reference', 'target_reference']
        # in general demo/target_reference and original_reference should not be compared
        # except for kpts case, which is independent of the local object frame
        # pos, quat, arti etc need a reference frame, but kpts are just pts

        if attr == "kpts":
            return self.compute_kpts_traj(obj)
        else:
            assert attr in ["pos", "quat", "arti"]
        if obj == "demo":
            key = f"object.{attr}"
            result = self.traj.get(key)
            if result is None and attr == "arti":
                n_frames = self.traj["object.pos"].shape[0]
                result = np.zeros((n_frames, 0), dtype=np.float32)
            elif result is not None:
                _d = result.shape[-1]
                if _d == 0:
                    result = result.reshape(result.shape[0], 0)
                else:
                    result = result.reshape(-1, _d)
        elif obj == "original_reference":
            ref_motion = self.original_demo.original_object_motion()
            key = f"obj_{attr}"
            result = ref_motion.get(key) if attr == "arti" else ref_motion[key]
        elif obj == "target_reference":
            ref_motion = self.original_demo.object_reference_motion
            key = f"obj_{attr}"
            result = ref_motion.get(key) if attr == "arti" else ref_motion[key]
        else:
            raise ValueError(f"Unknown object type: {obj}")

        if result is None and attr == "arti":
            if obj == "demo":
                n_frames = self.traj["object.pos"].shape[0]
            elif obj == "original_reference":
                n_frames = self.original_demo.original_object_motion()["obj_pos"].shape[
                    0
                ]
            else:  # target_reference
                n_frames = self.original_demo.object_reference_motion["obj_pos"].shape[
                    0
                ]
            result = np.zeros((n_frames, 0), dtype=np.float32)

        return torch.from_numpy(result).to(self.device)

    def _batched_chamfer_distance(
        self, kptsA: torch.Tensor, kptsB: torch.Tensor
    ) -> torch.Tensor:
        """Compute Chamfer distance for every (frame_A, frame_B) pair without OOM."""
        n_rows = kptsA.shape[0]
        m_cols = kptsB.shape[0]
        dist_matrix = torch.empty(n_rows, m_cols, device=self.device)

        max_pairs = max(1, getattr(self.cfg, "chamfer_pairs_per_batch", 512))
        row_chunk_size = max(1, getattr(self.cfg, "chamfer_row_chunk_size", 4))
        row_chunk_size = min(row_chunk_size, max_pairs)

        for row_start in range(0, n_rows, row_chunk_size):
            row_end = min(row_start + row_chunk_size, n_rows)
            row_chunk = kptsA[row_start:row_end]
            rows = row_end - row_start
            max_cols_for_chunk = max(1, max_pairs // rows)

            for col_start in range(0, m_cols, max_cols_for_chunk):
                col_end = min(col_start + max_cols_for_chunk, m_cols)
                col_chunk = kptsB[col_start:col_end]
                cols = col_end - col_start

                pair_kptsA = repeat(row_chunk, "r p d -> (r c) p d", c=cols)
                pair_kptsB = repeat(col_chunk, "c p d -> (r c) p d", r=rows)
                chunk_dist = chamfer_distance(
                    pair_kptsA, pair_kptsB, batch_reduction=None
                )[0]
                dist_matrix[row_start:row_end, col_start:col_end] = chunk_dist.view(
                    rows, cols
                )

        return dist_matrix

    def _compute_distance_matrix(
        self, motion_A: str, motion_B: str, attr: str, metric="euclidean"
    ) -> Optional[torch.Tensor]:
        # motion_A and motion_B belong to [demo, original_reference, target_reference]
        # attr is one of ['pos', 'quat', 'arti', 'kpts']
        if attr in ["hand_cfg", "hand_obj_rel"]:
            return self._compute_eval_feature_distance_matrix(
                motion_A, motion_B, attr, metric
            )
        if attr.startswith("hand_jvals_"):
            return self._compute_hand_jvals_distance_matrix(
                motion_A, motion_B, attr, metric
            )

        assert motion_A in ["demo", "original_reference", "target_reference"]
        assert motion_B in ["demo", "original_reference", "target_reference"]
        assert attr in ["pos", "quat", "arti", "kpts"]

        posA = self.get_gt(motion_A, "pos")
        quatA = self.get_gt(motion_A, "quat")
        posB = self.get_gt(motion_B, "pos")
        quatB = self.get_gt(motion_B, "quat")

        n_rows = posA.shape[0]
        m_cols = posB.shape[0]

        with torch.no_grad():

            if self.relative_traj:
                posA, quatA = self.construct_relative_gt(posA, quatA)
                posB, quatB = self.construct_relative_gt(posB, quatB)

            if (attr, metric) == ("pos", "euclidean"):
                dist_matrix = posA[:, None, :] - posB[None, :, :]
                dist_matrix = dist_matrix.norm(dim=-1)

            elif (attr, metric) == ("quat", "angular"):
                _quat_A = repeat(quatA, "n d -> n m d", m=m_cols)
                _quat_B = repeat(quatB, "m d -> n m d", n=n_rows)
                dist_matrix = quat_error_magnitude(_quat_A, _quat_B)

            elif (attr, metric) == ("arti", "euclidean"):
                artiA = self.get_gt(motion_A, "arti")
                artiB = self.get_gt(motion_B, "arti")
                _arti_A = repeat(artiA, "n d -> n m d", m=m_cols)
                _arti_B = repeat(artiB, "m d -> n m d", n=n_rows)
                dist_matrix = (_arti_A - _arti_B).norm(dim=-1)

            elif (attr, metric) == ("kpts", "euclidean"):
                kptsA = self.get_gt(motion_A, "kpts")
                kptsB = self.get_gt(motion_B, "kpts")
                n_pts_A, n_pts_B = kptsA.shape[1], kptsB.shape[1]
                if n_pts_A != n_pts_B:
                    return None
                # assert (
                #     n_pts_A == n_pts_B
                # ), f"Keypoint trajectories must have the same number of points. Comparing: {motion_A} and {motion_B}."
                # _kptsA: (t1, 1, n_pts, 3)
                # _kptsB: (1, t2, n_pts, 3)
                _kptsA = kptsA.unsqueeze(1)
                _kptsB = kptsB.unsqueeze(0)
                dist_matrix = torch.norm(_kptsA - _kptsB, dim=-1).mean(dim=-1)

            elif (attr, metric) == ("kpts", "chamfer"):
                kptsA = self.get_gt(motion_A, "kpts")
                kptsB = self.get_gt(motion_B, "kpts")
                dist_matrix = self._batched_chamfer_distance(kptsA, kptsB)
            else:
                raise ValueError(
                    f"Unknown attribute/metric combination: {attr}, {metric}"
                )

            assert dist_matrix.shape == (
                n_rows,
                m_cols,
            ), f"Distance matrix shape mismatch: {dist_matrix.shape} vs ({n_rows}, {m_cols})"

        return dist_matrix

    def _get_eval_feature_traj(self, motion: str, attr: str) -> Optional[torch.Tensor]:
        if motion not in ["demo", "command_reference"]:
            return None
        field = "rollout" if motion == "demo" else "reference"

        sides = ["left", "right"]
        base_attr = attr
        if attr.endswith("_left"):
            base_attr = attr[: -len("_left")]
            sides = ["left"]
        elif attr.endswith("_right"):
            base_attr = attr[: -len("_right")]
            sides = ["right"]

        chunks = []
        for side in sides:
            key = f"eval_metrics.{base_attr}.{side}.{field}"
            arr = self.traj.get(key)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr[:, None]
            chunks.append(
                torch.from_numpy(arr.astype(np.float32, copy=False)).to(self.device)
            )

        if not chunks:
            return None
        return torch.cat(chunks, dim=-1)

    def _compute_eval_feature_distance_matrix(
        self, motion_A: str, motion_B: str, attr: str, metric: str
    ) -> Optional[torch.Tensor]:
        if metric != "euclidean":
            raise ValueError(f"Unknown metric for {attr}: {metric}")

        featA = self._get_eval_feature_traj(motion_A, attr)
        featB = self._get_eval_feature_traj(motion_B, attr)
        if featA is None or featB is None:
            return None
        if featA.shape[-1] != featB.shape[-1]:
            return None
        return torch.cdist(featA, featB, p=2)

    def _get_hand_jvals_traj(self, motion: str, side: str) -> Optional[torch.Tensor]:
        if side not in ("left", "right"):
            raise ValueError(f"Unknown side for hand jvals: {side}")

        if motion == "demo":
            key = f"eval_metrics.hand_jvals.{side}.rollout"
            arr = self.traj.get(key)
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=np.float32)
        elif motion in ("target_reference", "command_reference"):
            retgt = getattr(self.original_demo, "retargeted_motion", None)
            if not isinstance(retgt, dict):
                return None
            key = f"{side}_jvals"
            arr = retgt.get(key)
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=np.float32)
        else:
            return None

        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            return None
        return torch.from_numpy(arr).to(self.device)

    def _compute_hand_jvals_distance_matrix(
        self, motion_A: str, motion_B: str, attr: str, metric: str
    ) -> Optional[torch.Tensor]:
        if metric != "euclidean":
            raise ValueError(f"Unknown metric for {attr}: {metric}")

        side = "left" if attr.endswith("_left") else "right"
        trajA = self._get_hand_jvals_traj(motion_A, side)
        trajB = self._get_hand_jvals_traj(motion_B, side)
        if trajA is None or trajB is None:
            return None
        if trajA.shape[-1] != trajB.shape[-1]:
            return None
        return torch.cdist(trajA, trajB, p=2)

    def compute_distance(self, motion_A: str, motion_B: str, attr: str, metric: str):
        # in DTW: motion_A is query, and motion_B is reference
        # but in here, we think of query being the reference and try to
        # find it in the recreated/demo motion.

        dist_matrix = self._compute_distance_matrix(motion_A, motion_B, attr, metric)
        if dist_matrix is None:
            return None, None, None
        alignment = dtw(
            dist_matrix.detach().cpu().double(),
            # open_begin=True,
            # step_pattern="asymmetric"
        )

        plt.plot(alignment.index1, alignment.index2)
        plt.xlabel("Frame in Motion A")
        plt.ylabel("Frame in Motion B")
        plt.title(f"DTW Alignment: {motion_A} vs {motion_B} ({attr}, {metric})")
        # plt.show()
        plt.savefig(
            self.save_dir
            / f"dtw_{SHORT_NAMES.get(motion_A, motion_A)}_{SHORT_NAMES.get(motion_B, motion_B)}_{attr}_{metric}.png"
        )

        # empty the plot
        plt.clf()

        # save heatmap for dist_matrix
        ax = sns.heatmap(dist_matrix.cpu().numpy())
        ax.set(xlabel=f"Frame in {motion_B}", ylabel=f"Frame in {motion_A}")
        # save heatmap
        plt.savefig(
            self.save_dir
            / f"heatmap_{SHORT_NAMES.get(motion_A, motion_A)}_{SHORT_NAMES.get(motion_B, motion_B)}_{attr}_{metric}.png"
        )
        plt.clf()

        # Compute the mean distance of frame-aligned pairs on the shared diagonal.
        n = min(dist_matrix.shape[0], dist_matrix.shape[1])
        diagonal = torch.diagonal(dist_matrix[:n, :n])
        corresponding_dist = diagonal.mean()

        return (
            float(alignment.normalizedDistance),
            float(corresponding_dist),
            float(alignment.distance / self.original_demo.n_frames),
        )

    def compute_metrics(
        self,
        only_demo=False,
        demo_name: str = "demo",
        skip_demo=False,
        save_pd=True,
    ):

        # list of all comparisons with different attributes and metrics and reference and query motions
        # (attr, metric, motion_A, motion_B)
        compare = [
            # pos
            (
                "pos",
                "euclidean",
                "original_reference",
                "original_reference",
            ),  # for sanity check, should be 0
            ("pos", "euclidean", "original_reference", "target_reference"),
            ("pos", "euclidean", "target_reference", "demo"),
            # quat
            (
                "quat",
                "angular",
                "original_reference",
                "original_reference",
            ),  # for sanity check, should be 0
            ("quat", "angular", "original_reference", "target_reference"),
            (
                "quat",
                "angular",
                "target_reference",
                "target_reference",
            ),  # for sanity check, should be 0
            ("quat", "angular", "target_reference", "demo"),
            # arti
            ("arti", "euclidean", "target_reference", "demo"),
            # kpts chamfer
            (
                "kpts",
                "chamfer",
                "original_reference",
                "original_reference",
            ),  # for sanity check, should be 0
            ("kpts", "chamfer", "original_reference", "target_reference"),
            (
                "kpts",
                "chamfer",
                "original_reference",
                "demo",
            ),  # for sanity check, should be 0
            ("kpts", "chamfer", "target_reference", "demo"),
            # kpts euclidean
            (
                "kpts",
                "euclidean",
                "original_reference",
                "original_reference",
            ),  # for sanity check, should be 0
            ("kpts", "euclidean", "original_reference", "target_reference"),
            (
                "kpts",
                "euclidean",
                "original_reference",
                "demo",
            ),  # for sanity check, should be 0
            ("kpts", "euclidean", "target_reference", "demo"),
            ("hand_cfg", "euclidean", "command_reference", "demo"),
            ("hand_obj_rel", "euclidean", "command_reference", "demo"),
            ("hand_jvals_left", "euclidean", "target_reference", "demo"),
            ("hand_jvals_right", "euclidean", "target_reference", "demo"),
        ]

        # create one dataframe with all the fields
        rows_list = []
        for attr, metric, motion_A, motion_B in compare:
            # Rigid objects have no articulation DOFs; skip articulation metrics entirely.
            if attr == "arti" and self.n_target_joints == 0:
                continue

            # skip the original_reference ones
            if "original_reference" in (motion_A, motion_B):
                continue

            if only_demo:
                if motion_A != "demo" and motion_B != "demo":
                    continue

            if skip_demo:
                if "demo" in (motion_A, motion_B):
                    continue

            dtw_dist, corres_dist, dtw_dist_unnormalized = self.compute_distance(
                motion_A, motion_B, attr, metric
            )
            if (
                dtw_dist is None
                and corres_dist is None
                and dtw_dist_unnormalized is None
            ):
                continue
            rows_list.append(
                {
                    "Attr": attr,
                    "Distance Metric": metric,
                    "Motion_A": motion_A if motion_A != "demo" else demo_name,
                    "Motion_B": motion_B if motion_B != "demo" else demo_name,
                    "DTW Distance": dtw_dist,
                    "Corresp. Distance": corres_dist,
                    "DTW Distance (Unnormalized)": dtw_dist_unnormalized,
                }
            )

        df = pd.DataFrame(rows_list)

        # assert whether all columns form [property fields + value fields]
        expected_columns = self.property_fields + self.value_fields
        assert set(df.columns) == set(
            expected_columns
        ), f"DataFrame columns mismatch: {df.columns} vs {expected_columns}"

        if save_pd:
            # dump to csv
            df.to_csv(
                self.save_dir / "eval_results" / f"{demo_name}_metrics.csv", index=False
            )

        else:
            print(df)

        return df

    @property
    def property_fields(self):
        return ["Attr", "Distance Metric", "Motion_A", "Motion_B"]

    @property
    def value_fields(self):
        return ["DTW Distance", "Corresp. Distance", "DTW Distance (Unnormalized)"]
