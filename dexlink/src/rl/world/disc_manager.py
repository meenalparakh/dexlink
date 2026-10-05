from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch
from torch.nn.functional import cosine_similarity

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import DiscObsCfg
    from dexlink.src.rl.world.env import Env
    from dexlink.src.rl.world.robot import Hand

from dexlink.src.retargeting.data.process_arctic import ArcticProcessor


@torch.jit.script
def _closest_points_indices(
    target_pcd: torch.Tensor, source_pcd: torch.Tensor
) -> torch.Tensor:
    """Return indices of the nearest target point for each source point."""
    cdist = torch.cdist(source_pcd, target_pcd)
    return torch.argmin(cdist, dim=-1)


@torch.jit.script
def _robot_rel_motion_vecs(
    lpos_t1: torch.Tensor,
    lpos_t2: torch.Tensor,
    obj_pcd_t1: torch.Tensor,
    obj_pcd_t2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute hand/object motion vectors using cached closest point indices."""
    closest_idx = _closest_points_indices(obj_pcd_t1, lpos_t1)
    closest_idx_expanded = closest_idx.unsqueeze(-1).expand(-1, -1, 3)
    obj_pts_t1 = torch.gather(obj_pcd_t1, 1, closest_idx_expanded)
    obj_pts_t2 = torch.gather(obj_pcd_t2, 1, closest_idx_expanded)
    hand_motion_vec = lpos_t2 - lpos_t1
    obj_motion_vec = obj_pts_t2 - obj_pts_t1
    return hand_motion_vec, obj_motion_vec


@torch.jit.script
def _bps_distance_vec(bps_t1: torch.Tensor, bps_t2: torch.Tensor) -> torch.Tensor:
    """Return concatenated BPS direction and normal deltas."""
    dir_t1 = bps_t1[..., :3]
    nrm_t1 = bps_t1[..., 3:]
    dir_t2 = bps_t2[..., :3]
    nrm_t2 = bps_t2[..., 3:]

    delta_dir = dir_t2 - dir_t1
    delta_nrm = nrm_t2 - nrm_t1

    return torch.cat([delta_dir, delta_nrm], dim=-1)


def _build_reference_time_gap_sampling(reference_time_gaps, total_steps: int):
    """Validate config and build per-gap timestep-pair sampling tables."""
    time_gaps = [int(gap) for gap, _ in reference_time_gaps]
    time_gap_ratios = np.array(
        [float(ratio) for _, ratio in reference_time_gaps], dtype=np.float64
    )

    assert len(time_gaps) > 0, "reference_time_gaps must not be empty."
    assert all(
        gap > 0 for gap in time_gaps
    ), f"All reference time gaps must be positive, got {time_gaps}."
    assert all(gap < total_steps for gap in time_gaps), (
        f"All reference time gaps must be smaller than total_steps={total_steps}, "
        f"got {time_gaps}."
    )
    assert np.all(
        time_gap_ratios >= 0.0
    ), "All reference time-gap ratios must be non-negative."

    ratio_sum = float(time_gap_ratios.sum())
    assert ratio_sum > 0.0, "At least one reference time-gap ratio must be positive."
    time_gap_ratios = time_gap_ratios / ratio_sum

    time_samples = []
    for gap in time_gaps:
        starts = np.arange(total_steps - gap, dtype=np.int64)
        time_samples.append(np.stack([starts, starts + gap], axis=1))

    return time_gaps, time_gap_ratios, time_samples


def _allocate_counts_from_ratios(ratios: np.ndarray, total: int) -> np.ndarray:
    """Allocate integer counts from ratios so counts sum exactly to `total`."""
    if total <= 0:
        return np.zeros_like(ratios, dtype=np.int64)

    raw_counts = ratios * total
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(total - counts.sum())
    if remainder > 0:
        frac = raw_counts - counts
        # Give extra samples to largest fractional remainders.
        extra_idx = np.argsort(-frac)[:remainder]
        counts[extra_idx] += 1
    return counts


class DiscObsManager:
    def __init__(self, cfg: DiscObsCfg, env: Env):
        self.cfg = cfg
        self.env = env
        self.physics = env.physics_manager
        self.original_demo: ArcticProcessor = env.task_manager.original_demo

        self.hand_obj_cos_dist = cfg.hand_obj_cos_dist and (
            not env.cfg.task_cfg.object_only_scene
        )
        self.hand_obj_dir_vec = cfg.hand_obj_dir_vec and (
            not env.cfg.task_cfg.object_only_scene
        )
        self.delta_lpos = cfg.delta_lpos and (not env.cfg.task_cfg.object_only_scene)
        self.delta_hand_obj_rel_bps = cfg.delta_hand_obj_rel_bps and (
            not env.cfg.task_cfg.object_only_scene
        )

        # sample time steps from the original demo
        (
            self.time_gaps,
            self.time_gap_ratios,
            self.time_samples,
        ) = _build_reference_time_gap_sampling(
            cfg.reference_time_gaps,
            env.task_manager.total_steps,
        )
        self._compute_reference_hand_pts()

        self.obs_terms = []
        self.term_dims = []
        self.term_fns = []

        if self.hand_obj_cos_dist:
            _dim = 0
            for side in ["left", "right"]:
                if side in self.env.physics_manager.scene_entities:
                    entity: Hand = self.env.physics_manager.scene_entities[side]
                    _dim += len(entity.human_joint_names)
            self.term_dims.append(_dim)
            self.obs_terms.append("hand_obj_cos_dist")
            self.term_fns.append(self._compute_cos_dist)

        if self.hand_obj_dir_vec:
            _dim = 0
            for side in ["left", "right"]:
                if side in self.env.physics_manager.scene_entities:
                    entity: Hand = self.env.physics_manager.scene_entities[side]
                    n_hand_pts = len(entity.human_joint_names)
                    _dim += n_hand_pts * 3
            self.term_dims.append(_dim)
            self.obs_terms.append("hand_obj_dir_vec")
            self.term_fns.append(self._compute_dir_vec)

        if self.delta_lpos:
            _dim = 0
            for side in ["left", "right"]:
                if side in self.env.physics_manager.scene_entities:
                    entity: Hand = self.env.physics_manager.scene_entities[side]
                    _dim += len(entity.human_joint_names) * 3
            self.term_dims.append(_dim)
            self.obs_terms.append("delta_lpos")
            self.term_fns.append(self._compute_delta_lpos)

        if self.delta_hand_obj_rel_bps:
            _dim = 0
            for side in ["left", "right"]:
                if side in self.env.physics_manager.scene_entities:
                    entity: Hand = self.env.physics_manager.scene_entities[side]
                    _dim += len(entity.human_joint_names) * 6
            self.term_dims.append(_dim)
            self.obs_terms.append("delta_hand_obj_rel_bps")
            self.term_fns.append(self._compute_delta_hand_obj_rel_bps)

    @property
    def observation_dim(self) -> int:
        return sum(self.term_dims)

    def initialize(self):
        pass

    def summary(self) -> str:
        spacing = 20
        lines = [
            "[DiscObsManager]",
            f"{'Disc Term':<{spacing}} {'Dimension':<10}",
        ]
        lines.append("-" * (spacing + 10))
        for term, dim in zip(self.obs_terms, self.term_dims):
            lines.append(f"{term:<{spacing}} {dim:<10}")
        lines.append("-" * (spacing + 10))
        lines.append(f"{'Total':<{spacing}} {self.observation_dim:<10}")
        return "\n".join(lines)

    def compute_disc_obs(self, current_named_obs, prev_named_obs) -> torch.Tensor:
        # obs provided to avoid recomputation of certain quantities
        # get direction vectors

        vecs = self.compute_robot_distance_vecs(current_named_obs, prev_named_obs)
        disc_obs = self._compute_disc_obs_from_vecs(vecs)
        return disc_obs

    def _compute_disc_obs_from_vecs(self, distance_vecs):
        obs_terms = []
        for fn in self.term_fns:
            obs_terms.append(fn(distance_vecs))

        disc_obs = torch.cat(obs_terms, dim=-1)
        return disc_obs

    def _compute_cos_dist(self, distance_vecs):
        result = []

        for side in ["left", "right"]:
            if side not in distance_vecs:
                continue
            hand_motion = distance_vecs[side]["hand_motion"]  # (B, n_hand_pts, 3)
            obj_motion = distance_vecs[side]["obj_motion"]  # (B, n_hand_pts, 3)

            cos = cosine_similarity(hand_motion, obj_motion, dim=-1)  # (B, n_hand_pts)

            result.append(cos)

        result = torch.cat(result, dim=-1)  # (B, total_n_hand_pts)
        return result

    def _compute_dir_vec(self, distance_vecs):
        result = []
        for side in ["left", "right"]:
            if side not in distance_vecs:
                continue
            hand_motion = distance_vecs[side]["hand_motion"]  # (B, n_hand_pts, 3)
            obj_motion = distance_vecs[side]["obj_motion"]  # (B, n_hand_pts, 3)

            dir_vec = hand_motion - obj_motion  # (B, n_hand_pts, 3)
            B, N, _ = dir_vec.shape
            dir_vec = dir_vec.reshape(B, N * 3)  # (B, n_hand_pts * 3)
            result.append(dir_vec)

        result = torch.cat(result, dim=-1)  # (B, total_n_hand_pts * 3)
        return result

    def _compute_delta_lpos(self, distance_vecs):
        result = []
        for side in ["left", "right"]:
            if side not in distance_vecs:
                continue
            delta_lpos = distance_vecs[side]["delta_lpos"]  # (B, n_hand_pts, 3)
            B, N, _ = delta_lpos.shape
            result.append(delta_lpos.reshape(B, N * 3))

        result = torch.cat(result, dim=-1)  # (B, total_n_hand_pts * 3)
        return result

    def _compute_delta_hand_obj_rel_bps(self, distance_vecs):
        result = []
        for side in ["left", "right"]:
            if side not in distance_vecs:
                continue
            delta_bps = distance_vecs[side][
                "delta_hand_obj_rel_bps"
            ]  # (B, n_hand_pts, 6)
            B, N, D = delta_bps.shape
            result.append(delta_bps.reshape(B, N * D))

        result = torch.cat(result, dim=-1)  # (B, total_n_hand_pts * 6)
        return result

    def _compute_bps_distance(self, bps_t1: torch.Tensor, bps_t2: torch.Tensor):
        return _bps_distance_vec(bps_t1=bps_t1, bps_t2=bps_t2)

    def named_view(self, obs: torch.Tensor) -> dict:
        """Return a dictionary mapping observation terms to their respective tensors."""
        splits = torch.split(obs, self.term_dims, dim=-1)
        named_splits = dict(zip(self.obs_terms, splits))
        return named_splits

    def expert_sampler(self, batch_size: int) -> torch.Tensor:
        """Sample expert discriminator observations from the original demo.

        Args:
            batch_size (int): Number of samples to draw.
        """
        if batch_size <= 0:
            return torch.empty((0, self.observation_dim), device=self.env.device)

        # allocate per-gap counts from configured time-gap ratios
        per_gap_counts = _allocate_counts_from_ratios(self.time_gap_ratios, batch_size)

        sampled_pairs = []
        for gap_idx, n_samples in enumerate(per_gap_counts):
            if n_samples == 0:
                continue
            gap_pairs = self.time_samples[gap_idx]
            sampled_idx = np.random.randint(0, gap_pairs.shape[0], size=n_samples)
            sampled_pairs.append(gap_pairs[sampled_idx])

        time_step_pairs = np.concatenate(sampled_pairs, axis=0)
        # no need to shuffle as order does not matter within a batch
        # np.random.shuffle(time_step_pairs)

        t1_indices = time_step_pairs[:, 0]
        t2_indices = time_step_pairs[:, 1]

        dist_vecs = self.compute_reference_distance_vecs(t1_indices, t2_indices)
        return self._compute_disc_obs_from_vecs(dist_vecs).to(self.env.device)

    def _compute_closest_pts_idx(self, target_pcd, source_pcd):
        """Compute closest points indices from source to target point clouds.

        Args:
            target_pcd (torch.Tensor): (B, M, 3) tensor of target point clouds.
            source_pcd (torch.Tensor): (B, N, 3) tensor of source point clouds.
        Returns:
            torch.Tensor: (B, N) tensor of indices of closest points in target for each
                point in source.
        """
        B, N, _ = source_pcd.shape
        M = target_pcd.shape[1]
        assert target_pcd.shape == (
            B,
            M,
            3,
        ), f"Expected target shape (B, M, 3), got {target_pcd.shape}"
        closest_pts_idx = _closest_points_indices(target_pcd, source_pcd)  # (B, N)
        assert closest_pts_idx.shape == (
            B,
            N,
        ), f"Expected closest_pts_idx shape (B, N), got {closest_pts_idx.shape}"

        # direction is not important here
        # direction is important from task perspective, where we may want to have
        # similar positioning of hand and objects, but for discriminator only the relative
        return closest_pts_idx

    def _compute_reference_hand_pts(self):
        # for now, these are just the important kpts on the robot hand
        # wrist, finger base and tip keypoints
        # ideally we want to sample more points on the hand surface
        hand_pts_indices = {}
        hand_pts = {}
        for side in ["left", "right"]:
            if side not in self.env.physics_manager.scene_entities:
                continue
            hand_entity: Hand = self.env.physics_manager.scene_entities[side]
            joint_names = hand_entity.human_joint_names
            joint_indices = [
                self.original_demo.hand_joint_map[jname] for jname in joint_names
            ]
            hand_kpts = self.original_demo.world_coord[f"joints.{side}"][
                :, joint_indices, :
            ]  # (num_frames, num_joints, 3)
            hand_pts[side] = torch.from_numpy(hand_kpts)
            hand_pts_indices[side] = joint_indices

        self.reference_hand_pt_indices = hand_pts_indices
        self.reference_hand_pts = hand_pts
        # return hand_pts

    def compute_reference_distance_vecs(self, t1, t2):
        # t1: (B,) the prev time step indices
        # t2: (B,) the next time step indices
        # t2 = t1 + 1 generally, but we want to also try different time gaps

        # for now, re use the kpts computed by the command manager
        com_mgr = self.env.commands_manager

        if "pcd" not in com_mgr.obs_command_terms:
            pcd = com_mgr._construct_pcd(append=False)
        else:
            pcd = com_mgr.obs_commands_buffer[com_mgr.obs_command_terms.index("pcd")]

        pcd = pcd.reshape(pcd.shape[0], -1, 6)  # (n, n_pts, 6)

        # choose only the positions for distance computations
        pcd_t1 = pcd[t1][..., :3].to(self.env.device)  # (B, n_pts, 3)
        pcd_t2 = pcd[t2][..., :3].to(self.env.device)  # (B, n_pts, 3)
        pcd_full_t1 = pcd[t1].to(self.env.device)  # (B, n_pts, 6)
        pcd_full_t2 = pcd[t2].to(self.env.device)  # (B, n_pts, 6)

        distance_vecs = {}

        for side in ["left", "right"]:
            if side not in self.reference_hand_pts:
                continue
            p_h_t1 = self.reference_hand_pts[side][t1].to(
                self.env.device
            )  # (B, n_hand_pts, 3)
            p_h_t2 = self.reference_hand_pts[side][t2].to(
                self.env.device
            )  # (B, n_hand_pts, 3)

            hand_motion_vec, obj_motion_vec = _robot_rel_motion_vecs(
                p_h_t1, p_h_t2, pcd_t1, pcd_t2
            )
            delta_lpos = p_h_t2 - p_h_t1

            bps_key = f"{side}_bps"
            if bps_key in com_mgr.obs_command_terms:
                ref_bps = com_mgr.obs_commands_buffer[
                    com_mgr.obs_command_terms.index(bps_key)
                ].reshape(pcd.shape[0], -1, 6)
                ref_bps_t1 = ref_bps[t1].to(self.env.device)
                ref_bps_t2 = ref_bps[t2].to(self.env.device)
            else:
                obs_mgr = self.env.observation_manager
                ref_bps_t1 = obs_mgr.compute_bps(pcd=pcd_full_t1, bps=p_h_t1)
                ref_bps_t2 = obs_mgr.compute_bps(pcd=pcd_full_t2, bps=p_h_t2)

            delta_hand_obj_rel_bps = self._compute_bps_distance(
                bps_t1=ref_bps_t1, bps_t2=ref_bps_t2
            )
            distance_vecs[side] = {
                "hand_motion": hand_motion_vec,
                "obj_motion": obj_motion_vec,
                "delta_lpos": delta_lpos,
                "delta_hand_obj_rel_bps": delta_hand_obj_rel_bps,
            }
        return distance_vecs

    def compute_robot_distance_vecs(self, current_named_obs, prev_named_obs):
        rew_mgr = self.env.reward_manager
        obs_mgr = self.env.observation_manager
        current_object_pcd = obs_mgr.get_object_pcd_from_named_obs(
            current_named_obs,
            expected_feat_dim=obs_mgr.privileged_object_pcd_dim,
        )
        previous_object_pcd = obs_mgr.get_object_pcd_from_named_obs(
            prev_named_obs,
            expected_feat_dim=obs_mgr.privileged_object_pcd_dim,
        )

        obj_pcd_t2 = current_object_pcd[..., :3]  # (B, n_obj_pts, 3)
        obj_pcd_t1 = previous_object_pcd[..., :3]

        # obj_pcd_t2 = rew_mgr._get_object_pcd(
        #     current_named_obs, normals=False
        # )  # (B, n_obj_pts, 3)
        # obj_pcd_t1 = rew_mgr._get_object_pcd(
        #     prev_named_obs, normals=False
        # )  # (B, n_obj_pts, 3)

        current_rel_bps = rew_mgr.get_obj_bps_relative_to_hand(
            named_obs=current_named_obs, use_prev_obj_pcd=False
        )
        prev_rel_bps = rew_mgr.get_obj_bps_relative_to_hand(
            named_obs=prev_named_obs, use_prev_obj_pcd=True
        )

        distance_vecs = {}
        for side in ["left", "right"]:
            obs_key = f"{side}_lpos"
            if obs_key not in current_named_obs:
                continue
            lpos_t2 = current_named_obs[obs_key]  # (B, n_hand_pts, 3)
            lpos_t1 = prev_named_obs[obs_key]  # (B, n_hand_pts, 3)
            B = lpos_t1.shape[0]
            lpos_t2 = lpos_t2.reshape(B, -1, 3)
            lpos_t1 = lpos_t1.reshape(B, -1, 3)

            n_hand_pts = len(
                self.env.physics_manager.scene_entities[side].human_joint_names
            )
            assert lpos_t1.shape == (
                B,
                n_hand_pts,
                3,
            ), f"Expected lpos_t1 shape (B, {n_hand_pts}, 3), got {lpos_t1.shape}"
            assert lpos_t2.shape == (
                B,
                n_hand_pts,
                3,
            ), f"Expected lpos_t2 shape (B, {n_hand_pts}, 3), got {lpos_t2.shape}"

            hand_motion_vec, obj_motion_vec = _robot_rel_motion_vecs(
                lpos_t1, lpos_t2, obj_pcd_t1, obj_pcd_t2
            )
            delta_lpos = lpos_t2 - lpos_t1

            if side in current_rel_bps and side in prev_rel_bps:
                bps_t2 = current_rel_bps[side].reshape(B, n_hand_pts, 6)
                bps_t1 = prev_rel_bps[side].reshape(B, n_hand_pts, 6)
            else:
                obs_mgr = self.env.observation_manager
                bps_t2 = obs_mgr.compute_bps(pcd=current_object_pcd, bps=lpos_t2)
                bps_t1 = obs_mgr.compute_bps(pcd=previous_object_pcd, bps=lpos_t1)
            delta_hand_obj_rel_bps = self._compute_bps_distance(
                bps_t1=bps_t1, bps_t2=bps_t2
            )

            distance_vecs[side] = {
                "hand_motion": hand_motion_vec,
                "obj_motion": obj_motion_vec,
                "delta_lpos": delta_lpos,
                "delta_hand_obj_rel_bps": delta_hand_obj_rel_bps,
            }
        return distance_vecs
