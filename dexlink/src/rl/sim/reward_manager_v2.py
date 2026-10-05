from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from dexlink.src.rl.sim.contact_info import get_filtered_contacts
from dexlink.src.retargeting.data.demo_loader import is_oakink_dataset
from dexlink.utils.retargeting.error_functions import angle_error
from dexlink.utils.retargeting.math_utils import check_shape

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import RewardCfgV2
    from dexlink.src.rl.sim.env import Env
    from dexlink.src.rl.sim.robot import Hand


def compute_pair_vec(pairs, lpos_dict):
    # lpos dict: joint_name -> (N, 3) tensor
    # N could be num_envs, or timesteps in reference motino

    vectors = []
    for joint1, joint2, mode in pairs:
        pos1 = lpos_dict[joint1]
        pos2 = lpos_dict[joint2]
        vec = pos2 - pos1
        if mode == "normalize":
            vec = vec / (torch.norm(vec, dim=-1, keepdim=True) + 1e-8)
        vectors.append(vec)
    return torch.stack(vectors, dim=1)  # (N, num_pairs, 3)


def compute_triplet_vec(triplets, lpos_dict):
    triplet_vecs = []
    for joint1, joint2, joint3, mode in triplets:
        lpos1 = lpos_dict[joint1]
        lpos2 = lpos_dict[joint2]
        lpos3 = lpos_dict[joint3]
        vec1 = lpos2 - lpos1
        vec2 = lpos3 - lpos1
        cross_vec = torch.cross(vec1, vec2, dim=-1)

        if mode == "normalize":
            cross_vec = cross_vec / (torch.norm(cross_vec, dim=-1, keepdim=True) + 1e-8)

        triplet_vecs.append(cross_vec)
    return torch.stack(triplet_vecs, dim=1)  # (N, num_triplets, 3)


class RewardCommonComputation:
    def __init__(self, cfg: RewardCfgV2, env: Env):
        self.cfg = cfg
        self.env = env
        self.discriminator = None
        self.log_metrics = cfg.log_metrics

        object_only = self.env.cfg.scene_cfg.object_only_scene
        hand_actions = self.env.cfg.action_cfg.control_hands
        self.has_hand_cfg = (
            (cfg.hand_cfg_weight > 0.0) and (not object_only) and hand_actions
        )
        self.has_hand_obj_rel = (
            (cfg.hand_obj_rel_weight > 0.0)
            and (not object_only)
            and hand_actions
            and self.env.cfg.commands_cfg.use_rel_bps
        )
        self.has_maniptrans = (
            (cfg.maniptrans_reward_weight > 0.0)
            and (not object_only)
            and hand_actions
            and self.env.cfg.commands_cfg.use_rel_bps
        )
        self.has_task_dense = cfg.task_weight > 0.0 and cfg.dense_task_weight > 0.0

        if self.has_hand_cfg:
            self.pairs_to_optimize = [
                ("wrist", "thumb_tip", "normalize"),
                ("wrist", "index_tip", "normalize"),
                ("wrist", "middle_tip", "normalize"),
                ("wrist", "ring_tip", "normalize"),
                ("wrist", "little_tip", "normalize"),
                ("thumb_base", "thumb_tip", "normalize"),
                ("index_base", "index_tip", "normalize"),
                ("middle_base", "middle_tip", "normalize"),
                ("ring_base", "ring_tip", "normalize"),
                ("little_base", "little_tip", "normalize"),
                ("thumb_tip", "index_tip", "unnormalize"),
                ("thumb_tip", "middle_tip", "unnormalize"),
                ("thumb_tip", "ring_tip", "unnormalize"),
                ("thumb_tip", "little_tip", "unnormalize"),
            ]
            self.triplets_to_optimize = [
                ("wrist", "thumb_base", "little_base", "normalize"),
            ]
            self.pairs_normalize = np.array(
                [mode == "normalize" for _, _, mode in self.pairs_to_optimize]
            )
            self.triplets_normalize = np.array(
                [mode == "normalize" for _, _, _, mode in self.triplets_to_optimize]
            )
        else:
            self.pairs_to_optimize = []
            self.triplets_to_optimize = []
            self.pairs_normalize = np.array([], dtype=bool)
            self.triplets_normalize = np.array([], dtype=bool)

        self.reference_pair_vectors = None
        self.reference_triplet_vectors = None
        self._current_commands = None
        self._object_tracking_cache = None
        self._rel_hand_obj_cache = None
        self._current_hand_obj_rel_bps = None
        self._prev_hand_obj_rel_bps = None
        self._last_eval_feature_info: dict[str, Any] = {}

    def reset_step_cache(self) -> None:
        self._current_commands = None
        self._object_tracking_cache = None
        self._rel_hand_obj_cache = None
        self._current_hand_obj_rel_bps = None
        self._prev_hand_obj_rel_bps = None

    def reset_eval_feature_cache(self) -> None:
        self._last_eval_feature_info = {}

    def log_eval_feature(
        self, group: str, side: str, rollout: torch.Tensor, reference: torch.Tensor
    ) -> None:
        if self.env.mode != "eval":
            return
        if group not in self._last_eval_feature_info:
            self._last_eval_feature_info[group] = {}
        self._last_eval_feature_info[group][side] = {
            "rollout": rollout.detach().clone(),
            "reference": reference.detach().clone(),
        }

    def get_eval_feature_info(self) -> dict[str, Any]:
        return self._last_eval_feature_info

    def set_discriminator(self, discriminator: Any) -> None:
        self.discriminator = discriminator

    def initialize(self) -> None:
        if self.pairs_to_optimize:
            self.construct_reference_vectors()

    def stage_completion_reward_mask(self) -> torch.Tensor:
        reward_mask = torch.ones(
            self.env.num_envs, dtype=torch.float32, device=self.env.device
        )
        if not is_oakink_dataset(self.env.scene_manager.cfg.demo_dataset):
            return reward_mask

        original_demo = self.env.scene_manager.original_demo
        if not hasattr(original_demo, "n_reach_frames"):
            return reward_mask

        reach_stage = self.env.command_time_step < int(original_demo.n_reach_frames)
        reward_mask[reach_stage] = 0.0
        if self.log_metrics:
            self.env.extras["log/reach_stage_command"] = reach_stage.to(torch.float32)
            self.env.extras["log/lift_stage_command"] = (~reach_stage).to(torch.float32)
        return reward_mask

    def current_reach_phase_mask(self) -> torch.Tensor:
        reach_phase = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        if not is_oakink_dataset(self.env.scene_manager.cfg.demo_dataset):
            return reach_phase

        original_demo = self.env.scene_manager.original_demo
        if not hasattr(original_demo, "n_reach_frames"):
            return reach_phase

        current_reference_timestep = self.env.start_timestep.to(
            self.env.device
        ) + self.env.timesteps.to(dtype=torch.long)
        reach_phase = current_reference_timestep < int(original_demo.n_reach_frames)
        if self.log_metrics:
            self.env.extras["log/current_reach_phase"] = reach_phase.to(torch.float32)
        return reach_phase

    def _reference_hand_pos_dict(self, side: str) -> dict[str, torch.Tensor]:
        original_demo = self.env.scene_manager.original_demo
        device = self.env.device

        if not self.cfg.use_retargeted_motion:
            if side == "left":
                ref_pos = original_demo.relevant_left_joint_pos
            elif side == "right":
                ref_pos = original_demo.relevant_right_joint_pos
            else:
                raise ValueError(f"Unknown side: {side}")
            return {
                k: torch.as_tensor(v, device=device).float() for k, v in ref_pos.items()
            }

        hand: Hand = self.env.scene_manager.scene_entities[side]
        ref_motion = original_demo.retargeted_motion
        assert ref_motion is not None
        side_link_pos = ref_motion.get(f"{side}_link_pos", None)
        if side_link_pos is None:
            raise ValueError(
                f"Retargeted motion missing {side}_link_pos for reward reference vectors."
            )

        return {
            human_name: torch.as_tensor(
                side_link_pos[link_name], device=device, dtype=torch.float32
            )
            for human_name, link_name in zip(
                hand.human_joint_names, hand.human_mapped_link_names
            )
        }

    def construct_reference_vectors(self) -> None:
        original_demo = self.env.scene_manager.original_demo
        self.reference_pair_vectors = {}
        self.reference_triplet_vectors = {}

        for side in ["left", "right"]:
            if side not in self.env.scene_manager.scene_entities:
                continue

            ref_pos_dict = self._reference_hand_pos_dict(side)
            pair_vectors = compute_pair_vec(self.pairs_to_optimize, ref_pos_dict)
            triplet_vectors = compute_triplet_vec(
                self.triplets_to_optimize, ref_pos_dict
            )

            check_shape(
                pair_vectors, (original_demo.n_frames, len(self.pairs_to_optimize), 3)
            )
            check_shape(
                triplet_vectors,
                (original_demo.n_frames, len(self.triplets_to_optimize), 3),
            )

            self.reference_pair_vectors[side] = pair_vectors
            self.reference_triplet_vectors[side] = triplet_vectors

    def get_commands(self, named_obs: dict[str, torch.Tensor]):
        if self._current_commands is None:
            command = self.env.commands_manager.get_commands_from_obs(
                named_obs=named_obs
            )
            if command is not None:
                self._current_commands = command
        return self._current_commands

    def get_obj_bps_relative_to_hand(
        self, named_obs: dict[str, torch.Tensor], use_prev_obj_pcd: bool = False
    ) -> dict[str, torch.Tensor]:
        if use_prev_obj_pcd:
            if self._prev_hand_obj_rel_bps is None:
                self._prev_hand_obj_rel_bps = self.compute_obj_bps_relative_to_hand(
                    named_obs=named_obs, obj_pcd=self.env._prev_obj_pcd
                )
            return self._prev_hand_obj_rel_bps

        if self._current_hand_obj_rel_bps is None:
            self._current_hand_obj_rel_bps = self.compute_obj_bps_relative_to_hand(
                named_obs=named_obs, obj_pcd=self.env._current_obj_pcd
            )
        return self._current_hand_obj_rel_bps

    def compute_obj_bps_relative_to_hand(
        self, named_obs: dict[str, torch.Tensor], obj_pcd=None
    ) -> dict[str, torch.Tensor]:
        obs_mgr = self.env.observation_manager
        if obj_pcd is None:
            obj_pcd = self.env._current_obj_pcd

        bps = {}
        for side in ["left", "right"]:
            obs_key = f"{side}_lpos"
            if obs_key not in named_obs:
                continue
            hand: Hand = self.env.scene_manager.scene_entities[side]
            n_links = len(hand.human_joint_names)
            lpos = named_obs[obs_key].view(self.env.num_envs, n_links, 3)
            result = obs_mgr.compute_bps(pcd=obj_pcd, bps=lpos)
            bps[side] = result.reshape(result.shape[0], -1)
        return bps

    def hand_config_matching_reward(self, named_obs, **kwargs):
        if not self.pairs_to_optimize:
            return torch.zeros_like(self.env.timesteps, dtype=torch.float32).to(
                self.env.device
            )

        if (
            self.reference_pair_vectors is None
            or self.reference_triplet_vectors is None
        ):
            self.construct_reference_vectors()

        command_ts = self.env.command_time_step.to(dtype=torch.long)
        rewards = torch.zeros_like(self.env.timesteps, dtype=torch.float32).to(
            self.env.device
        )
        n_hands = 0

        for side in ["left", "right"]:
            obs_key = f"{side}_lpos"
            if obs_key not in named_obs:
                continue

            hand: Hand = self.env.scene_manager.scene_entities[side]
            human_joint_names = hand.human_joint_names
            lpos = named_obs[obs_key].view(self.env.num_envs, len(human_joint_names), 3)
            lpos_dict = {k: lpos[:, i] for i, k in enumerate(human_joint_names)}

            pair_vectors = compute_pair_vec(self.pairs_to_optimize, lpos_dict)
            triplet_vectors = compute_triplet_vec(self.triplets_to_optimize, lpos_dict)

            ref_pair_vecs = self.reference_pair_vectors[side][command_ts].to(
                self.env.device
            )
            ref_triplet_vecs = self.reference_triplet_vectors[side][command_ts].to(
                self.env.device
            )

            self.log_eval_feature(
                "hand_cfg",
                side,
                rollout=torch.cat(
                    [
                        pair_vectors.reshape(self.env.num_envs, -1),
                        triplet_vectors.reshape(self.env.num_envs, -1),
                    ],
                    dim=-1,
                ),
                reference=torch.cat(
                    [
                        ref_pair_vecs.reshape(self.env.num_envs, -1),
                        ref_triplet_vecs.reshape(self.env.num_envs, -1),
                    ],
                    dim=-1,
                ),
            )

            norm_reward = torch.exp(
                -float(self.cfg.hand_cfg_norm_scale)
                * (
                    1.0
                    - F.cosine_similarity(
                        pair_vectors[:, self.pairs_normalize],
                        ref_pair_vecs[:, self.pairs_normalize],
                        dim=-1,
                    )
                )
            ).mean(dim=-1)
            dir_reward = torch.exp(
                -float(self.cfg.hand_cfg_dir_scale)
                * (
                    pair_vectors[:, ~self.pairs_normalize]
                    - ref_pair_vecs[:, ~self.pairs_normalize]
                )
                .abs()
                .sum(dim=-1)
            ).mean(dim=-1)
            triplet_reward = F.cosine_similarity(
                triplet_vectors,
                ref_triplet_vecs,
                dim=-1,
            ).mean(dim=-1)
            side_reward = (norm_reward + 5.0 * dir_reward + triplet_reward) / 7.0

            check_shape(norm_reward, (self.env.num_envs,))
            check_shape(dir_reward, (self.env.num_envs,))
            check_shape(triplet_reward, (self.env.num_envs,))

            rewards += side_reward
            n_hands += 1

        if n_hands > 0:
            rewards = rewards / n_hands
        return rewards

    def _hand_rel_obj_bps_reward(
        self, robot_bps: torch.Tensor, ref_bps: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        comp_displacement = robot_bps[..., :3]
        comp_normal = robot_bps[..., 3:6]
        ref_displacement = ref_bps[..., :3]
        ref_normal = ref_bps[..., 3:6]

        comp_distance_sq = torch.sum(comp_displacement**2, dim=-1) + 1e-4
        ref_distance_sq = torch.sum(ref_displacement**2, dim=-1) + 1e-4

        comp_direction = comp_displacement / torch.sqrt(comp_distance_sq).unsqueeze(-1)
        ref_direction = ref_displacement / torch.sqrt(ref_distance_sq).unsqueeze(-1)

        distance_cost = torch.abs(1.0 - comp_distance_sq / ref_distance_sq)
        direction_cost = 1.0 - torch.sum(comp_direction * ref_direction, dim=-1)
        displacement_cost = distance_cost + direction_cost

        nrm_diff = 1.0 - torch.sum(comp_normal * ref_normal, dim=-1)
        normal_cost = nrm_diff * torch.exp(-ref_distance_sq / (2.0 * (0.05**2)))

        displacement_reward = torch.exp(
            -float(self.cfg.hand_obj_displacement_scale) * displacement_cost
        )
        normal_reward = torch.exp(-float(self.cfg.hand_obj_normal_scale) * normal_cost)

        check_shape(displacement_cost, (self.env.num_envs, robot_bps.shape[1]))
        check_shape(normal_cost, (self.env.num_envs, robot_bps.shape[1]))
        check_shape(displacement_reward, (self.env.num_envs, robot_bps.shape[1]))
        check_shape(normal_reward, (self.env.num_envs, robot_bps.shape[1]))

        logs = {
            "distance_cost": distance_cost.mean(dim=-1),
            "direction_cost": direction_cost.mean(dim=-1),
            "displacement_cost": displacement_cost.mean(dim=-1),
            "normal_cost": normal_cost.mean(dim=-1),
            "displacement_reward": displacement_reward.mean(dim=-1),
            "normal_reward": normal_reward.mean(dim=-1),
        }
        hand_relative_pose_achieved = (
            (logs["distance_cost"] < self.cfg.hand_obj_rel_distance_completion_thd)
            & (logs["direction_cost"] < self.cfg.hand_obj_rel_direction_completion_thd)
            & (logs["normal_cost"] < self.cfg.hand_obj_rel_normal_completion_thd)
        )
        hand_far_away = (
            (logs["distance_cost"] > self.cfg.hand_obj_rel_distance_failure_thd)
            | (logs["direction_cost"] > self.cfg.hand_obj_rel_direction_failure_thd)
            | (logs["normal_cost"] > self.cfg.hand_obj_rel_normal_failure_thd)
        )
        hand_obj_rel_reward = displacement_reward * normal_reward
        return (
            hand_obj_rel_reward.mean(dim=-1),
            logs,
            hand_relative_pose_achieved,
            hand_far_away,
        )

    def hand_obj_relative_reward(self, named_obs, **kwargs):
        if self._rel_hand_obj_cache is not None:
            return self._rel_hand_obj_cache

        rewards = torch.zeros_like(self.env.timesteps, dtype=torch.float32).to(
            self.env.device
        )
        target_bps = self.get_obj_bps_relative_to_hand(
            named_obs=named_obs, use_prev_obj_pcd=False
        )
        if len(target_bps) == 0:
            hand_far_away = torch.zeros_like(self.env.timesteps, dtype=torch.bool).to(
                self.env.device
            )
            hand_relative_pose_achieved = torch.ones_like(
                self.env.timesteps, dtype=torch.bool
            ).to(self.env.device)
            return rewards, hand_far_away, hand_relative_pose_achieved

        commands = self.get_commands(named_obs)
        if commands is None:
            hand_far_away = torch.zeros_like(self.env.timesteps, dtype=torch.bool).to(
                self.env.device
            )
            hand_relative_pose_achieved = torch.ones_like(
                self.env.timesteps, dtype=torch.bool
            ).to(self.env.device)
            return rewards, hand_far_away, hand_relative_pose_achieved

        hand_far_away = torch.zeros_like(self.env.timesteps, dtype=torch.bool).to(
            self.env.device
        )
        hand_relative_pose_achieved = torch.ones_like(
            self.env.timesteps, dtype=torch.bool
        ).to(self.env.device)
        component_sums = {
            "distance_cost": torch.zeros_like(
                self.env.timesteps, dtype=torch.float32
            ).to(self.env.device),
            "direction_cost": torch.zeros_like(
                self.env.timesteps, dtype=torch.float32
            ).to(self.env.device),
            "displacement_cost": torch.zeros_like(
                self.env.timesteps, dtype=torch.float32
            ).to(self.env.device),
            "normal_cost": torch.zeros_like(self.env.timesteps, dtype=torch.float32).to(
                self.env.device
            ),
            "displacement_reward": torch.zeros_like(
                self.env.timesteps, dtype=torch.float32
            ).to(self.env.device),
            "normal_reward": torch.zeros_like(
                self.env.timesteps, dtype=torch.float32
            ).to(self.env.device),
        }

        named_commands = self.env.commands_manager.named_view(commands)
        n_hands = 0
        for side in target_bps:
            if is_oakink_dataset(self.env.scene_manager.cfg.demo_dataset):
                if side != self.env.scene_manager.original_demo.side:
                    continue

            reference_bps = named_commands[f"{side}_bps"].view(self.env.num_envs, -1, 6)
            robot_bps = target_bps[side].view(self.env.num_envs, -1, 6)
            self.log_eval_feature(
                "hand_obj_rel",
                side,
                rollout=robot_bps.reshape(self.env.num_envs, -1),
                reference=reference_bps.reshape(self.env.num_envs, -1),
            )
            (
                side_reward,
                component_logs,
                side_pose_achieved,
                side_far_away,
            ) = self._hand_rel_obj_bps_reward(
                robot_bps=robot_bps, ref_bps=reference_bps
            )
            rewards += side_reward
            for key, value in component_logs.items():
                component_sums[key] += value
            hand_relative_pose_achieved = torch.logical_and(
                hand_relative_pose_achieved, side_pose_achieved
            )
            hand_far_away = torch.logical_or(hand_far_away, side_far_away)
            n_hands += 1

        if n_hands > 0:
            rewards = rewards / n_hands
            for key in component_sums:
                component_sums[key] = component_sums[key] / n_hands

        hand_far_away = torch.where(
            self.current_reach_phase_mask(),
            torch.zeros_like(hand_far_away),
            hand_far_away,
        )

        if self.log_metrics:
            self.env.extras["log/cost_hand_obj_rel_distance"] = component_sums[
                "distance_cost"
            ]
            self.env.extras["log/cost_hand_obj_rel_direction"] = component_sums[
                "direction_cost"
            ]
            self.env.extras["log/cost_hand_obj_rel_displacement"] = component_sums[
                "displacement_cost"
            ]
            self.env.extras["log/cost_hand_obj_rel_normal"] = component_sums[
                "normal_cost"
            ]
            self.env.extras["log/reward_hand_obj_rel_displacement"] = component_sums[
                "displacement_reward"
            ]
            self.env.extras["log/reward_hand_obj_rel_normal"] = component_sums[
                "normal_reward"
            ]
            self.env.extras["log/hand_far_away"] = hand_far_away.to(torch.float32)
            self.env.extras["log/hand_relative_pose_achieved"] = (
                hand_relative_pose_achieved.to(torch.float32)
            )

        self._rel_hand_obj_cache = (
            rewards,
            hand_far_away,
            hand_relative_pose_achieved,
        )
        return self._rel_hand_obj_cache

    def maniptrans_reward(self, named_obs, **kwargs):
        commands = self.get_commands(named_obs)
        if commands is None:
            return torch.zeros_like(self.env.timesteps, dtype=torch.float32).to(
                self.env.device
            )

        named_commands = self.env.commands_manager.named_view(commands)
        proximity_thd = float(self.cfg.maniptrans_proximity_thd)
        lam = float(self.cfg.maniptrans_lambda)
        eps = 1e-6
        target_entity = (
            self.env.scene_manager.scene_entities["object"].object
            if self.env.cfg.obs_cfg.use_filtered_contact
            else None
        )

        rewards = torch.zeros_like(self.env.timesteps, dtype=torch.float32).to(
            self.env.device
        )
        n_hands = 0
        for side in ("left", "right"):
            bps_key = f"{side}_bps"
            if bps_key not in named_commands:
                continue
            if side not in self.env.scene_manager.scene_entities:
                continue

            hand: Hand = self.env.scene_manager.scene_entities[side]
            ref_bps = named_commands[bps_key].view(self.env.num_envs, -1, 6)
            keep_idx = [
                i
                for i, jn in enumerate(hand.human_joint_names)
                if jn.endswith("_tip") or jn.endswith("_base")
            ]
            if len(keep_idx) == 0:
                continue

            keep_idx_t = torch.as_tensor(
                keep_idx, dtype=torch.long, device=self.env.device
            )
            ref_dist_norm = torch.norm(ref_bps[..., :3], dim=-1)
            proximity_mask = (ref_dist_norm < proximity_thd).to(torch.float32)
            proximity_mask = proximity_mask[:, keep_idx_t]

            contact_info = get_filtered_contacts(
                entity_a=hand.robot,
                entity_b=target_entity,
                filter_links_a=torch.tensor(
                    hand.human_mapped_link_indices_local, dtype=torch.long
                ),
                return_link_force=True,
                return_link_pos=False,
            )
            force = contact_info["contact_force_link_a"]
            force_net = force.sum(dim=2)
            force_mag = torch.norm(force_net, dim=-1)
            force_mag = force_mag[:, keep_idx_t]

            effective_force = (force_mag * proximity_mask).sum(dim=-1)
            rewards += torch.exp(-lam / (effective_force + eps))
            n_hands += 1

        if n_hands > 0:
            rewards = rewards / n_hands
        return rewards

    def discriminator_reward(self, disc_obs, **kwargs):
        if self.discriminator is None:
            raise ValueError(
                "Discriminator reward requested but no discriminator is set."
            )
        with torch.no_grad():
            return self.discriminator.compute_rewards(disc_obs)

    def object_tracking_progress(self, named_obs, **kwargs):
        if self._object_tracking_cache is not None:
            return self._object_tracking_cache

        commands = self.get_commands(named_obs)
        if commands is None:
            empty_reward = torch.zeros_like(self.env.timesteps, dtype=torch.float32).to(
                self.env.device
            )
            object_pose_achieved = torch.zeros_like(
                self.env.timesteps, dtype=torch.bool
            ).to(self.env.device)
            early_terminate = torch.zeros_like(self.env.timesteps, dtype=torch.bool).to(
                self.env.device
            )
            self._object_tracking_cache = (
                empty_reward,
                object_pose_achieved,
                early_terminate,
            )
            return self._object_tracking_cache

        named_commands = self.env.commands_manager.named_view(commands)
        object_pose_achieved = torch.ones_like(self.env.timesteps, dtype=torch.bool).to(
            self.env.device
        )
        early_terminate = torch.zeros_like(self.env.timesteps, dtype=torch.bool).to(
            self.env.device
        )
        empty_reward = torch.zeros_like(self.env.timesteps, dtype=torch.float32).to(
            self.env.device
        )
        subtask_rewards: dict[str, torch.Tensor] = {}

        pos_error = torch.norm(
            named_obs["obj_pos"] - named_commands["gt_pos"], p=2, dim=-1
        )
        pos_completion = pos_error < self.cfg.completion_pos_thd
        object_pose_achieved = torch.logical_and(object_pose_achieved, pos_completion)
        pos_reward = torch.exp(-float(self.cfg.task_pos_scale) * pos_error)
        subtask_rewards["pos"] = pos_reward
        far_away = pos_error > self.cfg.obj_command_max_dist_pos
        early_terminate = torch.logical_or(early_terminate, far_away)
        if self.log_metrics:
            self.env.extras["log/pos_task_rew"] = pos_reward
            self.env.extras["log/pos_far_away"] = far_away.to(torch.float32)

        quat_error = angle_error(
            named_obs["obj_quat"],
            named_commands["gt_rot"],
            current_format=self.env.cfg.obs_cfg.rot_format,
            target_format=self.env.cfg.commands_cfg.rot_format,
        )
        quat_completion = quat_error < self.cfg.completion_rot_thd
        object_pose_achieved = torch.logical_and(object_pose_achieved, quat_completion)
        quat_reward = torch.exp(-float(self.cfg.task_rot_scale) * quat_error)
        subtask_rewards["rot"] = quat_reward
        far_away = quat_error > self.cfg.obj_command_max_dist_rot
        early_terminate = torch.logical_or(early_terminate, far_away)
        if self.log_metrics:
            self.env.extras["log/rot_task_rew"] = quat_reward
            self.env.extras["log/rot_far_away"] = far_away.to(torch.float32)

        if "gt_arti" in named_commands and "obj_arti" in named_obs:
            arti_error = (
                (named_obs["obj_arti"] - named_commands["gt_arti"]).abs().sum(dim=-1)
            )
            arti_completion = arti_error < self.cfg.completion_arti_thd
            object_pose_achieved = torch.logical_and(
                object_pose_achieved, arti_completion
            )
            arti_reward = torch.exp(-float(self.cfg.task_arti_scale) * arti_error)
            subtask_rewards["arti"] = arti_reward
            far_away = arti_error > self.cfg.obj_command_max_dist_arti
            early_terminate = torch.logical_or(early_terminate, far_away)
            if self.log_metrics:
                self.env.extras["log/arti_task_rew"] = arti_reward
                self.env.extras["log/arti_far_away"] = far_away.to(torch.float32)

        product_terms = [
            subtask_rewards[key]
            for key in ("pos", "rot", "arti")
            if key in subtask_rewards
        ]
        if len(product_terms) == 0:
            object_tracking_reward = empty_reward
        else:
            object_tracking_reward = product_terms[0]
            for term in product_terms[1:]:
                object_tracking_reward = object_tracking_reward * term
            if self.log_metrics:
                self.env.extras["log/object_tracking_product_rew"] = (
                    object_tracking_reward
                )

        self._object_tracking_cache = (
            object_tracking_reward,
            object_pose_achieved,
            early_terminate,
        )
        return self._object_tracking_cache


class RewardComputation:
    def __init__(self, common: RewardCommonComputation):
        self.common = common
        self.cfg = common.cfg
        self.env = common.env
        self.log_metrics = common.log_metrics
        self.reward_terms: dict[str, tuple[float, Any]] = {}

        if common.has_task_dense:
            self.reward_terms["object_tracking"] = (
                self.cfg.dense_task_weight,
                self.common.object_tracking_progress,
            )
        if common.has_hand_cfg:
            self.reward_terms["hand_cfg"] = (
                self.cfg.hand_cfg_weight,
                self.common.hand_config_matching_reward,
            )
        if common.has_hand_obj_rel:
            self.reward_terms["hand_obj_rel"] = (
                self.cfg.hand_obj_rel_weight,
                self.common.hand_obj_relative_reward,
            )
        if common.has_maniptrans:
            self.reward_terms["maniptrans"] = (
                self.cfg.maniptrans_reward_weight,
                self.common.maniptrans_reward,
            )
        if self.cfg.discriminator_weight > 0.0:
            self.reward_terms["discriminator"] = (
                self.cfg.discriminator_weight,
                self.common.discriminator_reward,
            )

        self.normalize_reward_weights()

    @property
    def uses_object_tracking(self) -> bool:
        return ("object_tracking" in self.reward_terms) or (
            self.cfg.task_completion_weight > 0.0
        )

    def normalize_reward_weights(self) -> None:
        total = sum(weight for weight, _ in self.reward_terms.values())
        if total <= 0.0:
            return
        self.reward_terms = {
            name: (weight / total, func)
            for name, (weight, func) in self.reward_terms.items()
        }

    def summary(self) -> str:
        spacing = 20
        lines = [
            "[RewardManagerV2]",
            f"{'Reward Term':<{spacing}} {'Weight':<10}",
        ]
        lines.append("-" * (spacing + 10))
        for name, (weight, _) in self.reward_terms.items():
            lines.append(f"{name:<{spacing}} {weight:<10.3f}")
        if self.cfg.task_completion_weight > 0.0:
            lines.append(
                f"{'task_completion':<{spacing}} {self.cfg.task_completion_weight:<10.3f}"
            )
        lines.append("-" * (spacing + 10))
        total_weight = sum(weight for weight, _ in self.reward_terms.values())
        lines.append(f"{'Total':<{spacing}} {total_weight:<10.3f}")
        return "\n".join(lines)

    def compute_total_reward(
        self,
        named_obs,
        disc_obs,
        action,
        prev_obs,
        actual_success: torch.Tensor,
        stage_completion: torch.Tensor,
    ) -> torch.Tensor:
        total_reward = torch.zeros_like(self.env.timesteps, dtype=torch.float32).to(
            self.env.device
        )
        for name, (weight, func) in self.reward_terms.items():
            result = func(
                named_obs=named_obs, disc_obs=disc_obs, action=action, prev_obs=prev_obs
            )
            reward_term = result[0] if isinstance(result, tuple) else result
            total_reward += weight * reward_term.float()
            if self.log_metrics:
                self.env.extras[f"log/reward_{name}"] = reward_term

        if self.cfg.task_completion_weight > 0.0:
            success_reward = actual_success.float()
            total_reward += self.cfg.task_completion_weight * success_reward
            if self.log_metrics:
                self.env.extras["log/reward_task_completion"] = success_reward
        if self.cfg.stage_completion_weight > 0.0:
            stage_reward = (
                stage_completion.float() * self.common.stage_completion_reward_mask()
            )
            total_reward += self.cfg.stage_completion_weight * stage_reward
            if self.log_metrics:
                self.env.extras["log/reward_stage_completion"] = stage_reward
        return total_reward


class RewardTerminationComputation:
    def __init__(self, common: RewardCommonComputation):
        self.common = common
        self.cfg = common.cfg
        self.env = common.env

    def object_pose_achieved_mask(self, named_obs, **kwargs) -> torch.Tensor:
        _, object_pose_achieved, _ = self.common.object_tracking_progress(
            named_obs=named_obs, **kwargs
        )
        return object_pose_achieved

    def hand_relative_pose_achieved_mask(self, named_obs, **kwargs) -> torch.Tensor:
        if not self.common.has_hand_obj_rel:
            return torch.ones_like(self.env.timesteps, dtype=torch.bool).to(
                self.env.device
            )
        _, _, hand_relative_pose_achieved = self.common.hand_obj_relative_reward(
            named_obs=named_obs, **kwargs
        )
        return hand_relative_pose_achieved

    def pose_achieved_mask(self, named_obs, disc_obs, action, prev_obs) -> torch.Tensor:
        object_pose_achieved = self.object_pose_achieved_mask(
            named_obs=named_obs,
            disc_obs=disc_obs,
            action=action,
            prev_obs=prev_obs,
        )
        hand_relative_pose_achieved = self.hand_relative_pose_achieved_mask(
            named_obs=named_obs,
            disc_obs=disc_obs,
            action=action,
            prev_obs=prev_obs,
        )
        pose_achieved = torch.logical_and(
            object_pose_achieved, hand_relative_pose_achieved
        )
        if self.common.log_metrics:
            self.env.extras["log/object_pose_achieved"] = object_pose_achieved.to(
                torch.float32
            )
            self.env.extras["log/pose_achieved"] = pose_achieved.to(torch.float32)
        return pose_achieved

    def far_away_mask(self, named_obs, disc_obs, action, prev_obs) -> torch.Tensor:
        hand_far_away = torch.zeros_like(self.env.timesteps, dtype=torch.bool).to(
            self.env.device
        )
        if self.common.has_hand_obj_rel:
            _, hand_far_away, _ = self.common.hand_obj_relative_reward(
                named_obs=named_obs,
                disc_obs=disc_obs,
                action=action,
                prev_obs=prev_obs,
            )

        _, _, object_far_away = self.common.object_tracking_progress(
            named_obs=named_obs,
            disc_obs=disc_obs,
            action=action,
            prev_obs=prev_obs,
        )
        far_away = torch.logical_or(hand_far_away, object_far_away)
        if self.common.log_metrics:
            self.env.extras["log/far_away"] = far_away.to(torch.float32)
        return far_away

    def actual_success_mask(self, pose_achieved: torch.Tensor) -> torch.Tensor:
        actual_success = torch.logical_and(
            pose_achieved,
            self.env.command_time_step == (self.env.scene_manager.total_steps - 2),
        )
        if self.common.log_metrics:
            self.env.extras["log/actual_success"] = actual_success.to(torch.float32)
        return actual_success

    def combine(
        self,
        actual_success: torch.Tensor | None = None,
        far_away: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if actual_success is None:
            actual_success = torch.zeros_like(self.env.timesteps, dtype=torch.bool).to(
                self.env.device
            )
        if far_away is None:
            far_away = torch.zeros_like(self.env.timesteps, dtype=torch.bool).to(
                self.env.device
            )

        termination = actual_success.clone()
        if self.cfg.early_terminate and (self.env.mode == "train"):
            termination = torch.logical_or(termination, far_away)
        if is_oakink_dataset(self.env.scene_manager.cfg.demo_dataset):
            termination = torch.logical_or(actual_success, far_away)
        return termination


class RewardTruncationComputation:
    def __init__(self, common: RewardCommonComputation):
        self.common = common
        self.cfg = common.cfg
        self.env = common.env

        max_command_horizon = self.env.cfg.obs_cfg.command_steps_range[1]
        self.first_stage_horizon = (
            self.cfg.first_command_horizon_scale * max_command_horizon
        )
        self.later_stage_horizon = (
            self.cfg.later_command_horizon_scale * max_command_horizon
        )
        self.use_custom_horizon = self.cfg.use_custom_horizon

    def check_stage_limit(self) -> torch.Tensor:
        if self.use_custom_horizon:
            stage_exceeded = self.env.timesteps > self.env.n_step_future_cmd
            return stage_exceeded
        else:
            first_stage_exceeded = self.env.timesteps > self.first_stage_horizon
            later_stage_exceeded = self.env.timesteps > self.later_stage_horizon
            return torch.where(
                self.env.is_first_stage,
                first_stage_exceeded,
                later_stage_exceeded,
            )

    def update_commands_and_termination(
        self, update_commands: torch.Tensor
    ) -> torch.Tensor:
        env_ids_to_update_command = torch.where(
            torch.logical_and(
                update_commands,
                self.env.command_time_step < (self.env.scene_manager.total_steps - 1),
            )
        )[0]
        truncation = update_commands & (
            self.env.command_time_step == (self.env.scene_manager.total_steps - 1)
        )

        self.env.is_first_stage[env_ids_to_update_command] = False
        next_commands_diff = torch.from_numpy(
            np.random.randint(
                self.env.cfg.obs_cfg.command_steps_range[0],
                self.env.cfg.obs_cfg.command_steps_range[1],
                size=len(env_ids_to_update_command),
            )
        ).to(self.env.device)
        next_commands = (
            self.env.command_time_step[env_ids_to_update_command] + next_commands_diff
        )
        next_commands = self.env.clamp_command_timesteps(next_commands)
        old_command_steps = self.env.command_time_step[env_ids_to_update_command]
        self.env.n_step_future_cmd[env_ids_to_update_command] = (
            next_commands - old_command_steps
        )
        self.env.command_time_step[env_ids_to_update_command] = next_commands

        if self.cfg.task_weight > 0.0:
            self.env.commands_manager.reset_obs_commands(env_ids_to_update_command)

        self.env.timesteps[env_ids_to_update_command] = 0
        return truncation


class RewardManager:
    def __init__(self, cfg: RewardCfgV2, env: Env):
        self.cfg = cfg
        self.env = env
        self.common = RewardCommonComputation(cfg, env)
        self.reward_computation = RewardComputation(self.common)
        self.termination_computation = RewardTerminationComputation(self.common)
        self.truncation_computation = RewardTruncationComputation(self.common)

    def __getattr__(self, name: str):
        for component_name in (
            "common",
            "reward_computation",
            "termination_computation",
            "truncation_computation",
        ):
            component = self.__dict__.get(component_name)
            if component is not None and hasattr(component, name):
                return getattr(component, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def summary(self) -> str:
        return self.reward_computation.summary()

    def initialize(self) -> None:
        self.common.initialize()

    def set_discriminator(self, discriminator: Any) -> None:
        self.common.set_discriminator(discriminator)

    def get_eval_feature_info(self) -> dict[str, Any]:
        return self.common.get_eval_feature_info()

    def compute_rewards(self, next_named_obs, disc_obs, action, prev_obs):
        self.common.reset_step_cache()
        self.common.reset_eval_feature_cache()
        named_obs = next_named_obs

        # this is per command stage pose achievement, not overall episode success
        pose_achieved = self.termination_computation.pose_achieved_mask(
            named_obs=named_obs,
            disc_obs=disc_obs,
            action=action,
            prev_obs=prev_obs,
        )
        actual_success = self.termination_computation.actual_success_mask(pose_achieved)

        need_command_update = torch.logical_or(
            self.truncation_computation.check_stage_limit(), pose_achieved
        )
        if need_command_update.any():
            truncation = self.truncation_computation.update_commands_and_termination(
                need_command_update
            )
        else:
            truncation = torch.zeros_like(
                self.env.timesteps, dtype=torch.bool, device=self.env.device
            )

        total_reward = self.reward_computation.compute_total_reward(
            named_obs=named_obs,
            disc_obs=disc_obs,
            action=action,
            prev_obs=prev_obs,
            actual_success=actual_success,
            stage_completion=pose_achieved,
        )

        far_away = self.termination_computation.far_away_mask(
            named_obs=named_obs,
            disc_obs=disc_obs,
            action=action,
            prev_obs=prev_obs,
        )

        # we do not early terminate. there is a single criteria to achieving
        # more rewards: do not go far away.
        termination = self.termination_computation.combine(
            # actual_success=actual_success,
            far_away=far_away if self.env.mode == "train" else None,
        )

        self.common.reset_step_cache()
        return total_reward, termination, truncation
