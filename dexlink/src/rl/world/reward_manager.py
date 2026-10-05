from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from dexlink.src.retargeting.data.demo_loader import is_oakink_dataset
from dexlink.utils.retargeting.error_functions import angle_error
from dexlink.utils.retargeting.math_utils import check_shape, quat_from_rotvec, quat_mul

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import RewardCfgV2
    from dexlink.src.rl.world.env import Env
    from dexlink.src.rl.world.robot import Hand


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
        self.physics = env.physics_manager
        self.discriminator = None
        self.log_metrics = cfg.log_metrics
        self.log_essential_metrics_only = cfg.log_essential_metrics_only

        object_only = self.env.cfg.task_cfg.object_only_scene
        hand_actions = len(self.env.task_manager.active_hand_sides) > 0
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
        self.has_retargeted_joint_mse_reward = (
            (cfg.retargeted_joint_mse_reward_weight > 0.0)
            and (not object_only)
            and hand_actions
        )
        self.has_wrist_velocity_reward = (
            (cfg.wrist_velocity_reward_weight > 0.0)
            and (not object_only)
            and hand_actions
        )
        self.has_task_dense = cfg.task_weight > 0.0 and cfg.dense_task_weight > 0.0
        self.has_joint_smoothness_reward = (
            (cfg.joint_smoothness_reward_weight > 0.0)
            and (not object_only)
            and hand_actions
        )

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
        self._reference_commands = None
        self._object_tracking_cache = None
        self._rel_hand_obj_cache = None
        self._current_hand_obj_rel_bps = None
        self._prev_hand_obj_rel_bps = None
        self._mixed_reference_hand_pos: dict[str, dict[str, torch.Tensor]] = {}
        self._mixed_reference_hand_pos_logged: set[str] = set()
        self._last_eval_feature_info: dict[str, Any] = {}

    def reset_step_cache(self) -> None:
        self._current_commands = None
        self._reference_commands = None
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

    def current_reach_phase_mask(self) -> torch.Tensor:
        reach_phase = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        if not is_oakink_dataset(self.env.task_cfg.demo_dataset):
            return reach_phase

        original_demo = self.env.task_manager.original_demo
        if not hasattr(original_demo, "n_reach_frames"):
            return reach_phase

        current_reference_timestep = self.current_reference_timesteps()
        transition_frames = int(self.env.cfg.reset_cfg.reach_to_lift_transition_frames)
        if transition_frames < 0:
            raise ValueError(
                "Expected reset_cfg.reach_to_lift_transition_frames to be non-negative, "
                f"got {transition_frames}."
            )
        reach_phase_end = max(int(original_demo.n_reach_frames) - transition_frames, 0)
        reach_phase = current_reference_timestep < reach_phase_end
        if self.log_metrics:
            self.env.extras["log/current_reach_phase"] = reach_phase.to(torch.float32)
        return reach_phase

    def current_pre_lift_phase_mask(self) -> torch.Tensor:
        pre_lift_phase = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        if not is_oakink_dataset(self.env.task_cfg.demo_dataset):
            return pre_lift_phase

        original_demo = self.env.task_manager.original_demo
        if not hasattr(original_demo, "n_reach_frames"):
            return pre_lift_phase

        lift_start_timestep = min(
            max(int(original_demo.n_reach_frames), 0),
            self.env.task_manager.total_steps - 1,
        )
        pre_lift_phase = self.current_reference_timesteps() < lift_start_timestep
        if self.log_metrics:
            self.env.extras["log/current_pre_lift_phase"] = pre_lift_phase.to(
                torch.float32
            )
        return pre_lift_phase

    def current_reference_timesteps(self) -> torch.Tensor:
        return self.env.command_time_step.to(dtype=torch.long)

    def should_log_full_metrics(self) -> bool:
        return bool(self.log_metrics)

    def should_log_essential_metrics(self) -> bool:
        return bool(self.log_metrics or self.log_essential_metrics_only)

    def lift_start_reference_timesteps(self) -> torch.Tensor:
        current_ts = self.current_reference_timesteps()
        if not is_oakink_dataset(self.env.task_cfg.demo_dataset):
            return torch.full_like(
                current_ts,
                max(self.env.task_manager.total_steps - 1, 0),
            )

        original_demo = self.env.task_manager.original_demo
        if not hasattr(original_demo, "n_reach_frames"):
            raise ValueError(
                "OakInk reward phase scheduling requires task_manager.original_demo.n_reach_frames."
            )

        lift_start_timestep = min(
            max(int(original_demo.n_reach_frames), 0),
            self.env.task_manager.total_steps - 1,
        )
        return torch.full_like(current_ts, lift_start_timestep)

    def reward_term_schedule(self, term_name: str):
        schedules = self.cfg.term_schedules
        if not hasattr(schedules, term_name):
            raise ValueError(f"Unknown reward term schedule for {term_name!r}.")
        return getattr(schedules, term_name)

    def _phase_progress(self, phase: str) -> tuple[torch.Tensor, torch.Tensor]:
        current_ts = self.current_reference_timesteps().to(torch.float32)
        transition_ts = self.lift_start_reference_timesteps().to(torch.float32)
        total_end = max(self.env.task_manager.total_steps - 1, 0)
        total_end_t = torch.full_like(current_ts, float(total_end))

        if phase == "all":
            denom = torch.clamp(total_end_t, min=1.0)
            progress = torch.clamp(current_ts / denom, min=0.0, max=1.0)
            mask = torch.ones_like(self.env.timesteps, dtype=torch.bool)
            return progress, mask

        if phase == "reach":
            mask = current_ts < transition_ts
            denom = torch.clamp(transition_ts, min=1.0)
            progress = torch.clamp(current_ts / denom, min=0.0, max=1.0)
            progress = torch.where(
                transition_ts <= 0.0,
                torch.where(
                    mask, torch.ones_like(progress), torch.zeros_like(progress)
                ),
                progress,
            )
            return progress, mask

        if phase == "lift":
            lift_duration = total_end_t - transition_ts
            denom = torch.clamp(lift_duration, min=1.0)
            progress = torch.clamp(
                (current_ts - transition_ts) / denom,
                min=0.0,
                max=1.0,
            )
            mask = current_ts >= transition_ts
            progress = torch.where(
                lift_duration <= 0.0,
                torch.where(
                    mask, torch.ones_like(progress), torch.zeros_like(progress)
                ),
                progress,
            )
            return progress, mask

        raise ValueError(
            f"Unsupported reward schedule phase: {phase!r}. "
            "Expected one of {'all', 'reach', 'lift'}."
        )

    def reward_term_multiplier(self, term_name: str) -> torch.Tensor:
        schedule = self.reward_term_schedule(term_name)
        scheme = schedule.scheme
        ones = torch.ones(
            self.env.num_envs, dtype=torch.float32, device=self.env.device
        )
        zeros = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.env.device
        )

        if scheme == "constant":
            return ones * float(schedule.value)

        if scheme == "stage_levels":
            reach_mask = self.current_reach_phase_mask()
            pre_lift_mask = self.current_pre_lift_phase_mask()
            transition_mask = torch.logical_and(pre_lift_mask, ~reach_mask)
            return torch.where(
                reach_mask,
                ones * float(schedule.reach_weight),
                torch.where(
                    transition_mask,
                    ones * float(schedule.transition_weight),
                    ones * float(schedule.lift_weight),
                ),
            )

        if scheme == "phase_levels":
            reach_mask = self.current_pre_lift_phase_mask()
            return torch.where(
                reach_mask,
                ones * float(schedule.reach_weight),
                ones * float(schedule.lift_weight),
            )

        if scheme == "phase_window":
            if not (
                0.0 <= float(schedule.start_frac) <= float(schedule.end_frac) <= 1.0
            ):
                raise ValueError(
                    "Reward schedule phase_window requires 0 <= start_frac <= end_frac <= 1."
                )
            progress, phase_mask = self._phase_progress(schedule.phase)
            window_mask = (
                phase_mask
                & (progress >= float(schedule.start_frac))
                & (progress <= float(schedule.end_frac))
            )
            return torch.where(window_mask, ones * float(schedule.value), zeros)

        if scheme == "reach_lift_exponential":
            progress_reach, reach_mask = self._phase_progress("reach")
            progress_lift, lift_mask = self._phase_progress("lift")
            rate = float(schedule.exponential_rate)
            if rate <= 0.0:
                raise ValueError(
                    "Reward schedule reach_lift_exponential requires exponential_rate > 0."
                )
            exp_neg_rate = float(np.exp(-rate))

            reach_curve = (torch.exp(-rate * progress_reach) - exp_neg_rate) / (
                1.0 - exp_neg_rate
            )
            lift_curve = (torch.exp(rate * progress_lift) - 1.0) / (
                float(np.exp(rate)) - 1.0
            )
            reach_curve = torch.clamp(reach_curve, min=0.0, max=1.0)
            lift_curve = torch.clamp(lift_curve, min=0.0, max=1.0)

            return torch.where(
                reach_mask,
                reach_curve,
                torch.where(lift_mask, lift_curve, zeros),
            )

        raise ValueError(
            f"Unsupported reward schedule scheme: {scheme!r}. "
            "Expected one of {'constant', 'stage_levels', 'phase_levels', 'phase_window', 'reach_lift_exponential'}."
        )

    def hand_obj_rel_reference_timesteps(self) -> torch.Tensor:
        phase_mode = self.cfg.hand_obj_rel_reference_mode
        current_ts = self.current_reference_timesteps()
        if phase_mode in {"all_phases", "lift_only"}:
            return current_ts
        if phase_mode == "lift_start_reference_before_lift":
            return torch.where(
                self.current_pre_lift_phase_mask(),
                self.lift_start_reference_timesteps(),
                current_ts,
            )
        raise ValueError(
            "Unsupported rewards_cfg.hand_obj_rel_reference_mode: "
            f"{phase_mode!r}. Expected one of "
            "{'all_phases', 'lift_only', 'lift_start_reference_before_lift'}."
        )

    def _hand_obj_rel_thresholds(self) -> dict[str, torch.Tensor]:
        if self.cfg.hand_obj_rel_reference_mode in {
            "lift_only",
            "lift_start_reference_before_lift",
        }:
            reach_phase = self.current_pre_lift_phase_mask()
        else:
            reach_phase = self.current_reach_phase_mask()

        def select(reach_value: float, lift_value: float) -> torch.Tensor:
            return torch.where(
                reach_phase,
                torch.full_like(
                    self.env.timesteps,
                    float(reach_value),
                    dtype=torch.float32,
                    device=self.env.device,
                ),
                torch.full_like(
                    self.env.timesteps,
                    float(lift_value),
                    dtype=torch.float32,
                    device=self.env.device,
                ),
            )

        return {
            "distance_completion": select(
                self.cfg.reach_hand_obj_rel_distance_completion_thd,
                self.cfg.hand_obj_rel_distance_completion_thd,
            ),
            "direction_completion": select(
                self.cfg.reach_hand_obj_rel_direction_completion_thd,
                self.cfg.hand_obj_rel_direction_completion_thd,
            ),
            "normal_completion": select(
                self.cfg.reach_hand_obj_rel_normal_completion_thd,
                self.cfg.hand_obj_rel_normal_completion_thd,
            ),
            "distance_failure": select(
                self.cfg.reach_hand_obj_rel_distance_failure_thd,
                self.cfg.hand_obj_rel_distance_failure_thd,
            ),
            "direction_failure": select(
                self.cfg.reach_hand_obj_rel_direction_failure_thd,
                self.cfg.hand_obj_rel_direction_failure_thd,
            ),
            "normal_failure": select(
                self.cfg.reach_hand_obj_rel_normal_failure_thd,
                self.cfg.hand_obj_rel_normal_failure_thd,
            ),
        }

    def _reference_hand_pos_dict(self, side: str) -> dict[str, torch.Tensor]:
        original_demo = self.env.task_manager.original_demo
        device = self.env.device
        reference_manager = self.env.reference_trajectory_manager

        if reference_manager.is_enabled() and self.cfg.use_retargeted_motion:
            hand: Hand = self.env.physics_manager.scene_entities[side]
            side_link_pos = reference_manager.hand_link_positions(side)
            return {
                human_name: side_link_pos[link_name].to(
                    device=device, dtype=torch.float32
                )
                for human_name, link_name in zip(
                    hand.human_joint_names, hand.human_mapped_link_names
                )
            }

        if self.env.cfg.reset_cfg.retargeted_hand_reference_mode == "canonical_fingers":
            hand: Hand = self.env.physics_manager.scene_entities[side]
            side_link_pos = reference_manager.hand_link_positions(side)
            return {
                human_name: side_link_pos[link_name].to(
                    device=device, dtype=torch.float32
                )
                for human_name, link_name in zip(
                    hand.human_joint_names, hand.human_mapped_link_names
                )
            }

        if not self.cfg.use_retargeted_motion:
            return self._mixed_human_retargeted_hand_pos_dict(side)

        hand: Hand = self.env.physics_manager.scene_entities[side]
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

    def _ensure_retargeted_hand_link_pos_for_reward(self, side: str) -> None:
        original_demo = self.env.task_manager.original_demo
        ref_motion = original_demo.retargeted_motion
        if ref_motion is None:
            raise ValueError(
                "Retargeted motion is required for reward robot-keypoint fallback."
            )
        if f"{side}_link_pos" not in ref_motion:
            self.env.task_manager.process_retargeted_motion()
        ref_motion = original_demo.retargeted_motion
        if ref_motion is None or f"{side}_link_pos" not in ref_motion:
            raise ValueError(
                f"Retargeted motion missing {side}_link_pos for reward reference vectors."
            )

    def _retargeted_hand_pos_dict(self, side: str) -> dict[str, torch.Tensor]:
        self._ensure_retargeted_hand_link_pos_for_reward(side)
        original_demo = self.env.task_manager.original_demo
        hand: Hand = self.env.physics_manager.scene_entities[side]
        ref_motion = original_demo.retargeted_motion
        if ref_motion is None:
            raise ValueError(
                "Retargeted motion is required for reward robot-keypoint fallback."
            )
        side_link_pos = ref_motion[f"{side}_link_pos"]
        return {
            human_name: torch.as_tensor(
                side_link_pos[link_name],
                device=self.env.device,
                dtype=torch.float32,
            )
            for human_name, link_name in zip(
                hand.human_joint_names,
                hand.human_mapped_link_names,
            )
        }

    def _human_hand_pos_dict(
        self, side: str
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        original_demo = self.env.task_manager.original_demo
        if side == "left":
            ref_pos = original_demo.relevant_left_joint_pos
        elif side == "right":
            ref_pos = original_demo.relevant_right_joint_pos
        else:
            raise ValueError(f"Unknown side: {side}")
        if len(ref_pos) == 0:
            raise ValueError(
                f"rewards_cfg.use_retargeted_motion=False requires human {side} "
                "reference keypoints."
            )

        first = next(iter(ref_pos.values()))
        n_frames = int(first.shape[0])
        human_valid = getattr(
            original_demo,
            f"human_reference_{side}_joint_valid",
            None,
        )
        if human_valid is None:
            valid_mask = np.ones((n_frames,), dtype=bool)
        else:
            valid_mask = np.asarray(human_valid, dtype=bool)
        if valid_mask.shape != (n_frames,):
            raise ValueError(
                f"Expected human_reference_{side}_joint_valid shape {(n_frames,)}, "
                f"got {valid_mask.shape}."
            )

        return (
            {
                k: torch.as_tensor(v, device=self.env.device, dtype=torch.float32)
                for k, v in ref_pos.items()
            },
            torch.from_numpy(valid_mask).bool().to(self.env.device),
        )

    def _mixed_human_retargeted_hand_pos_dict(
        self, side: str
    ) -> dict[str, torch.Tensor]:
        if side in self._mixed_reference_hand_pos:
            return self._mixed_reference_hand_pos[side]

        human_pos, human_valid = self._human_hand_pos_dict(side)
        if torch.all(human_valid):
            self._mixed_reference_hand_pos[side] = human_pos
            return human_pos
        if self.env.cfg.task_cfg.require_complete_human_hand_reference:
            missing = torch.nonzero(~human_valid, as_tuple=False).squeeze(-1).tolist()
            raise ValueError(
                "This preset requires complete human hand references and forbids "
                f"retargeted fallback, but {side!r} is missing frames {missing}."
            )

        robot_pos = self._retargeted_hand_pos_dict(side)
        if set(human_pos.keys()) != set(robot_pos.keys()):
            raise ValueError(
                f"Human and retargeted {side} keypoint names must match, got "
                f"human={sorted(human_pos.keys())}, retargeted={sorted(robot_pos.keys())}."
            )

        mixed_pos = {}
        for name in robot_pos:
            if tuple(human_pos[name].shape) != tuple(robot_pos[name].shape):
                raise ValueError(
                    f"Human and retargeted {side} keypoint {name!r} shapes must match, "
                    f"got {tuple(human_pos[name].shape)} and {tuple(robot_pos[name].shape)}."
                )
            value = robot_pos[name].clone()
            value[human_valid] = human_pos[name][human_valid]
            mixed_pos[name] = value

        if side not in self._mixed_reference_hand_pos_logged:
            n_human = int(human_valid.sum().item())
            n_total = int(human_valid.numel())
            print(
                f"[RewardManager] {side} reference hand keypoints: using human "
                f"keypoints for {n_human}/{n_total} frames and retargeted robot "
                f"keypoints for {n_total - n_human}/{n_total} frames."
            )
            self._mixed_reference_hand_pos_logged.add(side)
        self._mixed_reference_hand_pos[side] = mixed_pos
        return mixed_pos

    def construct_reference_vectors(self) -> None:
        original_demo = self.env.task_manager.original_demo
        self.reference_pair_vectors = {}
        self.reference_triplet_vectors = {}

        for side in ["left", "right"]:
            if side not in self.env.physics_manager.scene_entities:
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

    def query_reference_commands(
        self, reference_timesteps: torch.Tensor
    ) -> torch.Tensor:
        return self.env.commands_manager.query_commands_from_timesteps(
            reference_timesteps
        )

    def get_reference_commands(self) -> torch.Tensor:
        if self._reference_commands is None:
            self._reference_commands = self.query_reference_commands(
                self.current_reference_timesteps()
            )
        return self._reference_commands

    def _final_scene_arti_thresholds(self, valid_arti_count: int) -> torch.Tensor:
        if valid_arti_count <= 0:
            raise ValueError(
                f"Expected positive valid articulation count, got {valid_arti_count}."
            )
        entity = self.env.physics_manager.scene_entities["object"]
        joints = list(getattr(entity, "joints", []))
        if len(joints) != valid_arti_count:
            raise ValueError(
                f"Expected {valid_arti_count} object joints, got {len(joints)}."
            )
        thresholds = []
        for joint in joints:
            joint_type = str(joint.joint_type)
            if joint_type == "prismatic":
                thresholds.append(float(self.cfg.task_success_prismatic_arti_thd))
            elif joint_type in ("revolute", "continuous"):
                thresholds.append(float(self.cfg.task_success_revolute_arti_thd))
            else:
                raise ValueError(
                    f"Unsupported articulation joint type {joint_type!r}; expected "
                    "prismatic, revolute, or continuous."
                )
        return torch.as_tensor(
            thresholds,
            dtype=torch.float32,
            device=self.env.device,
        )

    def _log_paper_hand_obj_rel_metrics(
        self,
        named_obs: dict[str, torch.Tensor],
    ) -> None:
        if not self.env.cfg.commands_cfg.use_rel_bps:
            return
        target_bps = self.get_obj_bps_relative_to_hand(
            named_obs=named_obs,
            use_prev_obj_pcd=False,
        )
        if len(target_bps) == 0:
            return

        reference_timesteps = self.hand_obj_rel_reference_timesteps()
        named_commands = self.env.commands_manager.named_view(
            self.query_reference_commands(reference_timesteps)
        )
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
        n_hands = 0
        for side, side_target_bps in target_bps.items():
            if is_oakink_dataset(self.env.task_cfg.demo_dataset):
                if side != self.env.task_manager.original_demo.side:
                    continue
            bps_key = f"{side}_bps"
            if bps_key not in named_commands:
                raise ValueError(
                    f"Paper hand-object relation metrics expected reference command "
                    f"{bps_key!r}, but available commands are "
                    f"{sorted(named_commands)}."
                )
            robot_bps = side_target_bps.view(self.env.num_envs, -1, 6)
            reference_bps = named_commands[bps_key].view(self.env.num_envs, -1, 6)
            _, component_logs, _, _ = self._hand_rel_obj_bps_reward(
                robot_bps=robot_bps,
                ref_bps=reference_bps,
            )
            for key, value in component_logs.items():
                component_sums[key] += value
            n_hands += 1

        if n_hands == 0:
            return
        for key in component_sums:
            component_sums[key] = component_sums[key] / n_hands

        self.env.extras["log/paper_hand_obj_rel_distance_cost"] = component_sums[
            "distance_cost"
        ]
        self.env.extras["log/paper_hand_obj_rel_direction_cost"] = component_sums[
            "direction_cost"
        ]
        self.env.extras["log/paper_hand_obj_rel_displacement_cost"] = component_sums[
            "displacement_cost"
        ]
        self.env.extras["log/paper_hand_obj_rel_normal_cost"] = component_sums[
            "normal_cost"
        ]
        self.env.extras["log/paper_hand_obj_rel_displacement_reward"] = component_sums[
            "displacement_reward"
        ]
        self.env.extras["log/paper_hand_obj_rel_normal_reward"] = component_sums[
            "normal_reward"
        ]

    def log_paper_metrics(self, named_obs: dict[str, torch.Tensor]) -> None:
        if not self.should_log_essential_metrics():
            return

        named_current_commands = self.env.commands_manager.named_view(
            self.get_reference_commands()
        )
        pos_error = torch.linalg.norm(
            named_obs["obj_pos"] - named_current_commands["gt_pos"],
            dim=-1,
        )
        quat_error = angle_error(
            named_obs["obj_quat"],
            named_current_commands["gt_rot"],
            current_format=self.env.cfg.obs_cfg.rot_format,
            target_format=self.env.cfg.commands_cfg.rot_format,
        )
        self.env.extras["log/paper_object_pos_error_m"] = pos_error
        self.env.extras["log/paper_object_quat_error_rad"] = quat_error
        if "gt_arti" in named_current_commands and "obj_arti" in named_obs:
            arti_error = (
                (named_obs["obj_arti"] - named_current_commands["gt_arti"])
                .abs()
                .sum(dim=-1)
            )
            self.env.extras["log/paper_object_arti_error"] = arti_error

        final_reference_timestep = self.env.task_manager.total_steps - 1
        final_reference_timesteps = torch.full(
            (self.env.num_envs,),
            final_reference_timestep,
            dtype=torch.long,
            device=self.env.device,
        )
        named_final_commands = self.env.commands_manager.named_view(
            self.query_reference_commands(final_reference_timesteps)
        )
        final_pos_error = torch.linalg.norm(
            named_obs["obj_pos"] - named_final_commands["gt_pos"],
            dim=-1,
        )
        final_quat_error = angle_error(
            named_obs["obj_quat"],
            named_final_commands["gt_rot"],
            current_format=self.env.cfg.obs_cfg.rot_format,
            target_format=self.env.cfg.commands_cfg.rot_format,
        )
        final_success = torch.logical_and(
            final_pos_error <= float(self.cfg.task_success_pos_thd),
            final_quat_error <= float(self.cfg.task_success_rot_thd),
        )
        self.env.extras["log/paper_final_scene_max_pos_error_m"] = final_pos_error
        self.env.extras["log/paper_final_scene_max_quat_error_rad"] = final_quat_error
        if "gt_arti" in named_final_commands and "obj_arti" in named_obs:
            final_arti_abs_error = (
                named_obs["obj_arti"] - named_final_commands["gt_arti"]
            ).abs()
            final_arti_error = final_arti_abs_error.sum(dim=-1)
            thresholds = self._final_scene_arti_thresholds(
                valid_arti_count=final_arti_abs_error.shape[1]
            )
            final_success = torch.logical_and(
                final_success,
                (final_arti_abs_error <= thresholds.unsqueeze(0)).all(dim=-1),
            )
            self.env.extras["log/paper_final_scene_max_arti_error"] = final_arti_error
        self.env.extras["log/paper_final_scene_configuration_success"] = (
            final_success.to(torch.float32)
        )

        self._log_paper_hand_obj_rel_metrics(named_obs)

    def get_reference_hand_joint_positions(
        self, side: str, reference_timesteps: torch.Tensor | None = None
    ) -> torch.Tensor:
        reference_manager = self.env.reference_trajectory_manager
        if reference_manager.is_enabled() and self.cfg.use_retargeted_motion:
            frame_jvals = reference_manager.hand_joint_positions(side)
        else:
            frame_jvals = getattr(self.env.reset_manager, f"frame_{side}_jvals", None)
        if frame_jvals is None:
            raise ValueError(
                f"Retargeted joint values for side {side!r} are not available in ResetManager."
            )
        reference_ts = (
            self.current_reference_timesteps()
            if reference_timesteps is None
            else reference_timesteps
        )
        return frame_jvals.to(device=self.env.device, dtype=torch.float32)[reference_ts]

    def get_retargeted_reference_hand_joint_positions(
        self, side: str, reference_timesteps: torch.Tensor | None = None
    ) -> torch.Tensor:
        # This helper is intentionally independent of rewards_cfg.use_retargeted_motion.
        # It always returns the robot-hand reference trajectory used for retargeted
        # tracking rewards, either from the active optimized reference or from the
        # reset-manager robot reference buffers.
        reference_manager = self.env.reference_trajectory_manager
        if reference_manager.is_enabled():
            frame_jvals = reference_manager.hand_joint_positions(side)
        else:
            frame_jvals = getattr(self.env.reset_manager, f"frame_{side}_jvals", None)
        if frame_jvals is None:
            raise ValueError(
                f"Retargeted joint values for side {side!r} are not available."
            )
        reference_ts = (
            self.current_reference_timesteps()
            if reference_timesteps is None
            else reference_timesteps
        )
        return frame_jvals.to(device=self.env.device, dtype=torch.float32)[reference_ts]

    def get_retargeted_reference_wrist_pose(
        self, side: str, reference_timesteps: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Wrist reference for retargeted tracking always comes from the robot
        # reference trajectory, never from the human-reference reward path.
        reference_manager = self.env.reference_trajectory_manager
        if reference_manager.is_enabled():
            wrist_pos_all = reference_manager.hand_wrist_pos(side)
            wrist_quat_all = reference_manager.hand_wrist_quat(side)
        else:
            wrist_pos_all = getattr(
                self.env.reset_manager, f"frame_{side}_wrist_pos", None
            )
            wrist_quat_all = getattr(
                self.env.reset_manager, f"frame_{side}_wrist_quat", None
            )
        if wrist_pos_all is None or wrist_quat_all is None:
            raise ValueError(
                f"Retargeted wrist pose reference for side {side!r} is not available."
            )
        reference_ts = (
            self.current_reference_timesteps()
            if reference_timesteps is None
            else reference_timesteps
        )
        wrist_pos = wrist_pos_all.to(device=self.env.device, dtype=torch.float32)[
            reference_ts
        ]
        wrist_quat = wrist_quat_all.to(device=self.env.device, dtype=torch.float32)[
            reference_ts
        ]
        return wrist_pos, wrist_quat

    def get_obj_bps_relative_to_hand(
        self, named_obs: dict[str, torch.Tensor], use_prev_obj_pcd: bool = False
    ) -> dict[str, torch.Tensor]:
        obs_mgr = self.env.observation_manager
        target_named_obs = named_obs
        if use_prev_obj_pcd:
            if self.env.current_named_privileged_obs is None:
                raise ValueError(
                    "Current privileged named observation is not available."
                )
            target_named_obs = self.env.current_named_privileged_obs

        obj_pcd = obs_mgr.get_object_pcd_from_named_obs(
            target_named_obs,
            expected_feat_dim=obs_mgr.privileged_object_pcd_dim,
        )
        if use_prev_obj_pcd:
            if self._prev_hand_obj_rel_bps is None:
                self._prev_hand_obj_rel_bps = self.compute_obj_bps_relative_to_hand(
                    named_obs=named_obs,
                    obj_pcd=obj_pcd,
                )
            return self._prev_hand_obj_rel_bps

        if self._current_hand_obj_rel_bps is None:
            self._current_hand_obj_rel_bps = self.compute_obj_bps_relative_to_hand(
                named_obs=named_obs,
                obj_pcd=obj_pcd,
            )
        return self._current_hand_obj_rel_bps

    def compute_obj_bps_relative_to_hand(
        self, named_obs: dict[str, torch.Tensor], obj_pcd=None
    ) -> dict[str, torch.Tensor]:
        obs_mgr = self.env.observation_manager
        if obj_pcd is None:
            obj_pcd = obs_mgr.get_object_pcd_from_named_obs(
                named_obs,
                expected_feat_dim=obs_mgr.privileged_object_pcd_dim,
            )

        bps = {}
        for side in ["left", "right"]:
            obs_key = f"{side}_lpos"
            if obs_key not in named_obs:
                continue
            hand: Hand = self.env.physics_manager.scene_entities[side]
            n_links = len(hand.human_joint_names)
            lpos = named_obs[obs_key].view(self.env.num_envs, n_links, 3)
            result = obs_mgr.compute_bps(pcd=obj_pcd, bps=lpos)
            bps[side] = result.reshape(result.shape[0], -1)
        return bps

    def get_hand_obj_rel_bps_dim(self) -> int:
        total_dim = 0
        for side in self.env.task_manager.active_hand_sides:
            if side not in self.env.physics_manager.scene_entities:
                continue
            hand: Hand = self.env.physics_manager.scene_entities[side]
            total_dim += len(hand.human_joint_names) * 6
        return total_dim

    def pack_hand_obj_rel_bps(
        self, bps_by_side: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        packed: list[torch.Tensor] = []
        for side in self.env.task_manager.active_hand_sides:
            if side not in self.env.physics_manager.scene_entities:
                continue
            if side not in bps_by_side:
                raise ValueError(
                    f"Missing hand-object-relative BPS for active side {side!r}. "
                    f"Available sides: {list(bps_by_side.keys())}."
                )
            packed.append(bps_by_side[side])
        if len(packed) == 0:
            raise ValueError(
                "Expected at least one active hand side when packing hand-object-relative BPS."
            )
        return torch.cat(packed, dim=-1)

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

        reference_ts = self.current_reference_timesteps()
        rewards = torch.zeros_like(self.env.timesteps, dtype=torch.float32).to(
            self.env.device
        )
        n_hands = 0

        for side in ["left", "right"]:
            obs_key = f"{side}_lpos"
            if obs_key not in named_obs:
                continue

            hand: Hand = self.env.physics_manager.scene_entities[side]
            human_joint_names = hand.human_joint_names
            lpos = named_obs[obs_key].view(self.env.num_envs, len(human_joint_names), 3)
            lpos_dict = {k: lpos[:, i] for i, k in enumerate(human_joint_names)}

            pair_vectors = compute_pair_vec(self.pairs_to_optimize, lpos_dict)
            triplet_vectors = compute_triplet_vec(self.triplets_to_optimize, lpos_dict)

            ref_pair_vecs = self.reference_pair_vectors[side][reference_ts].to(
                self.env.device
            )
            ref_triplet_vecs = self.reference_triplet_vectors[side][reference_ts].to(
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
        thresholds = self._hand_obj_rel_thresholds()
        hand_relative_pose_achieved = (
            (logs["distance_cost"] < thresholds["distance_completion"])
            & (logs["direction_cost"] < thresholds["direction_completion"])
            & (logs["normal_cost"] < thresholds["normal_completion"])
        )
        hand_far_away = (
            (logs["distance_cost"] > thresholds["distance_failure"])
            | (logs["direction_cost"] > thresholds["direction_failure"])
            | (logs["normal_cost"] > thresholds["normal_failure"])
        )
        # print(f"Distance: {logs['distance_cost'][0]}, Completion: {thresholds['distance_completion'][0]}, Failure: {thresholds['distance_failure'][0]}")
        # print(f"Direction: {logs['direction_cost'][0]}, Completion: {thresholds['direction_completion'][0]}, Failure: {thresholds['direction_failure'][0]}")
        # print(f"Normal: {logs['normal_cost'][0]}, Completion: {thresholds['normal_completion'][0]}, Failure: {thresholds['normal_failure'][0]}")

        # print(f" Hand far away: {hand_far_away[0]}, Hand relative pose achieved: {hand_relative_pose_achieved[0]}")

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

        self.env.extras["hand_obj_rel_bps"] = self.pack_hand_obj_rel_bps(target_bps)

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

        reference_timesteps = self.hand_obj_rel_reference_timesteps()
        named_commands = self.env.commands_manager.named_view(
            self.query_reference_commands(reference_timesteps)
        )
        n_hands = 0
        for side in target_bps:
            if is_oakink_dataset(self.env.task_cfg.demo_dataset):
                if side != self.env.task_manager.original_demo.side:
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
            self.env.physics_manager.scene_entities["object"].object
            if self.cfg.use_filtered_contact
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
            if side not in self.env.physics_manager.scene_entities:
                continue

            hand: Hand = self.env.physics_manager.scene_entities[side]
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

            contact_info = self.physics.get_filtered_contact_info(
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

    def retargeted_joint_mse_reward(self, named_obs, **kwargs):
        return self.retargeted_joint_mse_tracking_score(
            apply_reference_noise=bool(self.cfg.retargeted_joint_mse_reference_noise),
            log_components=True,
        )

    def retargeted_joint_mse_tracking_score(
        self,
        *,
        apply_reference_noise: bool,
        log_components: bool,
    ) -> torch.Tensor:
        # This reward always tracks the robot retargeted reference, regardless of
        # rewards_cfg.use_retargeted_motion. It is defined on finger joint angles
        # plus wrist pose only; human link positions are intentionally excluded.
        joint_mode = str(self.cfg.retargeted_joint_mse_joint_mode)
        if joint_mode not in {"wrist_and_fingers", "wrist_only"}:
            raise ValueError(
                "Expected rewards_cfg.retargeted_joint_mse_joint_mode to be "
                f"'wrist_and_fingers' or 'wrist_only', got {joint_mode!r}."
            )
        rewards = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.env.device
        )
        n_hands = 0
        for side in ("left", "right"):
            if side not in self.env.physics_manager.scene_entities:
                continue
            hand: Hand = self.env.physics_manager.scene_entities[side]
            current_jpos = self.physics.get_hand_joint_positions(hand).to(
                device=self.env.device, dtype=torch.float32
            )
            reference_ts = self.current_reference_timesteps()
            reference_jpos = self.get_retargeted_reference_hand_joint_positions(
                side, reference_timesteps=reference_ts
            )
            if tuple(current_jpos.shape) != tuple(reference_jpos.shape):
                raise ValueError(
                    f"Current and retargeted joint positions shape mismatch for side {side!r}: "
                    f"{tuple(current_jpos.shape)} vs {tuple(reference_jpos.shape)}."
                )
            current_wrist_pos = self.env.current_wrist_pos_by_side.get(side)
            current_wrist_quat = self.env.current_wrist_quat_by_side.get(side)
            if current_wrist_pos is None or current_wrist_quat is None:
                raise ValueError(
                    f"Current wrist pose is not populated for hand {side!r} before reward computation."
                )
            reference_wrist_pos, reference_wrist_quat = (
                self.get_retargeted_reference_wrist_pose(
                    side, reference_timesteps=reference_ts
                )
            )
            if apply_reference_noise:
                reference_jpos, reference_wrist_pos, reference_wrist_quat = (
                    self._add_retargeted_joint_mse_reference_noise(
                        hand=hand,
                        reference_jpos=reference_jpos,
                        reference_wrist_pos=reference_wrist_pos,
                        reference_wrist_quat=reference_wrist_quat,
                    )
                )
            if tuple(current_wrist_pos.shape) != tuple(reference_wrist_pos.shape):
                raise ValueError(
                    f"Current and reference wrist position shape mismatch for side {side!r}: "
                    f"{tuple(current_wrist_pos.shape)} vs {tuple(reference_wrist_pos.shape)}."
                )
            if tuple(current_wrist_quat.shape) != tuple(reference_wrist_quat.shape):
                raise ValueError(
                    f"Current and reference wrist quaternion shape mismatch for side {side!r}: "
                    f"{tuple(current_wrist_quat.shape)} vs {tuple(reference_wrist_quat.shape)}."
                )
            wrist_pos_mse = torch.mean(
                (current_wrist_pos - reference_wrist_pos) ** 2, dim=-1
            )
            wrist_rot_error = angle_error(
                current_wrist_quat,
                reference_wrist_quat,
                current_format="quat",
                target_format="quat",
            )
            wrist_pose_error = wrist_pos_mse + wrist_rot_error
            reward_scale = float(self.cfg.retargeted_joint_mse_reward_scale)
            finger_mse = torch.zeros_like(wrist_pos_mse)
            if joint_mode == "wrist_and_fingers":
                finger_joint_indices = [
                    hand.joint_names.index(joint_name)
                    for joint_name in hand.finger_joint_names
                ]
                finger_joint_indices_t = torch.as_tensor(
                    finger_joint_indices,
                    dtype=torch.long,
                    device=self.env.device,
                )
                finger_mse = torch.mean(
                    (
                        current_jpos[:, finger_joint_indices_t]
                        - reference_jpos[:, finger_joint_indices_t]
                    )
                    ** 2,
                    dim=-1,
                )
                side_reward = torch.exp(
                    -reward_scale * finger_mse - reward_scale * wrist_pose_error
                )
            else:
                side_reward = torch.exp(-reward_scale * wrist_pose_error)
            rewards += side_reward
            if log_components and self.should_log_essential_metrics():
                self.env.extras[f"log/retargeted_finger_joint_mse_{side}"] = finger_mse
                self.env.extras[f"log/retargeted_wrist_pos_mse_{side}"] = wrist_pos_mse
                self.env.extras[f"log/retargeted_wrist_rot_error_{side}"] = (
                    wrist_rot_error
                )
                self.env.extras[f"log/retargeted_wrist_pose_error_{side}"] = (
                    wrist_pose_error
                )
            n_hands += 1

        if n_hands == 0:
            raise ValueError(
                "Retargeted joint MSE reward requires at least one active hand in the scene."
            )
        return rewards / n_hands

    def _add_retargeted_joint_mse_reference_noise(
        self,
        *,
        hand: Hand,
        reference_jpos: torch.Tensor,
        reference_wrist_pos: torch.Tensor,
        reference_wrist_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wrist_pos_std = float(self.cfg.retargeted_joint_mse_wrist_pos_noise_std)
        wrist_rot_std = float(self.cfg.retargeted_joint_mse_wrist_rot_noise_std)
        finger_std = float(self.cfg.retargeted_joint_mse_finger_joint_noise_std)
        if wrist_pos_std < 0.0 or wrist_rot_std < 0.0 or finger_std < 0.0:
            raise ValueError(
                "Retargeted joint MSE reference noise std values must be non-negative, "
                f"got wrist_pos={wrist_pos_std}, wrist_rot={wrist_rot_std}, "
                f"finger={finger_std}."
            )

        noisy_jpos = reference_jpos
        noisy_wrist_pos = reference_wrist_pos
        noisy_wrist_quat = reference_wrist_quat
        if wrist_pos_std > 0.0:
            noisy_wrist_pos = reference_wrist_pos + (
                torch.randn_like(reference_wrist_pos) * wrist_pos_std
            )
        if wrist_rot_std > 0.0:
            rotvec_noise = (
                torch.randn(
                    (reference_wrist_quat.shape[0], 3),
                    dtype=reference_wrist_quat.dtype,
                    device=reference_wrist_quat.device,
                )
                * wrist_rot_std
            )
            noisy_wrist_quat = quat_mul(
                reference_wrist_quat,
                quat_from_rotvec(rotvec_noise),
            )
        if finger_std > 0.0:
            finger_joint_indices = [
                hand.joint_names.index(joint_name)
                for joint_name in hand.finger_joint_names
            ]
            finger_joint_indices_t = torch.as_tensor(
                finger_joint_indices,
                dtype=torch.long,
                device=reference_jpos.device,
            )
            noisy_jpos = reference_jpos.clone()
            noisy_jpos[:, finger_joint_indices_t] = noisy_jpos[
                :, finger_joint_indices_t
            ] + (torch.randn_like(noisy_jpos[:, finger_joint_indices_t]) * finger_std)
        return noisy_jpos, noisy_wrist_pos, noisy_wrist_quat

    def wrist_velocity_reward(self, named_obs, **kwargs):
        del named_obs, kwargs
        rewards = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.env.device
        )
        n_hands = 0
        for side in ("left", "right"):
            if not self.env.task_manager.has_hand(side):
                continue
            hand = self.env.physics_manager.scene_entities[side]
            if len(hand.wrist_control_joint_idx) == 0:
                raise ValueError(
                    f"Wrist velocity reward requires wrist control joints for side {side!r}."
                )
            wrist_dof_indices = [
                hand.dof_map[joint_idx] for joint_idx in hand.wrist_control_joint_idx
            ]
            wrist_jvel = self.physics.get_entity_dofs_velocity(
                hand.robot,
                wrist_dof_indices,
            )
            expected_shape = (self.env.num_envs, len(wrist_dof_indices))
            if tuple(wrist_jvel.shape) != expected_shape:
                raise ValueError(
                    f"Wrist joint velocity shape mismatch for side {side!r}: "
                    f"expected {expected_shape}, got {tuple(wrist_jvel.shape)}."
                )
            velocity_mse = torch.mean(wrist_jvel**2, dim=-1)
            side_reward = torch.exp(
                -float(self.cfg.wrist_velocity_reward_scale) * velocity_mse
            )
            rewards += side_reward
            if self.log_metrics:
                self.env.extras[f"log/wrist_velocity_mse_{side}"] = velocity_mse
                self.env.extras[f"log/wrist_velocity_reward_{side}"] = side_reward
            n_hands += 1

        if n_hands == 0:
            raise ValueError(
                "Wrist velocity reward is enabled, but no active hands were found."
            )
        return rewards / n_hands

    def joint_smoothness_reward(self, named_obs, prev_named_obs, **kwargs):
        rewards = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.env.device
        )
        n_hands = 0
        for side in ("left", "right"):
            key = f"{side}_jpos"
            if key not in named_obs:
                continue
            if key not in prev_named_obs:
                raise ValueError(
                    f"Previous observation is missing required joint positions for {side!r} hand: {key!r}."
                )

            current_jpos = named_obs[key]
            previous_jpos = prev_named_obs[key]
            if tuple(current_jpos.shape) != tuple(previous_jpos.shape):
                raise ValueError(
                    f"Joint position shape mismatch for {side!r} hand: "
                    f"current={tuple(current_jpos.shape)}, previous={tuple(previous_jpos.shape)}."
                )

            joint_delta = current_jpos - previous_jpos
            side_reward = torch.exp(-torch.abs(joint_delta)).mean(dim=-1)
            rewards += side_reward
            if self.log_metrics:
                self.env.extras[f"log/joint_smoothness_{side}"] = side_reward
            n_hands += 1

        if n_hands == 0:
            raise ValueError(
                "Joint smoothness reward is enabled, but no hand joint-position observations were found."
            )
        return rewards / n_hands

    def object_tracking_progress(self, named_obs, **kwargs):
        if self._object_tracking_cache is not None:
            return self._object_tracking_cache

        named_commands = self.env.commands_manager.named_view(
            self.get_reference_commands()
        )
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

        pos_diff = named_obs["obj_pos"] - named_commands["gt_pos"]
        pos_error = torch.norm(pos_diff, p=2, dim=-1)
        pos_mse = torch.mean(pos_diff.square(), dim=-1)
        pos_completion = pos_error < self.cfg.completion_pos_thd
        object_pose_achieved = torch.logical_and(object_pose_achieved, pos_completion)
        pos_reward = torch.exp(-float(self.cfg.task_pos_scale) * pos_error)
        subtask_rewards["pos"] = pos_reward
        far_away = pos_error > self.cfg.obj_command_max_dist_pos
        early_terminate = torch.logical_or(early_terminate, far_away)
        if self.log_metrics:
            self.env.extras["log/pos_task_mse"] = pos_mse
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
        quat_mse = quat_error.square()
        far_away = quat_error > self.cfg.obj_command_max_dist_rot
        early_terminate = torch.logical_or(early_terminate, far_away)
        if self.log_metrics:
            self.env.extras["log/rot_task_mse"] = quat_mse
            self.env.extras["log/quat_error"] = quat_error
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
        if self.should_log_essential_metrics():
            self.env.extras["log/object_tracking_product_rew"] = object_tracking_reward

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
        self.log_essential_metrics_only = common.log_essential_metrics_only
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
        if common.has_retargeted_joint_mse_reward:
            self.reward_terms["retargeted_joint_mse"] = (
                self.cfg.retargeted_joint_mse_reward_weight,
                self.common.retargeted_joint_mse_reward,
            )
        if common.has_wrist_velocity_reward:
            self.reward_terms["wrist_velocity"] = (
                self.cfg.wrist_velocity_reward_weight,
                self.common.wrist_velocity_reward,
            )
        if common.has_joint_smoothness_reward:
            self.reward_terms["joint_smoothness"] = (
                self.cfg.joint_smoothness_reward_weight,
                self.common.joint_smoothness_reward,
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

    def reward_term_weight(self, name: str, weight: float) -> float | torch.Tensor:
        del name
        return weight

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
        prev_named_obs,
        actual_success: torch.Tensor,
    ) -> torch.Tensor:
        total_reward = torch.zeros_like(self.env.timesteps, dtype=torch.float32).to(
            self.env.device
        )
        for name, (weight, func) in self.reward_terms.items():
            result = func(
                named_obs=named_obs,
                disc_obs=disc_obs,
                action=action,
                prev_named_obs=prev_named_obs,
            )
            reward_term = result[0] if isinstance(result, tuple) else result
            reward_multiplier = self.common.reward_term_multiplier(name)
            weighted_reward_term = (
                self.reward_term_weight(name, weight)
                * reward_term.float()
                * reward_multiplier
            )
            total_reward += weighted_reward_term
            if self.common.should_log_essential_metrics():
                self.env.extras[f"log/reward_{name}"] = reward_term
                self.env.extras[f"log/reward_{name}_raw"] = reward_term
                self.env.extras[f"log/reward_{name}_weighted"] = weighted_reward_term

        if self.cfg.task_completion_weight > 0.0:
            success_reward = actual_success.float()
            success_multiplier = self.common.reward_term_multiplier("task_completion")
            weighted_success_reward = (
                self.cfg.task_completion_weight * success_reward * success_multiplier
            )
            total_reward += weighted_success_reward
            if self.common.should_log_essential_metrics():
                self.env.extras["log/reward_task_completion"] = success_reward
                self.env.extras["log/reward_task_completion_raw"] = success_reward
                self.env.extras["log/reward_task_completion_weighted"] = (
                    weighted_success_reward
                )
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

    def pose_achieved_mask(
        self, named_obs, disc_obs, action, prev_named_obs
    ) -> torch.Tensor:

        if self.env.cfg.backend == "real":
            return torch.zeros_like(self.env.timesteps, dtype=torch.bool).to(
                self.env.device
            )

        object_pose_achieved = self.object_pose_achieved_mask(
            named_obs=named_obs,
            disc_obs=disc_obs,
            action=action,
            prev_named_obs=prev_named_obs,
        )
        hand_relative_pose_achieved = self.hand_relative_pose_achieved_mask(
            named_obs=named_obs,
            disc_obs=disc_obs,
            action=action,
            prev_named_obs=prev_named_obs,
        )
        pose_achieved = torch.logical_and(
            object_pose_achieved, hand_relative_pose_achieved
        )
        if self.common.should_log_essential_metrics():
            self.env.extras["log/object_pose_achieved"] = object_pose_achieved.to(
                torch.float32
            )
            self.env.extras["log/pose_achieved"] = pose_achieved.to(torch.float32)
        return pose_achieved

    def far_away_mask(
        self, named_obs, disc_obs, action, prev_named_obs
    ) -> torch.Tensor:
        hand_far_away = torch.zeros_like(self.env.timesteps, dtype=torch.bool).to(
            self.env.device
        )
        if self.common.has_hand_obj_rel:
            _, hand_far_away, _ = self.common.hand_obj_relative_reward(
                named_obs=named_obs,
                disc_obs=disc_obs,
                action=action,
                prev_named_obs=prev_named_obs,
            )

        _, _, object_far_away = self.common.object_tracking_progress(
            named_obs=named_obs,
            disc_obs=disc_obs,
            action=action,
            prev_named_obs=prev_named_obs,
        )

        far_away = torch.logical_or(hand_far_away, object_far_away)
        if self.common.should_log_essential_metrics():
            self.env.extras["log/far_away"] = far_away.to(torch.float32)
        return far_away

    def actual_success_mask(
        self, pose_achieved: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        unclamped_reference_timestep = self.env.get_reference_timesteps_unclamped()
        max_reference_timestep = self.env.task_manager.total_steps - 1
        hold_steps = int(self.cfg.last_reference_hold_steps)
        if self.env.task_cfg.grasp_refinement_mode:
            metric_cfg = self.env.task_cfg.metric_cfg
            if metric_cfg.metric_id != "grasp_stability":
                raise ValueError(
                    "Grasp refinement terminal success requires "
                    "metric_id='grasp_stability', got "
                    f"{metric_cfg.metric_id!r}."
                )
            task_metrics = self.env.task_manager.compute_task_metrics()
            stability_success = task_metrics["grasp_stability_success"].to(
                dtype=torch.bool
            )
            actual_success = torch.logical_and(
                stability_success,
                unclamped_reference_timestep
                >= (max_reference_timestep + hold_steps),
            )
            if self.common.should_log_essential_metrics():
                self.env.extras["log/actual_success"] = actual_success.to(
                    torch.float32
                )
            return actual_success
        actual_success = torch.logical_and(
            pose_achieved,
            unclamped_reference_timestep >= (max_reference_timestep + hold_steps),
        )
        if self.common.should_log_essential_metrics():
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
        if is_oakink_dataset(self.env.task_cfg.demo_dataset):
            termination = torch.logical_or(actual_success, far_away)
        return termination


class RewardManager:
    def __init__(self, cfg: RewardCfgV2, env: Env):
        self.cfg = cfg
        self.env = env
        self.common = RewardCommonComputation(cfg, env)
        self.reward_computation = RewardComputation(self.common)
        self.termination_computation = RewardTerminationComputation(self.common)

    def __getattr__(self, name: str):
        for component_name in (
            "common",
            "reward_computation",
            "termination_computation",
        ):
            component = self.__dict__.get(component_name)
            if component is not None and hasattr(component, name):
                return getattr(component, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def summary(self) -> str:
        return self.reward_computation.summary()

    def initialize(self) -> None:
        self.common.initialize()

    def refresh_reference_buffers(self) -> None:
        self.common.reference_pair_vectors = None
        self.common.reference_triplet_vectors = None
        self.common._reference_commands = None
        self.common.construct_reference_vectors()

    def set_discriminator(self, discriminator: Any) -> None:
        self.common.set_discriminator(discriminator)

    def get_eval_feature_info(self) -> dict[str, Any]:
        return self.common.get_eval_feature_info()

    def compute_rewards(
        self,
        next_named_privileged_obs,
        disc_obs,
        action,
        prev_named_privileged_obs,
    ):
        self.common.reset_step_cache()
        self.common.reset_eval_feature_cache()
        named_obs = next_named_privileged_obs

        pose_achieved = self.termination_computation.pose_achieved_mask(
            named_obs=named_obs,
            disc_obs=disc_obs,
            action=action,
            prev_named_obs=prev_named_privileged_obs,
        )
        actual_success = self.termination_computation.actual_success_mask(
            pose_achieved,
            named_obs=named_obs,
            disc_obs=disc_obs,
            action=action,
            prev_named_obs=prev_named_privileged_obs,
        )
        unclamped_reference_timestep = self.env.get_reference_timesteps_unclamped()
        max_reference_timestep = self.env.task_manager.total_steps - 1
        hold_steps = int(self.cfg.last_reference_hold_steps)
        truncation = unclamped_reference_timestep > (
            max_reference_timestep + hold_steps
        )
        if self.common.should_log_essential_metrics():
            self.env.extras["log/episode_timestep"] = self.env.timesteps.to(
                torch.float32
            )
        if self.common.should_log_essential_metrics():
            # progress should consider the start timestep.
            # timesteps / (max_reference_timeste + hold_steps - start_timestep)
            # average of this quantity

            episode_horizon = torch.full(
                (self.env.num_envs,),
                max_reference_timestep + hold_steps,
                dtype=torch.float32,
                device=self.env.start_timestep.device,
            ) - self.env.start_timestep.to(torch.float32)
            self.env.extras["log/episode_progress"] = self.env.timesteps.to(
                torch.float32
            ) / episode_horizon.to(self.env.timesteps.device)

        if self.env.cfg.backend == "sim":

            total_reward = self.reward_computation.compute_total_reward(
                named_obs=named_obs,
                disc_obs=disc_obs,
                action=action,
                prev_named_obs=prev_named_privileged_obs,
                actual_success=actual_success,
            )

            far_away = self.termination_computation.far_away_mask(
                named_obs=named_obs,
                disc_obs=disc_obs,
                action=action,
                prev_named_obs=prev_named_privileged_obs,
            )
            self.common.log_paper_metrics(named_obs)
            self.env.record_eval_would_termination(far_away)

            termination = self.termination_computation.combine(
                far_away=far_away if self.env.mode == "train" else None,
                # far_away=far_away
            )

            self.common.reset_step_cache()
            return total_reward, termination, truncation

        else:
            self.common.reset_step_cache()
            rewards = torch.zeros_like(self.env.timesteps, dtype=torch.float32).to(
                self.env.device
            )
            termination = torch.zeros_like(self.env.timesteps, dtype=torch.bool).to(
                self.env.device
            )
            return rewards, termination, truncation
