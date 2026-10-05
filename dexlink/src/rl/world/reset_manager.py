from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np
import torch

from dexlink.src.retargeting.data.demo_loader import is_oakink_dataset
from dexlink.src.rl.world.robot import Hand

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import EnvConfig
    from dexlink.src.rl.world.art_object import ArtObject
    from dexlink.src.rl.world.env import Env


class ResetManager:
    EVAL_RESET_HAND_EXCLUSION_RADIUS_M = 0.02
    EVAL_OBJECT_XY_RANDOMIZATION_M = 0.10
    EVAL_OBJECT_Z_RANDOMIZATION_M = 0.05

    def __init__(self, cfg: EnvConfig, env: Env):
        self.cfg = cfg
        self.env = env
        self.task = env.task_manager
        self.physics = env.physics_manager
        self.reference_manager = env.reference_trajectory_manager
        self._configured_start_range: tuple[int, int] | None = None
        self._curriculum_start_range: tuple[int, int] | None = None
        self._last_curriculum_shift_policy_iter: int = 0

    def initialize(self):
        self.process_retargeted_results()
        self._initialize_reverse_curriculum()

        if self.env.cfg.backend == "sim":
            self.print_domain_randomization_summary()
        if self.cfg.reset_cfg.dr and (self.env.cfg.backend == "sim"):
            self.apply_domain_randomization(
                np.arange(self.env.num_envs, dtype=np.int64)
            )

    def print_domain_randomization_summary(self):
        reset_cfg = self.cfg.reset_cfg
        lines = ["[DomainRandomization] Summary"]
        lines.append(f"enabled: {reset_cfg.dr}")
        lines.append(
            f"ranges: object_mass={reset_cfg.dr_object_mass}, "
            f"object_surface_friction_abs={reset_cfg.dr_contact_friction}"
        )

        object_summary = self.env.physics_manager.object_domain_randomization_summary()
        lines.append(
            "object.mass available: "
            + self._format_dr_names(object_summary["mass_available"])
        )
        lines.append(
            "object.mass skipped: "
            + self._format_dr_names(object_summary["mass_skipped"])
        )

        # Hand stiffness/damping/joint-friction DR is intentionally disabled for
        # the final experiments; keep the summary focused on active DR terms.
        print("\n".join(lines))

    def _format_dr_names(self, names: list[str]) -> str:
        if len(names) == 0:
            return "none"
        return ", ".join(names)

    def apply_domain_randomization(self, env_ids: np.ndarray):
        if not self.cfg.reset_cfg.dr or len(env_ids) == 0:
            return

        self.env.physics_manager.apply_domain_randomization(self.cfg.reset_cfg, env_ids)

    def resample_random_force(self, env_ids: np.ndarray):
        if self.env.cfg.backend == "sim":
            if self.env.sim_cfg.apply_random_force:
                self.env.physics_manager.resample_random_force(env_ids=env_ids)

    def process_retargeted_results(self):
        self.sync_reference_buffers()

    def sync_reference_buffers(self) -> None:
        self.frame_pos = self.reference_manager.object_pos()
        self.frame_quat = self.reference_manager.object_quat()
        self.frame_arti = self.reference_manager.object_arti()
        self.drake_collision_refinement_unresolved_frame_mask = None
        self.drake_collision_refinement_unresolved_frame_indices = None
        self.drake_collision_refinement_valid_frame_mask = None
        self.drake_collision_refinement_valid_frame_indices = None
        self.drake_collision_refinement_invalid_frame_mask = None
        self.drake_collision_refinement_invalid_frame_indices = None
        self.drake_collision_refinement_final_collision_violation = None
        self.drake_collision_refinement_collision_distance = None
        for side in ("left", "right"):
            setattr(self, f"frame_{side}_jvals", None)
            setattr(self, f"frame_{side}_reset_jvals", None)
            setattr(self, f"frame_{side}_wrist_pos", None)
            setattr(self, f"frame_{side}_wrist_quat", None)
            setattr(self, f"frame_ref_{side}", None)
            if not self.task.has_hand(side):
                continue
            setattr(
                self,
                f"frame_{side}_jvals",
                self.reference_manager.hand_joint_positions(side),
            )
            setattr(
                self,
                f"frame_{side}_wrist_pos",
                self.reference_manager.hand_wrist_pos(side),
            )
            setattr(
                self,
                f"frame_{side}_wrist_quat",
                self.reference_manager.hand_wrist_quat(side),
            )
            setattr(
                self,
                f"frame_ref_{side}",
                self.reference_manager.hand_reference_obs(side),
            )
            setattr(
                self,
                f"frame_{side}_reset_jvals",
                getattr(self, f"frame_{side}_jvals"),
            )
        self._sync_drake_collision_refinement_reset_buffers()
        self._apply_hand_reference_mode_to_reset_buffers()

    def _apply_hand_reference_mode_to_reset_buffers(self) -> None:
        mode = str(self.cfg.reset_cfg.retargeted_hand_reference_mode)
        if mode == "retargeted":
            return
        if mode not in {"canonical_fingers", "default_configuration"}:
            raise ValueError(
                "Expected reset_cfg.retargeted_hand_reference_mode to be "
                "'retargeted', 'canonical_fingers', or "
                f"'default_configuration', got {mode!r}."
            )

        for side in self.task.active_hand_sides:
            robot: Hand = self.physics.scene_entities[side]
            reset_jvals = getattr(self, f"frame_{side}_reset_jvals", None)
            if reset_jvals is None:
                raise ValueError(
                    f"Canonical-finger reset mode requires frame_{side}_reset_jvals."
                )
            expected_shape = (self.task.total_steps, len(robot.joint_names))
            if tuple(reset_jvals.shape) != expected_shape:
                raise ValueError(
                    f"Expected frame_{side}_reset_jvals shape {expected_shape}, "
                    f"got {tuple(reset_jvals.shape)}."
                )
            finger_indices, canonical_values = robot.canonical_finger_joint_values(
                device=self.env.device,
                dtype=reset_jvals.dtype,
            )
            masked = (
                reset_jvals.clone()
                if mode == "canonical_fingers"
                else torch.zeros_like(reset_jvals)
            )
            masked[:, finger_indices] = canonical_values.unsqueeze(0)
            setattr(self, f"frame_{side}_reset_jvals", masked)

    def _apply_hand_reset_noise(
        self, *, side: str, jpos: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        if not torch.any(noise > 0.0):
            return jpos
        mode = str(self.cfg.reset_cfg.retargeted_hand_reference_mode)
        if mode == "retargeted":
            return jpos + torch.randn_like(jpos) * noise.unsqueeze(-1)
        if mode not in {"canonical_fingers", "default_configuration"}:
            raise ValueError(
                "Expected reset_cfg.retargeted_hand_reference_mode to be "
                "'retargeted', 'canonical_fingers', or "
                f"'default_configuration', got {mode!r}."
            )

        robot: Hand = self.physics.scene_entities[side]
        finger_indices, _ = robot.canonical_finger_joint_values(
            device=self.env.device,
            dtype=jpos.dtype,
        )
        noisy = jpos.clone()
        noisy[:, finger_indices] = noisy[:, finger_indices] + torch.randn_like(
            noisy[:, finger_indices]
        ) * noise.unsqueeze(-1)
        return noisy

    def _retargeted_motion(self) -> dict[str, Any]:
        motion = getattr(self.task.original_demo, "retargeted_motion", None)
        if motion is None:
            return {}
        return motion

    def _sync_drake_collision_refinement_reset_buffers(self) -> None:
        motion = self._retargeted_motion()
        if "drake_collision_refinement_jvals" not in motion:
            return

        required_keys = (
            "drake_collision_refinement_jvals",
            "drake_collision_refinement_jnames",
            "drake_collision_refinement_hand",
            "drake_collision_refinement_unresolved_frame_mask",
            "drake_collision_refinement_final_collision_violation",
            "drake_collision_refinement_collision_distance",
        )
        missing = [key for key in required_keys if key not in motion]
        if missing:
            raise ValueError(
                "Drake collision refinement motion is incomplete; missing "
                f"{missing}."
            )

        side = str(motion["drake_collision_refinement_hand"])
        if side not in self.task.active_hand_sides:
            raise ValueError(
                "Drake collision refinement hand is not active in this task: "
                f"{side!r}, active={self.task.active_hand_sides}."
            )
        robot: Hand = self.physics.scene_entities[side]
        refined_joint_names = [
            str(name) for name in motion["drake_collision_refinement_jnames"]
        ]
        missing_joints = [
            joint_name
            for joint_name in robot.joint_names
            if joint_name not in refined_joint_names
        ]
        if missing_joints:
            raise ValueError(
                "Drake collision refinement joint values are missing simulated "
                f"{side} hand joints: {missing_joints}."
            )

        jvals_key = f"drake_collision_refinement_{side}_jvals"
        if jvals_key in motion:
            refined_jvals_np = np.asarray(motion[jvals_key], dtype=np.float32)
        else:
            refined_jvals_np = np.asarray(
                motion["drake_collision_refinement_jvals"], dtype=np.float32
            )
        if refined_jvals_np.ndim != 2:
            raise ValueError(
                f"Expected {jvals_key} shape (T, J), got {refined_jvals_np.shape}."
            )
        if refined_jvals_np.shape[0] != self.task.total_steps:
            raise ValueError(
                "Expected Drake collision refinement jvals frame count to match "
                f"task.total_steps={self.task.total_steps}, got "
                f"{refined_jvals_np.shape[0]}."
            )

        joint_indices = [refined_joint_names.index(name) for name in robot.joint_names]
        reset_jvals = torch.as_tensor(
            refined_jvals_np[:, joint_indices],
            dtype=torch.float32,
            device=self.env.device,
        )
        setattr(self, f"frame_{side}_reset_jvals", reset_jvals)

        unresolved_mask = np.asarray(
            motion["drake_collision_refinement_unresolved_frame_mask"], dtype=bool
        )
        if unresolved_mask.shape != (self.task.total_steps,):
            raise ValueError(
                "Expected drake_collision_refinement_unresolved_frame_mask shape "
                f"({self.task.total_steps},), got {unresolved_mask.shape}."
            )
        if "drake_collision_refinement_valid_frame_mask" in motion:
            valid_mask = np.asarray(
                motion["drake_collision_refinement_valid_frame_mask"], dtype=bool
            )
            if valid_mask.shape != (self.task.total_steps,):
                raise ValueError(
                    "Expected drake_collision_refinement_valid_frame_mask shape "
                    f"({self.task.total_steps},), got {valid_mask.shape}."
                )
        else:
            valid_mask = ~unresolved_mask
        if "drake_collision_refinement_invalid_frame_mask" in motion:
            invalid_mask = np.asarray(
                motion["drake_collision_refinement_invalid_frame_mask"], dtype=bool
            )
            if invalid_mask.shape != (self.task.total_steps,):
                raise ValueError(
                    "Expected drake_collision_refinement_invalid_frame_mask shape "
                    f"({self.task.total_steps},), got {invalid_mask.shape}."
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
        final_violation = np.asarray(
            motion["drake_collision_refinement_final_collision_violation"],
            dtype=np.float32,
        )
        if final_violation.shape != (self.task.total_steps,):
            raise ValueError(
                "Expected drake_collision_refinement_final_collision_violation shape "
                f"({self.task.total_steps},), got {final_violation.shape}."
            )
        self.drake_collision_refinement_unresolved_frame_mask = unresolved_mask
        self.drake_collision_refinement_unresolved_frame_indices = np.flatnonzero(
            unresolved_mask
        ).astype(np.int64)
        self.drake_collision_refinement_valid_frame_mask = valid_mask
        self.drake_collision_refinement_valid_frame_indices = np.flatnonzero(
            valid_mask
        ).astype(np.int64)
        self.drake_collision_refinement_invalid_frame_mask = invalid_mask
        self.drake_collision_refinement_invalid_frame_indices = np.flatnonzero(
            invalid_mask
        ).astype(np.int64)
        self.drake_collision_refinement_final_collision_violation = final_violation
        self.drake_collision_refinement_collision_distance = float(
            np.asarray(motion["drake_collision_refinement_collision_distance"]).reshape(
                -1
            )[0]
        )

    def _resolve_refinement_reset_timesteps(self, timesteps: np.ndarray) -> np.ndarray:
        frame_idx = np.asarray(timesteps, dtype=np.int64)
        if frame_idx.ndim != 1:
            raise ValueError(
                f"Expected sampled timesteps shape (N,), got {frame_idx.shape}."
            )
        mask = self.drake_collision_refinement_invalid_frame_mask
        if mask is None:
            mask = self.drake_collision_refinement_unresolved_frame_mask
        if mask is None:
            return frame_idx

        resolved = frame_idx.copy()
        reference_end = self.reference_frame_end_timestep()
        checked_frame_idx = np.minimum(resolved, reference_end)
        failed_rows = np.flatnonzero(mask[checked_frame_idx])
        available_valid_frames = np.flatnonzero(~mask[: reference_end + 1])
        if available_valid_frames.shape[0] == 0:
            raise ValueError(
                "No Drake-refined collision-free reset frames are available."
            )
        for row in failed_rows.tolist():
            requested_frame = int(checked_frame_idx[row])
            previous = available_valid_frames[available_valid_frames <= requested_frame]
            if previous.shape[0] > 0:
                resolved[row] = int(previous[-1])
            else:
                resolved[row] = int(available_valid_frames[0])
        return resolved

    def normalize_env_ids(self, env_ids) -> np.ndarray:
        """Normalize env_ids to a numpy array of int64 indices.

        Accepts int, list/sequence, numpy array, torch tensor, or None (meaning all envs).
        """
        if isinstance(env_ids, (int, np.integer)):
            env_ids = np.array([int(env_ids)], dtype=np.int64)
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().numpy().astype(np.int64)
        elif env_ids is None:
            env_ids = np.arange(self.env.num_envs, dtype=np.int64)
        else:
            env_ids = np.array(env_ids, dtype=np.int64)
        return env_ids

    def reference_frame_end_timestep(self) -> int:
        return self.task.total_steps - 1

    def episode_end_timestep(self) -> int:
        hold_steps = int(self.cfg.rewards_cfg.last_reference_hold_steps)
        return self.reference_frame_end_timestep() + hold_steps

    def clamp_to_reference_frame_timesteps(
        self, timesteps: np.ndarray | list[int]
    ) -> np.ndarray:
        frame_idx = np.asarray(timesteps, dtype=np.int64)
        return np.minimum(frame_idx, self.reference_frame_end_timestep())

    def _resolve_start_range(self) -> tuple[int, int]:
        if (
            self.cfg.reset_cfg.reverse_curriculum
            and self._curriculum_start_range is not None
        ):
            return self._curriculum_start_range
        if self._configured_start_range is not None:
            return self._configured_start_range
        start_idx, end_idx = self._clamp_start_range(self.cfg.reset_cfg.start_range)
        self._configured_start_range = (start_idx, end_idx)
        return start_idx, end_idx

    def _clamp_start_range(self, start_range: tuple[int, int]) -> tuple[int, int]:
        start_idx, end_idx = start_range
        episode_num_steps = self.episode_end_timestep() + 1
        if end_idx < 0:
            end_idx = episode_num_steps + end_idx
        start_idx = max(int(start_idx), 0)
        end_idx = min(int(end_idx), self.episode_end_timestep())
        if start_idx > end_idx:
            raise ValueError(
                f"Invalid reset_cfg.start_range after clamping: ({start_idx}, {end_idx})."
            )
        return start_idx, end_idx

    def _initialize_reverse_curriculum(self) -> None:
        configured_start, configured_end = self._clamp_start_range(
            self.cfg.reset_cfg.start_range
        )
        self._configured_start_range = (configured_start, configured_end)
        if not self.cfg.reset_cfg.reverse_curriculum:
            self._curriculum_start_range = None
            return

        window_size = int(self.cfg.reset_cfg.reverse_curriculum_window_size)
        if window_size <= 0:
            raise ValueError(
                "Expected reset_cfg.reverse_curriculum_window_size > 0, got "
                f"{self.cfg.reset_cfg.reverse_curriculum_window_size}."
            )
        current_start = max(configured_end - window_size + 1, configured_start)
        self._curriculum_start_range = (current_start, configured_end)
        self._last_curriculum_shift_policy_iter = 0

    def reference_update_anchors(
        self, anchors: dict[int, dict[str, Any]] | None = None
    ) -> dict[int, dict[str, Any]] | None:
        # The reverse curriculum now expands the sampling window backward while
        # keeping the late end of the trajectory inside the active reset range.
        # Because later timesteps are still sampled directly, we do not need an
        # extra "solved suffix" anchor injection here.
        return anchors

    def maybe_advance_reverse_curriculum(
        self,
        *,
        policy_iter: int,
        reference_update_stats: dict[str, Any] | None,
    ) -> dict[str, float]:
        stats: dict[str, float] = {}
        if not self.cfg.reset_cfg.reverse_curriculum:
            return stats
        if self._configured_start_range is None or self._curriculum_start_range is None:
            raise ValueError("Reverse curriculum was not initialized.")

        configured_start, _ = self._configured_start_range
        current_start, current_end = self._curriculum_start_range
        stats["reverse_curriculum_start_idx"] = float(current_start)
        stats["reverse_curriculum_end_idx"] = float(current_end)
        stats["reverse_curriculum_shifted"] = 0.0
        if current_start <= configured_start:
            return stats

        interval = int(self.cfg.reset_cfg.reverse_curriculum_policy_iter_interval)
        if interval <= 0:
            raise ValueError(
                "Expected reset_cfg.reverse_curriculum_policy_iter_interval > 0, got "
                f"{self.cfg.reset_cfg.reverse_curriculum_policy_iter_interval}."
            )
        variance_threshold = float(
            self.cfg.reset_cfg.reverse_curriculum_variance_threshold
        )
        if variance_threshold < 0.0:
            raise ValueError(
                "Expected reset_cfg.reverse_curriculum_variance_threshold >= 0, got "
                f"{self.cfg.reset_cfg.reverse_curriculum_variance_threshold}."
            )

        iterations_since_shift = policy_iter - self._last_curriculum_shift_policy_iter
        shift_due_to_interval = iterations_since_shift >= interval
        max_gp_variance = self._extract_max_gp_variance(reference_update_stats)
        shift_due_to_variance = (
            max_gp_variance is not None and max_gp_variance <= variance_threshold
        )
        stats["reverse_curriculum_max_gp_variance"] = float(
            np.nan if max_gp_variance is None else max_gp_variance
        )
        if not (shift_due_to_interval or shift_due_to_variance):
            return stats

        shift_size = int(self.cfg.reset_cfg.reverse_curriculum_shift_size)
        if shift_size <= 0:
            raise ValueError(
                "Expected reset_cfg.reverse_curriculum_shift_size > 0, got "
                f"{self.cfg.reset_cfg.reverse_curriculum_shift_size}."
            )
        new_start = max(configured_start, current_start - shift_size)
        new_end = current_end
        self._curriculum_start_range = (new_start, new_end)
        self._last_curriculum_shift_policy_iter = policy_iter
        stats["reverse_curriculum_start_idx"] = float(new_start)
        stats["reverse_curriculum_end_idx"] = float(new_end)
        stats["reverse_curriculum_shifted"] = 1.0
        stats["reverse_curriculum_shift_reason_interval"] = float(shift_due_to_interval)
        stats["reverse_curriculum_shift_reason_variance"] = float(shift_due_to_variance)
        return stats

    def _extract_max_gp_variance(
        self, reference_update_stats: dict[str, Any] | None
    ) -> float | None:
        if not reference_update_stats:
            return None
        variance_values = [
            float(value)
            for key, value in reference_update_stats.items()
            if key.endswith("_variance")
        ]
        if len(variance_values) == 0:
            return None
        return max(variance_values)

    def _sample_uniform_start_timesteps(self, n_samples: int) -> np.ndarray:
        start_idx, end_idx = self._resolve_start_range()
        return np.random.randint(start_idx, end_idx + 1, size=n_samples)

    def _phase_intervals_in_start_range(
        self, start_idx: int, end_idx: int
    ) -> dict[str, tuple[int, int] | None]:
        if not is_oakink_dataset(self.env.task_cfg.demo_dataset):
            return {"reach": None, "transition": None, "lift": None}

        original_demo = self.task.original_demo
        transition_frames = int(self.cfg.reset_cfg.reach_to_lift_transition_frames)
        lift_start = min(
            max(int(original_demo.n_reach_frames), 0),
            self.reference_frame_end_timestep() + 1,
        )
        transition_start = max(lift_start - transition_frames, 0)

        def clamp_interval(lo: int, hi: int) -> tuple[int, int] | None:
            lo = max(lo, start_idx)
            hi = min(hi, end_idx)
            if lo > hi:
                return None
            return (lo, hi)

        return {
            "reach": clamp_interval(0, transition_start - 1),
            "transition": clamp_interval(transition_start, lift_start - 1),
            "lift": clamp_interval(lift_start, self.episode_end_timestep()),
        }

    def _sample_phase_weighted_start_timesteps(self, n_samples: int) -> np.ndarray:
        start_idx, end_idx = self._resolve_start_range()
        phase_probs = self.cfg.reset_cfg.start_phase_probs
        if (phase_probs is None) or (
            not is_oakink_dataset(self.env.task_cfg.demo_dataset)
        ):
            return self._sample_uniform_start_timesteps(n_samples)
        probs = np.asarray(phase_probs, dtype=np.float64)
        phase_names = ("reach", "transition", "lift")
        intervals = self._phase_intervals_in_start_range(start_idx, end_idx)
        active_probs = np.array(
            [
                probs[idx] if intervals[phase_name] is not None else 0.0
                for idx, phase_name in enumerate(phase_names)
            ],
            dtype=np.float64,
        )
        active_probs = active_probs / active_probs.sum()

        sampled_phase_indices = np.random.choice(
            len(phase_names), size=n_samples, p=active_probs
        )
        sampled_timesteps = np.empty(n_samples, dtype=np.int64)
        for phase_idx, phase_name in enumerate(phase_names):
            mask = sampled_phase_indices == phase_idx
            if not np.any(mask):
                continue
            interval = intervals[phase_name]
            lo, hi = interval
            sampled_timesteps[mask] = np.random.randint(
                lo, hi + 1, size=int(mask.sum())
            )
        return sampled_timesteps

    def sample_start_timesteps(self, env_ids: np.ndarray) -> np.ndarray:
        if self.env.mode == "eval":
            sampled = np.full(len(env_ids), self.cfg.reset_cfg.eval_start_timestep)
            return self._resolve_refinement_reset_timesteps(sampled)
        mode = str(self.cfg.reset_cfg.reset_timestep_sampling_mode)
        if mode in {"default", "phase_weighted"}:
            sampled = self._sample_phase_weighted_start_timesteps(len(env_ids))
        elif mode == "uniform":
            sampled = self._sample_uniform_start_timesteps(len(env_ids))
        else:
            raise ValueError(
                "Expected reset_cfg.reset_timestep_sampling_mode to be one of "
                "{'default', 'phase_weighted', 'uniform'}, got "
                f"{mode!r}."
            )
        return self._resolve_refinement_reset_timesteps(sampled)

    def hand_reset_noise(self, frame_idx: np.ndarray) -> torch.Tensor:
        frame_idx_array = np.asarray(frame_idx, dtype=np.int64)
        if frame_idx_array.ndim != 1:
            raise ValueError(
                "Expected frame_idx to be a 1D array of frame indices, "
                f"got shape {frame_idx_array.shape}."
            )

        lift_noise = float(self.cfg.reset_cfg.hand_noise_lift_phase)
        if not is_oakink_dataset(self.env.task_cfg.demo_dataset):
            return torch.full(
                (len(frame_idx_array),),
                lift_noise,
                dtype=torch.float32,
                device=self.env.device,
            )

        original_demo = self.task.original_demo
        if not hasattr(original_demo, "n_reach_frames"):
            raise ValueError(
                "OakInk demo is missing n_reach_frames required for phase-wise hand noise."
            )

        reach_noise = float(self.cfg.reset_cfg.hand_noise_reach_phase)
        transition_frames = int(self.cfg.reset_cfg.reach_to_lift_transition_frames)
        if transition_frames < 0:
            raise ValueError(
                "Expected reset_cfg.reach_to_lift_transition_frames to be non-negative, "
                f"got {transition_frames}."
            )

        frame_idx_t = torch.as_tensor(
            frame_idx_array,
            dtype=torch.float32,
            device=self.env.device,
        )
        reach_end = int(original_demo.n_reach_frames)
        transition_start = max(reach_end - transition_frames, 0)

        noise = torch.full(
            (len(frame_idx_array),),
            lift_noise,
            dtype=torch.float32,
            device=self.env.device,
        )
        if transition_frames == 0:
            reach_phase = frame_idx_t < float(reach_end)
            noise = torch.where(
                reach_phase,
                torch.full_like(noise, reach_noise),
                noise,
            )
            return noise

        reach_phase = frame_idx_t < float(transition_start)
        transition_phase = (frame_idx_t >= float(transition_start)) & (
            frame_idx_t < float(reach_end)
        )
        noise = torch.where(reach_phase, torch.full_like(noise, reach_noise), noise)
        transition_progress = (
            frame_idx_t[transition_phase] - float(transition_start)
        ) / float(transition_frames)
        noise[transition_phase] = reach_noise + transition_progress * (
            lift_noise - reach_noise
        )
        return noise

    def set_object_state_from_retargeted_results(
        self, frame_idx: np.ndarray, env_ids: np.ndarray
    ):
        noise = self.cfg.reset_cfg.obj_noise
        pos = self.frame_pos[frame_idx].to(self.env.device)
        quat = self.frame_quat[frame_idx].to(self.env.device)
        jpos_tensor = (
            None
            if self.frame_arti is None
            else self.frame_arti[frame_idx].to(self.env.device)
        )

        if noise > 0.0:
            pos += torch.randn_like(pos) * noise
            quat_noise = torch.randn_like(quat) * noise
            quat += quat_noise
            quat = quat / torch.norm(quat, dim=-1, keepdim=True)
            if jpos_tensor is not None:
                jpos_tensor += torch.randn_like(jpos_tensor) * noise

        asset: ArtObject = self.physics.scene_entities["object"]
        asset.set_state(pos=pos, quat=quat, jpos=jpos_tensor, env_idx=env_ids)
        return pos, quat, jpos_tensor

    def set_object_state_for_eval(
        self,
        frame_idx: np.ndarray,
        env_ids: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self.set_object_state_from_retargeted_results(
            frame_idx,
            env_ids,
        )

    def set_robot_state_from_retargeted_results(
        self,
        frame_idx: np.ndarray,
        env_ids: np.ndarray,
        random_actions: Dict[str, torch.Tensor] | None = None,
    ):
        noise = self.hand_reset_noise(frame_idx)
        # print(
        #     "[ResetManager.set_robot_state_from_retargeted_results] "
        #     f"env_ids.shape={np.asarray(env_ids).shape}, "
        #     f"env_ids={np.asarray(env_ids)}, "
        #     f"frame_idx.shape={np.asarray(frame_idx).shape}, "
        #     f"frame_idx={np.asarray(frame_idx)}"
        # )
        for side in self.task.active_hand_sides:
            jpos = getattr(self, f"frame_{side}_reset_jvals")[frame_idx].to(
                self.env.device
            )
            # print(
            #     "[ResetManager.set_robot_state_from_retargeted_results] "
            #     f"side={side}, raw_jpos.shape={tuple(jpos.shape)}"
            # )
            jpos = self._apply_hand_reset_noise(side=side, jpos=jpos, noise=noise)

            robot: Hand = self.physics.scene_entities[side]

            if self.cfg.reset_cfg.wrist_only_retargeting:
                root_joints = robot.root_joints
                if len(root_joints) == 0:
                    root_joints = robot.panda_joints
                    assert (
                        len(root_joints) > 0
                    ), f"No root joints found for wrist-only retargeting for robot {robot}"

                jpos_mask = torch.tensor(
                    [jname in root_joints for jname in robot.joint_names],
                    dtype=torch.bool,
                    device=self.env.device,
                ).unsqueeze(0)
                jpos = torch.where(jpos_mask, jpos, torch.zeros_like(jpos))

            # print(
            #     "[ResetManager.set_robot_state_from_retargeted_results] "
            #     f"side={side}, final_jpos.shape={tuple(jpos.shape)}, "
            #     f"expected_rows={len(env_ids)}"
            # )
            robot.set_robot_state(jpos=jpos, env_idx=env_ids)

    def _get_demo_object_pcd_center(self, frame_idx: np.ndarray) -> torch.Tensor:
        frame_idx_array = np.asarray(frame_idx, dtype=np.int64)
        pos = self.frame_pos[frame_idx_array].to(self.env.device)
        quat = self.frame_quat[frame_idx_array].to(self.env.device)
        if self.frame_arti is None:
            arti = torch.empty(
                (len(frame_idx_array), 0),
                dtype=torch.float32,
                device=self.env.device,
            )
        else:
            arti = self.frame_arti[frame_idx_array].to(self.env.device)

        demo_object_pcd = self.env.observation_manager.compute_object_pcd_obs(
            pos=pos,
            quat=quat,
            arti=arti,
            normals=False,
        )
        return demo_object_pcd.mean(dim=1)

    def reset_robot_state_for_eval(
        self,
        frame_idx: np.ndarray,
        env_ids: np.ndarray,
    ) -> None:

        print("Moving robot to demo-relative wrist pose for evaluations")
        if (
            self.env.cfg.backend == "sim"
            and self.cfg.reset_cfg.retargeted_hand_reference_mode
            == "default_configuration"
        ):
            self.set_robot_state_from_retargeted_results(frame_idx, env_ids)
            print("Robot moved to default-configuration eval reset pose.")
            return
        if not self.task.has_hand("right"):
            raise ValueError(
                "Evaluation reset requires an active right hand for action-manager wrist control."
            )

        right_robot: Hand = self.physics.scene_entities["right"]
        frame_right_jvals = getattr(self, "frame_right_reset_jvals", None)
        frame_right_wrist_pos = getattr(self, "frame_right_wrist_pos", None)
        frame_right_wrist_quat = getattr(self, "frame_right_wrist_quat", None)
        if (
            frame_right_jvals is None
            or frame_right_wrist_pos is None
            or frame_right_wrist_quat is None
        ):
            raise ValueError(
                "Evaluation reset requires retargeted right-hand joints and wrist pose."
            )

        if self.env.cfg.backend == "sim":
            # The object state has already been written for this reset; advance physics once
            # so policy object-point-cloud queries see the updated object configuration.
            self.env.physics_manager.step()

        self.env.observation_manager.reset_step_cache()
        reset_obs = self.env.observation_manager.compute_observations()
        named_policy_obs = self.env.observation_manager.named_view_policy(
            reset_obs["policy"]
        )
        object_pcd_rel = self.env.observation_manager.get_object_pcd_from_named_obs(
            named_policy_obs,
            expected_feat_dim=self.env.observation_manager.policy_object_pcd_dim,
        )
        current_wrist_pos = self.env.current_wrist_pos_by_side.get("right")
        if current_wrist_pos is None:
            raise ValueError(
                "Evaluation reset requires env.current_wrist_pos after computing observations."
            )

        current_qpos = self.physics.get_hand_dof_positions(
            right_robot, right_robot.dof_map
        ).detach()
        if self.env.cfg.backend == "real":
            hand_pcd_world = self.physics.get_policy_hand_pcd(
                right_robot,
                links_to_include=right_robot.hand_tree_link_names,
            )
        else:
            hand_pcd_world = right_robot.get_robot_pcd(
                current_qpos,
                links_to_include=right_robot.hand_tree_link_names,
            )
        hand_pcd_rel = hand_pcd_world - current_wrist_pos[:, None, :]
        object_pcd_world = object_pcd_rel + current_wrist_pos[:, None, :]

        noise = self.hand_reset_noise(frame_idx)
        right_jpos = frame_right_jvals[frame_idx].to(self.env.device)
        right_jpos = self._apply_hand_reset_noise(
            side="right",
            jpos=right_jpos,
            noise=noise,
        )

        finger_target_indices = torch.as_tensor(
            self.env.action_manager.hand_target_indices,
            dtype=torch.long,
            device=self.env.device,
        )
        target_hand_qpos = right_jpos.index_select(1, finger_target_indices)
        demo_object_pcd_center = self._get_demo_object_pcd_center(frame_idx)
        demo_wrist_offset = (
            frame_right_wrist_pos[frame_idx].to(self.env.device)
            - demo_object_pcd_center
        )
        target_wrist_quat = frame_right_wrist_quat[frame_idx].to(self.env.device)
        target_wrist_pos = torch.empty(
            (len(env_ids), 3),
            dtype=torch.float32,
            device=self.env.device,
        )

        exclusion_radius = float(self.EVAL_RESET_HAND_EXCLUSION_RADIUS_M)
        for row, env_id in enumerate(env_ids.tolist()):
            point_to_hand_dist = torch.cdist(
                object_pcd_rel[env_id],
                hand_pcd_rel[env_id],
            )
            keep_mask = point_to_hand_dist.min(dim=1).values > exclusion_radius
            filtered_object_pcd = object_pcd_world[env_id][keep_mask]
            if filtered_object_pcd.shape[0] == 0:
                raise ValueError(
                    "Evaluation reset removed all object PCD points after filtering points "
                    f"within {exclusion_radius:.3f} m of the hand for env {env_id}."
                )
            object_pcd_center = filtered_object_pcd.mean(dim=0)
            target_wrist_pos[row] = object_pcd_center + demo_wrist_offset[row]

        if self.env.cfg.backend == "real" and not self.env.real_cfg.initialize_robot:
            print(
                "Skipping real reset motion because env.real_cfg.initialize_robot=False."
            )
            return

        self.env.action_manager.move_to_reset_target(
            env_ids=env_ids,
            target_wrist_pos=target_wrist_pos,
            target_wrist_quat=target_wrist_quat,
            target_hand_qpos=target_hand_qpos,
        )
        print("Robot moved to demo-relative wrist pose.")

    # ///////////////////////////////// used when no retargeted results /////////////////////////////////
    def set_object_state(
        self, frame_idx: np.ndarray, env_ids: np.ndarray, noise: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:

        obj_trans = self.task.original_demo.object_reference_motion["obj_pos"]
        obj_quat = self.task.original_demo.object_reference_motion["obj_quat"]
        obj_arti = self.task.original_demo.object_reference_motion["obj_arti"]
        joint_names = self.task.original_demo.object_reference_motion["joint_names"]

        pos = torch.from_numpy(obj_trans[frame_idx]).to(self.env.device)
        quat = torch.from_numpy(obj_quat[frame_idx]).to(self.env.device)
        if obj_arti is not None and len(joint_names) > 0:
            cols = [
                torch.from_numpy(obj_arti[:, i][frame_idx])
                .to(self.env.device)
                .unsqueeze(-1)
                for i, _ in enumerate(joint_names)
            ]
            jpos_tensor = torch.cat(cols, dim=-1)
        else:
            jpos_tensor = None

        if noise > 0.0:
            pos += torch.randn_like(pos) * noise
            # For quaternion, add noise and renormalize
            quat_noise = torch.randn_like(quat) * noise
            quat += quat_noise
            quat = quat / torch.norm(quat, dim=-1, keepdim=True)
            if jpos_tensor is not None:
                jpos_tensor += torch.randn_like(jpos_tensor) * noise

        asset: ArtObject = self.physics.scene_entities["object"]
        asset.set_state(pos=pos, quat=quat, jpos=jpos_tensor, env_idx=env_ids)
        return pos, quat, jpos_tensor

    def set_robot_state(
        self,
        frame_idx: np.ndarray,
        env_ids: np.ndarray,
        random_actions: Dict[str, torch.Tensor] | None = None,
        noise: float = 0.0,
    ) -> None:
        for hand_side, robot in self.physics.scene_entities.items():
            if not isinstance(robot, Hand):
                continue
            robot: Hand
            assert (
                random_actions is not None
            ), "random_actions must be provided when not resetting with reference motion"
            jpos = random_actions[hand_side].to(self.env.device)
            robot.set_robot_state(jpos=jpos, env_idx=env_ids)

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////
