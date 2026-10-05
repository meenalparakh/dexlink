from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import torch
from dtw import dtw

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dexlink.utils.retargeting.grasp_eval_utils import (
    _robot_keypoint_dict_for_frame,
    _skeleton_angles,
)
from dexlink.utils.retargeting.math_utils import quat_error_magnitude


@dataclass(frozen=True)
class FinalRolloutMetricsConfig:
    lift_success_threshold_m: float = 0.15
    compute_template_metrics: bool = True
    dtw_window_fraction: float = 0.10
    dtw_plot_dir: Path | None = None
    dtw_plot_max_episodes: int = 50


class FinalRolloutMetrics:
    CONFIDENCE_LEVEL = 0.95
    CONFIDENCE_BOOTSTRAP_SAMPLES = 10_000
    CONFIDENCE_BOOTSTRAP_SEED = 0

    def __init__(self, env: Any, cfg: FinalRolloutMetricsConfig) -> None:
        self.env = env
        self.cfg = cfg
        self.reward_cfg = env.cfg.rewards_cfg
        self.object_ids = self._object_ids()
        self.object_static = self._object_static()
        self.ref_object_pos = self._reference_object_array("pos")
        self.ref_object_quat = self._reference_object_array("quat")
        self.ref_object_arti = self._reference_object_array("arti")
        self.object_active_mask = self._object_active_mask()
        self.ref_object_arti_valid_mask = self._reference_object_arti_valid_mask()
        self.active_hand_sides = tuple(env.task_manager.active_hand_sides)
        self.ref_hand_jvals = self._reference_hand_jvals()
        self.finger_joint_indices = self._finger_joint_indices()
        self.ref_hand_wrist_pos = self._reference_hand_wrist_attr("pos")
        self.ref_hand_wrist_quat = self._reference_hand_wrist_attr("quat")

    def compute_episode_metrics(
        self,
        episode: dict[str, Any],
        *,
        episode_index: int | None = None,
    ) -> dict[str, float]:
        start_timestep = int(episode["start_timestep"])
        length = int(episode["length"])
        if length <= 0:
            raise ValueError(f"Expected positive episode length, got {length}.")
        ref_slice = slice(start_timestep, self.ref_object_pos.shape[0])
        ref_indices = np.clip(
            np.asarray(episode["command_timestep"][:length], dtype=np.int64),
            0,
            self.ref_object_pos.shape[0] - 1,
        )
        moving_indices = self._moving_object_indices()
        articulated_indices = self._articulated_object_indices()

        metrics: dict[str, float] = {
            "episode_length": float(length),
        }
        metrics.update(
            self._object_dtw_metrics(
                episode,
                length,
                ref_slice,
                pose_object_indices=moving_indices,
                articulation_object_indices=articulated_indices,
                episode_index=episode_index,
            )
        )
        metrics.update(
            self._object_deviation_metrics(episode, length, ref_indices, moving_indices)
        )
        metrics.update(
            self._hand_dtw_metrics(
                episode,
                length,
                ref_slice,
                episode_index=episode_index,
            )
        )
        would_terminate_metrics = self._would_terminate_metrics(
            episode,
            length,
            ref_indices,
            moving_indices,
        )
        metrics.update(would_terminate_metrics)
        metrics.update(self._final_scene_configuration_metrics(episode, length))
        metrics.update(
            self._task_metrics(
                episode,
                length,
                would_terminate=bool(would_terminate_metrics["would_terminate"]),
            )
        )
        metrics.update(self._eval_only_task_metrics(episode, length))
        metrics.update(
            self._lift_metrics(episode, length, start_timestep, moving_indices)
        )
        if self.cfg.compute_template_metrics:
            metrics.update(
                self._template_angle_metrics(
                    episode=episode,
                    length=length,
                    ref_final_timestep=self.ref_object_pos.shape[0] - 1,
                )
            )
        return metrics

    def _eval_only_task_metrics(
        self,
        episode: dict[str, Any],
        length: int,
    ) -> dict[str, float]:
        eval_metric_id = self.env.task_manager.task_cfg.metric_cfg.eval_metric_id
        if eval_metric_id is None:
            return {}
        if eval_metric_id != "mug_grasp_change":
            raise ValueError(
                f"Unsupported evaluation-only task metric {eval_metric_id!r}."
            )
        return self._mug_grasp_change_metrics(episode, length)

    def _mug_grasp_change_metrics(
        self,
        episode: dict[str, Any],
        length: int,
    ) -> dict[str, float]:
        cfg = self.env.task_manager.task_cfg.metric_cfg
        side = cfg.mug_hand_side
        hand = self.env.physics_manager.scene_entities[side]
        keypoint_names = tuple(hand.robo_cfg["human_robot_link_map"].keys())
        phase_names = tuple(cfg.mug_reference_grasp_phase_names)
        reference_indices = self._reference_phase_midpoints(phase_names)

        rollout_jvals = np.asarray(episode[f"{side}_jvals"][:length])
        rollout_angles = np.stack(
            [
                self._robot_skeleton_angles(
                    hand=hand,
                    joint_values=joint_values,
                    keypoint_names=keypoint_names,
                )
                for joint_values in rollout_jvals
            ],
            axis=0,
        )
        reference_angles = np.stack(
            [
                self._robot_skeleton_angles(
                    hand=hand,
                    joint_values=self.ref_hand_jvals[side][frame_idx],
                    keypoint_names=keypoint_names,
                )
                for frame_idx in reference_indices
            ],
            axis=0,
        )
        angle_squared_error = (
            rollout_angles[:, None, :] - reference_angles[None, :, :]
        ) ** 2
        wrist_error = self._quat_pairwise_error_np(
            np.asarray(episode[f"{side}_wrist_quat"][:length]),
            self.ref_hand_wrist_quat[side][np.asarray(reference_indices)],
        )
        grasp_cost = np.sqrt(
            (
                angle_squared_error.sum(axis=-1)
                + np.asarray(wrist_error, dtype=np.float64) ** 2
            )
            / float(rollout_angles.shape[1] + 1)
        )

        mug_index = self.object_ids.index(cfg.mug_object_id)
        table_top_height = self.env.physics_manager.get_scene_table_top_height()
        mug_height = (
            np.asarray(episode["object_pos"][:length, mug_index, 2]) - table_top_height
        )
        airborne = mug_height >= float(cfg.mug_min_center_height_above_table_m)
        match = self._best_two_grasp_monotonic_match(grasp_cost, airborne)
        early_terminated = bool(np.asarray(episode["terminated"])[length - 1])
        if match is None or early_terminated:
            return {
                "mug_two_grasp_sequential_error_rad": float("nan"),
                "mug_two_grasp_sequential_similarity": 0.0,
                "mug_two_grasp_airborne_pair_valid": 0.0,
                "mug_top_grasp_match_timestep": -1.0,
                "mug_side_grasp_match_timestep": -1.0,
            }

        first_idx, second_idx, error = match
        return {
            "mug_two_grasp_sequential_error_rad": error,
            "mug_two_grasp_sequential_similarity": float(np.exp(-error)),
            "mug_two_grasp_airborne_pair_valid": 1.0,
            "mug_top_grasp_match_timestep": float(first_idx),
            "mug_side_grasp_match_timestep": float(second_idx),
        }

    def _reference_phase_midpoints(
        self,
        requested_phase_names: tuple[str, str],
    ) -> tuple[int, int]:
        phase_names = list(self.env.task_manager.phase_names)
        phase_lengths = np.asarray(
            self.env.task_manager.phase_lengths,
            dtype=np.int64,
        )
        starts = np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(phase_lengths[:-1])]
        )
        indices = []
        for name in requested_phase_names:
            matches = [
                idx for idx, phase_name in enumerate(phase_names) if phase_name == name
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one reference phase named {name!r}, got "
                    f"matches {matches} in {phase_names}."
                )
            phase_idx = matches[0]
            indices.append(
                int(starts[phase_idx] + (int(phase_lengths[phase_idx]) - 1) // 2)
            )
        if indices[0] >= indices[1]:
            raise ValueError(
                "Mug reference grasp phases must be sequential, got midpoint "
                f"indices {indices}."
            )
        return indices[0], indices[1]

    @staticmethod
    def _best_two_grasp_monotonic_match(
        grasp_cost: np.ndarray,
        airborne: np.ndarray,
    ) -> tuple[int, int, float] | None:
        grasp_cost = np.asarray(grasp_cost, dtype=np.float64)
        airborne = np.asarray(airborne, dtype=bool)
        if grasp_cost.ndim != 2 or grasp_cost.shape[1] != 2:
            raise ValueError(
                f"Expected two-reference grasp cost shape (T, 2), got {grasp_cost.shape}."
            )
        if airborne.shape != (grasp_cost.shape[0],):
            raise ValueError(
                f"Expected airborne mask shape {(grasp_cost.shape[0],)}, "
                f"got {airborne.shape}."
            )

        best_first_idx = -1
        best_first_cost = float("inf")
        best_match: tuple[int, int, float] | None = None
        for second_idx in range(grasp_cost.shape[0]):
            first_candidate = second_idx - 1
            if first_candidate >= 0 and airborne[first_candidate]:
                candidate_cost = float(grasp_cost[first_candidate, 0])
                if candidate_cost < best_first_cost:
                    best_first_cost = candidate_cost
                    best_first_idx = first_candidate
            if not airborne[second_idx] or best_first_idx < 0:
                continue
            error = 0.5 * (best_first_cost + float(grasp_cost[second_idx, 1]))
            if best_match is None or error < best_match[2]:
                best_match = (best_first_idx, second_idx, error)
        return best_match

    def _task_metrics(
        self,
        episode: dict[str, Any],
        length: int,
        *,
        would_terminate: bool,
    ) -> dict[str, float]:
        metrics = {}
        primary_name = self.env.task_manager.primary_task_metric_name
        early_terminated = (
            bool(np.asarray(episode["terminated"])[length - 1]) or would_terminate
        )
        for name in self.env.task_manager.task_metric_names:
            key = f"task_metric__{name}"
            if key not in episode:
                raise KeyError(
                    f"Episode is missing configured task metric {key!r}; available "
                    f"keys are {sorted(episode)}."
                )
            values = np.asarray(episode[key], dtype=np.float64)
            if values.shape != (length,):
                raise ValueError(
                    f"Expected episode task metric {key!r} shape {(length,)}, "
                    f"got {values.shape}."
                )
            value = float(values[length - 1])
            if early_terminated and name == primary_name:
                value = 0.0
            metrics[name] = value
        return metrics

    @staticmethod
    def summarize(episode_metrics: list[dict[str, float]]) -> dict[str, float]:
        if not episode_metrics:
            raise ValueError("Expected at least one episode metric row.")
        keys = sorted({key for row in episode_metrics for key in row})
        summary: dict[str, float] = {"num_episodes": float(len(episode_metrics))}
        for key in keys:
            values = np.asarray(
                [row[key] for row in episode_metrics if key in row],
                dtype=np.float64,
            )
            if values.size == 0:
                continue
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            summary[f"{key}_mean"] = float(np.mean(finite))
            summary[f"{key}_std"] = float(np.std(finite))
            summary[f"{key}_median"] = float(np.median(finite))
            ci_low, ci_high = FinalRolloutMetrics._bootstrap_mean_confidence_interval(
                finite
            )
            summary[f"{key}_ci95_low"] = ci_low
            summary[f"{key}_ci95_high"] = ci_high

        # Hammer impact is physically meaningful only when the authored strike
        # occurs. Keep the all-episode statistic above for auditing, and add a
        # success-conditioned statistic for reporting strike strength.
        completion_key = "hammer_single_swing_completed"
        impact_key = "hammer_peg_average_contact_impact_n"
        if completion_key in keys:
            if not all(
                completion_key in row and impact_key in row
                for row in episode_metrics
            ):
                raise ValueError(
                    "Hammer strike-conditioned impact requires both "
                    f"{completion_key!r} and {impact_key!r} in every episode row."
                )
            completion = np.asarray(
                [row[completion_key] for row in episode_metrics], dtype=np.float64
            )
            impact = np.asarray(
                [row[impact_key] for row in episode_metrics], dtype=np.float64
            )
            if not np.isfinite(completion).all() or not np.isin(
                completion, (0.0, 1.0)
            ).all():
                raise ValueError(
                    "Hammer completion values must be finite binary indicators, "
                    f"got {completion}."
                )
            if not np.isfinite(impact).all() or (impact < 0.0).any():
                raise ValueError(
                    "Hammer impact values must be finite and non-negative, got "
                    f"{impact}."
                )
            successful_impact = impact[completion == 1.0]
            metric_name = "hammer_peg_average_contact_impact_upon_strike_n"
            summary[f"{metric_name}_episode_count"] = float(successful_impact.size)
            if successful_impact.size > 0:
                summary[f"{metric_name}_mean"] = float(successful_impact.mean())
                summary[f"{metric_name}_std"] = float(successful_impact.std())
                summary[f"{metric_name}_median"] = float(
                    np.median(successful_impact)
                )
                ci_low, ci_high = (
                    FinalRolloutMetrics._bootstrap_mean_confidence_interval(
                        successful_impact
                    )
                )
                summary[f"{metric_name}_ci95_low"] = ci_low
                summary[f"{metric_name}_ci95_high"] = ci_high
        return summary

    @staticmethod
    def _bootstrap_mean_confidence_interval(
        values: np.ndarray,
    ) -> tuple[float, float]:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(
                "Expected a non-empty rank-1 array for confidence intervals, "
                f"got shape {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Confidence interval values must all be finite.")
        rng = np.random.default_rng(FinalRolloutMetrics.CONFIDENCE_BOOTSTRAP_SEED)
        sample_indices = rng.integers(
            0,
            values.size,
            size=(FinalRolloutMetrics.CONFIDENCE_BOOTSTRAP_SAMPLES, values.size),
        )
        bootstrap_means = values[sample_indices].mean(axis=1)
        tail = (1.0 - FinalRolloutMetrics.CONFIDENCE_LEVEL) / 2.0
        ci_low, ci_high = np.quantile(
            bootstrap_means,
            [tail, 1.0 - tail],
        )
        return float(ci_low), float(ci_high)

    def _object_ids(self) -> tuple[str, ...]:
        task = self.env.task_manager
        if hasattr(task, "scene_object_ids"):
            return tuple(str(object_id) for object_id in task.scene_object_ids)
        return ("object",)

    def _object_static(self) -> np.ndarray:
        task = self.env.task_manager
        if hasattr(task, "scene_object_static"):
            static = np.asarray(task.scene_object_static, dtype=bool)
        else:
            motion = self.env.task_manager.original_demo.retargeted_motion
            static = np.asarray([bool(motion.get("object_static", False))], dtype=bool)
        if static.shape != (len(self.object_ids),):
            raise ValueError(
                f"Expected object_static shape {(len(self.object_ids),)}, got {static.shape}."
            )
        return static

    def _object_active_mask(self) -> np.ndarray:
        task = self.env.task_manager
        if hasattr(task, "scene_object_active_mask"):
            active = np.asarray(task.scene_object_active_mask, dtype=bool)
        else:
            active = np.ones(
                (self.ref_object_pos.shape[0], len(self.object_ids)), dtype=bool
            )
        expected_shape = (self.ref_object_pos.shape[0], len(self.object_ids))
        if active.shape != expected_shape:
            raise ValueError(
                f"Expected object_active_mask shape {expected_shape}, got {active.shape}."
            )
        return active

    def _reference_object_array(self, attr: str) -> np.ndarray:
        task = self.env.task_manager
        if hasattr(task, f"scene_object_{attr}"):
            arr = np.asarray(getattr(task, f"scene_object_{attr}"), dtype=np.float32)
        elif attr == "arti" and hasattr(task, "scene_object_arti_vals"):
            arr = np.asarray(task.scene_object_arti_vals, dtype=np.float32)
        else:
            motion = task.original_demo.retargeted_motion
            key = {"pos": "obj_pos", "quat": "obj_quat", "arti": "obj_arti"}[attr]
            arr = np.asarray(motion[key], dtype=np.float32)[:, None, :]
        if arr.ndim != 3:
            raise ValueError(
                f"Expected reference object {attr} shape (T, O, D), got {arr.shape}."
            )
        if arr.shape[1] != len(self.object_ids):
            raise ValueError(
                f"Expected reference object {attr} object dim {len(self.object_ids)}, got {arr.shape[1]}."
            )
        return arr

    def _reference_object_arti_valid_mask(self) -> np.ndarray:
        task = self.env.task_manager
        if hasattr(task, "scene_object_arti_valid_mask"):
            mask = np.asarray(task.scene_object_arti_valid_mask, dtype=bool)
        else:
            mask = np.ones(
                (len(self.object_ids), self.ref_object_arti.shape[-1]),
                dtype=bool,
            )
        expected_shape = (len(self.object_ids), self.ref_object_arti.shape[-1])
        if mask.shape != expected_shape:
            raise ValueError(
                f"Expected articulation valid mask shape {expected_shape}, got {mask.shape}."
            )
        return mask

    def _reference_hand_jvals(self) -> dict[str, np.ndarray]:
        out = {}
        for side in self.active_hand_sides:
            reference_manager = self.env.reference_trajectory_manager
            if reference_manager.is_enabled():
                value = reference_manager.hand_joint_positions(side)
            else:
                value = getattr(self.env.reset_manager, f"frame_{side}_jvals", None)
            if value is None:
                raise ValueError(
                    f"Missing reference hand joint values for side {side!r}."
                )
            arr = np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected reference hand joint values for {side} to have shape (T, J), got {arr.shape}."
                )
            out[side] = arr
        return out

    def _finger_joint_indices(self) -> dict[str, np.ndarray]:
        out = {}
        for side in self.active_hand_sides:
            hand = self.env.physics_manager.scene_entities[side]
            indices = [hand.joint_names.index(name) for name in hand.finger_joint_names]
            if len(indices) == 0:
                raise ValueError(
                    f"Expected at least one finger joint for side {side!r}."
                )
            out[side] = np.asarray(indices, dtype=np.int64)
        return out

    def _reference_hand_wrist_attr(self, attr: str) -> dict[str, np.ndarray]:
        out = {}
        for side in self.active_hand_sides:
            value = getattr(self.env.reset_manager, f"frame_{side}_wrist_{attr}", None)
            if value is None:
                raise ValueError(f"Missing reference wrist {attr} for side {side!r}.")
            out[side] = value.detach().cpu().numpy().astype(np.float32, copy=False)
        return out

    def _moving_object_indices(self) -> np.ndarray:
        indices = np.where(~self.object_static)[0]
        if indices.size == 0:
            return indices.astype(np.int64)
        return indices.astype(np.int64)

    def _articulated_object_indices(self) -> np.ndarray:
        return np.where(np.any(self.ref_object_arti_valid_mask, axis=1))[0].astype(
            np.int64
        )

    def _object_dtw_metrics(
        self,
        episode: dict[str, Any],
        length: int,
        ref_slice: slice,
        pose_object_indices: np.ndarray,
        articulation_object_indices: np.ndarray,
        *,
        episode_index: int | None,
    ) -> dict[str, float]:
        values: dict[str, list[float]] = {
            "object_pos_dtw": [],
            "object_pos_dtw_path_mean": [],
            "object_quat_dtw": [],
            "object_quat_dtw_path_mean": [],
            "object_arti_dtw": [],
            "object_arti_dtw_path_mean": [],
        }
        for object_idx in pose_object_indices:
            object_id = self.object_ids[object_idx]
            dtw_normalized, dtw_path_mean = self._dtw_euclidean_stats(
                episode["object_pos"][:length, object_idx],
                self.ref_object_pos[ref_slice, object_idx],
                plot_path=self._dtw_plot_path(episode_index, f"object_{object_id}_pos"),
                title=f"Episode {episode_index}: {object_id} position",
                window_fraction=self.cfg.dtw_window_fraction,
            )
            values["object_pos_dtw"].append(dtw_normalized)
            values["object_pos_dtw_path_mean"].append(dtw_path_mean)

            dtw_normalized, dtw_path_mean = self._dtw_quat_stats(
                episode["object_quat"][:length, object_idx],
                self.ref_object_quat[ref_slice, object_idx],
                plot_path=self._dtw_plot_path(
                    episode_index, f"object_{object_id}_quat"
                ),
                title=f"Episode {episode_index}: {object_id} quaternion",
                window_fraction=self.cfg.dtw_window_fraction,
            )
            values["object_quat_dtw"].append(dtw_normalized)
            values["object_quat_dtw_path_mean"].append(dtw_path_mean)

        for object_idx in articulation_object_indices:
            object_id = self.object_ids[object_idx]
            valid_arti = self.ref_object_arti_valid_mask[object_idx]
            if not np.any(valid_arti):
                raise ValueError(
                    f"Articulation DTW received non-articulated object {object_id!r}."
                )
            rollout_arti = episode["object_arti"][:length, object_idx][:, valid_arti]
            ref_arti = self.ref_object_arti[ref_slice, object_idx][:, valid_arti]
            dtw_normalized, dtw_path_mean = self._dtw_euclidean_stats(
                rollout_arti,
                ref_arti,
                plot_path=self._dtw_plot_path(
                    episode_index, f"object_{object_id}_arti"
                ),
                title=f"Episode {episode_index}: {object_id} articulation",
                window_fraction=self.cfg.dtw_window_fraction,
            )
            values["object_arti_dtw"].append(dtw_normalized)
            values["object_arti_dtw_path_mean"].append(dtw_path_mean)

        return {
            key: float(np.mean(metric_values))
            for key, metric_values in values.items()
            if metric_values
        }

    def _object_deviation_metrics(
        self,
        episode: dict[str, Any],
        length: int,
        ref_indices: np.ndarray,
        object_indices: np.ndarray,
    ) -> dict[str, float]:
        max_pos = []
        max_quat = []
        max_arti = []
        for object_idx in object_indices:
            active = self.object_active_mask[ref_indices, object_idx]
            if not np.any(active):
                continue
            rollout_pos = episode["object_pos"][:length, object_idx][active]
            ref_pos = self.ref_object_pos[ref_indices, object_idx][active]
            max_pos.append(float(np.linalg.norm(rollout_pos - ref_pos, axis=-1).max()))

            rollout_quat = episode["object_quat"][:length, object_idx][active]
            ref_quat = self.ref_object_quat[ref_indices, object_idx][active]
            max_quat.append(float(self._quat_error_np(rollout_quat, ref_quat).max()))

            valid_arti = self.ref_object_arti_valid_mask[object_idx]
            if np.any(valid_arti):
                rollout_arti = episode["object_arti"][:length, object_idx][
                    :, valid_arti
                ][active]
                ref_arti = self.ref_object_arti[ref_indices, object_idx][:, valid_arti][
                    active
                ]
                max_arti.append(
                    float(np.abs(rollout_arti - ref_arti).sum(axis=-1).max())
                )

        out = {}
        if max_pos:
            out["object_max_pos_error_m"] = float(np.max(max_pos))
        if max_quat:
            out["object_max_quat_error_rad"] = float(np.max(max_quat))
        if max_arti:
            out["object_max_arti_error"] = float(np.max(max_arti))
        return out

    def _hand_dtw_metrics(
        self,
        episode: dict[str, Any],
        length: int,
        ref_slice: slice,
        *,
        episode_index: int | None,
    ) -> dict[str, float]:
        finger = []
        finger_path_mean = []
        wrist_pos = []
        wrist_pos_path_mean = []
        wrist_quat = []
        wrist_quat_path_mean = []
        for side in self.active_hand_sides:
            finger_indices = self.finger_joint_indices[side]
            rollout_jvals = episode[f"{side}_jvals"][:length, finger_indices]
            ref_jvals = self.ref_hand_jvals[side][ref_slice][:, finger_indices]
            if rollout_jvals.shape[-1] != ref_jvals.shape[-1]:
                raise ValueError(
                    f"Hand jval dim mismatch for {side}: rollout {rollout_jvals.shape}, ref {ref_jvals.shape}."
                )
            dtw_normalized, dtw_path_mean = self._dtw_euclidean_stats(
                rollout_jvals,
                ref_jvals,
                plot_path=self._dtw_plot_path(episode_index, f"{side}_finger_q"),
                title=f"Episode {episode_index}: {side} finger q",
                window_fraction=self.cfg.dtw_window_fraction,
            )
            finger.append(dtw_normalized)
            finger_path_mean.append(dtw_path_mean)

            dtw_normalized, dtw_path_mean = self._dtw_euclidean_stats(
                episode[f"{side}_wrist_pos"][:length],
                self.ref_hand_wrist_pos[side][ref_slice],
                plot_path=self._dtw_plot_path(episode_index, f"{side}_wrist_pos"),
                title=f"Episode {episode_index}: {side} wrist position",
                window_fraction=self.cfg.dtw_window_fraction,
            )
            wrist_pos.append(dtw_normalized)
            wrist_pos_path_mean.append(dtw_path_mean)

            dtw_normalized, dtw_path_mean = self._dtw_quat_stats(
                episode[f"{side}_wrist_quat"][:length],
                self.ref_hand_wrist_quat[side][ref_slice],
                plot_path=self._dtw_plot_path(episode_index, f"{side}_wrist_quat"),
                title=f"Episode {episode_index}: {side} wrist quaternion",
                window_fraction=self.cfg.dtw_window_fraction,
            )
            wrist_quat.append(dtw_normalized)
            wrist_quat_path_mean.append(dtw_path_mean)
        out = {}
        if finger:
            out["finger_q_dtw"] = float(np.mean(finger))
            out["finger_q_dtw_path_mean"] = float(np.mean(finger_path_mean))
        if wrist_pos:
            out["wrist_pos_dtw"] = float(np.mean(wrist_pos))
            out["wrist_pos_dtw_path_mean"] = float(np.mean(wrist_pos_path_mean))
        if wrist_quat:
            out["wrist_quat_dtw"] = float(np.mean(wrist_quat))
            out["wrist_quat_dtw_path_mean"] = float(np.mean(wrist_quat_path_mean))
        return out

    def _would_terminate_metrics(
        self,
        episode: dict[str, Any],
        length: int,
        ref_indices: np.ndarray,
        object_indices: np.ndarray,
    ) -> dict[str, float]:
        far_away = np.zeros(length, dtype=bool)
        for object_idx in object_indices:
            active = self.object_active_mask[ref_indices, object_idx]
            pos_error = np.linalg.norm(
                episode["object_pos"][:length, object_idx]
                - self.ref_object_pos[ref_indices, object_idx],
                axis=-1,
            )
            quat_error = self._quat_error_np(
                episode["object_quat"][:length, object_idx],
                self.ref_object_quat[ref_indices, object_idx],
            )
            object_far = (
                pos_error > float(self.reward_cfg.obj_command_max_dist_pos)
            ) | (quat_error > float(self.reward_cfg.obj_command_max_dist_rot))
            valid_arti = self.ref_object_arti_valid_mask[object_idx]
            if np.any(valid_arti):
                arti_error = np.abs(
                    episode["object_arti"][:length, object_idx][:, valid_arti]
                    - self.ref_object_arti[ref_indices, object_idx][:, valid_arti]
                ).sum(axis=-1)
                object_far |= arti_error > float(
                    self.reward_cfg.obj_command_max_dist_arti
                )
            far_away |= active & object_far

        if np.any(far_away):
            first = int(np.where(far_away)[0][0] + 1)
            horizon = max(
                1,
                self.ref_object_pos.shape[0]
                - 1
                + int(self.reward_cfg.last_reference_hold_steps)
                - int(episode["start_timestep"]),
            )
            return {
                "would_terminate": 1.0,
                "would_terminate_timestep": float(first),
                "would_terminate_progress": float(first / horizon),
            }
        return {
            "would_terminate": 0.0,
            "would_terminate_timestep": float(length),
            "would_terminate_progress": 1.0,
        }

    def _final_scene_configuration_metrics(
        self,
        episode: dict[str, Any],
        length: int,
    ) -> dict[str, float]:
        ref_final_timestep = self.ref_object_pos.shape[0] - 1
        rollout_pos = episode["object_pos"][length - 1]
        rollout_quat = episode["object_quat"][length - 1]
        rollout_arti = episode["object_arti"][length - 1]
        ref_pos = self.ref_object_pos[ref_final_timestep]
        ref_quat = self.ref_object_quat[ref_final_timestep]
        ref_arti = self.ref_object_arti[ref_final_timestep]

        pos_error = np.linalg.norm(rollout_pos - ref_pos, axis=-1)
        quat_error = self._quat_error_np(rollout_quat, ref_quat)
        pos_success = pos_error <= float(self.reward_cfg.task_success_pos_thd)
        quat_success = quat_error <= float(self.reward_cfg.task_success_rot_thd)

        has_arti = np.any(self.ref_object_arti_valid_mask, axis=1)
        arti_success = np.ones((len(self.object_ids),), dtype=bool)
        arti_error_sum = np.zeros((len(self.object_ids),), dtype=np.float64)
        for object_idx, object_id in enumerate(self.object_ids):
            valid_arti = self.ref_object_arti_valid_mask[object_idx]
            if not np.any(valid_arti):
                continue
            arti_error = np.abs(
                rollout_arti[object_idx, valid_arti] - ref_arti[object_idx, valid_arti]
            )
            thresholds = self._final_scene_arti_thresholds(
                object_id=object_id,
                valid_arti_count=int(np.sum(valid_arti)),
            )
            arti_success[object_idx] = bool(np.all(arti_error <= thresholds))
            arti_error_sum[object_idx] = float(np.sum(arti_error))

        object_success = pos_success & quat_success & arti_success
        out = {
            "final_scene_configuration_success": float(np.all(object_success)),
            "final_scene_pos_success": float(np.all(pos_success)),
            "final_scene_quat_success": float(np.all(quat_success)),
            "final_scene_max_pos_error_m": float(np.max(pos_error)),
            "final_scene_max_quat_error_rad": float(np.max(quat_error)),
        }
        if np.any(has_arti):
            out["final_scene_arti_success"] = float(np.all(arti_success[has_arti]))
            out["final_scene_max_arti_error"] = float(np.max(arti_error_sum[has_arti]))
        return out

    def _final_scene_arti_thresholds(
        self,
        *,
        object_id: str,
        valid_arti_count: int,
    ) -> np.ndarray:
        if valid_arti_count <= 0:
            raise ValueError(
                f"Expected positive valid articulation count for object {object_id!r}, "
                f"got {valid_arti_count}."
            )
        metric_cfg = self.env.task_manager.task_cfg.metric_cfg
        if (
            metric_cfg.metric_id == "drawer_articulation"
            and object_id == metric_cfg.drawer_object_id
        ):
            if valid_arti_count != 1:
                raise ValueError(
                    "Drawer task metric requires exactly one valid articulation, "
                    f"got {valid_arti_count} for object {object_id!r}."
                )
            return np.asarray(
                [metric_cfg.drawer_final_articulation_tolerance_m],
                dtype=np.float64,
            )
        entity = self._object_entity(object_id)
        joints = list(getattr(entity, "joints", []))
        if len(joints) != valid_arti_count:
            raise ValueError(
                f"Expected {valid_arti_count} simulated articulation joints for object "
                f"{object_id!r}, got {len(joints)}."
            )
        thresholds = []
        for joint in joints:
            joint_type = str(joint.joint_type)
            if joint_type == "prismatic":
                thresholds.append(
                    float(self.reward_cfg.task_success_prismatic_arti_thd)
                )
            elif joint_type in ("revolute", "continuous"):
                thresholds.append(float(self.reward_cfg.task_success_revolute_arti_thd))
            else:
                raise ValueError(
                    f"Unsupported articulation joint type {joint_type!r} for object "
                    f"{object_id!r}; expected prismatic, revolute, or continuous."
                )
        return np.asarray(thresholds, dtype=np.float64)

    def _object_entity(self, object_id: str) -> Any:
        physics = self.env.physics_manager
        if hasattr(physics, "object_entities"):
            object_entities = physics.object_entities
            if object_id not in object_entities:
                raise ValueError(
                    f"Physics object_entities is missing object {object_id!r}; "
                    f"available={sorted(object_entities)}."
                )
            return object_entities[object_id]
        if object_id != "object":
            raise ValueError(
                f"Single-object physics manager cannot resolve object {object_id!r}."
            )
        return physics.scene_entities["object"]

    def _lift_metrics(
        self,
        episode: dict[str, Any],
        length: int,
        start_timestep: int,
        object_indices: np.ndarray,
    ) -> dict[str, float]:
        if object_indices.size == 0:
            return {}
        object_idx = int(object_indices[0])
        initial_ref_z = float(self.ref_object_pos[start_timestep, object_idx, 2])
        final_z = float(episode["object_pos"][length - 1, object_idx, 2])
        lift_height = final_z - initial_ref_z
        return {
            "lift_height_m": float(lift_height),
            "lift_success_15cm": float(
                lift_height >= self.cfg.lift_success_threshold_m
            ),
        }

    def _template_angle_metrics(
        self,
        *,
        episode: dict[str, Any],
        length: int,
        ref_final_timestep: int,
    ) -> dict[str, float]:
        rollout_angles = []
        retargeted_angles = []
        template_angles = []
        processor = self.env.task_manager.original_demo
        for side in self.active_hand_sides:
            hand = self.env.physics_manager.scene_entities[side]
            keypoint_names = tuple(hand.robo_cfg["human_robot_link_map"].keys())
            template = self._template_skeleton_angles(
                processor=processor,
                side=side,
                keypoint_names=keypoint_names,
                frame_idx=ref_final_timestep,
            )
            template_angles.append(template)
            rollout_angles.append(
                self._robot_skeleton_angles(
                    hand=hand,
                    joint_values=episode[f"{side}_jvals"][length - 1],
                    keypoint_names=keypoint_names,
                )
            )
            retargeted_angles.append(
                self._robot_skeleton_angles(
                    hand=hand,
                    joint_values=self.ref_hand_jvals[side][ref_final_timestep],
                    keypoint_names=keypoint_names,
                )
            )
        if not template_angles:
            return {}
        template = np.concatenate(template_angles)
        rollout = np.concatenate(rollout_angles)
        retargeted = np.concatenate(retargeted_angles)
        return {
            "final_template_angle_mse_rad2": float(np.mean((rollout - template) ** 2)),
            "retargeted_template_angle_mse_rad2": float(
                np.mean((retargeted - template) ** 2)
            ),
            "template_angle_valid": 1.0,
        }

    def _template_skeleton_angles(
        self,
        *,
        processor: Any,
        side: str,
        keypoint_names: tuple[str, ...],
        frame_idx: int,
    ) -> np.ndarray:
        template_keypoints = self._demo_skeleton_keypoints_from_human_reference(
            processor=processor,
            side=side,
            keypoint_names=keypoint_names,
            frame_idx=frame_idx,
        )
        return _skeleton_angles(template_keypoints)

    @staticmethod
    def _demo_skeleton_keypoints_from_human_reference(
        *,
        processor: Any,
        side: str,
        keypoint_names: tuple[str, ...],
        frame_idx: int,
    ) -> dict[str, np.ndarray]:
        human_reference_joint_pos = getattr(
            processor,
            f"human_reference_{side}_joint_pos",
            None,
        )
        human_reference_joint_valid = getattr(
            processor,
            f"human_reference_{side}_joint_valid",
            None,
        )
        if human_reference_joint_pos is None and bool(
            getattr(processor, "requires_human_reference_joint_pos_for_eval", False)
        ):
            raise ValueError(
                f"Template angle metrics require human_reference_{side}_joint_pos "
                "for this processor. Regenerate the OakInk sequence export so it "
                f"includes {side}_human_reference_keypoint_pos."
            )
        if isinstance(human_reference_joint_pos, dict):
            if human_reference_joint_valid is not None:
                valid = np.asarray(human_reference_joint_valid, dtype=bool).reshape(-1)
                if frame_idx < 0 or frame_idx >= valid.shape[0]:
                    raise ValueError(
                        f"Template frame index {frame_idx} is out of range for "
                        f"human_reference_{side}_joint_valid with {valid.shape[0]} frames."
                    )
                if not bool(valid[frame_idx]):
                    raise ValueError(
                        f"Template frame {frame_idx} has invalid explicit human "
                        f"reference keypoints for {side} hand."
                    )
            keypoints: dict[str, np.ndarray] = {}
            for name in keypoint_names:
                if name not in human_reference_joint_pos:
                    raise KeyError(
                        f"human_reference_{side}_joint_pos is missing skeleton "
                        f"keypoint {name!r}."
                    )
                value = np.asarray(
                    human_reference_joint_pos[name][frame_idx], dtype=np.float64
                )
                if value.shape != (3,):
                    raise ValueError(
                        f"Expected human_reference_{side}_joint_pos[{name!r}]"
                        f"[{frame_idx}] shape (3,), got {value.shape}."
                    )
                keypoints[name] = value
            return keypoints

        joint_map = getattr(processor, "hand_joint_map", None)
        if not isinstance(joint_map, dict):
            raise ValueError(
                "Template angle metrics require processor.hand_joint_map to map "
                "semantic skeleton names to human joint indices."
            )

        raw_attr = f"raw_relevant_{side}_joint_pos"
        raw_joint_dict = getattr(processor, raw_attr, None)
        if isinstance(raw_joint_dict, dict):
            keypoints = {}
            for name in keypoint_names:
                if name not in raw_joint_dict:
                    raise KeyError(
                        f"Processor {raw_attr} is missing skeleton keypoint {name!r}."
                    )
                value = np.asarray(raw_joint_dict[name][frame_idx], dtype=np.float64)
                if value.shape != (3,):
                    raise ValueError(
                        f"Expected {raw_attr}[{name!r}][{frame_idx}] shape (3,), "
                        f"got {value.shape}."
                    )
                keypoints[name] = value
            return keypoints

        if not hasattr(processor, "world_coord"):
            raise ValueError(
                "Template angle metrics require processor.world_coord or "
                f"processor.{raw_attr}."
            )
        joints_key = f"joints.{side}"
        if joints_key not in processor.world_coord:
            raise ValueError(
                f"Processor world_coord is missing {joints_key!r} for template angles."
            )
        joints = np.asarray(processor.world_coord[joints_key], dtype=np.float64)
        if joints.ndim != 3 or joints.shape[-1] != 3:
            raise ValueError(
                f"Expected {joints_key} shape (T, J, 3), got {joints.shape}."
            )
        if frame_idx < 0 or frame_idx >= joints.shape[0]:
            raise ValueError(
                f"Template frame index {frame_idx} is out of range for {joints_key} "
                f"with {joints.shape[0]} frames."
            )
        frame_joints = joints[frame_idx]
        keypoints: dict[str, np.ndarray] = {}
        for name in keypoint_names:
            if name not in joint_map:
                raise KeyError(f"Processor hand_joint_map is missing {name!r}.")
            joint_idx = int(joint_map[name])
            if joint_idx >= frame_joints.shape[0]:
                raise ValueError(
                    f"Joint index {joint_idx} for {name!r} is out of range for "
                    f"{joints_key} frame shape {frame_joints.shape}."
                )
            keypoints[name] = frame_joints[joint_idx]
        return keypoints

    @staticmethod
    def _robot_skeleton_angles(
        *, hand: Any, joint_values: np.ndarray, keypoint_names: tuple[str, ...]
    ) -> np.ndarray:
        joint_values = np.asarray(joint_values).reshape(-1)
        if joint_values.shape != (len(hand.joint_names),):
            raise ValueError(
                f"Expected {len(hand.joint_names)} joint values for hand {hand.name!r}, got {joint_values.shape}."
            )
        pose_dict = {
            "pos": np.asarray(hand.base_pose[:3, 3], dtype=np.float64),
            "rot": np.asarray(hand.base_pose[:3, :3], dtype=np.float64),
            "joints": {
                name: float(value)
                for name, value in zip(hand.joint_names, joint_values)
            },
        }
        keypoints = _robot_keypoint_dict_for_frame(
            urdf_inst=hand.urdf_inst,
            robot_cfg=hand.robo_cfg,
            pose_dict=pose_dict,
            keypoint_names=keypoint_names,
        )
        return _skeleton_angles(keypoints)

    @staticmethod
    def _dtw_euclidean_stats(
        a: np.ndarray,
        b: np.ndarray,
        *,
        plot_path: Path | None = None,
        title: str | None = None,
        window_fraction: float,
    ) -> tuple[float, float]:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        if b.ndim == 1:
            b = b[:, None]
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
            raise ValueError(
                f"Expected DTW arrays with shapes (T, D), got {a.shape} and {b.shape}."
            )
        dist_matrix = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        alignment = FinalRolloutMetrics._dtw_with_window(
            dist_matrix,
            window_fraction=window_fraction,
        )
        FinalRolloutMetrics._save_dtw_alignment_plot(
            alignment,
            n_rollout=a.shape[0],
            n_reference=b.shape[0],
            plot_path=plot_path,
            title=title,
        )
        return (
            float(alignment.normalizedDistance),
            FinalRolloutMetrics._dtw_path_mean(alignment),
        )

    @classmethod
    def _dtw_quat_stats(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        *,
        plot_path: Path | None = None,
        title: str | None = None,
        window_fraction: float,
    ) -> tuple[float, float]:
        dist_matrix = cls._quat_pairwise_error_np(a, b)
        alignment = cls._dtw_with_window(
            dist_matrix,
            window_fraction=window_fraction,
        )
        cls._save_dtw_alignment_plot(
            alignment,
            n_rollout=np.asarray(a).shape[0],
            n_reference=np.asarray(b).shape[0],
            plot_path=plot_path,
            title=title,
        )
        return (
            float(alignment.normalizedDistance),
            cls._dtw_path_mean(alignment),
        )

    @staticmethod
    def _dtw_path_mean(alignment: Any) -> float:
        path_len = len(alignment.index1)
        if path_len <= 0:
            raise ValueError("Expected non-empty DTW alignment path.")
        return float(alignment.distance / path_len)

    @staticmethod
    def _dtw_with_window(dist_matrix: np.ndarray, *, window_fraction: float) -> Any:
        if not (0.0 <= window_fraction <= 1.0):
            raise ValueError(
                f"Expected DTW window_fraction in [0, 1], got {window_fraction}."
            )
        return dtw(
            np.asarray(dist_matrix, dtype=np.float64),
            window_type=FinalRolloutMetrics._normalized_sakoe_chiba_window,
            window_args={"window_fraction": float(window_fraction)},
        )

    @staticmethod
    def _normalized_sakoe_chiba_window(
        iw: np.ndarray,
        jw: np.ndarray,
        query_size: int,
        reference_size: int,
        window_fraction: float,
    ) -> np.ndarray:
        query_denom = max(int(query_size) - 1, 1)
        reference_denom = max(int(reference_size) - 1, 1)
        query_phase = iw / query_denom
        reference_phase = jw / reference_denom
        return np.abs(reference_phase - query_phase) <= float(window_fraction)

    def _dtw_plot_path(
        self, episode_index: int | None, metric_name: str
    ) -> Path | None:
        if self.cfg.dtw_plot_dir is None or episode_index is None:
            return None
        if episode_index >= int(self.cfg.dtw_plot_max_episodes):
            return None
        return self.cfg.dtw_plot_dir / f"episode_{episode_index:04d}_{metric_name}.png"

    @staticmethod
    def _save_dtw_alignment_plot(
        alignment: Any,
        *,
        n_rollout: int,
        n_reference: int,
        plot_path: Path | None,
        title: str | None,
    ) -> None:
        if plot_path is None:
            return
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=140)
        ax.plot(alignment.index1, alignment.index2, linewidth=1.5, label="DTW path")
        ax.plot(
            [0, max(n_rollout - 1, 0)],
            [0, max(n_reference - 1, 0)],
            linestyle="--",
            linewidth=1.0,
            color="black",
            label="start-end diagonal",
        )
        ax.set_xlabel("Rollout frame")
        ax.set_ylabel("Reference frame")
        if title is not None:
            ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    @staticmethod
    def _quat_error_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_t = torch.from_numpy(np.asarray(a, dtype=np.float32))
        b_t = torch.from_numpy(np.asarray(b, dtype=np.float32))
        return quat_error_magnitude(a_t, b_t).detach().cpu().numpy()

    @staticmethod
    def _quat_pairwise_error_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 4 or b.shape[1] != 4:
            raise ValueError(
                f"Expected quaternion arrays with shapes (T, 4), got {a.shape} and {b.shape}."
            )
        a_t = torch.from_numpy(
            np.broadcast_to(a[:, None, :], (a.shape[0], b.shape[0], 4)).copy()
        )
        b_t = torch.from_numpy(
            np.broadcast_to(b[None, :, :], (a.shape[0], b.shape[0], 4)).copy()
        )
        return quat_error_magnitude(a_t, b_t).detach().cpu().numpy()
