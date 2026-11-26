from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from dexlink.utils.retargeting.error_functions import angle_error
from dexlink.utils.retargeting.math_utils import matrix_from_quat
from dexlink.utils.retargeting.path_utils import process_urdf
from dexlink.utils.rl.dispenser_cycle_metrics import (
    DispenserCycleConfig,
    DispenserCycleTracker,
)
from dexlink.utils.rl.scissors_tracking_metrics import SCISSORS_METRIC_NAMES, ScissorsSample, ScissorsTracking
from dexlink.src.rl.world.art_object import ArtObject
from dexlink.src.rl.world.cardboard_object import CardboardObject
from dexlink.src.rl.world.robot import Hand
from dexlink.src.rl.world.task_manager import TaskManager


class MultiObjectTaskManager(TaskManager):
    """Task manager for exported sequence payloads with scene_object_* fields."""

    TASK_METRIC_NAMES = {
        "scissors_tracking": SCISSORS_METRIC_NAMES,
        "open_notebook": (
            "notebook_final_angle_error_rad",
            "notebook_final_angle_within_15deg",
        ),
        "block_stacking": (
            "stack_min_center_height_margin_m",
            "stack_success",
        ),
        "banana_handover": (
            "banana_center_height_above_table_m",
            "banana_in_air",
            "banana_not_fallen",
            "banana_object_weight_n",
            "banana_source_hand_contact_link_count",
            "banana_source_hand_net_contact_force_n",
            "banana_source_hand_net_contact_force_weight_fraction",
            "banana_source_hand_released",
            "banana_receiving_hand_contact_link_count",
            "banana_receiving_hand_net_contact_force_n",
            "banana_receiving_hand_upward_support_force_n",
            "banana_receiving_hand_upward_support_weight_fraction",
            "banana_receiving_hand_support",
            "banana_handover_success",
        ),
        "hammer_peg_impact": (
            "hammer_peg_average_contact_impact_n",
            "hammer_peg_contact_step_count",
        ),
        "hammer_single_swing": (
            "hammer_strike_contacted",
            "hammer_single_swing_completed",
            "hammer_single_swing_completion_time_steps",
            "hammer_single_swing_completion_time_s",
            "hammer_peg_average_contact_impact_n",
            "hammer_peg_contact_step_count",
        ),
        "hammer_two_swing": (
            "hammer_first_strike_contacted",
            "hammer_second_strike_contacted",
            "hammer_two_swings_completed",
            "hammer_two_swing_completion_time_steps",
            "hammer_two_swing_completion_time_s",
            "hammer_peg_average_contact_impact_n",
            "hammer_peg_contact_step_count",
        ),
        "object_pose": (
            "object_final_position_error_m",
            "object_final_orientation_error_rad",
            "object_pose_success",
        ),
        "object_lift": (
            "object_lift_height_m",
            "object_lift_reference_height_m",
            "object_lift_height_error_m",
            "object_lift_success",
        ),
        "bottle_interaction": (
            "object_lift_height_m",
            "object_lift_reference_height_m",
            "object_lift_height_error_m",
            "object_lift_success",
            "bottle_relative_wrist_position_error_m",
            "bottle_relative_wrist_orientation_error_rad",
            "bottle_requested_interaction_match",
            "bottle_interaction_success",
        ),
        "cube_strategy": (
            "cube_final_position_error_m",
            "cube_final_orientation_error_rad",
            "cube_final_pose_success",
            "cube_max_center_height_change_m",
            "cube_lifted",
            "cube_strategy_success",
        ),
        "dispenser_pressed": (
            "dispenser_min_head_joint_position_m",
            "dispenser_pressed_state_achieved",
        ),
        "dispenser_press_return_twice": (
            "dispenser_completed_press_return_cycles",
            "dispenser_head_returned",
            "dispenser_two_press_return_cycles_success",
        ),
        "drawer_articulation": (
            "drawer_final_articulation_error_m",
            "drawer_articulation_success",
        ),
        "eraser_tracking": (
            "eraser_erase_phase_max_position_error_m",
            "eraser_erase_phase_step_count",
            "eraser_erase_phase_complete",
            "eraser_erase_phase_success",
        ),
        "precision_mounting": (
            "mounting_origin_height_above_post_m",
            "mounting_origin_height_threshold_m",
            "mounting_bottom_axis_radial_error_m",
            "mounting_top_axis_radial_error_m",
            "mounting_max_axis_radial_error_m",
            "mounting_radial_threshold_m",
            "mounting_axis_tilt_rad",
            "mounting_tilt_threshold_rad",
            "mounting_success",
        ),
        "grasp_stability": (
            "grasp_current_object_drift_m",
            "grasp_max_object_drift_m",
            "grasp_stability_success",
        ),
    }
    PRIMARY_TASK_METRIC_NAMES = {
        "scissors_tracking": "scissors_in_air",
        "open_notebook": "notebook_final_angle_within_15deg",
        "block_stacking": "stack_success",
        "banana_handover": "banana_handover_success",
        "hammer_peg_impact": "hammer_peg_average_contact_impact_n",
        "hammer_single_swing": "hammer_single_swing_completed",
        "hammer_two_swing": "hammer_two_swings_completed",
        "object_pose": "object_pose_success",
        "object_lift": "object_lift_success",
        "bottle_interaction": "object_lift_success",
        "cube_strategy": "cube_strategy_success",
        "dispenser_pressed": "dispenser_pressed_state_achieved",
        "dispenser_press_return_twice": "dispenser_two_press_return_cycles_success",
        "drawer_articulation": "drawer_articulation_success",
        "eraser_tracking": "eraser_erase_phase_success",
        "precision_mounting": "mounting_success",
        "grasp_stability": "grasp_stability_success",
    }

    def initialize(self) -> None:
        super().initialize()
        self._initialize_phase_metadata()
        self._initialize_scene_object_metadata()
        self._initialize_task_metric()

    def _initialize_task_metric(self) -> None:
        cfg = self.task_cfg.metric_cfg
        metric_id = cfg.metric_id
        if metric_id is not None and metric_id not in self.TASK_METRIC_NAMES:
            raise ValueError(
                f"Unsupported task metric id {metric_id!r}; expected one of "
                f"{sorted(self.TASK_METRIC_NAMES)}."
            )

        if cfg.eval_metric_id not in (None, "mug_grasp_change"):
            raise ValueError(
                f"Unsupported eval task metric id {cfg.eval_metric_id!r}; "
                "expected None or 'mug_grasp_change'."
            )
        if metric_id is not None and cfg.eval_metric_id is not None:
            raise ValueError(
                "Configure either a train-time task metric or an evaluation-only "
                "task metric, not both."
            )
        if metric_id is None and cfg.eval_metric_id is None:
            return

        if cfg.eval_metric_id == "mug_grasp_change":
            self._validate_object_ids((cfg.mug_object_id,))
            if cfg.mug_hand_side not in self.active_hand_sides:
                raise ValueError(
                    "Mug grasp-change hand must be active, got "
                    f"{cfg.mug_hand_side!r} with active hands {self.active_hand_sides}."
                )
            phase_names = tuple(cfg.mug_reference_grasp_phase_names)
            if len(phase_names) != 2 or any(not name for name in phase_names):
                raise ValueError(
                    "Mug grasp-change criterion expects exactly two non-empty "
                    f"reference phase names, got {phase_names}."
                )
            missing_phases = [
                name for name in phase_names if name not in self.phase_names
            ]
            if missing_phases:
                raise ValueError(
                    f"Mug reference phases {missing_phases} are absent from "
                    f"task phases {self.phase_names}."
                )
            if cfg.mug_min_center_height_above_table_m <= 0.0:
                raise ValueError(
                    "Expected mug_min_center_height_above_table_m > 0, got "
                    f"{cfg.mug_min_center_height_above_table_m}."
                )
            self._mug_hand_tracking_score_sum = torch.zeros(
                self.env.num_envs, dtype=torch.float32, device=self.env.device
            )
            self._mug_hand_tracking_step_count = torch.zeros(
                self.env.num_envs, dtype=torch.long, device=self.env.device
            )

        if metric_id is None:
            return

        if metric_id == "scissors_tracking":
            self._validate_object_ids((cfg.scissors_object_id,))
            index = self.scene_object_ids.index(cfg.scissors_object_id)
            names = tuple(self.scene_object_arti_names[index])
            valid = self.scene_object_arti_valid_mask[index]
            if names != (cfg.scissors_joint_name,) or int(valid.sum()) != 1:
                raise ValueError(f"Expected exactly the scissors hinge {cfg.scissors_joint_name}, received {names}")
            if self.task_cfg.object_fixed or self.scene_object_static[index]:
                raise ValueError("Scissors retention tracking requires an explicitly free-floating scene object")
            self._scissors_index = index
            self._scissors_tracking = ScissorsTracking(
                Path(self.scene_object_urdf_paths[index]), self.env.num_envs,
                self.env.device, cfg.scissors_airborne_clearance_m,
            )
            self._scissors_reference = {
                "pos": torch.as_tensor(self.scene_object_pos[:, index], device=self.env.device),
                "quat": torch.as_tensor(self.scene_object_quat[:, index], device=self.env.device),
                "arti": torch.as_tensor(self.scene_object_arti_vals[:, index, valid], device=self.env.device),
            }
            return

        if metric_id == "open_notebook":
            self._validate_object_ids((cfg.notebook_object_id,))
            object_index = self.scene_object_ids.index(cfg.notebook_object_id)
            valid_arti = self.scene_object_arti_valid_mask[object_index]
            if int(valid_arti.sum()) != 1:
                raise ValueError(
                    "Open-notebook criterion expects exactly one articulation DoF "
                    f"for {cfg.notebook_object_id!r}, got mask {valid_arti}."
                )
            if cfg.notebook_angle_tolerance_rad <= 0.0:
                raise ValueError(
                    "Expected notebook_angle_tolerance_rad > 0, got "
                    f"{cfg.notebook_angle_tolerance_rad}."
                )
            return

        if metric_id == "block_stacking":
            object_ids = tuple(cfg.stacking_object_ids)
            offsets = tuple(cfg.stacking_min_center_height_offsets_m)
            if len(object_ids) < 2:
                raise ValueError(
                    f"Block-stacking criterion expects at least two objects, got {object_ids}."
                )
            self._validate_object_ids(object_ids)
            if len(offsets) != len(object_ids) - 1:
                raise ValueError(
                    "Block-stacking criterion expects one minimum center-height "
                    f"offset per adjacent pair, got objects={object_ids}, offsets={offsets}."
                )
            if any(offset <= 0.0 for offset in offsets):
                raise ValueError(
                    f"Expected positive stacking center-height offsets, got {offsets}."
                )
            return

        if metric_id in {
            "hammer_peg_impact",
            "hammer_single_swing",
            "hammer_two_swing",
        }:
            if self.env.cfg.backend != "sim":
                raise ValueError(
                    "Hammer peg impact requires the Genesis simulation backend, "
                    f"got backend={self.env.cfg.backend!r}."
                )
            self._validate_object_ids((cfg.hammer_object_id, cfg.hammer_peg_object_id))
            if cfg.hammer_object_id == cfg.hammer_peg_object_id:
                raise ValueError("Hammer and peg object ids must be different.")
            if cfg.hammer_peg_sensor_top_offset_m <= 0.0:
                raise ValueError(
                    "Expected hammer_peg_sensor_top_offset_m > 0, got "
                    f"{cfg.hammer_peg_sensor_top_offset_m}."
                )
            if cfg.hammer_contact_force_threshold <= 0.0:
                raise ValueError(
                    "Expected hammer_contact_force_threshold > 0, got "
                    f"{cfg.hammer_contact_force_threshold}."
                )
            self._hammer_contact_impact_sum = torch.zeros(
                self.env.num_envs, dtype=torch.float32, device=self.env.device
            )
            self._hammer_contact_step_count = torch.zeros(
                self.env.num_envs, dtype=torch.long, device=self.env.device
            )
            if metric_id == "hammer_single_swing":
                strike_phases = tuple(cfg.hammer_strike_phase_names)
                if not strike_phases:
                    raise ValueError(
                        "Hammer single-swing timing requires non-empty strike "
                        "phase names."
                    )
                missing_phases = [
                    name for name in strike_phases if name not in self.phase_names
                ]
                if missing_phases:
                    raise ValueError(
                        "Hammer strike phase names are absent from the reference: "
                        f"{missing_phases}; available={self.phase_names}."
                    )
                self._hammer_strike_phase_indices = torch.as_tensor(
                    [self.phase_names.index(name) for name in strike_phases],
                    dtype=torch.long,
                    device=self.env.device,
                )
                self._hammer_strike_contact_timestep = torch.full(
                    (self.env.num_envs,),
                    -1,
                    dtype=torch.long,
                    device=self.env.device,
                )
            elif metric_id == "hammer_two_swing":
                first_phases = tuple(cfg.hammer_first_strike_phase_names)
                second_phases = tuple(cfg.hammer_second_strike_phase_names)
                if not first_phases or not second_phases:
                    raise ValueError(
                        "Hammer two-swing timing requires non-empty first- and "
                        "second-strike phase-name tuples."
                    )
                overlap = sorted(set(first_phases) & set(second_phases))
                if overlap:
                    raise ValueError(
                        "Hammer first- and second-strike phase windows must be "
                        f"disjoint, got overlap {overlap}."
                    )
                missing_phases = [
                    name
                    for name in (*first_phases, *second_phases)
                    if name not in self.phase_names
                ]
                if missing_phases:
                    raise ValueError(
                        "Hammer timing phase names are absent from the reference: "
                        f"{missing_phases}; available={self.phase_names}."
                    )
                self._hammer_first_strike_phase_indices = torch.as_tensor(
                    [self.phase_names.index(name) for name in first_phases],
                    dtype=torch.long,
                    device=self.env.device,
                )
                self._hammer_second_strike_phase_indices = torch.as_tensor(
                    [self.phase_names.index(name) for name in second_phases],
                    dtype=torch.long,
                    device=self.env.device,
                )
                self._hammer_first_contact_timestep = torch.full(
                    (self.env.num_envs,),
                    -1,
                    dtype=torch.long,
                    device=self.env.device,
                )
                self._hammer_second_contact_timestep = torch.full(
                    (self.env.num_envs,),
                    -1,
                    dtype=torch.long,
                    device=self.env.device,
                )
            return

        if metric_id == "object_pose":
            self._validate_object_ids((cfg.object_pose_object_id,))
            if cfg.object_pose_position_tolerance_m <= 0.0:
                raise ValueError(
                    "Expected object_pose_position_tolerance_m > 0, got "
                    f"{cfg.object_pose_position_tolerance_m}."
                )
            if cfg.object_pose_orientation_tolerance_rad <= 0.0:
                raise ValueError(
                    "Expected object_pose_orientation_tolerance_rad > 0, got "
                    f"{cfg.object_pose_orientation_tolerance_rad}."
                )
            return

        if metric_id == "object_lift":
            self._validate_object_ids((cfg.object_lift_object_id,))
            object_index = self.scene_object_ids.index(cfg.object_lift_object_id)
            reference_lift_height = float(
                self.scene_object_pos[-1, object_index, 2]
                - self.scene_object_pos[0, object_index, 2]
            )
            if reference_lift_height <= 0.0:
                raise ValueError(
                    "Object-lift criterion requires a positive reference lift "
                    f"height, got {reference_lift_height} m for "
                    f"{cfg.object_lift_object_id!r}."
                )
            if self.env.rewards_cfg.task_success_pos_thd <= 0.0:
                raise ValueError(
                    "Object-lift criterion requires task_success_pos_thd > 0, got "
                    f"{self.env.rewards_cfg.task_success_pos_thd}."
                )
            return

        if metric_id == "bottle_interaction":
            self._validate_object_ids((cfg.bottle_interaction_object_id,))
            if cfg.bottle_interaction_hand_side not in self.active_hand_sides:
                raise ValueError(
                    "Bottle-interaction hand must be active, got "
                    f"{cfg.bottle_interaction_hand_side!r} with active hands "
                    f"{self.active_hand_sides}."
                )
            object_index = self.scene_object_ids.index(cfg.bottle_interaction_object_id)
            reference_lift_height = float(
                self.scene_object_pos[-1, object_index, 2]
                - self.scene_object_pos[0, object_index, 2]
            )
            if reference_lift_height <= 0.0:
                raise ValueError(
                    "Bottle-interaction criterion requires a positive reference "
                    f"lift height, got {reference_lift_height} m."
                )
            if cfg.bottle_relative_wrist_position_tolerance_m <= 0.0:
                raise ValueError(
                    "Expected bottle_relative_wrist_position_tolerance_m > 0, got "
                    f"{cfg.bottle_relative_wrist_position_tolerance_m}."
                )
            orientation_tolerance = cfg.bottle_relative_wrist_orientation_tolerance_rad
            if not 0.0 < orientation_tolerance <= np.pi:
                raise ValueError(
                    "Expected bottle_relative_wrist_orientation_tolerance_rad in "
                    f"(0, pi], got {orientation_tolerance}."
                )
            return

        if metric_id == "cube_strategy":
            self._validate_object_ids((cfg.cube_strategy_object_id,))
            if cfg.cube_strategy_min_lift_height_change_m <= 0.0:
                raise ValueError(
                    "Expected cube_strategy_min_lift_height_change_m > 0, got "
                    f"{cfg.cube_strategy_min_lift_height_change_m}."
                )
            if cfg.cube_strategy_position_tolerance_m <= 0.0:
                raise ValueError(
                    "Expected cube_strategy_position_tolerance_m > 0, got "
                    f"{cfg.cube_strategy_position_tolerance_m}."
                )
            if cfg.cube_strategy_orientation_tolerance_rad <= 0.0:
                raise ValueError(
                    "Expected cube_strategy_orientation_tolerance_rad > 0, got "
                    f"{cfg.cube_strategy_orientation_tolerance_rad}."
                )
            self._cube_initial_center_z = torch.zeros(
                self.env.num_envs, dtype=torch.float32, device=self.env.device
            )
            self._cube_max_center_z = torch.zeros(
                self.env.num_envs, dtype=torch.float32, device=self.env.device
            )
            return

        if metric_id == "dispenser_press_return_twice":
            self._validate_object_ids((cfg.dispenser_object_id,))
            object_index = self.scene_object_ids.index(cfg.dispenser_object_id)
            valid_arti = self.scene_object_arti_valid_mask[object_index]
            joint_names = tuple(self.scene_object_arti_names[object_index])
            if int(valid_arti.sum()) != 1 or joint_names != (cfg.dispenser_joint_name,):
                raise ValueError(
                    "Dispenser cycle criterion expects exactly the joint "
                    f"{cfg.dispenser_joint_name!r}, got names={joint_names} and "
                    f"valid mask={valid_arti}."
                )
            cycle_cfg = DispenserCycleConfig(
                press_depth_m=cfg.dispenser_cycle_press_depth_m,
                return_tolerance_m=cfg.dispenser_cycle_return_tolerance_m,
                dwell_steps=cfg.dispenser_cycle_dwell_steps,
            )
            reference_positions = self.scene_object_arti_vals[:, object_index, valid_arti]
            expected_shape = (self.total_steps, 1)
            if (reference_positions.shape != expected_shape
                    or not np.isfinite(reference_positions).all()):
                raise ValueError(
                    f"Expected finite dispenser reference articulation shape {expected_shape}, "
                    f"got {reference_positions.shape}."
                )
            if (reference_positions.min() > -cycle_cfg.press_depth_m
                    or not np.any(np.abs(reference_positions) <= cycle_cfg.return_tolerance_m)):
                raise ValueError("Dispenser cycle reference must reach the press and zero-rest return thresholds.")
            self._dispenser_cycle_tracker = DispenserCycleTracker(
                self.env.num_envs, self.env.device, cycle_cfg,
            )
            return

        if metric_id == "dispenser_pressed":
            self._validate_object_ids((cfg.dispenser_object_id,))
            object_index = self.scene_object_ids.index(cfg.dispenser_object_id)
            valid_arti = self.scene_object_arti_valid_mask[object_index]
            joint_names = tuple(self.scene_object_arti_names[object_index])
            if int(valid_arti.sum()) != 1 or joint_names != (cfg.dispenser_joint_name,):
                raise ValueError(
                    "Dispenser pressed-state criterion expects exactly the joint "
                    f"{cfg.dispenser_joint_name!r}, got names={joint_names} and "
                    f"valid mask={valid_arti}."
                )
            if cfg.dispenser_pressed_tolerance_m <= 0.0:
                raise ValueError(
                    "Expected dispenser_pressed_tolerance_m > 0, got "
                    f"{cfg.dispenser_pressed_tolerance_m}."
                )
            reference_positions = self.scene_object_arti_vals[
                :, object_index, valid_arti
            ]
            expected_shape = (self.total_steps, 1)
            if reference_positions.shape != expected_shape:
                raise ValueError(
                    "Expected dispenser reference articulation shape "
                    f"{expected_shape}, got {reference_positions.shape}."
                )
            pressed_position = float(reference_positions.min())
            raised_position = float(reference_positions.max())
            reference_stroke = raised_position - pressed_position
            if reference_stroke <= cfg.dispenser_pressed_tolerance_m:
                raise ValueError(
                    "Dispenser reference must contain a press whose stroke exceeds "
                    "the pressed-state tolerance, got "
                    f"stroke={reference_stroke} m and "
                    f"tolerance={cfg.dispenser_pressed_tolerance_m} m."
                )
            self._dispenser_pressed_reference_position_m = pressed_position
            self._dispenser_min_head_position_m = torch.full(
                (self.env.num_envs,),
                torch.inf,
                dtype=torch.float32,
                device=self.env.device,
            )
            return

        if metric_id == "drawer_articulation":
            self._validate_object_ids((cfg.drawer_object_id,))
            object_index = self.scene_object_ids.index(cfg.drawer_object_id)
            valid_arti = self.scene_object_arti_valid_mask[object_index]
            joint_names = tuple(self.scene_object_arti_names[object_index])
            if int(valid_arti.sum()) != 1 or joint_names != (cfg.drawer_joint_name,):
                raise ValueError(
                    "Drawer articulation criterion expects exactly the joint "
                    f"{cfg.drawer_joint_name!r}, got names={joint_names} and "
                    f"valid mask={valid_arti}."
                )
            if cfg.drawer_final_articulation_tolerance_m <= 0.0:
                raise ValueError(
                    "Expected drawer_final_articulation_tolerance_m > 0, got "
                    f"{cfg.drawer_final_articulation_tolerance_m}."
                )
            return

        if metric_id == "eraser_tracking":
            self._validate_object_ids((cfg.eraser_object_id,))
            phase_names = tuple(cfg.eraser_phase_names)
            if not phase_names:
                raise ValueError(
                    "Eraser tracking criterion requires non-empty erase phase names."
                )
            missing_phases = [
                phase_name
                for phase_name in phase_names
                if phase_name not in self.phase_names
            ]
            if missing_phases:
                raise ValueError(
                    "Eraser phase names are absent from the reference: "
                    f"{missing_phases}; available={self.phase_names}."
                )
            if cfg.eraser_max_position_error_m <= 0.0:
                raise ValueError(
                    "Expected eraser_max_position_error_m > 0, got "
                    f"{cfg.eraser_max_position_error_m}."
                )
            self._eraser_phase_indices = torch.as_tensor(
                [self.phase_names.index(name) for name in phase_names],
                dtype=torch.long,
                device=self.env.device,
            )
            phase_index_by_timestep = (
                self.phase_index_by_timestep.detach().cpu().numpy()
            )
            self._eraser_reference_step_count = int(
                np.isin(
                    phase_index_by_timestep, self._eraser_phase_indices.cpu().numpy()
                ).sum()
            )
            self._eraser_max_position_error_m = torch.full(
                (self.env.num_envs,),
                -torch.inf,
                dtype=torch.float32,
                device=self.env.device,
            )
            self._eraser_phase_step_count = torch.zeros(
                self.env.num_envs,
                dtype=torch.long,
                device=self.env.device,
            )
            return

        if metric_id == "precision_mounting":
            self._validate_object_ids(
                (cfg.mounting_cylinder_object_id, cfg.mounting_post_object_id)
            )
            if cfg.mounting_cylinder_object_id == cfg.mounting_post_object_id:
                raise ValueError("Mounting cylinder and post ids must be different.")
            positive_fields = {
                "mounting_cylinder_height_m": cfg.mounting_cylinder_height_m,
                "mounting_bore_clearance_m": cfg.mounting_bore_clearance_m,
                "mounting_max_origin_height_above_post_m": (
                    cfg.mounting_max_origin_height_above_post_m
                ),
                "mounting_max_axis_tilt_rad": cfg.mounting_max_axis_tilt_rad,
            }
            invalid_fields = {
                name: value for name, value in positive_fields.items() if value <= 0.0
            }
            if invalid_fields:
                raise ValueError(
                    "Precision-mounting metric requires positive geometric "
                    f"thresholds, got {invalid_fields}."
                )
            if cfg.mounting_max_axis_tilt_rad >= np.pi / 2.0:
                raise ValueError(
                    "Expected mounting_max_axis_tilt_rad < pi/2, got "
                    f"{cfg.mounting_max_axis_tilt_rad}."
                )
            return

        if metric_id == "grasp_stability":
            self._validate_object_ids((cfg.grasp_stability_object_id,))
            if not self.task_cfg.grasp_refinement_mode:
                raise ValueError(
                    "metric_id='grasp_stability' requires "
                    "task_cfg.grasp_refinement_mode=True."
                )
            if cfg.grasp_stability_max_drift_m <= 0.0:
                raise ValueError(
                    "Expected grasp_stability_max_drift_m > 0, got "
                    f"{cfg.grasp_stability_max_drift_m}."
                )
            if self.task_cfg.use_scene_support:
                raise ValueError(
                    "Grasp stability refinement requires use_scene_support=False."
                )
            object_index = self.scene_object_ids.index(cfg.grasp_stability_object_id)
            self._grasp_reference_object_position = torch.as_tensor(
                self.scene_object_pos[0, object_index],
                dtype=torch.float32,
                device=self.env.device,
            )
            self._grasp_max_object_drift_m = torch.zeros(
                self.env.num_envs,
                dtype=torch.float32,
                device=self.env.device,
            )
            return

        if metric_id != "banana_handover":
            raise ValueError(f"Unhandled task metric id {metric_id!r}.")

        self._validate_object_ids((cfg.banana_object_id,))
        if cfg.banana_source_hand not in self.active_hand_sides:
            raise ValueError(
                "Banana-handover source hand must be active, got "
                f"{cfg.banana_source_hand!r} with active hands {self.active_hand_sides}."
            )
        if cfg.banana_receiving_hand not in self.active_hand_sides:
            raise ValueError(
                "Banana-handover receiving hand must be active, got "
                f"{cfg.banana_receiving_hand!r} with active hands {self.active_hand_sides}."
            )
        if cfg.banana_source_hand == cfg.banana_receiving_hand:
            raise ValueError(
                "Banana-handover source and receiving hands must differ, got "
                f"{cfg.banana_source_hand!r}."
            )
        if cfg.banana_min_center_height_above_table_m <= 0.0:
            raise ValueError(
                "Expected banana_min_center_height_above_table_m > 0, got "
                f"{cfg.banana_min_center_height_above_table_m}."
            )
        source_fraction = cfg.banana_max_source_contact_weight_fraction
        if not (0.0 <= source_fraction < 1.0):
            raise ValueError(
                "Expected banana_max_source_contact_weight_fraction in [0, 1), "
                f"got {source_fraction}."
            )
        support_fraction = cfg.banana_min_receiving_support_weight_fraction
        if not (0.0 < support_fraction <= 1.0):
            raise ValueError(
                "Expected banana_min_receiving_support_weight_fraction in (0, 1], "
                f"got {support_fraction}."
            )
        if cfg.contact_force_threshold <= 0.0:
            raise ValueError(
                f"Expected contact_force_threshold > 0, got {cfg.contact_force_threshold}."
            )

    def _validate_object_ids(self, expected_ids: tuple[str, ...]) -> None:
        if any(not object_id for object_id in expected_ids):
            raise ValueError(
                f"Task criterion contains an empty object id: {expected_ids}."
            )
        missing = [
            object_id
            for object_id in expected_ids
            if object_id not in self.scene_object_ids
        ]
        if missing:
            raise ValueError(
                f"Task criterion objects {missing} are absent from scene objects "
                f"{self.scene_object_ids}."
            )

    @property
    def task_metric_names(self) -> tuple[str, ...]:
        metric_id = self.task_cfg.metric_cfg.metric_id
        if metric_id is None:
            return ()
        return self.TASK_METRIC_NAMES[metric_id]

    @property
    def primary_task_metric_name(self) -> str | None:
        metric_id = self.task_cfg.metric_cfg.metric_id
        if metric_id is None:
            return None
        return self.PRIMARY_TASK_METRIC_NAMES[metric_id]

    @property
    def checkpoint_metric_name(self) -> str | None:
        if self.task_cfg.metric_cfg.eval_metric_id == "mug_grasp_change":
            return "mug_hand_reference_tracking_score"
        return self.primary_task_metric_name

    def compute_task_metrics(self) -> dict[str, torch.Tensor]:
        metric_id = self.task_cfg.metric_cfg.metric_id
        if metric_id is None:
            return {}
        metric_fns = {
            "scissors_tracking": self._compute_scissors_tracking_metrics,
            "open_notebook": self._compute_open_notebook_metrics,
            "block_stacking": self._compute_block_stacking_metrics,
            "banana_handover": self._compute_banana_handover_metrics,
            "hammer_peg_impact": self._compute_hammer_peg_impact_metrics,
            "hammer_single_swing": self._compute_hammer_single_swing_metrics,
            "hammer_two_swing": self._compute_hammer_two_swing_metrics,
            "object_pose": self._compute_object_pose_metrics,
            "object_lift": self._compute_object_lift_metrics,
            "bottle_interaction": self._compute_bottle_interaction_metrics,
            "cube_strategy": self._compute_cube_strategy_metrics,
            "dispenser_pressed": self._compute_dispenser_pressed_metrics,
            "dispenser_press_return_twice": self._compute_dispenser_cycle_metrics,
            "drawer_articulation": self._compute_drawer_articulation_metrics,
            "eraser_tracking": self._compute_eraser_tracking_metrics,
            "precision_mounting": self._compute_precision_mounting_metrics,
            "grasp_stability": self._compute_grasp_stability_metrics,
        }
        metrics = metric_fns[metric_id]()
        if tuple(metrics) != self.task_metric_names:
            raise ValueError(
                f"Task metric {metric_id!r} returned keys {tuple(metrics)}, expected "
                f"{self.task_metric_names}."
            )
        for name, value in metrics.items():
            if tuple(value.shape) != (self.env.num_envs,):
                raise ValueError(
                    f"Task metric {name!r} must have shape {(self.env.num_envs,)}, "
                    f"got {tuple(value.shape)}."
                )
        return metrics

    def update_task_metric_accumulators(self) -> None:
        cfg = self.task_cfg.metric_cfg
        if cfg.metric_id == "scissors_tracking":
            self._scissors_tracking.update(self._scissors_tracking_sample())
        if cfg.metric_id in {
            "hammer_peg_impact",
            "hammer_single_swing",
            "hammer_two_swing",
        }:
            impact, contact_mask = (
                self.env.physics_manager.get_hammer_peg_contact_impact()
            )
            expected_shape = (self.env.num_envs,)
            if (
                tuple(impact.shape) != expected_shape
                or tuple(contact_mask.shape) != expected_shape
            ):
                raise ValueError(
                    "Expected hammer impact and contact mask shape "
                    f"{expected_shape}, got {tuple(impact.shape)} and "
                    f"{tuple(contact_mask.shape)}."
                )
            self._hammer_contact_impact_sum += impact
            self._hammer_contact_step_count += contact_mask.to(torch.long)
            if cfg.metric_id == "hammer_single_swing":
                phase_indices = self.phase_index_by_timestep[self.env.command_time_step]
                strike_window = torch.isin(
                    phase_indices,
                    self._hammer_strike_phase_indices,
                )
                first_contact = torch.logical_and(
                    torch.logical_and(contact_mask, strike_window),
                    self._hammer_strike_contact_timestep < 0,
                )
                self._hammer_strike_contact_timestep[first_contact] = (
                    self.env.episode_length[first_contact]
                )
            elif cfg.metric_id == "hammer_two_swing":
                phase_indices = self.phase_index_by_timestep[self.env.command_time_step]
                first_window = torch.isin(
                    phase_indices,
                    self._hammer_first_strike_phase_indices,
                )
                second_window = torch.isin(
                    phase_indices,
                    self._hammer_second_strike_phase_indices,
                )
                first_contact = torch.logical_and(
                    torch.logical_and(contact_mask, first_window),
                    self._hammer_first_contact_timestep < 0,
                )
                second_contact = torch.logical_and(
                    torch.logical_and(contact_mask, second_window),
                    self._hammer_second_contact_timestep < 0,
                )
                self._hammer_first_contact_timestep[first_contact] = (
                    self.env.episode_length[first_contact]
                )
                self._hammer_second_contact_timestep[second_contact] = (
                    self.env.episode_length[second_contact]
                )

        if cfg.metric_id == "cube_strategy":
            current_z = self._object_ground_truth(cfg.cube_strategy_object_id)["pos"][
                :, 2
            ]
            self._cube_max_center_z = torch.maximum(
                self._cube_max_center_z,
                current_z,
            )

        if cfg.metric_id == "dispenser_pressed":
            current_position = self._dispenser_head_joint_position()
            self._dispenser_min_head_position_m = torch.minimum(
                self._dispenser_min_head_position_m,
                current_position,
            )

        if cfg.metric_id == "dispenser_press_return_twice":
            self._dispenser_cycle_tracker.update(self._dispenser_head_joint_position())

        if cfg.metric_id == "eraser_tracking":
            self._accumulate_eraser_tracking_error()

        if cfg.metric_id == "grasp_stability":
            current_position = self._object_ground_truth(cfg.grasp_stability_object_id)[
                "pos"
            ]
            drift = torch.linalg.norm(
                current_position - self._grasp_reference_object_position,
                dim=-1,
            )
            self._grasp_max_object_drift_m = torch.maximum(
                self._grasp_max_object_drift_m,
                drift,
            )

        if cfg.eval_metric_id == "mug_grasp_change" and self.env.mode == "train":
            score = self.env.reward_manager.common.retargeted_joint_mse_tracking_score(
                apply_reference_noise=False,
                log_components=False,
            )
            expected_shape = (self.env.num_envs,)
            if tuple(score.shape) != expected_shape:
                raise ValueError(
                    f"Expected mug hand tracking score shape {expected_shape}, "
                    f"got {tuple(score.shape)}."
                )
            self._mug_hand_tracking_score_sum += score
            self._mug_hand_tracking_step_count += 1

    def reset_task_metric_accumulators(self, env_ids: np.ndarray) -> None:
        cfg = self.task_cfg.metric_cfg
        if cfg.metric_id == "scissors_tracking":
            self._scissors_tracking.reset(torch.as_tensor(env_ids, dtype=torch.long, device=self.env.device))
        if cfg.metric_id in {
            "hammer_peg_impact",
            "hammer_single_swing",
            "hammer_two_swing",
        }:
            self._hammer_contact_impact_sum[env_ids] = 0.0
            self._hammer_contact_step_count[env_ids] = 0
            if cfg.metric_id == "hammer_single_swing":
                self._hammer_strike_contact_timestep[env_ids] = -1
            elif cfg.metric_id == "hammer_two_swing":
                self._hammer_first_contact_timestep[env_ids] = -1
                self._hammer_second_contact_timestep[env_ids] = -1
        if cfg.metric_id == "cube_strategy":
            env_ids_tensor = torch.as_tensor(
                env_ids,
                dtype=torch.long,
                device=self.env.device,
            )
            current_z = self._object_ground_truth(cfg.cube_strategy_object_id)["pos"][
                env_ids_tensor, 2
            ]
            self._cube_initial_center_z[env_ids_tensor] = current_z
            self._cube_max_center_z[env_ids_tensor] = current_z
        if cfg.metric_id == "dispenser_pressed":
            env_ids_tensor = torch.as_tensor(
                env_ids,
                dtype=torch.long,
                device=self.env.device,
            )
            current_position = self._dispenser_head_joint_position()
            self._dispenser_min_head_position_m[env_ids_tensor] = current_position[
                env_ids_tensor
            ]
        if cfg.metric_id == "dispenser_press_return_twice":
            self._dispenser_cycle_tracker.reset(
                torch.as_tensor(env_ids, dtype=torch.long, device=self.env.device)
            )
        if cfg.metric_id == "eraser_tracking":
            env_ids_tensor = torch.as_tensor(
                env_ids,
                dtype=torch.long,
                device=self.env.device,
            )
            self._eraser_max_position_error_m[env_ids_tensor] = -torch.inf
            self._eraser_phase_step_count[env_ids_tensor] = 0
            self._accumulate_eraser_tracking_error(env_ids_tensor)
        if cfg.metric_id == "grasp_stability":
            env_ids_tensor = torch.as_tensor(
                env_ids,
                dtype=torch.long,
                device=self.env.device,
            )
            current_position = self._object_ground_truth(cfg.grasp_stability_object_id)[
                "pos"
            ]
            reset_drift = torch.linalg.norm(
                current_position[env_ids_tensor]
                - self._grasp_reference_object_position,
                dim=-1,
            )
            self._grasp_max_object_drift_m[env_ids_tensor] = reset_drift
        if cfg.eval_metric_id == "mug_grasp_change":
            self._mug_hand_tracking_score_sum[env_ids] = 0.0
            self._mug_hand_tracking_step_count[env_ids] = 0

    def compute_episode_end_task_metrics(
        self,
        env_ids: torch.Tensor,
        early_terminated: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        episode_metrics = super().compute_episode_end_task_metrics(
            env_ids, early_terminated
        )
        if self.task_cfg.metric_cfg.metric_id == "bottle_interaction":
            episode_metrics["bottle_interaction_success"][early_terminated] = 0.0
        return episode_metrics

    def compute_episode_end_checkpoint_metric(
        self,
        env_ids: torch.Tensor,
        early_terminated: torch.Tensor,
        episode_task_metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        if self.task_cfg.metric_cfg.metric_id in {
            "hammer_single_swing",
            "hammer_two_swing",
        }:
            completion = super().compute_episode_end_checkpoint_metric(
                env_ids,
                early_terminated,
                episode_task_metrics,
            )
            if completion is None:
                raise RuntimeError(
                    "Hammer checkpoint selection requires a completion metric."
                )
            if env_ids.ndim != 1 or env_ids.dtype != torch.long:
                raise ValueError(
                    "Expected rank-1 hammer checkpoint env_ids with dtype "
                    f"torch.long, got shape={tuple(env_ids.shape)} and "
                    f"dtype={env_ids.dtype}."
                )
            if completion.shape != env_ids.shape:
                raise ValueError(
                    "Expected one hammer completion value per completed "
                    f"environment, got completion={tuple(completion.shape)} and "
                    f"env_ids={tuple(env_ids.shape)}."
                )
            if (
                self.env.start_timestep.ndim != 1
                or self.env.start_timestep.dtype != torch.long
            ):
                raise ValueError(
                    "Expected env.start_timestep to be a rank-1 torch.long tensor, "
                    f"got shape={tuple(self.env.start_timestep.shape)} and "
                    f"dtype={self.env.start_timestep.dtype}."
                )

            # Only episodes matching the fixed-start evaluation protocol may
            # influence best-checkpoint selection. All terminal episodes remain
            # present in the ordinary task-metric learning curves.
            start_timestep_ids = env_ids.to(self.env.start_timestep.device)
            episode_start_timesteps = self.env.start_timestep[start_timestep_ids]
            eval_start_timestep = self.env.cfg.reset_cfg.eval_start_timestep
            if type(eval_start_timestep) is not int:
                raise TypeError(
                    "Expected reset_cfg.eval_start_timestep to be an int, got "
                    f"{type(eval_start_timestep).__name__}."
                )
            full_start_mask = episode_start_timesteps.eq(eval_start_timestep)
            return completion[full_start_mask.to(completion.device)]

        if self.task_cfg.metric_cfg.eval_metric_id != "mug_grasp_change":
            return super().compute_episode_end_checkpoint_metric(
                env_ids,
                early_terminated,
                episode_task_metrics,
            )
        counts = self._mug_hand_tracking_step_count[env_ids]
        if torch.any(counts <= 0):
            raise ValueError(
                "Mug checkpoint metric requires at least one accumulated training "
                f"step per completed episode, got counts {counts}."
            )
        score = self._mug_hand_tracking_score_sum[env_ids] / counts.to(torch.float32)
        score = score.clone()
        score[early_terminated] = 0.0
        return score

    def _object_ground_truth(self, object_id: str) -> dict[str, torch.Tensor]:
        return self.env.physics_manager.object_entities[
            object_id
        ].get_ground_truth_info()

    def _compute_open_notebook_metrics(self) -> dict[str, torch.Tensor]:
        cfg = self.task_cfg.metric_cfg
        object_index = self.scene_object_ids.index(cfg.notebook_object_id)
        valid_arti = self.scene_object_arti_valid_mask[object_index]
        reference_final = torch.as_tensor(
            self.scene_object_arti_vals[-1, object_index, valid_arti],
            dtype=torch.float32,
            device=self.env.device,
        )
        current = self._object_ground_truth(cfg.notebook_object_id)["arti"]
        expected_shape = (self.env.num_envs, int(valid_arti.sum()))
        if tuple(current.shape) != expected_shape:
            raise ValueError(
                f"Expected notebook articulation shape {expected_shape}, got {tuple(current.shape)}."
            )
        error = torch.abs(current - reference_final[None, :]).amax(dim=-1)
        within_tolerance = error <= float(cfg.notebook_angle_tolerance_rad)
        return {
            "notebook_final_angle_error_rad": error,
            "notebook_final_angle_within_15deg": within_tolerance.to(torch.float32),
        }

    def _compute_block_stacking_metrics(self) -> dict[str, torch.Tensor]:
        cfg = self.task_cfg.metric_cfg
        object_ids = tuple(cfg.stacking_object_ids)
        offsets = torch.as_tensor(
            cfg.stacking_min_center_height_offsets_m,
            dtype=torch.float32,
            device=self.env.device,
        )
        center_z = torch.stack(
            [
                self._object_ground_truth(object_id)["pos"][:, 2]
                for object_id in object_ids
            ],
            dim=1,
        )
        margins = center_z[:, 1:] - center_z[:, :-1] - offsets[None, :]
        min_margin = margins.amin(dim=-1)
        success = torch.all(margins >= 0.0, dim=-1)
        return {
            "stack_min_center_height_margin_m": min_margin,
            "stack_success": success.to(torch.float32),
        }

    def _scissors_tracking_sample(self) -> ScissorsSample:
        cfg = self.task_cfg.metric_cfg
        asset = self.env.physics_manager.object_entities[cfg.scissors_object_id]
        reference = {name: values[self.env.command_time_step] for name, values in self._scissors_reference.items()}
        return self._scissors_tracking.sample(asset.object, asset.get_ground_truth_info(), reference,
                                              self.env.physics_manager.get_scene_table_top_height())

    def _compute_scissors_tracking_metrics(self) -> dict[str, torch.Tensor]:
        return self._scissors_tracking.metrics(self._scissors_tracking_sample())

    def _compute_banana_handover_metrics(self) -> dict[str, torch.Tensor]:
        cfg = self.task_cfg.metric_cfg
        banana = self.env.physics_manager.object_entities[cfg.banana_object_id]
        banana_pos = banana.get_ground_truth_info()["pos"]
        table_top_height = self.env.physics_manager.get_scene_table_top_height()
        height_above_table = banana_pos[:, 2] - table_top_height
        not_fallen = height_above_table >= float(
            cfg.banana_min_center_height_above_table_m
        )

        def contact_diagnostics(
            hand_side: str,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            hand: Hand = self.env.physics_manager.scene_entities[hand_side]
            contact_signal, net_force_on_banana = (
                self.env.physics_manager.get_hand_object_contact_support_diagnostics(
                    hand,
                    banana,
                    threshold=float(cfg.contact_force_threshold),
                )
            )
            if contact_signal.ndim != 3:
                raise ValueError(
                    "Expected banana hand contact signal shape (N, L, 3) for "
                    f"{hand_side!r}, got {tuple(contact_signal.shape)}."
                )
            expected_force_shape = (self.env.num_envs, 3)
            if tuple(net_force_on_banana.shape) != expected_force_shape:
                raise ValueError(
                    "Expected net force on banana shape "
                    f"{expected_force_shape} for {hand_side!r}, got "
                    f"{tuple(net_force_on_banana.shape)}."
                )
            contact_link_count = torch.any(contact_signal.bool(), dim=-1).sum(dim=-1)
            return contact_link_count, net_force_on_banana

        source_contact_link_count, source_force_on_banana = contact_diagnostics(
            cfg.banana_source_hand
        )
        receiving_contact_link_count, receiving_force_on_banana = contact_diagnostics(
            cfg.banana_receiving_hand
        )
        env_link_mass = np.asarray(banana.env_link_mass, dtype=np.float32)
        if env_link_mass.ndim != 2 or env_link_mass.shape[0] != self.env.num_envs:
            raise ValueError(
                "Expected banana.env_link_mass shape (N, L), got "
                f"{env_link_mass.shape}."
            )
        gravity_magnitude = abs(float(self.env.sim_cfg.gravity))
        if gravity_magnitude <= 0.0:
            raise ValueError(
                "Banana support metric requires nonzero gravity, got "
                f"{self.env.sim_cfg.gravity}."
            )
        object_weight = torch.as_tensor(
            env_link_mass.sum(axis=1) * gravity_magnitude,
            device=self.env.device,
            dtype=torch.float32,
        )
        if torch.any(object_weight <= 0.0):
            raise ValueError(
                f"Banana support metric requires positive object weight, got {object_weight}."
            )
        source_net_contact_force = source_force_on_banana.norm(dim=-1)
        source_contact_force_weight_fraction = source_net_contact_force / object_weight
        source_released = source_contact_force_weight_fraction <= float(
            cfg.banana_max_source_contact_weight_fraction
        )
        receiving_upward_support_force = receiving_force_on_banana[:, 2].clamp_min(0.0)
        receiving_support_weight_fraction = (
            receiving_upward_support_force / object_weight
        )
        receiving_support = receiving_support_weight_fraction >= float(
            cfg.banana_min_receiving_support_weight_fraction
        )
        success = torch.logical_and(
            torch.logical_and(not_fallen, receiving_support),
            source_released,
        )
        return {
            "banana_center_height_above_table_m": height_above_table,
            "banana_in_air": not_fallen.to(torch.float32),
            "banana_not_fallen": not_fallen.to(torch.float32),
            "banana_object_weight_n": object_weight,
            "banana_source_hand_contact_link_count": source_contact_link_count.to(
                torch.float32
            ),
            "banana_source_hand_net_contact_force_n": source_net_contact_force,
            "banana_source_hand_net_contact_force_weight_fraction": (
                source_contact_force_weight_fraction
            ),
            "banana_source_hand_released": source_released.to(torch.float32),
            "banana_receiving_hand_contact_link_count": (
                receiving_contact_link_count.to(torch.float32)
            ),
            "banana_receiving_hand_net_contact_force_n": (
                receiving_force_on_banana.norm(dim=-1)
            ),
            "banana_receiving_hand_upward_support_force_n": (
                receiving_upward_support_force
            ),
            "banana_receiving_hand_upward_support_weight_fraction": (
                receiving_support_weight_fraction
            ),
            "banana_receiving_hand_support": receiving_support.to(torch.float32),
            "banana_handover_success": success.to(torch.float32),
        }

    def _compute_hammer_peg_impact_metrics(self) -> dict[str, torch.Tensor]:
        counts = self._hammer_contact_step_count
        average_impact = torch.where(
            counts > 0,
            self._hammer_contact_impact_sum / counts.clamp_min(1).to(torch.float32),
            torch.zeros_like(self._hammer_contact_impact_sum),
        )
        return {
            "hammer_peg_average_contact_impact_n": average_impact,
            "hammer_peg_contact_step_count": counts.to(torch.float32),
        }

    def _compute_hammer_single_swing_metrics(self) -> dict[str, torch.Tensor]:
        force_metrics = self._compute_hammer_peg_impact_metrics()
        contacted = self._hammer_strike_contact_timestep >= 0
        completion_steps = self._hammer_strike_contact_timestep.to(torch.float32)
        completion_steps = torch.where(
            contacted,
            completion_steps,
            torch.full_like(completion_steps, torch.nan),
        )
        control_period_s = float(self.env.sim_cfg.dt) * int(self.env.cfg.decimation)
        if control_period_s <= 0.0:
            raise ValueError(
                f"Expected a positive control period, got {control_period_s}."
            )
        completion = contacted.to(torch.float32)
        return {
            "hammer_strike_contacted": completion,
            "hammer_single_swing_completed": completion,
            "hammer_single_swing_completion_time_steps": completion_steps,
            "hammer_single_swing_completion_time_s": completion_steps
            * control_period_s,
            **force_metrics,
        }

    def _compute_hammer_two_swing_metrics(self) -> dict[str, torch.Tensor]:
        force_metrics = self._compute_hammer_peg_impact_metrics()
        first_strike_contacted = self._hammer_first_contact_timestep >= 0
        second_strike_contacted = self._hammer_second_contact_timestep >= 0
        completed = torch.logical_and(
            first_strike_contacted,
            second_strike_contacted,
        )
        completion_steps = self._hammer_second_contact_timestep.to(torch.float32)
        completion_steps = torch.where(
            completed,
            completion_steps,
            torch.full_like(completion_steps, torch.nan),
        )
        control_period_s = float(self.env.sim_cfg.dt) * int(self.env.cfg.decimation)
        if control_period_s <= 0.0:
            raise ValueError(
                f"Expected a positive control period, got {control_period_s}."
            )
        return {
            "hammer_first_strike_contacted": first_strike_contacted.to(torch.float32),
            "hammer_second_strike_contacted": second_strike_contacted.to(torch.float32),
            "hammer_two_swings_completed": completed.to(torch.float32),
            "hammer_two_swing_completion_time_steps": completion_steps,
            "hammer_two_swing_completion_time_s": (completion_steps * control_period_s),
            **force_metrics,
        }

    def _object_final_pose_errors(
        self,
        object_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        object_index = self.scene_object_ids.index(object_id)
        reference_position = torch.as_tensor(
            self.scene_object_pos[-1, object_index],
            dtype=torch.float32,
            device=self.env.device,
        )
        reference_quaternion = torch.as_tensor(
            self.scene_object_quat[-1, object_index],
            dtype=torch.float32,
            device=self.env.device,
        )
        current = self._object_ground_truth(object_id)
        position_error = torch.linalg.norm(
            current["pos"] - reference_position.unsqueeze(0),
            dim=-1,
        )
        orientation_error = angle_error(
            current["quat"],
            reference_quaternion.unsqueeze(0).expand(self.env.num_envs, -1),
            current_format="quat",
            target_format="quat",
        )
        return position_error, orientation_error

    def _compute_object_pose_metrics(self) -> dict[str, torch.Tensor]:
        cfg = self.task_cfg.metric_cfg
        position_error, orientation_error = self._object_final_pose_errors(
            cfg.object_pose_object_id
        )
        success = torch.logical_and(
            position_error <= float(cfg.object_pose_position_tolerance_m),
            orientation_error <= float(cfg.object_pose_orientation_tolerance_rad),
        )
        return {
            "object_final_position_error_m": position_error,
            "object_final_orientation_error_rad": orientation_error,
            "object_pose_success": success.to(torch.float32),
        }

    def _object_lift_values(
        self, object_id: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        object_index = self.scene_object_ids.index(object_id)
        reference_initial_height = float(self.scene_object_pos[0, object_index, 2])
        reference_lift_height = float(
            self.scene_object_pos[-1, object_index, 2] - reference_initial_height
        )
        current_height = self._object_ground_truth(object_id)["pos"][:, 2]
        lift_height = current_height - reference_initial_height
        reference_lift = torch.full_like(lift_height, reference_lift_height)
        height_error = torch.abs(lift_height - reference_lift)
        success = height_error <= float(self.env.rewards_cfg.task_success_pos_thd)
        return lift_height, reference_lift, height_error, success

    def _compute_object_lift_metrics(self) -> dict[str, torch.Tensor]:
        cfg = self.task_cfg.metric_cfg
        lift_height, reference_lift, height_error, success = self._object_lift_values(
            cfg.object_lift_object_id
        )
        return {
            "object_lift_height_m": lift_height,
            "object_lift_reference_height_m": reference_lift,
            "object_lift_height_error_m": height_error,
            "object_lift_success": success.to(torch.float32),
        }

    def _compute_bottle_interaction_metrics(self) -> dict[str, torch.Tensor]:
        cfg = self.task_cfg.metric_cfg
        object_id = cfg.bottle_interaction_object_id
        hand_side = cfg.bottle_interaction_hand_side
        lift_height, reference_lift, height_error, lift_success = (
            self._object_lift_values(object_id)
        )

        bottle = self._object_ground_truth(object_id)
        hand: Hand = self.env.physics_manager.scene_entities[hand_side]
        wrist_position = self.env.physics_manager.get_entity_link_pos(
            hand.robot,
            link_name=hand.wrist_link_name,
            ref="link_origin",
        ).squeeze(1)
        wrist_quaternion = self.env.physics_manager.get_entity_link_quat(
            hand.robot,
            link_name=hand.wrist_link_name,
        ).squeeze(1)
        expected_position_shape = (self.env.num_envs, 3)
        expected_quaternion_shape = (self.env.num_envs, 4)
        if tuple(wrist_position.shape) != expected_position_shape:
            raise ValueError(
                "Expected bottle-interaction wrist position shape "
                f"{expected_position_shape}, got {tuple(wrist_position.shape)}."
            )
        if tuple(wrist_quaternion.shape) != expected_quaternion_shape:
            raise ValueError(
                "Expected bottle-interaction wrist quaternion shape "
                f"{expected_quaternion_shape}, got {tuple(wrist_quaternion.shape)}."
            )

        object_index = self.scene_object_ids.index(object_id)
        reference_wrist_position = getattr(
            self.env.reset_manager, f"frame_{hand_side}_wrist_pos"
        )[-1].to(device=self.env.device, dtype=wrist_position.dtype)
        reference_wrist_quaternion = getattr(
            self.env.reset_manager, f"frame_{hand_side}_wrist_quat"
        )[-1].to(device=self.env.device, dtype=wrist_quaternion.dtype)
        reference_object_position = torch.as_tensor(
            self.scene_object_pos[-1, object_index],
            device=self.env.device,
            dtype=wrist_position.dtype,
        )
        reference_object_quaternion = torch.as_tensor(
            self.scene_object_quat[-1, object_index],
            device=self.env.device,
            dtype=wrist_quaternion.dtype,
        )
        expected_reference_shapes = {
            "wrist position": (reference_wrist_position, (3,)),
            "wrist quaternion": (reference_wrist_quaternion, (4,)),
            "object position": (reference_object_position, (3,)),
            "object quaternion": (reference_object_quaternion, (4,)),
        }
        for name, (value, expected_shape) in expected_reference_shapes.items():
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"Expected final reference {name} shape {expected_shape}, got "
                    f"{tuple(value.shape)}."
                )

        current_object_rotation = matrix_from_quat(bottle["quat"])
        current_wrist_rotation = matrix_from_quat(wrist_quaternion)
        reference_object_rotation = matrix_from_quat(
            reference_object_quaternion.unsqueeze(0)
        )[0]
        reference_wrist_rotation = matrix_from_quat(
            reference_wrist_quaternion.unsqueeze(0)
        )[0]
        current_relative_position = torch.matmul(
            current_object_rotation.transpose(-1, -2),
            (wrist_position - bottle["pos"]).unsqueeze(-1),
        ).squeeze(-1)
        reference_relative_position = torch.matmul(
            reference_object_rotation.transpose(-1, -2),
            reference_wrist_position - reference_object_position,
        )
        relative_position_error = torch.linalg.norm(
            current_relative_position - reference_relative_position.unsqueeze(0),
            dim=-1,
        )

        current_relative_rotation = torch.matmul(
            current_object_rotation.transpose(-1, -2), current_wrist_rotation
        )
        reference_relative_rotation = torch.matmul(
            reference_object_rotation.transpose(-1, -2), reference_wrist_rotation
        )
        rotation_delta = torch.matmul(
            reference_relative_rotation.transpose(-1, -2).unsqueeze(0),
            current_relative_rotation,
        )
        rotation_trace = torch.diagonal(rotation_delta, dim1=-2, dim2=-1).sum(-1)
        relative_orientation_error = torch.acos(
            ((rotation_trace - 1.0) * 0.5).clamp(-1.0, 1.0)
        )
        interaction_match = torch.logical_and(
            relative_position_error
            <= float(cfg.bottle_relative_wrist_position_tolerance_m),
            relative_orientation_error
            <= float(cfg.bottle_relative_wrist_orientation_tolerance_rad),
        )
        interaction_success = torch.logical_and(lift_success, interaction_match)
        return {
            "object_lift_height_m": lift_height,
            "object_lift_reference_height_m": reference_lift,
            "object_lift_height_error_m": height_error,
            "object_lift_success": lift_success.to(torch.float32),
            "bottle_relative_wrist_position_error_m": relative_position_error,
            "bottle_relative_wrist_orientation_error_rad": (relative_orientation_error),
            "bottle_requested_interaction_match": interaction_match.to(torch.float32),
            "bottle_interaction_success": interaction_success.to(torch.float32),
        }

    def _compute_cube_strategy_metrics(self) -> dict[str, torch.Tensor]:
        cfg = self.task_cfg.metric_cfg
        position_error, orientation_error = self._object_final_pose_errors(
            cfg.cube_strategy_object_id
        )
        pose_success = torch.logical_and(
            position_error <= float(cfg.cube_strategy_position_tolerance_m),
            orientation_error <= float(cfg.cube_strategy_orientation_tolerance_rad),
        )
        max_height_change = self._cube_max_center_z - self._cube_initial_center_z
        lifted = max_height_change >= float(cfg.cube_strategy_min_lift_height_change_m)
        expected_lift = bool(cfg.cube_strategy_expected_lift)
        strategy_match = lifted if expected_lift else torch.logical_not(lifted)
        strategy_success = torch.logical_and(pose_success, strategy_match)
        return {
            "cube_final_position_error_m": position_error,
            "cube_final_orientation_error_rad": orientation_error,
            "cube_final_pose_success": pose_success.to(torch.float32),
            "cube_max_center_height_change_m": max_height_change,
            "cube_lifted": lifted.to(torch.float32),
            "cube_strategy_success": strategy_success.to(torch.float32),
        }

    def _dispenser_head_joint_position(self) -> torch.Tensor:
        cfg = self.task_cfg.metric_cfg
        articulation = self._object_ground_truth(cfg.dispenser_object_id)["arti"]
        expected_shape = (self.env.num_envs, 1)
        if tuple(articulation.shape) != expected_shape:
            raise ValueError(
                "Expected one dispenser articulation value per environment with "
                f"shape {expected_shape}, got {tuple(articulation.shape)}."
            )
        return articulation[:, 0]

    def _compute_dispenser_pressed_metrics(self) -> dict[str, torch.Tensor]:
        cfg = self.task_cfg.metric_cfg
        pressed_threshold = self._dispenser_pressed_reference_position_m + float(
            cfg.dispenser_pressed_tolerance_m
        )
        achieved = self._dispenser_min_head_position_m <= pressed_threshold
        return {
            "dispenser_min_head_joint_position_m": (
                self._dispenser_min_head_position_m
            ),
            "dispenser_pressed_state_achieved": achieved.to(torch.float32),
        }

    def _compute_dispenser_cycle_metrics(self) -> dict[str, torch.Tensor]:
        return self._dispenser_cycle_tracker.metrics()

    def _drawer_joint_position(self) -> torch.Tensor:
        cfg = self.task_cfg.metric_cfg
        articulation = self._object_ground_truth(cfg.drawer_object_id)["arti"]
        expected_shape = (self.env.num_envs, 1)
        if tuple(articulation.shape) != expected_shape:
            raise ValueError(
                "Expected one drawer articulation value per environment with "
                f"shape {expected_shape}, got {tuple(articulation.shape)}."
            )
        return articulation[:, 0]

    def _compute_drawer_articulation_metrics(self) -> dict[str, torch.Tensor]:
        cfg = self.task_cfg.metric_cfg
        object_index = self.scene_object_ids.index(cfg.drawer_object_id)
        valid_arti = self.scene_object_arti_valid_mask[object_index]
        reference_final = float(
            self.scene_object_arti_vals[-1, object_index, valid_arti][0]
        )
        error = torch.abs(self._drawer_joint_position() - reference_final)
        success = error <= float(cfg.drawer_final_articulation_tolerance_m)
        return {
            "drawer_final_articulation_error_m": error,
            "drawer_articulation_success": success.to(torch.float32),
        }

    def _accumulate_eraser_tracking_error(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        cfg = self.task_cfg.metric_cfg
        if env_ids is None:
            env_ids = torch.arange(
                self.env.num_envs,
                dtype=torch.long,
                device=self.env.device,
            )
        if env_ids.ndim != 1 or env_ids.dtype != torch.long:
            raise ValueError(
                "Expected eraser accumulator env_ids to be rank-1 torch.long, got "
                f"shape={tuple(env_ids.shape)} and dtype={env_ids.dtype}."
            )
        command_timesteps = self.env.command_time_step[env_ids]
        phase_indices = self.phase_index_by_timestep[command_timesteps]
        erase_mask = torch.isin(phase_indices, self._eraser_phase_indices)
        if not torch.any(erase_mask):
            return
        erase_env_ids = env_ids[erase_mask]
        erase_timesteps = command_timesteps[erase_mask]
        object_index = self.scene_object_ids.index(cfg.eraser_object_id)
        reference_positions = torch.as_tensor(
            self.scene_object_pos[:, object_index],
            dtype=torch.float32,
            device=self.env.device,
        )[erase_timesteps]
        current_positions = self._object_ground_truth(cfg.eraser_object_id)["pos"]
        expected_shape = (self.env.num_envs, 3)
        if tuple(current_positions.shape) != expected_shape:
            raise ValueError(
                "Expected one eraser position per environment with shape "
                f"{expected_shape}, got {tuple(current_positions.shape)}."
            )
        position_error = torch.linalg.norm(
            current_positions[erase_env_ids] - reference_positions,
            dim=-1,
        )
        self._eraser_max_position_error_m[erase_env_ids] = torch.maximum(
            self._eraser_max_position_error_m[erase_env_ids],
            position_error,
        )
        self._eraser_phase_step_count[erase_env_ids] += 1

    def _compute_eraser_tracking_metrics(self) -> dict[str, torch.Tensor]:
        cfg = self.task_cfg.metric_cfg
        phase_complete = (
            self._eraser_phase_step_count >= self._eraser_reference_step_count
        )
        max_error = torch.where(
            self._eraser_phase_step_count > 0,
            self._eraser_max_position_error_m,
            torch.full_like(self._eraser_max_position_error_m, torch.inf),
        )
        success = torch.logical_and(
            phase_complete,
            max_error < float(cfg.eraser_max_position_error_m),
        )
        return {
            "eraser_erase_phase_max_position_error_m": max_error,
            "eraser_erase_phase_step_count": self._eraser_phase_step_count.to(
                torch.float32
            ),
            "eraser_erase_phase_complete": phase_complete.to(torch.float32),
            "eraser_erase_phase_success": success.to(torch.float32),
        }

    @staticmethod
    def _point_distance_from_axis(
        points: torch.Tensor,
        axis_origins: torch.Tensor,
        unit_axes: torch.Tensor,
    ) -> torch.Tensor:
        offsets = points - axis_origins
        axial_offsets = torch.sum(offsets * unit_axes, dim=-1, keepdim=True)
        radial_offsets = offsets - axial_offsets * unit_axes
        return torch.linalg.norm(radial_offsets, dim=-1)

    def _compute_precision_mounting_metrics(self) -> dict[str, torch.Tensor]:
        cfg = self.task_cfg.metric_cfg
        cylinder = self._object_ground_truth(cfg.mounting_cylinder_object_id)
        post = self._object_ground_truth(cfg.mounting_post_object_id)
        expected_pos_shape = (self.env.num_envs, 3)
        expected_quat_shape = (self.env.num_envs, 4)
        for object_id, state in (
            (cfg.mounting_cylinder_object_id, cylinder),
            (cfg.mounting_post_object_id, post),
        ):
            if tuple(state["pos"].shape) != expected_pos_shape:
                raise ValueError(
                    f"Expected mounting object {object_id!r} position shape "
                    f"{expected_pos_shape}, got {tuple(state['pos'].shape)}."
                )
            if tuple(state["quat"].shape) != expected_quat_shape:
                raise ValueError(
                    f"Expected mounting object {object_id!r} quaternion shape "
                    f"{expected_quat_shape}, got {tuple(state['quat'].shape)}."
                )

        # The local z axes are the cylinder bore and mounting-post axes. Checking
        # both cylinder endpoints makes the complete straight bore stay within
        # the physical radial clearance while still allowing a valid lean.
        cylinder_axis = matrix_from_quat(cylinder["quat"])[..., :, 2]
        post_axis = matrix_from_quat(post["quat"])[..., :, 2]
        cylinder_top = (
            cylinder["pos"] + float(cfg.mounting_cylinder_height_m) * cylinder_axis
        )
        bottom_radial_error = self._point_distance_from_axis(
            cylinder["pos"], post["pos"], post_axis
        )
        top_radial_error = self._point_distance_from_axis(
            cylinder_top, post["pos"], post_axis
        )
        max_radial_error = torch.maximum(bottom_radial_error, top_radial_error)

        origin_height = torch.sum(
            (cylinder["pos"] - post["pos"]) * post_axis,
            dim=-1,
        )
        axis_dot = torch.sum(cylinder_axis * post_axis, dim=-1).clamp(-1.0, 1.0)
        axis_tilt = torch.acos(axis_dot)
        height_threshold = torch.full_like(
            origin_height,
            float(cfg.mounting_max_origin_height_above_post_m),
        )
        radial_threshold = torch.full_like(
            max_radial_error,
            float(cfg.mounting_bore_clearance_m),
        )
        tilt_threshold = torch.full_like(
            axis_tilt,
            float(cfg.mounting_max_axis_tilt_rad),
        )
        success = torch.logical_and(
            origin_height <= height_threshold,
            torch.logical_and(
                max_radial_error <= radial_threshold,
                axis_tilt <= tilt_threshold,
            ),
        )
        return {
            "mounting_origin_height_above_post_m": origin_height,
            "mounting_origin_height_threshold_m": height_threshold,
            "mounting_bottom_axis_radial_error_m": bottom_radial_error,
            "mounting_top_axis_radial_error_m": top_radial_error,
            "mounting_max_axis_radial_error_m": max_radial_error,
            "mounting_radial_threshold_m": radial_threshold,
            "mounting_axis_tilt_rad": axis_tilt,
            "mounting_tilt_threshold_rad": tilt_threshold,
            "mounting_success": success.to(torch.float32),
        }

    def _compute_grasp_stability_metrics(self) -> dict[str, torch.Tensor]:
        cfg = self.task_cfg.metric_cfg
        current_position = self._object_ground_truth(cfg.grasp_stability_object_id)[
            "pos"
        ]
        current_drift = torch.linalg.norm(
            current_position - self._grasp_reference_object_position,
            dim=-1,
        )
        max_drift = torch.maximum(
            self._grasp_max_object_drift_m,
            current_drift,
        )
        # The object may settle while the policy closes the grasp. Completion is
        # therefore determined at the current (terminal when logged) pose, while
        # maximum drift remains available as an episode-wide diagnostic.
        success = current_drift <= float(cfg.grasp_stability_max_drift_m)
        return {
            "grasp_current_object_drift_m": current_drift,
            "grasp_max_object_drift_m": max_drift,
            "grasp_stability_success": success.to(torch.float32),
        }

    def _initialize_phase_metadata(self) -> None:
        assert self.original_demo is not None
        phase_names = list(getattr(self.original_demo, "phase_names", []))
        phase_lengths = np.asarray(
            getattr(self.original_demo, "phase_lengths", np.asarray([])),
            dtype=np.int64,
        )
        if len(phase_names) == 0 or phase_lengths.shape != (len(phase_names),):
            raise ValueError(
                "MultiObjectTaskManager requires phase_names and phase_lengths "
                f"with matching lengths, got {len(phase_names)} names and "
                f"phase_lengths shape {phase_lengths.shape}."
            )
        if np.any(phase_lengths <= 0):
            raise ValueError(
                f"Expected all phase_lengths to be positive, got {phase_lengths}."
            )
        if int(phase_lengths.sum()) != self.total_steps:
            raise ValueError(
                "Expected phase_lengths to sum to task_manager.total_steps, got "
                f"{int(phase_lengths.sum())} and {self.total_steps}."
            )

        self.phase_names = phase_names
        self.phase_lengths = phase_lengths
        self.phase_index_by_timestep = torch.repeat_interleave(
            torch.arange(len(phase_names), dtype=torch.long, device=self.env.device),
            torch.as_tensor(phase_lengths, dtype=torch.long, device=self.env.device),
        )

    def _initialize_scene_object_metadata(self) -> None:
        assert self.original_demo is not None
        retargeted_motion = getattr(self.original_demo, "retargeted_motion", None)
        if not isinstance(retargeted_motion, dict):
            raise ValueError(
                "MultiObjectTaskManager requires original_demo.retargeted_motion."
            )

        required_keys = (
            "scene_object_ids",
            "scene_object_names",
            "scene_object_urdf_paths",
            "scene_object_pos",
            "scene_object_quat",
            "scene_object_arti_vals",
            "scene_object_arti_valid_mask",
            "scene_object_arti_names",
            "scene_object_active_mask",
            "scene_object_static",
        )
        missing = [key for key in required_keys if key not in retargeted_motion]
        if missing:
            present_scene_keys = [
                key for key in required_keys if key in retargeted_motion
            ]
            if present_scene_keys:
                raise ValueError(
                    "Retargeted motion contains partial multi-object scene fields. "
                    f"Present scene keys: {present_scene_keys}; missing {missing}."
                )
            self._initialize_single_object_scene_metadata(retargeted_motion, missing)
            return

        self.scene_object_ids = self._object_list(retargeted_motion["scene_object_ids"])
        self.scene_object_names = self._object_list(
            retargeted_motion["scene_object_names"]
        )
        self.scene_object_urdf_paths = self._object_list(
            retargeted_motion["scene_object_urdf_paths"]
        )
        self.scene_object_arti_names = [
            self._object_list(names)
            for names in np.asarray(
                retargeted_motion["scene_object_arti_names"], dtype=object
            )
        ]
        self.scene_object_pos = np.asarray(
            retargeted_motion["scene_object_pos"], dtype=np.float32
        )
        self.scene_object_quat = np.asarray(
            retargeted_motion["scene_object_quat"], dtype=np.float32
        )
        self.scene_object_arti_vals = np.asarray(
            retargeted_motion["scene_object_arti_vals"], dtype=np.float32
        )
        self.scene_object_arti_valid_mask = np.asarray(
            retargeted_motion["scene_object_arti_valid_mask"], dtype=bool
        )
        self.scene_object_active_mask = np.asarray(
            retargeted_motion["scene_object_active_mask"], dtype=bool
        )
        self.scene_object_static = np.asarray(
            retargeted_motion["scene_object_static"], dtype=bool
        )
        self.scene_object_friction = np.asarray(
            retargeted_motion.get(
                "scene_object_friction",
                np.ones(len(self.scene_object_ids), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        self.scene_active_object_index = np.asarray(
            retargeted_motion.get(
                "scene_active_object_index",
                self._derive_active_object_index(self.scene_object_active_mask),
            ),
            dtype=np.int64,
        )
        self.scene_canonical_object_id = str(
            retargeted_motion.get(
                "scene_canonical_object_id",
                self.scene_object_ids[0],
            )
        )
        self.scene_is_synthetic_single_object = False
        self.scene_table_height = float(
            retargeted_motion.get(
                "scene_table_height",
                self.task_cfg.lift_support_height,
            )
        )
        self.scene_table_margin = float(
            retargeted_motion.get(
                "scene_table_margin",
                0.001,
            )
        )
        self.scene_table_size = np.asarray(
            retargeted_motion.get(
                "scene_table_size",
                np.asarray([0.8, 1.2], dtype=np.float32),
            ),
            dtype=np.float32,
        )
        self.scene_uses_block = bool(retargeted_motion.get("scene_uses_block", False))
        self.scene_block_height = 0.05
        self.scene_support_block_ids = self._object_list(
            retargeted_motion.get("scene_support_block_ids", [])
        )
        self.scene_support_block_pos = np.asarray(
            retargeted_motion.get(
                "scene_support_block_pos",
                np.zeros((0, 3), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        self.scene_support_block_size = np.asarray(
            retargeted_motion.get(
                "scene_support_block_size",
                np.zeros((0, 3), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        self._validate_scene_object_metadata()

    def _initialize_single_object_scene_metadata(
        self,
        retargeted_motion: dict[str, Any],
        missing_scene_keys: list[str],
    ) -> None:
        single_object_keys = ("obj_pos", "obj_quat", "obj_arti", "joint_names")
        missing_single_object = [
            key for key in single_object_keys if key not in retargeted_motion
        ]
        if missing_single_object:
            raise ValueError(
                "Multi-object env requires either full scene object fields "
                f"or single-object retargeted motion fields {single_object_keys}; "
                f"missing scene keys {missing_scene_keys} and single-object keys "
                f"{missing_single_object}."
            )

        object_id = "object"
        object_name = str(getattr(self.original_demo, "target_object_name", object_id))
        urdf_path = str(getattr(self.original_demo, "target_obj_urdf_path"))
        if len(urdf_path) == 0:
            raise ValueError(
                "Single-object multi-object compatibility requires "
                "original_demo.target_obj_urdf_path."
            )

        obj_pos = np.asarray(retargeted_motion["obj_pos"], dtype=np.float32)
        obj_quat = np.asarray(retargeted_motion["obj_quat"], dtype=np.float32)
        obj_arti = np.asarray(retargeted_motion["obj_arti"], dtype=np.float32)
        joint_names = self._object_list(retargeted_motion["joint_names"])

        if obj_pos.shape != (self.total_steps, 3):
            raise ValueError(
                f"Expected single-object obj_pos shape {(self.total_steps, 3)}, "
                f"got {obj_pos.shape}."
            )
        if obj_quat.shape != (self.total_steps, 4):
            raise ValueError(
                f"Expected single-object obj_quat shape {(self.total_steps, 4)}, "
                f"got {obj_quat.shape}."
            )
        if obj_arti.ndim != 2 or obj_arti.shape[0] != self.total_steps:
            raise ValueError(
                "Expected single-object obj_arti shape (T, A), got "
                f"{obj_arti.shape}."
            )
        if obj_arti.shape[1] != len(joint_names):
            raise ValueError(
                "Expected single-object joint_names length to match obj_arti dim, "
                f"got {len(joint_names)} names and obj_arti shape {obj_arti.shape}."
            )

        self.scene_object_ids = [object_id]
        self.scene_object_names = [object_name]
        self.scene_object_urdf_paths = [urdf_path]
        self.scene_object_arti_names = [joint_names]
        self.scene_object_pos = obj_pos[:, None, :]
        self.scene_object_quat = obj_quat[:, None, :]
        self.scene_object_arti_vals = obj_arti[:, None, :]
        self.scene_object_arti_valid_mask = np.ones(
            (1, obj_arti.shape[1]),
            dtype=bool,
        )
        self.scene_object_active_mask = np.ones(
            (self.total_steps, 1),
            dtype=bool,
        )
        self.scene_object_static = np.array(
            [bool(retargeted_motion.get("object_static", False))],
            dtype=bool,
        )
        self.scene_object_friction = np.ones(1, dtype=np.float32)
        self.scene_active_object_index = np.zeros(self.total_steps, dtype=np.int64)
        self.scene_canonical_object_id = object_id
        self.scene_is_synthetic_single_object = True
        self.scene_table_margin = 0.001
        self.scene_table_size = np.asarray([0.8, 1.2], dtype=np.float32)
        self.scene_block_height = 0.05
        self.scene_support_block_ids = []
        self.scene_support_block_pos = np.zeros((0, 3), dtype=np.float32)
        self.scene_support_block_size = np.zeros((0, 3), dtype=np.float32)
        lowest_z = self._single_object_lowest_z(
            urdf_path=urdf_path,
            obj_pos=obj_pos[0],
            obj_quat=obj_quat[0],
            obj_arti=obj_arti[0],
            joint_names=joint_names,
        )
        self.scene_table_height = (
            lowest_z - self.scene_table_margin - self.scene_block_height
        )
        if self.scene_table_height <= 0.0:
            raise ValueError(
                "Synthetic single-object support would place the table below a "
                f"non-positive height: lowest_z={lowest_z}, "
                f"margin={self.scene_table_margin}, "
                f"block_height={self.scene_block_height}."
            )
        self.scene_uses_block = True
        self._validate_scene_object_metadata()

    def _single_object_lowest_z(
        self,
        *,
        urdf_path: str,
        obj_pos: np.ndarray,
        obj_quat: np.ndarray,
        obj_arti: np.ndarray,
        joint_names: list[str],
    ) -> float:
        urdf_inst = URDF.load(str(process_urdf(urdf_path)))
        joint_dict = dict(
            zip(
                joint_names,
                np.asarray(obj_arti, dtype=np.float32).reshape(-1),
            )
        )
        fk = urdf_inst.visual_trimesh_fk(joint_dict)
        if len(fk) == 0:
            fk = urdf_inst.collision_trimesh_fk(joint_dict)
        if len(fk) == 0:
            raise ValueError(
                f"Single-object scene object {urdf_path!r} has no visual or "
                "collision meshes."
            )
        mesh_pieces = []
        for mesh, pose in fk.items():
            transformed = mesh.copy()
            transformed.apply_transform(pose)
            mesh_pieces.append(transformed)
        mesh = trimesh.util.concatenate(mesh_pieces)
        object_pose = np.eye(4, dtype=np.float64)
        object_pose[:3, :3] = R.from_quat(np.roll(obj_quat, -1)).as_matrix()
        object_pose[:3, 3] = np.asarray(obj_pos, dtype=np.float64).reshape(3)
        mesh.apply_transform(object_pose)
        return float(mesh.bounds[0, 2])

    def _validate_scene_object_metadata(self) -> None:
        n_objects = len(self.scene_object_ids)
        if n_objects == 0:
            raise ValueError("Multi-object env requires at least one scene object.")
        if len(set(self.scene_object_ids)) != n_objects:
            raise ValueError(
                f"Expected unique scene_object_ids, got {self.scene_object_ids}."
            )
        for key, value in (
            ("scene_object_names", self.scene_object_names),
            ("scene_object_urdf_paths", self.scene_object_urdf_paths),
            ("scene_object_arti_names", self.scene_object_arti_names),
        ):
            if len(value) != n_objects:
                raise ValueError(
                    f"Expected {key} length {n_objects}, got {len(value)}."
                )
        expected_pos_shape = (self.total_steps, n_objects, 3)
        if tuple(self.scene_object_pos.shape) != expected_pos_shape:
            raise ValueError(
                f"Expected scene_object_pos shape {expected_pos_shape}, "
                f"got {self.scene_object_pos.shape}."
            )
        expected_quat_shape = (self.total_steps, n_objects, 4)
        if tuple(self.scene_object_quat.shape) != expected_quat_shape:
            raise ValueError(
                f"Expected scene_object_quat shape {expected_quat_shape}, "
                f"got {self.scene_object_quat.shape}."
            )
        if self.scene_object_arti_vals.ndim != 3 or self.scene_object_arti_vals.shape[
            :2
        ] != (self.total_steps, n_objects):
            raise ValueError(
                "Expected scene_object_arti_vals shape (T, N, Amax), got "
                f"{self.scene_object_arti_vals.shape}."
            )
        if (
            self.scene_object_arti_valid_mask.shape
            != self.scene_object_arti_vals.shape[1:]
        ):
            raise ValueError(
                "Expected scene_object_arti_valid_mask shape (N, Amax), got "
                f"{self.scene_object_arti_valid_mask.shape}."
            )
        if tuple(self.scene_object_active_mask.shape) != (
            self.total_steps,
            n_objects,
        ):
            raise ValueError(
                "Expected scene_object_active_mask shape (T, N), got "
                f"{self.scene_object_active_mask.shape}."
            )
        if tuple(self.scene_object_static.shape) != (n_objects,):
            raise ValueError(
                f"Expected scene_object_static shape {(n_objects,)}, got "
                f"{self.scene_object_static.shape}."
            )
        if tuple(self.scene_object_friction.shape) != (n_objects,):
            raise ValueError(
                f"Expected scene_object_friction shape {(n_objects,)}, got "
                f"{self.scene_object_friction.shape}."
            )
        if np.any(self.scene_object_friction <= 0.0):
            raise ValueError(
                f"Expected positive scene_object_friction, got {self.scene_object_friction}."
            )
        if tuple(self.scene_active_object_index.shape) != (self.total_steps,):
            raise ValueError(
                "Expected scene_active_object_index shape (T,), got "
                f"{self.scene_active_object_index.shape}."
            )
        if self.scene_canonical_object_id not in self.scene_object_ids:
            raise ValueError(
                "scene_canonical_object_id must refer to a scene object id; got "
                f"{self.scene_canonical_object_id!r} from {self.scene_object_ids}."
            )
        if tuple(self.scene_table_size.shape) != (2,):
            raise ValueError(
                f"Expected scene_table_size shape (2,), got {self.scene_table_size.shape}."
            )
        if np.any(self.scene_table_size <= 0.0):
            raise ValueError(
                f"Expected positive scene_table_size values, got {self.scene_table_size}."
            )
        n_blocks = len(self.scene_support_block_ids)
        if len(set(self.scene_support_block_ids)) != n_blocks:
            raise ValueError(
                f"Expected unique scene_support_block_ids, got {self.scene_support_block_ids}."
            )
        if tuple(self.scene_support_block_pos.shape) != (n_blocks, 3):
            raise ValueError(
                f"Expected scene_support_block_pos shape {(n_blocks, 3)}, "
                f"got {self.scene_support_block_pos.shape}."
            )
        if tuple(self.scene_support_block_size.shape) != (n_blocks, 3):
            raise ValueError(
                f"Expected scene_support_block_size shape {(n_blocks, 3)}, "
                f"got {self.scene_support_block_size.shape}."
            )
        if np.any(self.scene_support_block_size <= 0.0):
            raise ValueError(
                "Expected positive scene_support_block_size values, got "
                f"{self.scene_support_block_size}."
            )

    def _scene_object_jpos_for_verify(
        self,
        *,
        object_index: int,
        asset: ArtObject,
        frame_idx: int,
    ) -> torch.Tensor | None:
        if len(asset.joints) == 0:
            return None

        valid_indices = np.flatnonzero(self.scene_object_arti_valid_mask[object_index])
        exported_joint_names = list(self.scene_object_arti_names[object_index])
        if len(exported_joint_names) != len(valid_indices):
            raise ValueError(
                "Expected scene_object_arti_names length to match valid articulation "
                f"count for object index {object_index}, got "
                f"{len(exported_joint_names)} names and {len(valid_indices)} "
                "valid values."
            )
        missing = [
            joint.name
            for joint in asset.joints
            if joint.name not in exported_joint_names
        ]
        if missing:
            raise ValueError(
                "Scene object articulation is missing joints required by the "
                f"simulated object {asset.obj_cfg['name']!r}: {missing}."
            )
        joint_indices = [
            exported_joint_names.index(joint.name) for joint in asset.joints
        ]
        valid_values = self.scene_object_arti_vals[
            frame_idx, object_index, valid_indices
        ]
        return torch.as_tensor(
            valid_values[joint_indices],
            dtype=torch.float32,
            device=self.env.device,
        )

    def verify_target_object_motion(
        self,
        mode: str = "sim",
        *,
        frame_callback: Callable[[int, np.ndarray], None] | None = None,
    ) -> None:
        if mode == "real":
            return
        if not self.task_cfg.save_reference_video:
            return
        if mode != "sim":
            return

        reset_mgr = self.env.reset_manager
        physics_manager = self.env.physics_manager
        object_entities = physics_manager.object_entities
        env_idx = np.arange(self.env.num_envs)

        hand_jvals = {
            side: getattr(reset_mgr, f"frame_{side}_reset_jvals").to(self.env.device)
            for side in self.active_hand_sides
        }

        while True:
            physics_manager.start_recording()

            for frame_idx in range(self.total_steps):
                for object_index, object_id in enumerate(self.scene_object_ids):
                    asset = object_entities[object_id]
                    pos = torch.as_tensor(
                        self.scene_object_pos[frame_idx, object_index],
                        dtype=torch.float32,
                        device=self.env.device,
                    )
                    quat = torch.as_tensor(
                        self.scene_object_quat[frame_idx, object_index],
                        dtype=torch.float32,
                        device=self.env.device,
                    )
                    arti = self._scene_object_jpos_for_verify(
                        object_index=object_index,
                        asset=asset,
                        frame_idx=frame_idx,
                    )
                    state_kwargs = {
                        "pos": pos,
                        "quat": quat,
                        "jpos": arti,
                        "env_idx": env_idx,
                    }
                    if isinstance(asset, CardboardObject):
                        if object_index != self.canonical_object_index:
                            raise ValueError(
                                "Reference video only supports cardboard rest "
                                "states for the canonical scene object."
                            )
                        rest_values = reset_mgr.frame_cardboard_rest_arti
                        rest_names = reset_mgr.cardboard_rest_arti_names
                        if rest_values is None:
                            raise ValueError(
                                "Cardboard reference video requires recorded "
                                "cardboard_rest_arti_vals."
                            )
                        asset_joint_names = [joint.name for joint in asset.joints]
                        missing_rest_joints = [
                            name for name in asset_joint_names if name not in rest_names
                        ]
                        if missing_rest_joints:
                            raise ValueError(
                                "Cardboard reference video rest state is missing "
                                f"joints {missing_rest_joints}."
                            )
                        rest_indices = [
                            rest_names.index(name) for name in asset_joint_names
                        ]
                        state_kwargs["rest_jpos"] = (
                            rest_values[frame_idx, rest_indices]
                            .unsqueeze(0)
                            .repeat(self.env.num_envs, 1)
                        )
                    asset.set_state(**state_kwargs)

                for side in self.active_hand_sides:
                    robot: Hand = physics_manager.scene_entities[side]
                    hand_jpos = (
                        hand_jvals[side][frame_idx]
                        .unsqueeze(0)
                        .repeat(self.env.num_envs, 1)
                    )
                    robot.set_robot_state(jpos=hand_jpos, env_idx=env_idx)

                physics_manager.step()
                image = physics_manager.video_recording_hook(timestep=0)
                if frame_callback is not None:
                    if image is None:
                        raise ValueError("Reference frame callback requires an enabled recording camera.")
                    frame_callback(frame_idx, image)

            fname = f"{physics_manager.logs_dir}/reference_motion.mp4"
            physics_manager.stop_recording(save_to_filename=fname, fps=30)

            replay = "n"
            if replay.lower() != "y":
                break

    def _derive_active_object_index(self, active_mask: np.ndarray) -> np.ndarray:
        if active_mask.ndim != 2:
            raise ValueError(
                f"Expected scene_object_active_mask shape (T, N), got {active_mask.shape}."
            )
        active_index = np.full(active_mask.shape[0], -1, dtype=np.int64)
        for frame_idx, frame_mask in enumerate(active_mask):
            active = np.flatnonzero(frame_mask)
            if active.shape[0] == 1:
                active_index[frame_idx] = int(active[0])
            elif active.shape[0] > 1:
                raise ValueError(
                    "Expected at most one active scene object per frame, got "
                    f"{active.tolist()} at frame {frame_idx}."
                )
        return active_index

    def _object_list(self, value: Any) -> list[str]:
        arr = np.asarray(value, dtype=object)
        if arr.shape == ():
            item = arr.item()
            if isinstance(item, (list, tuple, np.ndarray)):
                return [str(v) for v in item]
            return [str(item)]
        return [str(v) for v in arr.tolist()]

    @property
    def canonical_object_index(self) -> int:
        return self.scene_object_ids.index(self.scene_canonical_object_id)

    def scene_object_entity_key(self, object_id: str) -> str:
        if object_id not in self.scene_object_ids:
            raise ValueError(
                f"Unknown scene object id {object_id!r}; expected one of "
                f"{self.scene_object_ids}."
            )
        return f"object:{object_id}"
