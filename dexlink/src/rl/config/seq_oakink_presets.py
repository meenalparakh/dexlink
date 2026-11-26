from math import asin, pi

from dexlink.src.rl.config.oakink_presets import ppo_preset_xhand

KEY_TASK_PROTOCOL_VERSION = "gate0_v0.5"
CONTROLLABILITY_PROTOCOL_VERSION = "v3_01_controllability_v0.3"
GATE4_PROTOCOL_VERSION = "gate4_v0.1"
BOTTLE_INTERACTION_PROTOCOL_VERSION = "gate4_bottle_interaction_v0.1"
PRECISION_MOUNTING_PROTOCOL_VERSION = "v3_08_precision_mounting_v0.1"

SEQ_BIMANUAL_REQUIRED_SETTINGS = {
    "agent.policy_inputs": (
        "left",
        "ref_left",
        "right",
        "ref_right",
        "contact",
        "command",
        "object",
    ),
    "agent.critic_inputs": (
        "left",
        "ref_left",
        "right",
        "ref_right",
        "contact",
        "command",
        "object",
    ),
}

SEQ_BIMANUAL_NO_REFERENCE_REQUIRED_SETTINGS = {
    "agent.policy_inputs": (
        "left",
        "right",
        "contact",
        "command",
        "object",
    ),
    "agent.critic_inputs": (
        "left",
        "right",
        "contact",
        "command",
        "object",
    ),
}

SEQ_MOTION_FOLLOWING_REQUIRED_SETTINGS = {
    # For these tasks, the desired motion is the meaningful objective; the
    # current task-success signal does not encode the intended behavior.
    "env.rewards_cfg.task_completion_weight": 0.0,
}

SEQ_ZERO_GRAVITY_REQUIRED_SETTINGS = {
    "env.sim_cfg.gravity": 0.0,
}

KEY_TASK_PROTOCOL_SETTINGS = {
    "env.key_task_protocol_version": KEY_TASK_PROTOCOL_VERSION,
    "env.sim_cfg.num_envs": 1024,
    "agent.timesteps": 50_000,
    "agent.eval_freq": 0,
    "agent.eval_metrics_freq": 0,
    "agent.eval_video_freq": 0,
}

NOTEBOOK_TASK_METRIC_SETTINGS = {
    "env.task_cfg.metric_cfg.metric_id": "open_notebook",
    "env.task_cfg.metric_cfg.notebook_object_id": "notebook",
    "env.task_cfg.metric_cfg.notebook_angle_tolerance_rad": 15.0 * pi / 180.0,
}

STACKING_TASK_METRIC_SETTINGS = {
    "env.task_cfg.metric_cfg.metric_id": "block_stacking",
    "env.task_cfg.metric_cfg.stacking_object_ids": ("red", "green", "blue"),
    # Each upper center must be at least half of the supporting cube's side
    # length above the lower center: red=6 cm, green=4.5 cm.
    "env.task_cfg.metric_cfg.stacking_min_center_height_offsets_m": (0.03, 0.0225),
}

BANANA_HANDOVER_TASK_METRIC_SETTINGS = {
    "env.task_cfg.metric_cfg.metric_id": "banana_handover",
    "env.task_cfg.metric_cfg.banana_object_id": "banana",
    "env.task_cfg.metric_cfg.banana_source_hand": "right",
    "env.task_cfg.metric_cfg.banana_receiving_hand": "left",
    "env.task_cfg.metric_cfg.banana_min_center_height_above_table_m": 0.05,
    "env.task_cfg.metric_cfg.banana_min_receiving_support_weight_fraction": 0.5,
    "env.task_cfg.metric_cfg.banana_max_source_contact_weight_fraction": 0.1,
}

MUG_GRASP_CHANGE_TASK_METRIC_SETTINGS = {
    # The two-grasp sequence criterion is intentionally evaluation-only. During
    # training, best-checkpoint selection uses the hand-reference tracking score.
    "env.task_cfg.metric_cfg.eval_metric_id": "mug_grasp_change",
    "env.task_cfg.metric_cfg.mug_object_id": "mug",
    "env.task_cfg.metric_cfg.mug_hand_side": "right",
    "env.task_cfg.metric_cfg.mug_reference_grasp_phase_names": (
        "top_grasp",
        "side_grasp",
    ),
    "env.task_cfg.metric_cfg.mug_min_center_height_above_table_m": 0.05,
}

HAMMER_TASK_METRIC_SETTINGS = {
    "env.task_cfg.metric_cfg.metric_id": "hammer_peg_impact",
    "env.task_cfg.metric_cfg.hammer_object_id": "hammer",
    "env.task_cfg.metric_cfg.hammer_peg_object_id": "peg",
    "env.task_cfg.metric_cfg.hammer_peg_sensor_top_offset_m": 0.01,
    "env.task_cfg.metric_cfg.hammer_contact_force_threshold": 1.0e-4,
}

HAMMER_TIMING_TASK_METRIC_SETTINGS = {
    **HAMMER_TASK_METRIC_SETTINGS,
    "env.task_cfg.metric_cfg.metric_id": "hammer_two_swing",
    "env.task_cfg.metric_cfg.hammer_first_strike_phase_names": (
        "strike",
        "hold_impact",
    ),
    "env.task_cfg.metric_cfg.hammer_second_strike_phase_names": (
        "strike_again",
        "hold_second_impact",
    ),
}

HAMMER_SINGLE_SWING_TASK_METRIC_SETTINGS = {
    **HAMMER_TASK_METRIC_SETTINGS,
    "env.task_cfg.metric_cfg.metric_id": "hammer_single_swing",
    "env.task_cfg.metric_cfg.hammer_strike_phase_names": (
        "strike",
        "hold_impact",
    ),
    # The reference itself ends with five post-impact frames. No additional
    # global hold is needed before the episode may finish.
    "env.rewards_cfg.last_reference_hold_steps": 0,
}

CUP_POSE_TASK_METRIC_SETTINGS = {
    "env.task_cfg.metric_cfg.metric_id": "object_pose",
    "env.task_cfg.metric_cfg.object_pose_object_id": "cup",
    "env.task_cfg.metric_cfg.object_pose_position_tolerance_m": 0.03,
    "env.task_cfg.metric_cfg.object_pose_orientation_tolerance_rad": 15.0 * pi / 180.0,
}

CUBE_PICK_STRATEGY_TASK_METRIC_SETTINGS = {
    "env.task_cfg.metric_cfg.metric_id": "cube_strategy",
    "env.task_cfg.metric_cfg.cube_strategy_object_id": "red",
    "env.task_cfg.metric_cfg.cube_strategy_expected_lift": True,
    "env.task_cfg.metric_cfg.cube_strategy_min_lift_height_change_m": 0.02,
    "env.task_cfg.metric_cfg.cube_strategy_position_tolerance_m": 0.03,
    "env.task_cfg.metric_cfg.cube_strategy_orientation_tolerance_rad": 30.0
    * pi
    / 180.0,
}

CUBE_PUSH_STRATEGY_TASK_METRIC_SETTINGS = {
    **CUBE_PICK_STRATEGY_TASK_METRIC_SETTINGS,
    "env.task_cfg.metric_cfg.cube_strategy_expected_lift": False,
}

DISPENSER_PRESSED_TASK_METRIC_SETTINGS = {
    "env.task_cfg.metric_cfg.metric_id": "dispenser_pressed",
    "env.task_cfg.metric_cfg.dispenser_object_id": "dispenser",
    "env.task_cfg.metric_cfg.dispenser_joint_name": "neck_to_head_prismatic",
    # Count a press once the head is within 2 mm of its reference fully-pressed
    # position. The episode accumulator preserves the event after spring return.
    "env.task_cfg.metric_cfg.dispenser_pressed_tolerance_m": 0.002,
}

DRAWER_TASK_METRIC_SETTINGS = {
    "env.task_cfg.metric_cfg.metric_id": "drawer_articulation",
    "env.task_cfg.metric_cfg.drawer_object_id": "drawer",
    "env.task_cfg.metric_cfg.drawer_joint_name": "drawer_slide",
    "env.task_cfg.metric_cfg.drawer_final_articulation_tolerance_m": 0.01,
}

ERASER_TASK_METRIC_SETTINGS = {
    "env.task_cfg.metric_cfg.metric_id": "eraser_tracking",
    "env.task_cfg.metric_cfg.eraser_object_id": "eraser",
    "env.task_cfg.metric_cfg.eraser_phase_names": tuple(
        f"erase_{phase_index}" for phase_index in range(1, 8)
    ),
    "env.task_cfg.metric_cfg.eraser_max_position_error_m": 0.02,
}

MOUNTING_CYLINDER_HEIGHT_M = 0.09
MOUNTING_MAX_ORIGIN_HEIGHT_ABOVE_POST_M = 0.018
MOUNTING_CLEARANCE_BY_EXPERIMENT_ID = {
    "v3_08_mounting_03": 0.003,
    "v3_08_mounting_05": 0.005,
    "v3_08_mounting_07": 0.007,
}


def _precision_mounting_task_metric_settings(
    clearance_m: float,
) -> dict[str, object]:
    if not 0.0 < clearance_m < MOUNTING_CYLINDER_HEIGHT_M / 2.0:
        raise ValueError(
            "Expected mounting clearance in (0, cylinder_height/2), got "
            f"{clearance_m}."
        )
    # If both ends of the 9 cm bore stay within the radial clearance, their
    # maximum possible separation is twice that clearance. This is the largest
    # physically admissible lean rather than an arbitrary pose tolerance.
    max_axis_tilt_rad = asin(2.0 * clearance_m / MOUNTING_CYLINDER_HEIGHT_M)
    return {
        "env.task_cfg.metric_cfg.metric_id": "precision_mounting",
        "env.task_cfg.metric_cfg.mounting_cylinder_object_id": "cylinder",
        "env.task_cfg.metric_cfg.mounting_post_object_id": "post",
        "env.task_cfg.metric_cfg.mounting_cylinder_height_m": (
            MOUNTING_CYLINDER_HEIGHT_M
        ),
        "env.task_cfg.metric_cfg.mounting_bore_clearance_m": clearance_m,
        "env.task_cfg.metric_cfg.mounting_max_origin_height_above_post_m": (
            MOUNTING_MAX_ORIGIN_HEIGHT_ABOVE_POST_M
        ),
        "env.task_cfg.metric_cfg.mounting_max_axis_tilt_rad": max_axis_tilt_rad,
    }


def _object_lift_task_metric_settings(object_id: str) -> dict[str, object]:
    if not object_id:
        raise ValueError("Expected a non-empty object_id for the lift metric.")
    return {
        "env.task_cfg.metric_cfg.metric_id": "object_lift",
        "env.task_cfg.metric_cfg.object_lift_object_id": object_id,
    }


def _bottle_interaction_task_metric_settings() -> dict[str, object]:
    return {
        "env.task_cfg.metric_cfg.metric_id": "bottle_interaction",
        "env.task_cfg.metric_cfg.bottle_interaction_object_id": "bottle",
        "env.task_cfg.metric_cfg.bottle_interaction_hand_side": "right",
        "env.task_cfg.metric_cfg.bottle_relative_wrist_position_tolerance_m": 0.05,
        "env.task_cfg.metric_cfg.bottle_relative_wrist_orientation_tolerance_rad": (
            30.0 * pi / 180.0
        ),
    }


BOWL_CENTER_HEIGHT_OFFSET_M = 0.05711411


def _bowl_stack_task_metric_settings(bowl_count: int) -> dict[str, object]:
    if bowl_count < 2:
        raise ValueError(f"Expected bowl_count >= 2, got {bowl_count}.")
    return {
        "env.task_cfg.metric_cfg.metric_id": "block_stacking",
        "env.task_cfg.metric_cfg.stacking_object_ids": tuple(
            f"bowl_{index}" for index in range(1, bowl_count + 1)
        ),
        "env.task_cfg.metric_cfg.stacking_min_center_height_offsets_m": (
            (BOWL_CENTER_HEIGHT_OFFSET_M,) * (bowl_count - 1)
        ),
    }


CONTROLLABILITY_PROTOCOL_SETTINGS = {
    **KEY_TASK_PROTOCOL_SETTINGS,
    "env.key_task_protocol_version": CONTROLLABILITY_PROTOCOL_VERSION,
}

GATE4_PROTOCOL_SETTINGS = {
    **KEY_TASK_PROTOCOL_SETTINGS,
    "env.key_task_protocol_version": GATE4_PROTOCOL_VERSION,
}

BOTTLE_INTERACTION_PROTOCOL_SETTINGS = {
    **GATE4_PROTOCOL_SETTINGS,
    "env.key_task_protocol_version": BOTTLE_INTERACTION_PROTOCOL_VERSION,
}

PRECISION_MOUNTING_PROTOCOL_SETTINGS = {
    **KEY_TASK_PROTOCOL_SETTINGS,
    "env.key_task_protocol_version": PRECISION_MOUNTING_PROTOCOL_VERSION,
}

V3_03_REUSABILITY_EXPERIMENT_IDS = {
    f"v3_03_reuse_{template}_scale_{scale}"
    for template in (
        "mug_top",
        "mug_side",
        "mug_handle",
        "bottle_top",
        "bottle_side",
    )
    for scale in ("070", "085", "100", "125")
}

BOTTLE_INTERACTION_EXPERIMENT_IDS = {
    "v3_03_reuse_bottle_side_scale_100",
    "v3_03_reuse_bottle_top_scale_100",
}

V3_04_COMPOSITION_EXPERIMENT_IDS = {
    f"v3_04_stack_{bowl_count}_bowls_normal" for bowl_count in (2, 3, 5)
}

V3_05_TEMPLATE_TRANSFER_EXPERIMENT_IDS = {
    "v3_05_mug_top_transferred",
    "v3_05_mug_top_exact_target",
    "v3_05_mug_handle_transferred",
    "v3_05_mug_handle_exact_target",
    "v3_05_bottle_top_transferred",
    "v3_05_bottle_top_exact_target",
    "v3_05_bottle_side_a15015_transferred",
    "v3_05_bottle_side_a15015_exact_target",
    "v3_05_apple_to_camera_top_transferred",
    "v3_05_apple_to_camera_top_exact_target",
    "v3_05_toothbrush_to_fryingpan_handle_transferred",
    "v3_05_toothbrush_to_fryingpan_handle_exact_target",
    "v3_05_trigger_sprayer_spraying_a01027_transferred",
    "v3_05_trigger_sprayer_spraying_a01027_exact_target",
    "v3_05_hammer_to_wineglass_stem_transferred",
    "v3_05_hammer_to_wineglass_stem_exact_target",
    "v3_05_apple_to_mouse_top_transferred",
    "v3_05_apple_to_mouse_top_exact_target",
}

GATE4_EXPERIMENT_IDS = (
    V3_03_REUSABILITY_EXPERIMENT_IDS
    | V3_04_COMPOSITION_EXPERIMENT_IDS
    | V3_05_TEMPLATE_TRANSFER_EXPERIMENT_IDS
)

SEQ_KEY_TASK_SETTINGS_BY_EXPERIMENT_ID = {
    "v2_03_open_notebook": NOTEBOOK_TASK_METRIC_SETTINGS,
    "v2_03_stack_rgb_blocks_different_sizes": STACKING_TASK_METRIC_SETTINGS,
    "v2_03_banana_handover": BANANA_HANDOVER_TASK_METRIC_SETTINGS,
    "v2_03_mug_grasp_change_sequence": MUG_GRASP_CHANGE_TASK_METRIC_SETTINGS,
    "v2_03_hammer_strike_peg": HAMMER_TASK_METRIC_SETTINGS,
    "v2_03_dispenser_press_twice": DISPENSER_PRESSED_TASK_METRIC_SETTINGS,
    "v2_03_erase_board": ERASER_TASK_METRIC_SETTINGS,
    "v2_03_invert_cup": CUP_POSE_TASK_METRIC_SETTINGS,
    "v2_03_open_drawer": DRAWER_TASK_METRIC_SETTINGS,
    "v3_01_hammer_timing_slow": HAMMER_TIMING_TASK_METRIC_SETTINGS,
    "v3_01_hammer_timing_medium": HAMMER_TIMING_TASK_METRIC_SETTINGS,
    "v3_01_hammer_timing_fast": HAMMER_TIMING_TASK_METRIC_SETTINGS,
    "v3_01_cup_pose_tilt_045deg": CUP_POSE_TASK_METRIC_SETTINGS,
    "v3_01_cup_pose_tilt_090deg": CUP_POSE_TASK_METRIC_SETTINGS,
    "v3_01_cup_pose_tilt_120deg": CUP_POSE_TASK_METRIC_SETTINGS,
    "v3_01_cube_strategy_pick": CUBE_PICK_STRATEGY_TASK_METRIC_SETTINGS,
    "v3_01_cube_strategy_push": CUBE_PUSH_STRATEGY_TASK_METRIC_SETTINGS,
    "v3_controllability_hammer2_slow": HAMMER_SINGLE_SWING_TASK_METRIC_SETTINGS,
    "v3_controllability_hammer2_fast": HAMMER_SINGLE_SWING_TASK_METRIC_SETTINGS,
    **{
        experiment_id: _precision_mounting_task_metric_settings(clearance_m)
        for experiment_id, clearance_m in (MOUNTING_CLEARANCE_BY_EXPERIMENT_ID.items())
    },
    **{
        experiment_id: (
            _bottle_interaction_task_metric_settings()
            if experiment_id in BOTTLE_INTERACTION_EXPERIMENT_IDS
            else _object_lift_task_metric_settings(
                "mug" if "_mug_" in experiment_id else "bottle"
            )
        )
        for experiment_id in V3_03_REUSABILITY_EXPERIMENT_IDS
    },
    **{
        experiment_id: _object_lift_task_metric_settings("template_transfer")
        for experiment_id in V3_05_TEMPLATE_TRANSFER_EXPERIMENT_IDS
    },
    **{
        f"v3_04_stack_{bowl_count}_bowls_normal": (
            _bowl_stack_task_metric_settings(bowl_count)
        )
        for bowl_count in (2, 3, 5)
    },
}

CONTROLLABILITY_EXPERIMENT_IDS = {
    experiment_id
    for experiment_id in SEQ_KEY_TASK_SETTINGS_BY_EXPERIMENT_ID
    if experiment_id.startswith("v3_01_")
}

SEQ_REQUIRED_SETTINGS_BY_FLAG = {
    "bimanual": SEQ_BIMANUAL_REQUIRED_SETTINGS,
    "motion_following": SEQ_MOTION_FOLLOWING_REQUIRED_SETTINGS,
    "zero_gravity": SEQ_ZERO_GRAVITY_REQUIRED_SETTINGS,
}

SEQ_REQUIRED_TASK_FLAGS_BY_EXPERIMENT_ID = {
    "v2_03_banana_handover": ("bimanual",),
    "v2_03_banana_handover_template_stride_3": ("bimanual",),
    "v2_03_banana_handover_template_stride_6": ("bimanual",),
    "v2_03_banana_handover_template_stride_10": ("bimanual",),
    "v2_03_banana_handover_template_start_end": ("bimanual",),
    "v2_03_lid_on_container_from_table": ("bimanual",),
    "v3_08_lid_on_container": ("bimanual",),
    "v2_03_lid_on_container_in_hand": ("bimanual",),
    "v2_14_throw_catch_cube": ("bimanual",),
    "v2_16_fold_carton_dual_point_gripper": ("bimanual", "motion_following"),
    "v2_16_fold_mailer_dual_point_gripper": ("bimanual", "motion_following"),
    "v2_03_dispenser_press_twice": ("motion_following",),
    "v2_03_erase_board": ("motion_following",),
    "v2_03_hammer_strike_peg": ("motion_following",),
    "v3_01_hammer_timing_slow": ("motion_following",),
    "v3_01_hammer_timing_medium": ("motion_following",),
    "v3_01_hammer_timing_fast": ("motion_following",),
    "v3_controllability_hammer2_slow": ("motion_following",),
    "v3_controllability_hammer2_fast": ("motion_following",),
    "v2_03_lift_toothbrush_flat": ("zero_gravity",),
}

SEQ_NO_REFERENCE_INPUT_PRESETS = set()


def seq_experiment_id_from_exported_path(exported_path: str) -> str | None:
    path_parts = exported_path.split("/")
    pipeline_markers = (
        "v2_03_pipeline",
        "v2_10_pipeline",
        "v2_11_pipeline",
        "v2_14_pipeline",
        "v2_16_box_folding_point_gripper",
        "v3_01_controllability_pipeline",
        "v3_03_reusability_pipeline",
        "v3_04_compositionality_pipeline",
        "v3_05_template_transfer_pipeline",
        "v3_07_controllability_hammer2_pipeline",
        "v3_08_precision_mounting_pipeline",
        "v3_08_lid_on_container_pipeline",
    )
    pipeline_idx = None
    for marker in pipeline_markers:
        if marker in path_parts:
            pipeline_idx = path_parts.index(marker)
            break
    if pipeline_idx is None:
        return None
    if pipeline_idx + 1 >= len(path_parts):
        raise ValueError(
            "Expected exported sequence-pipeline path to include an experiment id after "
            f"one of {pipeline_markers}, got {exported_path!r}."
        )
    return path_parts[pipeline_idx + 1]


def required_seq_task_settings(
    exported_path: str, preset_name: str | None = None
) -> dict[str, object]:
    experiment_id = seq_experiment_id_from_exported_path(exported_path)
    if experiment_id is None:
        return {}
    settings: dict[str, object] = {}
    if experiment_id in SEQ_KEY_TASK_SETTINGS_BY_EXPERIMENT_ID:
        if experiment_id in CONTROLLABILITY_EXPERIMENT_IDS:
            protocol_settings = CONTROLLABILITY_PROTOCOL_SETTINGS
        elif experiment_id in BOTTLE_INTERACTION_EXPERIMENT_IDS:
            protocol_settings = BOTTLE_INTERACTION_PROTOCOL_SETTINGS
        elif experiment_id in GATE4_EXPERIMENT_IDS:
            protocol_settings = GATE4_PROTOCOL_SETTINGS
        elif experiment_id in MOUNTING_CLEARANCE_BY_EXPERIMENT_ID:
            protocol_settings = PRECISION_MOUNTING_PROTOCOL_SETTINGS
        else:
            protocol_settings = KEY_TASK_PROTOCOL_SETTINGS
        settings.update(protocol_settings)
        task_metric_settings = SEQ_KEY_TASK_SETTINGS_BY_EXPERIMENT_ID[experiment_id]
        if task_metric_settings:
            settings.update(task_metric_settings)
    for flag in SEQ_REQUIRED_TASK_FLAGS_BY_EXPERIMENT_ID.get(experiment_id, ()):
        if flag == "bimanual" and preset_name in SEQ_NO_REFERENCE_INPUT_PRESETS:
            settings.update(SEQ_BIMANUAL_NO_REFERENCE_REQUIRED_SETTINGS)
        else:
            settings.update(SEQ_REQUIRED_SETTINGS_BY_FLAG[flag])
    return settings


SEQ_DEFAULT_PRESET_SIM = ppo_preset_xhand.copy()
SEQ_DEFAULT_PRESET_SIM.update(
    {
        "preset": "seq_oakink_default",
        "logger.project": "condext_v6",
        "logger.group": "seq_oakink",
        "logger.runname": "seq_oakink_default",
        "env.env_class": "multi_object",
        "env.backend": "sim",
        # Use the exported OakInk dispatcher so sequence payloads instantiate
        # OakinkProcessorSequence based on demo_kind="sequence".
        "env.task_cfg.oakink_exported_path": "logs/trajectory_templates/01_sparse_bowl_s01_t01_lift/reach_grasp_lift_oakink_sequence_exported.npy",
        "env.task_cfg.demo_dataset": "oakink",
        "env.task_cfg.start_timestep": 0,
        "env.task_cfg.end_timestep": -1,
        "env.task_cfg.skip": 1,
        "env.task_cfg.object_only_scene": False,
        "env.task_cfg.object_fixed": False,
        "env.sim_cfg.video_save_freq": 5_000,
        "env.action_cfg.action_mode": "diff",
        "env.action_cfg.arm_translation_delta_scale": 0.025,
        "env.action_cfg.arm_rotation_delta_scale": 0.45,
        "env.action_cfg.hand_joint_delta_scale": 0.45,
        "env.commands_cfg.use_gt_pos": True,
        "env.commands_cfg.use_gt_rot": True,
        "env.commands_cfg.use_gt_arti": True,
        "env.commands_cfg.use_pcd": False,
        "env.commands_cfg.use_bps": False,
        "env.commands_cfg.use_rel_bps": True,
        "env.commands_cfg.use_retargeted_motion": True,
        "env.obs_cfg.use_hand_obs": True,
        "env.obs_cfg.use_contact": True,
        "env.obs_cfg.use_object_obs": True,
        "env.obs_cfg.use_object_vel_obs": False,
        "env.obs_cfg.use_policy_object_pcd": False,
        "env.obs_cfg.use_policy_hand_pcd": False,
        "env.obs_cfg.use_privileged_object_pcd": True,
        "env.obs_cfg.task_command": True,
        "env.obs_cfg.use_command_timestep": True,
        "env.obs_cfg.embed_commands_from_timestep": True,
        "agent.policy_inputs": (
            "right",
            "ref_right",
            "contact",
            "command",
            "object",
        ),
        "agent.critic_inputs": (
            "right",
            "ref_right",
            "contact",
            "command",
            "object",
        ),
        "env.reference_cfg.enabled": False,
        "env.reset_cfg.reverse_curriculum": False,
        "env.reset_cfg.start_phase_probs": None,
        "env.rewards_cfg.use_retargeted_motion": True,
        "env.rewards_cfg.pcd_distance_metric": "bps",
        "env.rewards_cfg.maniptrans_reward_weight": 0.0,
        "env.rewards_cfg.discriminator_weight": 0.0,
        "env.rewards_cfg.term_schedules.object_tracking.scheme": "constant",
        "env.rewards_cfg.term_schedules.object_tracking.value": 1.0,
        # "env.rewards_cfg.use_retargeting_motion": False,
        # "env.commands_cfg.use_retargeting_motion": False,
        "env.rewards_cfg.early_terminate": True,
        "env.rewards_cfg.last_reference_hold_steps": 40,
        "env.rewards_cfg.rest_task_pos_scale": 2.0,
        "env.rewards_cfg.rest_task_rot_scale": 1.0,
        "env.rewards_cfg.task_arti_scale": 20.0,
        "env.rewards_cfg.rest_task_arti_scale": 10.0,
        "agent.log_essential_metrics_only": True,
        "env.rewards_cfg.log_metrics": False,
        "env.rewards_cfg.log_essential_metrics_only": True,
        "agent.timesteps": 100_000,
        "logger.desired_model_saves": 50,
    }
)

SEQ_RESIDUAL_TARGET_TRAJECTORY_WEIGHTED_POLICY_V3_PRESET_SIM = (
    SEQ_DEFAULT_PRESET_SIM.copy()
)
SEQ_RESIDUAL_TARGET_TRAJECTORY_WEIGHTED_POLICY_V3_PRESET_SIM.update(
    {
        "preset": "seq_oakink_residual_target_trajectory_weighted_policy_v3",
        "logger.group": "seq_oakink_residual_target_trajectory_weighted_policy_v3",
        "logger.runname": "seq_oakink_residual_target_trajectory_weighted_policy_v3",
        "env.action_cfg.action_mode": "absolute_target",
        "env.action_cfg.policy_variant": "trajectory_weighted_residual_target_v3",
        "agent.policy_embed_dim": 96,
        "agent.hand_synergy_dim": 96,
        "agent.obs_proc_embed_dim": 64,
        "agent.value_embed_dim": 96,
    }
)

SEQ_RESIDUAL_TARGET_TRAJECTORY_WEIGHTED_POLICY_V3_PCD_PRESET_SIM = (
    SEQ_RESIDUAL_TARGET_TRAJECTORY_WEIGHTED_POLICY_V3_PRESET_SIM.copy()
)
SEQ_RESIDUAL_TARGET_TRAJECTORY_WEIGHTED_POLICY_V3_PCD_PRESET_SIM.update(
    {
        "preset": "seq_oakink_residual_target_trajectory_weighted_policy_v3_pcd",
        "logger.group": "seq_oakink_residual_target_trajectory_weighted_policy_v3_pcd",
        "logger.runname": "seq_oakink_residual_target_trajectory_weighted_policy_v3_pcd",
        "env.task_cfg.hand_name": "xhand",
        "env.obs_cfg.use_policy_object_pcd": True,
        "env.obs_cfg.use_policy_hand_pcd": True,
        "agent.policy_inputs": (
            "right",
            "ref_right",
            "contact",
            "command",
            "object_pcd",
            "hand_pcd",
        ),
    }
)

SEQ_DUAL_POINT_GRIPPER_PRESET_SIM = SEQ_DEFAULT_PRESET_SIM.copy()
SEQ_DUAL_POINT_GRIPPER_PRESET_SIM.update(
    {
        "preset": "seq_oakink_dual_point_gripper",
        "logger.group": "seq_oakink_dual_point_gripper",
        "logger.runname": "seq_oakink_dual_point_gripper",
        "env.task_cfg.hand_name": "point_gripper",
        "env.sim_cfg.num_envs": 512,
        # Root commands move in centimeters/radians; jaw commands move in meters.
        "env.action_cfg.arm_translation_delta_scale": 0.015,
        "env.action_cfg.arm_rotation_delta_scale": 0.25,
        "env.action_cfg.hand_joint_delta_scale": 0.002,
        "agent.policy_inputs": (
            "left",
            "ref_left",
            "right",
            "ref_right",
            "contact",
            "command",
            "object",
        ),
        "agent.critic_inputs": (
            "left",
            "ref_left",
            "right",
            "ref_right",
            "contact",
            "command",
            "object",
        ),
    }
)

SEQ_SIM_EVAL_PRESET_OVERRIDES = {
    "preset": "seq_oakink_eval_sim",
    "env.backend": "sim",
    "env.sim_cfg.num_envs": 1,
    "env.sim_cfg.n_rendered_envs": 1,
    "env.sim_cfg.show_viewer": True,
    "env.sim_cfg.max_FPS": 60,
    "env.sim_cfg.apply_random_force": False,
    "env.visualize_policy_pcd": False,
    "env.visualize_policy_pcd_on": "step",
    "agent.resume": True,
    "agent.eval_freq": 5000,
    "agent.eval_metrics_freq": -1,
    "agent.eval_video_freq": -1,
    "logger.group": "seq_sim_eval",
    "logger.runname": "seq_eval_sim",
    "logger.disable": True,
    "logger.timestamp": True,
}

SEQ_OAKINK_PRESETS = {
    "seq_oakink_default": SEQ_DEFAULT_PRESET_SIM,
    "seq_oakink_residual_target_trajectory_weighted_policy_v3": (
        SEQ_RESIDUAL_TARGET_TRAJECTORY_WEIGHTED_POLICY_V3_PRESET_SIM
    ),
    "seq_oakink_residual_target_trajectory_weighted_policy_v3_pcd": (
        SEQ_RESIDUAL_TARGET_TRAJECTORY_WEIGHTED_POLICY_V3_PCD_PRESET_SIM
    ),
    "seq_oakink_dual_point_gripper": SEQ_DUAL_POINT_GRIPPER_PRESET_SIM,
    "seq_oakink_eval_sim": SEQ_SIM_EVAL_PRESET_OVERRIDES,
}
