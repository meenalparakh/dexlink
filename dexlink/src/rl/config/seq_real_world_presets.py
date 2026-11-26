from __future__ import annotations

from dexlink.src.rl.config.seq_oakink_presets import (
    SEQ_RESIDUAL_TARGET_TRAJECTORY_WEIGHTED_POLICY_V3_PCD_PRESET_SIM,
)

SEQ_REAL_WORLD_BASE_PRESET = (
    SEQ_RESIDUAL_TARGET_TRAJECTORY_WEIGHTED_POLICY_V3_PCD_PRESET_SIM.copy()
)
SEQ_REAL_WORLD_BASE_PRESET.update(
    {
        "preset": "seq_oakink_real_base",
        "logger.project": "real_evals",
        "logger.group": "seq_oakink_real",
        "logger.runname": "seq_oakink_real",
        "env.env_class": "multi_object",
        "env.backend": "real",
        "env.task_cfg.hand_name": "xhand",
        "env.real_cfg.real_control_hand_name": "franka_xhand",
        "env.real_cfg.initialize_robot": True,
        "env.real_cfg.initialize_camera": True,
        "env.real_cfg.read_robot_state_on_initialize": True,
        "env.real_cfg.start_camera_polling_on_initialize": True,
        "env.real_cfg.visualization_only": False,
        "env.real_cfg.bypass_low_level_control": False,
        "env.real_cfg.reset_to_default_on_initialize": True,
        "env.real_cfg.reset_motion_franka_tol_rad": 0.5,
        "env.real_cfg.reset_motion_xhand_tol_rad": 3.14,
        "env.real_cfg.reset_motion_max_steps": 1000,
        "env.real_cfg.reset_motion_franka_max_step_rad": 0.2,
        "env.real_cfg.reset_motion_xhand_steps": 2,
        "env.real_cfg.control_target_max_joint_step_rad": 0.714,
        "env.real_cfg.control_target_progress_per_update": 1.0,
        "env.real_cfg.phase_cached_object_pcd_motion_start_phase": "grasp",
        "env.real_cfg.run_world_frame_calibration": False,
        "env.real_cfg.reuse_cached_world_calibration": True,
        "env.real_cfg.world_calibration_cache_path": (
            "real_deployment/deoxys_xhand/calib_camera_data/"
            "seq_world_frame_calibration.npz"
        ),
        "env.real_cfg.wrist_to_marker_pos": (0.0, -0.023, -0.055),
        "env.real_cfg.wrist_to_marker_rotmat": (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0),
        ),
        "env.real_cfg.world_x_min": 0.25,
        "env.real_cfg.world_z_min": 0.07,
        "env.real_cfg.table_plane_height_margin_m": 0.005,
        "env.real_cfg.table_plane_sample_points": 200,
        "env.real_cfg.table_plane_synthetic_half_extent_m": 0.30,
        "env.real_cfg.table_plane_synthetic_depth_m": 0.005,
        "env.real_cfg.robot_pcd_exclusion_radius_m": 0.03,
        "env.real_cfg.largest_cluster_eps_m": 0.03,
        "env.real_cfg.largest_cluster_min_points": 20,
        "env.real_cfg.visualize_runtime_full_arm_mesh": False,
        "env.sim_cfg.num_envs": 1,
        "env.obs_cfg.use_policy_object_pcd": True,
        "env.obs_cfg.use_policy_hand_pcd": True,
        "env.obs_cfg.use_privileged_object_pcd": False,
        "env.obs_cfg.use_object_vel_obs": False,
        "agent.resume": True,
        "logger.disable": False,
    }
)


SEQ_REAL_WORLD_CALIBRATION_PRESET = SEQ_REAL_WORLD_BASE_PRESET.copy()
SEQ_REAL_WORLD_CALIBRATION_PRESET.update(
    {
        "preset": "seq_oakink_real_calibration",
        "logger.group": "seq_oakink_real_calibration",
        "logger.runname": "seq_oakink_real_calibration",
        "env.real_cfg.run_world_frame_calibration": True,
        "env.real_cfg.reuse_cached_world_calibration": False,
        "env.real_cfg.bypass_low_level_control": False,
        "env.real_cfg.reset_to_default_on_initialize": False,
        "env.real_cfg.visualize_runtime_debug": True,
        "env.real_cfg.pause_on_visualization_update": True,
    }
)


SEQ_REAL_WORLD_EVAL_PRESET = SEQ_REAL_WORLD_BASE_PRESET.copy()
SEQ_REAL_WORLD_EVAL_PRESET.update(
    {
        "preset": "seq_oakink_eval_real",
        "logger.group": "seq_oakink_eval_real",
        "logger.runname": "seq_oakink_eval_real",
        "real_eval.rollout_length": 200,
        "real_eval.logdir": "real_eval",
        "real_eval.seed": 42,
        # "real_eval.command_substeps": 4,
        "real_eval.command_substeps": 1,
        "real_eval.initialization_only": False,
        "real_eval.single_step_only": False,
    }
)


# Scene deployment inherits the trained actor/schema instead of a training preset's
# input terms, architecture or point budget. Calibration is still operator supplied.
SEQ_REAL_WORLD_SCENE_CLOUD_PRESET = {
    key: value
    for key, value in SEQ_REAL_WORLD_EVAL_PRESET.items()
    if key.startswith(("env.real_cfg.", "logger.", "real_eval."))
}
SEQ_REAL_WORLD_SCENE_CLOUD_PRESET.update(
    {
        "preset": "seq_oakink_eval_real_scene_cloud_v1",
        "env.env_class": "multi_object",
        "env.backend": "real",
        "env.obs_cfg.policy_object_pcd_mode": "scene_cloud_v1",
        "env.obs_cfg.use_privileged_object_pcd": False,
        "env.real_cfg.use_phase_cached_object_pcd": False,
        "env.real_cfg.table_plane_sample_points": 0,
    }
)


SEQ_REAL_WORLD_PRESETS = {
    "seq_oakink_real_base": SEQ_REAL_WORLD_BASE_PRESET,
    "seq_oakink_real_calibration": SEQ_REAL_WORLD_CALIBRATION_PRESET,
    "seq_oakink_eval_real": SEQ_REAL_WORLD_EVAL_PRESET,
    "seq_oakink_eval_real_scene_cloud_v1": SEQ_REAL_WORLD_SCENE_CLOUD_PRESET,
}
