from math import pi

from dexlink.utils.retargeting.path_utils import (
    get_external_data_dir,
    get_logs_dir,
)

"""
OakInk-specific presets.
"""

OAKINK_DEFAULT_PRESET = {
    "oakink_default": {
        # ExperimentConfig
        "preset": "oakink_default",
        # EnvConfig
        "env.device": "cuda",
        "env.seed": 42,
        "env.decimation": 8,
        "env.backend": "sim",
        "env.real_cfg.wris_pos_alpha": 1.0,
        "env.real_cfg.wrist_rot_alpha": 1.0,
        "env.real_cfg.fingers_pos_alpha": 1.0,
        "env.visualize_policy_pcd": False,
        "env.visualize_policy_pcd_on": "reset",
        "env.visualize_policy_pcd_point_size": 0.005,
        # ObsCfg
        "env.obs_cfg.use_hand_obs": True,
        "env.obs_cfg.hand_cfg.use_jpos": True,
        "env.obs_cfg.hand_cfg.use_jvel": False,
        "env.obs_cfg.hand_cfg.use_wrist_pose": True,
        "env.obs_cfg.hand_cfg.use_jacc_privileged": False,
        "env.obs_cfg.hand_cfg.use_lpos": True,
        "env.obs_cfg.hand_cfg.use_lvel": False,
        "env.obs_cfg.use_contact": True,
        "env.obs_cfg.use_policy_object_pcd": True,
        "env.obs_cfg.use_privileged_object_pcd": True,
        "env.obs_cfg.privileged_object_pcd_n_pts": 512,
        "env.obs_cfg.policy_object_pcd_n_pts": 256,
        "env.obs_cfg.policy_object_pcd_truncation_drop_ratio_range": (0.2, 0.7),
        "env.obs_cfg.policy_object_pcd_jitter_std": 0.002,
        # "env.obs_cfg.policy_object_pcd_n_pts": 480,
        # "env.obs_cfg.policy_object_pcd_truncation_drop_ratio_range": (0.005, 0.01),
        # "env.obs_cfg.policy_object_pcd_jitter_std": 1e-6,
        "env.obs_cfg.use_object_obs": True,
        "env.obs_cfg.use_object_vel_obs": False,
        "env.obs_cfg.rot_format": "6d",
        "env.obs_cfg.task_command": True,
        "env.obs_cfg.use_command_timestep": True,
        "env.obs_cfg.embed_commands_from_timestep": True,
        # TaskCfg
        "env.task_cfg.save_reference_video": True,
        "env.task_cfg.demo_dataset": "oakink_lift",
        "env.task_cfg.oakink_exported_path": str(
            # get_external_data_dir() / "franka_apple_low/apple.npy"
            get_logs_dir()
            / "trajopt_results/bowl/C12001_1e99dbed82_export_000/sequence_motion_franka_substitution_franka_oakink_lift_exported.npy"
        ),
        # "env.task_cfg.original_demo_path": str(
        #     get_external_data_dir()
        #     / "original_processed_verts"
        #     / "seqs"
        #     / "s01"
        #     / "laptop_use_02.npy"
        # ),
        "env.task_cfg.start_timestep": 0,
        "env.task_cfg.end_timestep": -1,
        "env.task_cfg.skip": 1,
        # "env.task_cfg.target_object": "arctic_laptop",
        # "env.task_cfg.reference_object_urdf": str(
        #     get_assets_dir() / "arctic_laptop/decomp/laptop_decomp_2.urdf"
        # ),
        # "env.task_cfg.target_object_urdf": str(
        #     get_assets_dir() / "arctic_laptop/decomp/laptop_decomp_2.urdf"
        # ),
        # "env.task_cfg.retargeted_results": (
        #     "20260118_172148/trajopt_return_initial_guess_processed.npz"
        # ),
        # "env.task_cfg.retargeted_results_upsample": False,
        "env.task_cfg.hand_name": "franka_xhand",
        "env.task_cfg.object_only_scene": False,
        "env.task_cfg.object_fixed": False,
        "env.task_cfg.lift_support_height": 0.75,
        "env.task_cfg.lift_support_margin": 0.002,
        "env.task_cfg.max_hand_pts_per_link": 200,
        "env.task_cfg.dense_hand_pts_per_link": 200,
        "env.task_cfg.max_pts_per_link": 2048,
        "env.task_cfg.ref_motion_noise_sig": (0.0, 0.0),
        # SimCfg
        "env.sim_cfg.gravity": -9.81,
        "env.sim_cfg.apply_random_force": False,
        "env.sim_cfg.random_force_mag": 1e-2,
        "env.sim_cfg.camera_look_at": (0.0, 0.0, 1.2),
        "env.sim_cfg.camera_ht": 1.8,
        "env.sim_cfg.camera_radius": 2.0,
        "env.sim_cfg.camera_theta": -3.14 / 2.5,
        "env.sim_cfg.resolution": (640, 480),
        "env.sim_cfg.camera_fov": 30,
        "env.sim_cfg.camera_mov": False,
        "env.sim_cfg.dt": 1.0 / 120.0,
        "env.sim_cfg.max_FPS": 200,
        "env.sim_cfg.show_viewer": False,
        "env.sim_cfg.batch_dofs_info": True,
        "env.sim_cfg.batch_links_info": True,
        "env.sim_cfg.n_rendered_envs": None,
        "env.sim_cfg.enable_joint_limit": True,
        "env.sim_cfg.max_collision_pairs": 1000,
        "env.sim_cfg.num_envs": 1024,
        "env.sim_cfg.video_save_freq": 5_000,
        "env.sim_cfg.video_save_length": 50,
        "env.sim_cfg.video_save_dir": "videos",
        # RealCfg
        "env.real_cfg.num_envs": 1,
        "env.real_cfg.video_save_dir": "real_videos",
        "env.real_cfg.joint_limit_soft_margin": 0.95,
        "env.real_cfg.visualization_only": False,
        "env.real_cfg.reset_to_default_on_initialize": False,
        "env.real_cfg.initialize_robot": True,
        "env.real_cfg.initialize_camera": True,
        "env.real_cfg.use_threaded_state_query": False,
        "env.real_cfg.use_threaded_control": False,
        "env.real_cfg.state_poll_period_s": 0.01,
        "env.real_cfg.control_period_s": 0.01,
        "env.real_cfg.robot_init_wait_s": 2.0,
        "env.real_cfg.robot_init_state_timeout_s": 10.0,
        "env.real_cfg.robot_init_state_poll_s": 0.05,
        "env.real_cfg.run_world_frame_calibration": False,
        "env.real_cfg.reuse_cached_world_calibration": True,
        "env.real_cfg.world_calibration_cache_path": (
            "real_deployment/deoxys_xhand/calib_camera_data/world_frame_calibration.npz"
        ),
        "env.real_cfg.robot_control_config_path": (
            "real_deployment/deoxys_xhand/configs/franka_xhand_base.yml"
        ),
        "env.real_cfg.robot_control_log_filename": "robot_control.log",
        "env.real_cfg.franka_joint_controller_cfg": "joint-impedance-controller.yml",
        "env.real_cfg.franka_osc_controller_cfg": "osc-pose-controller.yml",
        "env.real_cfg.franka_control_mode": "joint_impedance",
        "env.real_cfg.xhand_control_mode": "pos",
        "env.real_cfg.camera_query_resolution": False,
        "env.real_cfg.camera_img_w": 640,
        "env.real_cfg.camera_img_h": 480,
        "env.real_cfg.camera_fps": 30,
        "env.real_cfg.camera_capture_timeout_s": 2.0,
        "env.real_cfg.recording_frame_interval_s": 2.0,
        "env.real_cfg.marker_dict_name": "DICT_4X4_50",
        "env.real_cfg.base_marker_id": 1,
        "env.real_cfg.hand_outer_palm_marker_id": 2,
        "env.real_cfg.hand_palm_side_marker_id": 3,
        "env.real_cfg.thumb_marker_id": 4,
        "env.real_cfg.base_marker_size_m": 0.06,
        "env.real_cfg.hand_outer_palm_marker_size_m": 0.04,
        "env.real_cfg.hand_palm_side_marker_size_m": 0.01,
        "env.real_cfg.thumb_marker_size_m": 0.01,
        "env.real_cfg.wrist_to_marker_pos": (0.0, -0.027, -0.055),
        "env.real_cfg.wrist_to_marker_rotmat": (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0),
        ),
        "env.real_cfg.static_calibration_num_frames": 10,
        "env.real_cfg.world_calibration_num_robot_configs": 20,
        "env.real_cfg.world_calibration_frame_stride": 3,
        "env.real_cfg.world_calibration_settle_time_s": 0.25,
        "env.real_cfg.world_calibration_panda_fallback_tries": 8,
        "env.real_cfg.world_calibration_panda_fallback_std_rad": 0.2,
        "env.real_cfg.world_calibration_num_visible_candidates": 3,
        "env.real_cfg.world_calibration_marker_delta_translation_sigma_m": 0.02,
        "env.real_cfg.world_calibration_marker_delta_rotation_sigma_deg": 10.0,
        "env.real_cfg.world_calibration_marker_delta_regularization_weight": 0.1,
        "env.real_cfg.world_calibration_visibility_margin_px": 40.0,
        "env.real_cfg.world_calibration_overlay_point_size_px": 2,
        "env.real_cfg.world_calibration_num_manual_reviews": 3,
        "env.real_cfg.world_calibration_auto_accept_pos_margin_m": 0.06,
        "env.real_cfg.world_calibration_auto_accept_rot_margin_deg": 20.0,
        "env.real_cfg.depth_delta_threshold_m": 0.03,
        "env.real_cfg.rgb_delta_threshold": 45.0,
        "env.real_cfg.mask_erode_kernel_size": 3,
        "env.real_cfg.mask_erode_iterations": 1,
        "env.real_cfg.min_depth_m": 0.10,
        "env.real_cfg.max_depth_m": 1.50,
        "env.real_cfg.object_pcd_max_points": 4096,
        "env.real_cfg.world_x_min": 0.10,
        "env.real_cfg.table_plane_height_margin_m": 0.005,
        "env.real_cfg.world_calibration_robot_min_z_offset_m": 0.05,
        "env.real_cfg.robot_pcd_exclusion_radius_m": 0.02,
        "env.real_cfg.largest_cluster_eps_m": 0.03,
        "env.real_cfg.largest_cluster_min_points": 20,
        "env.real_cfg.visualize_runtime_debug": False,
        "env.real_cfg.visualize_runtime_point_size": 0.006,
        "env.real_cfg.pause_on_visualization_update": True,
        "env.real_cfg.visualize_policy_pcd": False,
        "env.real_cfg.visualize_policy_pcd_on": "reset",
        "env.real_cfg.visualize_policy_pcd_point_size": 0.01,
        # ActionCfg
        "env.action_cfg.action_mode": "diff",
        "env.action_cfg.residual_beta": 0.5,
        "env.action_cfg.arm_translation_delta_scale": 0.1,
        "env.action_cfg.arm_rotation_delta_scale": 0.7,
        "env.action_cfg.hand_joint_delta_scale": 0.5,
        "env.action_cfg.position_target_blend_alpha_start": 0.5,
        "env.action_cfg.position_target_blend_alpha_end": 1.0,
        # DiscObsCfg
        "env.disc_obs_cfg.hand_obj_cos_dist": True,
        "env.disc_obs_cfg.hand_obj_dir_vec": True,
        "env.disc_obs_cfg.delta_lpos": True,
        "env.disc_obs_cfg.delta_hand_obj_rel_bps": True,
        "env.disc_obs_cfg.reference_time_gaps": (
            (1, 0.4),
            (2, 0.3),
            (4, 0.2),
            (6, 0.1),
        ),
        # DiscCfg
        "env.disc_cfg.hidden_dims": (64, 64),
        "env.disc_cfg.seq_len": 1,
        "env.disc_cfg.memory_size": 8,
        "env.disc_cfg.input_mask_prob": 0.8,
        "env.disc_cfg.lr": 3e-4,
        "env.disc_cfg.weight_decay": 0.01,
        "env.disc_cfg.batch_size": 4096,
        "env.disc_cfg.n_steps": 1,
        # RewardCfgV2
        "env.rewards_cfg.use_retargeted_motion": True,
        "env.reference_cfg.enabled": True,
        "env.reference_cfg.source_motion": "retargeted",
        "env.reference_cfg.update_topk_percentile": 0.2,
        "env.reference_cfg.ema_alpha": 0.2,
        "env.reference_cfg.gp_lengthscale": 0.15,
        "env.reference_cfg.gp_observation_noise": 1e-3,
        "env.reference_cfg.object_gp_prior_position_std_m": 0.01,
        "env.reference_cfg.object_gp_prior_rotation_std_rad": 0.05,
        "env.reference_cfg.gp_prior_position_std_m": 0.05,
        "env.reference_cfg.gp_prior_rotation_std_rad": 0.3,
        "env.reference_cfg.gp_prior_joint_std_rad": 0.3,
        "env.reference_cfg.min_rollouts_for_update": 2,
        "env.reference_cfg.max_position_residual_m": 0.05,
        "env.reference_cfg.max_rotation_residual_rad": 0.5,
        "env.reference_cfg.max_joint_residual_rad": 0.3,
        "env.reference_cfg.pin_start": True,
        "env.reference_cfg.pin_end": False,
        "env.reference_cfg.builtin_pin_variance_scale": 0.1,
        "env.rewards_cfg.log_metrics": False,
        "env.rewards_cfg.log_essential_metrics_only": True,
        "env.rewards_cfg.dense_task_weight": 0.5,
        "env.rewards_cfg.pcd_distance_metric": "bps",
        "env.rewards_cfg.hand_cfg_weight": 0.1,
        "env.rewards_cfg.term_schedules.hand_cfg.scheme": "phase_window",
        "env.rewards_cfg.term_schedules.hand_cfg.phase": "lift",
        "env.rewards_cfg.term_schedules.hand_cfg.start_frac": 0.5,
        "env.rewards_cfg.term_schedules.hand_cfg.end_frac": 1.0,
        "env.rewards_cfg.term_schedules.hand_obj_rel.scheme": "phase_window",
        "env.rewards_cfg.term_schedules.hand_obj_rel.phase": "lift",
        "env.rewards_cfg.term_schedules.hand_obj_rel.start_frac": 0.5,
        "env.rewards_cfg.term_schedules.hand_obj_rel.end_frac": 1.0,
        "env.rewards_cfg.hand_obj_rel_weight": 0.15,
        "env.rewards_cfg.hand_obj_rel_reference_mode": "all_phases",
        "env.rewards_cfg.reach_hand_obj_rel_distance_completion_thd": 0.1,
        "env.rewards_cfg.reach_hand_obj_rel_direction_completion_thd": 0.1,
        "env.rewards_cfg.reach_hand_obj_rel_normal_completion_thd": 0.1,
        "env.rewards_cfg.reach_hand_obj_rel_distance_failure_thd": 10.0,
        "env.rewards_cfg.reach_hand_obj_rel_direction_failure_thd": 1.0,
        "env.rewards_cfg.reach_hand_obj_rel_normal_failure_thd": 1.0,
        "env.rewards_cfg.hand_obj_rel_distance_completion_thd": 0.1,
        "env.rewards_cfg.hand_obj_rel_direction_completion_thd": 0.1,
        "env.rewards_cfg.hand_obj_rel_normal_completion_thd": 0.1,
        "env.rewards_cfg.hand_obj_rel_distance_failure_thd": 1.0,
        "env.rewards_cfg.hand_obj_rel_direction_failure_thd": 1.0,
        "env.rewards_cfg.hand_obj_rel_normal_failure_thd": 1.0,
        "env.rewards_cfg.maniptrans_reward_weight": 0.0,
        "env.rewards_cfg.maniptrans_proximity_thd": 0.01,
        "env.rewards_cfg.maniptrans_lambda": 1.0,
        "env.rewards_cfg.task_completion_weight": 10.0,
        "env.rewards_cfg.last_reference_hold_steps": 10,
        "env.rewards_cfg.early_terminate": True,
        "env.rewards_cfg.task_weight": 1.0,
        "env.rewards_cfg.discriminator_weight": 0.0,
        "env.rewards_cfg.completion_pos_thd": 0.01,
        "env.rewards_cfg.completion_rot_thd": 10 * pi / 180,
        "env.rewards_cfg.completion_arti_thd": 10 * pi / 180,
        "env.rewards_cfg.completion_pcd_thd": 0.01,
        "env.rewards_cfg.hand_obj_max_dist_pos": 1.0,
        "env.rewards_cfg.obj_command_max_dist_pos": 0.15,
        "env.rewards_cfg.obj_command_max_dist_rot": 0.57,
        "env.rewards_cfg.obj_command_max_dist_arti": 0.57,
        "env.rewards_cfg.task_pos_scale": 10.0,
        "env.rewards_cfg.task_rot_scale": 2.0,
        "env.rewards_cfg.task_arti_scale": 5.0,
        "env.rewards_cfg.task_pcd_scale": 1.0,
        "env.rewards_cfg.hand_cfg_norm_scale": 2.0,
        "env.rewards_cfg.hand_cfg_dir_scale": 2.0,
        "env.rewards_cfg.hand_obj_displacement_scale": 2.0,
        "env.rewards_cfg.hand_obj_normal_scale": 1.0,
        "env.rewards_cfg.retargeted_joint_mse_reward_weight": 0.2,
        "env.rewards_cfg.term_schedules.retargeted_joint_mse.scheme": "reach_lift_exponential",
        "env.rewards_cfg.term_schedules.retargeted_joint_mse.exponential_rate": 5.0,
        "env.rewards_cfg.retargeted_joint_mse_reward_scale": 1.0,
        "env.rewards_cfg.joint_smoothness_reward_weight": 0.05,
        "env.rewards_cfg.term_schedules.joint_smoothness.scheme": "phase_levels",
        "env.rewards_cfg.term_schedules.joint_smoothness.reach_weight": 1.0,
        "env.rewards_cfg.term_schedules.joint_smoothness.lift_weight": 1.0,
        "env.rewards_cfg.use_filtered_contact": True,
        # ResetCfg
        "env.reset_cfg.start_range": (0, -5),
        "env.reset_cfg.start_phase_probs": (0.25, 0.5, 0.25),
        "env.reset_cfg.obj_noise": 0.00,
        "env.reset_cfg.hand_noise_reach_phase": 0.002,
        "env.reset_cfg.hand_noise_lift_phase": 0.001,
        "env.reset_cfg.reach_to_lift_transition_frames": 10,
        "env.reset_cfg.dr": False,
        "env.reset_cfg.dr_stiffness": (0.5, 1.5),
        "env.reset_cfg.dr_damping": (0.5, 1.5),
        "env.reset_cfg.dr_friction": (0.5, 1.5),
        "env.reset_cfg.dr_object_mass": (0.8, 1.2),
        "env.reset_cfg.eval_start_timestep": 0,
        "env.reset_cfg.wrist_only_retargeting": False,
        # CurriculumCfg
        "env.curriculum_cfg.use_curriculum": False,
        "env.curriculum_cfg.end_timestep": 50_000,
        "env.curriculum_cfg.update_freq": 1_000,
        "env.curriculum_cfg.task_dense_product_scale": (20.0, 40.0),
        # CommandsCfg
        "env.commands_cfg.use_gt_pos": True,
        "env.commands_cfg.use_gt_rot": True,
        "env.commands_cfg.rot_format": "6d",
        "env.commands_cfg.use_gt_arti": True,
        "env.commands_cfg.use_pcd": False,
        "env.commands_cfg.use_bps": False,
        "env.commands_cfg.bps_n_pts": 128,
        "env.commands_cfg.use_rel_bps": True,
        "env.commands_cfg.object_pcd_num_pts_sample": -1,
        "env.commands_cfg.relative_commands": False,
        "env.commands_cfg.use_retargeted_motion": False,
        # AgentConfig
        "agent.algo": "ppo",
        "agent.policy_lr": 1e-4,
        "agent.value_lr": 1e-4,
        "agent.gamma": 0.95,
        "agent.lam": 0.95,
        "agent.n_rollout_steps": 8,
        "agent.batch_size": 1024,
        "agent.update_epochs": 4,
        "agent.clip_ratio": 0.2,
        "agent.entropy_coef": 1e-3,
        "agent.normalize_advantages": True,
        "agent.vf_coef": 0.5,
        "agent.timesteps": 400_000,
        "agent.history_len": 0,
        "agent.resume": False,
        "agent.seed": 42,
        "agent.device": "cuda",
        "agent.check_overheat": False,
        "agent.use_discriminator": False,
        "agent.policy_embed_dim": 128,
        "agent.hand_synergy_dim": 32,
        "agent.value_embed_dim": 128,
        "agent.obs_proc_embed_dim": 64,
        "agent.obs_proc_layer_norm": True,
        "agent.obs_proc_pcd_hidden_dims": (64,),
        "agent.obs_proc_pcd_pool": "mean",
        "agent.hand_obj_bps_aux_loss_coef": 0.0,
        "agent.hand_obj_bps_aux_hidden_dims": (256,),
        "agent.resume_from_ckpt": None,
        "agent.policy_inputs": (
            "right",
            "ref_right",
            "contact",
            "command",
            "object_pcd",
            "hand_pcd",
        ),
        "agent.critic_inputs": ("right", "contact", "command", "object"),
        "agent.eval_freq": 0,
        "agent.eval_metrics_freq": 0,
        "agent.eval_video_freq": -1,
        "agent.eval_episodes": 1,
        "agent.num_eval_envs": 10,
        # LoggerConfig
        "logger.project": "dexlink",
        "logger.runname": "default_run",
        "logger.logdir": str(get_logs_dir()),
        "logger.group": "default_group",
        "logger.disable": False,
        "logger.seed": True,
        "logger.timestamp": True,
        "logger.log_freq": -1,
        "logger.model_save_freq": -1,
        "logger.desired_num_logs": 2500,
        "logger.desired_model_saves": 150,
        # EvalConfig
        "eval.num_additional_kpts": 200,
        "eval.relative_traj": True,
        "eval.logdir": "logs/eval",
        "eval.chamfer_pairs_per_batch": 512,
        "eval.chamfer_row_chunk_size": 4,
        # RealEvalConfig
        "real_eval.rollout_length": 200,
        "real_eval.logdir": "real_eval",
        "real_eval.seed": 42,
        "real_eval.initialization_only": False,
        "real_eval.single_step_only": False,
        "real_eval.command_substeps": 10,
    }
}


# create seeds
seeds = [42, 60, 72]


for idx, seed in enumerate(seeds):
    preset_name = f"oakink_default_v6_{idx}"
    preset = OAKINK_DEFAULT_PRESET["oakink_default"].copy()
    preset["preset"] = preset_name
    preset["logger.group"] = "oakink_default_v6"
    preset["env.seed"] = seed

    OAKINK_DEFAULT_PRESET[preset_name] = preset


for idx, seed in enumerate(seeds):
    preset_name = f"oakink_default_v7_{idx}"
    preset = OAKINK_DEFAULT_PRESET["oakink_default"].copy()
    preset["preset"] = preset_name
    preset["logger.group"] = "oakink_default_v7"
    preset["env.seed"] = seed
    preset["env.rewards_cfg.term_schedules.object_tracking.scheme"] = "stage_levels"
    preset["env.rewards_cfg.term_schedules.object_tracking.reach_weight"] = 0.0
    preset["env.rewards_cfg.term_schedules.object_tracking.transition_weight"] = 0.0
    preset["env.rewards_cfg.term_schedules.object_tracking.lift_weight"] = 1.0
    preset["env.rewards_cfg.term_schedules.hand_cfg.scheme"] = "constant"
    preset["env.rewards_cfg.term_schedules.hand_cfg.value"] = 1.0
    preset["env.rewards_cfg.term_schedules.hand_obj_rel.scheme"] = "constant"
    preset["env.rewards_cfg.term_schedules.hand_obj_rel.value"] = 1.0
    preset["env.rewards_cfg.term_schedules.retargeted_joint_mse.scheme"] = "constant"
    preset["env.rewards_cfg.term_schedules.retargeted_joint_mse.value"] = 1.0
    preset["env.rewards_cfg.dense_task_weight"] = 0.75
    preset["env.rewards_cfg.hand_cfg_weight"] = 0.1
    preset["env.rewards_cfg.hand_obj_rel_weight"] = 0.1
    preset["env.rewards_cfg.joint_smoothness_reward_weight"] = 0.05

    OAKINK_DEFAULT_PRESET[preset_name] = preset

for idx, seed in enumerate(seeds):
    preset_name = f"oakink_default_v8_{idx}"
    preset = OAKINK_DEFAULT_PRESET["oakink_default"].copy()
    preset["preset"] = preset_name
    preset["logger.group"] = "oakink_default_v8"
    preset["env.seed"] = seed
    # preset["agent.update_epochs"] = 2
    preset["agent.policy_lr"] = 1e-5
    # preset['agent.n_rollout_steps'] = 16
    preset["agent.policy_embed_dim"] = 256
    preset["agent.hand_synergy_dim"] = 256
    preset["agent.value_embed_dim"] = 256
    preset["agent.obs_proc_embed_dim"] = 128
    preset["agent.obs_proc_layer_norm"] = False
    preset["agent.obs_proc_pcd_hidden_dims"] = (128,)
    # set retargeted joint mse reward to 0
    preset["env.rewards_cfg.retargeted_joint_mse_reward_weight"] = 0.0
    OAKINK_DEFAULT_PRESET[preset_name] = preset


gp_base_preset = OAKINK_DEFAULT_PRESET["oakink_default_v7_0"].copy()
gp_base_preset.update(
    {
        "env.reference_cfg.enabled": True,
        "env.reference_cfg.source_motion": "retargeted",
        "env.reference_cfg.update_freq_policy_iters": 10,
        "env.reference_cfg.update_topk_percentile": 0.05,
        "env.reference_cfg.ema_alpha": 0.2,
        "env.reference_cfg.gp_lengthscale": 0.30,
        "env.reference_cfg.gp_observation_noise": 1e-3,
        "env.reference_cfg.object_gp_prior_position_std_m": 0.01,
        "env.reference_cfg.object_gp_prior_rotation_std_rad": 0.1,
        "env.reference_cfg.min_rollouts_for_update": 2,
        "env.reference_cfg.max_position_residual_m": 0.05,
        "env.reference_cfg.max_rotation_residual_rad": 0.5,
        "env.reference_cfg.max_joint_residual_rad": 0.3,
        "env.reference_cfg.pin_start": True,
        "env.reference_cfg.pin_end": True,
        "env.reference_cfg.builtin_pin_variance_scale": 0.1,
        "env.reference_cfg.anchor_observation_weight": 100.0,
        "env.reset_cfg.obj_noise": 0.0,
        "env.reset_cfg.hand_noise_reach_phase": 0.002,
        "env.reset_cfg.hand_noise_lift_phase": 0.001,
        "env.reset_cfg.reverse_curriculum": True,
        "env.reset_cfg.reverse_curriculum_window_size": 5,
        "env.reset_cfg.reverse_curriculum_shift_size": 2,
        "env.reset_cfg.reverse_curriculum_policy_iter_interval": 20,
        "env.reset_cfg.reverse_curriculum_variance_threshold": 0.01,
        # "env.sim_cfg.num_envs": 256,
        "env.task_cfg.hand_name": "xhand",
        "env.task_cfg.oakink_exported_path": str(
            get_logs_dir()
            / "trajopt_results/bowl/C12001_1e99dbed82_export_000/sequence_motion_oakink_lift_exported.npy"
        ),
    }
)

for idx, seed in enumerate(seeds):
    preset_name = f"gp_v6_{idx}"
    preset = gp_base_preset.copy()
    preset["preset"] = preset_name
    preset["logger.group"] = "gp_v6"
    preset["logger.runname"] = preset_name
    preset["env.seed"] = seed
    OAKINK_DEFAULT_PRESET[preset_name] = preset


ppo_base_preset = OAKINK_DEFAULT_PRESET["oakink_default_v7_0"].copy()
ppo_base_preset.update(
    {
        "env.reference_cfg.enabled": False,
        "env.reset_cfg.reverse_curriculum": False,
    }
)

ppo_preset_xhand = ppo_base_preset.copy()
ppo_preset_xhand.update(
    {
        "env.task_cfg.hand_name": "xhand",
        "env.task_cfg.oakink_exported_path": str(
            get_logs_dir()
            / "trajopt_results/bowl/C12001_1e99dbed82_export_000/sequence_motion_oakink_lift_exported.npy"
        ),
        "logger.group": "ppo_xhand",
        "logger.runname": "ppo_xhand",
    }
)

OAKINK_DEFAULT_PRESET["ppo_xhand"] = ppo_preset_xhand


for idx, seed in enumerate(seeds):
    preset_name = f"ppo_{idx}"
    preset = ppo_base_preset.copy()
    preset["preset"] = preset_name
    preset["logger.group"] = "ppo"
    preset["logger.runname"] = preset_name
    preset["env.seed"] = seed
    OAKINK_DEFAULT_PRESET[preset_name] = preset


ppo_lowlog_base_preset = ppo_base_preset.copy()
ppo_lowlog_base_preset.update(
    {
        "agent.log_essential_metrics_only": True,
        "env.rewards_cfg.log_metrics": False,
        "env.rewards_cfg.log_essential_metrics_only": True,
    }
)

for idx, seed in enumerate(seeds):
    preset_name = f"ppo_lowlog_{idx}"
    preset = ppo_lowlog_base_preset.copy()
    preset["preset"] = preset_name
    preset["logger.group"] = "ppo_lowlog"
    preset["logger.runname"] = preset_name
    preset["env.seed"] = seed
    OAKINK_DEFAULT_PRESET[preset_name] = preset


gp_lowlog_base_preset = gp_base_preset.copy()
gp_lowlog_base_preset.update(
    {
        "agent.log_essential_metrics_only": True,
        "env.rewards_cfg.log_metrics": False,
        "env.rewards_cfg.log_essential_metrics_only": True,
    }
)

for idx, seed in enumerate(seeds):
    preset_name = f"gp_lowlog_v5_{idx}"
    preset = gp_lowlog_base_preset.copy()
    preset["preset"] = preset_name
    preset["logger.group"] = "gp_lowlog_v5"
    preset["logger.runname"] = preset_name
    preset["env.seed"] = seed
    OAKINK_DEFAULT_PRESET[preset_name] = preset


finetune_preset = {
    "agent.timesteps": 100_000,
    "agent.resume": True,
    "logger.desired_num_logs": 1500,
    "logger.desired_model_saves": 50,
    "logger.group": "finetune_v2",
    "env.rewards_cfg.dense_task_weight": 0.75,
    "env.rewards_cfg.hand_cfg_weight": 0.1,
    "env.rewards_cfg.hand_obj_rel_weight": 0.10,
    "env.rewards_cfg.joint_smoothness_reward_weight": 0.05,
    "env.sim_cfg.apply_random_force": False,
}


new_demo_path = get_external_data_dir() / "tennis_ball" / "ball.npy"
base_preset = (OAKINK_DEFAULT_PRESET["oakink_default_v6_0"]).copy()
base_preset.update(finetune_preset)
base_preset["env.task_cfg.oakink_exported_path"] = str(new_demo_path)
base_preset["agent.resume_from_ckpt"] = None
base_preset["logger.runname"] = "oakink_tennis_finetune"
OAKINK_DEFAULT_PRESET["oakink_tennis_finetune"] = base_preset

sim_eval_preset = OAKINK_DEFAULT_PRESET["oakink_default_v6_0"].copy()
sim_eval_preset.update(
    {
        "preset": "oakink_eval_sim",
        "env.backend": "sim",
        "env.sim_cfg.num_envs": 1,
        "env.sim_cfg.n_rendered_envs": 1,
        "env.sim_cfg.show_viewer": True,
        "env.sim_cfg.max_FPS": 60,
        "env.sim_cfg.apply_random_force": False,
        "env.reset_cfg.hand_noise_reach_phase": 0.5,
        "env.reset_cfg.hand_noise_lift_phase": 0.001,
        "env.visualize_policy_pcd": False,
        "env.visualize_policy_pcd_on": "step",
        "agent.resume": True,
        "agent.resume_from_ckpt": None,
        "agent.eval_freq": 5000,
        "agent.eval_metrics_freq": -1,
        "agent.eval_video_freq": -1,
        "logger.group": "sim_eval",
        "logger.runname": "eval_sim",
        "logger.disable": True,
        "logger.timestamp": True,
    }
)
OAKINK_DEFAULT_PRESET["oakink_eval_sim"] = sim_eval_preset

real_deployment_preset = OAKINK_DEFAULT_PRESET["oakink_default_v6_0"].copy()
real_deployment_preset.update(
    {
        "preset": "real_deployment",
        "env.backend": "real",
        "env.sim_cfg.show_viewer": False,
        "env.reset_cfg.hand_noise_reach_phase": 0.001,
        "env.reset_cfg.hand_noise_lift_phase": 0.001,
        "env.visualize_policy_pcd": True,
        "env.real_cfg.initialize_robot": True,
        "env.real_cfg.initialize_camera": True,
        "env.real_cfg.use_threaded_state_query": False,
        "env.real_cfg.use_threaded_control": False,
        "env.real_cfg.control_freq_hz": 10.0,
        "env.real_cfg.calibration_control_freq_hz": 1.0,
        "env.real_cfg.control_target_progress_per_update": 0.32,
        "env.real_cfg.control_period_s": 0.01,
        "env.real_cfg.reset_motion_franka_tol_rad": 0.02,
        "env.real_cfg.reset_motion_xhand_tol_rad": 0.1,
        # "env.real_cfg.wris_pos_alpha": 0.35,  # this is applied on top of the scale from simulation
        # "env.real_cfg.wrist_rot_alpha": 0.31,  # this is applied on top of the scale from simulation
        # "env.real_cfg.fingers_pos_alpha": 0.3,  # this is applied on top of the scale from simulation
        "env.real_cfg.wris_pos_alpha": 1.0,  # this is applied on top of the scale from simulation
        "env.real_cfg.wrist_rot_alpha": 1.0,  # this is applied on top of the scale from simulation
        "env.real_cfg.fingers_pos_alpha": 1.0,  # this is applied on top of the scale from simulation
        "env.action_cfg.position_target_blend_alpha_start": 0.001,
        "env.action_cfg.position_target_blend_alpha_end": 0.35,
        "env.decimation": 10,
        "env.real_cfg.table_plane_height_margin_m": 0.075,
        "env.real_cfg.use_phase_cached_object_pcd": True,
        "env.real_cfg.reuse_cached_world_calibration": True,
        "env.real_cfg.run_world_frame_calibration": False,
        "env.real_cfg.world_calibration_cache_path": (
            "real_deployment/deoxys_xhand/calib_camera_data/world_frame_calibration.npz"
        ),
        "env.real_cfg.recording_frame_interval_s": 0.2,
        "agent.resume": True,
        "agent.resume_from_ckpt": None,
        "real_eval.rollout_length": 200,
        "real_eval.logdir": "real_eval",
        "real_eval.seed": 42,
        "real_eval.command_substeps": 8,
    }
)
OAKINK_DEFAULT_PRESET["real_deployment"] = real_deployment_preset

real_deployment_fresh_object_pcd_preset = real_deployment_preset.copy()
real_deployment_fresh_object_pcd_preset.update(
    {
        "preset": "real_deployment_fresh_object_pcd",
        "env.real_cfg.use_phase_cached_object_pcd": False,
    }
)

real_deployment_heuristic_object_pcd_preset = real_deployment_preset.copy()
real_deployment_heuristic_object_pcd_preset.update(
    {
        "preset": "real_deployment_heuristic_object_pcd",
        "env.real_cfg.use_phase_cached_object_pcd": True,
    }
)

real_deployment_meshcat_debug_preset = real_deployment_preset.copy()
real_deployment_meshcat_debug_preset.update(
    {
        "preset": "real_deployment_meshcat_debug",
        "env.real_cfg.visualize_runtime_debug": True,
        "env.real_cfg.pause_on_visualization_update": True,
        "env.visualize_policy_pcd": True,
        "env.visualize_policy_pcd_on": "step",
    }
)

real_deployment_bypass_control_preset = real_deployment_meshcat_debug_preset.copy()
real_deployment_bypass_control_preset.update(
    {
        "preset": "real_deployment_bypass_control",
        "env.real_cfg.bypass_low_level_control": True,
        "real_eval.print_command_timing": True,
    }
)

real_startup_timing_only_preset = real_deployment_preset.copy()
real_startup_timing_only_preset.update(
    {
        "preset": "real_startup_timing_only",
        "env.real_cfg.bypass_low_level_control": True,
        "real_eval.print_command_timing": True,
        "real_eval.startup_timing_only": True,
    }
)

real_control_test_preset = real_deployment_preset.copy()
real_control_test_preset.update(
    {
        "preset": "real_control_test",
        "env.real_cfg.initialize_camera": False,
        "env.real_cfg.run_world_frame_calibration": False,
        "env.real_cfg.control_target_progress_per_update": 0.3,
        "env.real_cfg.control_period_s": 0.01,
        "env.real_cfg.reuse_cached_world_calibration": False,
        "env.real_cfg.use_threaded_control": False,
        "env.real_cfg.use_threaded_state_query": False,
    }
)

world_frame_calibration_preset = real_deployment_preset.copy()
world_frame_calibration_preset.update(
    {
        "preset": "world_frame_calibration",
        "real_eval.initialization_only": True,
        "env.real_cfg.run_world_frame_calibration": True,
        "env.real_cfg.reuse_cached_world_calibration": False,
        # "env.visualize_policy_pcd": True,
        # "env.visualize_policy_pcd_on": "step",
        "env.real_cfg.visualize_runtime_debug": True,
        "env.real_cfg.pause_on_visualization_update": True,
        "env.visualize_policy_pcd": True,
        "env.visualize_policy_pcd_on": "step",
    }
)

check_scene_with_calibration_only_preset = world_frame_calibration_preset.copy()
check_scene_with_calibration_only_preset.update(
    {
        "env.real_cfg.run_world_frame_calibration": False,
        "env.real_cfg.reuse_cached_world_calibration": True,
        "env.real_cfg.initialize_robot": False,
        "env.real_cfg.initialize_camera": True,
        "preset": "check_scene_with_calib_only",
    }
)

# first step only
real_first_step_only_preset = real_deployment_meshcat_debug_preset.copy()
real_first_step_only_preset.update(
    {
        "preset": "real_first_step_only",
        "real_eval.single_step_only": True,
    }
)

OAKINK_DEFAULT_PRESET["real_deployment_meshcat_debug"] = (
    real_deployment_meshcat_debug_preset
)
OAKINK_DEFAULT_PRESET["real_deployment_fresh_object_pcd"] = (
    real_deployment_fresh_object_pcd_preset
)
OAKINK_DEFAULT_PRESET["real_deployment_heuristic_object_pcd"] = (
    real_deployment_heuristic_object_pcd_preset
)
OAKINK_DEFAULT_PRESET["real_deployment_bypass_control"] = (
    real_deployment_bypass_control_preset
)
OAKINK_DEFAULT_PRESET["real_startup_timing_only"] = real_startup_timing_only_preset
OAKINK_DEFAULT_PRESET["real_control_test"] = real_control_test_preset
OAKINK_DEFAULT_PRESET["check_scene_with_calib_only"] = (
    check_scene_with_calibration_only_preset
)
OAKINK_DEFAULT_PRESET["real_first_step_only"] = real_first_step_only_preset
OAKINK_DEFAULT_PRESET["world_frame_calibration"] = world_frame_calibration_preset
