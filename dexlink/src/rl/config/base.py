from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from math import pi
from typing import Any, Dict, List, get_args, get_type_hints

from dexlink.utils.retargeting.path_utils import (
    get_assets_dir,
    get_external_data_dir,
    get_logs_dir,
    get_src,
)
from dexlink.src.rl.config.oakink_presets import OAKINK_DEFAULT_PRESET
from dexlink.src.rl.config.seq_oakink_final_experiment_presets import (
    SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS,
)
from dexlink.src.rl.config.seq_oakink_method_ablation_presets import (
    SEQ_OAKINK_METHOD_ABLATION_PRESETS,
)
from dexlink.src.rl.config.seq_oakink_presets import (
    SEQ_OAKINK_PRESETS,
    required_seq_task_settings,
)


@dataclass
class ResetCfg:
    start_range: tuple[int, int] = (
        0,
        -5,
    )  # range of timesteps to randomly start from at reset, -10 means total_steps - 10
    start_phase_probs: tuple[float, float, float] | None = (
        None  # reach, transition, lift
    )
    obj_noise: float = 0.00
    hand_noise_reach_phase: float = 0.005
    hand_noise_lift_phase: float = 0.005
    reach_to_lift_transition_frames: int = 5
    dr: bool = False  # whether to use domain randomization
    dr_stiffness: tuple[float, float] = (0.5, 1.5)  # randomize joint stiffness
    dr_damping: tuple[float, float] = (0.5, 1.5)  # randomize joint damping
    dr_friction: tuple[float, float] = (0.5, 1.5)  # randomize joint friction
    dr_object_mass: tuple[float, float] = (0.01, 0.35)  # randomize total object mass
    dr_object_mass_min_abs: float = 0.01
    dr_contact_friction: tuple[float, float] = (
        0.8,
        1.2,
    )  # randomize absolute object surface friction coefficient
    # use this timestep to start the eval with
    eval_start_timestep: int = 0
    wrist_only_retargeting: bool = False  # if True, only use wrist values.
    retargeted_hand_reference_mode: str = "retargeted"
    reverse_curriculum: bool = False
    reverse_curriculum_window_size: int = 5
    reverse_curriculum_shift_size: int = 2
    reverse_curriculum_policy_iter_interval: int = 200
    reverse_curriculum_variance_threshold: float = 0.01
    reset_timestep_sampling_mode: str = "default"
    multi_object_reset_timestep_density_ema_alpha: float = 0.05


@dataclass
class CurriculumCfg:
    use_curriculum: bool = True
    end_timestep: int = 50_000
    update_freq: int = 1_000
    # the following reward_cfg attribute will be updated.
    task_dense_product_scale: tuple[float, float] = (20.0, 40.0)


@dataclass
class ReferenceTrajectoryCfg:
    enabled: bool = False
    source_motion: str = "retargeted"
    update_freq_policy_iters: int = 10
    update_topk_percentile: float = 0.2
    ema_alpha: float = 0.2
    gp_lengthscale: float = 0.15
    gp_observation_noise: float = 1.0e-4
    object_gp_prior_position_std_m: float = 0.01
    object_gp_prior_rotation_std_rad: float = 0.05
    gp_prior_position_std_m: float = 0.05
    gp_prior_rotation_std_rad: float = 0.3
    gp_prior_joint_std_rad: float = 0.3
    min_rollouts_for_update: int = 2
    max_position_residual_m: float = 0.05
    max_rotation_residual_rad: float = 0.5
    max_joint_residual_rad: float = 0.3
    pin_start: bool = True
    pin_end: bool = False
    pinned_timesteps: tuple[int, ...] = ()
    builtin_pin_variance_scale: float = 0.1
    anchor_observation_weight: float = 100.0


@dataclass
class TaskMetricCfg:
    metric_id: str | None = None
    eval_metric_id: str | None = None
    notebook_object_id: str = ""
    notebook_angle_tolerance_rad: float = 15.0 * pi / 180.0
    stacking_object_ids: tuple[str, ...] = ()
    stacking_min_center_height_offsets_m: tuple[float, ...] = ()
    banana_object_id: str = ""
    banana_source_hand: str = ""
    banana_receiving_hand: str = ""
    banana_min_center_height_above_table_m: float = 0.05
    banana_min_receiving_support_weight_fraction: float = 0.5
    banana_max_source_contact_weight_fraction: float = 0.1
    contact_force_threshold: float = 1.0e-4
    hammer_object_id: str = ""
    hammer_peg_object_id: str = ""
    hammer_peg_sensor_top_offset_m: float = 0.01
    hammer_contact_force_threshold: float = 1.0e-4
    hammer_strike_phase_names: tuple[str, ...] = ()
    hammer_first_strike_phase_names: tuple[str, ...] = ()
    hammer_second_strike_phase_names: tuple[str, ...] = ()
    object_pose_object_id: str = ""
    object_pose_position_tolerance_m: float = 0.03
    object_pose_orientation_tolerance_rad: float = 15.0 * pi / 180.0
    object_lift_object_id: str = ""
    bottle_interaction_object_id: str = ""
    bottle_interaction_hand_side: str = ""
    bottle_relative_wrist_position_tolerance_m: float = 0.05
    bottle_relative_wrist_orientation_tolerance_rad: float = 30.0 * pi / 180.0
    cube_strategy_object_id: str = ""
    cube_strategy_expected_lift: bool = False
    cube_strategy_min_lift_height_change_m: float = 0.02
    cube_strategy_position_tolerance_m: float = 0.03
    cube_strategy_orientation_tolerance_rad: float = 30.0 * pi / 180.0
    dispenser_object_id: str = ""
    dispenser_joint_name: str = ""
    dispenser_pressed_tolerance_m: float = 0.002
    drawer_object_id: str = ""
    drawer_joint_name: str = ""
    drawer_final_articulation_tolerance_m: float = 0.01
    eraser_object_id: str = ""
    eraser_phase_names: tuple[str, ...] = ()
    eraser_max_position_error_m: float = 0.02
    mounting_cylinder_object_id: str = ""
    mounting_post_object_id: str = ""
    mounting_cylinder_height_m: float = 0.0
    mounting_bore_clearance_m: float = 0.0
    mounting_max_origin_height_above_post_m: float = 0.0
    mounting_max_axis_tilt_rad: float = 0.0
    grasp_stability_object_id: str = ""
    grasp_stability_max_drift_m: float = 0.01
    mug_object_id: str = ""
    mug_hand_side: str = ""
    mug_reference_grasp_phase_names: tuple[str, str] = ("", "")
    mug_min_center_height_above_table_m: float = 0.05


@dataclass
class TaskCfg:
    save_reference_video: bool = True
    demo_dataset: str = (
        "oakink_lift"  # "oakink", "oakink_grasp", "oakink_lift", "arctic", or "random_traj"
    )
    oakink_exported_path: str = str(
        get_external_data_dir() / "oakink_exported" / "handspray12.npy"
    )
    original_demo_path: str = str(
        get_external_data_dir()
        / "original_processed_verts"
        / "seqs"
        / "s01"
        / "laptop_use_02.npy"
    )
    start_timestep: int = 100
    end_timestep: int = 200  # -1 means till the end
    skip: int = 1  # subsample the demo by this factor
    target_object: str = "arctic_laptop"
    reference_object_urdf: str = str(
        get_assets_dir() / "arctic_laptop/decomp/laptop_decomp_2.urdf"
    )
    target_object_urdf: str = str(
        get_assets_dir() / "arctic_laptop/decomp/laptop_decomp_2.urdf"
    )
    # wrt to directory: get_logs_dir()/trajopt_results/
    retargeted_results: str | None = (
        "20260118_172148/trajopt_return_initial_guess_processed.npz"
    )
    retargeted_results_upsample: bool = False
    hand_name: str = "franka_xhand"
    object_only_scene: bool = False
    object_fixed: bool = False
    lift_support_height: float = 0.75
    lift_support_margin: float = 0.002
    use_scene_support: bool = True
    grasp_refinement_mode: bool = False
    max_pts_per_link: int = 200
    max_hand_pts_per_link: int = 100
    dense_hand_pts_per_link: int = 200
    metric_cfg: TaskMetricCfg = field(default_factory=TaskMetricCfg)
    require_complete_human_hand_reference: bool = False
    # noise for reference/retargeted motion
    # for translation (wrist joints) and revolute joints (wrist and fingers)
    ref_motion_noise_sig: tuple[float, float] = (0.0, 0.0)


@dataclass
class SimCfg:
    gravity: float = -9.81  # -9.81 is the true val, 0.0 will disable gravity
    apply_random_force: bool = False
    random_force_mag: float = 5.0

    camera_look_at: tuple[float, float, float] = (0.0, 0.0, 1.2)
    camera_ht: float = 1.8
    camera_radius: float = 2.0
    camera_theta: float = -3.14 / 2.5  # in radians
    resolution: tuple[int, int] = (640, 480)
    camera_fov: int = 30
    camera_mov: bool = False

    dt: float = 1.0 / 120.0  # 120 hz
    max_FPS: int = 200
    show_viewer: bool = False
    batch_dofs_info: bool = True
    batch_links_info: bool = True
    n_rendered_envs: int | None = None
    enable_joint_limit: bool = True
    max_collision_pairs: int = 1000
    rapid_collision_resolution: bool = True
    num_envs: int = 4096

    video_save_freq: int = 20_000
    video_save_length: int = 100
    video_save_dir: str = "videos"


@dataclass
class RealCfg:
    num_envs: int = 1  # for consistency, should not be changed
    video_save_dir: str = "real_videos"
    control_freq_hz: float = 10.0
    calibration_control_freq_hz: float = 1.0
    control_target_progress_per_update: float = 0.3
    control_target_max_joint_step_rad: float = 0.05
    wris_pos_alpha: float = 1.0
    wrist_rot_alpha: float = 1.0
    fingers_pos_alpha: float = 1.0
    joint_limit_soft_margin: float = 0.95
    adaptive_decimation_joint_step_deg: float = 10.0
    visualization_only: bool = False
    reset_to_default_on_initialize: bool = False
    bypass_low_level_control: bool = False
    initialize_robot: bool = True
    initialize_camera: bool = True
    read_robot_state_on_initialize: bool = True
    start_camera_polling_on_initialize: bool = True
    use_threaded_state_query: bool = False
    use_threaded_control: bool = False
    state_poll_period_s: float = 0.01
    control_period_s: float = 0.01
    reset_motion_franka_tol_rad: float = 0.05
    reset_motion_xhand_tol_rad: float = 0.15
    reset_motion_max_steps: int = 200
    reset_motion_franka_max_step_rad: float = 0.05
    reset_motion_xhand_steps: int = 50
    robot_init_wait_s: float = 2.0
    robot_init_state_timeout_s: float = 10.0
    robot_init_state_poll_s: float = 0.05
    run_world_frame_calibration: bool = False
    reuse_cached_world_calibration: bool = True
    world_calibration_cache_path: str = str(
        get_src() / "camera_calib" / "world_frame_calibration.npz"
    )
    robot_control_config_path: str = (
        "real_deployment/deoxys_xhand/configs/franka_xhand_base.yml"
    )
    robot_control_log_filename: str = "robot_control.log"
    real_control_hand_name: str = "franka_xhand"
    franka_joint_controller_cfg: str = "joint-impedance-controller.yml"
    franka_osc_controller_cfg: str = "osc-pose-controller.yml"
    franka_control_mode: str = "joint_impedance"
    xhand_control_mode: str = "pos"
    camera_query_resolution: bool = False
    camera_img_w: int = 640
    camera_img_h: int = 480
    camera_fps: int = 30
    camera_capture_timeout_s: float = 2.0
    recording_frame_interval_s: float = 2.0
    marker_dict_name: str = "DICT_4X4_50"
    base_marker_id: int = 1
    hand_outer_palm_marker_id: int = 2
    hand_palm_side_marker_id: int = 3
    thumb_marker_id: int = 4
    base_marker_size_m: float = 0.06
    hand_outer_palm_marker_size_m: float = 0.04
    hand_palm_side_marker_size_m: float = 0.01
    thumb_marker_size_m: float = 0.01
    wrist_to_marker_pos: tuple[float, float, float] = (0.0, -0.02, -0.055)
    wrist_to_marker_rotmat: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
    static_calibration_num_frames: int = 10
    world_calibration_num_robot_configs: int = 20
    world_calibration_frame_stride: int = 3
    world_calibration_settle_time_s: float = 0.25
    world_calibration_panda_fallback_tries: int = 8
    world_calibration_panda_fallback_std_rad: float = 0.2
    world_calibration_candidate_min_joint_l1_deg: float = 20.0
    world_calibration_num_visible_candidates: int = 3
    world_calibration_marker_delta_translation_sigma_m: float = 0.02
    world_calibration_marker_delta_rotation_sigma_deg: float = 10.0
    world_calibration_marker_delta_regularization_weight: float = 0.1
    world_calibration_visibility_margin_px: float = 40.0
    world_calibration_overlay_point_size_px: int = 2
    world_calibration_num_manual_reviews: int = 3
    world_calibration_auto_accept_pos_margin_m: float = 0.06
    world_calibration_auto_accept_rot_margin_deg: float = 20.0
    depth_delta_threshold_m: float = 0.03
    rgb_delta_threshold: float = 45.0
    mask_erode_kernel_size: int = 3
    mask_erode_iterations: int = 1
    min_depth_m: float = 0.10
    max_depth_m: float = 1.50
    use_phase_cached_object_pcd: bool = True
    phase_cached_object_pcd_motion_start_phase: str = "lift"
    object_pcd_max_points: int = 4096
    world_x_min: float = 0.10
    world_z_min: float = 0.0
    table_plane_height_margin_m: float = 0.005
    table_plane_sample_points: int = 200
    table_plane_synthetic_half_extent_m: float = 0.30
    table_plane_synthetic_depth_m: float = 0.005
    world_calibration_robot_min_z_offset_m: float = 0.05
    robot_pcd_exclusion_radius_m: float = 0.03
    largest_cluster_eps_m: float = 0.03
    largest_cluster_min_points: int = 20
    visualize_runtime_debug: bool = False
    visualize_runtime_point_size: float = 0.006
    visualize_runtime_full_arm_mesh: bool = True
    runtime_meshcat_port: int = 6000
    pause_on_visualization_update: bool = True
    print_real_control_stages: bool = False

    visualize_policy_pcd: bool = False
    visualize_policy_pcd_on: str = "reset"  # one of: 'reset', 'step', 'both'
    visualize_policy_pcd_point_size: float = 0.01


@dataclass
class EnvConfig:
    device: str = "cuda"
    seed: int = 42
    decimation: int = 1
    backend: str = "sim"  # "sim" or "real"
    env_class: str = "multi_object"  # "single_object" or "multi_object"
    key_task_protocol_version: str = ""

    visualize_policy_pcd: bool = False
    visualize_policy_pcd_on: str = "reset"  # one of: 'reset', 'step', 'both'
    visualize_policy_pcd_point_size: float = 0.01

    obs_cfg: ObsCfg = field(default_factory=lambda: ObsCfg())
    task_cfg: TaskCfg = field(default_factory=lambda: TaskCfg())
    sim_cfg: SimCfg = field(default_factory=lambda: SimCfg())
    real_cfg: RealCfg = field(default_factory=lambda: RealCfg())

    action_cfg: ActionCfg = field(default_factory=lambda: ActionCfg())
    disc_obs_cfg: DiscObsCfg = field(default_factory=lambda: DiscObsCfg())
    disc_cfg: DiscCfg = field(default_factory=lambda: DiscCfg())
    rewards_cfg: RewardCfgV2 = field(default_factory=lambda: RewardCfgV2())
    reset_cfg: ResetCfg = field(default_factory=lambda: ResetCfg())
    curriculum_cfg: CurriculumCfg = field(default_factory=lambda: CurriculumCfg())
    reference_cfg: ReferenceTrajectoryCfg = field(
        default_factory=lambda: ReferenceTrajectoryCfg()
    )
    commands_cfg: CommandsCfg = field(default_factory=lambda: CommandsCfg())


# class ObjObsCfg:
#     use_pos: bool = True
#     use_rot: bool = True
#     rot_format: str = "6d"  # 'quat' or 'euler' or '6d
#     use_pcd: bool = False  # point cloud on the object surface


@dataclass
class RobotObsCfg:
    use_jpos: bool = True
    use_jvel: bool = True
    use_wrist_pose: bool = True
    use_jacc_privileged: bool = True
    use_lpos: bool = True  # important link positions
    use_lvel: bool = False  # important link linear velocities


@dataclass
class ObsCfg:
    use_hand_obs: bool = True
    hand_cfg: RobotObsCfg = field(default_factory=lambda: RobotObsCfg())
    use_contact: bool = True  # binary tactile contact signal per link and axis
    use_policy_object_pcd: bool = True
    use_policy_hand_pcd: bool = True
    use_privileged_object_pcd: bool = True
    privileged_object_pcd_n_pts: int = 256
    policy_object_pcd_n_pts: int = 128
    policy_object_pcd_truncation_drop_ratio_range: tuple[float, float] = (0.0, 0.4)
    policy_object_pcd_jitter_std: float = 0.002
    use_object_obs: bool = True
    use_object_vel_obs: bool = False
    rot_format: str = "6d"  # 'quat' or 'euler' or '6d
    task_command: bool = True
    use_command_timestep: bool = (
        True  # if True, use the command timestep as input observation, the model can query env to get the actual commands from the timesteps
    )
    embed_commands_from_timestep: bool = (
        True  # if True and use_command_timestep is enabled, embed the normalized timestep scalar instead of the queried command values.
    )
    use_phase_index: bool = (
        False  # if True, expose the exported sequence phase index as a separate observation term.
    )
    mask_reference_hand_action_obs: bool = False


@dataclass
class CommandsCfg:
    use_gt_pos: bool = True  # use ground truth object position
    use_gt_rot: bool = True  # use ground truth object rotation
    rot_format: str = "6d"  # 'quat' or 'euler' or '6d
    use_gt_arti: bool = True  # use ground truth object articulation
    use_pcd: bool = False  # use point cloud as command input
    use_bps: bool = (
        False  # use bps representation for obs inputs, but the reward still uses the above pcd
    )
    bps_n_pts: int = 128  # number of bps points if using bps representation
    use_rel_bps: bool = True  # use object's bps representation relative to hands
    object_pcd_num_pts_sample: int = -1
    # pcd_n_pts: int = 128  # number of points in the point cloud command
    relative_commands: bool = (  # not implemented yet
        False  # whether to use relative commands (rel to start frame of the same sequence)
    )
    use_retargeted_motion: bool = False


@dataclass
class DiscObsCfg:
    # what observations to use for the discriminator
    hand_obj_cos_dist: bool = True
    hand_obj_dir_vec: bool = True
    delta_lpos: bool = True
    delta_hand_obj_rel_bps: bool = True
    reference_time_gaps: tuple = (
        (1, 0.4),
        (2, 0.3),
        (4, 0.2),
        (6, 0.1),
    )  # (time_gap, data_ratio)


@dataclass
class ActionCfg:
    action_mode: str = "diff"
    policy_variant: str = "default"
    trajectory_weighted_policy_sigma: float = 5.0
    trajectory_weighted_policy_timestep_dropout: float = 0.0
    residual_target_reference_mode: str = "retargeted"
    residual_beta: float = 1.0
    lock_wrist_targets_to_reference: bool = False
    arm_translation_delta_scale: float = 0.025
    arm_rotation_delta_scale: float = 0.1
    hand_joint_delta_scale: float = 0.2
    hand_prismatic_joint_delta_scale_factor: float = 0.1
    position_target_blend_alpha_start: float = 0.5
    position_target_blend_alpha_end: float = 1.0


@dataclass
class RewardTermScheduleCfg:
    scheme: str = "constant"
    value: float = 1.0
    reach_weight: float = 1.0
    transition_weight: float = 1.0
    lift_weight: float = 1.0
    exponential_rate: float = 5.0
    phase: str = "lift"
    start_frac: float = 0.0
    end_frac: float = 1.0
    min_value: float = 0.0


@dataclass
class RewardTermSchedulesCfg:
    object_tracking: RewardTermScheduleCfg = field(
        default_factory=lambda: RewardTermScheduleCfg(
            scheme="stage_levels",
            reach_weight=0.0,
            transition_weight=1.0,
            lift_weight=1.0,
        )
    )
    hand_cfg: RewardTermScheduleCfg = field(
        default_factory=lambda: RewardTermScheduleCfg(
            scheme="phase_window",
            phase="lift",
            start_frac=0.5,
            end_frac=1.0,
        )
    )
    hand_obj_rel: RewardTermScheduleCfg = field(
        default_factory=lambda: RewardTermScheduleCfg(
            scheme="phase_window",
            phase="lift",
            start_frac=0.5,
            end_frac=1.0,
        )
    )
    maniptrans: RewardTermScheduleCfg = field(default_factory=RewardTermScheduleCfg)
    retargeted_joint_mse: RewardTermScheduleCfg = field(
        default_factory=lambda: RewardTermScheduleCfg(
            scheme="reach_lift_exponential",
            exponential_rate=5.0,
        )
    )
    wrist_velocity: RewardTermScheduleCfg = field(
        default_factory=lambda: RewardTermScheduleCfg(
            scheme="phase_levels",
            reach_weight=1.0,
            lift_weight=1.0,
        )
    )
    joint_smoothness: RewardTermScheduleCfg = field(
        default_factory=lambda: RewardTermScheduleCfg(
            scheme="phase_levels",
            reach_weight=1.0,
            lift_weight=1.0,
        )
    )
    discriminator: RewardTermScheduleCfg = field(default_factory=RewardTermScheduleCfg)
    task_completion: RewardTermScheduleCfg = field(
        default_factory=RewardTermScheduleCfg
    )


@dataclass
class RewardCfgV2:
    use_retargeted_motion: bool = False
    log_metrics: bool = True
    log_essential_metrics_only: bool = False
    dense_task_weight: float = 0.2
    pcd_distance_metric: str = "bps"
    term_schedules: RewardTermSchedulesCfg = field(
        default_factory=RewardTermSchedulesCfg
    )
    hand_cfg_weight: float = 0.2
    hand_obj_rel_weight: float = 0.2
    hand_obj_rel_reference_mode: str = "all_phases"
    reach_hand_obj_rel_distance_completion_thd: float = 2.0
    reach_hand_obj_rel_direction_completion_thd: float = 3.0
    reach_hand_obj_rel_normal_completion_thd: float = 3.0
    reach_hand_obj_rel_distance_failure_thd: float = 20.0
    reach_hand_obj_rel_direction_failure_thd: float = 5.0
    reach_hand_obj_rel_normal_failure_thd: float = 5.0
    hand_obj_rel_distance_completion_thd: float = 1.0
    hand_obj_rel_direction_completion_thd: float = 2.0
    hand_obj_rel_normal_completion_thd: float = 2.0
    hand_obj_rel_distance_failure_thd: float = 5.0
    hand_obj_rel_direction_failure_thd: float = 5.0
    hand_obj_rel_normal_failure_thd: float = 5.0
    maniptrans_reward_weight: float = 0.0
    maniptrans_proximity_thd: float = 0.01
    maniptrans_lambda: float = 1.0
    task_completion_weight: float = 10.0
    last_reference_hold_steps: int = 10
    early_terminate: bool = True
    task_weight: float = 1.0
    discriminator_weight: float = 0.0
    completion_pos_thd: float = 0.01
    completion_rot_thd: float = 10 * pi / 180
    completion_arti_thd: float = 10 * pi / 180
    completion_pcd_thd: float = 0.01
    task_success_pos_thd: float = 0.03
    task_success_rot_thd: float = pi / 6
    task_success_prismatic_arti_thd: float = 0.02
    task_success_revolute_arti_thd: float = 10 * pi / 180
    hand_obj_max_dist_pos: float = 1.0
    obj_command_max_dist_pos: float = 0.15
    obj_command_max_dist_rot: float = 1.4
    obj_command_max_dist_arti: float = 1.57
    task_pos_scale: float = 10.0
    task_rot_scale: float = 5.0
    task_arti_scale: float = 5.0
    rest_task_pos_scale: float = 2.0
    rest_task_rot_scale: float = 1.0
    rest_task_arti_scale: float = 1.0
    task_pcd_scale: float = 1.0
    hand_cfg_norm_scale: float = 2.0
    hand_cfg_dir_scale: float = 2.0
    hand_obj_displacement_scale: float = 1.0
    hand_obj_normal_scale: float = 10.0
    retargeted_joint_mse_reward_weight: float = 0.0
    retargeted_joint_mse_reward_scale: float = 1.0
    retargeted_joint_mse_joint_mode: str = "wrist_and_fingers"
    retargeted_joint_mse_reference_noise: bool = False
    retargeted_joint_mse_wrist_pos_noise_std: float = 0.0
    retargeted_joint_mse_wrist_rot_noise_std: float = 0.0
    retargeted_joint_mse_finger_joint_noise_std: float = 0.0
    wrist_velocity_reward_weight: float = 0.0
    wrist_velocity_reward_scale: float = 1.0
    joint_smoothness_reward_weight: float = 0.05
    use_filtered_contact: bool = True


@dataclass
class LoggerConfig:
    project: str = "dexlink"
    runname: str = "default_run"
    logdir: str = str(get_logs_dir())
    group: str = "default_group"
    disable: bool = False
    seed: bool = True  # appended to runname if True
    timestamp: bool = True  # appends timestamp to runname if True
    # If either freq is <= 0, PPO computes it from desired counts below.
    log_freq: int = -1  # every N update steps (global_step)
    model_save_freq: int = -1  # every N update steps (global_step)
    desired_num_logs: int = 500  # target number of logs over full training
    desired_model_saves: int = 10  # target number of model saves over full training


@dataclass
class AgentConfig:
    # Placeholder agent config; will grow once algorithms are added
    resume_from_ckpt: str | None = (
        None  # path to checkpoint to resume from, if any, relative to logs dir
    )
    resume_start_timestep: int = -1
    resume_latest_fixed_budget: bool = False
    algo: str = "ppo"
    policy_lr: float = 5e-5
    value_lr: float = 5e-5
    gamma: float = 0.95
    lam: float = 0.95
    n_rollout_steps: int = 8  # small compared to horizon
    batch_size: int = 1024
    update_epochs: int = 4
    clip_ratio: float = 0.2
    entropy_coef: float = 1e-3
    normalize_advantages: bool = True
    # init_log_std: float = 0.1  # directly change in the code
    vf_coef: float = 0.5
    timesteps: int = 1_000_000  # number of train iterations
    history_len: int = 0  # use past n observations
    resume: bool = True  # resume from the latest checkpoint for the run if available
    seed: int = 42  # overriden by env.seed
    device: str = "cuda"  # overriden by env.device
    check_overheat: bool = False
    use_discriminator: bool = True  # overriden by env.use_discriminator
    # n_eval_episodes: int = 1000
    # networks: policy, value, and obs processor embed_dims
    policy_embed_dim: int = 256
    hand_synergy_dim: int = 256  # effectively no synergy here.
    value_embed_dim: int = 256
    obs_proc_embed_dim: int = 256
    obs_proc_layer_norm: bool = False
    obs_proc_pcd_hidden_dims: tuple[int, ...] = (256, 256)
    obs_proc_pcd_pool: str = "max"
    hand_obj_bps_aux_loss_coef: float = 0.1
    hand_obj_bps_aux_hidden_dims: tuple[int, ...] = (256,)
    policy_inputs: tuple[str, ...] = (
        "left",
        "ref_left",
        "right",
        "ref_right",
        "contact",
        "command",
        "object_pcd",
        # "hand_pcd",
    )
    critic_inputs: tuple[str, ...] = ("left", "right", "contact", "command", "object")
    # For stream-specific frequencies, 0 disables that stream and negative values
    # fall back to eval_freq (or model-save cadence when eval_freq <= 0).
    eval_freq: int = 0
    eval_metrics_freq: int = 0
    eval_video_freq: int = -1
    eval_episodes: int = 1
    num_eval_envs: int = (
        10  # all envs are processed, but information is logged only for eval_episodes
    )
    log_essential_metrics_only: bool = False


@dataclass
class DiscCfg:  # the discriminator (network) config
    hidden_dims: tuple[int, ...] = (64, 64)  # small network
    seq_len: int = 1
    memory_size: int = 8
    input_mask_prob: float = 0.8  # random input masking probability for disc obs
    lr: float = 3e-4
    weight_decay: float = 0.01  # strong regularization
    batch_size: int = 4096
    n_steps: int = (
        1  # number of discriminator update minibatches; +1 minibatch is used for eval
    )


@dataclass
class EvalConfig:
    num_additional_kpts: int = 200  # these are beside the mesh vertices
    relative_traj: bool = (
        True  # whether to use relative traj (rel to start frame of the same sequence)
    )
    logdir: str = "logs/eval"  # the log directory, overriden by logger.experiment_dir
    # number of (row, column) frame pairs evaluated per Chamfer batch
    chamfer_pairs_per_batch: int = 512
    # how many frames from the first trajectory to consider per chunk
    chamfer_row_chunk_size: int = 4


@dataclass
class RealEvalConfig:
    # unlike simulation, here we don't get the ground truth.
    # here we are doing independent policy rollouts, and
    # saving the camera recordings for each rollout appropriately in the logdir
    rollout_length: int = 200
    logdir: str = "real_eval"
    seed: int = 42
    initialization_only: bool = False
    single_step_only: bool = False
    command_substeps: int = 10
    print_command_timing: bool = False
    startup_timing_only: bool = False
    max_rollout_steps: int = 200


@dataclass
class RealControlTestConfig:
    logdir: str = str(get_logs_dir() / "real_control_tests")
    save_filename: str = "control_trace.npz"
    command_freq_hz: float = 10.0
    frame_stride: int = 1
    max_frames: int = -1


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    real_eval: RealEvalConfig = field(default_factory=RealEvalConfig)
    real_control_test: RealControlTestConfig = field(
        default_factory=RealControlTestConfig
    )
    preset: str = "default"


PRESETS: Dict[str, Dict[str, Any]] = {}
PRESETS.update(OAKINK_DEFAULT_PRESET)
PRESETS.update(SEQ_OAKINK_PRESETS)
PRESETS.update(SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS)
PRESETS.update(SEQ_OAKINK_METHOD_ABLATION_PRESETS)


# ---- Utilities ----


def _set_nested_attr(obj: Any, path: List[str], value: Any) -> None:
    if not path:
        raise ValueError("Expected a non-empty configuration path.")
    if len(path) == 1:
        if not hasattr(obj, path[0]):
            raise KeyError(
                f"Unknown config field {path[0]!r} on {type(obj).__name__}."
            )
        setattr(obj, path[0], value)
        return
    head = path[0]
    tail = path[1:]
    if not hasattr(obj, head):
        raise KeyError(f"Unknown config field {head!r} on {type(obj).__name__}.")
    child = getattr(obj, head)
    _set_nested_attr(child, tail, value)


def _cast_scalar(value_str: str, target_type: type) -> Any:
    if target_type is bool:
        normalized = value_str.lower()
        if normalized not in {"1", "0", "true", "false"}:
            raise ValueError(
                "Expected a boolean override encoded as true, false, 1, or 0; "
                f"got {value_str!r}."
            )
        return normalized in {"1", "true"}
    if target_type in {int, float, str}:
        return target_type(value_str)
    raise TypeError(f"Unsupported scalar override type {target_type!r}.")


def _cast_to_type(value_str: str, existing_value: Any, annotation: Any) -> Any:
    t = type(existing_value)
    if t in {bool, int, float, str}:
        return _cast_scalar(value_str, t)
    # basic list support: comma separated
    if isinstance(existing_value, (list, tuple)):
        parts = [p.strip() for p in value_str.split(",") if p.strip()]
        if not existing_value:
            raise ValueError(
                "Cannot infer the element type for an empty list or tuple override."
            )
        et = type(existing_value[0])
        converted = [et(part) for part in parts]
        return tuple(converted) if isinstance(existing_value, tuple) else converted
    if existing_value is None:
        if value_str.lower() == "none":
            return None
        candidate_types = [item for item in get_args(annotation) if item is not type(None)]
        if len(candidate_types) != 1 or candidate_types[0] not in {bool, int, float, str}:
            raise ValueError(
                "Cannot infer a single scalar type for this optional override. "
                f"Field annotation: {annotation!r}."
            )
        return _cast_scalar(value_str, candidate_types[0])
    raise TypeError(
        f"Unsupported override type {type(existing_value).__name__} for value "
        f"{existing_value!r}."
    )


def apply_overrides_from_dict(
    cfg: ExperimentConfig, overrides: Dict[str, Any]
) -> ExperimentConfig:
    # here we don't need to cast types, as the dict already has correct types
    for key, value in overrides.items():
        path = key.split(".")
        _set_nested_attr(cfg, path, value)
    return cfg


def apply_overrides(cfg: ExperimentConfig, overrides: List[str]) -> ExperimentConfig:
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override '{ov}' must be in key=val format")
        key, value_str = ov.split("=", 1)
        path = key.split(".")
        # Resolve current value for type casting
        cur = cfg
        for p in path[:-1]:
            cur = getattr(cur, p)
        if not hasattr(cur, path[-1]):
            raise KeyError(
                f"Unknown config field {path[-1]!r} on {type(cur).__name__}."
            )
        existing_value = getattr(cur, path[-1])
        annotation = get_type_hints(type(cur)).get(path[-1])
        value = _cast_to_type(value_str, existing_value, annotation)
        _set_nested_attr(cfg, path, value)
    return cfg


def apply_preset(cfg: ExperimentConfig, preset_name: str) -> ExperimentConfig:
    # import the preset from the current globals
    # preset = globals().get(preset_name, None)
    preset = PRESETS.get(preset_name, None)
    if preset is None:
        raise ValueError(f"Unknown preset '{preset_name}'.")
    return apply_overrides_from_dict(cfg, preset)
    # Convert preset dict items (flat with dot-notation) to overrides
    # overrides = [f"{k}={v}" for k, v in preset.items()]
    # print(f"Applying preset '{preset_name}': with {overrides}")
    # return apply_overrides(cfg, overrides)


def load_experiment_config(
    preset: str | None = None, overrides: List[str] | None = None
) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if preset and preset != "default":
        cfg = apply_preset(cfg, preset)
        cfg.preset = preset
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    if cfg.preset.startswith("seq_oakink"):
        cfg = apply_overrides_from_dict(
            cfg,
            required_seq_task_settings(
                cfg.env.task_cfg.oakink_exported_path,
                preset_name=cfg.preset,
            ),
        )
    return cfg


def _update_dataclass_from_dict(instance: Any, values: Dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown config key '{key}' for {type(instance).__name__}")
        current_value = getattr(instance, key)
        if is_dataclass(current_value) and isinstance(value, dict):
            _update_dataclass_from_dict(current_value, value)
        else:
            setattr(instance, key, value)


def load_experiment_config_from_dict(cfg_dict: Dict[str, Any]) -> ExperimentConfig:
    cfg = ExperimentConfig()
    _update_dataclass_from_dict(cfg, cfg_dict)
    return cfg


def config_to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    return asdict(cfg)


def print_cfg_summary(cfg: ExperimentConfig) -> None:
    import pprint

    cfg_dict = config_to_dict(cfg)
    pprint.pprint(cfg_dict)


def extract_overrides(namespace, unknown_tokens):
    """Extract dot-notation overrides from tokens like --env.num_envs 8.
    Returns list of strings: ['env.num_envs=8', ...]
    """
    overrides = []
    it = iter(unknown_tokens)
    for token in it:
        if not token.startswith("--"):
            continue
        key = token[2:]
        if "." not in key:
            # ignore non dot-notation unknowns
            continue
        try:
            value = next(it)
        except StopIteration:
            raise ValueError(f"Missing value for override '{token}'")
        overrides.append(f"{key}={value}")
    return overrides


__all__ = [
    "EnvConfig",
    "AgentConfig",
    "ExperimentConfig",
    "load_experiment_config",
    "config_to_dict",
    "print_cfg_summary",
    "extract_overrides",
]
