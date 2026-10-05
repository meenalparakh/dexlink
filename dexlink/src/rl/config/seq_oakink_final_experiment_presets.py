"""Final-experiment sequential OakInk RL presets.

These variants are the named presets used by the current experiment generators:
the grasp-refinement recipe and the v2_07/v2_09 ablations.
"""

from __future__ import annotations

from math import pi
from typing import Any

from dexlink.src.rl.config.seq_oakink_presets import (
    SEQ_BIMANUAL_REQUIRED_SETTINGS,
    SEQ_OAKINK_PRESETS,
)

OUR_METHOD_PRESET_NAME = "seq_oakink_residual_target_trajectory_weighted_policy_v3"


def _named_preset(
    base_preset: dict[str, Any],
    *,
    name: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    preset = base_preset.copy()
    preset.update(
        {
            "preset": name,
            "logger.group": name,
            "logger.runname": name,
        }
    )
    preset.update(overrides)
    return preset


DEFAULT_SINGLE_HAND_POLICY_SIZE_OVERRIDES = {
    "agent.policy_embed_dim": 250,
    "agent.hand_synergy_dim": 250,
    "agent.obs_proc_embed_dim": 64,
}

DEFAULT_BIMANUAL_POLICY_SIZE_OVERRIDES = {
    **SEQ_BIMANUAL_REQUIRED_SETTINGS,
    "agent.policy_embed_dim": 286,
    "agent.hand_synergy_dim": 286,
    "agent.obs_proc_embed_dim": 64,
}

OUR_METHOD_POLICY_SIZE_OVERRIDES = {
    "agent.policy_embed_dim": 96,
    "agent.hand_synergy_dim": 96,
    "agent.obs_proc_embed_dim": 64,
}

OUR_METHOD_BIMANUAL_POLICY_SIZE_OVERRIDES = {
    **SEQ_BIMANUAL_REQUIRED_SETTINGS,
    "agent.policy_embed_dim": 96,
    "agent.hand_synergy_dim": 96,
    "agent.obs_proc_embed_dim": 64,
}

OBJECT_TRACKING_ONLY_OVERRIDES = {
    "env.action_cfg.action_mode": "absolute_target",
    "env.action_cfg.policy_variant": "trajectory_weighted_residual_target_v3",
    "env.action_cfg.residual_target_reference_mode": "initial_frame",
    "env.action_cfg.residual_beta": 1.0,
    "env.reset_cfg.retargeted_hand_reference_mode": "retargeted",
    "env.commands_cfg.use_retargeted_motion": False,
    "env.rewards_cfg.use_retargeted_motion": False,
    "env.obs_cfg.mask_reference_hand_action_obs": True,
    "env.reset_cfg.start_range": (0, 0),
    "env.reset_cfg.start_phase_probs": None,
    "env.reset_cfg.reverse_curriculum": False,
    "env.rewards_cfg.dense_task_weight": 1.0,
    "env.rewards_cfg.task_weight": 1.0,
    "env.rewards_cfg.task_completion_weight": 10.0,
    "env.rewards_cfg.joint_smoothness_reward_weight": 0.05,
    "env.rewards_cfg.hand_cfg_weight": 0.0,
    "env.rewards_cfg.hand_obj_rel_weight": 0.0,
    "env.rewards_cfg.maniptrans_reward_weight": 0.0,
    "env.rewards_cfg.discriminator_weight": 0.0,
    "env.rewards_cfg.retargeted_joint_mse_reward_weight": 0.2,
    "env.rewards_cfg.retargeted_joint_mse_joint_mode": "wrist_only",
    "env.rewards_cfg.retargeted_joint_mse_reference_noise": False,
    "env.rewards_cfg.wrist_velocity_reward_weight": 0.0,
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
    **OUR_METHOD_POLICY_SIZE_OVERRIDES,
}

OBJECT_TRACKING_ONLY_BIMANUAL_OVERRIDES = {
    **OBJECT_TRACKING_ONLY_OVERRIDES,
    **OUR_METHOD_BIMANUAL_POLICY_SIZE_OVERRIDES,
}

OBJECT_TRACKING_NOISY_RETARGETED_MSE_OVERRIDES = {
    **OBJECT_TRACKING_ONLY_OVERRIDES,
    "env.rewards_cfg.retargeted_joint_mse_joint_mode": "wrist_and_fingers",
    "env.rewards_cfg.retargeted_joint_mse_reference_noise": True,
    "env.rewards_cfg.retargeted_joint_mse_wrist_pos_noise_std": 0.03,
    "env.rewards_cfg.retargeted_joint_mse_wrist_rot_noise_std": 20 * pi / 180,
    "env.rewards_cfg.retargeted_joint_mse_finger_joint_noise_std": 20 * pi / 180,
}

OBJECT_TRACKING_RETARGETED_MSE_OVERRIDES = {
    **OBJECT_TRACKING_NOISY_RETARGETED_MSE_OVERRIDES,
    "env.rewards_cfg.retargeted_joint_mse_reference_noise": False,
    "env.rewards_cfg.retargeted_joint_mse_wrist_pos_noise_std": 0.0,
    "env.rewards_cfg.retargeted_joint_mse_wrist_rot_noise_std": 0.0,
    "env.rewards_cfg.retargeted_joint_mse_finger_joint_noise_std": 0.0,
}

OBJECT_TRACKING_RETARGETED_MSE_BIMANUAL_OVERRIDES = {
    **OBJECT_TRACKING_RETARGETED_MSE_OVERRIDES,
    **OUR_METHOD_BIMANUAL_POLICY_SIZE_OVERRIDES,
}

OBJECT_TRACKING_NOISY_RETARGETED_MSE_BIMANUAL_OVERRIDES = {
    **OBJECT_TRACKING_NOISY_RETARGETED_MSE_OVERRIDES,
    **OUR_METHOD_BIMANUAL_POLICY_SIZE_OVERRIDES,
}

OBJECT_TRACKING_WRIST_RESIDUAL_OVERRIDES = {
    **OBJECT_TRACKING_ONLY_OVERRIDES,
    "env.action_cfg.residual_target_reference_mode": (
        "retargeted_wrist_initial_fingers"
    ),
}

OBJECT_TRACKING_RETARGETED_MSE_WRIST_RESIDUAL_OVERRIDES = {
    **OBJECT_TRACKING_RETARGETED_MSE_OVERRIDES,
    "env.action_cfg.residual_target_reference_mode": (
        "retargeted_wrist_initial_fingers"
    ),
}

WRIST_CANONICAL_HAND_PRIOR_OVERRIDES = {
    "env.action_cfg.action_mode": "residual",
    "env.action_cfg.residual_beta": 0.5,
    "env.action_cfg.policy_variant": "default",
    "env.reset_cfg.retargeted_hand_reference_mode": "canonical_fingers",
    "env.action_cfg.lock_wrist_targets_to_reference": False,
    "env.rewards_cfg.hand_obj_rel_weight": 0.0,
    "env.rewards_cfg.retargeted_joint_mse_joint_mode": "wrist_only",
    **DEFAULT_SINGLE_HAND_POLICY_SIZE_OVERRIDES,
}

WRIST_CANONICAL_HAND_PRIOR_BIMANUAL_OVERRIDES = {
    **WRIST_CANONICAL_HAND_PRIOR_OVERRIDES,
    **DEFAULT_BIMANUAL_POLICY_SIZE_OVERRIDES,
}

WRIST_CANONICAL_HAND_PRIOR_BPS_OVERRIDES = {
    **WRIST_CANONICAL_HAND_PRIOR_OVERRIDES,
    "env.rewards_cfg.hand_obj_rel_weight": 0.2,
}

WRIST_CANONICAL_HAND_PRIOR_BPS_BIMANUAL_OVERRIDES = {
    **WRIST_CANONICAL_HAND_PRIOR_BIMANUAL_OVERRIDES,
    "env.rewards_cfg.hand_obj_rel_weight": 0.2,
}

NO_TEMPLATE_REWARD_OVERRIDES = {
    "env.rewards_cfg.hand_cfg_weight": 0.0,
    "env.rewards_cfg.hand_obj_rel_weight": 0.0,
}

UNIFORM_RESET_OVERRIDES = {
    "env.reset_cfg.reset_timestep_sampling_mode": "uniform",
}

FIRST_RESET_ONLY_OVERRIDES = {
    "env.reset_cfg.start_range": (0, 0),
    "env.reset_cfg.start_phase_probs": None,
    "env.reset_cfg.reverse_curriculum": False,
}

STANDARD_RESIDUAL_SINGLE_HAND_OVERRIDES = {
    "env.action_cfg.action_mode": "absolute_target",
    "env.action_cfg.residual_beta": 0.5,
    "env.action_cfg.policy_variant": "trajectory_mean_residual_target_v3",
    "env.action_cfg.residual_target_reference_mode": "retargeted",
    **OUR_METHOD_POLICY_SIZE_OVERRIDES,
}

STANDARD_RESIDUAL_BIMANUAL_OVERRIDES = {
    "env.action_cfg.action_mode": "absolute_target",
    "env.action_cfg.residual_beta": 0.5,
    "env.action_cfg.policy_variant": "trajectory_mean_residual_target_v3",
    "env.action_cfg.residual_target_reference_mode": "retargeted",
    **OUR_METHOD_BIMANUAL_POLICY_SIZE_OVERRIDES,
}


SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS: dict[str, dict[str, Any]] = {}

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS["seq_oakink_grasp_refinement"] = _named_preset(
    SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
    name="seq_oakink_grasp_refinement",
    overrides={
        "env.sim_cfg.gravity": 0.0,
        "env.sim_cfg.apply_random_force": True,
        "env.sim_cfg.random_force_mag": 1.0,
        "env.task_cfg.use_scene_support": False,
        "env.reset_cfg.start_range": (0, 1),
        "env.reset_cfg.dr": True,
        "env.reset_cfg.dr_object_mass": (0.01, 0.3),
        "env.reset_cfg.dr_contact_friction": (0.8, 1.2),
        "agent.timesteps": 10_000,
        "logger.desired_model_saves": 20,
        "logger.desired_num_logs": 250,
        "agent.eval_freq": 0,
        "agent.eval_metrics_freq": 0,
        "agent.eval_video_freq": -1,
    },
)

# Stability refinement starts from one collision-refined grasp and learns to keep
# the unsupported object at that pose under a fixed 1 N episode disturbance.
SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS[
    "seq_oakink_grasp_stability_refinement"
] = _named_preset(
    SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
    name="seq_oakink_grasp_stability_refinement",
    overrides={
        "env.task_cfg.demo_dataset": "oakink",
        "env.task_cfg.grasp_refinement_mode": True,
        "env.task_cfg.use_scene_support": False,
        "env.task_cfg.object_fixed": False,
        "env.task_cfg.save_reference_video": False,
        "env.task_cfg.start_timestep": 0,
        "env.task_cfg.end_timestep": -1,
        "env.task_cfg.skip": 1,
        "env.task_cfg.metric_cfg.metric_id": "grasp_stability",
        "env.task_cfg.metric_cfg.grasp_stability_object_id": "object",
        "env.task_cfg.metric_cfg.grasp_stability_max_drift_m": 0.01,
        "env.sim_cfg.gravity": 0.0,
        "env.sim_cfg.apply_random_force": True,
        "env.sim_cfg.random_force_mag": 1.0,
        "env.sim_cfg.video_save_freq": 0,
        "env.reset_cfg.start_range": (0, 0),
        "env.reset_cfg.start_phase_probs": None,
        "env.reset_cfg.obj_noise": 0.0,
        "env.reset_cfg.hand_noise_reach_phase": 0.0,
        "env.reset_cfg.hand_noise_lift_phase": 0.0,
        "env.reset_cfg.dr": False,
        "env.reset_cfg.reverse_curriculum": False,
        "env.curriculum_cfg.use_curriculum": False,
        "env.rewards_cfg.task_weight": 1.0,
        "env.rewards_cfg.dense_task_weight": 0.2,
        "env.rewards_cfg.early_terminate": False,
        "env.rewards_cfg.last_reference_hold_steps": 1,
        "agent.timesteps": 7_000,
        "logger.desired_model_saves": 10,
        "logger.desired_num_logs": 100,
        "agent.eval_freq": 0,
        "agent.eval_metrics_freq": 0,
        "agent.eval_video_freq": -1,
    },
)

# This short pilot isolates the three requested rewards while retaining the
# exact retargeted reset and the per-episode 1 N disturbance direction.
SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS[
    "seq_oakink_grasp_stability_refinement_random_force_2k"
] = _named_preset(
    SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS[
        "seq_oakink_grasp_stability_refinement"
    ],
    name="seq_oakink_grasp_stability_refinement_random_force_2k",
    overrides={
        "env.sim_cfg.gravity": 0.0,
        "env.sim_cfg.apply_random_force": True,
        "env.sim_cfg.random_force_mag": 1.0,
        "env.rewards_cfg.task_weight": 1.0,
        "env.rewards_cfg.dense_task_weight": 1.0,
        "env.rewards_cfg.hand_cfg_weight": 1.0,
        "env.rewards_cfg.hand_obj_rel_weight": 1.0,
        "env.rewards_cfg.task_completion_weight": 0.0,
        "env.rewards_cfg.maniptrans_reward_weight": 0.0,
        "env.rewards_cfg.retargeted_joint_mse_reward_weight": 0.0,
        "env.rewards_cfg.wrist_velocity_reward_weight": 0.0,
        "env.rewards_cfg.joint_smoothness_reward_weight": 0.0,
        "env.rewards_cfg.discriminator_weight": 0.0,
        "agent.timesteps": 2_000,
        "logger.desired_model_saves": 10,
        "logger.desired_num_logs": 100,
    },
)

# The selected physical refinement asks the hand to recover and hold the object
# against gravity. Success is evaluated from terminal drift by the task metric.
SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS[
    "seq_oakink_grasp_stability_refinement_gravity_terminal"
] = _named_preset(
    SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS[
        "seq_oakink_grasp_stability_refinement"
    ],
    name="seq_oakink_grasp_stability_refinement_gravity_terminal",
    overrides={
        "env.sim_cfg.gravity": -9.81,
        "env.sim_cfg.apply_random_force": False,
        "env.sim_cfg.random_force_mag": 0.0,
        "agent.timesteps": 5_000,
    },
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS["seq_oakink_v207_object_tracking_only"] = (
    _named_preset(
        SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
        name="seq_oakink_v207_object_tracking_only",
        overrides=OBJECT_TRACKING_ONLY_OVERRIDES,
    )
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS["seq_oakink_v207_object_tracking_only_bimanual"] = (
    _named_preset(
        SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
        name="seq_oakink_v207_object_tracking_only_bimanual",
        overrides=OBJECT_TRACKING_ONLY_BIMANUAL_OVERRIDES,
    )
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS[
    "seq_oakink_v207_object_tracking_noisy_retargeted_mse"
] = _named_preset(
    SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
    name="seq_oakink_v207_object_tracking_noisy_retargeted_mse",
    overrides=OBJECT_TRACKING_NOISY_RETARGETED_MSE_OVERRIDES,
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS[
    "seq_oakink_v207_object_tracking_retargeted_mse"
] = _named_preset(
    SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
    name="seq_oakink_v207_object_tracking_retargeted_mse",
    overrides=OBJECT_TRACKING_RETARGETED_MSE_OVERRIDES,
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS[
    "seq_oakink_v207_object_tracking_retargeted_mse_bimanual"
] = _named_preset(
    SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
    name="seq_oakink_v207_object_tracking_retargeted_mse_bimanual",
    overrides=OBJECT_TRACKING_RETARGETED_MSE_BIMANUAL_OVERRIDES,
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS[
    "seq_oakink_v207_object_tracking_noisy_retargeted_mse_bimanual"
] = _named_preset(
    SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
    name="seq_oakink_v207_object_tracking_noisy_retargeted_mse_bimanual",
    overrides=OBJECT_TRACKING_NOISY_RETARGETED_MSE_BIMANUAL_OVERRIDES,
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS[
    "seq_oakink_v207_object_tracking_wrist_residual"
] = _named_preset(
    SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
    name="seq_oakink_v207_object_tracking_wrist_residual",
    overrides=OBJECT_TRACKING_WRIST_RESIDUAL_OVERRIDES,
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS[
    "seq_oakink_v207_object_tracking_retargeted_mse_wrist_residual"
] = _named_preset(
    SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
    name="seq_oakink_v207_object_tracking_retargeted_mse_wrist_residual",
    overrides=OBJECT_TRACKING_RETARGETED_MSE_WRIST_RESIDUAL_OVERRIDES,
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS["seq_oakink_v207_wrist_canonical_hand_prior"] = (
    _named_preset(
        SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
        name="seq_oakink_v207_wrist_canonical_hand_prior",
        overrides=WRIST_CANONICAL_HAND_PRIOR_OVERRIDES,
    )
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS[
    "seq_oakink_v207_wrist_canonical_hand_prior_bimanual"
] = _named_preset(
    SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
    name="seq_oakink_v207_wrist_canonical_hand_prior_bimanual",
    overrides=WRIST_CANONICAL_HAND_PRIOR_BIMANUAL_OVERRIDES,
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS[
    "seq_oakink_v207_wrist_canonical_hand_prior_bps"
] = _named_preset(
    SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
    name="seq_oakink_v207_wrist_canonical_hand_prior_bps",
    overrides=WRIST_CANONICAL_HAND_PRIOR_BPS_OVERRIDES,
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS[
    "seq_oakink_v207_wrist_canonical_hand_prior_bps_bimanual"
] = _named_preset(
    SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
    name="seq_oakink_v207_wrist_canonical_hand_prior_bps_bimanual",
    overrides=WRIST_CANONICAL_HAND_PRIOR_BPS_BIMANUAL_OVERRIDES,
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS["seq_oakink_v209_no_template_reward"] = (
    _named_preset(
        SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
        name="seq_oakink_v209_no_template_reward",
        overrides=NO_TEMPLATE_REWARD_OVERRIDES,
    )
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS["seq_oakink_v209_uniform_reset"] = _named_preset(
    SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
    name="seq_oakink_v209_uniform_reset",
    overrides=UNIFORM_RESET_OVERRIDES,
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS["seq_oakink_v209_first_reset_only"] = _named_preset(
    SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
    name="seq_oakink_v209_first_reset_only",
    overrides=FIRST_RESET_ONLY_OVERRIDES,
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS["seq_oakink_v209_standard_residual_single_hand"] = (
    _named_preset(
        SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
        name="seq_oakink_v209_standard_residual_single_hand",
        overrides=STANDARD_RESIDUAL_SINGLE_HAND_OVERRIDES,
    )
)

SEQ_OAKINK_FINAL_EXPERIMENT_PRESETS["seq_oakink_v209_standard_residual_bimanual"] = (
    _named_preset(
        SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME],
        name="seq_oakink_v209_standard_residual_bimanual",
        overrides=STANDARD_RESIDUAL_BIMANUAL_OVERRIDES,
    )
)
