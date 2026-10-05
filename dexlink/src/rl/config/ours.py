from dexlink.utils.retargeting.train_utils import merge_configs

REWARD_COMMON = {
    "env.rewards_cfg.task_completion_weight": 5.0,
    "env.rewards_cfg.hand_cfg_weight": 0.15,
    "env.rewards_cfg.hand_obj_rel_weight": 0.35,
    "env.rewards_cfg.term_schedules.hand_obj_rel.scheme": "phase_window",
    "env.rewards_cfg.term_schedules.hand_obj_rel.phase": "lift",
    "env.rewards_cfg.term_schedules.hand_obj_rel.start_frac": 0.8,
    "env.rewards_cfg.term_schedules.hand_obj_rel.end_frac": 1.0,
    "env.rewards_cfg.dense_task_weight": 1.0,
    "env.rewards_cfg.term_schedules.joint_smoothness.scheme": "phase_levels",
    "env.rewards_cfg.term_schedules.joint_smoothness.reach_weight": 1.0,
    "env.rewards_cfg.term_schedules.joint_smoothness.lift_weight": 1.0,
    "env.rewards_cfg.joint_smoothness_reward_weight": 0.05,
    "env.rewards_cfg.completion_pos_thd": 0.005,
}

USE_RETARGETED_MOTION = {
    # directly use the hand motion ground truths instead of retargeted results.
    "env.commands_cfg.use_retargeted_motion": False,
    "env.rewards_cfg.use_retargeted_motion": False,
}

ACTION_COMMON = {
    "env.action_cfg.action_mode": "diff",
    "env.action_cfg.residual_beta": 1.0,
}

OURS = merge_configs(
    REWARD_COMMON,
    USE_RETARGETED_MOTION,
    ACTION_COMMON,
)
