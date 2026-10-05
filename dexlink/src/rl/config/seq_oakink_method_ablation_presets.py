"""One-factor method ablations for the Gate-0 sequential task suite."""

from __future__ import annotations

from typing import Any

from dexlink.src.rl.config.seq_oakink_final_experiment_presets import (
    OUR_METHOD_POLICY_SIZE_OVERRIDES,
    OUR_METHOD_PRESET_NAME,
)
from dexlink.src.rl.config.seq_oakink_presets import (
    SEQ_NO_REFERENCE_INPUT_PRESETS,
    SEQ_OAKINK_PRESETS,
)

METHOD_ABLATION_PRESET_PREFIX = "seq_oakink_v3_02"
METHOD_ABLATION_FULL_PRESET = f"{METHOD_ABLATION_PRESET_PREFIX}_full_method"
METHOD_ABLATION_FIXED_POLICY_PRESET = f"{METHOD_ABLATION_PRESET_PREFIX}_fixed_policy"
METHOD_ABLATION_CURRENT_JPOS_DIFF_PRESET = (
    f"{METHOD_ABLATION_PRESET_PREFIX}_current_jpos_diff"
)
METHOD_ABLATION_NO_TEMPLATE_REWARD_PRESET = (
    f"{METHOD_ABLATION_PRESET_PREFIX}_no_template_reward"
)
METHOD_ABLATION_FIRST_RESET_ONLY_PRESET = (
    f"{METHOD_ABLATION_PRESET_PREFIX}_first_reset_only"
)
METHOD_ABLATION_FIRST_RESET_ONLY_NO_EARLY_TERMINATION_PRESET = (
    f"{METHOD_ABLATION_PRESET_PREFIX}_first_reset_only_no_early_termination"
)
METHOD_ABLATION_NO_HAND_TRAJOPT_PRESET = (
    f"{METHOD_ABLATION_PRESET_PREFIX}_no_hand_trajopt"
)
SEQ_NO_REFERENCE_INPUT_PRESETS.add(METHOD_ABLATION_NO_HAND_TRAJOPT_PRESET)

# The fixed policy is parameter-matched once, on the 69-frame single-hand
# hammer task, against the full method's width-96 trajectory policy. The same
# hidden widths are then frozen for every Gate-0 task. Input and output layers
# still change when a task is bimanual.
FIXED_POLICY_SIZE_OVERRIDES = {
    "agent.policy_embed_dim": 250,
    "agent.hand_synergy_dim": 250,
    "agent.obs_proc_embed_dim": 64,
}


def _named_preset(
    *,
    name: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    preset = SEQ_OAKINK_PRESETS[OUR_METHOD_PRESET_NAME].copy()
    preset.update(
        {
            "preset": name,
            "logger.group": name,
            "logger.runname": name,
        }
    )
    preset.update(overrides)
    return preset


NO_TEMPLATE_REWARD_OVERRIDES = {
    "env.rewards_cfg.hand_cfg_weight": 0.0,
    "env.rewards_cfg.hand_obj_rel_weight": 0.0,
    "env.rewards_cfg.retargeted_joint_mse_reward_weight": 0.0,
    "env.rewards_cfg.wrist_velocity_reward_weight": 0.0,
}

FIRST_RESET_ONLY_OVERRIDES = {
    "env.reset_cfg.start_range": (0, 0),
    "env.reset_cfg.start_phase_probs": None,
    "env.reset_cfg.reverse_curriculum": False,
}

FIRST_RESET_ONLY_NO_EARLY_TERMINATION_OVERRIDES = {
    **FIRST_RESET_ONLY_OVERRIDES,
    "env.rewards_cfg.early_terminate": False,
}

NO_HAND_TRAJOPT_OVERRIDES = {
    # The actor receives no optimized robot-hand reference and predicts deltas
    # from the current configuration with a task-independent fixed-size MLP.
    "env.action_cfg.action_mode": "diff",
    "env.action_cfg.policy_variant": "default",
    "env.commands_cfg.use_retargeted_motion": False,
    "env.rewards_cfg.use_retargeted_motion": False,
    "env.obs_cfg.mask_reference_hand_action_obs": True,
    "env.task_cfg.require_complete_human_hand_reference": True,
    "env.reset_cfg.retargeted_hand_reference_mode": "default_configuration",
    **FIRST_RESET_ONLY_OVERRIDES,
    # Keep object tracking and the two human-template shaping terms. Explicitly
    # disable reward terms that require an optimized robot reference trajectory.
    "env.rewards_cfg.retargeted_joint_mse_reward_weight": 0.0,
    "env.rewards_cfg.wrist_velocity_reward_weight": 0.0,
    "agent.policy_inputs": ("right", "contact", "command", "object"),
    "agent.critic_inputs": ("right", "contact", "command", "object"),
    **FIXED_POLICY_SIZE_OVERRIDES,
}


SEQ_OAKINK_METHOD_ABLATION_PRESETS: dict[str, dict[str, Any]] = {
    METHOD_ABLATION_FULL_PRESET: _named_preset(
        name=METHOD_ABLATION_FULL_PRESET,
        overrides={},
    ),
    METHOD_ABLATION_FIXED_POLICY_PRESET: _named_preset(
        name=METHOD_ABLATION_FIXED_POLICY_PRESET,
        overrides={
            "env.action_cfg.action_mode": "absolute_target",
            "env.action_cfg.policy_variant": "fixed_residual_target_v1",
            "env.action_cfg.residual_target_reference_mode": "retargeted",
            **FIXED_POLICY_SIZE_OVERRIDES,
        },
    ),
    METHOD_ABLATION_CURRENT_JPOS_DIFF_PRESET: _named_preset(
        name=METHOD_ABLATION_CURRENT_JPOS_DIFF_PRESET,
        overrides={
            "env.action_cfg.action_mode": "diff",
            "env.action_cfg.policy_variant": "trajectory_weighted_v2",
            **OUR_METHOD_POLICY_SIZE_OVERRIDES,
        },
    ),
    METHOD_ABLATION_NO_TEMPLATE_REWARD_PRESET: _named_preset(
        name=METHOD_ABLATION_NO_TEMPLATE_REWARD_PRESET,
        overrides=NO_TEMPLATE_REWARD_OVERRIDES,
    ),
    METHOD_ABLATION_FIRST_RESET_ONLY_PRESET: _named_preset(
        name=METHOD_ABLATION_FIRST_RESET_ONLY_PRESET,
        overrides=FIRST_RESET_ONLY_OVERRIDES,
    ),
    METHOD_ABLATION_FIRST_RESET_ONLY_NO_EARLY_TERMINATION_PRESET: _named_preset(
        name=METHOD_ABLATION_FIRST_RESET_ONLY_NO_EARLY_TERMINATION_PRESET,
        overrides=FIRST_RESET_ONLY_NO_EARLY_TERMINATION_OVERRIDES,
    ),
    METHOD_ABLATION_NO_HAND_TRAJOPT_PRESET: _named_preset(
        name=METHOD_ABLATION_NO_HAND_TRAJOPT_PRESET,
        overrides=NO_HAND_TRAJOPT_OVERRIDES,
    ),
}
