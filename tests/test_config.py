from __future__ import annotations

import pytest

from dexlink.src.rl.config.base import load_experiment_config


def test_multi_object_environment_is_the_default() -> None:
    config = load_experiment_config()
    assert config.env.env_class == "multi_object"
    assert config.logger.project == "dexlink"


def test_full_method_public_preset_keeps_training_invariants() -> None:
    config = load_experiment_config("seq_oakink_v3_02_full_method")
    assert config.env.env_class == "multi_object"
    assert config.agent.normalize_advantages is True
    assert config.agent.policy_lr == pytest.approx(1.0e-4)
    assert config.agent.value_lr == pytest.approx(1.0e-4)


def test_config_overrides_are_strict_and_type_preserving() -> None:
    config = load_experiment_config(
        overrides=["env.sim_cfg.num_envs=8", "agent.normalize_advantages=false"]
    )
    assert config.env.sim_cfg.num_envs == 8
    assert isinstance(config.env.sim_cfg.num_envs, int)
    assert config.agent.normalize_advantages is False

    with pytest.raises(KeyError, match="unknown_field"):
        load_experiment_config(overrides=["agent.unknown_field=1"])
    with pytest.raises(ValueError, match="boolean override"):
        load_experiment_config(overrides=["agent.normalize_advantages=maybe"])
