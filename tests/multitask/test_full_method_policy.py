"""Native-preset inheritance and the declared task-conditioning/local-window extension."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace
import io

import torch

from dexlink.src.rl.agent.agent_utils import ObservationProcessor
from dexlink.src.rl.config.base import load_experiment_config

from dexlink.multitask_cup.full_method_config import (
    ConditioningArchitecture,
    FULL_METHOD_PRESET,
    FullMethodRunConfig,
    task_configuration,
)
from dexlink.multitask_cup.full_method_policy import (
    FullMethodInputs,
    NativeObservationLayout,
    NativeTaskObservationProcessor,
    make_full_method_policy,
)
from .test_geometry_policy import example


LAYOUT = NativeObservationLayout(
    ("right", "ref_right", "contact", "command", "object"), (27, 21, 20, 1, 18)
)


def fixture():
    torch.manual_seed(42)
    _, old = example(point_count=32)
    method = FullMethodRunConfig().method()
    architecture = ConditioningArchitecture(point_count=32)
    model = make_full_method_policy(LAYOUT, method, architecture)
    raw = torch.randn(2, sum(LAYOUT.term_dims))
    raw[:, 68] = torch.tensor([0.0, 49.0])
    inputs = FullMethodInputs(
        native_obs=raw,
        normalized_reference_time=torch.tensor([0.0, 1.0]),
        **{
            name: getattr(old, name)
            for name in (
                "window",
                "window_mask",
                "geometry_points",
                "geometry_metadata",
                "geometry_index",
                "program_tokens",
                "program_mask",
                "program_index",
                "current_phase",
                "progress",
            )
        },
    )
    return model, inputs


def test_only_author_requested_resource_and_budget_fields_change_the_method() -> None:
    original = load_experiment_config(FULL_METHOD_PRESET)
    resolved = FullMethodRunConfig().method()
    resolved.agent.n_rollout_steps = original.agent.n_rollout_steps
    resolved.agent.timesteps = original.agent.timesteps
    resolved.env.sim_cfg.num_envs = original.env.sim_cfg.num_envs
    assert asdict(resolved) == asdict(original)
    task = task_configuration(
        FullMethodRunConfig(), {"reference": "approved-task.npy"}, 128, evaluation=False
    )
    assert task.env.reset_cfg == original.env.reset_cfg
    assert task.env.rewards_cfg == original.env.rewards_cfg
    assert task.env.action_cfg == original.env.action_cfg


def test_native_observation_embeddings_match_for_task_normalized_time() -> None:
    method = FullMethodRunConfig().method()
    current = NativeTaskObservationProcessor(LAYOUT, method, method.agent.policy_inputs)
    native_context = SimpleNamespace(
        cfg=method.env,
        observation_manager=LAYOUT,
        observation_space=SimpleNamespace(shape=(2, sum(LAYOUT.term_dims))),
        task_manager=SimpleNamespace(total_steps=50),
    )
    native = ObservationProcessor(
        native_context,
        embed_dim=method.agent.obs_proc_embed_dim,
        layer_norm=method.agent.obs_proc_layer_norm,
        selected_terms=method.agent.policy_inputs,
    )
    native.load_state_dict(current.state_dict(), strict=True)
    raw = torch.randn(2, sum(LAYOUT.term_dims))
    raw[:, 68] = torch.tensor([0.0, 49.0])
    torch.testing.assert_close(
        current(raw, torch.tensor([0.0, 1.0])), native(raw), rtol=0, atol=0
    )


def test_full_method_backbones_optimizer_and_exploration_are_inherited() -> None:
    model, inputs = fixture()
    distribution, values, weights = model(inputs)
    assert distribution.mean.shape == (2, 18) and values.shape == (2,)
    torch.testing.assert_close(
        distribution.stddev, torch.full((2, 18), torch.exp(torch.tensor(-1.0)))
    )
    assert model.actor.network.embed_dim == 96
    assert model.actor.network.hand_synergy_dim == 96
    assert model.critic.network.embed_dim == 96
    assert model.actor.sigma == 5.0
    optimizer = model.optimizer()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 1e-4]
    assert all(group["weight_decay"] == 0.01 for group in optimizer.param_groups)
    assert torch.equal(
        weights[~inputs.window_mask], torch.zeros_like(weights[~inputs.window_mask])
    )
    (distribution.mean.square().mean() + values.square().mean()).backward()
    for module in (
        model.actor.features.geometry,
        model.actor.features.program,
        model.actor.features.state_norm,
        model.critic.features.geometry,
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        )
        assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_reference_padding_cannot_change_the_actions_or_values() -> None:
    model, inputs = fixture()
    first = model(inputs)
    changed = inputs.window.clone()
    changed[~inputs.window_mask] = 1e6
    second = model(replace(inputs, window=changed))
    torch.testing.assert_close(first[0].mean, second[0].mean)
    torch.testing.assert_close(first[1], second[1])


def test_native_conditioned_checkpoint_and_adamw_reload_in_memory() -> None:
    model, inputs = fixture()
    optimizer = model.optimizer()
    distribution, values, _ = model(inputs)
    loss = (
        distribution.mean.square().mean()
        + values.square().mean()
        + distribution.entropy().mean()
    )
    loss.backward()
    optimizer.step()
    archive = io.BytesIO()
    torch.save(
        dict(
            policy=model.state_dict(),
            optimizer=optimizer.state_dict(),
            schema=model.checkpoint_schema(),
        ),
        archive,
    )
    archive.seek(0)
    saved = torch.load(archive, weights_only=True)
    restored = make_full_method_policy(LAYOUT, model.method, model.architecture)
    restored.load_state_dict(saved["policy"], strict=True)
    restored_optimizer = restored.optimizer()
    restored_optimizer.load_state_dict(saved["optimizer"])
    assert restored.checkpoint_schema() == saved["schema"]
    assert {value["step"].item() for value in restored_optimizer.state.values()} == {1}
    torch.testing.assert_close(model(inputs)[0].mean, restored(inputs)[0].mean)
