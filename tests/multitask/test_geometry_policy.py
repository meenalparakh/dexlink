"""Geometry, program and local-window invariants for the proposed v2 architecture."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from dexlink.multitask_cup.config import GeometryPolicyConfig
from dexlink.multitask_cup.geometry_policy import (
    GeometryPolicyInputs,
    GeometryPolicySchema,
    make_geometry_policy,
)
from dexlink.multitask_cup.policy import local_window


def example(phases: int = 6, length: int = 50, point_count: int = 32):
    torch.manual_seed(42)
    config = GeometryPolicyConfig(point_count=point_count)
    model = make_geometry_policy(GeometryPolicySchema(), config)
    normals = torch.nn.functional.normalize(torch.randn(2, point_count, 3), dim=-1)
    points = torch.cat((torch.rand(2, point_count, 3) * 0.12 - 0.06, normals), -1)
    window, mask = local_window(torch.randn(length, 30), torch.tensor([0, length - 1]))
    inputs = GeometryPolicyInputs(
        state=torch.randn(2, 67),
        window=window,
        window_mask=mask,
        geometry_points=points,
        geometry_metadata=torch.tensor(
            [[0.12, 0.12, 0.12, 0, 0, 0, 0.1, 0, 0, 0, 0.001, 0.001, 0.001, 0, 0, 0]]
        )
        .expand(2, -1)
        .clone(),
        geometry_index=torch.tensor([0, 1]),
        program_tokens=torch.randn(2, phases, 32),
        program_mask=torch.ones(2, phases, dtype=torch.bool),
        program_index=torch.tensor([0, 1]),
        current_phase=torch.zeros(2, dtype=torch.long),
        progress=torch.zeros(2, 4),
    )
    return model, inputs


def test_default_full_geometry_and_long_program_forward() -> None:
    model, inputs = example(phases=12, length=124, point_count=512)
    distribution, value, _ = model(inputs)
    assert torch.isfinite(distribution.mean).all() and torch.isfinite(value).all()
    assert model.checkpoint_schema()["model"]["point_count"] == 512


def test_geometry_is_point_order_invariant_but_retains_shape_and_physical_size() -> (
    None
):
    model, inputs = example()
    encoder = model.geometry_encoder
    baseline = encoder(inputs.geometry_points, inputs.geometry_metadata)
    permuted = encoder(
        inputs.geometry_points[:, torch.randperm(32)], inputs.geometry_metadata
    )
    torch.testing.assert_close(baseline, permuted)
    changed = inputs.geometry_points.clone()
    changed[..., 0] *= 0.5
    assert not torch.allclose(baseline, encoder(changed, inputs.geometry_metadata))
    scaled = inputs.geometry_points.clone()
    scaled[..., :3] *= 2
    metadata = inputs.geometry_metadata.clone()
    metadata[:, :6] *= 2
    assert not torch.allclose(baseline, encoder(scaled, metadata))
    metadata = inputs.geometry_metadata.clone()
    metadata[:, 6] *= 2
    assert not torch.allclose(baseline, encoder(inputs.geometry_points, metadata))


def test_geometry_program_and_both_modulators_affect_learned_actions() -> None:
    model, inputs = example()
    distribution, value, _ = model(inputs)
    swapped = model(replace(inputs, geometry_index=inputs.geometry_index.flip(0)))[0]
    assert not torch.allclose(distribution.mean, swapped.mean)
    changed = model(replace(inputs, program_index=inputs.program_index.flip(0)))[0]
    assert not torch.allclose(distribution.mean, changed.mean)
    (distribution.mean.square().mean() + value.square().mean()).backward()
    for module in (
        model.geometry_encoder,
        model.program_encoder,
        model.state_norm.modulation,
        model.reference_norm.modulation,
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        )
        assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_program_order_matters_and_padding_cannot_change_actions() -> None:
    model, inputs = example()
    baseline = model(inputs)[0].mean
    assert not torch.allclose(
        baseline,
        model(replace(inputs, program_tokens=inputs.program_tokens.flip(1)))[0].mean,
    )
    mask = inputs.program_mask.clone()
    mask[:, -2:] = False
    inputs = replace(inputs, program_mask=mask)
    baseline = model(inputs)[0].mean
    program = inputs.program_tokens.clone()
    program[~mask] = 1e6
    window = inputs.window.clone()
    window[~inputs.window_mask] = 1e6
    torch.testing.assert_close(
        baseline, model(replace(inputs, program_tokens=program, window=window))[0].mean
    )


@pytest.mark.parametrize(
    ("phases", "length"), [(6, 44), (8, 79), (10, 99), (11, 124), (12, 119)]
)
def test_variable_program_and_trajectory_lengths_keep_parameter_shapes(
    phases: int, length: int
) -> None:
    model, inputs = example(phases, length)
    distribution, value, weights = model(inputs)
    assert distribution.mean.shape == (2, 18) and value.shape == (2,)
    assert (weights[~inputs.window_mask] == 0).all()
    residual = distribution.sample()
    torch.testing.assert_close(
        model.log_prob(inputs, residual), distribution.log_prob(residual).sum(-1)
    )
    baseline, _ = example()
    assert [
        (name, tuple(parameter.shape)) for name, parameter in model.named_parameters()
    ] == [
        (name, tuple(parameter.shape))
        for name, parameter in baseline.named_parameters()
    ]


def test_bad_geometry_and_program_selection_are_rejected() -> None:
    model, inputs = example()
    with pytest.raises(ValueError, match="geometry points"):
        model(replace(inputs, geometry_points=inputs.geometry_points[:, :8]))
    mask = inputs.program_mask.clone()
    mask[:, 0] = False
    with pytest.raises(ValueError, match="valid program token"):
        model(replace(inputs, program_mask=mask))
    with pytest.raises(ValueError, match="geometry_index"):
        model(replace(inputs, geometry_index=torch.tensor([0, 2])))
    metadata = inputs.geometry_metadata.clone()
    metadata[:, 10] = -0.001
    with pytest.raises(ValueError, match="positive-definite"):
        model(replace(inputs, geometry_metadata=metadata))


def test_factory_keeps_geometry_policy_on_cpu_after_simulator_default_changes() -> None:
    baseline, inputs = example()
    with torch.device("meta"):
        model = make_geometry_policy(baseline.schema, baseline.config)
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        assert model(inputs)[0].mean.device.type == "cpu"
