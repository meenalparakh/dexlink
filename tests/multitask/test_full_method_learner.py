"""Uneven-rank gradients must equal the exact native global PPO objective."""

from __future__ import annotations

import torch

from dexlink.multitask_cup.full_method_config import FullMethodRunConfig
from dexlink.multitask_cup.full_method_learner import (
    PPOMinibatch,
    local_global_minibatches,
    native_ppo_loss,
    standardized_advantages,
)


def test_uneven_eight_rank_loss_matches_native_global_gradient() -> None:
    torch.manual_seed(42)
    agent = FullMethodRunConfig().method().agent
    counts = (40, 40, 40, 40, 40, 40, 30, 30)
    size = sum(counts)
    features = torch.randn(size, 3, dtype=torch.float64)
    advantages = torch.randn(size, dtype=torch.float64)
    returns = torch.randn(size, dtype=torch.float64)
    old = torch.zeros(size, dtype=torch.float64)
    parameter = torch.tensor(
        [0.02, 0.03, -0.04], dtype=torch.float64, requires_grad=True
    )

    def data(selected: torch.Tensor) -> PPOMinibatch:
        value = features[selected] @ parameter
        return PPOMinibatch(
            value,
            old[selected],
            advantages[selected],
            value.square(),
            returns[selected],
            value.exp(),
        )

    whole, _ = native_ppo_loss(data(torch.arange(size)), agent, 1, size)
    reference = torch.autograd.grad(whole, parameter, retain_graph=True)[0]
    gradients = []
    start = 0
    for count in counts:
        loss, _ = native_ppo_loss(
            data(torch.arange(start, start + count)), agent, 8, size
        )
        gradients.append(torch.autograd.grad(loss, parameter, retain_graph=True)[0])
        start += count
    torch.testing.assert_close(
        torch.stack(gradients).mean(0), reference, rtol=1e-12, atol=1e-12
    )


def test_global_minibatch_partition_is_exact_with_uneven_ranks() -> None:
    counts = (4096, 4096, 4096, 4096, 4096, 4096, 3072, 3072)
    torch.manual_seed(42)
    order = torch.randperm(sum(counts))
    pieces = [
        list(local_global_minibatches(order, counts, rank, 1024)) for rank in range(8)
    ]
    assert len(pieces[0]) == sum(counts) // 1024
    for step in range(len(pieces[0])):
        reconstructed = torch.cat(
            [pieces[rank][step][0] + sum(counts[:rank]) for rank in range(8)]
        )
        torch.testing.assert_close(
            reconstructed.sort().values,
            order[step * 1024 : (step + 1) * 1024].sort().values,
        )
    for rank in range(8):
        torch.testing.assert_close(
            torch.cat([indices for indices, _ in pieces[rank]]).sort().values,
            torch.arange(counts[rank]),
        )


def test_advantage_standardization_matches_native_unbiased_std() -> None:
    torch.manual_seed(42)
    values = torch.randn(40, dtype=torch.float64)
    moments = torch.stack(
        (values.sum(), values.square().sum(), values.new_tensor(len(values)))
    )
    torch.testing.assert_close(
        standardized_advantages(values, moments),
        (values - values.mean()) / (values.std() + 1e-8),
    )
