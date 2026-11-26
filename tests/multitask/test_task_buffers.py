import pytest
import torch

from dexlink.multitask_cup.config import PPOConfig
from dexlink.multitask_cup.learner import gae
from dexlink.multitask_cup.task_buffers import (
    TaskExploration,
    TaskResetStatistics,
    TaskRolloutBuffer,
)


def row(episode=0):
    return {
        "task_id": torch.tensor([0, 1]),
        "episode_id": torch.full((2,), episode, dtype=torch.long),
        "residual": torch.zeros(2, 18),
        "log_prob": torch.zeros(2),
        "reward": torch.ones(2),
        "value": torch.zeros(2),
        "next_value": torch.zeros(2),
        "terminated": torch.tensor([False, False]),
        "truncated": torch.tensor([False, False]),
    }


def test_buffer_consumed_once_and_rejects_stale():
    buffer = TaskRolloutBuffer(3, torch.tensor([0, 1]), 2)
    with pytest.raises(ValueError, match="stale"):
        buffer.append(2, row())
    buffer.append(3, row())
    with pytest.raises(ValueError, match="complete"):
        buffer.consume(3)
    buffer.append(3, row())
    assert buffer.consume(3)["residual"].shape == (2, 2, 18)
    assert not buffer.rows
    with pytest.raises(ValueError, match="unconsumed"):
        buffer.consume(3)


def test_buffer_episode_boundaries_and_task_identity():
    buffer = TaskRolloutBuffer(0, torch.tensor([0, 1]), 2)
    first = row()
    first["truncated"][0] = True
    buffer.append(0, first)
    with pytest.raises(ValueError, match="Episode"):
        buffer.append(0, row())
    second = row()
    second["episode_id"][0] = 1
    second["task_id"] = torch.tensor([1, 0])
    with pytest.raises(ValueError, match="slots"):
        buffer.append(0, second)


def test_gae_keeps_tasks_and_episodes_separate():
    rewards = torch.tensor([[1.0, 10.0], [100.0, 1000.0]])
    values = torch.zeros_like(rewards)
    terminal = torch.tensor([[True, False], [False, False]])
    truncated = torch.tensor([[False, True], [False, False]])
    next_values = torch.tensor([[99.0, 3.0], [0.0, 0.0]])
    advantages, _ = gae(
        rewards,
        values,
        next_values,
        terminal,
        truncated,
        PPOConfig(gamma=1.0, gae_lambda=1.0),
    )
    torch.testing.assert_close(advantages, torch.tensor([[1.0, 13.0], [100.0, 1000.0]]))


def test_reset_statistics_are_task_local_and_resume_rng():
    first = TaskResetStatistics(torch.tensor([3, 8]))
    second = TaskResetStatistics(torch.tensor([3, 8]))
    first.record(torch.tensor([0, 0, 0]), torch.tensor([1, 1, 2]))
    assert torch.equal(first.visits[1], second.visits[1])
    first.sample(0, 100)
    assert torch.equal(first.sample(1, 20), second.sample(1, 20))
    saved = first.state_dict()
    restored = TaskResetStatistics(torch.tensor([3, 8]))
    restored.load_state_dict(saved)
    assert torch.equal(first.sample(0, 100), restored.sample(0, 100))
    assert (restored.sample(0, 100) < 3).all()
    with pytest.raises(ValueError, match="range"):
        first.record(torch.tensor([0]), torch.tensor([3]))


def test_buffer_rejects_nonfinite_experience_and_invalid_slots():
    with pytest.raises(ValueError, match="positive"):
        TaskRolloutBuffer(0, torch.tensor([0]), 0)
    buffer = TaskRolloutBuffer(0, torch.tensor([0, 1]), 1)
    invalid = row()
    invalid["reward"][1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        buffer.append(0, invalid)


def test_reset_statistics_storage_is_cpu_with_non_cpu_default():
    lengths = torch.tensor([3, 8], device="cpu")
    with torch.device("meta"):
        statistics = TaskResetStatistics(lengths)
    assert statistics.visits.device.type == "cpu"


def test_exploration_is_invariant_to_task_grouping_and_resumes():
    from torch.distributions import Normal

    first, second = TaskExploration(3), TaskExploration(3)
    distribution = Normal(torch.zeros(4, 18), torch.ones(4, 18))
    joint = first.sample(distribution, torch.tensor([0, 0, 2, 2]))
    single = Normal(torch.zeros(2, 18), torch.ones(2, 18))
    second.sample(single, torch.tensor([1, 1]))
    separate = second.sample(single, torch.tensor([2, 2]))
    torch.testing.assert_close(joint[2:], separate)
    restored = TaskExploration(3)
    restored.load_state_dict(first.state_dict())
    labels = torch.tensor([0, 0, 2, 2])
    torch.testing.assert_close(
        first.sample(distribution, labels), restored.sample(distribution, labels)
    )
