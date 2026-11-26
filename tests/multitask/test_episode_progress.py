"""Distinguish contiguous tracking, recovery coverage and elapsed reference clocks."""

from __future__ import annotations

import pytest
import torch

from dexlink.multitask_cup.episode_progress import (
    episode_tracking_progress,
    progress_metrics,
)
from dexlink.multitask_cup.task_buffers import BehaviorPolicyIdentity


def test_tracking_prefix_stops_at_first_error_but_coverage_can_recover() -> None:
    reference = torch.zeros((5, 7))
    reference[:, 3] = 1
    pose = reference[:, None].repeat(1, 2, 1)
    pose[3, 0, 0] = 0.04
    pose[1, 1, 0] = 0.04
    pose = torch.cat((pose, torch.zeros((10, 2, 7))))
    result = episode_tracking_progress(pose, reference)
    assert result["episode_progress"] == 0.25
    assert result["reference_tracking_fraction"] == 0.75


def test_initial_frame_and_padding_do_not_change_progress() -> None:
    reference = torch.zeros((5, 7))
    reference[:, 3] = 1
    pose = reference[:, None].clone()
    pose[0, :, 0] = 1
    assert episode_tracking_progress(pose, reference) == dict(
        episode_progress=1.0, reference_tracking_fraction=1.0
    )
    with pytest.raises(ValueError, match="pose"):
        episode_tracking_progress(pose[:3], reference)
    pose[2, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        episode_tracking_progress(pose, reference)


def test_task_balanced_progress_requires_complete_cohort() -> None:
    rows = [
        dict(task_id="a", episode_progress=0.2, reference_tracking_fraction=0.6),
        dict(task_id="b", episode_progress=0.8, reference_tracking_fraction=1.0),
    ]
    result = progress_metrics(rows, ("a", "b"))
    assert result["eval/episode_progress"] == 0.5
    assert result["eval/reference_tracking_fraction"] == 0.8
    assert result["eval/task/a/episode_progress"] == 0.2
    for invalid in (rows[:1], [rows[0], rows[0]]):
        with pytest.raises(ValueError, match="exactly once"):
            progress_metrics(invalid, ("a", "b"))


def test_rollout_identity_rejects_stale_changed_or_reused_policy() -> None:
    identity = BehaviorPolicyIdentity(4, "a" * 64)
    for version, digest in ((3, "a" * 64), (4, "b" * 64)):
        with pytest.raises(ValueError, match="current policy"):
            identity.consume(version, digest)
    identity.consume(4, "a" * 64)
    with pytest.raises(ValueError, match="unconsumed"):
        identity.consume(4, "a" * 64)
