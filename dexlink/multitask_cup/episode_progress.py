"""Pose-tracking diagnostics, independent of the frozen functional success tests."""

from __future__ import annotations

import torch

from .source_metrics import pose_within


def episode_tracking_progress(
    pose: torch.Tensor, reference: torch.Tensor
) -> dict[str, float]:
    """Score frames after initial-state reset; padding and final holds do not earn progress."""
    if reference.ndim != 2 or reference.shape[1] != 7 or len(reference) < 2:
        raise ValueError(
            f"Expected reference (frames >= 2,7), received {tuple(reference.shape)}."
        )
    if (
        pose.ndim != 3
        or pose.shape[2] != 7
        or pose.shape[0] < len(reference)
        or pose.shape[1] < 1
    ):
        raise ValueError(
            f"Expected pose (frames >= {len(reference)},episodes >= 1,7), received {tuple(pose.shape)}."
        )
    if not torch.isfinite(pose).all() or not torch.isfinite(reference).all():
        raise ValueError("Episode tracking requires finite poses and reference values.")
    tracked = pose_within(pose[1 : len(reference)], reference[1:, None])
    prefix = tracked.long().cumprod(dim=0).float().mean(dim=0)
    coverage = tracked.float().mean(dim=0)
    return {
        "episode_progress": prefix.mean().item(),
        "reference_tracking_fraction": coverage.mean().item(),
    }


def progress_metrics(
    results: list[dict], task_ids: tuple[str, ...]
) -> dict[str, float]:
    """Publish task-balanced means only for a complete, unique evaluation cohort."""
    if len(results) != len(task_ids) or {row["task_id"] for row in results} != set(
        task_ids
    ):
        raise ValueError(
            "Episode-progress aggregation requires every declared task exactly once."
        )
    metrics = {}
    for name in ("episode_progress", "reference_tracking_fraction"):
        values = torch.tensor([row[name] for row in results], dtype=torch.float64)
        if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f"Expected finite {name} fractions in [0,1].")
        for row in results:
            metrics[f"eval/task/{row['task_id']}/{name}"] = row[name]
        metrics[f"eval/{name}"] = values.mean().item()
    return metrics
