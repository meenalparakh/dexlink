from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from dexlink.utils.retargeting.math_utils import axis_angle_from_quat, quat_box_minus


def _compute_projected_alpha(
    start_value: torch.Tensor,
    target_value: torch.Tensor,
    achieved_value: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    target_delta = target_value - start_value
    achieved_delta = achieved_value - start_value
    denom = torch.sum(target_delta * target_delta, dim=-1)
    numer = torch.sum(target_delta * achieved_delta, dim=-1)
    return torch.where(denom > eps, numer / denom, torch.zeros_like(numer))


def _compute_delta_ratio_metrics(
    start_value: torch.Tensor,
    target_value: torch.Tensor,
    achieved_value: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, float, float]:
    target_delta = target_value - start_value
    achieved_delta = achieved_value - start_value
    ratio = torch.full_like(target_delta, float("nan"))
    valid_mask = target_delta.abs() > eps
    ratio[valid_mask] = achieved_delta[valid_mask] / target_delta[valid_mask]

    finite_ratio = ratio[torch.isfinite(ratio)]
    mean_ratio = (
        float(finite_ratio.mean().item()) if finite_ratio.numel() > 0 else float("nan")
    )
    median_ratio = (
        float(finite_ratio.median().item())
        if finite_ratio.numel() > 0
        else float("nan")
    )
    return ratio, mean_ratio, median_ratio


def compute_action_achievement_metrics(
    start_qpos: torch.Tensor,
    target_qpos: torch.Tensor,
    achieved_qpos: torch.Tensor,
    start_hand_qpos: torch.Tensor,
    target_hand_qpos: torch.Tensor,
    achieved_hand_qpos: torch.Tensor,
    start_wrist_pos: torch.Tensor,
    target_wrist_pos: torch.Tensor,
    achieved_wrist_pos: torch.Tensor,
    start_wrist_quat: torch.Tensor,
    target_wrist_quat: torch.Tensor,
    achieved_wrist_quat: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, float | list[float]]:
    alpha_l2 = _compute_projected_alpha(
        start_value=start_qpos,
        target_value=target_qpos,
        achieved_value=achieved_qpos,
        eps=eps,
    )
    joint_ratio, mean_joint_ratio, median_joint_ratio = _compute_delta_ratio_metrics(
        start_value=start_qpos,
        target_value=target_qpos,
        achieved_value=achieved_qpos,
        eps=eps,
    )

    alpha_finger_l2 = _compute_projected_alpha(
        start_value=start_hand_qpos,
        target_value=target_hand_qpos,
        achieved_value=achieved_hand_qpos,
        eps=eps,
    )
    finger_ratio, mean_finger_ratio, median_finger_ratio = _compute_delta_ratio_metrics(
        start_value=start_hand_qpos,
        target_value=target_hand_qpos,
        achieved_value=achieved_hand_qpos,
        eps=eps,
    )

    alpha_wrist_pos_l2 = _compute_projected_alpha(
        start_value=start_wrist_pos,
        target_value=target_wrist_pos,
        achieved_value=achieved_wrist_pos,
        eps=eps,
    )

    target_wrist_rotvec = quat_box_minus(target_wrist_quat, start_wrist_quat)
    achieved_wrist_rotvec = quat_box_minus(achieved_wrist_quat, start_wrist_quat)
    alpha_wrist_rot_l2 = _compute_projected_alpha(
        start_value=torch.zeros_like(target_wrist_rotvec),
        target_value=target_wrist_rotvec,
        achieved_value=achieved_wrist_rotvec,
        eps=eps,
    )

    wrist_pos_error = torch.linalg.norm(target_wrist_pos - achieved_wrist_pos, dim=-1)
    wrist_rot_error = torch.linalg.norm(
        quat_box_minus(target_wrist_quat, achieved_wrist_quat), dim=-1
    )

    return {
        "alpha_l2": float(alpha_l2[0].item()),
        "alpha_finger_l2": float(alpha_finger_l2[0].item()),
        "alpha_wrist_pos_l2": float(alpha_wrist_pos_l2[0].item()),
        "alpha_wrist_rot_l2": float(alpha_wrist_rot_l2[0].item()),
        "mean_joint_ratio": mean_joint_ratio,
        "median_joint_ratio": median_joint_ratio,
        "per_joint_ratio": joint_ratio[0].detach().cpu().tolist(),
        "mean_finger_ratio": mean_finger_ratio,
        "median_finger_ratio": median_finger_ratio,
        "per_finger_ratio": finger_ratio[0].detach().cpu().tolist(),
        "wrist_pos_error": float(wrist_pos_error[0].item()),
        "wrist_rot_error": float(wrist_rot_error[0].item()),
    }


def save_action_tracking_plot(
    episode_idx: int,
    joint_names: list[str],
    target_qpos: torch.Tensor,
    achieved_qpos: torch.Tensor,
    plot_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    n_joints = len(joint_names)
    n_cols = 3
    n_rows = math.ceil(n_joints / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 2.8 * n_rows),
        sharex=True,
    )
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    x = np.arange(target_qpos.shape[0])
    target_np = target_qpos.detach().cpu().numpy()
    achieved_np = achieved_qpos.detach().cpu().numpy()

    for joint_idx, joint_name in enumerate(joint_names):
        ax = axes[joint_idx // n_cols, joint_idx % n_cols]
        ax.plot(x, target_np[:, joint_idx], label="target", linewidth=1.5)
        ax.plot(x, achieved_np[:, joint_idx], label="achieved", linewidth=1.2)
        ax.set_title(joint_name)
        ax.grid(True, alpha=0.3)
        if joint_idx // n_cols == n_rows - 1:
            ax.set_xlabel("Env Step")
        ax.set_ylabel("Joint Pos")

    for unused_idx in range(n_joints, n_rows * n_cols):
        axes[unused_idx // n_cols, unused_idx % n_cols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"Episode {episode_idx} Action Targets vs Achieved Joint Positions")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def save_wrist_tracking_plot(
    episode_idx: int,
    target_wrist_pos: torch.Tensor,
    achieved_wrist_pos: torch.Tensor,
    target_wrist_quat: torch.Tensor,
    achieved_wrist_quat: torch.Tensor,
    plot_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    wrist_rot_labels = ["rot_x", "rot_y", "rot_z"]
    wrist_pos_labels = ["pos_x", "pos_y", "pos_z"]
    target_wrist_rotvec = axis_angle_from_quat(target_wrist_quat)
    achieved_wrist_rotvec = axis_angle_from_quat(achieved_wrist_quat)

    fig, axes = plt.subplots(2, 3, figsize=(18, 6), sharex=True)
    x = np.arange(target_wrist_pos.shape[0])
    target_wrist_pos_np = target_wrist_pos.detach().cpu().numpy()
    achieved_wrist_pos_np = achieved_wrist_pos.detach().cpu().numpy()
    target_wrist_rotvec_np = target_wrist_rotvec.detach().cpu().numpy()
    achieved_wrist_rotvec_np = achieved_wrist_rotvec.detach().cpu().numpy()

    for axis_idx, axis_name in enumerate(wrist_pos_labels):
        ax = axes[0, axis_idx]
        ax.plot(x, target_wrist_pos_np[:, axis_idx], label="target", linewidth=1.5)
        ax.plot(
            x,
            achieved_wrist_pos_np[:, axis_idx],
            label="achieved",
            linewidth=1.2,
        )
        ax.set_title(axis_name)
        ax.set_ylabel("Meters")
        ax.grid(True, alpha=0.3)

    for axis_idx, axis_name in enumerate(wrist_rot_labels):
        ax = axes[1, axis_idx]
        ax.plot(
            x,
            target_wrist_rotvec_np[:, axis_idx],
            label="target",
            linewidth=1.5,
        )
        ax.plot(
            x,
            achieved_wrist_rotvec_np[:, axis_idx],
            label="achieved",
            linewidth=1.2,
        )
        ax.set_title(axis_name)
        ax.set_xlabel("Env Step")
        ax.set_ylabel("Radians")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"Episode {episode_idx} Wrist Pose Targets vs Achieved")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def finalize_action_tracking_episode(
    *,
    episode_idx: int,
    episode_dir: Path,
    episode_action_traces: list[dict],
) -> dict[str, float | int | str]:
    if len(episode_action_traces) == 0:
        raise ValueError("Expected at least one action trace to finalize an episode.")

    joint_names = list(episode_action_traces[0]["joint_names"])
    target_qpos = torch.stack(
        [trace["target_qpos"] for trace in episode_action_traces], dim=0
    )
    achieved_qpos = torch.stack(
        [trace["achieved_qpos"] for trace in episode_action_traces], dim=0
    )
    start_qpos = torch.stack(
        [trace["start_qpos"] for trace in episode_action_traces], dim=0
    )
    hand_target_indices = torch.as_tensor(
        episode_action_traces[0]["hand_target_indices"],
        dtype=torch.long,
        device=start_qpos.device,
    )
    start_wrist_pos = torch.stack(
        [trace["start_wrist_pos"] for trace in episode_action_traces], dim=0
    )
    target_wrist_pos = torch.stack(
        [trace["target_wrist_pos"] for trace in episode_action_traces], dim=0
    )
    achieved_wrist_pos = torch.stack(
        [trace["achieved_wrist_pos"] for trace in episode_action_traces], dim=0
    )
    start_wrist_quat = torch.stack(
        [trace["start_wrist_quat"] for trace in episode_action_traces], dim=0
    )
    target_wrist_quat = torch.stack(
        [trace["target_wrist_quat"] for trace in episode_action_traces], dim=0
    )
    achieved_wrist_quat = torch.stack(
        [trace["achieved_wrist_quat"] for trace in episode_action_traces], dim=0
    )
    start_hand_qpos = start_qpos.index_select(1, hand_target_indices)
    target_hand_qpos = target_qpos.index_select(1, hand_target_indices)
    achieved_hand_qpos = achieved_qpos.index_select(1, hand_target_indices)

    step_metrics = [
        compute_action_achievement_metrics(
            start_qpos=start_qpos[step_idx].unsqueeze(0),
            target_qpos=target_qpos[step_idx].unsqueeze(0),
            achieved_qpos=achieved_qpos[step_idx].unsqueeze(0),
            start_hand_qpos=start_hand_qpos[step_idx].unsqueeze(0),
            target_hand_qpos=target_hand_qpos[step_idx].unsqueeze(0),
            achieved_hand_qpos=achieved_hand_qpos[step_idx].unsqueeze(0),
            start_wrist_pos=start_wrist_pos[step_idx].unsqueeze(0),
            target_wrist_pos=target_wrist_pos[step_idx].unsqueeze(0),
            achieved_wrist_pos=achieved_wrist_pos[step_idx].unsqueeze(0),
            start_wrist_quat=start_wrist_quat[step_idx].unsqueeze(0),
            target_wrist_quat=target_wrist_quat[step_idx].unsqueeze(0),
            achieved_wrist_quat=achieved_wrist_quat[step_idx].unsqueeze(0),
        )
        for step_idx in range(len(episode_action_traces))
    ]

    alpha_l2_values = [float(metric["alpha_l2"]) for metric in step_metrics]
    alpha_finger_l2_values = [
        float(metric["alpha_finger_l2"]) for metric in step_metrics
    ]
    alpha_wrist_pos_l2_values = [
        float(metric["alpha_wrist_pos_l2"]) for metric in step_metrics
    ]
    alpha_wrist_rot_l2_values = [
        float(metric["alpha_wrist_rot_l2"]) for metric in step_metrics
    ]
    mean_joint_ratio_values = [
        float(metric["mean_joint_ratio"])
        for metric in step_metrics
        if math.isfinite(float(metric["mean_joint_ratio"]))
    ]
    median_joint_ratio_values = [
        float(metric["median_joint_ratio"])
        for metric in step_metrics
        if math.isfinite(float(metric["median_joint_ratio"]))
    ]
    mean_finger_ratio_values = [
        float(metric["mean_finger_ratio"])
        for metric in step_metrics
        if math.isfinite(float(metric["mean_finger_ratio"]))
    ]
    median_finger_ratio_values = [
        float(metric["median_finger_ratio"])
        for metric in step_metrics
        if math.isfinite(float(metric["median_finger_ratio"]))
    ]
    wrist_pos_error_values = [
        float(metric["wrist_pos_error"]) for metric in step_metrics
    ]
    wrist_rot_error_values = [
        float(metric["wrist_rot_error"]) for metric in step_metrics
    ]

    plot_path = episode_dir / "joint_targets_vs_achieved.png"
    wrist_plot_path = episode_dir / "wrist_pose_targets_vs_achieved.png"
    trace_path = episode_dir / "action_tracking.pt"
    summary_path = episode_dir / "action_tracking_summary.json"
    save_action_tracking_plot(
        episode_idx=episode_idx,
        joint_names=joint_names,
        target_qpos=target_qpos,
        achieved_qpos=achieved_qpos,
        plot_path=plot_path,
    )
    save_wrist_tracking_plot(
        episode_idx=episode_idx,
        target_wrist_pos=target_wrist_pos,
        achieved_wrist_pos=achieved_wrist_pos,
        target_wrist_quat=target_wrist_quat,
        achieved_wrist_quat=achieved_wrist_quat,
        plot_path=wrist_plot_path,
    )

    trace_payload = {
        "joint_names": joint_names,
        "target_qpos": target_qpos,
        "achieved_qpos": achieved_qpos,
        "start_qpos": start_qpos,
        "start_wrist_pos": start_wrist_pos,
        "target_wrist_pos": target_wrist_pos,
        "achieved_wrist_pos": achieved_wrist_pos,
        "start_wrist_quat": start_wrist_quat,
        "target_wrist_quat": target_wrist_quat,
        "achieved_wrist_quat": achieved_wrist_quat,
        "step_metrics": step_metrics,
        "alpha_schedule": [trace["alpha_schedule"] for trace in episode_action_traces],
        "actions": torch.stack(
            [trace["action"] for trace in episode_action_traces], dim=0
        ),
    }
    torch.save(trace_payload, trace_path)

    summary = {
        "episode_idx": int(episode_idx),
        "num_env_steps": int(len(episode_action_traces)),
        "mean_alpha_l2": float(np.mean(alpha_l2_values)),
        "median_alpha_l2": float(np.median(alpha_l2_values)),
        "mean_alpha_finger_l2": float(np.mean(alpha_finger_l2_values)),
        "median_alpha_finger_l2": float(np.median(alpha_finger_l2_values)),
        "mean_alpha_wrist_pos_l2": float(np.mean(alpha_wrist_pos_l2_values)),
        "median_alpha_wrist_pos_l2": float(np.median(alpha_wrist_pos_l2_values)),
        "mean_alpha_wrist_rot_l2": float(np.mean(alpha_wrist_rot_l2_values)),
        "median_alpha_wrist_rot_l2": float(np.median(alpha_wrist_rot_l2_values)),
        "mean_joint_ratio": (
            float(np.mean(mean_joint_ratio_values))
            if len(mean_joint_ratio_values) > 0
            else float("nan")
        ),
        "median_joint_ratio": (
            float(np.mean(median_joint_ratio_values))
            if len(median_joint_ratio_values) > 0
            else float("nan")
        ),
        "mean_finger_ratio": (
            float(np.mean(mean_finger_ratio_values))
            if len(mean_finger_ratio_values) > 0
            else float("nan")
        ),
        "median_finger_ratio": (
            float(np.mean(median_finger_ratio_values))
            if len(median_finger_ratio_values) > 0
            else float("nan")
        ),
        "mean_wrist_pos_error": float(np.mean(wrist_pos_error_values)),
        "mean_wrist_rot_error": float(np.mean(wrist_rot_error_values)),
        "plot_path": str(plot_path),
        "wrist_plot_path": str(wrist_plot_path),
        "trace_path": str(trace_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
