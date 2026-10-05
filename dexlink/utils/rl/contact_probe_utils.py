from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch


def save_contact_tracking_plot(
    episode_idx: int,
    finger_names: list[str],
    per_finger_contact: torch.Tensor,
    plot_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    n_fingers = len(finger_names)
    n_cols = 3
    n_rows = math.ceil(n_fingers / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 2.6 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    x = np.arange(per_finger_contact.shape[0])
    per_finger_contact_np = per_finger_contact.detach().cpu().numpy()

    for finger_idx, finger_name in enumerate(finger_names):
        ax = axes[finger_idx // n_cols, finger_idx % n_cols]
        contact = per_finger_contact_np[:, finger_idx]
        ax.step(x, contact, where="post", linewidth=1.5)
        ax.fill_between(x, 0.0, contact, step="post", alpha=0.25)
        ax.set_title(finger_name)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 1.0])
        ax.grid(True, alpha=0.3)
        if finger_idx // n_cols == n_rows - 1:
            ax.set_xlabel("Env Step")
        ax.set_ylabel("Contact")

    for unused_idx in range(n_fingers, n_rows * n_cols):
        axes[unused_idx // n_cols, unused_idx % n_cols].axis("off")

    fig.suptitle(f"Episode {episode_idx} Contact Signals Per Finger")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def save_contact_axis_tracking_plot(
    episode_idx: int,
    finger_names: list[str],
    per_finger_contact_xyz: torch.Tensor,
    plot_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if per_finger_contact_xyz.ndim != 3:
        raise ValueError(
            "Expected per_finger_contact_xyz to have shape (num_steps, num_fingers, 3), "
            f"got {tuple(per_finger_contact_xyz.shape)}."
        )
    if per_finger_contact_xyz.shape[2] != 3:
        raise ValueError(
            "Expected per_finger_contact_xyz last dimension to be 3 (x, y, z), "
            f"got {per_finger_contact_xyz.shape[2]}."
        )
    if per_finger_contact_xyz.shape[1] != len(finger_names):
        raise ValueError(
            "Finger-name count mismatch for per-finger xyz contact plot: "
            f"tensor has {per_finger_contact_xyz.shape[1]} fingers but got {len(finger_names)} names."
        )

    axis_labels = ["x", "y", "z"]
    n_fingers = len(finger_names)
    fig, axes = plt.subplots(
        n_fingers,
        3,
        figsize=(16, 2.6 * n_fingers),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x = np.arange(per_finger_contact_xyz.shape[0])
    per_finger_contact_xyz_np = per_finger_contact_xyz.detach().cpu().numpy()

    for finger_idx, finger_name in enumerate(finger_names):
        for axis_idx, axis_name in enumerate(axis_labels):
            ax = axes[finger_idx, axis_idx]
            signal = per_finger_contact_xyz_np[:, finger_idx, axis_idx]
            ax.step(x, signal, where="post", linewidth=1.4)
            ax.fill_between(x, 0.0, signal, step="post", alpha=0.25)
            if finger_idx == 0:
                ax.set_title(f"{axis_name}-axis")
            if axis_idx == 0:
                ax.set_ylabel(f"{finger_name}\nContact")
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0.0, 1.0])
            ax.grid(True, alpha=0.3)
            if finger_idx == (n_fingers - 1):
                ax.set_xlabel("Env Step")

    fig.suptitle(f"Episode {episode_idx} Tactile Contact Signals Per Finger and Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def finalize_contact_tracking_episode(
    *,
    episode_idx: int,
    episode_dir: Path,
    episode_contact_traces: list[dict],
) -> dict[str, float | int | str | dict[str, float]]:
    if len(episode_contact_traces) == 0:
        raise ValueError("Expected at least one contact trace to finalize an episode.")

    finger_names = list(episode_contact_traces[0]["finger_names"])
    contact = torch.stack([trace["contact"] for trace in episode_contact_traces], dim=0)
    per_finger_contact = torch.stack(
        [trace["per_finger_contact"] for trace in episode_contact_traces], dim=0
    )

    plot_path = episode_dir / "contact_signals_per_finger.png"
    trace_path = episode_dir / "contact_tracking.pt"
    summary_path = episode_dir / "contact_tracking_summary.json"
    save_contact_tracking_plot(
        episode_idx=episode_idx,
        finger_names=finger_names,
        per_finger_contact=per_finger_contact,
        plot_path=plot_path,
    )

    trace_payload = {
        "finger_names": finger_names,
        "contact": contact,
        "per_finger_contact": per_finger_contact,
    }
    torch.save(trace_payload, trace_path)

    mean_contact_by_finger = (
        per_finger_contact.to(torch.float32).mean(dim=0).detach().cpu().tolist()
    )
    summary = {
        "episode_idx": int(episode_idx),
        "num_env_steps": int(len(episode_contact_traces)),
        "mean_contact_fraction": float(
            per_finger_contact.to(torch.float32).mean().item()
        ),
        "mean_contact_fraction_by_finger": {
            finger_name: float(contact_fraction)
            for finger_name, contact_fraction in zip(
                finger_names, mean_contact_by_finger
            )
        },
        "plot_path": str(plot_path),
        "trace_path": str(trace_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
