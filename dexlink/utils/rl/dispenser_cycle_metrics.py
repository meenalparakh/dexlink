"""Ordered press/return accounting from actual dispenser joint positions."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch


@dataclass(frozen=True)
class DispenserCycleConfig:
    press_depth_m: float = 0.008
    return_tolerance_m: float = 0.0005
    dwell_steps: int = 2

    def __post_init__(self) -> None:
        if not (isfinite(self.press_depth_m) and isfinite(self.return_tolerance_m)
                and 0 < self.return_tolerance_m < self.press_depth_m):
            raise ValueError(
                "Expected finite 0 < return_tolerance_m < press_depth_m, got "
                f"{self.return_tolerance_m} and {self.press_depth_m}"
            )
        if type(self.dwell_steps) is not int or self.dwell_steps < 1:
            raise ValueError(f"Expected a positive integer dwell_steps, got {self.dwell_steps!r}")


class DispenserCycleTracker:
    """Count up–press–up cycles independently for each environment.

    The prismatic rest coordinate is zero and compression is negative. Dwell is
    measured in calls to update, once per control step in the task manager.
    Initial compression earns no credit before a sustained raised state.
    """

    def __init__(self, num_envs: int, device: torch.device | str,
                 config: DispenserCycleConfig) -> None:
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError(f"Expected a positive integer num_envs, got {num_envs!r}")
        self.config = config
        self.num_envs = num_envs
        self.completed_cycles = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.device = self.completed_cycles.device
        self.ready_for_press = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.awaiting_return = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.press_dwell = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.return_dwell = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def update(self, joint_position_m: torch.Tensor) -> None:
        if (joint_position_m.shape != (self.num_envs,)
                or joint_position_m.device != self.device
                or not joint_position_m.is_floating_point()
                or not torch.isfinite(joint_position_m).all()):
            raise ValueError(
                f"Expected finite floating joint positions with shape {(self.num_envs,)} "
                f"on {self.device}, got shape {tuple(joint_position_m.shape)}, "
                f"dtype {joint_position_m.dtype}, device {joint_position_m.device}"
            )
        cfg = self.config
        pressed = joint_position_m <= -cfg.press_depth_m
        returned = joint_position_m.abs() <= cfg.return_tolerance_m

        # Separate thresholds and consecutive dwell prevent boundary chatter.
        self.press_dwell = torch.where(
            self.ready_for_press & pressed,
            (self.press_dwell + 1).clamp(max=cfg.dwell_steps), 0,
        )
        self.return_dwell = torch.where(
            returned, (self.return_dwell + 1).clamp(max=cfg.dwell_steps), 0,
        )
        sustained_press = self.ready_for_press & (self.press_dwell >= cfg.dwell_steps)
        self.awaiting_return |= sustained_press
        self.ready_for_press &= ~sustained_press
        sustained_return = self.return_dwell >= cfg.dwell_steps
        completed = self.awaiting_return & sustained_return
        self.completed_cycles += completed.to(torch.long)
        self.awaiting_return &= ~completed
        self.ready_for_press |= sustained_return

    def reset(self, env_ids: torch.Tensor) -> None:
        if (env_ids.ndim != 1 or env_ids.dtype != torch.long or env_ids.device != self.device
                or torch.any(env_ids < 0) or torch.any(env_ids >= self.num_envs)):
            raise ValueError(
                f"Expected one-dimensional torch.long environment indices on {self.device} "
                f"within [0, {self.num_envs}), got {env_ids}"
            )
        self.completed_cycles[env_ids] = 0
        self.ready_for_press[env_ids] = False
        self.awaiting_return[env_ids] = False
        self.press_dwell[env_ids] = 0
        self.return_dwell[env_ids] = 0

    def metrics(self) -> dict[str, torch.Tensor]:
        head_returned = self.return_dwell >= self.config.dwell_steps
        success = (self.completed_cycles >= 2) & head_returned & ~self.awaiting_return
        return {
            "dispenser_completed_press_return_cycles": self.completed_cycles.to(torch.float32),
            "dispenser_head_returned": head_returned.to(torch.float32),
            "dispenser_two_press_return_cycles_success": success.to(torch.float32),
        }
