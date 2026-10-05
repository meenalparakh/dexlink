import torch

from dexlink.utils.retargeting.math_utils import (
    quat_error_magnitude,
    quat_from_6d,
)


def mse_error(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((current - target) ** 2, dim=-1)


def exp_reward_from_diff(diff, scale: float, norm=1) -> torch.Tensor:
    error = torch.norm(diff, dim=-1, p=norm)
    return torch.exp(-scale * error)


def exp_reward_from_error(error: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.exp(-scale * error)


def angle_error(current, target, current_format="6d", target_format="6d"):

    # if 6d, convert to quat w, x, y, z
    if current_format == "6d":
        current_quat = quat_from_6d(current)
    else:
        current_quat = current

    if target_format == "6d":
        target_quat = quat_from_6d(target)
    else:
        target_quat = target

    error = quat_error_magnitude(current_quat, target_quat)
    return error
