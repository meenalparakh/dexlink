"""Explicit unlabeled scene-cloud contract shared by simulation and deployment."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import ObsCfg, RealCfg


def scene_cloud_contract(cfg: ObsCfg) -> dict | None:
    """Validate opt-in configuration; legacy point budgets remain per object."""
    if cfg.policy_object_pcd_mode == "legacy_per_object":
        return None
    if cfg.policy_object_pcd_mode not in {"scene_cloud_v1", "rendered_depth_v1"}:
        raise ValueError(
            "Expected policy_object_pcd_mode 'legacy_per_object', 'scene_cloud_v1' or 'rendered_depth_v1', "
            f"got {cfg.policy_object_pcd_mode!r}."
        )
    if (
        not cfg.use_policy_object_pcd
        or type(cfg.policy_object_pcd_n_pts) is not int
        or cfg.policy_object_pcd_n_pts <= 0
    ):
        raise ValueError(
            "scene_cloud_v1 requires policy object PCDs and a positive total point budget."
        )
    lower, upper = cfg.scene_cloud_world_min, cfg.scene_cloud_world_max
    if lower is None or upper is None:
        raise ValueError(
            "scene_cloud_v1 requires explicit scene_cloud_world_min/max in calibrated world meters."
        )
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError(
            f"Expected scene cloud min/max XYZ triples, got {lower}, {upper}."
        )
    bounds = torch.tensor([lower, upper], dtype=torch.float64)
    if not torch.isfinite(bounds).all() or not (bounds[0] < bounds[1]).all():
        raise ValueError(
            f"Expected finite scene cloud bounds (2, 3) with min < max, got {lower}, {upper}."
        )
    contract = {
        "mode": cfg.policy_object_pcd_mode,
        "world_min": list(lower),
        "world_max": list(upper),
        "n_points": cfg.policy_object_pcd_n_pts,
        "features": "unlabeled_xyz_m",
        "sampling": "uniform_with_replacement",
        "actor_frame": "world_axes_minus_active_wrist_translation",
        "synthetic_table": False,
        "augmentation": "none",
    }
    if cfg.policy_object_pcd_mode == "rendered_depth_v1":
        contract["camera"] = cfg.rendered_depth_cfg.observation_contract()
    return contract


def validate_scene_cloud_actor_inputs(cfg: ObsCfg, terms: tuple[str, ...]) -> None:
    """Keep simulated object state and contact out of the deployable actor."""
    if scene_cloud_contract(cfg) is None:
        return
    if "object_pcd" not in terms:
        raise ValueError("scene_cloud_v1 actor requires object_pcd in selected_terms.")
    privileged_terms = {"object", "object_vel", "contact"}.intersection(terms)
    if privileged_terms:
        raise ValueError(
            f"scene_cloud_v1 actor requires deployable terms; received {sorted(privileged_terms)}."
        )


def sample_scene_cloud(points_world: torch.Tensor, contract: dict) -> torch.Tensor:
    """Crop all clusters together, then uniformly sample a fixed total budget."""
    if (
        points_world.ndim != 3
        or points_world.shape[-1] != 3
        or not points_world.is_floating_point()
    ):
        raise ValueError(
            f"Expected floating world XYZ shape (B, N, 3), got {points_world.shape}, {points_world.dtype}."
        )
    if not torch.isfinite(points_world).all():
        raise ValueError("Expected finite world XYZ; received NaN or infinity.")
    lower = points_world.new_tensor(contract["world_min"])
    upper = points_world.new_tensor(contract["world_max"])
    keep = ((points_world >= lower) & (points_world <= upper)).all(dim=-1)
    empty = torch.where(keep.sum(dim=1) == 0)[0]
    if empty.numel():
        raise ValueError(
            f"Scene cloud crop is empty for batch indices {empty.tolist()}; no zero padding is permitted."
        )
    indices = torch.multinomial(
        keep.to(points_world.dtype), contract["n_points"], replacement=True
    )
    return points_world.gather(1, indices[..., None].expand(-1, -1, 3))


def validate_scene_cloud_real_config(obs_cfg: ObsCfg, real_cfg: RealCfg) -> None:
    """Reject incompatible live perception settings before hardware initialization."""
    if scene_cloud_contract(obs_cfg) is None:
        return
    if real_cfg.use_phase_cached_object_pcd:
        raise ValueError(
            "scene_cloud_v1 requires use_phase_cached_object_pcd=False; independent objects need fresh frames."
        )
    if real_cfg.table_plane_sample_points != 0:
        raise ValueError(
            "scene_cloud_v1 requires table_plane_sample_points=0; synthetic table points are outside the contract."
        )


def require_positive_policy_checkpoint(policy_path: Path) -> int:
    """Require an explicit positive periodic/final checkpoint; never infer 'best'."""
    if policy_path.name != "policy.pt" or not policy_path.parent.name.isdecimal():
        raise ValueError(
            f"Expected explicit models/<positive timestep>/policy.pt, got {policy_path}."
        )
    timestep = int(policy_path.parent.name)
    if timestep <= 0:
        raise ValueError(
            f"Expected strictly positive checkpoint timestep, got {timestep}."
        )
    return timestep


def validate_scene_cloud_checkpoint(policy_path: Path, obs_cfg: ObsCfg) -> None:
    """Check cloud semantics before constructing a live environment."""
    expected = scene_cloud_contract(obs_cfg)
    processor_path = policy_path.parent / "policy_obs_processor.pt"
    if not processor_path.is_file():
        if expected is not None:
            raise FileNotFoundError(
                f"scene_cloud_v1 requires processor metadata: {processor_path}"
            )
        return
    state = torch.load(processor_path, map_location="cpu", weights_only=True)
    metadata = state.get("_extra_state")
    received = metadata.get("scene_cloud") if isinstance(metadata, dict) else None
    if received != expected:
        raise ValueError(
            f"Scene-cloud checkpoint contract mismatch: expected {expected}, received {received}."
        )
