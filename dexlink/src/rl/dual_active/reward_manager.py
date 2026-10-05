"""Reward manager that measures each hand against the object it supports."""

from __future__ import annotations

import torch

from dexlink.src.rl.dual_active.registry import hand_objects_for_exported_path
from dexlink.src.rl.multi_object.reward_manager import (
    MultiObjectRewardCommonComputation,
    MultiObjectRewardComputation,
    MultiObjectRewardManager,
    MultiObjectRewardTerminationComputation,
)


class DualActiveRewardCommonComputation(MultiObjectRewardCommonComputation):
    """Shape each hand against its own object rather than a shared one.

    The default computation walks both hands but measures both against one point
    cloud, the canonical object's. That is right when one hand works and the
    other steadies something, and wrong when each hand supports an object of its
    own: a policy could hold one object properly and leave the other unsupported
    without the term noticing.

    Here each side is measured against the object it is assigned, so the term
    says "this hand is placed correctly on its own object" for both hands.
    """

    @property
    def hand_objects(self) -> dict[str, str] | None:
        return hand_objects_for_exported_path(
            getattr(self.env.cfg.task_cfg, "oakink_exported_path", "")
        )

    def compute_obj_bps_relative_to_hand(
        self, named_obs: dict[str, torch.Tensor], obj_pcd=None
    ) -> dict[str, torch.Tensor]:
        assignment = self.hand_objects
        # A caller that supplies a point cloud has already chosen the object, and
        # a task that has not opted in keeps the shared-object behaviour.
        if assignment is None or obj_pcd is not None:
            return super().compute_obj_bps_relative_to_hand(named_obs, obj_pcd)

        obs_mgr = self.env.observation_manager
        bps: dict[str, torch.Tensor] = {}
        for side in ("left", "right"):
            obs_key = f"{side}_lpos"
            if obs_key not in named_obs:
                continue
            object_id = assignment.get(side)
            if object_id is None:
                raise ValueError(
                    f"Hand {side!r} is active but no object is assigned to it; "
                    f"the assignment names {sorted(assignment)}."
                )
            side_pcd = obs_mgr.get_object_pcd_from_named_obs(
                named_obs,
                expected_feat_dim=obs_mgr.privileged_object_pcd_dim,
                object_id=object_id,
            )
            hand = self.env.physics_manager.scene_entities[side]
            n_links = len(hand.human_joint_names)
            lpos = named_obs[obs_key].view(self.env.num_envs, n_links, 3)
            result = obs_mgr.compute_bps(pcd=side_pcd, bps=lpos)
            bps[side] = result.reshape(result.shape[0], -1)
        return bps


class DualActiveRewardManager(MultiObjectRewardManager):
    """The multi-object reward manager with the per-hand shaping swapped in."""

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.common = DualActiveRewardCommonComputation(cfg, env)
        self.reward_computation = MultiObjectRewardComputation(self.common)
        self.termination_computation = MultiObjectRewardTerminationComputation(
            self.common
        )
