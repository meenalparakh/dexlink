from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from dexlink.utils.retargeting.vis_server import VizServer

if TYPE_CHECKING:
    from dexlink.src.rl.world.env import Env
    from dexlink.src.rl.world.robot import Hand

# the other classes in env only rely on the scene manager properties
# which might be internally wired to task and physics manager
# so scene manager acts as a unified interface currently


class SceneManager:
    _HAND_SKELETON_EDGES = (
        ("wrist", "thumb_base"),
        ("thumb_base", "thumb_proximal"),
        ("thumb_proximal", "thumb_tip"),
        ("wrist", "index_base"),
        ("index_base", "index_proximal"),
        ("index_proximal", "index_tip"),
        ("wrist", "middle_base"),
        ("middle_base", "middle_proximal"),
        ("middle_proximal", "middle_tip"),
        ("wrist", "ring_base"),
        ("ring_base", "ring_proximal"),
        ("ring_proximal", "ring_tip"),
        ("wrist", "little_base"),
        ("little_base", "little_proximal"),
        ("little_proximal", "little_tip"),
    )

    def __init__(self, env: Env):
        self.env = env
        self.policy_pcd_viz: VizServer | None = None

    def initialize(self) -> None:
        if not self.env.visualize_policy_pcd:
            return
        if self.env.visualize_policy_pcd_on not in {"reset", "step", "both"}:
            raise ValueError(
                "env.visualize_policy_pcd_on must be one of 'reset', 'step', or 'both', "
                f"got {self.env.visualize_policy_pcd_on!r}."
            )
        self.policy_pcd_viz = VizServer()
        print("Initialized separate VizServer for policy point cloud visualization.")

    def maybe_visualize_policy_point_cloud(
        self, policy_named_obs: dict[str, torch.Tensor], *, event: str
    ) -> None:
        if not self.env.visualize_policy_pcd:
            return
        if self.env.visualize_policy_pcd_on not in {event, "both"}:
            return
        if self.policy_pcd_viz is None:
            raise ValueError(
                "Policy point cloud visualization requested before SceneManager.initialize()."
            )

        object_pcd = policy_named_obs.get("object_pcd")
        if object_pcd is not None:
            self._view_policy_xyz_point_cloud(
                object_pcd,
                name="policy_object_pcd",
                color=np.array([0.1, 0.6, 1.0], dtype=np.float32),
            )
        hand_pcd = policy_named_obs.get("hand_pcd")
        if hand_pcd is not None:
            self._view_policy_xyz_point_cloud(
                hand_pcd,
                name="policy_hand_pcd",
                color=np.array([1.0, 0.35, 0.2], dtype=np.float32),
            )

        for side, color in (
            ("left_lpos", np.array([1.0, 0.45, 0.15], dtype=np.float32)),
            ("right_lpos", np.array([0.15, 0.85, 0.25], dtype=np.float32)),
        ):
            link_pos = policy_named_obs.get(side)
            if link_pos is None:
                continue
            self._view_policy_xyz_point_cloud(
                link_pos,
                name=f"{side}_pcd",
                color=color,
            )
        for side, color in (
            ("left", np.array([0.95, 0.45, 0.15], dtype=np.float32)),
            ("right", np.array([0.15, 0.85, 0.25], dtype=np.float32)),
        ):
            self._view_policy_hand_skeleton(policy_named_obs, side=side, color=color)
        pause_helper = getattr(
            self.env.physics_manager, "pause_if_visualization_active", None
        )
        if pause_helper is not None:
            pause_helper("Showing policy-input visualization.")
        else:
            input("Press Enter to continue...")

    def _view_policy_xyz_point_cloud(
        self,
        flat_points: torch.Tensor,
        *,
        name: str,
        color: np.ndarray,
    ) -> None:
        if self.policy_pcd_viz is None:
            raise ValueError(
                "SceneManager policy point cloud visualizer is not initialized."
            )
        if flat_points.shape[-1] % 3 != 0:
            raise ValueError(
                f"Expected {name} last dim to be divisible by 3, got {flat_points.shape[-1]}."
            )
        env0_points = (
            flat_points[0].view(-1, 3).detach().cpu().numpy().astype(np.float32)
        )
        colors = np.tile(color[None, :], (env0_points.shape[0], 1))
        self.policy_pcd_viz.view_pcd(
            env0_points,
            colors=colors,
            name=name,
            size=float(self.env.visualize_policy_pcd_point_size),
        )

    def _view_policy_hand_skeleton(
        self,
        policy_named_obs: dict[str, torch.Tensor],
        *,
        side: str,
        color: np.ndarray,
    ) -> None:
        if self.policy_pcd_viz is None:
            raise ValueError(
                "SceneManager policy point cloud visualizer is not initialized."
            )
        lpos_key = f"{side}_lpos"
        flat_points = policy_named_obs.get(lpos_key)
        if flat_points is None:
            return
        if side not in self.env.physics_manager.scene_entities:
            return

        hand: Hand = self.env.physics_manager.scene_entities[side]
        if flat_points.shape[-1] != 3 * len(hand.human_joint_names):
            raise ValueError(
                f"Expected {lpos_key} to contain one xyz triplet per human joint "
                f"({len(hand.human_joint_names)} total), got shape {tuple(flat_points.shape)}."
            )

        env0_points = (
            flat_points[0]
            .view(len(hand.human_joint_names), 3)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        name_to_point = {
            joint_name: env0_points[idx]
            for idx, joint_name in enumerate(hand.human_joint_names)
        }

        for parent_name, child_name in self._HAND_SKELETON_EDGES:
            parent = name_to_point.get(parent_name)
            child = name_to_point.get(child_name)
            if parent is None or child is None:
                continue
            self.policy_pcd_viz.view_line_segment(
                parent,
                child,
                name=f"{side}_skeleton/{parent_name}_to_{child_name}",
                color=color,
                linewidth=2.0,
                opacity=1.0,
            )
