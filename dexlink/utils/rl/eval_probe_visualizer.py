from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from dexlink.utils.retargeting.vis_server import VizServer

if TYPE_CHECKING:
    from dexlink.src.rl.agent.ppo_eval import PPOEval
    from dexlink.src.rl.world.env import Env


@dataclass
class EvalProbeVisualizerConfig:
    point_size: float = 0.006
    action_mesh_opacity: float = 0.45
    show_action_mesh: bool = True
    meshcat_port: int = 6000


class EvalProbeVisualizer:
    def __init__(
        self,
        *,
        env: Env,
        agent: PPOEval,
        cfg: EvalProbeVisualizerConfig,
    ) -> None:
        self.env = env
        self.agent = agent
        self.cfg = cfg
        self.viz = VizServer(port_vis=int(cfg.meshcat_port))
        self._printed_summary = False

    def print_agent_input_summary(self) -> None:
        obs_proc = self.agent.policy_obs_processor
        dim_by_term = dict(zip(obs_proc.full_obs_terms, obs_proc.full_obs_term_dims))
        print("\nAgent policy input terms:")
        print(f"{'term':<24} {'raw_dim':>8} {'current_shape':>24}")
        print("-" * 60)
        named_obs = self.env.current_named_policy_obs
        if named_obs is None:
            raise ValueError("Expected env.current_named_policy_obs to be populated.")
        for term in obs_proc.selected_terms:
            value = named_obs.get(term)
            if value is None:
                raise ValueError(f"Selected policy input term {term!r} is missing.")
            print(f"{term:<24} {dim_by_term[term]:>8} {str(tuple(value.shape)):>24}")
        print("-" * 60)
        print(f"{'total raw obs':<24} {obs_proc.true_obs_dim:>8}")
        print(f"{'processed policy obs':<24} {obs_proc.obs_dim:>8}\n")
        self._printed_summary = True

    def render_step(
        self,
        *,
        step_idx: int,
        policy_named_obs: dict[str, torch.Tensor],
        action_probe: dict,
    ) -> None:
        if not self._printed_summary:
            self.print_agent_input_summary()
        self._print_step_term_summary(step_idx, policy_named_obs)
        self._view_policy_point_cloud_terms(policy_named_obs)
        if self.cfg.show_action_mesh:
            self._view_action_target_mesh(action_probe)
            self._view_real_control_action_target_mesh()

    def _print_step_term_summary(
        self, step_idx: int, policy_named_obs: dict[str, torch.Tensor]
    ) -> None:
        obs_proc = self.agent.policy_obs_processor
        pieces = []
        for term in obs_proc.selected_terms:
            value = policy_named_obs.get(term)
            if value is None:
                raise ValueError(f"Selected policy input term {term!r} is missing.")
            pieces.append(f"{term}{tuple(value.shape)}")
        print(f"[probe step {step_idx}] " + ", ".join(pieces))

    def _view_policy_point_cloud_terms(
        self, policy_named_obs: dict[str, torch.Tensor]
    ) -> None:
        colors = {
            "object_pcd": np.array([0.10, 0.60, 1.00], dtype=np.float32),
            "hand_pcd": np.array([1.00, 0.35, 0.20], dtype=np.float32),
            "left_lpos": np.array([1.00, 0.45, 0.15], dtype=np.float32),
            "right_lpos": np.array([0.15, 0.85, 0.25], dtype=np.float32),
        }
        for term in self.agent.policy_obs_processor.selected_terms:
            value = policy_named_obs.get(term)
            if value is None:
                raise ValueError(f"Selected policy input term {term!r} is missing.")
            if term in {"object_pcd", "hand_pcd", "left_lpos", "right_lpos"}:
                self._view_flat_xyz_tensor(
                    value,
                    name=f"policy_input/{term}",
                    color=colors[term],
                )

        object_pcd_by_id = policy_named_obs.get("object_pcd_by_id")
        if (
            "object_pcd" in self.agent.policy_obs_processor.selected_terms
            and isinstance(object_pcd_by_id, dict)
        ):
            palette = (
                np.array([0.10, 0.60, 1.00], dtype=np.float32),
                np.array([0.70, 0.20, 1.00], dtype=np.float32),
                np.array([0.00, 0.75, 0.45], dtype=np.float32),
                np.array([1.00, 0.65, 0.05], dtype=np.float32),
            )
            for idx, object_id in enumerate(sorted(object_pcd_by_id)):
                self._view_flat_xyz_tensor(
                    object_pcd_by_id[object_id],
                    name=f"policy_input/object_pcd_by_id/{object_id}",
                    color=palette[idx % len(palette)],
                )

    def _view_flat_xyz_tensor(
        self,
        value: torch.Tensor,
        *,
        name: str,
        color: np.ndarray,
    ) -> None:
        if value.ndim < 2:
            raise ValueError(
                f"Expected {name} to have batch dimension, got {value.ndim}."
            )
        env0 = value[0]
        if env0.ndim == 1:
            if env0.shape[0] % 3 != 0:
                raise ValueError(
                    f"Expected flat xyz term {name!r} last dim divisible by 3, "
                    f"got {env0.shape[0]}."
                )
            points = env0.view(-1, 3)
        elif env0.ndim == 2:
            if env0.shape[-1] != 3:
                raise ValueError(
                    f"Expected xyz term {name!r} shape (N, 3), got {tuple(env0.shape)}."
                )
            points = env0
        else:
            raise ValueError(
                f"Expected {name!r} env0 tensor rank 1 or 2, got {env0.ndim}."
            )
        points_np = points.detach().cpu().numpy().astype(np.float32)
        colors_np = np.tile(color[None, :], (points_np.shape[0], 1))
        self.viz.view_pcd(
            points_np,
            colors=colors_np,
            name=name,
            size=float(self.cfg.point_size),
        )

    def _view_action_target_mesh(self, action_probe: dict) -> None:
        for side, side_probe in self._iter_action_side_probes(action_probe):
            hand = self.env.physics_manager.scene_entities.get(side)
            if hand is None:
                raise ValueError(f"Action probe references unknown hand side {side!r}.")
            target_qpos = side_probe.get("target_qpos")
            joint_names = side_probe.get("joint_names")
            if not isinstance(target_qpos, torch.Tensor):
                raise ValueError(f"Action probe for {side!r} is missing target_qpos.")
            if not isinstance(joint_names, list):
                raise ValueError(f"Action probe for {side!r} is missing joint_names.")
            target_env0 = target_qpos[0].detach().cpu().numpy().astype(np.float32)
            if len(joint_names) != target_env0.shape[0]:
                raise ValueError(
                    f"Action probe {side!r} joint_names/target_qpos mismatch: "
                    f"{len(joint_names)} names vs {target_env0.shape[0]} values."
                )
            self.viz.view_robot(
                hand.urdf_inst,
                joint_names=joint_names,
                joint_vals=target_env0,
                name=f"action_target/{side}",
                color=0xD95F02,
                alpha=float(self.cfg.action_mesh_opacity),
                base_pose=np.asarray(hand.base_pose, dtype=np.float32),
            )

    def _iter_action_side_probes(self, action_probe: dict):
        if "target_qpos" in action_probe:
            side = action_probe.get("side")
            if not isinstance(side, str):
                raise ValueError("Single-hand action probe is missing string 'side'.")
            yield side, action_probe
            return
        for side, side_probe in action_probe.items():
            if not isinstance(side_probe, dict):
                raise ValueError(
                    f"Expected action probe entry for side {side!r} to be a dict, "
                    f"got {type(side_probe).__name__}."
                )
            yield str(side), side_probe

    def _view_real_control_action_target_mesh(self) -> None:
        physics = self.env.physics_manager
        if not bool(self.env.real_cfg.visualize_runtime_full_arm_mesh):
            return
        real_control_hands = getattr(physics, "real_control_hands", None)
        real_control_qpos = getattr(physics, "_latest_real_control_qpos_by_side", None)
        if not isinstance(real_control_hands, dict) or not isinstance(
            real_control_qpos, dict
        ):
            return
        for side, hand in real_control_hands.items():
            target_qpos = real_control_qpos.get(side)
            if target_qpos is None:
                continue
            target_qpos = torch.as_tensor(target_qpos, dtype=torch.float32)
            expected_shape = (1, len(hand.joint_names))
            if tuple(target_qpos.shape) != expected_shape:
                raise ValueError(
                    "Real-control action target qpos shape mismatch for "
                    f"{side!r}: expected {expected_shape}, got {tuple(target_qpos.shape)}."
                )
            self.viz.view_robot(
                hand.urdf_inst,
                joint_names=list(hand.joint_names),
                joint_vals=target_qpos[0].detach().cpu().numpy().astype(np.float32),
                name=f"action_target_real_control/{side}",
                color=0x7570B3,
                alpha=float(self.cfg.action_mesh_opacity),
                base_pose=np.asarray(hand.base_pose, dtype=np.float32),
            )
