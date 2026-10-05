from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from dexlink.utils.retargeting.math_utils import (
    matrix_from_quat,
    quat_box_minus,
    quat_from_matrix,
    quat_from_rotvec,
    quat_mul,
    rot6d_from_matrix,
)

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import ReferenceTrajectoryCfg
    from dexlink.src.rl.world.art_object import ArtObject
    from dexlink.src.rl.world.env import Env
    from dexlink.src.rl.world.robot import Hand


class ReferenceTrajectoryManager:
    def __init__(self, cfg: ReferenceTrajectoryCfg, env: Env):
        self.cfg = cfg
        self.env = env
        self.device = torch.device(env.device)
        self.base_state: dict[str, Any] | None = None
        self.optimized_state: dict[str, Any] | None = None
        self.last_update_stats: dict[str, Any] = {}
        self.last_gp_plot_data: dict[str, Any] = {}
        self.gp_plot_axis_limits: dict[str, dict[str, tuple[float, float]]] = {}

    def initialize(self) -> None:
        state = self._build_initial_state()
        self.base_state = self._clone_state(state)
        self.optimized_state = self._clone_state(state)
        self.last_update_stats = self._compute_reference_stats(self.optimized_state)
        self.last_gp_plot_data = {}
        self.gp_plot_axis_limits = {}

    def is_enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def active_state(self) -> dict[str, Any]:
        if self.optimized_state is None:
            raise ValueError("ReferenceTrajectoryManager was not initialized.")
        return self.optimized_state if self.is_enabled() else self.base_state

    def base_reference_state(self) -> dict[str, Any]:
        if self.base_state is None:
            raise ValueError("ReferenceTrajectoryManager was not initialized.")
        return self.base_state

    def object_joint_names(self) -> list[str]:
        return list(self.active_state()["joint_names"])

    def object_pos(self) -> torch.Tensor:
        return self.active_state()["obj_pos"]

    def object_quat(self) -> torch.Tensor:
        return self.active_state()["obj_quat"]

    def object_arti(self) -> torch.Tensor | None:
        return self.active_state()["obj_arti"]

    def hand_joint_positions(self, side: str) -> torch.Tensor:
        return self.active_state()["hands"][side]["jvals"]

    def hand_wrist_pos(self, side: str) -> torch.Tensor:
        return self.active_state()["hands"][side]["wrist_pos"]

    def hand_wrist_quat(self, side: str) -> torch.Tensor:
        return self.active_state()["hands"][side]["wrist_quat"]

    def hand_reference_obs(self, side: str) -> torch.Tensor:
        return self.active_state()["hands"][side]["ref"]

    def hand_link_positions(self, side: str) -> dict[str, torch.Tensor]:
        return self.active_state()["hands"][side]["link_pos"]

    def update_from_rollout_batch(
        self,
        *,
        timesteps: torch.Tensor | np.ndarray,
        returns: torch.Tensor | np.ndarray,
        object_pos: torch.Tensor | np.ndarray | None = None,
        object_quat: torch.Tensor | np.ndarray | None = None,
        hand_wrist_pos: dict[str, torch.Tensor | np.ndarray] | None = None,
        hand_wrist_quat: dict[str, torch.Tensor | np.ndarray] | None = None,
        hand_joint_positions: dict[str, torch.Tensor | np.ndarray] | None = None,
        anchors: dict[int, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # This is the main external update API: PPO (or eval code) passes in a
        # batch of rollout suffixes and their returns, and we turn the best ones
        # into a smooth optimized robot reference trajectory.
        if not self.is_enabled():
            raise ValueError(
                "Reference trajectory updates require reference_cfg.enabled=True."
            )
        if self.base_state is None or self.optimized_state is None:
            raise ValueError("ReferenceTrajectoryManager was not initialized.")

        rollout_ts = self._to_long_tensor(timesteps)
        returns_t = self._to_float_tensor(returns)
        if rollout_ts.ndim != 2:
            raise ValueError(
                f"Expected timesteps with shape (num_rollouts, horizon), got {tuple(rollout_ts.shape)}."
            )
        if returns_t.ndim != 1 or returns_t.shape[0] != rollout_ts.shape[0]:
            raise ValueError(
                "Expected returns with shape (num_rollouts,) matching timesteps, "
                f"got returns={tuple(returns_t.shape)} timesteps={tuple(rollout_ts.shape)}."
            )

        selected_indices, sample_weights = self._select_top_rollouts(returns_t)
        selected_ts = rollout_ts.index_select(0, selected_indices)
        candidate_state = self._clone_state(self.optimized_state)
        posterior_stats: dict[str, float] = {}
        gp_plot_data: dict[str, Any] = {"hands": {}, "objects": {}}

        if object_pos is not None:
            selected_pos = self._select_rollout_tensor(object_pos, selected_indices, 3)
            fitted_obj_pos, obj_pos_var, obj_pos_std = self._fit_translation_signal(
                base_signal=self.base_state["obj_pos"],
                rollout_signal=selected_pos,
                rollout_timesteps=selected_ts,
                sample_weights=sample_weights,
                clip_limit=float(self.cfg.max_position_residual_m),
                prior_std=float(self.cfg.object_gp_prior_position_std_m),
                anchors=anchors,
                key="obj_pos",
            )
            candidate_state["obj_pos"] = fitted_obj_pos
            posterior_stats["obj_pos_variance"] = obj_pos_var
            gp_plot_data["objects"]["obj_pos_std"] = obj_pos_std.detach().cpu()

        if object_quat is not None:
            selected_quat = self._select_rollout_tensor(
                object_quat, selected_indices, 4
            )
            fitted_obj_quat, obj_quat_var, obj_quat_std = self._fit_rotation_signal(
                base_quat=self.base_state["obj_quat"],
                rollout_quat=selected_quat,
                rollout_timesteps=selected_ts,
                sample_weights=sample_weights,
                clip_limit=float(self.cfg.max_rotation_residual_rad),
                prior_std=float(self.cfg.object_gp_prior_rotation_std_rad),
                anchors=anchors,
                key="obj_quat",
            )
            candidate_state["obj_quat"] = fitted_obj_quat
            posterior_stats["obj_quat_variance"] = obj_quat_var
            gp_plot_data["objects"]["obj_quat_std"] = obj_quat_std.detach().cpu()

        hand_sides = set()
        if hand_wrist_pos is not None:
            hand_sides.update(hand_wrist_pos.keys())
        if hand_wrist_quat is not None:
            hand_sides.update(hand_wrist_quat.keys())
        if hand_joint_positions is not None:
            hand_sides.update(hand_joint_positions.keys())

        for side in hand_sides:
            if side not in self.base_state["hands"]:
                raise ValueError(f"Unknown hand side {side!r} in rollout batch.")
            if hand_wrist_pos is not None and side in hand_wrist_pos:
                selected_wrist_pos = self._select_rollout_tensor(
                    hand_wrist_pos[side], selected_indices, 3
                )
                fitted_wrist_pos, wrist_pos_var, wrist_pos_std = (
                    self._fit_translation_signal(
                        base_signal=self.base_state["hands"][side]["wrist_pos"],
                        rollout_signal=selected_wrist_pos,
                        rollout_timesteps=selected_ts,
                        sample_weights=sample_weights,
                        clip_limit=float(self.cfg.max_position_residual_m),
                        prior_std=float(self.cfg.gp_prior_position_std_m),
                        anchors=anchors,
                        key=f"{side}_wrist_pos",
                    )
                )
                candidate_state["hands"][side]["wrist_pos"] = fitted_wrist_pos
                posterior_stats[f"{side}_wrist_pos_variance"] = wrist_pos_var
                gp_plot_data["hands"].setdefault(side, {})[
                    "wrist_pos_std"
                ] = wrist_pos_std.detach().cpu()
            if hand_wrist_quat is not None and side in hand_wrist_quat:
                selected_wrist_quat = self._select_rollout_tensor(
                    hand_wrist_quat[side], selected_indices, 4
                )
                fitted_wrist_quat, wrist_quat_var, wrist_quat_std = (
                    self._fit_rotation_signal(
                        base_quat=self.base_state["hands"][side]["wrist_quat"],
                        rollout_quat=selected_wrist_quat,
                        rollout_timesteps=selected_ts,
                        sample_weights=sample_weights,
                        clip_limit=float(self.cfg.max_rotation_residual_rad),
                        prior_std=float(self.cfg.gp_prior_rotation_std_rad),
                        anchors=anchors,
                        key=f"{side}_wrist_quat",
                    )
                )
                candidate_state["hands"][side]["wrist_quat"] = fitted_wrist_quat
                posterior_stats[f"{side}_wrist_quat_variance"] = wrist_quat_var
                gp_plot_data["hands"].setdefault(side, {})[
                    "wrist_quat_std"
                ] = wrist_quat_std.detach().cpu()
            if hand_joint_positions is not None and side in hand_joint_positions:
                selected_joints = self._select_rollout_tensor(
                    hand_joint_positions[side],
                    selected_indices,
                    self.base_state["hands"][side]["finger_jvals"].shape[1],
                )
                fitted_joints, joint_var, joint_std = self._fit_joint_signal(
                    base_signal=self.base_state["hands"][side]["finger_jvals"],
                    rollout_signal=selected_joints,
                    rollout_timesteps=selected_ts,
                    sample_weights=sample_weights,
                    clip_limit=float(self.cfg.max_joint_residual_rad),
                    prior_std=float(self.cfg.gp_prior_joint_std_rad),
                    anchors=anchors,
                    key=f"{side}_finger_jvals",
                )
                candidate_state["hands"][side]["finger_jvals"] = fitted_joints
                posterior_stats[f"{side}_finger_jvals_variance"] = joint_var
                gp_plot_data["hands"].setdefault(side, {})[
                    "finger_joint_std"
                ] = joint_std.detach().cpu()

        self._apply_custom_anchors(candidate_state, anchors)
        self._reconstruct_hand_joint_trajectory(candidate_state)
        self.optimized_state = self._ema_update(self.optimized_state, candidate_state)
        self._apply_custom_anchors(self.optimized_state, anchors)
        self._reconstruct_hand_joint_trajectory(self.optimized_state)
        self.last_update_stats = {
            "selected_rollouts": int(selected_indices.numel()),
            "selected_rollout_indices": selected_indices.detach().cpu().tolist(),
            **posterior_stats,
            **self._compute_reference_stats(self.optimized_state),
        }
        self.last_gp_plot_data = gp_plot_data
        return dict(self.last_update_stats)

    def save_gp_plot(
        self,
        *,
        output_dir: str | Path,
        update_step: int,
        side: str = "right",
    ) -> Path | None:
        # Save one panel per GP state dimension so the update can be inspected
        # after training. The first saved plot fixes the y-axis limits per
        # channel, and later plots reuse those limits so changes over time are
        # visually comparable.
        if self.base_state is None or self.optimized_state is None:
            raise ValueError("ReferenceTrajectoryManager was not initialized.")
        if side not in self.base_state["hands"]:
            raise ValueError(f"Unknown hand side {side!r} for GP plotting.")
        hand_plot_data = self.last_gp_plot_data.get("hands", {}).get(side)
        object_plot_data = self.last_gp_plot_data.get("objects", {})
        if hand_plot_data is None and len(object_plot_data) == 0:
            return None
        wrist_pos_std = (
            None if hand_plot_data is None else hand_plot_data.get("wrist_pos_std")
        )
        wrist_quat_std = (
            None if hand_plot_data is None else hand_plot_data.get("wrist_quat_std")
        )
        finger_joint_std = (
            None if hand_plot_data is None else hand_plot_data.get("finger_joint_std")
        )
        obj_pos_std = object_plot_data.get("obj_pos_std")
        obj_quat_std = object_plot_data.get("obj_quat_std")

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        robot: Hand = self.env.physics_manager.scene_entities[side]
        phase = self._normalized_query_times().detach().cpu().numpy()
        wrist_pos_std_np = (
            None if wrist_pos_std is None else wrist_pos_std.detach().cpu().numpy()
        )
        wrist_quat_std_np = (
            None if wrist_quat_std is None else wrist_quat_std.detach().cpu().numpy()
        )
        finger_joint_std_np = (
            None
            if finger_joint_std is None
            else finger_joint_std.detach().cpu().numpy()
        )
        obj_pos_std_np = (
            None if obj_pos_std is None else obj_pos_std.detach().cpu().numpy()
        )
        obj_quat_std_np = (
            None if obj_quat_std is None else obj_quat_std.detach().cpu().numpy()
        )

        base_hand_state = self.base_state["hands"][side]
        optimized_hand_state = self.optimized_state["hands"][side]
        channels: list[tuple[str, torch.Tensor, torch.Tensor, np.ndarray]] = []

        if obj_pos_std_np is not None:
            base_obj_pos = self.base_state["obj_pos"]
            optimized_obj_pos = self.optimized_state["obj_pos"]
            obj_pos_labels = ["obj_pos_x", "obj_pos_y", "obj_pos_z"]
            for axis_idx, axis_name in enumerate(obj_pos_labels):
                channels.append(
                    (
                        axis_name,
                        base_obj_pos[:, axis_idx],
                        optimized_obj_pos[:, axis_idx],
                        obj_pos_std_np,
                    )
                )

        if obj_quat_std_np is not None:
            base_obj_quat = self.base_state["obj_quat"]
            optimized_obj_quat = self.optimized_state["obj_quat"]
            base_obj_rotvec = torch.zeros(
                (base_obj_quat.shape[0], 3),
                dtype=base_obj_quat.dtype,
                device=base_obj_quat.device,
            )
            optimized_obj_rotvec = quat_box_minus(optimized_obj_quat, base_obj_quat)
            obj_rotvec_labels = ["obj_rotvec_x", "obj_rotvec_y", "obj_rotvec_z"]
            for axis_idx, axis_name in enumerate(obj_rotvec_labels):
                channels.append(
                    (
                        axis_name,
                        base_obj_rotvec[:, axis_idx],
                        optimized_obj_rotvec[:, axis_idx],
                        obj_quat_std_np,
                    )
                )

        if (
            wrist_pos_std_np is not None
            and wrist_quat_std_np is not None
            and finger_joint_std_np is not None
        ):
            base_wrist_pos = base_hand_state["wrist_pos"]
            optimized_wrist_pos = optimized_hand_state["wrist_pos"]
            wrist_pos_labels = ["wrist_pos_x", "wrist_pos_y", "wrist_pos_z"]
            for axis_idx, axis_name in enumerate(wrist_pos_labels):
                channels.append(
                    (
                        axis_name,
                        base_wrist_pos[:, axis_idx],
                        optimized_wrist_pos[:, axis_idx],
                        wrist_pos_std_np,
                    )
                )

            base_wrist_quat = base_hand_state["wrist_quat"]
            optimized_wrist_quat = optimized_hand_state["wrist_quat"]
            # Plot wrist rotation as tangent-space residuals relative to the base
            # reference. Raw quaternion channels can differ by a sign flip even when
            # the represented rotation is identical.
            base_wrist_rotvec = torch.zeros(
                (base_wrist_quat.shape[0], 3),
                dtype=base_wrist_quat.dtype,
                device=base_wrist_quat.device,
            )
            optimized_wrist_rotvec = quat_box_minus(
                optimized_wrist_quat, base_wrist_quat
            )
            wrist_rotvec_labels = [
                "wrist_rotvec_x",
                "wrist_rotvec_y",
                "wrist_rotvec_z",
            ]
            for axis_idx, axis_name in enumerate(wrist_rotvec_labels):
                channels.append(
                    (
                        axis_name,
                        base_wrist_rotvec[:, axis_idx],
                        optimized_wrist_rotvec[:, axis_idx],
                        wrist_quat_std_np,
                    )
                )

            for joint_idx, joint_name in enumerate(robot.finger_joint_names):
                channels.append(
                    (
                        joint_name,
                        base_hand_state["finger_jvals"][:, joint_idx],
                        optimized_hand_state["finger_jvals"][:, joint_idx],
                        finger_joint_std_np,
                    )
                )

        if len(channels) == 0:
            return None

        side_axis_limits = self.gp_plot_axis_limits.setdefault(side, {})
        n_cols = 3
        n_rows = int(np.ceil(len(channels) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 2.8 * n_rows),
            sharex=True,
        )
        axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
        for channel_idx, (
            channel_name,
            base_values,
            updated_values,
            channel_std,
        ) in enumerate(channels):
            ax = axes[channel_idx // n_cols, channel_idx % n_cols]
            base_np = base_values.detach().cpu().numpy()
            updated_np = updated_values.detach().cpu().numpy()
            ax.plot(phase, base_np, label="retargeted_ref", linewidth=1.5)
            ax.plot(phase, updated_np, label="updated_ref", linewidth=1.5)
            ax.fill_between(
                phase,
                updated_np - channel_std,
                updated_np + channel_std,
                color="tab:orange",
                alpha=0.2,
                label="gp_std" if channel_idx == 0 else None,
            )
            ax.set_title(channel_name)
            ax.set_xlim(float(phase[0]), float(phase[-1]))
            y_limits = side_axis_limits.get(channel_name)
            if y_limits is None:
                lower = float(min(base_np.min(), (updated_np - channel_std).min()))
                upper = float(max(base_np.max(), (updated_np + channel_std).max()))
                if np.isclose(lower, upper):
                    padding = max(abs(lower) * 0.05, 1.0e-3)
                else:
                    padding = 0.05 * (upper - lower)
                y_limits = (lower - padding, upper + padding)
                side_axis_limits[channel_name] = y_limits
            ax.set_ylim(*y_limits)
            ax.grid(True, alpha=0.3)
            if channel_idx // n_cols == n_rows - 1:
                ax.set_xlabel("Phase")

        for unused_idx in range(len(channels), n_rows * n_cols):
            axes[unused_idx // n_cols, unused_idx % n_cols].axis("off")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.suptitle(f"Reference GP Update {update_step} ({side} hand)")
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        output_path = (
            Path(output_dir) / f"reference_gp_update_{update_step:06d}_{side}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def _build_initial_state(self) -> dict[str, Any]:
        motion = self._source_motion_dict()
        obj_pos = torch.as_tensor(
            motion["obj_pos"], dtype=torch.float32, device=self.device
        )
        obj_quat = torch.as_tensor(
            motion["obj_quat"], dtype=torch.float32, device=self.device
        )
        obj_arti = self._aligned_object_arti(motion)
        hands = {}
        for side in self.env.task_manager.active_hand_sides:
            hands[side] = self._extract_hand_state(motion=motion, side=side)
        state = {
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "obj_arti": obj_arti,
            "joint_names": self._sim_object_joint_names(),
            "hands": hands,
        }
        self._recompute_hand_state(state)
        return state

    def _source_motion_dict(self) -> dict[str, Any]:
        source = str(self.cfg.source_motion)
        original_demo = self.env.task_manager.original_demo
        if source == "retargeted":
            motion = getattr(original_demo, "retargeted_motion", None)
            if motion is None:
                raise ValueError(
                    "reference_cfg.source_motion='retargeted' requires original_demo.retargeted_motion."
                )
            return motion
        if source == "object_reference":
            if self.env.task_manager.active_hand_sides:
                raise ValueError(
                    "reference_cfg.source_motion='object_reference' cannot provide robot hand trajectories."
                )
            return original_demo.object_reference_motion
        raise ValueError(
            f"Unsupported reference_cfg.source_motion {source!r}. Expected 'retargeted' or 'object_reference'."
        )

    def _aligned_object_arti(self, motion: dict[str, Any]) -> torch.Tensor | None:
        arti_np = motion.get("obj_arti", None)
        sim_joint_names = self._sim_object_joint_names()
        if arti_np is None or len(sim_joint_names) == 0:
            return None
        motion_joint_names = [str(name) for name in motion["joint_names"]]
        joint_indices = [motion_joint_names.index(name) for name in sim_joint_names]
        return torch.as_tensor(
            arti_np[:, joint_indices], dtype=torch.float32, device=self.device
        )

    def _extract_hand_state(self, motion: dict[str, Any], side: str) -> dict[str, Any]:
        robot: Hand = self.env.physics_manager.scene_entities[side]
        mode = str(self.env.cfg.reset_cfg.retargeted_hand_reference_mode)
        if mode == "default_configuration":
            jvals = torch.zeros(
                (self.env.task_manager.total_steps, len(robot.joint_names)),
                dtype=torch.float32,
                device=self.device,
            )
            finger_indices, canonical_values = robot.canonical_finger_joint_values(
                device=self.device,
                dtype=jvals.dtype,
            )
            jvals[:, finger_indices] = canonical_values.unsqueeze(0)
            return {
                "jvals": jvals,
                "finger_jvals": None,
                "wrist_pos": None,
                "wrist_quat": None,
                "ref": None,
                "link_pos": {},
            }

        if f"{side}_jvals" not in motion or f"{side}_jnames" not in motion:
            raise ValueError(
                f"Reference motion is missing {side!r} joint values required for robot hand tracking."
            )
        motion_joint_names = [str(name) for name in motion[f"{side}_jnames"]]
        joint_indices = [motion_joint_names.index(name) for name in robot.joint_names]
        jvals = torch.as_tensor(
            motion[f"{side}_jvals"][:, joint_indices],
            dtype=torch.float32,
            device=self.device,
        )
        jvals = self._apply_hand_reference_mode(robot=robot, jvals=jvals)
        return {
            "jvals": jvals,
            "finger_jvals": None,
            "wrist_pos": None,
            "wrist_quat": None,
            "ref": None,
            "link_pos": {},
        }

    def _apply_hand_reference_mode(
        self, *, robot: Hand, jvals: torch.Tensor
    ) -> torch.Tensor:
        mode = str(self.env.cfg.reset_cfg.retargeted_hand_reference_mode)
        if mode == "retargeted":
            return jvals
        if mode not in {"canonical_fingers", "default_configuration"}:
            raise ValueError(
                "Expected reset_cfg.retargeted_hand_reference_mode to be "
                "'retargeted', 'canonical_fingers', or "
                f"'default_configuration', got {mode!r}."
            )

        expected_shape = (self.env.task_manager.total_steps, len(robot.joint_names))
        if tuple(jvals.shape) != expected_shape:
            raise ValueError(
                "Hand reference masking expected jvals shape "
                f"{expected_shape}, got {tuple(jvals.shape)}."
            )
        finger_indices, canonical_values = robot.canonical_finger_joint_values(
            device=self.device,
            dtype=jvals.dtype,
        )
        masked = (
            jvals.clone() if mode == "canonical_fingers" else torch.zeros_like(jvals)
        )
        masked[:, finger_indices] = canonical_values.unsqueeze(0)
        return masked

    def _recompute_hand_state(self, state: dict[str, Any]) -> None:
        for side, side_state in state["hands"].items():
            robot: Hand = self.env.physics_manager.scene_entities[side]
            jvals = side_state["jvals"]
            joint_dict = {
                joint.name: jvals[:, joint_idx].detach().cpu().numpy()
                for joint_idx, joint in enumerate(robot.joints)
            }
            link_fk = robot.urdf_inst.link_fk_batch(cfgs=joint_dict)
            base_pose = torch.as_tensor(
                robot.base_pose, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            base_pose = base_pose.expand(jvals.shape[0], -1, -1)

            wrist_link = robot.urdf_inst.link_map[robot.wrist_link_name]
            wrist_pose_np = link_fk.get(wrist_link)
            if wrist_pose_np is None:
                raise ValueError(
                    f"URDF FK did not return wrist poses for side {side!r}."
                )
            wrist_pose = torch.as_tensor(
                wrist_pose_np, dtype=torch.float32, device=self.device
            )
            wrist_pose_world = torch.bmm(base_pose, wrist_pose)
            side_state["wrist_pos"] = wrist_pose_world[:, :3, 3].clone()
            side_state["wrist_quat"] = self._canonicalize_quaternion_sequence(
                quat_from_matrix(wrist_pose_world[:, :3, :3]).clone()
            )

            link_pos: dict[str, torch.Tensor] = {}
            for link_name in robot.human_mapped_link_names:
                fk_link = robot.urdf_inst.link_map[link_name]
                link_pose_np = link_fk.get(fk_link)
                if link_pose_np is None:
                    raise ValueError(
                        f"URDF FK did not return poses for link {link_name!r} on side {side!r}."
                    )
                link_pose = torch.as_tensor(
                    link_pose_np, dtype=torch.float32, device=self.device
                )
                link_pose_world = torch.bmm(base_pose, link_pose)
                link_pos[link_name] = link_pose_world[:, :3, 3].clone()
            side_state["link_pos"] = link_pos

            finger_joint_indices = [
                robot.joint_names.index(joint_name)
                for joint_name in robot.finger_joint_names
            ]
            finger_joint_indices_t = torch.as_tensor(
                finger_joint_indices,
                dtype=torch.long,
                device=self.device,
            )
            finger_jvals = jvals.index_select(1, finger_joint_indices_t)
            side_state["finger_jvals"] = finger_jvals.clone()
            wrist_rot_world = matrix_from_quat(side_state["wrist_quat"])
            ref_wrist_pos = side_state["wrist_pos"] - state["obj_pos"]
            ref_wrist_rot_6d = rot6d_from_matrix(wrist_rot_world)
            side_state["ref"] = torch.cat(
                [ref_wrist_pos, ref_wrist_rot_6d, finger_jvals], dim=-1
            )

    def _select_top_rollouts(
        self, returns: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep only the best-performing trajectories and convert their returns
        # into soft weights. Those weights later become inverse-noise terms in
        # the GP so stronger trajectories contribute more.
        percentile = float(self.cfg.update_topk_percentile)
        if percentile <= 0.0 or percentile > 1.0:
            raise ValueError(
                "Expected reference_cfg.update_topk_percentile in (0, 1], "
                f"got {percentile}."
            )
        num_rollouts = int(returns.shape[0])
        num_selected = max(int(np.ceil(num_rollouts * percentile)), 1)
        num_selected = max(num_selected, int(self.cfg.min_rollouts_for_update))
        if num_selected > num_rollouts:
            raise ValueError(
                "Not enough rollout trajectories to update the optimized reference. "
                f"Need at least {num_selected}, got {num_rollouts}."
            )
        selected_indices = torch.topk(returns, k=num_selected, largest=True).indices
        selected_returns = returns.index_select(0, selected_indices)
        min_return = selected_returns.min()
        max_return = selected_returns.max()
        if torch.isclose(max_return, min_return):
            sample_weights = torch.ones_like(selected_returns)
        else:
            sample_weights = (selected_returns - min_return) / (max_return - min_return)
            sample_weights = 0.1 + 0.9 * sample_weights
        return selected_indices, sample_weights

    def _fit_translation_signal(
        self,
        *,
        base_signal: torch.Tensor,
        rollout_signal: torch.Tensor,
        rollout_timesteps: torch.Tensor,
        sample_weights: torch.Tensor,
        clip_limit: float,
        prior_std: float,
        anchors: dict[int, dict[str, Any]] | None,
        key: str,
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        # Fit a GP to residual translation values relative to the base
        # reference. We update the residual, not the absolute trajectory,
        # so the optimized motion stays anchored to the trusted baseline.
        obs_times = rollout_timesteps.reshape(-1)
        residuals = rollout_signal - base_signal[rollout_timesteps]
        obs_values = residuals.reshape(-1, residuals.shape[-1])
        obs_weights = (
            sample_weights.unsqueeze(1).expand_as(rollout_timesteps).reshape(-1)
        )
        fitted_residuals, posterior_var, posterior_std = self._gp_posterior(
            obs_times=obs_times,
            obs_values=obs_values,
            obs_weights=obs_weights,
            clip_limit=clip_limit,
            prior_std=prior_std,
            anchor_values=self._translation_anchor_residuals(anchors, key),
        )
        return base_signal + fitted_residuals, posterior_var, posterior_std

    def _fit_joint_signal(
        self,
        *,
        base_signal: torch.Tensor,
        rollout_signal: torch.Tensor,
        rollout_timesteps: torch.Tensor,
        sample_weights: torch.Tensor,
        clip_limit: float,
        prior_std: float,
        anchors: dict[int, dict[str, Any]] | None,
        key: str,
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        # Finger / articulation joints are treated the same way as translation:
        # smooth the residual over phase, then add it back to the base motion.
        obs_times = rollout_timesteps.reshape(-1)
        residuals = rollout_signal - base_signal[rollout_timesteps]
        obs_values = residuals.reshape(-1, residuals.shape[-1])
        obs_weights = (
            sample_weights.unsqueeze(1).expand_as(rollout_timesteps).reshape(-1)
        )
        fitted_residuals, posterior_var, posterior_std = self._gp_posterior(
            obs_times=obs_times,
            obs_values=obs_values,
            obs_weights=obs_weights,
            clip_limit=clip_limit,
            prior_std=prior_std,
            anchor_values=self._translation_anchor_residuals(anchors, key),
        )
        return base_signal + fitted_residuals, posterior_var, posterior_std

    def _fit_rotation_signal(
        self,
        *,
        base_quat: torch.Tensor,
        rollout_quat: torch.Tensor,
        rollout_timesteps: torch.Tensor,
        sample_weights: torch.Tensor,
        clip_limit: float,
        prior_std: float,
        anchors: dict[int, dict[str, Any]] | None,
        key: str,
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        # Rotations are smoothed in tangent space. We first convert
        # base->rollout quaternion differences into rotation vectors, run the
        # GP on those 3D residuals, and then compose the smoothed residual back
        # onto the base quaternion trajectory.
        base_at_obs = base_quat[rollout_timesteps]
        residual_rotvec = quat_box_minus(
            rollout_quat.reshape(-1, 4),
            base_at_obs.reshape(-1, 4),
        ).reshape(rollout_quat.shape[0], rollout_quat.shape[1], 3)
        obs_times = rollout_timesteps.reshape(-1)
        obs_values = residual_rotvec.reshape(-1, 3)
        obs_weights = (
            sample_weights.unsqueeze(1).expand_as(rollout_timesteps).reshape(-1)
        )
        fitted_rotvec, posterior_var, posterior_std = self._gp_posterior(
            obs_times=obs_times,
            obs_values=obs_values,
            obs_weights=obs_weights,
            clip_limit=clip_limit,
            prior_std=prior_std,
            anchor_values=self._rotation_anchor_residuals(anchors, key),
        )
        fitted_delta_quat = quat_from_rotvec(fitted_rotvec)
        fitted_quat = quat_mul(fitted_delta_quat, base_quat)
        return fitted_quat, posterior_var, posterior_std

    def _gp_posterior(
        self,
        *,
        obs_times: torch.Tensor,
        obs_values: torch.Tensor,
        obs_weights: torch.Tensor,
        clip_limit: float,
        prior_std: float,
        anchor_values: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        # Closed-form GP regression over normalized timestep / phase.
        # The rollout weights act as inverse observation noise. Built-in start/end
        # pins and any custom anchors are injected as extra observations with a
        # configurable weight so they bias the GP softly rather than hard-clamping
        # the trajectory.
        if prior_std <= 0.0:
            raise ValueError(f"Expected GP prior_std > 0, got {prior_std}.")
        prior_variance = float(prior_std) ** 2
        query_times = self._normalized_query_times()
        obs_times_f = self._normalize_timesteps(obs_times)
        obs_values_f = obs_values.to(self.device, dtype=torch.float32)
        obs_weights_f = obs_weights.to(self.device, dtype=torch.float32).clamp(
            min=1.0e-3
        )
        if anchor_values is not None:
            anchor_times, anchor_residuals, anchor_weights = anchor_values
            anchor_times = self._normalize_timesteps(anchor_times.to(self.device))
            obs_times_f = torch.cat([obs_times_f, anchor_times], dim=0)
            obs_values_f = torch.cat(
                [obs_values_f, anchor_residuals.to(self.device, dtype=torch.float32)],
                dim=0,
            )
            obs_weights_f = torch.cat(
                [
                    obs_weights_f,
                    anchor_weights.to(self.device, dtype=torch.float32).clamp(
                        min=1.0e-3
                    ),
                ],
                dim=0,
            )

        kernel_xx = prior_variance * self._rbf_kernel(obs_times_f, obs_times_f)
        noise = float(self.cfg.gp_observation_noise)
        kernel_xx = kernel_xx + torch.diag(noise / obs_weights_f)
        kernel_qx = prior_variance * self._rbf_kernel(query_times, obs_times_f)
        solve = torch.linalg.solve(kernel_xx, obs_values_f)
        posterior_mean = kernel_qx @ solve

        solve_kernel = torch.linalg.solve(kernel_xx, kernel_qx.transpose(0, 1))
        kernel_qq_diag = torch.full(
            (query_times.shape[0],),
            prior_variance,
            dtype=torch.float32,
            device=self.device,
        )
        posterior_var = kernel_qq_diag - (kernel_qx * solve_kernel.transpose(0, 1)).sum(
            dim=-1
        )
        posterior_var = posterior_var.clamp(min=0.0)

        # Residual magnitude clipping keeps one update from drifting too far
        # away from the base reference, even if a few rollout samples are noisy.
        norm = torch.linalg.norm(posterior_mean, dim=-1, keepdim=True)
        scale = torch.clamp(
            torch.full_like(norm, clip_limit) / norm.clamp(min=1.0e-8),
            max=1.0,
        )
        posterior_mean = posterior_mean * scale
        posterior_std = torch.sqrt(posterior_var.clamp(min=0.0))
        return posterior_mean, float(posterior_var.mean().item()), posterior_std

    def _rbf_kernel(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        lengthscale = float(self.cfg.gp_lengthscale)
        if lengthscale <= 0.0:
            raise ValueError(
                f"Expected reference_cfg.gp_lengthscale > 0, got {lengthscale}."
            )
        delta = x_a[:, None] - x_b[None, :]
        return torch.exp(-0.5 * (delta / lengthscale) ** 2)

    def _normalized_query_times(self) -> torch.Tensor:
        num_steps = int(self.env.task_manager.total_steps)
        if num_steps <= 1:
            return torch.zeros((num_steps,), dtype=torch.float32, device=self.device)
        return torch.linspace(0.0, 1.0, steps=num_steps, device=self.device)

    def _normalize_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        denom = max(int(self.env.task_manager.total_steps) - 1, 1)
        return timesteps.to(self.device, dtype=torch.float32) / float(denom)

    def _ema_update(
        self, previous_state: dict[str, Any], candidate_state: dict[str, Any]
    ) -> dict[str, Any]:
        alpha = float(self.cfg.ema_alpha)
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError(
                f"Expected reference_cfg.ema_alpha in (0, 1], got {alpha}."
            )

        # Object state stays fixed to the original demo trajectory. The only EMA
        # update we apply is on the robot reference trajectory.
        updated_state = self._clone_state(previous_state)
        for side in updated_state["hands"]:
            updated_state["hands"][side]["jvals"] = (1.0 - alpha) * previous_state[
                "hands"
            ][side]["jvals"] + alpha * candidate_state["hands"][side]["jvals"]
            updated_state["hands"][side]["finger_jvals"] = (
                1.0 - alpha
            ) * previous_state["hands"][side]["finger_jvals"] + alpha * candidate_state[
                "hands"
            ][
                side
            ][
                "finger_jvals"
            ]
            updated_state["hands"][side]["wrist_pos"] = (1.0 - alpha) * previous_state[
                "hands"
            ][side]["wrist_pos"] + alpha * candidate_state["hands"][side]["wrist_pos"]
            updated_state["hands"][side]["wrist_quat"] = self._slerp_quat(
                previous_state["hands"][side]["wrist_quat"],
                candidate_state["hands"][side]["wrist_quat"],
                alpha,
            )
        return updated_state

    def _slerp_quat(
        self, start_quat: torch.Tensor, end_quat: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        rel_rotvec = quat_box_minus(end_quat, start_quat)
        delta_quat = quat_from_rotvec(rel_rotvec * float(alpha))
        return quat_mul(delta_quat, start_quat)

    def _builtin_pinned_timesteps(self) -> set[int]:
        pinned_timesteps = set(int(t) for t in self.cfg.pinned_timesteps)
        if self.cfg.pin_start:
            pinned_timesteps.add(0)
        if self.cfg.pin_end:
            pinned_timesteps.add(self.env.task_manager.total_steps - 1)
        for timestep in pinned_timesteps:
            if timestep < 0 or timestep >= self.env.task_manager.total_steps:
                raise ValueError(
                    f"Pinned timestep {timestep} is outside [0, {self.env.task_manager.total_steps - 1}]."
                )
        return pinned_timesteps

    def _apply_custom_anchors(
        self, state: dict[str, Any], anchors: dict[int, dict[str, Any]] | None
    ) -> None:
        if anchors is None:
            return
        for timestep, values in anchors.items():
            if timestep < 0 or timestep >= self.env.task_manager.total_steps:
                raise ValueError(
                    f"Anchor timestep {timestep} is outside [0, {self.env.task_manager.total_steps - 1}]."
                )
            if "obj_pos" in values:
                state["obj_pos"][timestep] = self._to_float_tensor(
                    values["obj_pos"]
                ).reshape(3)
            if "obj_quat" in values:
                quat = self._to_float_tensor(values["obj_quat"]).reshape(4)
                state["obj_quat"][timestep] = quat / quat.norm().clamp(min=1.0e-8)
            if "obj_arti" in values:
                if state["obj_arti"] is None:
                    raise ValueError(
                        "Received obj_arti anchor for a non-articulated object reference."
                    )
                state["obj_arti"][timestep] = self._to_float_tensor(
                    values["obj_arti"]
                ).reshape(-1)
            hand_values = values.get("hands", {})
            for side, side_values in hand_values.items():
                if side not in state["hands"]:
                    raise ValueError(f"Unknown hand side {side!r} in anchors.")
                if "jvals" in side_values:
                    state["hands"][side]["jvals"][timestep] = self._to_float_tensor(
                        side_values["jvals"]
                    ).reshape(-1)
                if "finger_jvals" in side_values:
                    state["hands"][side]["finger_jvals"][timestep] = (
                        self._to_float_tensor(side_values["finger_jvals"]).reshape(-1)
                    )
                if "wrist_pos" in side_values:
                    state["hands"][side]["wrist_pos"][timestep] = self._to_float_tensor(
                        side_values["wrist_pos"]
                    ).reshape(3)
                if "wrist_quat" in side_values:
                    quat = self._to_float_tensor(side_values["wrist_quat"]).reshape(4)
                    state["hands"][side]["wrist_quat"][
                        timestep
                    ] = quat / quat.norm().clamp(min=1.0e-8)

    def _builtin_pin_observation_weight(self) -> float:
        variance_scale = float(self.cfg.builtin_pin_variance_scale)
        if variance_scale <= 0.0:
            raise ValueError(
                "Expected reference_cfg.builtin_pin_variance_scale > 0, got "
                f"{self.cfg.builtin_pin_variance_scale}."
            )
        return 1.0 / variance_scale

    def _custom_anchor_observation_weight(self) -> float:
        anchor_weight = float(self.cfg.anchor_observation_weight)
        if anchor_weight <= 0.0:
            raise ValueError(
                "Expected reference_cfg.anchor_observation_weight > 0, got "
                f"{self.cfg.anchor_observation_weight}."
            )
        return anchor_weight

    def _translation_anchor_residuals(
        self, anchors: dict[int, dict[str, Any]] | None, key: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        anchor_times = []
        anchor_residuals = []
        anchor_weights = []
        builtin_pin_weight = self._builtin_pin_observation_weight()
        for timestep in sorted(self._builtin_pinned_timesteps()):
            if self._has_custom_anchor_for_key(anchors, timestep, key):
                continue
            residual = self._builtin_translation_anchor_residual(timestep, key)
            if residual is None:
                continue
            anchor_times.append(timestep)
            anchor_residuals.append(residual)
            anchor_weights.append(builtin_pin_weight)
        if anchors is None:
            anchors = {}
        custom_anchor_weight = self._custom_anchor_observation_weight()
        for timestep, values in anchors.items():
            if key == "obj_pos" and "obj_pos" in values:
                anchor_times.append(timestep)
                anchor_residuals.append(
                    self._to_float_tensor(values["obj_pos"]).reshape(3)
                    - self.base_state["obj_pos"][timestep]
                )
                anchor_weights.append(custom_anchor_weight)
            if key.endswith("_finger_jvals"):
                side = key.replace("_finger_jvals", "")
                side_values = values.get("hands", {}).get(side, {})
                if "finger_jvals" in side_values:
                    anchor_times.append(timestep)
                    anchor_residuals.append(
                        self._to_float_tensor(side_values["finger_jvals"]).reshape(-1)
                        - self.base_state["hands"][side]["finger_jvals"][timestep]
                    )
                    anchor_weights.append(custom_anchor_weight)
            if key.endswith("_wrist_pos"):
                side = key.replace("_wrist_pos", "")
                side_values = values.get("hands", {}).get(side, {})
                if "wrist_pos" in side_values:
                    anchor_times.append(timestep)
                    anchor_residuals.append(
                        self._to_float_tensor(side_values["wrist_pos"]).reshape(3)
                        - self.base_state["hands"][side]["wrist_pos"][timestep]
                    )
                    anchor_weights.append(custom_anchor_weight)
        if len(anchor_times) == 0:
            return None
        return (
            torch.as_tensor(anchor_times, dtype=torch.long, device=self.device),
            torch.stack(anchor_residuals, dim=0).to(self.device),
            torch.as_tensor(anchor_weights, dtype=torch.float32, device=self.device),
        )

    def _rotation_anchor_residuals(
        self, anchors: dict[int, dict[str, Any]] | None, key: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        anchor_times = []
        anchor_residuals = []
        anchor_weights = []
        builtin_pin_weight = self._builtin_pin_observation_weight()
        for timestep in sorted(self._builtin_pinned_timesteps()):
            if self._has_custom_anchor_for_key(anchors, timestep, key):
                continue
            residual = self._builtin_rotation_anchor_residual(timestep, key)
            if residual is None:
                continue
            anchor_times.append(timestep)
            anchor_residuals.append(residual)
            anchor_weights.append(builtin_pin_weight)
        if anchors is None:
            anchors = {}
        custom_anchor_weight = self._custom_anchor_observation_weight()
        for timestep, values in anchors.items():
            if key == "obj_quat" and "obj_quat" in values:
                anchor_quat = self._to_float_tensor(values["obj_quat"]).reshape(4)
                anchor_times.append(timestep)
                anchor_residuals.append(
                    quat_box_minus(
                        anchor_quat.unsqueeze(0),
                        self.base_state["obj_quat"][timestep].unsqueeze(0),
                    ).squeeze(0)
                )
                anchor_weights.append(custom_anchor_weight)
            if key.endswith("_wrist_quat"):
                side = key.replace("_wrist_quat", "")
                side_values = values.get("hands", {}).get(side, {})
                if "wrist_quat" not in side_values:
                    continue
                anchor_quat = self._to_float_tensor(side_values["wrist_quat"]).reshape(
                    4
                )
                anchor_times.append(timestep)
                anchor_residuals.append(
                    quat_box_minus(
                        anchor_quat.unsqueeze(0),
                        self.base_state["hands"][side]["wrist_quat"][
                            timestep
                        ].unsqueeze(0),
                    ).squeeze(0)
                )
                anchor_weights.append(custom_anchor_weight)
        if len(anchor_times) == 0:
            return None
        return (
            torch.as_tensor(anchor_times, dtype=torch.long, device=self.device),
            torch.stack(anchor_residuals, dim=0).to(self.device),
            torch.as_tensor(anchor_weights, dtype=torch.float32, device=self.device),
        )

    def _has_custom_anchor_for_key(
        self, anchors: dict[int, dict[str, Any]] | None, timestep: int, key: str
    ) -> bool:
        if anchors is None or timestep not in anchors:
            return False
        values = anchors[timestep]
        if key == "obj_pos":
            return "obj_pos" in values
        if key == "obj_quat":
            return "obj_quat" in values
        if key.endswith("_finger_jvals"):
            side = key.replace("_finger_jvals", "")
            return "finger_jvals" in values.get("hands", {}).get(side, {})
        if key.endswith("_wrist_pos"):
            side = key.replace("_wrist_pos", "")
            return "wrist_pos" in values.get("hands", {}).get(side, {})
        if key.endswith("_wrist_quat"):
            side = key.replace("_wrist_quat", "")
            return "wrist_quat" in values.get("hands", {}).get(side, {})
        return False

    def _builtin_translation_anchor_residual(
        self, timestep: int, key: str
    ) -> torch.Tensor | None:
        if key == "obj_pos":
            base_value = self.base_state["obj_pos"][timestep]
            return torch.zeros_like(base_value)
        if key.endswith("_finger_jvals"):
            side = key.replace("_finger_jvals", "")
            base_value = self.base_state["hands"][side]["finger_jvals"][timestep]
            return torch.zeros_like(base_value)
        if key.endswith("_wrist_pos"):
            side = key.replace("_wrist_pos", "")
            base_value = self.base_state["hands"][side]["wrist_pos"][timestep]
            return torch.zeros_like(base_value)
        return None

    def _builtin_rotation_anchor_residual(
        self, timestep: int, key: str
    ) -> torch.Tensor | None:
        if key == "obj_quat":
            base_quat = self.base_state["obj_quat"][timestep]
            return torch.zeros((3,), dtype=base_quat.dtype, device=base_quat.device)
        if key.endswith("_wrist_quat"):
            side = key.replace("_wrist_quat", "")
            base_quat = self.base_state["hands"][side]["wrist_quat"][timestep]
            return torch.zeros((3,), dtype=base_quat.dtype, device=base_quat.device)
        return None

    def _compute_reference_stats(self, state: dict[str, Any]) -> dict[str, float]:
        stats = {}
        pos_residual = state["obj_pos"] - self.base_state["obj_pos"]
        stats["optimized_reference_obj_pos_residual_m"] = float(
            pos_residual.norm(dim=-1).mean().item()
        )
        quat_residual = quat_box_minus(state["obj_quat"], self.base_state["obj_quat"])
        stats["optimized_reference_obj_rot_residual_rad"] = float(
            quat_residual.norm(dim=-1).mean().item()
        )
        for side in state["hands"]:
            joint_residual = (
                state["hands"][side]["jvals"] - self.base_state["hands"][side]["jvals"]
            )
            stats[f"optimized_reference_{side}_joint_residual_rad"] = float(
                joint_residual.abs().mean().item()
            )
        return stats

    def _clone_state(self, state: dict[str, Any]) -> dict[str, Any]:
        cloned_hands = {}
        for side, side_state in state["hands"].items():
            cloned_hands[side] = {
                "jvals": side_state["jvals"].clone(),
                "finger_jvals": (
                    None
                    if side_state["finger_jvals"] is None
                    else side_state["finger_jvals"].clone()
                ),
                "wrist_pos": (
                    None
                    if side_state["wrist_pos"] is None
                    else side_state["wrist_pos"].clone()
                ),
                "wrist_quat": (
                    None
                    if side_state["wrist_quat"] is None
                    else side_state["wrist_quat"].clone()
                ),
                "ref": None if side_state["ref"] is None else side_state["ref"].clone(),
                "link_pos": {
                    name: value.clone()
                    for name, value in side_state["link_pos"].items()
                },
            }
        return {
            "obj_pos": state["obj_pos"].clone(),
            "obj_quat": state["obj_quat"].clone(),
            "obj_arti": (
                None if state["obj_arti"] is None else state["obj_arti"].clone()
            ),
            "joint_names": list(state["joint_names"]),
            "hands": cloned_hands,
        }

    def _select_rollout_tensor(
        self,
        values: torch.Tensor | np.ndarray,
        selected_indices: torch.Tensor,
        trailing_dim: int,
    ) -> torch.Tensor:
        tensor = self._to_float_tensor(values)
        if tensor.ndim != 3 or tensor.shape[2] != trailing_dim:
            raise ValueError(
                f"Expected rollout tensor with shape (num_rollouts, horizon, {trailing_dim}), "
                f"got {tuple(tensor.shape)}."
            )
        return tensor.index_select(0, selected_indices.to(tensor.device)).to(
            self.device
        )

    def _sim_object_joint_names(self) -> list[str]:
        asset: ArtObject = self.env.physics_manager.scene_entities["object"]
        return [joint.name for joint in asset.joints]

    def _to_float_tensor(
        self, value: torch.Tensor | np.ndarray | list[float]
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _to_long_tensor(self, value: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=self.device, dtype=torch.long)
        return torch.as_tensor(value, dtype=torch.long, device=self.device)

    def _canonicalize_quaternion_sequence(self, quats: torch.Tensor) -> torch.Tensor:
        if quats.ndim != 2 or quats.shape[1] != 4:
            raise ValueError(
                "Expected quaternion sequence with shape (num_steps, 4), got "
                f"{tuple(quats.shape)}."
            )
        canonical = quats.clone()
        for timestep in range(1, canonical.shape[0]):
            if torch.dot(canonical[timestep - 1], canonical[timestep]) < 0.0:
                canonical[timestep] = -canonical[timestep]
        return canonical

    def _reconstruct_hand_joint_trajectory(self, state: dict[str, Any]) -> None:
        # Rebuild full robot joints from GP wrist pose + finger joints using a
        # sequential IK rollout. The previous timestep solution becomes q_init
        # for the next timestep so the arm stays on a consistent configuration branch.
        for side, side_state in state["hands"].items():
            robot: Hand = self.env.physics_manager.scene_entities[side]
            if (
                side_state["wrist_pos"] is None
                or side_state["wrist_quat"] is None
                or side_state["finger_jvals"] is None
            ):
                raise ValueError(
                    f"Missing wrist/finger GP state for hand side {side!r}."
                )
            finger_indices = [
                robot.joint_names.index(name) for name in robot.finger_joint_names
            ]
            arm_ik_dof_indices = np.asarray(
                [robot.dof_map[idx] for idx in robot.wrist_control_joint_idx],
                dtype=np.int64,
            )
            qpos = side_state["jvals"].clone()
            q_init = (
                self.optimized_state["hands"][side]["jvals"][0].unsqueeze(0).clone()
            )
            ik_batch_size = int(self.env.num_envs)
            for timestep in range(qpos.shape[0]):
                q_init[:, finger_indices] = side_state["finger_jvals"][
                    timestep
                ].unsqueeze(0)
                batched_target_pos = (
                    side_state["wrist_pos"][timestep]
                    .unsqueeze(0)
                    .repeat(ik_batch_size, 1)
                )
                batched_target_quat = (
                    side_state["wrist_quat"][timestep]
                    .unsqueeze(0)
                    .repeat(ik_batch_size, 1)
                )
                batched_q_init = q_init.repeat(ik_batch_size, 1)
                ik_qpos = self.env.physics_manager.solve_entity_ik(
                    robot.robot,
                    link_name=robot.wrist_link_name,
                    target_pos=batched_target_pos,
                    target_quat=batched_target_quat,
                    dofs_idx_local=arm_ik_dof_indices.tolist(),
                    init_qpos=batched_q_init,
                    init_qpos_dofs_idx_local=robot.dof_map,
                )
                qpos[timestep] = ik_qpos[0]
                qpos[timestep, finger_indices] = side_state["finger_jvals"][timestep]
                q_init = qpos[timestep].unsqueeze(0).clone()
            side_state["jvals"] = qpos
        self._recompute_hand_state(state)
