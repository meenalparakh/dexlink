from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import torch

from dexlink.src.rl.sim.robot import Hand

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import EnvConfig
    from dexlink.src.rl.sim.art_object import ArtObject
    from dexlink.src.rl.sim.env import Env


class ResetManager:
    def __init__(self, cfg: EnvConfig, env: Env):
        self.cfg = cfg
        self.env = env
        self.scene = env.scene_manager

    def initialize(self):
        self.process_retargeted_results()
        self.print_domain_randomization_summary()
        if self.cfg.reset_cfg.dr:
            self.apply_domain_randomization(
                np.arange(self.env.num_envs, dtype=np.int64)
            )

    def print_domain_randomization_summary(self):
        reset_cfg = self.cfg.reset_cfg
        lines = ["[DomainRandomization] Summary"]
        lines.append(f"enabled: {reset_cfg.dr}")
        lines.append(
            f"ranges: stiffness={reset_cfg.dr_stiffness}, damping={reset_cfg.dr_damping}, "
            f"friction={reset_cfg.dr_friction}, object_mass={reset_cfg.dr_object_mass}"
        )

        object_entity: ArtObject = self.scene.scene_entities["object"]
        object_summary = object_entity.domain_randomization_summary()
        lines.append(
            "object.mass available: "
            + self._format_dr_names(object_summary["mass_available"])
        )
        lines.append(
            "object.mass skipped: "
            + self._format_dr_names(object_summary["mass_skipped"])
        )

        for side in self.scene.active_hand_sides:
            hand: Hand = self.scene.scene_entities[side]
            hand_summary = hand.domain_randomization_summary()
            lines.append(
                f"{side}.stiffness available: "
                + self._format_dr_names(hand_summary["stiffness_available"])
            )
            lines.append(
                f"{side}.stiffness skipped: "
                + self._format_dr_names(hand_summary["stiffness_skipped"])
            )
            lines.append(
                f"{side}.damping available: "
                + self._format_dr_names(hand_summary["damping_available"])
            )
            lines.append(
                f"{side}.damping skipped: "
                + self._format_dr_names(hand_summary["damping_skipped"])
            )
            lines.append(
                f"{side}.friction available: "
                + self._format_dr_names(hand_summary["friction_available"])
            )
            lines.append(
                f"{side}.friction skipped: "
                + self._format_dr_names(hand_summary["friction_skipped"])
            )
        print("\n".join(lines))

    def _format_dr_names(self, names: list[str]) -> str:
        if len(names) == 0:
            return "none"
        return ", ".join(names)

    def apply_domain_randomization(self, env_ids: np.ndarray):
        if not self.cfg.reset_cfg.dr or len(env_ids) == 0:
            return

        object_entity: ArtObject = self.scene.scene_entities["object"]
        object_entity.randomize_dynamics(self.cfg.reset_cfg, env_ids)

        for side in self.scene.active_hand_sides:
            hand: Hand = self.scene.scene_entities[side]
            hand.randomize_dynamics(self.cfg.reset_cfg, env_ids)

    def resample_random_force(self, env_ids: np.ndarray):
        if self.scene.cfg.apply_random_force:
            self.scene.resample_random_force(env_ids=env_ids)

    def process_retargeted_results(self):
        motion = self.scene.original_demo.retargeted_motion
        if motion is None:
            print("No retargeted motion found for ResetManager")
            return

        print("Processing retargeted results for ResetManager")
        self.frame_pos = torch.from_numpy(motion["obj_pos"])
        self.frame_quat = torch.from_numpy(motion["obj_quat"])
        if (motion["obj_arti"] is not None) and (len(motion["joint_names"]) > 0):
            sim_asset: ArtObject = self.scene.scene_entities["object"]
            sim_jnames = [j.name for j in sim_asset.joints]
            retargeted_jnames = motion["joint_names"]
            retargeted_jvals = motion["obj_arti"]
            joint_indices = [retargeted_jnames.index(jn) for jn in sim_jnames]

            arti = retargeted_jvals[:, joint_indices]
            self.frame_arti = torch.from_numpy(arti).float()
        else:
            self.frame_arti = None

        for side in ("left", "right"):
            setattr(self, f"frame_{side}_jvals", None)
            if not self.scene.has_hand(side):
                continue
            side_jnames = motion[f"{side}_jnames"]
            side_jvals = motion[f"{side}_jvals"]
            sim_robot: Hand = self.scene.scene_entities[side]
            sim_jnames = sim_robot.joint_names
            jidx_map = [side_jnames.index(jn) for jn in sim_jnames]
            frame_jvals = torch.from_numpy(side_jvals[:, jidx_map]).float()
            setattr(self, f"frame_{side}_jvals", frame_jvals)

    def normalize_env_ids(self, env_ids) -> np.ndarray:
        """Normalize env_ids to a numpy array of int64 indices.

        Accepts int, list/sequence, numpy array, torch tensor, or None (meaning all envs).
        """
        if isinstance(env_ids, (int, np.integer)):
            env_ids = np.array([int(env_ids)], dtype=np.int64)
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().numpy().astype(np.int64)
        elif env_ids is None:
            env_ids = np.arange(self.env.num_envs, dtype=np.int64)
        else:
            env_ids = np.array(env_ids, dtype=np.int64)
        return env_ids

    def set_object_state_from_retargeted_results(
        self, frame_idx: np.ndarray, env_ids: np.ndarray
    ):
        noise = self.cfg.reset_cfg.obj_noise
        pos = self.frame_pos[frame_idx].to(self.env.device)
        quat = self.frame_quat[frame_idx].to(self.env.device)
        jpos_tensor = (
            None
            if self.frame_arti is None
            else self.frame_arti[frame_idx].to(self.env.device)
        )

        if noise > 0.0:
            pos += torch.randn_like(pos) * noise
            quat_noise = torch.randn_like(quat) * noise
            quat += quat_noise
            quat = quat / torch.norm(quat, dim=-1, keepdim=True)
            if jpos_tensor is not None:
                jpos_tensor += torch.randn_like(jpos_tensor) * noise

        asset: ArtObject = self.scene.scene_entities["object"]
        asset.set_state(pos=pos, quat=quat, jpos=jpos_tensor, env_idx=env_ids)
        return pos, quat, jpos_tensor

    def set_robot_state_from_retargeted_results(
        self,
        frame_idx: np.ndarray,
        env_ids: np.ndarray,
        random_actions: Dict[str, torch.Tensor] | None = None,
    ):
        noise = self.cfg.reset_cfg.hand_noise
        # print(
        #     "[ResetManager.set_robot_state_from_retargeted_results] "
        #     f"env_ids.shape={np.asarray(env_ids).shape}, "
        #     f"env_ids={np.asarray(env_ids)}, "
        #     f"frame_idx.shape={np.asarray(frame_idx).shape}, "
        #     f"frame_idx={np.asarray(frame_idx)}"
        # )
        for side in self.scene.active_hand_sides:
            jpos = getattr(self, f"frame_{side}_jvals")[frame_idx].to(self.env.device)
            # print(
            #     "[ResetManager.set_robot_state_from_retargeted_results] "
            #     f"side={side}, raw_jpos.shape={tuple(jpos.shape)}"
            # )
            if noise > 0.0:
                jpos += torch.randn_like(jpos) * noise

            robot: Hand = self.scene.scene_entities[side]

            if self.cfg.reset_cfg.wrist_only_retargeting:
                root_joints = robot.root_joints
                if len(root_joints) == 0:
                    root_joints = robot.panda_joints
                    assert (
                        len(root_joints) > 0
                    ), f"No root joints found for wrist-only retargeting for robot {robot}"

                jpos_mask = torch.tensor(
                    [jname in root_joints for jname in robot.joint_names],
                    dtype=torch.bool,
                    device=self.env.device,
                ).unsqueeze(0)
                jpos = torch.where(jpos_mask, jpos, torch.zeros_like(jpos))

            # print(
            #     "[ResetManager.set_robot_state_from_retargeted_results] "
            #     f"side={side}, final_jpos.shape={tuple(jpos.shape)}, "
            #     f"expected_rows={len(env_ids)}"
            # )
            robot.set_robot_state(jpos=jpos, env_idx=env_ids)

    # ///////////////////////////////// used when no retargeted results /////////////////////////////////
    def set_object_state(
        self, frame_idx: np.ndarray, env_ids: np.ndarray, noise: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:

        obj_trans = self.scene.original_demo.object_reference_motion["obj_pos"]
        obj_quat = self.scene.original_demo.object_reference_motion["obj_quat"]
        obj_arti = self.scene.original_demo.object_reference_motion["obj_arti"]
        joint_names = self.scene.original_demo.object_reference_motion["joint_names"]

        pos = torch.from_numpy(obj_trans[frame_idx]).to(self.env.device)
        quat = torch.from_numpy(obj_quat[frame_idx]).to(self.env.device)
        if obj_arti is not None and len(joint_names) > 0:
            cols = [
                torch.from_numpy(obj_arti[:, i][frame_idx])
                .to(self.env.device)
                .unsqueeze(-1)
                for i, _ in enumerate(joint_names)
            ]
            jpos_tensor = torch.cat(cols, dim=-1)
        else:
            jpos_tensor = None

        if noise > 0.0:
            pos += torch.randn_like(pos) * noise
            # For quaternion, add noise and renormalize
            quat_noise = torch.randn_like(quat) * noise
            quat += quat_noise
            quat = quat / torch.norm(quat, dim=-1, keepdim=True)
            if jpos_tensor is not None:
                jpos_tensor += torch.randn_like(jpos_tensor) * noise

        asset: ArtObject = self.scene.scene_entities["object"]
        asset.set_state(pos=pos, quat=quat, jpos=jpos_tensor, env_idx=env_ids)
        return pos, quat, jpos_tensor

    def set_robot_state(
        self,
        frame_idx: np.ndarray,
        env_ids: np.ndarray,
        random_actions: Dict[str, torch.Tensor] | None = None,
        noise: float = 0.0,
    ) -> None:
        for hand_side, robot in self.scene.scene_entities.items():
            if not isinstance(robot, Hand):
                continue
            robot: Hand
            assert (
                random_actions is not None
            ), "random_actions must be provided when not resetting with reference motion"
            jpos = random_actions[hand_side].to(self.env.device)
            robot.set_robot_state(jpos=jpos, env_idx=env_ids)

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////
