from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from urdfpy import URDF

from dexlink.src.rl.sim.art_object import ArtObject
from dexlink.utils.retargeting.math_utils import (
    matrix_from_quat,
    quat_from_matrix,
    rot6d_from_matrix,
)
from dexlink.utils.retargeting.mesh_utils import (
    compute_bps,
    compute_kpts_traj,
    construct_pytorch_mesh,
    sample_kpts_with_normals,
    sample_unit_ball,
)

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import CommandsCfg
    from dexlink.src.rl.sim.env import Env
    from dexlink.src.rl.sim.robot import Hand
    from dexlink.src.retargeting.data.process_arctic import ArcticProcessor


class CommandsManager:
    def __init__(self, cfg: CommandsCfg, env: Env):
        self.cfg = cfg
        self.env = env

        self.original_demo: ArcticProcessor = self.env.scene_manager.original_demo

        self.obs_commands_buffer = []
        self.obs_command_terms = []
        self.obs_command_dims = []
        self.process_pcd = self.cfg.use_pcd and self.cfg.use_bps

        if self.cfg.use_gt_pos:
            print("Using GT Position commands, requires retargeted motion")
            self._construct_gt_pos()
            self.obs_command_terms.append("gt_pos")
            self.obs_command_dims.append(3)

        if self.cfg.use_gt_rot:
            print("Using GT Rotation commands, requires retargeted motion")
            rot_dim = 4
            if self.cfg.rot_format == "6d":
                rot_dim = 6
            self._construct_gt_rot()
            self.obs_command_terms.append("gt_rot")
            self.obs_command_dims.append(rot_dim)

        if self.cfg.use_gt_arti:
            print("Using GT Articulation commands, requires retargeted motion")
            asset: ArtObject = self.env.scene_manager.scene_entities["object"]
            arti_dim = len(asset.dof_map)
            if arti_dim > 0:
                self._construct_gt_arti()
                self.obs_command_terms.append("gt_arti")
                self.obs_command_dims.append(arti_dim)
            else:
                print("Skipping gt_arti commands: object has no movable joints.")

        if self.cfg.use_pcd:
            pcd_time_series = self._construct_pcd()
            pcd_dim = pcd_time_series.shape[-1]

            self.obs_command_terms.append("pcd")
            self.obs_command_dims.append(pcd_dim)

        # this is different from rel bps, where the bps are always hand points
        # here bps is the actual bps representation of the object point cloud
        # this is not appended to command buffer here, but called by the observation processor in the agent
        if self.process_pcd:
            self._construct_actual_bps()
        #     bps_dim = self.actual_bps.shape[-1]
        #     self.obs_command_terms.append("pcd")
        #     self.obs_command_dims.append(bps_dim)
        # else:

        if self.cfg.use_rel_bps:
            # the following also tackles command terms and dims, along with the buffer
            self._construct_obj_bps()

        self.check_commands_buffer()

        self.obs_command_tensor = self._construct_reference_obs_commands().to("cpu")
        self.policy_command_tensor = self._construct_reference_policy_commands().to(
            "cpu"
        )

        if not self.env.cfg.obs_cfg.use_command_timestep:
            self.env_obs_commands = torch.zeros(
                self.env.num_envs,
                self.obs_command_tensor.shape[-1],
                device=self.env.device,
            )
        else:
            self.env_obs_commands = None

    def _ensure_retargeted_hand_link_pos(self) -> None:
        if not self.cfg.use_retargeted_motion:
            return

        ref_motion = self.original_demo.retargeted_motion
        if ref_motion is None:
            raise ValueError(
                "commands_cfg.use_retargeted_motion=True requires retargeted motion."
            )

        need_left = (
            "left" in self.env.scene_manager.scene_entities
            and "left_link_pos" not in ref_motion
        )
        need_right = (
            "right" in self.env.scene_manager.scene_entities
            and "right_link_pos" not in ref_motion
        )
        if not (need_left or need_right):
            return

        self.original_demo.process_retargeted_motion(
            left_hand=self.env.scene_manager.scene_entities.get("left"),
            right_hand=self.env.scene_manager.scene_entities.get("right"),
        )

    def _reference_hand_points_tensor(self, side: str, hand: Hand) -> torch.Tensor:
        if not self.cfg.use_retargeted_motion:
            human_joint_names = hand.human_joint_names
            joint_indices = [
                self.original_demo.hand_joint_map[name] for name in human_joint_names
            ]
            jpos = self.original_demo.world_coord[f"joints.{side}"][:, joint_indices, :]
            return torch.from_numpy(jpos).float().to(self.env.device)

        self._ensure_retargeted_hand_link_pos()
        ref_motion = self.original_demo.retargeted_motion
        assert ref_motion is not None

        side_link_pos = ref_motion.get(f"{side}_link_pos", None)
        if side_link_pos is None:
            raise ValueError(
                f"Retargeted motion missing {side}_link_pos. "
                "Ensure retargeted hand FK was processed."
            )

        ordered = np.stack(
            [side_link_pos[link_name] for link_name in hand.human_mapped_link_names],
            axis=1,
        )
        return torch.from_numpy(ordered).float().to(self.env.device)

    def check_commands_buffer(self):
        # sanity check dims with buffer values
        for term, dim, tensor in zip(
            self.obs_command_terms, self.obs_command_dims, self.obs_commands_buffer
        ):
            assert (
                dim == tensor.shape[-1]
            ), f"Dimension mismatch for term '{term}': expected {dim}, got {tensor.shape[-1]}"
            # assert tensor.device == torch.device('cpu'), f"Tensor for term '{term}' is not on CPU"

    def _construct_reference_policy_commands(self):
        # these use actual_bps if needed
        if self.process_pcd:
            # instead of pcd, use actual bps
            print("Constructing BPS representation for policy inputs of commands")
            new_list = []
            total_dim = 0
            term_dims = []
            for idx, (term, dim) in enumerate(
                zip(self.obs_command_terms, self.obs_command_dims)
            ):
                if term == "pcd":
                    total_dim += self.actual_bps.shape[-1]
                    term_dims.append(self.actual_bps.shape[-1])
                    new_list.append(self.actual_bps)
                else:
                    total_dim += dim
                    term_dims.append(dim)
                    new_list.append(self.obs_commands_buffer[idx])
            self.policy_command_dim = total_dim
            self.policy_command_dims = term_dims
            return torch.cat(new_list, dim=-1)
        else:
            self.policy_command_dim = self.obs_command_dim
            self.policy_command_dims = self.obs_command_dims
            return self.obs_command_tensor

    def _construct_reference_obs_commands(self):
        # make sure all tensors are on same device
        return torch.cat(self.obs_commands_buffer, dim=-1)

    def _construct_gt_pos(self):
        # pos = self.original_demo.object_reference_motion["obj_pos"]
        if self.original_demo.retargeted_motion:
            pos = self.original_demo.retargeted_motion["obj_pos"]
        else:
            raise ValueError("instead of gt pos, retargeted motion should be used")
            pos = self.original_demo.object_reference_motion["obj_pos"]
        pos = torch.from_numpy(pos).float()
        self.obs_commands_buffer.append(pos)

    def _construct_gt_rot(self):
        if self.original_demo.retargeted_motion:
            quat = self.original_demo.retargeted_motion["obj_quat"]
        else:
            raise ValueError("instead of gt rot, retargeted motion should be used")
            quat = self.original_demo.object_reference_motion["obj_quat"]

        rot = torch.from_numpy(quat).float()
        if self.cfg.rot_format == "6d":
            rot = rot6d_from_matrix(matrix_from_quat(rot))

        self.obs_commands_buffer.append(rot)

    def _construct_gt_arti(self):
        # use the joint names as per the asset info
        asset: ArtObject = self.env.scene_manager.scene_entities["object"]
        sim_asset_joint_names = [j.name for j in asset.joints]
        if len(sim_asset_joint_names) == 0:
            return

        assert (
            self.original_demo.retargeted_motion
        ), "Retargeted motion must be available for gt arti"

        ref_motion = self.original_demo.retargeted_motion

        retargeted_jnames = ref_motion["joint_names"]
        joint_indices = [
            retargeted_jnames.index(jname) for jname in sim_asset_joint_names
        ]

        retargeted_jvals = ref_motion["obj_arti"]
        if retargeted_jvals is None:
            raise ValueError(
                "Retargeted motion is missing object articulation values for an articulated object."
            )
        arti = retargeted_jvals[:, joint_indices]

        arti = torch.from_numpy(arti).float()
        self.obs_commands_buffer.append(arti)

    def _construct_pcd(self, append=True):
        # IMPORTANT: used from reference.
        # optionally use target object geometry + retargeted target-object motion

        # for each time step, construct the mesh with the given pose and arti
        # sample points from the mesh.
        if self.cfg.use_retargeted_motion:
            if self.original_demo.retargeted_motion is None:
                raise ValueError(
                    "commands_cfg.use_retargeted_motion=True requires retargeted motion."
                )
            obj_urdf_path = self.original_demo.target_obj_urdf_path
            object_motion = self.original_demo.object_reference_motion
        else:
            obj_urdf_path = self.original_demo.obj_urdf_path
            object_motion = self.original_demo.original_object_motion()

        obj_urdf_inst = URDF.load(obj_urdf_path)
        obj_mesh = construct_pytorch_mesh(
            urdf_inst=obj_urdf_inst, device=self.env.device
        )
        kpts, normals_list = sample_kpts_with_normals(
            obj_mesh,
            num_additional_pts=self.cfg.num_additional_kpts,
            max_pts=self.cfg.max_pts_per_link,
        )
        pos = object_motion["obj_pos"]
        quat = object_motion["obj_quat"]
        pos = torch.from_numpy(pos).float().to(self.env.device)
        quat = torch.from_numpy(quat).float().to(self.env.device)

        if self.cfg.relative_commands:
            raise NotImplementedError(
                "Relative commands not implemented yet. Refer to eval_utils.py example"
            )
            pos, quat = self.construct_relative_gt(pos, quat)

        motion_joint_names = object_motion["joint_names"]
        arti_np = object_motion["obj_arti"]
        n_frames = pos.shape[0]
        if arti_np is None:
            arti = torch.zeros((n_frames, len(motion_joint_names)), dtype=torch.float32)
        else:
            arti = torch.from_numpy(arti_np).float()
        arti = arti.to(self.env.device)

        pcd_time_series = compute_kpts_traj(
            local_frame_kpts_list=kpts,
            pos=pos,
            quat=quat,
            arti=arti,
            obj_inst=obj_urdf_inst,
            jnames=motion_joint_names,
            local_frame_normals_list=normals_list,
        )
        n_kpts = pcd_time_series.shape[1]
        assert pcd_time_series.shape == (
            pos.shape[0],
            n_kpts,
            6,
        ), f"Expected shape {(pos.shape[0], n_kpts, 6)}, got {pcd_time_series.shape}"

        pcd_time_series = pcd_time_series.reshape(pos.shape[0], n_kpts * 6)
        pcd_time_series = pcd_time_series.to("cpu")

        if append:
            self.obs_commands_buffer.append(pcd_time_series)

        return pcd_time_series

    def _construct_obj_bps(self):
        # IMPORTANT: reference motion needed
        # need to add option of using target object (that is, the results of retargeted motion)
        # and using the retargeted motion of robot hands (not human hands)

        if "pcd" not in self.obs_command_terms:
            pcd = self._construct_pcd(append=False)
        else:
            pcd = self.obs_commands_buffer[self.obs_command_terms.index("pcd")]

        pcd = pcd.reshape(pcd.shape[0], -1, 6)  # (n, n_pts, 6)

        bps = {}
        for side in ["left", "right"]:
            if side not in self.env.scene_manager.scene_entities:
                continue
            hand: Hand = self.env.scene_manager.scene_entities[side]
            human_joint_names = hand.human_joint_names
            jpos = self._reference_hand_points_tensor(side=side, hand=hand)  # (n, j, 3)
            pcd = pcd.to(self.env.device)

            result = compute_bps(pcd, jpos)  # (n, n_bps, 6)
            assert result.shape[1:] == (
                len(human_joint_names),
                6,
            ), f"Expected bps shape {(len(human_joint_names), 6)}, got {result.shape[1:]}"
            bps[side] = result.to("cpu")
            self.obs_command_terms.append(f"{side}_bps")
            self.obs_command_dims.append(result.shape[1] * result.shape[2])
            self.obs_commands_buffer.append(
                result.reshape(result.shape[0], -1).to("cpu")
            )

        return bps

    def _construct_actual_bps(self):
        # sample bps points
        # pcd should already be available

        pcd = self.obs_commands_buffer[self.obs_command_terms.index("pcd")].clone()
        pcd = pcd.reshape(pcd.shape[0], -1, 6)  # (n, n_pts, 6)
        pcd = pcd.to(self.env.device)

        pcd_mean = pcd[:, :, :3].mean(dim=1, keepdim=True)  # (n, 1, 3)
        pcd[..., :3] = pcd[:, :, :3] - pcd_mean  # center the
        pcd_norm = torch.norm(pcd[..., :3], dim=-1, keepdim=True) + 1e-8
        pcd[..., :3] = pcd[..., :3] / pcd_norm

        bps = (
            sample_unit_ball(self.cfg.bps_n_pts)
            .unsqueeze(0)
            .expand(pcd.shape[0], -1, -1)
            .to(self.env.device)
        )  # (n, bps_n_pts, 3)
        bps_repr = compute_bps(pcd, bps)
        # add mean to the bps repr
        bps_repr[..., :3] = bps_repr[..., :3] + pcd_mean
        self.actual_bps = bps_repr.reshape(pcd.shape[0], self.cfg.bps_n_pts * 6).to(
            "cpu"
        )  # (n, bps_n_pts, 6)

    def construct_relative_gt(self, pos, quat):
        ff_pose = torch.zeros(4, 4)
        ff_pose[3, 3] = 1.0
        ff_pose[:3, 3] = pos[0]
        ff_pose[:3, :3] = matrix_from_quat(quat[0])
        ff_pose_inv = ff_pose.inverse()

        n = len(pos)
        poses = torch.zeros(n, 4, 4)
        poses[:, 3, 3] = 1.0
        poses[:, :3, 3] = pos
        poses[:, :3, :3] = matrix_from_quat(quat)

        relative_poses = ff_pose_inv[None, ...] @ poses
        pos = relative_poses[:, :3, 3]
        quat = quat_from_matrix(relative_poses[:, :3, :3])
        return pos, quat

    @property
    def obs_command_dim(self) -> int:
        return sum(self.obs_command_dims)

    def initialize(self):
        pass

    def summary(self) -> str:
        spacing = 20
        lines = [
            "[CommandsManager]",
            f"{'Command Term':<{spacing}} {'Dimension':<10}",
        ]
        lines.append("-" * (spacing + 10))
        for term, dim in zip(self.obs_command_terms, self.obs_command_dims):
            lines.append(f"{term:<{spacing}} {dim:<10}")
        lines.append("-" * (spacing + 10))
        lines.append(f"{'Total':<{spacing}} {self.obs_command_dim:<10}")
        return "\n".join(lines)

    def reset_obs_commands(self, env_ids: list[int] | np.ndarray) -> None:
        if self.env_obs_commands is None:
            return
        command_timesteps = self.env.command_time_step[env_ids]
        commands = self.obs_command_tensor[command_timesteps]
        self.env_obs_commands[env_ids] = commands.to(self.env.device)

    def get_commands_for_env_ids(
        self, env_ids: list[int] | np.ndarray | None = None
    ) -> torch.Tensor:

        env_ids = self.env.reset_manager.normalize_env_ids(env_ids)

        if self.env_obs_commands is not None:
            return self.env_obs_commands[env_ids]
        else:
            command_timesteps = self.env.command_time_step[env_ids]
            commands = self.obs_command_tensor[command_timesteps]
            return commands.to(self.env.device)

    def query_commands_from_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """timesteps: (batch_size,) tensor of timesteps"""
        commands = self.obs_command_tensor[timesteps.cpu()]
        return commands.to(self.env.device)

    def query_policy_commands_from_timesteps(
        self, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """timesteps: (batch_size,) tensor of timesteps"""
        commands = self.policy_command_tensor[timesteps.cpu()]
        return commands.to(self.env.device)

    def get_commands_from_obs(self, named_obs: dict) -> torch.Tensor | None:
        command = named_obs.get("command", None)
        if command is None:
            return None
        if self.env.cfg.obs_cfg.use_command_timestep:
            return self.query_commands_from_timesteps(command[:, 0].long())
        else:
            return command

    def named_view(self, command: torch.Tensor) -> dict:
        """Return a dictionary mapping observation terms to their respective tensors."""
        splits = torch.split(command, self.obs_command_dims, dim=-1)
        named_splits = dict(zip(self.obs_command_terms, splits))
        return named_splits
