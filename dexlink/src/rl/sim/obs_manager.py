from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from dexlink.src.rl.sim.contact_info import get_filtered_contacts
from dexlink.utils.retargeting.math_utils import quat_from_matrix
from dexlink.utils.retargeting.mesh_utils import (
    compute_bps,
    compute_kpts_traj,
    construct_pytorch_mesh,
    sample_kpts_with_normals,
)

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import ObsCfg
    from dexlink.src.rl.sim.art_object import ArtObject
    from dexlink.src.rl.sim.env import Env
    from dexlink.src.rl.sim.robot import Hand


class ObservationManager:
    def __init__(self, cfg: ObsCfg, env: Env):
        self.cfg = cfg
        self.env = env
        self.active_hand_sides = self.env.scene_manager.active_hand_sides
        self.hand_obs = (
            cfg.use_hand_obs
            and (not env.cfg.scene_cfg.object_only_scene)
            and len(self.active_hand_sides) > 0
        )
        self.object_obs = cfg.use_object_obs
        self.object_vel_obs = cfg.use_object_vel_obs
        self.task_command_obs = cfg.task_command

        self.rot_format = cfg.rot_format
        self.obs_terms = []
        self.term_dims = []
        self.term_fns = []
        if self.hand_obs:
            for side, fn in [
                ("left", self._compute_left_obs),
                ("right", self._compute_right_obs),
            ]:
                if not self.env.scene_manager.has_hand(side):
                    continue
                hand = self.env.scene_manager.scene_entities[side]
                self.obs_terms.append(side)
                self.term_dims.append(hand.obs_dim)
                self.term_fns.append(fn)

        if self.object_obs:
            self.obs_terms.append("object")
            object_entity = self.env.scene_manager.scene_entities["object"]
            rot_dim = 4
            if self.rot_format == "6d":
                rot_dim = 6
            obj_dim = 3 + rot_dim + len(object_entity.dof_map)
            self.term_dims.append(obj_dim)
            self.term_fns.append(self._compute_object_obs)

        if self.object_vel_obs:
            self.obs_terms.append("object_vel")
            object_entity = self.env.scene_manager.scene_entities["object"]
            obj_vel_dim = 6 + len(object_entity.dof_map)
            self.term_dims.append(obj_vel_dim)
            self.term_fns.append(self._compute_object_vel_obs)

        if self.task_command_obs:
            self.obs_terms.append("command")
            if self.cfg.use_command_timestep:
                command_dim = 1  # just the timestep index
            else:
                command_dim = self.env.commands_manager.obs_command_dim
            self.term_dims.append(command_dim)
            self.term_fns.append(self._get_commands)

        self.use_filtered_contact = self.cfg.use_filtered_contact
        self.use_contact_force = self.cfg.use_contact_force
        self.use_contact_pos = self.cfg.use_contact_pos

        # for now we use either both, or neither

        contact_obs = self.hand_obs and (self.use_contact_force or self.use_contact_pos)
        if contact_obs:
            term_dim = 0
            for side in self.active_hand_sides:
                hand: Hand = self.env.scene_manager.scene_entities[side]
                term_dim += len(hand.tactile_links_global_indices) * 6
            if term_dim > 0:
                self.obs_terms.append("contact")
                self.term_dims.append(term_dim)
                self.term_fns.append(self._compute_contact_obs)

        # compute the object keypoints,
        asset: ArtObject = self.env.scene_manager.scene_entities["object"]
        urdf_inst = asset.urdf_inst
        asset_mesh = construct_pytorch_mesh(urdf_inst=urdf_inst)
        command_cfg = self.env.cfg.commands_cfg
        command_kpts, command_normals = sample_kpts_with_normals(
            asset_mesh,
            num_additional_pts=command_cfg.num_additional_kpts,
            max_pts=command_cfg.max_pts_per_link,
        )
        self.object_command_representation_kpts = command_kpts
        self.object_command_representation_normals = command_normals

        for l_idx in range(len(command_kpts)):
            assert (
                command_kpts[l_idx].shape == command_normals[l_idx].shape
            ), f"Expected keypoints and normals to have same shape for link {l_idx}, got {command_kpts[l_idx].shape} and {command_normals[l_idx].shape}"

        self.asset = asset

    @property
    def observation_dim(self) -> int:
        return sum(self.term_dims)

    def initialize(self):
        pass

    def summary(self) -> str:
        spacing = 20
        lines = [
            "[ObservationManager]",
            f"{'Observation Term':<{spacing}} {'Dimension':<10}",
        ]
        lines.append("-" * (spacing + 10))
        for term, dim in zip(self.obs_terms, self.term_dims):
            lines.append(f"{term:<{spacing}} {dim:<10}")
        lines.append("-" * (spacing + 10))
        lines.append(f"{'Total':<{spacing}} {self.observation_dim:<10}")
        return "\n".join(lines)

    def _compute_left_obs(self):
        left_hand: Hand = self.env.scene_manager.scene_entities["left"]
        left_obs = left_hand.get_obs()  # (num_envs, dof)
        return left_obs

    def _compute_right_obs(self):
        right_hand: Hand = self.env.scene_manager.scene_entities["right"]
        right_obs = right_hand.get_obs()  # (num_envs, dof)
        return right_obs

    def _compute_object_obs(self):
        object_entity: ArtObject = self.env.scene_manager.scene_entities["object"]
        object_obs = object_entity.get_obs(
            rot_format=self.rot_format
        )  # (num_envs, 3+4/9+[obj_dofs])

        return object_obs

    def _compute_object_vel_obs(self):
        object_entity: ArtObject = self.env.scene_manager.scene_entities["object"]
        return object_entity.get_vel_obs()  # (num_envs, 6+[obj_dofs])

    def _get_commands(self):

        if self.cfg.use_command_timestep:
            # add small epsilon so that rounding works fine
            return (
                (self.env.command_time_step[:, None] + 1e-2)
                .to(torch.float32)
                .to(self.env.device)
            )
        else:
            return self.env.commands_manager.get_commands_for_env_ids(env_ids=None)

    def _get_object_pcd_from_named_obs(self, named_obs, normals: bool = True):
        # if self._current_obj_pcd is None:
        #     self._current_obj_pcd = self.env.observation_manager.compute_object_pcd_obs(
        #         pos=named_obs["obj_pos"],
        #         quat=named_obs["obj_quat"],
        #         arti=named_obs["obj_arti"],
        #         normals=normals,
        #     )
        # return self._current_obj_pcd
        return self.env.observation_manager.compute_object_pcd_obs(
            pos=named_obs["obj_pos"],
            quat=named_obs["obj_quat"],
            arti=named_obs["obj_arti"],
            normals=normals,
        )

    def compute_object_pcd_obs(self, pos, quat, arti, normals=False):

        rot_dim = quat.shape[-1]
        if rot_dim == 4:
            quat = quat
        elif rot_dim == 6:
            # convert to quat
            x_axis = quat[:, :3]
            y_axis = quat[:, 3:6]
            z_axis = torch.cross(x_axis, y_axis, dim=-1)
            mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (N, 3, 3)
            quat = quat_from_matrix(mat)

        pcd = compute_kpts_traj(
            self.object_command_representation_kpts,
            pos,
            quat,
            arti=arti,
            obj_inst=self.asset.urdf_inst,
            jnames=[j.name for j in self.asset.joints],
            local_frame_normals_list=(
                self.object_command_representation_normals if normals else None
            ),
        )
        # pcd of shape (N, n_pts, 3)
        pcd_dim = pcd.shape[-1]
        if normals:
            assert pcd_dim == 6, f"Expected pcd dim 6 with normals, got {pcd_dim}"
        else:
            assert pcd_dim == 3, f"Expected pcd dim 3 without normals, got {pcd_dim}"
        return pcd

    # def (self, named_obs: dict):
    #     pos = named_obs["obj_pos"]
    #     quat = named_obs["obj_quat"]
    #     arti = named_obs["obj_arti"]

    #     return self._compute_object_pcd_obs(pos, quat, arti, normals=self.cfg.)

    def compute_bps(self, pcd, bps):
        # pcd: (N, n_pts, 6) -> x, y, z, nx, ny, nz
        # bps: (N, n_bps, 3)
        # returns: (N, n_bps, 6) -> direction to closest point + normal at that point
        return compute_bps(pcd, bps)

    def _compute_contact_obs(self):
        if self.use_filtered_contact:
            target_entity = self.env.scene_manager.scene_entities["object"].object
        else:
            target_entity = None

        _obs = []
        for side in self.active_hand_sides:
            hand: Hand = self.env.scene_manager.scene_entities[side]
            contact_info = get_filtered_contacts(
                entity_a=hand.robot,
                entity_b=target_entity,
                filter_links_a=torch.tensor(
                    hand.tactile_links_global_indices, dtype=torch.long
                ),
                return_link_force=self.use_contact_force,
                return_link_pos=True,  # we use the validity information from here
            )

            force = contact_info["contact_force_link_a"]
            force_net = force.sum(dim=2)
            force_mag = force.norm(dim=-1)
            # for pos, use the index in dim=2, that has the maximum force mag
            # (N, n_links_a)

            # we want to select the pos corresponding to max_force_indices
            pos = contact_info["contact_pos_link_a"]  # (N, n_links_a, n_links_b, 3)
            # max_force_indices = (
            #     force_mag.argmax(dim=-1)
            #     .unsqueeze(-1)
            #     .unsqueeze(-1)
            #     # .expand(-1, -1, -1, pos.shape[-1])
            # )
            max_force_indices = force_mag.argmax(dim=-1)
            # build one-hot mask to avoid shape-dependent expand/gather errors
            max_force_mask = F.one_hot(max_force_indices, num_classes=pos.shape[2])
            max_force_mask = max_force_mask.to(pos.dtype).unsqueeze(-1)
            grouped_contact_pos = (pos * max_force_mask).sum(dim=2)

            contact_obs = torch.cat([grouped_contact_pos, force_net], dim=-1)
            n_envs = contact_obs.shape[0]
            contact_obs = contact_obs.reshape(
                n_envs, len(hand.tactile_links_global_indices) * 6
            )

            _obs.append(contact_obs)

        return torch.cat(_obs, dim=-1)

    def compute_observations(self) -> torch.Tensor:
        obs = []
        for _term, _fn in zip(self.obs_terms, self.term_fns):
            obs.append(_fn())

        return torch.cat(obs, dim=-1)

    def named_view(self, obs: torch.Tensor) -> dict:
        """Return a dictionary mapping observation terms to their respective tensors."""
        splits = torch.split(obs, self.term_dims, dim=-1)
        named_splits = dict(zip(self.obs_terms, splits))

        # also add hand observations separately for convenience
        if "left" in named_splits:
            left_obs = named_splits["left"]
            left_hand: Hand = self.env.scene_manager.scene_entities["left"]
            named_left_obs = left_hand.obs_named_view(left_obs)
            named_splits.update({f"left_{k}": v for k, v in named_left_obs.items()})

        if "right" in named_splits:
            right_obs = named_splits["right"]
            right_hand: Hand = self.env.scene_manager.scene_entities["right"]
            named_right_obs = right_hand.obs_named_view(right_obs)
            named_splits.update({f"right_{k}": v for k, v in named_right_obs.items()})

        # also add object pos, quat, arti separately for convenience
        if "object" in named_splits:
            object_obs = named_splits["object"]
            object_entity = self.env.scene_manager.scene_entities["object"]
            n_obj_dofs = len(object_entity.dof_map)
            pos = object_obs[:, :3]
            if self.rot_format == "6d":
                rot_dim = 6
            else:
                rot_dim = 4
            quat = object_obs[:, 3 : 3 + rot_dim]
            arti = object_obs[:, 3 + rot_dim : 3 + rot_dim + n_obj_dofs]
            named_splits["obj_pos"] = pos
            named_splits["obj_quat"] = quat
            named_splits["obj_arti"] = arti

        if "object_vel" in named_splits:
            object_vel_obs = named_splits["object_vel"]
            object_entity = self.env.scene_manager.scene_entities["object"]
            n_obj_dofs = len(object_entity.dof_map)
            named_splits["obj_lin_vel"] = object_vel_obs[:, :3]
            named_splits["obj_ang_vel"] = object_vel_obs[:, 3:6]
            named_splits["obj_arti_vel"] = object_vel_obs[:, 6 : 6 + n_obj_dofs]

        return named_splits

    def get_ground_truth_info(self) -> dict:
        # the observations do not necessarily contain all the information
        # should be used only for evaluation: how good the trajectory is
        gt = {}

        asset: ArtObject = self.env.scene_manager.scene_entities["object"]
        gt["object"] = asset.get_ground_truth_info()

        for side in self.active_hand_sides:
            hand = self.env.scene_manager.scene_entities[side]
            hand_info = hand.get_ground_truth_info()
            gt[side] = hand_info

        gt["command_timestep"] = self.env.command_time_step.clone()
        eval_feature_info = dict(self.env.reward_manager.get_eval_feature_info())
        hand_jvals_eval_info = self._get_hand_joint_eval_rollout_info()
        if hand_jvals_eval_info:
            eval_feature_info["hand_jvals"] = hand_jvals_eval_info
        if eval_feature_info:
            gt["eval_metrics"] = eval_feature_info

        return gt

    def _get_hand_joint_eval_rollout_info(self) -> dict:
        """Build per-side rollout joint-value features for eval logs."""
        original_demo = self.env.scene_manager.original_demo
        retargeted_motion = getattr(original_demo, "retargeted_motion", None)
        if not isinstance(retargeted_motion, dict):
            return {}

        out = {}
        for side in ("left", "right"):
            if side not in self.env.scene_manager.scene_entities:
                continue
            jvals_key = f"{side}_jvals"
            jnames_key = f"{side}_jnames"
            if (jvals_key not in retargeted_motion) or (
                jnames_key not in retargeted_motion
            ):
                continue

            hand: Hand = self.env.scene_manager.scene_entities[side]
            retgt_jnames = [str(n) for n in retargeted_motion[jnames_key]]
            if len(retgt_jnames) != len(hand.joint_names) or any(
                jn not in hand.joint_names for jn in retgt_jnames
            ):
                continue

            rollout_jvals = hand.robot.get_dofs_position(hand.dof_map).detach()
            hand_idx_map = [hand.joint_names.index(jn) for jn in retgt_jnames]
            rollout_jvals = rollout_jvals[:, hand_idx_map]

            out[side] = {
                "rollout": rollout_jvals.clone(),
            }
        return out
