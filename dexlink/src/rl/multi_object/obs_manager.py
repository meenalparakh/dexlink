from __future__ import annotations

import torch

from dexlink.utils.retargeting.math_utils import (
    matrix_from_quat,
    quat_from_matrix,
    rot6d_from_matrix,
)
from dexlink.utils.retargeting.mesh_utils import compute_kpts_traj
from dexlink.src.rl.world.art_object import ArtObject
from dexlink.src.rl.world.obs_manager import ObservationManager
from dexlink.src.rl.world.robot import Hand


class MultiObjectObservationManager(ObservationManager):
    """Observation manager that packs every scene object into object terms."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.object_ids = list(env.task_manager.scene_object_ids)
        self.object_entities: dict[str, ArtObject] = env.physics_manager.object_entities
        self.canonical_object_id = env.task_manager.scene_canonical_object_id
        self.object_obs_dims_by_id = {
            object_id: self._object_obs_dim(entity)
            for object_id, entity in self.object_entities.items()
        }
        self.object_vel_dims_by_id = {
            object_id: 6 + len(entity.dof_map)
            for object_id, entity in self.object_entities.items()
        }
        self.object_surface_n_pts_by_id = {
            object_id: sum(kpts.shape[0] for kpts in entity.object_surface_kpts)
            for object_id, entity in self.object_entities.items()
        }
        self.privileged_object_pcd_n_pts_by_id = {
            object_id: self._privileged_pcd_n_pts_for_object(object_id)
            for object_id in self.object_ids
        }
        self.object_pcd_policy_offsets = self._pcd_offsets(self.policy_object_pcd_n_pts)
        self.object_pcd_privileged_offsets = self._pcd_offsets(
            self.privileged_object_pcd_n_pts_by_id
        )
        self._replace_registered_term(
            "policy",
            "object",
            sum(self.object_obs_dims_by_id.values()),
            self._compute_object_obs,
        )
        self._replace_registered_term(
            "privileged",
            "object",
            sum(self.object_obs_dims_by_id.values()),
            self._compute_object_obs,
        )
        self._replace_registered_term(
            "policy",
            "object_vel",
            sum(self.object_vel_dims_by_id.values()),
            self._compute_object_vel_obs,
        )
        self._replace_registered_term(
            "privileged",
            "object_vel",
            sum(self.object_vel_dims_by_id.values()),
            self._compute_object_vel_obs,
        )
        if self.cfg.use_phase_index:
            self._register_term("policy", "phase_index", 1, self._get_phase_indices)
            self._register_term("privileged", "phase_index", 1, self._get_phase_indices)
        if self.policy_object_pcd_obs:
            self._replace_registered_term(
                "policy",
                "object_pcd",
                len(self.object_ids)
                * self.policy_object_pcd_n_pts
                * self.policy_object_pcd_dim,
                self._compute_policy_object_pcd_obs,
            )
        if self.privileged_object_pcd_obs:
            self._replace_registered_term(
                "privileged",
                "object_pcd",
                sum(self.privileged_object_pcd_n_pts_by_id.values())
                * self.privileged_object_pcd_dim,
                self._compute_privileged_object_pcd_obs,
            )

    def _replace_registered_term(self, view: str, term: str, dim: int, fn) -> None:
        terms = self._get_terms(view)
        dims = self._get_term_dims(view)
        fns = self._get_term_fns(view)
        if term not in terms:
            return
        index = terms.index(term)
        dims[index] = dim
        fns[index] = fn

    def _object_obs_dim(self, entity: ArtObject) -> int:
        rot_dim = 6 if self.rot_format == "6d" else 4
        return 3 + rot_dim + len(entity.dof_map)

    def _privileged_pcd_n_pts_for_object(self, object_id: str) -> int:
        surface_n_pts = self.object_surface_n_pts_by_id[object_id]
        if self.cfg.privileged_object_pcd_n_pts == -1:
            return surface_n_pts
        if self.cfg.privileged_object_pcd_n_pts > 0:
            return min(self.cfg.privileged_object_pcd_n_pts, surface_n_pts)
        raise ValueError(
            "obs_cfg.privileged_object_pcd_n_pts must be -1 or a positive integer, "
            f"got {self.cfg.privileged_object_pcd_n_pts}."
        )

    def _pcd_offsets(self, n_pts_by_object: int | dict[str, int]) -> dict[str, slice]:
        offsets: dict[str, slice] = {}
        start = 0
        for object_id in self.object_ids:
            n_pts = (
                int(n_pts_by_object)
                if isinstance(n_pts_by_object, int)
                else int(n_pts_by_object[object_id])
            )
            offsets[object_id] = slice(start, start + n_pts)
            start += n_pts
        return offsets

    def _compute_object_obs(self):
        return torch.cat(
            [
                self.object_entities[object_id].get_obs(rot_format=self.rot_format)
                for object_id in self.object_ids
            ],
            dim=-1,
        )

    def _compute_object_vel_obs(self):
        return torch.cat(
            [
                self.object_entities[object_id].get_vel_obs()
                for object_id in self.object_ids
            ],
            dim=-1,
        )

    def _get_phase_indices(self) -> torch.Tensor:
        phase_indices = self.env.task_manager.phase_index_by_timestep[
            self.env.command_time_step
        ]
        return phase_indices[:, None].to(dtype=torch.float32)

    def compute_object_pcd_obs_for_entity(
        self,
        entity: ArtObject,
        *,
        pos: torch.Tensor,
        quat: torch.Tensor,
        arti: torch.Tensor,
        normals: bool,
        num_pts_sample: int,
    ) -> torch.Tensor:
        rot_dim = quat.shape[-1]
        if rot_dim == 6:
            x_axis = quat[:, :3]
            y_axis = quat[:, 3:6]
            z_axis = torch.cross(x_axis, y_axis, dim=-1)
            quat = quat_from_matrix(torch.stack([x_axis, y_axis, z_axis], dim=-1))
        elif rot_dim != 4:
            raise ValueError(f"Expected object rotation dim 4 or 6, got {rot_dim}.")

        pcd = compute_kpts_traj(
            entity.object_surface_kpts,
            pos,
            quat,
            arti=arti,
            obj_inst=entity.urdf_inst,
            jnames=[joint.name for joint in entity.joints],
            local_frame_normals_list=entity.object_surface_normals if normals else None,
            num_pts_sample=num_pts_sample,
        )
        expected_dim = (
            self.privileged_object_pcd_dim if normals else self.policy_object_pcd_dim
        )
        if pcd.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected object PCD feature dim {expected_dim}, got {pcd.shape[-1]}."
            )
        return pcd

    def _current_object_state_for_id(
        self, object_id: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entity = self.object_entities[object_id]
        obs = entity.get_obs(rot_format="quat")
        pos = obs[:, :3]
        quat = obs[:, 3:7]
        arti = obs[:, 7 : 7 + len(entity.dof_map)]
        return pos, quat, arti

    def _compute_privileged_object_pcd_from_object_state(self) -> torch.Tensor:
        chunks = []
        for object_id in self.object_ids:
            entity = self.object_entities[object_id]
            pos, quat, arti = self._current_object_state_for_id(object_id)
            chunks.append(
                self.compute_object_pcd_obs_for_entity(
                    entity,
                    pos=pos,
                    quat=quat,
                    arti=arti,
                    normals=True,
                    num_pts_sample=self.privileged_object_pcd_n_pts_by_id[object_id],
                )
            )
        return torch.cat(chunks, dim=1)

    def _compute_policy_object_pcd_obs(self):
        policy_object_pcd = self.physics.get_object_pcd(normals=False)
        expected_shape = (
            self.env.num_envs,
            len(self.object_ids) * self.policy_object_pcd_n_pts,
            self.policy_object_pcd_dim,
        )
        if tuple(policy_object_pcd.shape) != expected_shape:
            raise ValueError(
                f"Expected policy object PCD shape {expected_shape}, "
                f"got {tuple(policy_object_pcd.shape)}."
            )
        return policy_object_pcd.reshape(self.env.num_envs, -1)

    def _compute_contact_obs(self):
        object_targets = list(self.object_entities.values())
        chunks = []
        for side in self.active_hand_sides:
            hand: Hand = self.env.physics_manager.scene_entities[side]
            side_signal = None
            for target_entity in object_targets:
                contact_signal = self.physics.get_hand_object_binary_contact_signal(
                    hand,
                    target_entity,
                    threshold=self.contact_signal_threshold,
                )
                side_signal = (
                    contact_signal
                    if side_signal is None
                    else torch.logical_or(side_signal.bool(), contact_signal.bool())
                )
            if side_signal is None:
                raise ValueError("Expected at least one object target for contact obs.")
            n_envs = side_signal.shape[0]
            chunks.append(
                side_signal.reshape(
                    n_envs,
                    len(hand.tactile_links_global_indices) * 3,
                ).to(dtype=torch.float32)
            )
        return torch.cat(chunks, dim=-1)

    def _named_view_tensor(self, obs: torch.Tensor, view: str) -> dict:
        named = super()._named_view_tensor(obs=obs, view=view)
        if "object" in named:
            self._add_named_object_slices(named, named["object"])
        if "object_vel" in named:
            self._add_named_object_velocity_slices(named, named["object_vel"])
        if "object_pcd" in named:
            offsets = (
                self.object_pcd_policy_offsets
                if view == "policy"
                else self.object_pcd_privileged_offsets
            )
            self._add_named_object_pcd_slices(
                named,
                named["object_pcd"],
                expected_feat_dim=(
                    self.policy_object_pcd_dim
                    if view == "policy"
                    else self.privileged_object_pcd_dim
                ),
                offsets=offsets,
            )
        return named

    def _add_named_object_slices(
        self, named: dict[str, torch.Tensor], object_obs: torch.Tensor
    ) -> None:
        offset = 0
        objects = {}
        pos_chunks = []
        quat_chunks = []
        max_arti = max(len(entity.dof_map) for entity in self.object_entities.values())
        arti_chunks = []
        for object_id in self.object_ids:
            entity = self.object_entities[object_id]
            dim = self.object_obs_dims_by_id[object_id]
            chunk = object_obs[:, offset : offset + dim]
            offset += dim
            rot_dim = 6 if self.rot_format == "6d" else 4
            pos = chunk[:, :3]
            quat = chunk[:, 3 : 3 + rot_dim]
            arti = chunk[:, 3 + rot_dim : 3 + rot_dim + len(entity.dof_map)]
            padded_arti = torch.zeros(
                (self.env.num_envs, max_arti),
                dtype=chunk.dtype,
                device=chunk.device,
            )
            if arti.shape[1] > 0:
                padded_arti[:, : arti.shape[1]] = arti
            objects[object_id] = {
                "obs": chunk,
                "pos": pos,
                "quat": quat,
                "arti": arti,
            }
            named[f"object_{object_id}"] = chunk
            named[f"obj_pos_{object_id}"] = pos
            named[f"obj_quat_{object_id}"] = quat
            named[f"obj_arti_{object_id}"] = arti
            pos_chunks.append(pos)
            quat_chunks.append(quat)
            arti_chunks.append(padded_arti)
        if offset != object_obs.shape[1]:
            raise ValueError(
                f"Consumed {offset} object obs dims, got term dim {object_obs.shape[1]}."
            )
        named["objects"] = objects
        named["obj_pos_all"] = torch.stack(pos_chunks, dim=1)
        named["obj_quat_all"] = torch.stack(quat_chunks, dim=1)
        named["obj_arti_all"] = torch.stack(arti_chunks, dim=1)
        canonical = objects[self.canonical_object_id]
        named["obj_pos"] = canonical["pos"]
        named["obj_quat"] = canonical["quat"]
        named["obj_arti"] = canonical["arti"]

    def _add_named_object_velocity_slices(
        self, named: dict[str, torch.Tensor], object_vel_obs: torch.Tensor
    ) -> None:
        offset = 0
        for object_id in self.object_ids:
            dim = self.object_vel_dims_by_id[object_id]
            chunk = object_vel_obs[:, offset : offset + dim]
            offset += dim
            named[f"object_vel_{object_id}"] = chunk
            named[f"obj_lin_vel_{object_id}"] = chunk[:, :3]
            named[f"obj_ang_vel_{object_id}"] = chunk[:, 3:6]
            named[f"obj_arti_vel_{object_id}"] = chunk[:, 6:dim]
        canonical_dim = self.object_vel_dims_by_id[self.canonical_object_id]
        canonical_offset = sum(
            self.object_vel_dims_by_id[object_id]
            for object_id in self.object_ids
            if self.object_ids.index(object_id)
            < self.object_ids.index(self.canonical_object_id)
        )
        canonical = object_vel_obs[
            :, canonical_offset : canonical_offset + canonical_dim
        ]
        named["obj_lin_vel"] = canonical[:, :3]
        named["obj_ang_vel"] = canonical[:, 3:6]
        named["obj_arti_vel"] = canonical[:, 6:canonical_dim]

    def _add_named_object_pcd_slices(
        self,
        named: dict[str, torch.Tensor],
        object_pcd_flat: torch.Tensor,
        *,
        expected_feat_dim: int,
        offsets: dict[str, slice],
    ) -> None:
        object_pcd = object_pcd_flat.view(
            object_pcd_flat.shape[0],
            -1,
            expected_feat_dim,
        )
        pcd_by_id = {}
        for object_id, offset in offsets.items():
            pcd = object_pcd[:, offset, :]
            pcd_by_id[object_id] = pcd
            named[f"object_pcd_{object_id}"] = pcd.reshape(self.env.num_envs, -1)
        named["object_pcd_by_id"] = pcd_by_id

    def get_named_object_obs(
        self, named_obs: dict[str, torch.Tensor], object_id: str
    ) -> dict[str, torch.Tensor]:
        objects = named_obs.get("objects")
        if not isinstance(objects, dict) or object_id not in objects:
            raise ValueError(
                f"Named obs is missing object {object_id!r}; available objects: "
                f"{list(objects) if isinstance(objects, dict) else []}."
            )
        return objects[object_id]

    def get_object_pcd_from_named_obs(
        self,
        named_obs: dict[str, torch.Tensor],
        expected_feat_dim: int,
        object_id: str | None = None,
    ) -> torch.Tensor:
        if object_id is None:
            return super().get_object_pcd_from_named_obs(
                named_obs,
                expected_feat_dim=expected_feat_dim,
            )
        pcd_by_id = named_obs.get("object_pcd_by_id")
        if not isinstance(pcd_by_id, dict) or object_id not in pcd_by_id:
            raise ValueError(
                f"Named obs is missing object_pcd for object {object_id!r}."
            )
        pcd = pcd_by_id[object_id]
        if pcd.shape[-1] != expected_feat_dim:
            raise ValueError(
                f"Expected object {object_id!r} PCD feature dim {expected_feat_dim}, "
                f"got {pcd.shape[-1]}."
            )
        return pcd

    def post_process_policy_obs(self, named_obs):
        processed = dict(named_obs)
        if len(self.active_hand_sides) == 0:
            return processed
        frame_side = (
            "right" if "right" in self.active_hand_sides else self.active_hand_sides[0]
        )
        wrist_pos, wrist_rot, wrist_rot_t = self._process_hand_policy_obs(
            processed, frame_side=frame_side
        )
        self._process_multi_object_policy_obs(processed, wrist_pos, wrist_rot_t)
        self._process_policy_pcd_obs(processed, wrist_pos)
        processed[f"{frame_side}_wrist_pose"][:, :3] = torch.zeros_like(wrist_pos)
        return processed

    def _process_hand_policy_obs(
        self, processed: dict[str, torch.Tensor], *, frame_side: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_wrist_key = f"{frame_side}_wrist_pose"
        frame_wrist_pose = processed.get(frame_wrist_key)
        if frame_wrist_pose is None or frame_wrist_pose.shape[-1] != 9:
            raise ValueError(
                f"Expected policy-frame wrist observation {frame_wrist_key!r} with dim 9."
            )
        wrist_pos = frame_wrist_pose[:, :3]
        wrist_x_axis = frame_wrist_pose[:, 3:6]
        wrist_y_axis = frame_wrist_pose[:, 6:9]
        wrist_z_axis = torch.cross(wrist_x_axis, wrist_y_axis, dim=-1)
        wrist_rot = torch.stack([wrist_x_axis, wrist_y_axis, wrist_z_axis], dim=-1)
        wrist_rot_t = wrist_rot.transpose(1, 2)

        for side in self.active_hand_sides:
            if side not in processed:
                raise ValueError(
                    f"Policy observation is missing active hand term {side!r}."
                )
            hand: Hand = self.env.physics_manager.scene_entities[side]
            side_named_obs = hand.obs_named_view(processed[side])
            side_wrist_key = f"{side}_wrist_pose"
            side_wrist_pose = processed.get(side_wrist_key)
            if side_wrist_pose is None or side_wrist_pose.shape[-1] != 9:
                raise ValueError(
                    f"Expected wrist pose observation {side_wrist_key!r} with dim 9."
                )
            side_wrist_x_axis = side_wrist_pose[:, 3:6]
            side_wrist_y_axis = side_wrist_pose[:, 6:9]
            side_wrist_z_axis = torch.cross(
                side_wrist_x_axis, side_wrist_y_axis, dim=-1
            )
            side_wrist_rot = torch.stack(
                [side_wrist_x_axis, side_wrist_y_axis, side_wrist_z_axis], dim=-1
            )
            self.env.current_wrist_pos_by_side[side] = side_wrist_pose[:, :3].clone()
            self.env.current_wrist_quat_by_side[side] = quat_from_matrix(
                side_wrist_rot
            ).clone()
            if "jpos" in side_named_obs:
                side_jpos = side_named_obs["jpos"]
                expected_shape = (self.env.num_envs, len(hand.dof_map))
                if tuple(side_jpos.shape) != expected_shape:
                    raise ValueError(
                        f"Expected {side!r}-hand jpos shape {expected_shape}, "
                        f"got {tuple(side_jpos.shape)}."
                    )
                self.env.current_wrist_control_jpos_by_side[side] = side_jpos[
                    :, hand.wrist_control_joint_idx
                ].clone()
                masked_side_jpos = side_jpos.clone()
                masked_side_jpos[:, hand.wrist_control_joint_idx] = 0.0
                side_named_obs["jpos"] = masked_side_jpos
                processed[f"{side}_jpos"] = masked_side_jpos
            if "lpos" in side_named_obs:
                lpos = side_named_obs["lpos"].view(self.env.num_envs, -1, 3)
                lpos = lpos - wrist_pos[:, None, :]
                side_named_obs["lpos"] = lpos.reshape(self.env.num_envs, -1)
                processed[f"{side}_lpos"] = side_named_obs["lpos"]
            processed[side] = torch.cat(
                [side_named_obs[term] for term in hand.obs_terms],
                dim=-1,
            )
        return wrist_pos, wrist_rot, wrist_rot_t

    def _process_multi_object_policy_obs(
        self,
        processed: dict[str, torch.Tensor],
        wrist_pos: torch.Tensor,
        wrist_rot_t: torch.Tensor,
    ) -> None:
        if "object" not in processed:
            return
        chunks = []
        for object_id in self.object_ids:
            obj = self.get_named_object_obs(processed, object_id)
            rel_pos = torch.matmul(
                wrist_rot_t,
                (obj["pos"] - wrist_pos).unsqueeze(-1),
            ).squeeze(-1)
            if self.rot_format == "6d":
                obj_x_axis = obj["quat"][:, :3]
                obj_y_axis = obj["quat"][:, 3:6]
                obj_z_axis = torch.cross(obj_x_axis, obj_y_axis, dim=-1)
                obj_rot_mat = torch.stack(
                    [obj_x_axis, obj_y_axis, obj_z_axis],
                    dim=-1,
                )
                rel_rot = rot6d_from_matrix(torch.matmul(wrist_rot_t, obj_rot_mat))
            else:
                rel_rot = quat_from_matrix(
                    torch.matmul(wrist_rot_t, matrix_from_quat(obj["quat"]))
                )
            rel_chunk = torch.cat([rel_pos, rel_rot, obj["arti"]], dim=-1)
            chunks.append(rel_chunk)
            obj["obs"] = rel_chunk
            obj["pos"] = rel_pos
            obj["quat"] = rel_rot
            processed[f"object_{object_id}"] = rel_chunk
            processed[f"obj_pos_{object_id}"] = rel_pos
            processed[f"obj_quat_{object_id}"] = rel_rot
        processed["object"] = torch.cat(chunks, dim=-1)
        canonical = processed["objects"][self.canonical_object_id]
        processed["obj_pos"] = canonical["pos"]
        processed["obj_quat"] = canonical["quat"]
        processed["obj_arti"] = canonical["arti"]

    def _process_policy_pcd_obs(
        self,
        processed: dict[str, torch.Tensor],
        wrist_pos: torch.Tensor,
    ) -> None:
        if "object_pcd" in processed:
            object_pcd = self.get_object_pcd_from_named_obs(
                processed,
                expected_feat_dim=self.policy_object_pcd_dim,
            )
            if object_pcd.shape[-1] != 3:
                raise ValueError(
                    "Expected policy object_pcd feature dim 3, got "
                    f"{object_pcd.shape[-1]}."
                )
            rel_object_pcd = object_pcd - wrist_pos[:, None, :]
            processed["object_pcd"] = rel_object_pcd.reshape(self.env.num_envs, -1)
            self._add_named_object_pcd_slices(
                processed,
                processed["object_pcd"],
                expected_feat_dim=self.policy_object_pcd_dim,
                offsets=self.object_pcd_policy_offsets,
            )
        if "hand_pcd" in processed:
            hand_pcd = self.get_hand_pcd_from_named_obs(
                processed,
                expected_feat_dim=self.policy_hand_pcd_dim,
            )
            if hand_pcd.shape[-1] != 3:
                raise ValueError(
                    f"Expected hand_pcd feature dim 3, got {hand_pcd.shape[-1]}."
                )
            rel_hand_pcd = hand_pcd - wrist_pos[:, None, :]
            processed["hand_pcd"] = rel_hand_pcd.reshape(self.env.num_envs, -1)

    def get_ground_truth_info(self) -> dict:
        gt = super().get_ground_truth_info()
        gt["objects"] = {
            object_id: object_entity.get_ground_truth_info()
            for object_id, object_entity in self.object_entities.items()
        }
        task_metrics = self.env.task_manager.compute_task_metrics()
        if task_metrics:
            gt["task_metrics"] = task_metrics
        return gt
