from __future__ import annotations

import torch

from dexlink.utils.retargeting.math_utils import matrix_from_quat, rot6d_from_matrix
from dexlink.utils.retargeting.mesh_utils import compute_bps, compute_kpts_traj
from dexlink.src.rl.world.commands_manager import CommandsManager
from dexlink.src.rl.world.robot import Hand


class MultiObjectCommandsManager(CommandsManager):
    """Commands manager that packs reference commands for all scene objects."""

    def refresh_reference_buffers(self) -> None:
        self.obs_commands_buffer = []
        self.obs_command_terms = []
        self.obs_command_dims = []
        self._mixed_reference_hand_points = {}
        self.object_ids = list(self.env.task_manager.scene_object_ids)
        self.object_command_dims_by_id = self._object_command_dims_by_id()
        self.object_command_offsets = self._object_command_offsets(
            self.object_command_dims_by_id
        )

        if self.cfg.use_gt_pos:
            self._construct_gt_pos()
            self.obs_command_terms.append("gt_pos")
            self.obs_command_dims.append(3 * len(self.object_ids))

        if self.cfg.use_gt_rot:
            rot_dim = 6 if self.cfg.rot_format == "6d" else 4
            self._construct_gt_rot()
            self.obs_command_terms.append("gt_rot")
            self.obs_command_dims.append(rot_dim * len(self.object_ids))

        if self.cfg.use_gt_arti:
            arti_dim = int(self.env.task_manager.scene_object_arti_vals.shape[2])
            if arti_dim > 0:
                self._construct_gt_arti()
                self.obs_command_terms.append("gt_arti")
                self.obs_command_dims.append(arti_dim * len(self.object_ids))

        if self.cfg.use_pcd:
            pcd_time_series = self._construct_pcd()
            self.obs_command_terms.append("pcd")
            self.obs_command_dims.append(pcd_time_series.shape[-1])

        if self.process_pcd:
            self._construct_actual_bps()

        if self.cfg.use_rel_bps:
            self._construct_obj_bps()

        self.check_commands_buffer()
        self.obs_command_tensor = self._construct_reference_obs_commands().to("cpu")
        self.policy_command_tensor = self._construct_reference_policy_commands().to(
            "cpu"
        )

    def _object_command_dims_by_id(self) -> dict[str, dict[str, int]]:
        rot_dim = 6 if self.cfg.rot_format == "6d" else 4
        arti_dim = int(self.env.task_manager.scene_object_arti_vals.shape[2])
        return {
            object_id: {
                "pos": 3,
                "rot": rot_dim,
                "arti": arti_dim,
            }
            for object_id in self.env.task_manager.scene_object_ids
        }

    def _object_command_offsets(
        self, dims_by_id: dict[str, dict[str, int]]
    ) -> dict[str, dict[str, slice]]:
        offsets: dict[str, dict[str, slice]] = {}
        starts = {"pos": 0, "rot": 0, "arti": 0}
        for object_id in self.object_ids:
            offsets[object_id] = {}
            for key in ("pos", "rot", "arti"):
                dim = dims_by_id[object_id][key]
                offsets[object_id][key] = slice(starts[key], starts[key] + dim)
                starts[key] += dim
        return offsets

    def _construct_gt_pos(self):
        pos = torch.as_tensor(
            self.env.task_manager.scene_object_pos,
            dtype=torch.float32,
        )
        self.obs_commands_buffer.append(pos.reshape(pos.shape[0], -1).to("cpu"))

    def _construct_gt_rot(self):
        quat = torch.as_tensor(
            self.env.task_manager.scene_object_quat,
            dtype=torch.float32,
        )
        if self.cfg.rot_format == "6d":
            rot = rot6d_from_matrix(matrix_from_quat(quat.reshape(-1, 4))).reshape(
                quat.shape[0],
                quat.shape[1],
                6,
            )
        else:
            rot = quat
        self.obs_commands_buffer.append(rot.reshape(rot.shape[0], -1).to("cpu"))

    def _construct_gt_arti(self):
        arti = torch.as_tensor(
            self.env.task_manager.scene_object_arti_vals,
            dtype=torch.float32,
        )
        self.obs_commands_buffer.append(arti.reshape(arti.shape[0], -1).to("cpu"))

    def _construct_pcd(self, append=True):
        chunks = []
        pcd_points_by_id: dict[str, int] = {}
        task = self.env.task_manager
        for object_index, object_id in enumerate(self.object_ids):
            entity = self.env.physics_manager.object_entities[object_id]
            pos = torch.as_tensor(
                task.scene_object_pos[:, object_index],
                dtype=torch.float32,
                device=self.env.device,
            )
            quat = torch.as_tensor(
                task.scene_object_quat[:, object_index],
                dtype=torch.float32,
                device=self.env.device,
            )
            arti, joint_names = self._reference_object_arti_for_entity(
                object_index=object_index,
                object_id=object_id,
            )
            chunk = compute_kpts_traj(
                local_frame_kpts_list=entity.object_surface_kpts,
                pos=pos,
                quat=quat,
                arti=arti,
                obj_inst=entity.urdf_inst,
                jnames=joint_names,
                local_frame_normals_list=entity.object_surface_normals,
                num_pts_sample=self.cfg.object_pcd_num_pts_sample,
            )
            if chunk.ndim != 3 or chunk.shape[2] != 6:
                raise ValueError(
                    f"Expected reference PCD for object {object_id!r} to have "
                    f"shape (T, P, 6), got {tuple(chunk.shape)}."
                )
            chunks.append(chunk)
            pcd_points_by_id[object_id] = int(chunk.shape[1])
        self.pcd_points_by_id = pcd_points_by_id
        pcd_time_series = torch.cat(chunks, dim=1).reshape(chunks[0].shape[0], -1)
        pcd_time_series = pcd_time_series.to("cpu")
        if append:
            self.obs_commands_buffer.append(pcd_time_series)
        return pcd_time_series

    def _reference_object_arti_for_entity(
        self, *, object_index: int, object_id: str
    ) -> tuple[torch.Tensor, list[str]]:
        task = self.env.task_manager
        entity = self.env.physics_manager.object_entities[object_id]
        if len(entity.joints) == 0:
            return (
                torch.zeros(
                    (task.total_steps, 0),
                    dtype=torch.float32,
                    device=self.env.device,
                ),
                [],
            )
        valid_indices = (
            torch.as_tensor(
                task.scene_object_arti_valid_mask[object_index],
                dtype=torch.bool,
                device=self.env.device,
            )
            .nonzero(as_tuple=False)
            .reshape(-1)
        )
        exported_joint_names = list(task.scene_object_arti_names[object_index])
        if len(exported_joint_names) != int(valid_indices.numel()):
            raise ValueError(
                "Expected scene_object_arti_names length to match valid articulation "
                f"count for object {object_id!r}, got {len(exported_joint_names)} "
                f"names and {int(valid_indices.numel())} valid values."
            )
        missing = [
            joint.name
            for joint in entity.joints
            if joint.name not in exported_joint_names
        ]
        if missing:
            raise ValueError(
                f"Reference articulation for object {object_id!r} is missing "
                f"simulated joints {missing}."
            )
        joint_indices = torch.as_tensor(
            [exported_joint_names.index(joint.name) for joint in entity.joints],
            dtype=torch.long,
            device=self.env.device,
        )
        raw_arti = torch.as_tensor(
            task.scene_object_arti_vals[:, object_index],
            dtype=torch.float32,
            device=self.env.device,
        )
        valid_values = raw_arti.index_select(1, valid_indices)
        return valid_values.index_select(1, joint_indices), [
            joint.name for joint in entity.joints
        ]

    def _construct_obj_bps(self):
        if "pcd" not in self.obs_command_terms:
            pcd = self._construct_pcd(append=False)
        else:
            pcd = self.obs_commands_buffer[self.obs_command_terms.index("pcd")]

        pcd = pcd.reshape(pcd.shape[0], -1, 6).to(self.env.device)
        bps = {}
        for side in ("left", "right"):
            if side not in self.env.physics_manager.scene_entities:
                continue
            hand: Hand = self.env.physics_manager.scene_entities[side]
            jpos = self._reference_hand_points_tensor(side=side, hand=hand)
            result = compute_bps(pcd, jpos)
            if result.shape[1:] != (len(hand.human_joint_names), 6):
                raise ValueError(
                    f"Expected {side} BPS shape {(len(hand.human_joint_names), 6)}, "
                    f"got {tuple(result.shape[1:])}."
                )
            bps[side] = result.to("cpu")
            self.obs_command_terms.append(f"{side}_bps")
            self.obs_command_dims.append(result.shape[1] * result.shape[2])
            self.obs_commands_buffer.append(
                result.reshape(result.shape[0], -1).to("cpu")
            )
        return bps

    def named_view(self, command: torch.Tensor) -> dict:
        named_splits = super().named_view(command)
        self._add_object_command_slices(named_splits)
        return named_splits

    def _add_object_command_slices(self, named: dict[str, torch.Tensor]) -> None:
        if "gt_pos" in named:
            pos = named["gt_pos"].view(
                named["gt_pos"].shape[0],
                len(self.object_ids),
                3,
            )
            named["gt_pos_all"] = pos
            for object_index, object_id in enumerate(self.object_ids):
                named[f"gt_pos_{object_id}"] = pos[:, object_index]
        if "gt_rot" in named:
            rot_dim = 6 if self.cfg.rot_format == "6d" else 4
            rot = named["gt_rot"].view(
                named["gt_rot"].shape[0],
                len(self.object_ids),
                rot_dim,
            )
            named["gt_rot_all"] = rot
            for object_index, object_id in enumerate(self.object_ids):
                named[f"gt_rot_{object_id}"] = rot[:, object_index]
        if "gt_arti" in named:
            arti_dim = int(self.env.task_manager.scene_object_arti_vals.shape[2])
            arti = named["gt_arti"].view(
                named["gt_arti"].shape[0],
                len(self.object_ids),
                arti_dim,
            )
            named["gt_arti_all"] = arti
            for object_index, object_id in enumerate(self.object_ids):
                named[f"gt_arti_{object_id}"] = arti[:, object_index]
        if "pcd" in named:
            if not hasattr(self, "pcd_points_by_id"):
                raise ValueError("Command PCD slices requested before PCD was built.")
            pcd = named["pcd"].view(named["pcd"].shape[0], -1, 6)
            named["pcd_all"] = pcd
            start = 0
            for object_id in self.object_ids:
                n_points = self.pcd_points_by_id[object_id]
                named[f"pcd_{object_id}"] = pcd[:, start : start + n_points]
                start += n_points
            if start != pcd.shape[1]:
                raise ValueError(
                    f"Expected command PCD to contain {start} points from object "
                    f"slices, got {pcd.shape[1]}."
                )
