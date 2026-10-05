from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from dexlink.utils.retargeting.math_utils import (
    matrix_from_quat,
    quat_from_matrix,
    rot6d_from_matrix,
)
from dexlink.utils.retargeting.mesh_utils import compute_bps, compute_kpts_traj

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import ObsCfg
    from dexlink.src.rl.world.art_object import ArtObject
    from dexlink.src.rl.world.env import Env
    from dexlink.src.rl.world.robot import Hand


class ObservationManager:
    def __init__(self, cfg: ObsCfg, env: Env):
        self.cfg = cfg
        self.env = env
        self.active_hand_sides = self.env.task_manager.active_hand_sides
        self.hand_obs = (
            cfg.use_hand_obs
            and (not env.cfg.task_cfg.object_only_scene)
            and len(self.active_hand_sides) > 0
        )
        self.policy_object_pcd_obs = cfg.use_policy_object_pcd
        self.privileged_object_pcd_obs = cfg.use_privileged_object_pcd
        self.object_obs = cfg.use_object_obs
        self.object_vel_obs = cfg.use_object_vel_obs
        self.task_command_obs = cfg.task_command
        self.physics = self.env.physics_manager

        self.rot_format = cfg.rot_format
        self.policy_obs_terms = []
        self.policy_term_dims = []
        self.policy_term_fns = []
        self.privileged_obs_terms = []
        self.privileged_term_dims = []
        self.privileged_term_fns = []
        if self.hand_obs:
            for side, fn in [
                ("left", self._compute_left_obs),
                ("right", self._compute_right_obs),
            ]:
                if not self.env.task_manager.has_hand(side):
                    continue
                hand = self.env.physics_manager.scene_entities[side]
                self._register_term("policy", side, hand.obs_dim, fn)
                self._register_term("privileged", side, hand.obs_dim, fn)
                ref_dim = 3 + 6 + len(hand.finger_joint_names)
                self._register_term(
                    "policy",
                    f"ref_{side}",
                    ref_dim,
                    lambda side=side: self._compute_reference_hand_action_obs(side),
                )
                self._register_term(
                    "privileged",
                    f"ref_{side}",
                    ref_dim,
                    lambda side=side: self._compute_reference_hand_action_obs(side),
                )
                if hand.obs_cfg.use_jacc_privileged:
                    self._register_term(
                        "privileged",
                        f"{side}_jacc",
                        len(hand.dof_map),
                        lambda side=side: self._compute_hand_joint_acceleration(side),
                    )

        object_entity = self.env.physics_manager.scene_entities["object"]
        self.asset: ArtObject = object_entity
        rot_dim = 4
        if self.rot_format == "6d":
            rot_dim = 6
        self.policy_object_pcd_dim = 3
        self.privileged_object_pcd_dim = 6
        self.policy_hand_pcd_dim = 3
        self.object_surface_n_pts = sum(
            kpts.shape[0] for kpts in self.asset.object_surface_kpts
        )
        self.policy_object_pcd_n_pts = cfg.policy_object_pcd_n_pts
        if cfg.privileged_object_pcd_n_pts == -1:
            self.privileged_object_pcd_n_pts = self.object_surface_n_pts
        elif cfg.privileged_object_pcd_n_pts > 0:
            self.privileged_object_pcd_n_pts = min(
                cfg.privileged_object_pcd_n_pts,
                self.object_surface_n_pts,
            )
        else:
            raise ValueError(
                "obs_cfg.privileged_object_pcd_n_pts must be -1 or a positive integer, "
                f"got {cfg.privileged_object_pcd_n_pts}."
            )
        obj_dim = 3 + rot_dim + len(object_entity.dof_map)
        if self.policy_object_pcd_obs:
            self._register_term(
                "policy",
                "object_pcd",
                self.policy_object_pcd_n_pts * self.policy_object_pcd_dim,
                self._compute_policy_object_pcd_obs,
            )
        if self.env.task_manager.has_hand("right"):
            right_hand: Hand = self.env.physics_manager.scene_entities["right"]
            self.right_hand_pcd_link_names = right_hand.hand_tree_link_names
            self.policy_hand_pcd_n_pts = right_hand.get_robot_pcd_n_pts(
                links_to_include=self.right_hand_pcd_link_names
            )
            if cfg.use_policy_hand_pcd:
                self._register_term(
                    "policy",
                    "hand_pcd",
                    self.policy_hand_pcd_n_pts * self.policy_hand_pcd_dim,
                    self._compute_policy_hand_pcd_obs,
                )
        else:
            self.right_hand_pcd_link_names = []
            self.policy_hand_pcd_n_pts = 0
        if self.privileged_object_pcd_obs:
            self._register_term(
                "privileged",
                "object_pcd",
                self.privileged_object_pcd_n_pts * self.privileged_object_pcd_dim,
                self._compute_privileged_object_pcd_obs,
            )
        self._register_term("privileged", "object", obj_dim, self._compute_object_obs)
        if self.object_obs:
            self._register_term("policy", "object", obj_dim, self._compute_object_obs)

        obj_vel_dim = 6 + len(object_entity.dof_map)
        self._register_term(
            "privileged",
            "object_vel",
            obj_vel_dim,
            self._compute_object_vel_obs,
        )
        if self.object_vel_obs:
            self._register_term(
                "policy",
                "object_vel",
                obj_vel_dim,
                self._compute_object_vel_obs,
            )

        if self.task_command_obs:
            if self.cfg.use_command_timestep:
                command_dim = 1  # just the timestep index
            else:
                command_dim = self.env.commands_manager.obs_command_dim
            self._register_term("policy", "command", command_dim, self._get_commands)
            self._register_term(
                "privileged", "command", command_dim, self._get_commands
            )

        self.use_contact = self.cfg.use_contact
        self.contact_signal_threshold = 1e-4
        self._cached_privileged_object_pcd: torch.Tensor | None = None

        contact_obs = self.hand_obs and self.use_contact
        if contact_obs:
            term_dim = 0
            for side in self.active_hand_sides:
                hand: Hand = self.env.physics_manager.scene_entities[side]
                term_dim += len(hand.tactile_links_global_indices) * 3
            if term_dim > 0:
                self._register_term(
                    "policy", "contact", term_dim, self._compute_contact_obs
                )
                self._register_term(
                    "privileged", "contact", term_dim, self._compute_contact_obs
                )
        # Backward-compatible aliases: "obs" means policy obs.
        self.obs_terms = self.policy_obs_terms
        self.term_dims = self.policy_term_dims
        self.term_fns = self.policy_term_fns

    def _register_term(self, view: str, term: str, dim: int, fn) -> None:

        if (self.env.cfg.backend == "real") and (view == "privileged"):
            return

        terms = self._get_terms(view)
        dims = self._get_term_dims(view)
        fns = self._get_term_fns(view)
        terms.append(term)
        dims.append(dim)
        fns.append(fn)

    def _get_terms(self, view: str) -> list[str]:
        if view == "policy":
            return self.policy_obs_terms
        if view == "privileged":
            return self.privileged_obs_terms
        raise ValueError(f"Unknown observation view: {view}")

    def _get_term_dims(self, view: str) -> list[int]:
        if view == "policy":
            return self.policy_term_dims
        if view == "privileged":
            return self.privileged_term_dims
        raise ValueError(f"Unknown observation view: {view}")

    def _get_term_fns(self, view: str) -> list:
        if view == "policy":
            return self.policy_term_fns
        if view == "privileged":
            return self.privileged_term_fns
        raise ValueError(f"Unknown observation view: {view}")

    @property
    def observation_dim(self) -> int:
        return sum(self.policy_term_dims)

    @property
    def privileged_observation_dim(self) -> int:
        return sum(self.privileged_term_dims)

    def initialize(self):
        pass

    def reset_step_cache(self) -> None:
        self._cached_privileged_object_pcd = None

    def summary(self) -> str:
        spacing = 20
        lines = ["[ObservationManager]"]
        for view, terms, dims in (
            ("Policy", self.policy_obs_terms, self.policy_term_dims),
            ("Privileged", self.privileged_obs_terms, self.privileged_term_dims),
        ):
            lines.append(f"{view} Observations")
            lines.append(f"{'Observation Term':<{spacing}} {'Dimension':<10}")
            lines.append("-" * (spacing + 10))
            for term, dim in zip(terms, dims):
                lines.append(f"{term:<{spacing}} {dim:<10}")
                for subterm, subdim in self._summary_subterms(term):
                    lines.append(f"  - {subterm:<{spacing - 4}} {subdim:<10}")
            lines.append("-" * (spacing + 10))
            lines.append(f"{'Total':<{spacing}} {sum(dims):<10}")
        return "\n".join(lines)

    def _summary_subterms(self, term: str) -> list[tuple[str, int]]:
        if term in {"left", "right"}:
            if term not in self.env.physics_manager.scene_entities:
                return []
            hand: Hand = self.env.physics_manager.scene_entities[term]
            return list(zip(hand.obs_terms, hand.obs_dims))

        if term.startswith("ref_"):
            side = term.removeprefix("ref_")
            if side not in self.env.physics_manager.scene_entities:
                return []
            hand: Hand = self.env.physics_manager.scene_entities[side]
            return [
                ("wrist_pos", 3),
                ("wrist_rot_6d", 6),
                ("finger_jpos", len(hand.finger_joint_names)),
            ]

        if term == "object":
            subterms = [("pos", 3)]
            rot_dim = 6 if self.rot_format == "6d" else 4
            subterms.append(("rot", rot_dim))
            if self.asset.joints:
                subterms.append(("arti", len(self.asset.dof_map)))
            return subterms

        if term == "object_vel":
            subterms = [("lin_vel", 3), ("ang_vel", 3)]
            if self.asset.joints:
                subterms.append(("arti_vel", len(self.asset.dof_map)))
            return subterms

        return []

    def _compute_left_obs(self):
        left_hand: Hand = self.env.physics_manager.scene_entities["left"]
        left_obs = left_hand.get_obs()  # (num_envs, dof)
        return left_obs

    def _compute_right_obs(self):
        right_hand: Hand = self.env.physics_manager.scene_entities["right"]
        right_obs = right_hand.get_obs()  # (num_envs, dof)
        return right_obs

    def _compute_hand_joint_acceleration(self, side: str):
        hand: Hand = self.env.physics_manager.scene_entities[side]
        return hand._jacc_obs()

    def _compute_reference_hand_action_obs(self, side: str) -> torch.Tensor:
        ref_obs = getattr(self.env.reset_manager, f"frame_ref_{side}")
        if ref_obs is None:
            raise ValueError(
                f"Reference action observation for side {side!r} is not available in ResetManager."
            )
        command_timestep = torch.clamp(
            self.env.command_time_step,
            max=self.env.reset_manager.reference_frame_end_timestep(),
        ).to(
            dtype=torch.long,
            device=self.env.device,
        )
        selected_ref_obs = ref_obs.to(self.env.device)[command_timestep]
        if self.cfg.mask_reference_hand_action_obs:
            return torch.zeros_like(selected_ref_obs)
        return selected_ref_obs

    def _compute_object_obs(self):
        object_entity: ArtObject = self.env.physics_manager.scene_entities["object"]
        object_obs = object_entity.get_obs(
            rot_format=self.rot_format
        )  # (num_envs, 3+4/9+[obj_dofs])

        return object_obs

    def _compute_object_vel_obs(self):
        object_entity: ArtObject = self.env.physics_manager.scene_entities["object"]
        return object_entity.get_vel_obs()  # (num_envs, 6+[obj_dofs])

    def _compute_privileged_object_pcd(self) -> torch.Tensor:
        if self._cached_privileged_object_pcd is None:
            self._cached_privileged_object_pcd = (
                self._compute_privileged_object_pcd_from_object_state()
            )
        return self._cached_privileged_object_pcd

    def _get_current_object_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        object_obs = self._compute_object_obs()
        n_obj_dofs = len(self.asset.dof_map)
        rot_dim = 6 if self.rot_format == "6d" else 4
        pos = object_obs[:, :3]
        quat = object_obs[:, 3 : 3 + rot_dim]
        arti = object_obs[:, 3 + rot_dim : 3 + rot_dim + n_obj_dofs]
        return pos, quat, arti

    def _compute_privileged_object_pcd_from_object_state(self) -> torch.Tensor:
        pos, quat, arti = self._get_current_object_state()
        return self.compute_object_pcd_obs(pos=pos, quat=quat, arti=arti, normals=True)

    def _compute_privileged_object_pcd_obs(self):
        privileged_object_pcd = self._compute_privileged_object_pcd()
        return privileged_object_pcd.reshape(self.env.num_envs, -1)

    def _compute_policy_object_pcd_obs(self):
        policy_object_pcd = self.physics.get_object_pcd(normals=False)
        expected_shape = (
            self.env.num_envs,
            self.policy_object_pcd_n_pts,
            self.policy_object_pcd_dim,
        )
        if tuple(policy_object_pcd.shape) != expected_shape:
            raise ValueError(
                f"Expected policy object PCD shape {expected_shape}, got {tuple(policy_object_pcd.shape)}."
            )
        return policy_object_pcd.reshape(self.env.num_envs, -1)

    def _compute_policy_hand_pcd_obs(self):
        right_hand: Hand = self.env.physics_manager.scene_entities["right"]
        right_jpos = right_hand._jpos_obs()
        if self.env.cfg.backend == "real":
            hand_pcd = self.physics.get_policy_hand_pcd(
                right_hand,
                links_to_include=self.right_hand_pcd_link_names,
            )
        else:
            hand_pcd = right_hand.get_robot_pcd(
                right_jpos,
                links_to_include=self.right_hand_pcd_link_names,
            )
        expected_shape = (
            self.env.num_envs,
            self.policy_hand_pcd_n_pts,
            self.policy_hand_pcd_dim,
        )
        if tuple(hand_pcd.shape) != expected_shape:
            raise ValueError(
                f"Expected policy hand PCD shape {expected_shape}, got {tuple(hand_pcd.shape)}."
            )
        return hand_pcd.reshape(self.env.num_envs, -1)

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

    def get_object_pcd_from_named_obs(
        self,
        named_obs: dict[str, torch.Tensor],
        expected_feat_dim: int,
    ) -> torch.Tensor:
        object_pcd = named_obs.get("object_pcd")
        if object_pcd is None:
            raise ValueError("Observation is missing required 'object_pcd' term.")
        return object_pcd.view(object_pcd.shape[0], -1, expected_feat_dim)

    def get_hand_pcd_from_named_obs(
        self,
        named_obs: dict[str, torch.Tensor],
        expected_feat_dim: int,
    ) -> torch.Tensor:
        hand_pcd = named_obs.get("hand_pcd")
        if hand_pcd is None:
            raise ValueError("Observation is missing required 'hand_pcd' term.")
        return hand_pcd.view(hand_pcd.shape[0], -1, expected_feat_dim)

    def compute_bps(self, pcd, bps):
        # pcd: (N, n_pts, 6) -> x, y, z, nx, ny, nz
        # bps: (N, n_bps, 3)
        # returns: (N, n_bps, 6) -> direction to closest point + normal at that point
        return compute_bps(pcd, bps)

    def compute_object_pcd_obs(
        self, pos, quat, arti, normals: bool = False
    ) -> torch.Tensor:
        rot_dim = quat.shape[-1]
        if rot_dim == 4:
            pass
        elif rot_dim == 6:
            x_axis = quat[:, :3]
            y_axis = quat[:, 3:6]
            z_axis = torch.cross(x_axis, y_axis, dim=-1)
            mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)
            quat = quat_from_matrix(mat)
        else:
            raise ValueError(f"Expected object rotation dim 4 or 6, got {rot_dim}.")

        pcd = compute_kpts_traj(
            self.asset.object_surface_kpts,
            pos,
            quat,
            arti=arti,
            obj_inst=self.asset.urdf_inst,
            jnames=[j.name for j in self.asset.joints],
            local_frame_normals_list=(
                self.asset.object_surface_normals if normals else None
            ),
            num_pts_sample=self.privileged_object_pcd_n_pts,
        )
        expected_dim = (
            self.privileged_object_pcd_dim if normals else self.policy_object_pcd_dim
        )
        if pcd.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected object PCD feature dim {expected_dim}, got {pcd.shape[-1]}."
            )
        return pcd

    def compute_observations(self) -> dict[str, torch.Tensor]:
        policy_obs = self._compute_view_observations("policy")
        policy_named_obs = self.named_view_policy(policy_obs)
        policy_named_obs = self.post_process_policy_obs(policy_named_obs)
        policy_obs = self.flatten_policy_named_obs(policy_named_obs)
        return {
            "policy": policy_obs,
            "privileged": self._compute_view_observations("privileged"),
        }

    def get_contact_probe(
        self, named_policy_obs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | list[str]]:
        contact_obs = named_policy_obs.get("contact")
        if contact_obs is None:
            raise ValueError(
                "Policy observation term 'contact' is missing. Contact probes require obs_cfg.use_contact=True."
            )

        expected_dim = 0
        for side in self.active_hand_sides:
            hand: Hand = self.env.physics_manager.scene_entities[side]
            expected_dim += len(hand.tactile_link_names) * 3
        expected_shape = (self.env.num_envs, expected_dim)
        if tuple(contact_obs.shape) != expected_shape:
            raise ValueError(
                "Contact observation shape mismatch: expected "
                f"{expected_shape}, got {tuple(contact_obs.shape)}."
            )

        finger_order = ["thumb", "index", "middle", "ring", "little"]
        per_finger_contacts: list[torch.Tensor] = []
        finger_names: list[str] = []
        offset = 0
        for side in self.active_hand_sides:
            hand = self.env.physics_manager.scene_entities[side]
            n_tactile_links = len(hand.tactile_link_names)
            if n_tactile_links != len(finger_order):
                raise ValueError(
                    f"Expected {len(finger_order)} tactile fingertip links for hand {side!r}, "
                    f"got {n_tactile_links}: {hand.tactile_link_names}."
                )
            side_dim = n_tactile_links * 3
            side_contact = contact_obs[:, offset : offset + side_dim]
            expected_side_shape = (self.env.num_envs, side_dim)
            if tuple(side_contact.shape) != expected_side_shape:
                raise ValueError(
                    f"Contact slice for hand {side!r} has shape {tuple(side_contact.shape)}, "
                    f"expected {expected_side_shape}."
                )
            side_contact = side_contact.reshape(self.env.num_envs, n_tactile_links, 3)
            per_finger_contacts.append(side_contact.amax(dim=-1))
            finger_names.extend(
                [f"{side}_{finger_name}" for finger_name in finger_order]
            )
            offset += side_dim

        if offset != contact_obs.shape[1]:
            raise ValueError(
                "Contact observation flattening mismatch: consumed "
                f"{offset} dims, but contact term has {contact_obs.shape[1]} dims."
            )
        if len(per_finger_contacts) == 0:
            raise ValueError(
                "No contact observations were collected for the active hands."
            )

        return {
            "contact": contact_obs.detach().clone(),
            "per_finger_contact": torch.cat(per_finger_contacts, dim=-1)
            .detach()
            .clone(),
            "finger_names": finger_names,
        }

    def _compute_contact_obs(self):
        target_entity = self.env.physics_manager.scene_entities["object"]

        _obs = []
        for side in self.active_hand_sides:
            hand: Hand = self.env.physics_manager.scene_entities[side]
            contact_signal = self.physics.get_hand_object_binary_contact_signal(
                hand,
                target_entity,
                threshold=self.contact_signal_threshold,
            )
            n_envs = contact_signal.shape[0]
            contact_signal = contact_signal.reshape(
                n_envs, len(hand.tactile_links_global_indices) * 3
            )

            _obs.append(contact_signal)

        return torch.cat(_obs, dim=-1)

    def _compute_view_observations(self, view: str) -> torch.Tensor:
        obs = []
        for _term, _fn in zip(self._get_terms(view), self._get_term_fns(view)):
            obs.append(_fn())

        if len(obs) == 0:
            return torch.zeros(
                (self.env.num_envs, 0), dtype=torch.float32, device=self.env.device
            )
        return torch.cat(obs, dim=-1)

    def _named_view_tensor(self, obs: torch.Tensor, view: str) -> dict:
        """Return a dictionary mapping observation terms to their respective tensors."""
        term_dims = self._get_term_dims(view)
        obs_terms = self._get_terms(view)
        splits = torch.split(obs, term_dims, dim=-1)
        named_splits = dict(zip(obs_terms, splits))

        # also add hand observations separately for convenience
        if "left" in named_splits:
            left_obs = named_splits["left"]
            left_hand: Hand = self.env.physics_manager.scene_entities["left"]
            named_left_obs = left_hand.obs_named_view(left_obs)
            named_splits.update({f"left_{k}": v for k, v in named_left_obs.items()})

        if "right" in named_splits:
            right_obs = named_splits["right"]
            right_hand: Hand = self.env.physics_manager.scene_entities["right"]
            named_right_obs = right_hand.obs_named_view(right_obs)
            named_splits.update({f"right_{k}": v for k, v in named_right_obs.items()})

        # also add object pos, quat, arti separately for convenience
        if "object" in named_splits:
            object_obs = named_splits["object"]
            object_entity = self.env.physics_manager.scene_entities["object"]
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
            object_entity = self.env.physics_manager.scene_entities["object"]
            n_obj_dofs = len(object_entity.dof_map)
            named_splits["obj_lin_vel"] = object_vel_obs[:, :3]
            named_splits["obj_ang_vel"] = object_vel_obs[:, 3:6]
            named_splits["obj_arti_vel"] = object_vel_obs[:, 6 : 6 + n_obj_dofs]

        return named_splits

    def named_view_policy(self, obs: torch.Tensor) -> dict:
        return self._named_view_tensor(obs=obs, view="policy")

    def named_view_privileged(self, obs: torch.Tensor) -> dict:
        return self._named_view_tensor(obs=obs, view="privileged")

    def get_ground_truth_info(self) -> dict:
        # the observations do not necessarily contain all the information
        # should be used only for evaluation: how good the trajectory is
        gt = {}

        asset: ArtObject = self.env.physics_manager.scene_entities["object"]
        gt["object"] = asset.get_ground_truth_info()

        for side in self.active_hand_sides:
            hand = self.env.physics_manager.scene_entities[side]
            hand_info = hand.get_ground_truth_info()
            wrist_pos = self.env.current_wrist_pos_by_side.get(side)
            wrist_quat = self.env.current_wrist_quat_by_side.get(side)
            if wrist_pos is None or wrist_quat is None:
                raise ValueError(
                    f"Current wrist pose is not populated for hand {side!r}."
                )
            hand_info.update(
                {
                    "wrist_pos": wrist_pos.clone(),
                    "wrist_quat": wrist_quat.clone(),
                    "jvals": self.physics.get_hand_joint_positions(hand).detach(),
                    "jnames": list(hand.joint_names),
                }
            )
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
        original_demo = self.env.task_manager.original_demo
        retargeted_motion = getattr(original_demo, "retargeted_motion", None)
        if not isinstance(retargeted_motion, dict):
            return {}

        out = {}
        for side in ("left", "right"):
            if side not in self.env.physics_manager.scene_entities:
                continue
            jvals_key = f"{side}_jvals"
            jnames_key = f"{side}_jnames"
            if (jvals_key not in retargeted_motion) or (
                jnames_key not in retargeted_motion
            ):
                continue

            hand: Hand = self.env.physics_manager.scene_entities[side]
            retgt_jnames = [str(n) for n in retargeted_motion[jnames_key]]
            if len(retgt_jnames) != len(hand.joint_names) or any(
                jn not in hand.joint_names for jn in retgt_jnames
            ):
                continue

            rollout_jvals = self.physics.get_hand_joint_positions(hand).detach()
            hand_idx_map = [hand.joint_names.index(jn) for jn in retgt_jnames]
            rollout_jvals = rollout_jvals[:, hand_idx_map]

            out[side] = {
                "rollout": rollout_jvals.clone(),
            }
        return out

    def post_process_policy_obs(self, named_obs):
        """
        for object pcd, transform it relative to the right hand wrist pose
        which should also be queried from the named obs (wrist_pose I think)
        the robot joint pos should be as it is.
        the link pos of hand fingers should be relative to the wrist (subtract the wrist translation from the link pos)
        the contact signal is already computed in the desired binarized form
        the other terms (object pose) should also be transformed relative to the wrist pose
        X_WoWr = wrist pose in world frame
        X_WoO = object pose in world frame
        X_WrO = object pose in wrist frame = X_WoWr^-1 * X_WoO
        for object pcd instead, we do a similar transformation to get the pcd in the wrist frame.
        """
        processed = dict(named_obs)
        if len(self.active_hand_sides) == 0:
            return processed
        frame_side = (
            "right" if "right" in self.active_hand_sides else self.active_hand_sides[0]
        )
        frame_wrist_key = f"{frame_side}_wrist_pose"
        frame_wrist_pose = processed.get(frame_wrist_key)
        if frame_wrist_pose is None:
            raise ValueError(
                f"Policy-frame wrist observation {frame_wrist_key!r} is missing."
            )
        if frame_wrist_pose.shape[-1] != 9:
            raise ValueError(
                f"Expected {frame_wrist_key} to have dim 9, got {frame_wrist_pose.shape[-1]}."
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
                        f"Expected {side!r}-hand jpos observation shape "
                        f"{expected_shape}, got {tuple(side_jpos.shape)}."
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
                [side_named_obs[term] for term in hand.obs_terms], dim=-1
            )

        if "object" in processed:
            object_obs = processed["object"]
            object_entity = self.env.physics_manager.scene_entities["object"]
            n_obj_dofs = len(object_entity.dof_map)
            rot_dim = 6 if self.rot_format == "6d" else 4
            obj_pos = object_obs[:, :3]
            obj_rot = object_obs[:, 3 : 3 + rot_dim]
            obj_arti = object_obs[:, 3 + rot_dim : 3 + rot_dim + n_obj_dofs]

            rel_obj_pos = torch.matmul(
                wrist_rot_t, (obj_pos - wrist_pos).unsqueeze(-1)
            ).squeeze(-1)
            if self.rot_format == "6d":
                obj_x_axis = obj_rot[:, :3]
                obj_y_axis = obj_rot[:, 3:6]
                obj_z_axis = torch.cross(obj_x_axis, obj_y_axis, dim=-1)
                obj_rot_mat = torch.stack([obj_x_axis, obj_y_axis, obj_z_axis], dim=-1)
                rel_obj_rot_mat = torch.matmul(wrist_rot_t, obj_rot_mat)
                rel_obj_rot = rot6d_from_matrix(rel_obj_rot_mat)
            else:
                obj_rot_mat = matrix_from_quat(obj_rot)
                rel_obj_rot_mat = torch.matmul(wrist_rot_t, obj_rot_mat)
                rel_obj_rot = quat_from_matrix(rel_obj_rot_mat)

            processed["object"] = torch.cat(
                [rel_obj_pos, rel_obj_rot, obj_arti], dim=-1
            )
            processed["obj_pos"] = rel_obj_pos
            processed["obj_quat"] = rel_obj_rot
            processed["obj_arti"] = obj_arti

        if "object_pcd" in processed:
            object_pcd = self.get_object_pcd_from_named_obs(
                processed, expected_feat_dim=self.policy_object_pcd_dim
            )
            if object_pcd.shape[-1] != 3:
                raise ValueError(
                    "Expected object_pcd points to have xyz coordinates with shape "
                    f"(num_envs, n_pts, 3), got {tuple(object_pcd.shape)}."
                )
            rel_object_pcd = object_pcd - wrist_pos[:, None, :]
            processed["object_pcd"] = rel_object_pcd.reshape(self.env.num_envs, -1)

        if "hand_pcd" in processed:
            hand_pcd = self.get_hand_pcd_from_named_obs(
                processed, expected_feat_dim=self.policy_hand_pcd_dim
            )
            if hand_pcd.shape[-1] != 3:
                raise ValueError(
                    "Expected hand_pcd points to have xyz coordinates with shape "
                    f"(num_envs, n_pts, 3), got {tuple(hand_pcd.shape)}."
                )
            rel_hand_pcd = hand_pcd - wrist_pos[:, None, :]
            processed["hand_pcd"] = rel_hand_pcd.reshape(self.env.num_envs, -1)

        if (
            self.env.cfg.backend == "sim"
            and "object_pcd" in processed
            and "hand_pcd" in processed
        ):
            object_pcd = self.get_object_pcd_from_named_obs(
                processed, expected_feat_dim=self.policy_object_pcd_dim
            )
            hand_pcd = self.get_hand_pcd_from_named_obs(
                processed, expected_feat_dim=self.policy_hand_pcd_dim
            )
            if object_pcd.shape[-1] != 3:
                raise ValueError(
                    "Expected object_pcd points to have xyz coordinates with shape "
                    f"(num_envs, n_pts, 3), got {tuple(object_pcd.shape)}."
                )
            if hand_pcd.shape[-1] != 3:
                raise ValueError(
                    "Expected hand_pcd points to have xyz coordinates with shape "
                    f"(num_envs, n_pts, 3), got {tuple(hand_pcd.shape)}."
                )

            n_object_pts = object_pcd.shape[1]
            n_hand_pts = hand_pcd.shape[1]
            if n_hand_pts <= 0:
                raise ValueError("Expected hand_pcd to contain at least one point.")

            replace_ratio = 0.2 + 0.4 * torch.rand(
                (), device=object_pcd.device, dtype=object_pcd.dtype
            )
            replace_count = int(
                torch.clamp(
                    torch.round(replace_ratio * n_object_pts).to(torch.long),
                    min=1,
                    max=n_object_pts,
                ).item()
            )

            mixed_object_pcd = object_pcd.clone()
            noise_std = self.cfg.policy_object_pcd_jitter_std
            if noise_std < 0:
                raise ValueError(
                    "Expected obs_cfg.policy_object_pcd_jitter_std >= 0, "
                    f"got {noise_std}."
                )
            object_indices = torch.randperm(n_object_pts, device=object_pcd.device)[
                :replace_count
            ]
            hand_indices = torch.randint(
                low=0,
                high=n_hand_pts,
                size=(self.env.num_envs, replace_count),
                device=object_pcd.device,
            )
            batch_indices = torch.arange(
                self.env.num_envs, device=object_pcd.device
            ).unsqueeze(1)
            sampled_hand_points = hand_pcd[batch_indices, hand_indices]
            if noise_std > 0:
                sampled_hand_points = sampled_hand_points + (
                    torch.randn_like(sampled_hand_points) * noise_std
                )
            mixed_object_pcd[:, object_indices] = sampled_hand_points
            processed["object_pcd"] = mixed_object_pcd.reshape(self.env.num_envs, -1)

        # finally center the position of the wrist_pose
        processed[frame_wrist_key][:, :3] = torch.zeros_like(wrist_pos)

        return processed

    def flatten_policy_named_obs(
        self, named_obs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return torch.cat([named_obs[term] for term in self.policy_obs_terms], dim=-1)
