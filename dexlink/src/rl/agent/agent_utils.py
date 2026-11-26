from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from dexlink.src.rl.core_agent.agent_utils import (
    DiscObservationProcessor,
    PolicyNetwork,
    ValueNetwork,
)
from dexlink.utils.retargeting.math_utils import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_from_rotvec,
    quat_mul,
)
from dexlink.utils.rl.scene_cloud import (
    scene_cloud_contract,
    validate_scene_cloud_actor_inputs,
)

if TYPE_CHECKING:
    from dexlink.src.rl.world.env import Env


class CommandsProcessor(nn.Module):
    """Embed raw command observations, either from timestep or full command vectors."""

    def __init__(
        self,
        env: Env,
        embed_dim: int = 64,
        embed_from_timestep: bool = False,
    ):
        super().__init__()
        self.env = env
        self.embed_dim = embed_dim
        self.embed_from_timestep = embed_from_timestep
        obs_cfg = env.cfg.obs_cfg
        self.use_command_timestep = (
            obs_cfg.use_command_timestep and obs_cfg.task_command
        )

        if self.embed_from_timestep and not self.use_command_timestep:
            raise ValueError(
                "embed_from_timestep=True requires command timestep observations."
            )

        if self.embed_from_timestep:
            self.command_terms = ["command_timestep"]
            self.command_dims = [1]
            self.expected_input_dim = 1
        else:
            self.command_terms = list(env.commands_manager.obs_command_terms)
            if self.use_command_timestep:
                self.command_dims = list(env.commands_manager.policy_command_dims)
                self.expected_input_dim = 1
            else:
                self.command_dims = list(env.commands_manager.obs_command_dims)
                self.expected_input_dim = sum(self.command_dims)

        self.embed_layers = nn.ModuleDict()
        for term, dim in zip(self.command_terms, self.command_dims):
            layer = nn.Linear(dim, embed_dim)
            nn.init.xavier_uniform_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
            self.embed_layers[term] = layer

    def _normalize_command_timestep(
        self, command_timestep: torch.Tensor
    ) -> torch.Tensor:
        max_timestep = self.env.task_manager.total_steps - 1
        if max_timestep <= 0:
            raise ValueError(
                f"Expected task_manager.total_steps > 1, got {self.env.task_manager.total_steps}."
            )
        return command_timestep / max_timestep

    def forward(self, commands: torch.Tensor) -> torch.Tensor:
        if commands.shape[-1] != self.expected_input_dim:
            raise ValueError(
                f"Command dim mismatch: expected {self.expected_input_dim}, got {commands.shape[-1]}."
            )

        if self.use_command_timestep:
            command_timestep = commands.squeeze(-1).long()
            if self.embed_from_timestep:
                command_inputs = self._normalize_command_timestep(
                    command_timestep.to(dtype=commands.dtype)
                ).unsqueeze(-1)
            else:
                command_inputs = (
                    self.env.commands_manager.query_policy_commands_from_timesteps(
                        command_timestep
                    )
                )
        else:
            command_inputs = commands

        splits = torch.split(command_inputs, self.command_dims, dim=-1)
        embeds = []
        for term, chunk in zip(self.command_terms, splits):
            embeds.append(self.embed_layers[term](chunk))

        embed = torch.stack(embeds, dim=0).sum(dim=0)
        return embed


class PointCloudProcessor(nn.Module):
    """Encode object/hand point clouds with a PointNet-style backbone."""

    def __init__(
        self,
        *,
        object_n_points: int,
        hand_n_points: int,
        point_feat_dim: int,
        embed_dim: int,
        hidden_dims: tuple[int, ...],
        pool: str = "max",
        layer_norm: bool = True,
    ):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size.")
        if pool not in {"max", "mean"}:
            raise ValueError(f"Unsupported point cloud pooling '{pool}'.")
        self.object_n_points = object_n_points
        self.hand_n_points = hand_n_points
        self.point_feat_dim = point_feat_dim
        in_dim = point_feat_dim
        self.point_blocks = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.point_blocks.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                )
            )
            in_dim = 2 * hidden_dim
        self.output_proj = nn.Linear(hidden_dims[-1], embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()

    def _reshape_flat_pcd(
        self, pcd: torch.Tensor, *, n_points: int, name: str
    ) -> torch.Tensor:
        expected_dim = n_points * self.point_feat_dim
        if pcd.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected flattened {name} dim {expected_dim}, got {pcd.shape[-1]}."
            )
        return pcd.reshape(-1, n_points, self.point_feat_dim)

    def _encode_flat_pcd(
        self, pcd: torch.Tensor, *, n_points: int, name: str
    ) -> torch.Tensor:
        original_shape = pcd.shape[:-1]
        point_features = self._reshape_flat_pcd(pcd, n_points=n_points, name=name)
        for point_block in self.point_blocks:
            point_features = point_block(point_features)
            global_features = point_features.mean(dim=1, keepdim=True)
            point_features = torch.cat(
                [
                    point_features,
                    global_features.expand(-1, point_features.shape[1], -1),
                ],
                dim=-1,
            )
        final_point_features = point_features[..., : self.output_proj.in_features]
        pooled = final_point_features.mean(dim=1)
        pooled = self.output_proj(pooled)
        pooled = pooled.reshape(*original_shape, -1)
        return self.output_norm(pooled)

    def forward(
        self,
        object_pcd: torch.Tensor,
        hand_pcd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hand_pcd is not None:
            if hand_pcd.shape[:-1] != object_pcd.shape[:-1]:
                raise ValueError(
                    "Expected object_pcd and hand_pcd batch dimensions to match, got "
                    f"{tuple(object_pcd.shape[:-1])} and {tuple(hand_pcd.shape[:-1])}."
                )
            object_plus_hand_embed = self._encode_flat_pcd(
                torch.cat([object_pcd, hand_pcd], dim=-1),
                n_points=self.object_n_points + self.hand_n_points,
                name="object_plus_hand_pcd",
            )
            hand_embed = self._encode_flat_pcd(
                hand_pcd,
                n_points=self.hand_n_points,
                name="hand_pcd",
            )
            return object_plus_hand_embed - hand_embed
        return self._encode_flat_pcd(
            object_pcd,
            n_points=self.object_n_points,
            name="object_pcd",
        )


class AuxiliaryDecoder(nn.Module):
    """Small MLP head for observation-encoder auxiliary supervision."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (256,),
    ):
        super().__init__()
        if output_dim <= 0:
            raise ValueError(f"Expected positive output_dim, got {output_dim}.")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError(
                    f"Expected positive auxiliary hidden dim, got {hidden_dim}."
                )
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObservationProcessor(nn.Module):
    """Observation processor for DexLink policy inputs, including point clouds."""

    def __init__(
        self,
        env: Env,
        embed_dim: int = 64,
        layer_norm: bool = True,
        pcd_hidden_dims: tuple[int, ...] = (64, 64),
        pcd_pool: str = "max",
        selected_terms: tuple[str, ...] | None = None,
    ):
        super().__init__()
        self.env = env
        self.embed_dim = embed_dim
        self.obs_save_dim = env.observation_space.shape[1]
        self.true_obs_dim = self.obs_save_dim

        self.obs_terms = deepcopy(env.observation_manager.obs_terms)
        self.obs_term_dims = deepcopy(env.observation_manager.term_dims)

        self.full_obs_terms = deepcopy(self.obs_terms)
        self.full_obs_term_dims = deepcopy(self.obs_term_dims)
        if selected_terms is None:
            selected_terms = tuple(self.full_obs_terms)
        if len(selected_terms) == 0:
            raise ValueError(
                "selected_terms must contain at least one observation term."
            )
        missing_terms = [
            term for term in selected_terms if term not in self.full_obs_terms
        ]
        if missing_terms:
            raise ValueError(
                f"Requested observation terms {missing_terms} are not available in policy observations. "
                f"Available terms: {self.full_obs_terms}."
            )
        self.selected_terms = list(selected_terms)
        self.scene_cloud_contract = scene_cloud_contract(env.cfg.obs_cfg)
        self.shared_pcd_terms = [
            term for term in ("object_pcd", "hand_pcd") if term in self.selected_terms
        ]
        if (
            "hand_pcd" in self.shared_pcd_terms
            and "object_pcd" not in self.shared_pcd_terms
        ):
            raise ValueError(
                "hand_pcd requires object_pcd in selected_terms because the shared point-cloud "
                "processor computes net(object_pcd + hand_pcd) - net(hand_pcd)."
            )
        self.embed_layers = nn.ModuleDict()
        obs_mgr = env.observation_manager
        self.pcd_processor: PointCloudProcessor | None = None
        if self.shared_pcd_terms:
            pcd_feat_dim = obs_mgr.policy_object_pcd_dim
            term_dim_by_name = dict(zip(self.full_obs_terms, self.full_obs_term_dims))
            object_pcd_dim = term_dim_by_name["object_pcd"]
            if object_pcd_dim % pcd_feat_dim != 0:
                raise ValueError(
                    "Expected object_pcd observation dim to be divisible by "
                    f"policy_object_pcd_dim={pcd_feat_dim}, got {object_pcd_dim}."
                )
            object_pcd_n_points = object_pcd_dim // pcd_feat_dim
            hand_pcd_n_points = 0
            if "hand_pcd" in self.shared_pcd_terms:
                hand_pcd_feat_dim = obs_mgr.policy_hand_pcd_dim
                if hand_pcd_feat_dim != pcd_feat_dim:
                    raise ValueError(
                        "Expected object_pcd and hand_pcd to share the same point feature dim, "
                        f"got {pcd_feat_dim} and {hand_pcd_feat_dim}."
                    )
                hand_pcd_dim = term_dim_by_name["hand_pcd"]
                if hand_pcd_dim % hand_pcd_feat_dim != 0:
                    raise ValueError(
                        "Expected hand_pcd observation dim to be divisible by "
                        f"policy_hand_pcd_dim={hand_pcd_feat_dim}, got {hand_pcd_dim}."
                    )
                hand_pcd_n_points = hand_pcd_dim // hand_pcd_feat_dim
            self.pcd_processor = PointCloudProcessor(
                object_n_points=object_pcd_n_points,
                hand_n_points=hand_pcd_n_points,
                point_feat_dim=pcd_feat_dim,
                embed_dim=self.embed_dim,
                hidden_dims=pcd_hidden_dims,
                pool=pcd_pool,
                layer_norm=layer_norm,
            )
        for term, dim in zip(self.obs_terms, self.obs_term_dims):
            if term == "command":
                layer = CommandsProcessor(
                    env,
                    embed_dim=self.embed_dim,
                    embed_from_timestep=env.cfg.obs_cfg.embed_commands_from_timestep,
                )
            elif term in {"object_pcd", "hand_pcd"}:
                continue
            else:
                layer = nn.Linear(dim, self.embed_dim)
                nn.init.xavier_uniform_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
            self.embed_layers[term] = layer

        self.layer_norm = nn.LayerNorm(self.embed_dim) if layer_norm else nn.Identity()

    @property
    def obs_dim(self) -> int:
        return self.embed_dim

    def get_extra_state(self) -> dict:
        """Persist semantics as well as shapes; equal shapes need not mean equal inputs."""
        return {
            "scene_cloud": self.scene_cloud_contract,
            "selected_terms": self.selected_terms,
            "full_terms": self.full_obs_terms,
            "full_dims": self.full_obs_term_dims,
        }

    def set_extra_state(self, state: dict) -> None:
        expected = self.get_extra_state()
        if state != expected:
            raise ValueError(
                f"Observation checkpoint contract mismatch: expected {expected}, received {state}."
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        key = prefix + "_extra_state"
        if key not in state_dict and self.scene_cloud_contract is None:
            # Metadata-free historical checkpoints are valid only in explicit legacy mode.
            state_dict[key] = self.get_extra_state()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _split_raw_terms(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        if obs.shape[-1] != self.true_obs_dim:
            raise ValueError(
                f"Obs shape mismatch: expected {self.true_obs_dim}, got {obs.shape[-1]}."
            )
        chunks = torch.split(obs, self.full_obs_term_dims, dim=-1)
        return dict(zip(self.full_obs_terms, chunks))

    def command_timesteps(self, obs: torch.Tensor) -> torch.Tensor:
        term_to_chunk = self._split_raw_terms(obs)
        if "command" not in term_to_chunk:
            raise ValueError(
                "ObservationProcessor requires a 'command' observation term to "
                "extract command timesteps."
            )
        command = term_to_chunk["command"]
        if command.shape[-1] != 1:
            raise ValueError(
                "Expected command timestep observations to have dim 1, got "
                f"{command.shape[-1]}."
            )
        return command.squeeze(-1).long()

    def forward_with_command_timesteps(
        self, obs: torch.Tensor, update_stats: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(obs, update_stats=update_stats), self.command_timesteps(obs)

    def forward(self, obs: torch.Tensor, update_stats: bool = False) -> torch.Tensor:
        del update_stats
        if obs.shape[-1] != self.true_obs_dim:
            raise ValueError(
                f"Obs shape mismatch: expected {self.true_obs_dim}, got {obs.shape[-1]}."
            )

        term_to_chunk = self._split_raw_terms(obs)
        embeds = []

        if self.shared_pcd_terms:
            if self.pcd_processor is None:
                raise ValueError("Expected pcd_processor to be initialized.")
            hand_pcd = (
                term_to_chunk["hand_pcd"]
                if "hand_pcd" in self.shared_pcd_terms
                else None
            )
            embeds.append(
                self.pcd_processor(
                    object_pcd=term_to_chunk["object_pcd"],
                    hand_pcd=hand_pcd,
                )
            )

        for term in self.selected_terms:
            if term in {"object_pcd", "hand_pcd"}:
                continue
            embeds.append(self.embed_layers[term](term_to_chunk[term]))

        embed = torch.stack(embeds, dim=0).sum(dim=0)
        return self.layer_norm(embed)


class TrajectoryWeightedPolicyNetwork(PolicyNetwork):
    """Predict per-timestep actions, then blend by proximity and active object."""

    def __init__(
        self,
        *,
        env: Env,
        obs_dim: int,
        action_dim: int,
        sigma: float,
        embed_dim: int = 1024,
        hand_synergy_dim: int = 4,
        history_len: int = 0,
        device: str = "cuda",
        last_activation=None,
        action_residuals: bool = False,
    ):
        from dexlink.src.rl.multi_object.env import MultiObjectEnv

        if not isinstance(env, MultiObjectEnv):
            raise ValueError(
                "Trajectory-weighted policies require MultiObjectEnv or a derived environment; "
                f"received {type(env).__name__}."
            )
        if not env.cfg.obs_cfg.task_command or not env.cfg.obs_cfg.use_command_timestep:
            raise ValueError(
                "trajectory_weighted_v2 policy requires command timestep observations."
            )
        if sigma <= 0.0:
            raise ValueError(
                f"Expected trajectory_weighted_policy_sigma > 0, got {sigma}."
            )
        timestep_dropout = float(
            env.cfg.action_cfg.trajectory_weighted_policy_timestep_dropout
        )
        if not 0.0 <= timestep_dropout < 1.0:
            raise ValueError(
                "Expected trajectory_weighted_policy_timestep_dropout to be in "
                f"[0, 1), got {timestep_dropout}."
            )
        num_timesteps = int(env.task_manager.total_steps)
        final_action_dim = int(action_dim)
        super().__init__(
            obs_dim=obs_dim,
            action_dim=num_timesteps * final_action_dim,
            embed_dim=embed_dim,
            hand_synergy_dim=hand_synergy_dim,
            history_len=history_len,
            device=device,
            last_activation=last_activation,
            action_residuals=action_residuals,
        )
        self.env = env
        self.num_timesteps = num_timesteps
        self.final_action_dim = final_action_dim
        self.sigma = float(sigma)
        self.timestep_dropout = timestep_dropout
        init_log_std = -1.0 if action_residuals else 0.5
        self.log_std = nn.Parameter(torch.ones(self.final_action_dim) * init_log_std)
        self.register_buffer(
            "trajectory_timesteps",
            torch.arange(self.num_timesteps, dtype=torch.float32),
        )
        active_object_index = torch.as_tensor(
            env.task_manager.scene_active_object_index,
            dtype=torch.long,
        )
        if tuple(active_object_index.shape) != (self.num_timesteps,):
            raise ValueError(
                "Expected scene_active_object_index shape "
                f"{(self.num_timesteps,)}, got {tuple(active_object_index.shape)}."
            )
        self.register_buffer("active_object_index", active_object_index)

    def _weights(
        self, command_timesteps: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        if command_timesteps.ndim != 1:
            raise ValueError(
                f"Expected command_timesteps shape (B,), got {tuple(command_timesteps.shape)}."
            )
        if torch.any(command_timesteps < 0) or torch.any(
            command_timesteps >= self.num_timesteps
        ):
            raise ValueError(
                "Expected command timesteps in "
                f"[0, {self.num_timesteps - 1}], got {command_timesteps}."
            )
        timesteps = self.trajectory_timesteps.to(dtype=dtype)
        command_as_float = command_timesteps.to(dtype=dtype)[:, None]
        gaussian = torch.exp(
            -0.5 * ((timesteps[None, :] - command_as_float) / self.sigma) ** 2
        )
        current_active_object = self.active_object_index[command_timesteps]
        same_object = (
            self.active_object_index[None, :] == current_active_object[:, None]
        )
        masked = gaussian * same_object.to(dtype=dtype)
        if self.training and self.timestep_dropout > 0.0:
            keep = torch.rand_like(masked) >= self.timestep_dropout
            keep.scatter_(1, command_timesteps[:, None], True)
            masked = masked * keep.to(dtype=dtype)
        denom = masked.sum(dim=-1, keepdim=True)
        if torch.any(denom <= 0):
            raise ValueError(
                "Expected at least one same-active-object timestep per row."
            )
        return masked / denom

    def forward(
        self,
        obs,
        command_timesteps: torch.Tensor,
        history_obs=None,
        history_dones=None,
    ):
        policy_input = obs
        if self.history_encoder is not None:
            history_embed = self.history_encoder(history_obs, history_dones)
            if history_embed is None:
                history_embed = torch.zeros(
                    obs.shape[0],
                    self.embed_dim,
                    device=obs.device,
                    dtype=obs.dtype,
                )
            policy_input = torch.cat([obs, history_embed], dim=-1)

        candidate_means = self.net(policy_input).reshape(
            obs.shape[0], self.num_timesteps, self.final_action_dim
        )
        weights = self._weights(command_timesteps, dtype=obs.dtype)
        mean = torch.sum(candidate_means * weights[..., None], dim=1)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def act(self, obs, command_timesteps, history_obs=None, history_dones=None):
        mean, std = self.forward(
            obs,
            command_timesteps=command_timesteps,
            history_obs=history_obs,
            history_dones=history_dones,
        )
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def act_greedy(self, obs, command_timesteps, history_obs=None, history_dones=None):
        mean, std = self.forward(
            obs,
            command_timesteps=command_timesteps,
            history_obs=history_obs,
            history_dones=history_dones,
        )
        dist = torch.distributions.Normal(mean, std)
        return mean, dist.log_prob(mean).sum(dim=-1)

    def get_log_prob(
        self,
        obs,
        action,
        command_timesteps,
        history_obs=None,
        history_dones=None,
    ):
        mean, std = self.forward(
            obs,
            command_timesteps=command_timesteps,
            history_obs=history_obs,
            history_dones=history_dones,
        )
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1), dist.entropy().sum(dim=-1)


class TrajectoryWeightedResidualTargetPolicyNetwork(TrajectoryWeightedPolicyNetwork):
    """Learn residual trajectories, but expose blended absolute targets to the env."""

    def __init__(self, **kwargs):
        env = kwargs["env"]
        if env.cfg.action_cfg.action_mode != "absolute_target":
            raise ValueError(
                "trajectory_weighted_residual_target_v3 requires action_mode='absolute_target'."
            )
        self.reference_mode = str(env.cfg.action_cfg.residual_target_reference_mode)
        if self.reference_mode not in {
            "retargeted",
            "initial_frame",
            "retargeted_wrist_initial_fingers",
        }:
            raise ValueError(
                "Expected action_cfg.residual_target_reference_mode to be "
                "'retargeted', 'initial_frame', or "
                f"'retargeted_wrist_initial_fingers', got {self.reference_mode!r}."
            )
        self.target_action_dim = int(kwargs.pop("action_dim"))
        residual_action_dim = sum(
            spec.action_dim for spec in env.action_manager.hand_specs.values()
        )
        super().__init__(action_dim=residual_action_dim, **kwargs)
        self.residual_action_dim = residual_action_dim
        expected_target_dim = sum(
            7 + len(spec.hand_target_indices)
            for spec in env.action_manager.hand_specs.values()
        )
        if self.target_action_dim != expected_target_dim:
            raise ValueError(
                "Absolute-target action dimension mismatch: expected "
                f"{expected_target_dim}, got {self.target_action_dim}."
            )
        for side, spec in env.action_manager.hand_specs.items():
            if len(spec.hand_dof_indices) != len(spec.hand_target_indices):
                raise ValueError(
                    f"Residual-target policy requires matching hand residual and target dims for {side!r}."
                )

    def _reference_trajectory(
        self, side: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        manager = self.env.reference_trajectory_manager
        spec = self.env.action_manager.hand_specs[side]
        jvals = manager.hand_joint_positions(side).to(
            device=self.trajectory_timesteps.device
        )
        wrist_pos = manager.hand_wrist_pos(side).to(
            device=self.trajectory_timesteps.device
        )
        wrist_quat = manager.hand_wrist_quat(side).to(
            device=self.trajectory_timesteps.device
        )
        finger_indices = torch.as_tensor(
            spec.hand_target_indices, dtype=torch.long, device=jvals.device
        )
        finger_jvals = jvals.index_select(1, finger_indices)
        if self.reference_mode == "initial_frame":
            wrist_pos = wrist_pos[0:1].expand(self.num_timesteps, -1)
            wrist_quat = wrist_quat[0:1].expand(self.num_timesteps, -1)
            finger_jvals = finger_jvals[0:1].expand(self.num_timesteps, -1)
        elif self.reference_mode == "retargeted_wrist_initial_fingers":
            finger_jvals = finger_jvals[0:1].expand(self.num_timesteps, -1)
        return wrist_pos, wrist_quat, finger_jvals

    def _candidate_residual_means(self, obs, history_obs, history_dones):
        policy_input = obs
        if self.history_encoder is not None:
            history_embed = self.history_encoder(history_obs, history_dones)
            if history_embed is None:
                history_embed = torch.zeros(
                    obs.shape[0], self.embed_dim, device=obs.device, dtype=obs.dtype
                )
            policy_input = torch.cat([obs, history_embed], dim=-1)
        return self.net(policy_input).reshape(
            obs.shape[0], self.num_timesteps, self.residual_action_dim
        )

    def _effective_residual_mean(
        self, obs, command_timesteps, history_obs, history_dones
    ):
        candidates = self._candidate_residual_means(obs, history_obs, history_dones)
        weights = self._weights(command_timesteps, dtype=obs.dtype)
        return torch.sum(candidates * weights[..., None], dim=1), weights

    def _targets_from_residual(
        self,
        residual: torch.Tensor,
        command_timesteps: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        targets = []
        residual_offset = 0
        for side, spec in self.env.action_manager.hand_specs.items():
            side_dim = spec.action_dim
            side_residual = residual[:, residual_offset : residual_offset + side_dim]
            residual_offset += side_dim
            wrist_pos_all, wrist_quat_all, finger_all = self._reference_trajectory(side)
            wrist_pos_all = wrist_pos_all.to(
                device=residual.device, dtype=residual.dtype
            )
            wrist_quat_all = wrist_quat_all.to(
                device=residual.device, dtype=residual.dtype
            )
            finger_all = finger_all.to(device=residual.device, dtype=residual.dtype)
            base_pos = torch.sum(wrist_pos_all[None, :, :] * weights[..., None], dim=1)
            base_finger = torch.sum(finger_all[None, :, :] * weights[..., None], dim=1)
            pos = base_pos + side_residual[:, :3] * float(
                self.env.cfg.action_cfg.arm_translation_delta_scale
            )
            rotvec = side_residual[:, 3:6] * float(
                self.env.cfg.action_cfg.arm_rotation_delta_scale
            )
            current_ref_quat = wrist_quat_all.index_select(0, command_timesteps)
            quat = quat_mul(current_ref_quat, quat_from_rotvec(rotvec))
            finger_scale = float(
                self.env.cfg.action_cfg.hand_joint_delta_scale
            ) * spec.hand_delta_scale_factors.to(
                device=residual.device, dtype=residual.dtype
            )
            finger = base_finger + side_residual[:, 6:] * finger_scale.unsqueeze(0)
            targets.append(torch.cat([pos, quat, finger], dim=-1))
        return torch.cat(targets, dim=-1)

    def _residual_from_targets(
        self,
        targets: torch.Tensor,
        command_timesteps: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        residuals = []
        target_offset = 0
        for side, spec in self.env.action_manager.hand_specs.items():
            target_dim = 7 + len(spec.hand_target_indices)
            side_target = targets[:, target_offset : target_offset + target_dim]
            target_offset += target_dim
            wrist_pos_all, wrist_quat_all, finger_all = self._reference_trajectory(side)
            wrist_pos_all = wrist_pos_all.to(device=targets.device, dtype=targets.dtype)
            wrist_quat_all = wrist_quat_all.to(
                device=targets.device, dtype=targets.dtype
            )
            finger_all = finger_all.to(device=targets.device, dtype=targets.dtype)
            base_pos = torch.sum(wrist_pos_all[None, :, :] * weights[..., None], dim=1)
            base_finger = torch.sum(finger_all[None, :, :] * weights[..., None], dim=1)
            pos_residual = (side_target[:, :3] - base_pos) / float(
                self.env.cfg.action_cfg.arm_translation_delta_scale
            )
            current_ref_quat = wrist_quat_all.index_select(0, command_timesteps)
            quat_delta = quat_mul(quat_conjugate(current_ref_quat), side_target[:, 3:7])
            rot_residual = axis_angle_from_quat(quat_delta) / float(
                self.env.cfg.action_cfg.arm_rotation_delta_scale
            )
            finger_scale = float(
                self.env.cfg.action_cfg.hand_joint_delta_scale
            ) * spec.hand_delta_scale_factors.to(
                device=targets.device, dtype=targets.dtype
            )
            finger_residual = (
                side_target[:, 7:] - base_finger
            ) / finger_scale.unsqueeze(0)
            residuals.append(
                torch.cat([pos_residual, rot_residual, finger_residual], dim=-1)
            )
        return torch.cat(residuals, dim=-1)

    def forward(self, obs, command_timesteps, history_obs=None, history_dones=None):
        residual_mean, weights = self._effective_residual_mean(
            obs, command_timesteps, history_obs, history_dones
        )
        std = torch.exp(self.log_std).expand_as(residual_mean)
        return residual_mean, std, weights

    def act(self, obs, command_timesteps, history_obs=None, history_dones=None):
        mean, std, weights = self.forward(
            obs, command_timesteps, history_obs, history_dones
        )
        dist = torch.distributions.Normal(mean, std)
        residual = dist.rsample()
        return self._targets_from_residual(
            residual, command_timesteps, weights
        ), dist.log_prob(residual).sum(dim=-1)

    def act_greedy(self, obs, command_timesteps, history_obs=None, history_dones=None):
        mean, std, weights = self.forward(
            obs, command_timesteps, history_obs, history_dones
        )
        dist = torch.distributions.Normal(mean, std)
        return self._targets_from_residual(
            mean, command_timesteps, weights
        ), dist.log_prob(mean).sum(dim=-1)

    def get_log_prob(
        self, obs, action, command_timesteps, history_obs=None, history_dones=None
    ):
        mean, std, weights = self.forward(
            obs, command_timesteps, history_obs, history_dones
        )
        residual = self._residual_from_targets(action, command_timesteps, weights)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(residual).sum(dim=-1), dist.entropy().sum(dim=-1)


class TrajectoryMeanResidualTargetPolicyNetwork(
    TrajectoryWeightedResidualTargetPolicyNetwork
):
    """Predict per-timestep residuals, average them, and apply to current reference."""

    def _effective_residual_mean(
        self, obs, command_timesteps, history_obs, history_dones
    ):
        candidates = self._candidate_residual_means(obs, history_obs, history_dones)
        weights = torch.full(
            (obs.shape[0], self.num_timesteps),
            1.0 / float(self.num_timesteps),
            dtype=obs.dtype,
            device=obs.device,
        )
        return candidates.mean(dim=1), weights

    def _current_reference_base(
        self,
        side: str,
        command_timesteps: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wrist_pos_all, wrist_quat_all, finger_all = self._reference_trajectory(side)
        wrist_pos_all = wrist_pos_all.to(device=device, dtype=dtype)
        wrist_quat_all = wrist_quat_all.to(device=device, dtype=dtype)
        finger_all = finger_all.to(device=device, dtype=dtype)
        return (
            wrist_pos_all.index_select(0, command_timesteps),
            wrist_quat_all.index_select(0, command_timesteps),
            finger_all.index_select(0, command_timesteps),
        )

    def _targets_from_residual(
        self,
        residual: torch.Tensor,
        command_timesteps: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        del weights
        targets = []
        residual_offset = 0
        for side, spec in self.env.action_manager.hand_specs.items():
            side_dim = spec.action_dim
            side_residual = residual[:, residual_offset : residual_offset + side_dim]
            residual_offset += side_dim
            base_pos, base_quat, base_finger = self._current_reference_base(
                side,
                command_timesteps,
                dtype=residual.dtype,
                device=residual.device,
            )
            pos = base_pos + side_residual[:, :3] * float(
                self.env.cfg.action_cfg.arm_translation_delta_scale
            )
            rotvec = side_residual[:, 3:6] * float(
                self.env.cfg.action_cfg.arm_rotation_delta_scale
            )
            quat = quat_mul(base_quat, quat_from_rotvec(rotvec))
            finger_scale = float(
                self.env.cfg.action_cfg.hand_joint_delta_scale
            ) * spec.hand_delta_scale_factors.to(
                device=residual.device, dtype=residual.dtype
            )
            finger = base_finger + side_residual[:, 6:] * finger_scale.unsqueeze(0)
            targets.append(torch.cat([pos, quat, finger], dim=-1))
        return torch.cat(targets, dim=-1)

    def _residual_from_targets(
        self,
        targets: torch.Tensor,
        command_timesteps: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        del weights
        residuals = []
        target_offset = 0
        for side, spec in self.env.action_manager.hand_specs.items():
            target_dim = 7 + len(spec.hand_target_indices)
            side_target = targets[:, target_offset : target_offset + target_dim]
            target_offset += target_dim
            base_pos, base_quat, base_finger = self._current_reference_base(
                side,
                command_timesteps,
                dtype=targets.dtype,
                device=targets.device,
            )
            pos_residual = (side_target[:, :3] - base_pos) / float(
                self.env.cfg.action_cfg.arm_translation_delta_scale
            )
            quat_delta = quat_mul(quat_conjugate(base_quat), side_target[:, 3:7])
            rot_residual = axis_angle_from_quat(quat_delta) / float(
                self.env.cfg.action_cfg.arm_rotation_delta_scale
            )
            finger_scale = float(
                self.env.cfg.action_cfg.hand_joint_delta_scale
            ) * spec.hand_delta_scale_factors.to(
                device=targets.device, dtype=targets.dtype
            )
            finger_residual = (
                side_target[:, 7:] - base_finger
            ) / finger_scale.unsqueeze(0)
            residuals.append(
                torch.cat([pos_residual, rot_residual, finger_residual], dim=-1)
            )
        return torch.cat(residuals, dim=-1)


class FixedResidualTargetPolicyNetwork(TrajectoryWeightedResidualTargetPolicyNetwork):
    """Predict one reference-centered residual with a fixed-size policy MLP."""

    def __init__(self, **kwargs):
        obs_dim = int(kwargs["obs_dim"])
        embed_dim = int(kwargs["embed_dim"])
        hand_synergy_dim = int(kwargs["hand_synergy_dim"])
        history_len = int(kwargs["history_len"])
        device = str(kwargs["device"])
        last_activation = kwargs["last_activation"]
        action_residuals = bool(kwargs["action_residuals"])
        super().__init__(**kwargs)

        fixed_policy = PolicyNetwork(
            obs_dim=obs_dim,
            action_dim=self.residual_action_dim,
            embed_dim=embed_dim,
            hand_synergy_dim=hand_synergy_dim,
            history_len=history_len,
            device=device,
            last_activation=last_activation,
            action_residuals=action_residuals,
        )
        self.net = fixed_policy.net
        self.history_encoder = fixed_policy.history_encoder
        self.log_std = fixed_policy.log_std

    def _effective_residual_mean(
        self, obs, command_timesteps, history_obs, history_dones
    ):
        policy_input = obs
        if self.history_encoder is not None:
            history_embed = self.history_encoder(history_obs, history_dones)
            if history_embed is None:
                history_embed = torch.zeros(
                    obs.shape[0],
                    self.embed_dim,
                    device=obs.device,
                    dtype=obs.dtype,
                )
            policy_input = torch.cat([obs, history_embed], dim=-1)
        residual_mean = self.net(policy_input)
        weights = self._weights(command_timesteps, dtype=obs.dtype)
        return residual_mean, weights


class PolicyModel(nn.Module):
    """Policy plus its observation processing and optional auxiliary decoder."""

    def __init__(
        self,
        *,
        env: Env,
        selected_terms: tuple[str, ...],
        obs_embed_dim: int,
        obs_layer_norm: bool,
        pcd_hidden_dims: tuple[int, ...],
        pcd_pool: str,
        action_dim: int,
        policy_embed_dim: int,
        hand_synergy_dim: int,
        history_len: int,
        variant: str,
        aux_output_dim: int = 0,
        aux_hidden_dims: tuple[int, ...] = (256,),
    ):
        super().__init__()
        validate_scene_cloud_actor_inputs(env.cfg.obs_cfg, selected_terms)
        self.variant = variant
        self.history_len = history_len
        self.obs_processor = ObservationProcessor(
            env,
            embed_dim=obs_embed_dim,
            layer_norm=obs_layer_norm,
            pcd_hidden_dims=pcd_hidden_dims,
            pcd_pool=pcd_pool,
            selected_terms=selected_terms,
        )
        if variant == "default":
            self.net = PolicyNetwork(
                self.obs_processor.obs_dim,
                action_dim,
                embed_dim=policy_embed_dim,
                hand_synergy_dim=hand_synergy_dim,
                history_len=history_len,
                device=env.cfg.device,
                last_activation=None,
                action_residuals=True,
            )
        elif variant == "trajectory_weighted_v2":
            self.net = TrajectoryWeightedPolicyNetwork(
                env=env,
                obs_dim=self.obs_processor.obs_dim,
                action_dim=action_dim,
                sigma=env.cfg.action_cfg.trajectory_weighted_policy_sigma,
                embed_dim=policy_embed_dim,
                hand_synergy_dim=hand_synergy_dim,
                history_len=history_len,
                device=env.cfg.device,
                last_activation=None,
                action_residuals=True,
            )
        elif variant in {
            "trajectory_weighted_residual_target_v3",
            "trajectory_mean_residual_target_v3",
            "fixed_residual_target_v1",
        }:
            network_cls = {
                "trajectory_weighted_residual_target_v3": (
                    TrajectoryWeightedResidualTargetPolicyNetwork
                ),
                "trajectory_mean_residual_target_v3": (
                    TrajectoryMeanResidualTargetPolicyNetwork
                ),
                "fixed_residual_target_v1": FixedResidualTargetPolicyNetwork,
            }[variant]
            self.net = network_cls(
                env=env,
                obs_dim=self.obs_processor.obs_dim,
                action_dim=action_dim,
                sigma=env.cfg.action_cfg.trajectory_weighted_policy_sigma,
                embed_dim=policy_embed_dim,
                hand_synergy_dim=hand_synergy_dim,
                history_len=history_len,
                device=env.cfg.device,
                last_activation=None,
                action_residuals=True,
            )
        else:
            raise ValueError(
                "Expected action_cfg.policy_variant to be one of "
                "{'default', 'trajectory_weighted_v2', 'trajectory_weighted_residual_target_v3', "
                "'trajectory_mean_residual_target_v3', 'fixed_residual_target_v1'}, "
                f"got {variant!r}."
            )
        self.aux_decoder = (
            AuxiliaryDecoder(
                input_dim=self.obs_processor.obs_dim,
                output_dim=aux_output_dim,
                hidden_dims=aux_hidden_dims,
            )
            if aux_output_dim > 0
            else None
        )

    @property
    def obs_dim(self) -> int:
        return self.obs_processor.obs_dim

    @property
    def obs_save_dim(self) -> int:
        return self.obs_processor.obs_save_dim

    @property
    def log_std(self) -> torch.Tensor:
        return self.net.log_std

    def _process_history_obs(
        self, history_obs: torch.Tensor | None
    ) -> torch.Tensor | None:
        if history_obs is None:
            return None
        if history_obs.shape[0] == 0:
            return torch.zeros(
                history_obs.shape[1],
                0,
                self.obs_dim,
                device=history_obs.device,
            )
        seq_len, batch_size = history_obs.shape[:2]
        processed = self.obs_processor(
            history_obs.reshape(seq_len * batch_size, -1), update_stats=False
        )
        return processed.view(seq_len, batch_size, self.obs_dim).transpose(0, 1)

    def _encode(
        self, obs: torch.Tensor, update_stats: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.variant in {
            "trajectory_weighted_v2",
            "trajectory_weighted_residual_target_v3",
            "trajectory_mean_residual_target_v3",
            "fixed_residual_target_v1",
        }:
            processed, command_timesteps = (
                self.obs_processor.forward_with_command_timesteps(
                    obs, update_stats=update_stats
                )
            )
            return processed, command_timesteps
        return self.obs_processor(obs, update_stats=update_stats), None

    def act(
        self, obs, history_obs=None, history_dones=None, update_stats: bool = False
    ):
        processed, command_timesteps = self._encode(obs, update_stats=update_stats)
        processed_history = self._process_history_obs(history_obs)
        if command_timesteps is None:
            return self.net.act(
                processed,
                history_obs=processed_history,
                history_dones=history_dones,
            )
        return self.net.act(
            processed,
            command_timesteps=command_timesteps,
            history_obs=processed_history,
            history_dones=history_dones,
        )

    def act_greedy(self, obs, history_obs=None, history_dones=None):
        processed, command_timesteps = self._encode(obs, update_stats=False)
        processed_history = self._process_history_obs(history_obs)
        if command_timesteps is None:
            return self.net.act_greedy(
                processed,
                history_obs=processed_history,
                history_dones=history_dones,
            )
        return self.net.act_greedy(
            processed,
            command_timesteps=command_timesteps,
            history_obs=processed_history,
            history_dones=history_dones,
        )

    def get_log_prob(self, obs, action, history_obs=None, history_dones=None):
        processed, command_timesteps = self._encode(obs, update_stats=False)
        processed_history = self._process_history_obs(history_obs)
        if command_timesteps is None:
            return self.net.get_log_prob(
                processed,
                action,
                history_obs=processed_history,
                history_dones=history_dones,
            )
        return self.net.get_log_prob(
            processed,
            action,
            command_timesteps=command_timesteps,
            history_obs=processed_history,
            history_dones=history_dones,
        )

    def predict_aux(self, obs: torch.Tensor) -> torch.Tensor:
        if self.aux_decoder is None:
            raise ValueError("PolicyModel has no auxiliary decoder.")
        processed, _ = self._encode(obs, update_stats=False)
        return self.aux_decoder(processed)


class ValueModel(nn.Module):
    """Value network plus its observation processing."""

    def __init__(
        self,
        *,
        env: Env,
        selected_terms: tuple[str, ...],
        obs_embed_dim: int,
        obs_layer_norm: bool,
        pcd_hidden_dims: tuple[int, ...],
        pcd_pool: str,
        value_embed_dim: int,
        history_len: int,
    ):
        super().__init__()
        self.obs_processor = ObservationProcessor(
            env,
            embed_dim=obs_embed_dim,
            layer_norm=obs_layer_norm,
            pcd_hidden_dims=pcd_hidden_dims,
            pcd_pool=pcd_pool,
            selected_terms=selected_terms,
        )
        self.net = ValueNetwork(
            self.obs_processor.obs_dim,
            embed_dim=value_embed_dim,
            history_len=history_len,
            device=env.cfg.device,
        )

    @property
    def obs_dim(self) -> int:
        return self.obs_processor.obs_dim

    def _process_history_obs(
        self, history_obs: torch.Tensor | None
    ) -> torch.Tensor | None:
        if history_obs is None:
            return None
        if history_obs.shape[0] == 0:
            return torch.zeros(
                history_obs.shape[1],
                0,
                self.obs_dim,
                device=history_obs.device,
            )
        seq_len, batch_size = history_obs.shape[:2]
        processed = self.obs_processor(
            history_obs.reshape(seq_len * batch_size, -1), update_stats=False
        )
        return processed.view(seq_len, batch_size, self.obs_dim).transpose(0, 1)

    def forward(
        self, obs, history_obs=None, history_dones=None, update_stats: bool = False
    ):
        processed = self.obs_processor(obs, update_stats=update_stats)
        processed_history = self._process_history_obs(history_obs)
        return self.net(
            processed,
            history_obs=processed_history,
            history_dones=history_dones,
        )


__all__ = [
    "CommandsProcessor",
    "DiscObservationProcessor",
    "ObservationProcessor",
    "PointCloudProcessor",
    "PolicyModel",
    "PolicyNetwork",
    "TrajectoryWeightedPolicyNetwork",
    "TrajectoryWeightedResidualTargetPolicyNetwork",
    "ValueModel",
    "ValueNetwork",
]
