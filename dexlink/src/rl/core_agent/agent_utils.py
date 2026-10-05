from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from dexlink.utils.retargeting.math_utils import (
    matrix_from_quat,
    quat_from_matrix,
    rot6d_from_matrix,
)
from dexlink.utils.retargeting.torch_utils import FCN

if TYPE_CHECKING:
    from dexlink.src.rl.sim.env import Env


class CommandsProcessor(nn.Module):
    """Projects individual command terms to a shared embedding space."""

    def __init__(
        self,
        env: Env,
        embed_dim: int = 64,
        layer_norm: bool = True,
        embed_from_timestep: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = env.cfg.device
        self.embed_from_timestep = embed_from_timestep

        if self.embed_from_timestep:
            if not env.cfg.obs_cfg.use_command_timestep:
                raise ValueError(
                    "embed_from_timestep=True requires obs_cfg.use_command_timestep=True."
                )
            self.command_terms = ["command_timestep"]
            self.command_dims = [1]
        else:
            self.command_terms = list(env.commands_manager.obs_command_terms)
            if env.cfg.obs_cfg.use_command_timestep:
                self.command_dims = list(env.commands_manager.policy_command_dims)
            else:
                self.command_dims = list(env.commands_manager.obs_command_dims)

        self.embed_layers = nn.ModuleDict()
        for _term, _dim in zip(self.command_terms, self.command_dims):
            layer = nn.Linear(_dim, embed_dim)
            nn.init.xavier_uniform_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
            self.embed_layers[_term] = layer

        if layer_norm:
            self.layer_norm = nn.LayerNorm(self.embed_dim)
        else:
            self.layer_norm = nn.Identity()

    @property
    def out_dim(self) -> int:
        return self.embed_dim

    def forward(self, commands: torch.Tensor) -> torch.Tensor:
        assert commands.shape[-1] == sum(
            self.command_dims
        ), f"Command dim mismatch: expected {sum(self.command_dims)}, got {commands.shape[-1]}"
        splits = torch.split(commands, self.command_dims, dim=-1)
        embeds = []
        for term, chunk in zip(self.command_terms, splits):
            embeds.append(self.embed_layers[term](chunk))

        stacked = torch.stack(embeds, dim=0)
        embed = stacked.sum(dim=0)
        return self.layer_norm(embed)


class ObservationProcessor(nn.Module):
    """Handles command expansion, normalization, and term-wise embedding for policy/value nets."""

    def __init__(
        self,
        env: Env,
        embed_dim: int = 64,
        clip_range: float = 100.0,
        eps: float = 1e-8,
        alpha=0.5,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.env = env
        self.eps = eps
        self.embed_dim = embed_dim
        # how fast to change stats: low value means means and var changes slowly
        self.alpha = alpha
        obs_cfg = env.cfg.obs_cfg
        self.embed_commands_from_timestep = (
            obs_cfg.embed_commands_from_timestep and obs_cfg.use_command_timestep
        )
        # dim for saving obs in buffer
        self.obs_save_dim = env.observation_space.shape[1]
        self.add_commands = obs_cfg.use_command_timestep and obs_cfg.task_command

        if self.add_commands:
            self.command_dim = env.commands_manager.policy_command_dim
            obs_dim = self.obs_save_dim - 1 + self.command_dim
            obs_mgr = env.observation_manager
            command_idx = obs_mgr.obs_terms.index("command")
            self.pre_obs_idx = sum(obs_mgr.term_dims[:command_idx])
            if env.observation_space.shape[1] == (self.pre_obs_idx + 1):
                self.post_obs_dim = 0
            else:
                self.post_obs_dim = sum(obs_mgr.term_dims[command_idx + 1 :])
            assert (
                obs_dim == self.pre_obs_idx + self.post_obs_dim + self.command_dim
            ), f"obs_dim mismatch: expected {obs_dim}, got {self.pre_obs_idx + self.post_obs_dim + self.command_dim}"
        else:
            self.pre_obs_idx = 0
            self.post_obs_dim = 0
            obs_dim = self.obs_save_dim
            if (
                "command" in env.observation_manager.obs_terms
            ) and obs_cfg.task_command:
                cmd_idx = env.observation_manager.obs_terms.index("command")
                self.command_dim = env.observation_manager.term_dims[cmd_idx]
            else:
                self.command_dim = 0

        self.true_obs_dim = obs_dim  # dim before embedding
        self.clip_range = clip_range
        device = env.cfg.device
        self.register_buffer("mean", torch.zeros(self.true_obs_dim, device=device))
        self.register_buffer("var", torch.ones(self.true_obs_dim, device=device))
        self.register_buffer("count", torch.tensor(0.0, device=device))

        # embedding setup
        self.obs_terms = deepcopy(env.observation_manager.obs_terms)
        self.obs_term_dims = deepcopy(env.observation_manager.term_dims)

        # fix the command dim to the actual command dim
        if "command" in self.obs_terms:
            cmd_idx = self.obs_terms.index("command")
            self.obs_term_dims[cmd_idx] = self.command_dim

        self.embed_layers = nn.ModuleDict()

        for term, dim in zip(self.obs_terms, self.obs_term_dims):
            if term == "command":
                layer = CommandsProcessor(
                    env,
                    embed_dim=self.embed_dim,
                    layer_norm=layer_norm,
                    embed_from_timestep=self.embed_commands_from_timestep,
                )
            else:
                layer = nn.Linear(dim, self.embed_dim)
                nn.init.xavier_uniform_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
            self.embed_layers[term] = layer

        if layer_norm:
            self.layer_norm = nn.LayerNorm(self.embed_dim)
        else:
            self.layer_norm = nn.Identity()

    def get_relative_obs(
        self, obs: torch.Tensor, command_timestep: torch.Tensor
    ) -> torch.Tensor:
        hand_rel = self.env.cfg.obs_cfg.hand_cfg.use_retgt_relative_obs
        obj_rel = self.env.cfg.obs_cfg.use_relative_obj_obs
        if not (hand_rel or obj_rel):
            return obs

        out = obs.clone()
        splits = list(torch.split(out, self.obs_term_dims, dim=-1))
        term_to_idx = {term: i for i, term in enumerate(self.obs_terms)}

        command_ts = command_timestep.to(device=obs.device, dtype=torch.long)

        if hand_rel:
            reset_mgr = self.env.reset_manager
            for side in ("left", "right"):
                if side not in term_to_idx:
                    continue
                hand = self.env.scene_manager.scene_entities[side]
                chunk = splits[term_to_idx[side]]
                named = hand.obs_named_view(chunk)

                frame_jvals = (
                    reset_mgr.frame_left_jvals
                    if side == "left"
                    else reset_mgr.frame_right_jvals
                ).to(device=obs.device, dtype=chunk.dtype)
                cmd_jpos = frame_jvals[command_ts, ...]
                if "jpos" in named:
                    named["jpos"] = named["jpos"] - cmd_jpos

                if "lpos" in named:
                    ref_motion = self.env.scene_manager.original_demo.retargeted_motion
                    if ref_motion is not None and f"{side}_link_pos" in ref_motion:
                        side_link_pos = ref_motion[f"{side}_link_pos"]
                        cmd_link_pos = torch.stack(
                            [
                                torch.as_tensor(
                                    side_link_pos[link_name],
                                    device=obs.device,
                                    dtype=chunk.dtype,
                                )
                                for link_name in hand.human_mapped_link_names
                            ],
                            dim=1,
                        )
                        cmd_link_pos = cmd_link_pos[command_ts, ...].reshape(
                            chunk.shape[0], 3 * len(hand.human_mapped_link_names)
                        )
                        named["lpos"] = named["lpos"] - cmd_link_pos

                splits[term_to_idx[side]] = torch.cat(
                    [named[t] for t in hand.obs_terms], dim=-1
                )

        if obj_rel and ("object" in term_to_idx):
            object_chunk = splits[term_to_idx["object"]]
            object_entity = self.env.scene_manager.scene_entities["object"]
            n_envs = object_chunk.shape[0]
            n_obj_dofs = len(object_entity.dof_map)
            rot_dim = 6 if self.env.cfg.obs_cfg.rot_format == "6d" else 4
            dtype = object_chunk.dtype
            device = object_chunk.device

            command_vals = self.env.commands_manager.query_commands_from_timesteps(
                command_ts
            )
            named_command = self.env.commands_manager.named_view(command_vals)

            gt_pos = named_command.get(
                "gt_pos", torch.zeros((n_envs, 3), device=device, dtype=dtype)
            ).to(device=device, dtype=dtype)

            gt_rot = named_command.get("gt_rot", None)
            if gt_rot is None:
                gt_rot = torch.zeros((n_envs, rot_dim), device=device, dtype=dtype)
            else:
                gt_rot = gt_rot.to(device=device, dtype=dtype)
                if gt_rot.shape[-1] != rot_dim:
                    if rot_dim == 6 and gt_rot.shape[-1] == 4:
                        gt_rot = rot6d_from_matrix(matrix_from_quat(gt_rot))
                    elif rot_dim == 4 and gt_rot.shape[-1] == 6:
                        x_axis = gt_rot[:, :3]
                        y_axis = gt_rot[:, 3:6]
                        z_axis = torch.cross(x_axis, y_axis, dim=-1)
                        mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)
                        gt_rot = quat_from_matrix(mat)
                    else:
                        raise ValueError(
                            f"Unsupported gt_rot dim {gt_rot.shape[-1]} for rot_dim {rot_dim}."
                        )

            gt_arti = named_command.get(
                "gt_arti", torch.zeros((n_envs, n_obj_dofs), device=device, dtype=dtype)
            ).to(device=device, dtype=dtype)

            splits[term_to_idx["object"]] = object_chunk - torch.cat(
                [gt_pos, gt_rot, gt_arti], dim=-1
            )

        return torch.cat(splits, dim=-1)

    @property
    def obs_dim(self):
        return self.embed_dim

    def _normalize_command_timestep(
        self,
        command_timestep: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        max_timestep = self.env.scene_manager.total_steps - 1
        if max_timestep <= 0:
            raise ValueError(
                f"Expected scene_manager.total_steps > 1, got {self.env.scene_manager.total_steps}."
            )
        return (
            command_timestep.to(device=device, dtype=dtype) / max_timestep
        ).unsqueeze(-1)

    def _preprocess(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.add_commands:
            raise AssertionError(
                "ObservationProcessor assumes use_command_timestep=True for preprocessing."
            )

        assert (
            obs.shape[-1] == self.obs_save_dim
        ), f"obs shape mismatch: expected {self.obs_save_dim}, got {obs.shape[-1]}"

        pre_obs = obs[..., : self.pre_obs_idx]
        command_timestep = obs[..., self.pre_obs_idx].long()
        command = self.env.commands_manager.query_policy_commands_from_timesteps(
            command_timestep
        )
        assert (
            command.shape[-1] == self.command_dim
        ), f"command shape mismatch: expected {self.command_dim}, got {command.shape[-1]}"

        if self.post_obs_dim > 0:
            post_obs = obs[..., self.pre_obs_idx + 1 :]
            new_obs = torch.cat([pre_obs, command, post_obs], dim=-1)
        else:
            new_obs = torch.cat([pre_obs, command], dim=-1)

        assert (
            new_obs.shape[-1] == self.true_obs_dim
        ), f"new_obs shape mismatch: expected {self.true_obs_dim}, got {new_obs.shape[-1]}"
        new_obs = self.get_relative_obs(new_obs, command_timestep=command_timestep)
        if self.embed_commands_from_timestep:
            normalized_command_timestep = self._normalize_command_timestep(
                command_timestep,
                dtype=new_obs.dtype,
                device=new_obs.device,
            )
        else:
            normalized_command_timestep = None
        return new_obs, normalized_command_timestep

    def update(self, proc: torch.Tensor) -> None:
        with torch.no_grad():
            batch_mean = torch.mean(proc, dim=0)
            batch_var = torch.var(proc, dim=0)
            batch_count = proc.shape[0]

            if self.count.item() == 0:
                self.mean.copy_(batch_mean)
                self.var.copy_(batch_var + self.eps)
                self.count.add_(batch_count)
                return

            total_count = self.count + batch_count
            combined_mean = (
                self.mean * self.count + batch_mean * batch_count
            ) / total_count

            delta = batch_mean - self.mean
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
            combined_var = M2 / total_count

            alpha = self.alpha
            smoothed_mean = (1 - alpha) * self.mean + alpha * combined_mean
            smoothed_var = (1 - alpha) * self.var + alpha * combined_var

            self.mean.copy_(smoothed_mean)
            self.var.copy_(torch.clamp(smoothed_var, min=self.eps))
            self.count.copy_(total_count)

    def _embed_terms(
        self,
        normalized_obs: torch.Tensor,
        normalized_command_timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Split normalized obs into terms, embed each, and sum."""
        assert (
            normalized_obs.shape[-1] == self.true_obs_dim
        ), f"Normalized obs shape mismatch: expected {self.true_obs_dim}, got {normalized_obs.shape[-1]}"
        splits = torch.split(normalized_obs, self.obs_term_dims, dim=-1)
        embeds = []

        for term, chunk in zip(self.obs_terms, splits):
            if term == "command" and self.embed_commands_from_timestep:
                if normalized_command_timestep is None:
                    raise ValueError(
                        "normalized_command_timestep is required when embedding commands from timestep."
                    )
                embeds.append(self.embed_layers[term](normalized_command_timestep))
            else:
                embeds.append(self.embed_layers[term](chunk))

        embed = torch.sum(torch.stack(embeds, dim=0), dim=0)
        return self.layer_norm(embed)

    def forward(self, obs: torch.Tensor, update_stats: bool = False) -> torch.Tensor:
        proc, normalized_command_timestep = self._preprocess(obs)
        if update_stats:
            self.update(proc)
        normalized_obs = (proc - self.mean) / torch.sqrt(self.var + self.eps)
        normalized_obs = torch.clamp(normalized_obs, -self.clip_range, self.clip_range)
        return self._embed_terms(
            normalized_obs,
            normalized_command_timestep=normalized_command_timestep,
        )


class DiscObservationProcessor(nn.Module):
    """Embed discriminator observations term-wise before feeding the discriminator."""

    def __init__(
        self,
        env: Env,
        embed_dim: int = 64,
        layer_norm: bool = True,
        mask_prob: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        if not (0.0 <= mask_prob < 1.0):
            raise ValueError(f"mask_prob must be in [0.0, 1.0), got {mask_prob}")
        self.mask_prob = mask_prob
        self.obs_terms = list(env.disc_observation_manager.obs_terms)
        self.term_dims = list(env.disc_observation_manager.term_dims)

        self.embed_layers = nn.ModuleDict()
        for term, dim in zip(self.obs_terms, self.term_dims):
            layer = nn.Linear(dim, embed_dim)
            nn.init.xavier_uniform_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
            self.embed_layers[term] = layer

        if layer_norm:
            self.layer_norm = nn.LayerNorm(self.embed_dim)
        else:
            self.layer_norm = nn.Identity()

    @property
    def out_dim(self) -> int:
        return self.embed_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        assert obs.shape[-1] == sum(
            self.term_dims
        ), f"Disc obs dim mismatch: expected {sum(self.term_dims)}, got {obs.shape[-1]}"
        if self.mask_prob > 0.0:
            keep_mask = torch.rand_like(obs) > self.mask_prob
            obs = obs * keep_mask.to(obs.dtype)
        splits = torch.split(obs, self.term_dims, dim=-1)
        embeds = [
            self.embed_layers[term](chunk)
            for term, chunk in zip(self.obs_terms, splits)
        ]
        embed = torch.stack(embeds, dim=0).sum(dim=0)
        return self.layer_norm(embed)


class ObservationHistoryEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

    def forward(
        self, history_obs: torch.Tensor | None, history_dones: torch.Tensor | None
    ) -> torch.Tensor | None:
        if history_obs is None:
            return None

        if history_obs.ndim != 3:
            raise ValueError(
                f"Expected history_obs to have shape (batch, history_len, obs_dim), got {history_obs.shape}."
            )

        if history_obs.shape[1] == 0:
            return torch.zeros(
                history_obs.shape[0],
                self.hidden_dim,
                device=history_obs.device,
                dtype=history_obs.dtype,
            )

        if history_dones is None:
            valid_mask = torch.ones_like(history_obs[..., :1])
        else:
            done_flags = history_dones.to(dtype=history_obs.dtype)
            valid_mask = (
                torch.cumsum(done_flags.flip(dims=(1,)), dim=1).flip(dims=(1,)) == 0
            ).to(dtype=history_obs.dtype)

        masked_history = history_obs * valid_mask
        _, hidden = self.gru(masked_history)
        return self.layer_norm(hidden[-1])


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        embed_dim: int = 1024,
        hand_synergy_dim: int = 4,
        history_len: int = 0,
        device="cuda",
        last_activation=nn.Tanh,
        action_residuals: bool = False,
    ):
        super().__init__()
        self.device = device
        self.action_residuals = action_residuals
        self.history_len = history_len

        self.embed_dim = embed_dim
        self.hand_synergy_dim = hand_synergy_dim
        self.history_encoder = (
            ObservationHistoryEncoder(obs_dim=obs_dim, hidden_dim=embed_dim)
            if history_len > 0
            else None
        )
        self.net = FCN(
            input_dim=obs_dim + (embed_dim if history_len > 0 else 0),
            output_dim=action_dim,
            hidden_sizes=[self.embed_dim, self.hand_synergy_dim, self.embed_dim],
            last_activation=last_activation,
            # last_activation=None,
        )

        init_log_std = 0.5
        gain = 0.2
        if self.action_residuals:
            init_log_std = -1.0
            gain = 0.01

        # save to buffer the obs normalization stats
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

        # set gain to smaller value to avoid large initial actions
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, obs, history_obs=None, history_dones=None):
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

        mean = self.net(policy_input)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def act(self, obs, history_obs=None, history_dones=None):
        mean, std = self.forward(
            obs, history_obs=history_obs, history_dones=history_dones
        )
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def act_greedy(self, obs, history_obs=None, history_dones=None):
        mean, std = self.forward(
            obs, history_obs=history_obs, history_dones=history_dones
        )
        dist = torch.distributions.Normal(mean, std)
        action = mean
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def get_log_prob(self, obs, action, history_obs=None, history_dones=None):
        mean, std = self.forward(
            obs, history_obs=history_obs, history_dones=history_dones
        )
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        # print(f"Log prob shape: {log_prob.shape}, Entropy shape: {entropy.shape}")
        return log_prob, entropy


class ValueNetwork(nn.Module):
    def __init__(
        self, obs_dim, embed_dim: int = 1024, history_len: int = 0, device="cuda"
    ):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.history_len = history_len
        self.history_encoder = (
            ObservationHistoryEncoder(obs_dim=obs_dim, hidden_dim=embed_dim)
            if history_len > 0
            else None
        )
        self.net = FCN(
            input_dim=obs_dim + (embed_dim if history_len > 0 else 0),
            output_dim=1,
            hidden_sizes=[embed_dim, embed_dim],
            last_activation=None,
        )

    def forward(self, obs, history_obs=None, history_dones=None):
        value_input = obs
        if self.history_encoder is not None:
            history_embed = self.history_encoder(history_obs, history_dones)
            if history_embed is None:
                history_embed = torch.zeros(
                    obs.shape[0],
                    self.embed_dim,
                    device=obs.device,
                    dtype=obs.dtype,
                )
            value_input = torch.cat([obs, history_embed], dim=-1)

        value = self.net(value_input)
        return value.squeeze(-1)
