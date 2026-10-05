"""Adversarial Discriminator for GAIL.

This module implements a simple MLP-based binary classifier D(s) where
the input s are discriminator-specific state features provided by the
environment's ObservationBuilder (not the agent observation).

Usage:
- Initialize with Env instance; we read expert data from env.reference_demo
  and rely on env.obs_builder to assemble discriminator inputs.
- Call add_fake_batch(...) after each rollout to append policy-generated
  samples into an in-memory buffer.
- Call train_step(...) periodically to update D on expert vs. fake data.
- Call reward(obs_dict) to compute per-env rewards: r = -log(1 - D(s)).

Notes:
- For now, we use only object motion features as inputs (pos, quat, jpos
  if present) concatenated into a flat tensor per env.
- Buffer is in-memory; can be extended to on-disk for large-scale runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dexlink.src.rl.core_agent.memory import Memory

if TYPE_CHECKING:
    from dexlink.src.rl.core_agent.wandb_logger import WandbLogger
    from dexlink.src.rl.config.base import DiscCfg

# def build_object_features(obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
#     """Extract and concatenate object-centric features from ObservationBuilder outputs.

#     Expected keys (if present):
#     - obj_pos: (n_envs, 3)
#     - obj_quat: (n_envs, 4)
#     - obj_jpos: (n_envs, J) optional articulated object DOFs

#     Returns tensor of shape (n_envs, feat_dim).
#     """
#     feats = []
#     for k in ("obj_pos", "obj_quat", "obj_jpos"):
#         if k in obs_dict and isinstance(obs_dict[k], torch.Tensor):
#             feats.append(obs_dict[k])
#     if len(feats) == 0:
#         raise ValueError("Discriminator: No object features found in obs_dict.")
#     return torch.cat(feats, dim=-1).detach()  # ensure features don't carry grad


class DiscNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
        dropout_p: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=dropout_p))
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expert_sampler: Callable,
        logger: WandbLogger,
        memory_buffer: Memory,
        disc_cfg: DiscCfg,
        # hidden_sizes: Tuple[int, int] = (256, 256),
        # lr: float = 3e-4,
        # batch_size: int = 256,
        # n_steps: int = 4,
        input_processor: nn.Module | None = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.buffer_capacity = memory_buffer.rollouts * memory_buffer.num_envs
        self.input_processor = input_processor
        self.raw_feat_dim = input_dim
        self.feat_dim = (
            input_processor.out_dim if input_processor is not None else input_dim
        )
        self.expert_sampler = expert_sampler
        self.fake_samples_buffer = memory_buffer
        self.logger = logger
        self.net = DiscNet(
            input_dim=self.feat_dim,
            hidden_sizes=disc_cfg.hidden_dims,
            device=device,
        )

        # opt_params = list(self.net.parameters())
        # if self.input_processor is not None:
        #     opt_params += list(self.input_processor.parameters())

        opt_params = self.parameters()
        self.optimizer = torch.optim.AdamW(
            opt_params, lr=disc_cfg.lr, weight_decay=disc_cfg.weight_decay
        )

        self.batch_size = disc_cfg.batch_size
        self.n_steps = (
            disc_cfg.n_steps
        )  # perform net updates 4 times per call to update_net()

        self.use_transitions = self.fake_samples_buffer.seq_len > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # process inputs if needed and forward through net
        proc_x = self._process_inputs(x)
        return self.net(proc_x)

    def _process_inputs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.input_processor is None:
            return obs
        return self.input_processor(obs)

    def disc_loss_fn(self, logits, labels, targets):
        # ensure matching shapes for stable MSE gradients
        logits = logits.squeeze(-1)
        targets = targets.view_as(logits)
        return F.mse_loss(logits, targets)

    def update_net(self, timestep: int, train_iters: int):

        n_experts = self.batch_size // 2
        n_fakes = self.batch_size - n_experts

        fake_sample_indices = self.fake_samples_buffer.sample_indices(
            batch_size=n_fakes
        )
        if len(fake_sample_indices) == 0:
            return

        # Use exactly n_steps batches for updates and reserve one final batch for eval.
        max_batches = min(len(fake_sample_indices), max(self.n_steps, 0) + 1)
        fake_sample_indices = fake_sample_indices[:max_batches]
        update_sample_indices = fake_sample_indices[:-1]
        eval_sample_indices = fake_sample_indices[-1]

        losses = []
        grad_norms = []
        grad_max = []
        logits_mean = []
        logits_std = []
        expert_logits_mean = []
        fake_logits_mean = []
        self.train()

        for _sample_indices in update_sample_indices:

            with torch.no_grad():
                fake_samples = self.fake_samples_buffer.query_indices(
                    _sample_indices, device=self.device
                )
                # fake_obs = self._process_inputs(
                #     fake_samples["disc_obs"]
                # )  # (batch_size, feat_dim)

                # expert_obs = self._process_inputs(
                #     self.expert_sampler(batch_size=n_fakes)
                # )
                fake_obs = fake_samples["disc_obs"]  # (batch_size, obs_dim)
                expert_obs = self.expert_sampler(batch_size=n_experts)

                disc_inputs = torch.cat([expert_obs, fake_obs], dim=0)
                labels = torch.ones(
                    disc_inputs.shape[0], device=self.device
                ).bool()  # placeholder
                labels[-n_fakes:] = False
                targets = torch.where(
                    labels,
                    torch.ones_like(labels.float()),
                    -torch.ones_like(labels.float()),
                )

            # update step
            self.optimizer.zero_grad()
            logits = self.forward(disc_inputs)
            loss = self.disc_loss_fn(logits, labels, targets)
            loss.backward()
            # also log avg grad norm
            grads = [p.grad.view(-1) for p in self.parameters() if p.grad is not None]
            flat_grads = torch.cat(grads)
            grad_norms.append(flat_grads.abs().mean().item())
            grad_max.append(flat_grads.abs().max().item())

            logits = logits.squeeze(-1)
            logits_mean.append(logits.mean().item())
            logits_std.append(logits.std().item())
            expert_logits = logits[labels]
            fake_logits = logits[~labels]
            expert_logits_mean.append(expert_logits.mean().item())
            fake_logits_mean.append(fake_logits.mean().item())

            self.optimizer.step()
            losses.append(loss.item())

        # eval step on reserved last batch
        with torch.no_grad():
            fake_samples = self.fake_samples_buffer.query_indices(
                eval_sample_indices, device=self.device
            )
            fake_obs = fake_samples["disc_obs"]
            expert_obs = self.expert_sampler(batch_size=n_experts)
            disc_inputs = torch.cat([expert_obs, fake_obs], dim=0)
            labels = torch.ones(disc_inputs.shape[0], device=self.device).bool()
            labels[-n_fakes:] = False
            targets = torch.where(
                labels,
                torch.ones_like(labels.float()),
                -torch.ones_like(labels.float()),
            )

            eval_logits = self.forward(disc_inputs).squeeze(-1)
            eval_loss = self.disc_loss_fn(eval_logits, labels, targets)
            eval_preds = eval_logits > 0.0
            eval_acc = (eval_preds == labels).float().mean().item()
            self.logger.log(
                {
                    "discriminator/accuracy": eval_acc,
                    "discriminator/eval_loss": eval_loss.item(),
                },
                timestep=timestep,
                train_iters=train_iters,
            )

        # logging
        self.logger.log(
            {
                "discriminator/train_loss": np.mean(losses) if len(losses) > 0 else 0.0,
                "discriminator/avg_grad_norm": (
                    np.mean(grad_norms) if len(grad_norms) > 0 else 0.0
                ),
                "discriminator/max_grad": (
                    np.max(grad_max) if len(grad_max) > 0 else 0.0
                ),
                "discriminator/logits_mean": (
                    np.mean(logits_mean) if len(logits_mean) > 0 else 0.0
                ),
                "discriminator/logits_std": (
                    np.mean(logits_std) if len(logits_std) > 0 else 0.0
                ),
                "discriminator/expert_logits_mean": (
                    np.mean(expert_logits_mean) if len(expert_logits_mean) > 0 else 0.0
                ),
                "discriminator/fake_logits_mean": (
                    np.mean(fake_logits_mean) if len(fake_logits_mean) > 0 else 0.0
                ),
                "discriminator/num_update_steps": len(update_sample_indices),
            },
            timestep=timestep,
            train_iters=train_iters,
        )

    def compute_rewards(self, disc_obs):
        """Return per-env GAIL reward: r = -log(1 - D(s)).

        D(s) = sigmoid(net(s)); clamp for numerical stability.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(disc_obs).squeeze(-1)
            disc_rew = 1 - 0.25 * torch.square(logits - 1)
            disc_rew = torch.clamp(disc_rew, min=0.0)

        return disc_rew
