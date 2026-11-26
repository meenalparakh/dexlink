"""Native full-method observation/backbone modules with the declared multi-task conditioning and local head."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.distributions import Normal

from dexlink.src.rl.core_agent.agent_utils import PolicyNetwork, ValueNetwork
from dexlink.src.rl.agent.agent_utils import ObservationProcessor
from dexlink.src.rl.config.base import EnvConfig, ExperimentConfig

from .config import GeometryPolicyConfig
from .full_method_config import ConditioningArchitecture, FULL_METHOD_PRESET
from .geometry_policy import (
    GeometryEncoder,
    GeometryPolicySchema,
    ProgramEncoder,
    REFERENCE_SCHEMA,
    PROGRAM_SCHEMA,
    GEOMETRY_SCHEMA,
)
from .policy import AdaptiveRMSNorm, OFFSETS


@dataclass(frozen=True)
class NativeObservationLayout:
    obs_terms: tuple[str, ...]
    term_dims: tuple[int, ...]

    def validate(self) -> None:
        if (
            not self.obs_terms
            or len(self.obs_terms) != len(self.term_dims)
            or len(set(self.obs_terms)) != len(self.obs_terms)
            or min(self.term_dims) < 1
        ):
            raise ValueError(
                "Expected unique native observation terms with matching positive dimensions."
            )


@dataclass(frozen=True)
class ObservationSpaceShape:
    shape: tuple[int, int]


@dataclass(frozen=True)
class ObservationConstructionContext:
    cfg: EnvConfig
    observation_manager: NativeObservationLayout
    observation_space: ObservationSpaceShape


@dataclass(frozen=True)
class FullMethodInputs:
    native_obs: torch.Tensor
    normalized_reference_time: torch.Tensor
    window: torch.Tensor
    window_mask: torch.Tensor
    geometry_points: torch.Tensor
    geometry_metadata: torch.Tensor
    geometry_index: torch.Tensor
    program_tokens: torch.Tensor
    program_mask: torch.Tensor
    program_index: torch.Tensor
    current_phase: torch.Tensor
    progress: torch.Tensor


class NativeTaskObservationProcessor(ObservationProcessor):
    """Keep native term embeddings/LayerNorm; normalize each task's clock by its own horizon."""

    def __init__(
        self,
        layout: NativeObservationLayout,
        method: ExperimentConfig,
        selected: tuple[str, ...],
    ) -> None:
        layout.validate()
        if (
            not method.env.obs_cfg.embed_commands_from_timestep
            or method.agent.history_len != 0
        ):
            raise ValueError(
                "This native full-method contract requires timestep commands and history_len=0."
            )
        # ObservationProcessor construction needs metadata, not a live simulator or reference length.
        context = ObservationConstructionContext(
            method.env, layout, ObservationSpaceShape((0, sum(layout.term_dims)))
        )
        super().__init__(
            context,
            embed_dim=method.agent.obs_proc_embed_dim,
            layer_norm=method.agent.obs_proc_layer_norm,
            pcd_hidden_dims=method.agent.obs_proc_pcd_hidden_dims,
            pcd_pool=method.agent.obs_proc_pcd_pool,
            selected_terms=selected,
        )
        if self.shared_pcd_terms:
            raise ValueError(
                "Expected the validated full-method vector-input preset; geometry points are a separate declared conditioning input."
            )

    def forward(self, obs: torch.Tensor, normalized_time: torch.Tensor) -> torch.Tensor:
        chunks = self._split_raw_terms(obs)
        if (
            normalized_time.shape != (len(obs),)
            or not torch.isfinite(normalized_time).all()
            or ((normalized_time < 0) | (normalized_time > 1)).any()
        ):
            raise ValueError(
                "Expected a finite normalized reference-time vector in[0,1]."
            )
        embeddings = []
        for term in self.selected_terms:
            if term == "command":
                command = self.embed_layers[term]
                if command.command_terms != ["command_timestep"]:
                    raise ValueError(
                        "Native timestep-command embedding contract changed."
                    )
                embeddings.append(
                    command.embed_layers["command_timestep"](normalized_time[:, None])
                )
            else:
                embeddings.append(self.embed_layers[term](chunks[term]))
        return self.layer_norm(torch.stack(embeddings).sum(0))


class TaskConditionedFeatures(nn.Module):
    """Add geometry/program modulation and the local reference window to native processed observations."""

    def __init__(
        self,
        layout: NativeObservationLayout,
        method: ExperimentConfig,
        architecture: ConditioningArchitecture,
        selected: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.observations = NativeTaskObservationProcessor(layout, method, selected)
        config = GeometryPolicyConfig(
            **asdict(architecture),
            hidden_dim=method.agent.policy_embed_dim,
            initial_log_std=-1.0,
            window_sigma=method.env.action_cfg.trajectory_weighted_policy_sigma,
        )
        self.geometry = GeometryEncoder(config)
        self.program = ProgramEncoder(GeometryPolicySchema(), config)
        context_width = method.agent.policy_embed_dim
        self.context = nn.Sequential(
            nn.Linear(
                architecture.geometry_embed_dim
                + 2 * architecture.program_embed_dim
                + 4,
                context_width,
            ),
            nn.SiLU(),
            nn.Linear(context_width, context_width),
        )
        width = method.agent.obs_proc_embed_dim
        self.reference = nn.Linear(10 * 31, width)
        nn.init.xavier_uniform_(self.reference.weight, gain=2**0.5)
        nn.init.zeros_(self.reference.bias)
        self.state_norm = AdaptiveRMSNorm(width, context_width)
        self.reference_norm = AdaptiveRMSNorm(width, context_width)

    def forward(self, inputs: FullMethodInputs) -> torch.Tensor:
        geometry = self.geometry(inputs.geometry_points, inputs.geometry_metadata)[
            inputs.geometry_index
        ]
        phases, program = self.program(inputs.program_tokens, inputs.program_mask)
        if not inputs.program_mask[inputs.program_index, inputs.current_phase].all():
            raise ValueError(
                "Current phase must select a valid native task-program token."
            )
        context = self.context(
            torch.cat(
                (
                    geometry,
                    program[inputs.program_index],
                    phases[inputs.program_index, inputs.current_phase],
                    inputs.progress,
                ),
                -1,
            )
        )
        state = self.observations(inputs.native_obs, inputs.normalized_reference_time)
        reference = torch.cat(
            (
                inputs.window.masked_fill(~inputs.window_mask[..., None], 0),
                inputs.window_mask[..., None],
            ),
            -1,
        ).flatten(1)
        return self.state_norm(state, context) + self.reference_norm(
            self.reference(reference), context
        )


class FullMethodActor(nn.Module):
    def __init__(
        self,
        layout: NativeObservationLayout,
        method: ExperimentConfig,
        architecture: ConditioningArchitecture,
    ) -> None:
        super().__init__()
        self.features = TaskConditionedFeatures(
            layout, method, architecture, method.agent.policy_inputs
        )
        self.network = PolicyNetwork(
            method.agent.obs_proc_embed_dim,
            10 * 18,
            embed_dim=method.agent.policy_embed_dim,
            hand_synergy_dim=method.agent.hand_synergy_dim,
            history_len=method.agent.history_len,
            device="cpu",
            last_activation=None,
            action_residuals=True,
        )
        # Native trajectory policies use one18D sampling distribution after blending candidates.
        self.network.log_std = nn.Parameter(torch.full((18,), -1.0))
        self.sigma = method.env.action_cfg.trajectory_weighted_policy_sigma
        if method.env.action_cfg.trajectory_weighted_policy_timestep_dropout != 0:
            raise ValueError(
                "The validated full-method preset must have zero timestep dropout."
            )

    def forward(self, inputs: FullMethodInputs) -> tuple[Normal, torch.Tensor]:
        candidates = self.network.net(self.features(inputs)).reshape(-1, 10, 18)
        logits = -0.5 * (inputs.native_obs.new_tensor(OFFSETS) / self.sigma).square()
        weights = (
            logits.expand(len(candidates), -1)
            .masked_fill(~inputs.window_mask, -torch.inf)
            .softmax(-1)
        )
        mean = (candidates * weights[..., None]).sum(1)
        return Normal(mean, self.network.log_std.exp().expand_as(mean)), weights


class FullMethodCritic(nn.Module):
    def __init__(
        self,
        layout: NativeObservationLayout,
        method: ExperimentConfig,
        architecture: ConditioningArchitecture,
    ) -> None:
        super().__init__()
        self.features = TaskConditionedFeatures(
            layout, method, architecture, method.agent.critic_inputs
        )
        self.network = ValueNetwork(
            method.agent.obs_proc_embed_dim,
            embed_dim=method.agent.value_embed_dim,
            history_len=method.agent.history_len,
            device="cpu",
        )

    def forward(self, inputs: FullMethodInputs) -> torch.Tensor:
        return self.network(self.features(inputs))


class FullMethodPolicy(nn.Module):
    def __init__(
        self,
        layout: NativeObservationLayout,
        method: ExperimentConfig,
        architecture: ConditioningArchitecture,
    ) -> None:
        super().__init__()
        if (
            method.agent.hand_obj_bps_aux_loss_coef != 0
            or method.agent.use_discriminator
        ):
            raise ValueError(
                "Expected the validated full-method preset with no auxiliary BPS loss or discriminator."
            )
        self.layout, self.method, self.architecture = layout, method, architecture
        self.actor = FullMethodActor(layout, method, architecture)
        self.critic = FullMethodCritic(layout, method, architecture)

    def forward(
        self, inputs: FullMethodInputs
    ) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        batch = len(inputs.native_obs)
        if batch < 1:
            raise ValueError("Expected a nonempty native observation batch.")
        shapes = {
            "native_obs": (batch, sum(self.layout.term_dims)),
            "window": (batch, 10, 30),
            "window_mask": (batch, 10),
            "progress": (batch, 4),
            "normalized_reference_time": (batch,),
            "geometry_index": (batch,),
            "program_index": (batch,),
            "current_phase": (batch,),
        }
        for name, shape in shapes.items():
            value = getattr(inputs, name)
            if (
                value.shape != shape
                or value.device != inputs.native_obs.device
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"Expected finite{name} with shape{shape} on the native observation device."
                )
        if inputs.window_mask.dtype != torch.bool or not inputs.window_mask[:, 4].all():
            raise ValueError(
                "Expected a boolean local-window mask with a valid current frame."
            )
        for name, limit in (
            ("geometry_index", len(inputs.geometry_points)),
            ("program_index", len(inputs.program_tokens)),
            ("current_phase", inputs.program_tokens.shape[1]),
        ):
            value = getattr(inputs, name)
            if value.dtype != torch.long or (value < 0).any() or (value >= limit).any():
                raise ValueError(f"Expected{name} as int64 indices in[0,{limit - 1}].")
        distribution, weights = self.actor(inputs)
        return distribution, self.critic(inputs), weights

    def checkpoint_schema(self) -> dict:
        return dict(
            version="c11.full_method.conditioned.local10.v1",
            preset=FULL_METHOD_PRESET,
            native_observation_layout=asdict(self.layout),
            agent=asdict(self.method.agent),
            action=asdict(self.method.env.action_cfg),
            architecture=asdict(self.architecture),
            offsets=list(OFFSETS),
            reference_fields=REFERENCE_SCHEMA,
            program_fields=PROGRAM_SCHEMA,
            geometry_fields=GEOMETRY_SCHEMA,
        )

    def optimizer(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            [
                {
                    "params": list(self.actor.parameters()),
                    "lr": self.method.agent.policy_lr,
                },
                {
                    "params": list(self.critic.parameters()),
                    "lr": self.method.agent.value_lr,
                },
            ]
        )


def make_full_method_policy(
    layout: NativeObservationLayout,
    method: ExperimentConfig,
    architecture: ConditioningArchitecture,
) -> FullMethodPolicy:
    with torch.device("cpu"):
        return FullMethodPolicy(layout, method, architecture)
