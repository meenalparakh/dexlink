"""Full-method inheritance with only conditional geometry and program inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from dataclasses import fields, is_dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path

from dexlink.src.rl.config.base import ExperimentConfig, load_experiment_config
from .task_catalog import validate_training_catalog


FULL_METHOD_PRESET = "seq_oakink_residual_target_trajectory_weighted_policy_v3"
INTERNAL_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ConditioningArchitecture:
    point_count: int = 512
    geometry_hidden_dims: tuple[int, ...] = (64, 128, 128)
    geometry_embed_dim: int = 128
    program_embed_dim: int = 128
    program_layers: int = 2
    program_heads: int = 4
    max_program_phases: int = 12


@dataclass(frozen=True)
class TransferInitialization:
    arm: str = "source"
    study: str = ""
    study_sha256: str = ""
    checkpoint: str = ""
    checkpoint_sha256: str = ""
    source_control_step: int = 0


@dataclass
class FullMethodRunConfig:
    data: str = ""
    run_directory: str = ""
    run_id: str = ""
    seed: int = 42
    world_size: int = 8
    environments_per_task: int = 2048
    rollout_steps: int = 16
    control_steps: int = 50000
    checkpoint_interval: int = 1000
    evaluation_interval: int = 1000
    evaluation_episodes: int = 50
    task_indices: tuple[int, ...] = tuple(range(30))
    validation_mode: bool = False
    cache_directory: str = str(INTERNAL_ROOT / "outputs/cache/multitask")
    storage_root: str = str(INTERNAL_ROOT / "outputs/multitask")
    wandb_entity: str = os.environ.get("WANDB_ENTITY", "")
    wandb_project: str = os.environ.get("WANDB_PROJECT", "dexlink")
    architecture: ConditioningArchitecture = field(
        default_factory=ConditioningArchitecture
    )
    transfer: TransferInitialization = field(default_factory=TransferInitialization)
    checkpoint_controls: tuple[int, ...] = ()
    execution_platform: str = "local"

    def checkpoint_storage_root(self) -> Path:
        return Path(self.storage_root).expanduser()

    def wandb_mode(self) -> str:
        return "online"

    def method(self) -> ExperimentConfig:
        """Read inherited hyperparameters from their authoritative preset, never generic PPO defaults."""
        resolved = load_experiment_config(FULL_METHOD_PRESET)
        resolved.agent.timesteps = self.control_steps
        resolved.agent.n_rollout_steps = self.rollout_steps
        resolved.agent.seed = self.seed
        resolved.env.seed = self.seed
        resolved.env.sim_cfg.num_envs = self.environments_per_task
        return resolved

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()

    def validate(self) -> None:
        if not self.data or not self.run_directory or not self.run_id or self.seed < 0:
            raise ValueError(
                "Expected task data, a unique run directory/ID and a nonnegative seed."
            )
        if self.world_size not in (1, 4, 8) or self.rollout_steps not in (8, 16):
            raise ValueError(
                "Expected one/four/eight GPU ranks and rollout length 8 or 16."
            )
        if self.execution_platform not in ("local", "cluster"):
            raise ValueError("Expected local or cluster execution_platform.")
        manifest = json.loads((Path(self.data) / "manifest.json").read_text())
        count = validate_training_catalog(manifest)
        if (
            not self.task_indices
            or len(set(self.task_indices)) != len(self.task_indices)
            or min(self.task_indices) < 0
            or max(self.task_indices) >= count
        ):
            raise ValueError(
                f"Expected distinct task indices in 0..{count - 1}, received {self.task_indices}."
            )
        if len(self.task_indices) < self.world_size or self.environments_per_task < 2:
            raise ValueError(
                "Every rank requires tasks and every task requires at least two environments."
            )
        if (
            self.control_steps <= 0
            or self.control_steps % 8
            or not 1 <= self.checkpoint_interval <= 10000
        ):
            raise ValueError(
                "Expected positive controls divisible by 8 and checkpoint cadence <= 10k."
            )
        if (
            self.evaluation_interval < 0
            or self.evaluation_interval % self.checkpoint_interval
            or self.evaluation_episodes < 50
        ):
            raise ValueError(
                "Evaluation must align with checkpoints and use at least 50 episodes per task."
            )
        if self.checkpoint_controls and (
            tuple(sorted(set(self.checkpoint_controls))) != self.checkpoint_controls
            or self.checkpoint_controls[0] <= 0
            or self.checkpoint_controls[-1] != self.control_steps
            or any(step % 8 for step in self.checkpoint_controls)
        ):
            raise ValueError(
                "Checkpoint controls must increase from positive multiples of 8 to the endpoint."
            )
        if self.transfer.arm not in ("source", "warm_start", "scratch"):
            raise ValueError("Expected source, warm_start or scratch initialization.")
        if self.transfer.arm != "source":
            study_path = Path(self.transfer.study)
            if (
                self.world_size != 1
                or len(self.task_indices) != 1
                or not study_path.is_file()
            ):
                raise ValueError(
                    "Transfer requires one GPU, one selected target and an explicit study manifest."
                )
            if (
                hashlib.sha256(study_path.read_bytes()).hexdigest()
                != self.transfer.study_sha256
            ):
                raise ValueError("Transfer study SHA256 does not match.")
            study = json.loads(study_path.read_text())
            if (
                self.control_steps != study["target_controls"]
                or hashlib.sha256(
                    (Path(self.data) / "manifest.json").read_bytes()
                ).hexdigest()
                != study["target_manifest_sha256"]
            ):
                raise ValueError(
                    "Transfer task data or control budget differs from its manifest."
                )
            if self.transfer.arm == "warm_start" and (
                self.transfer.source_control_step <= 0
                or not self.transfer.checkpoint
                or not self.transfer.checkpoint_sha256
            ):
                raise ValueError(
                    "Warm initialization requires a positive source checkpoint and its SHA256."
                )
            if self.transfer.arm == "scratch" and (
                self.transfer.checkpoint
                or self.transfer.checkpoint_sha256
                or self.transfer.source_control_step
            ):
                raise ValueError(
                    "Scratch initialization cannot contain learned policy state."
                )
        persistent = self.checkpoint_storage_root().resolve()
        if persistent.is_relative_to(Path("/tmp")) or not Path(
            self.run_directory
        ).resolve().is_relative_to(persistent):
            raise ValueError(
                "Run directory must be inside the configured persistent storage_root, outside /tmp."
            )
        persistent.mkdir(parents=True, exist_ok=True)


def task_configuration(
    config: FullMethodRunConfig, variant: dict, total_slots: int, *, evaluation: bool
) -> ExperimentConfig:
    """Apply task assets, approved resources and evaluation protocol to the unchanged native method."""
    resolved = config.method()
    resolved.env.seed = config.seed
    resolved.env.device = "cuda:0"
    resolved.env.task_cfg.oakink_exported_path = variant["reference"]
    resolved.env.task_cfg.lift_support_height = 0.0
    resolved.env.task_cfg.lift_support_margin = 0.002
    resolved.env.task_cfg.save_reference_video = False
    resolved.env.sim_cfg.num_envs = total_slots
    resolved.env.sim_cfg.show_viewer = False
    resolved.env.sim_cfg.video_save_freq = 0
    resolved.env.sim_cfg.n_rendered_envs = 0
    resolved.env.task_cfg.metric_cfg.metric_id = "object_pose"
    resolved.env.task_cfg.metric_cfg.object_pose_object_id = "cup"
    resolved.env.task_cfg.metric_cfg.object_pose_position_tolerance_m = 0.03
    # This is the approved cup functional metric; it does not replace training reward terms.
    import math

    resolved.env.task_cfg.metric_cfg.object_pose_orientation_tolerance_rad = (
        math.pi / 12
    )
    if evaluation:
        resolved.env.rewards_cfg.last_reference_hold_steps = 10
        resolved.env.reset_cfg.eval_start_timestep = 0
    return resolved


def load_full_config(path: Path) -> FullMethodRunConfig:
    values = json.loads(path.read_text())
    values["task_indices"] = tuple(values["task_indices"])
    architecture = values.pop("architecture")
    architecture["geometry_hidden_dims"] = tuple(architecture["geometry_hidden_dims"])
    if "transfer" in values:
        values["transfer"] = TransferInitialization(**values["transfer"])
    if "checkpoint_controls" in values:
        values["checkpoint_controls"] = tuple(values["checkpoint_controls"])
    return FullMethodRunConfig(
        **values, architecture=ConditioningArchitecture(**architecture)
    )


def parse_full_config(argv: list[str] | None = None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, required=True)
    known, _ = bootstrap.parse_known_args(argv)
    config = load_full_config(known.config)
    parser = argparse.ArgumentParser(parents=[bootstrap], description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop_after_control", type=int, default=0)
    parser.add_argument("--exit_before_initialization", action="store_true")

    def register(owner: object, prefix: str = "") -> None:
        for item in fields(owner):
            value = getattr(owner, item.name)
            key = prefix + item.name
            if is_dataclass(value):
                register(value, key + ".")
            else:
                convert = type(value)
                if isinstance(value, tuple):
                    convert = lambda text: tuple(int(part) for part in text.split(","))
                elif isinstance(value, bool):

                    def convert(text: str) -> bool:
                        if text not in ("true", "false"):
                            raise argparse.ArgumentTypeError("Expected true or false.")
                        return text == "true"

                parser.add_argument("--" + key, type=convert, default=argparse.SUPPRESS)

    register(config)
    options = parser.parse_args(argv)
    values = asdict(config)
    for key, value in vars(options).items():
        if key in (
            "config",
            "resume",
            "stop_after_control",
            "exit_before_initialization",
        ):
            continue
        parts = key.split(".")
        owner = values
        for part in parts[:-1]:
            owner = owner[part]
        owner[parts[-1]] = value
    architecture = values.pop("architecture")
    values["transfer"] = TransferInitialization(**values["transfer"])
    config = FullMethodRunConfig(
        **values, architecture=ConditioningArchitecture(**architecture)
    )
    config.validate()
    return config, options


def next_checkpoint_control(config: FullMethodRunConfig, control: int) -> int:
    if not 0 <= control < config.control_steps:
        raise ValueError(
            "Next-checkpoint lookup requires an unfinished nonnegative training control count."
        )
    if config.checkpoint_controls:
        return next(step for step in config.checkpoint_controls if step > control)
    return min(
        config.control_steps,
        (control // config.checkpoint_interval + 1) * config.checkpoint_interval,
    )


def evaluation_due(config: FullMethodRunConfig, control: int) -> bool:
    if control <= 0 or not config.evaluation_interval:
        return False
    if config.checkpoint_controls:
        return control in config.checkpoint_controls
    return control % config.evaluation_interval == 0 or control == config.control_steps
