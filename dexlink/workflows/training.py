"""Task deployment around the native full-method PPO implementation."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any

from .catalog import ROOT, repository_path

if TYPE_CHECKING:
    from .cli import TrainArguments


def train_task(
    task: dict[str, Any], arguments: TrainArguments, overrides: list[str]
) -> None:
    started = monotonic()
    import setuptools  # noqa: F401 - initialize distutils before Torch/Triton
    from dexlink.src.rl.core_agent.wandb_logger import WandbLogger
    from dexlink.src.rl.agent.ppo import PPO
    from dexlink.src.rl.config.base import (
        apply_overrides,
        apply_overrides_from_dict,
        extract_overrides,
        load_experiment_config,
    )
    from dexlink.src.rl.multi_object.env_factory import create_env
    from .cli import FULL_METHOD_PRESET

    if task["reference_format"] == "native_xhand":
        from dexlink.src.rl.library.config import TrainingConfig, task_config

        configuration = task_config(
            TrainingConfig(
                library=ROOT / "tasks/library40",
                instance=task["id"],
                num_envs=arguments.num_envs,
                timesteps=arguments.control_steps,
                seed=arguments.seed,
            )
        )
    else:
        configuration = load_experiment_config(FULL_METHOD_PRESET)
        import argparse

        configuration = apply_overrides(
            configuration,
            extract_overrides(argparse.Namespace(), task["training_overrides"]),
        )
    changes = {
        "env.task_cfg.oakink_exported_path": str(
            Path(task["reference"])
            if Path(task["reference"]).is_absolute()
            else repository_path(task["reference"])
        ),
        "env.sim_cfg.num_envs": arguments.num_envs,
        "env.sim_cfg.show_viewer": False,
        "env.seed": arguments.seed,
        "agent.seed": arguments.seed,
        "agent.timesteps": arguments.control_steps,
        "agent.resume": False,
        "agent.resume_from_ckpt": None,
        "logger.project": arguments.project,
        "logger.group": "tasks",
        "logger.runname": arguments.run_directory.name,
        "logger.timestamp": False,
        "logger.seed": False,
        "logger.disable": False,
        "logger.model_save_freq": 125,
        "logger.log_freq": 1 if arguments.numerical_audit else 25,
    }
    configuration = apply_overrides_from_dict(configuration, changes)
    if overrides:
        import argparse

        configuration = apply_overrides(
            configuration, extract_overrides(argparse.Namespace(), overrides)
        )
    from dexlink.src.rl.config.seq_oakink_presets import required_seq_task_settings

    configuration = apply_overrides_from_dict(
        configuration,
        required_seq_task_settings(
            configuration.env.task_cfg.oakink_exported_path,
            preset_name=configuration.preset,
        ),
    )
    if (
        configuration.logger.disable
        or os.environ.get("WANDB_MODE", "online") != "online"
    ):
        raise ValueError(
            "Policy learning requires online W&B logging. Run wandb login and choose a writable entity/project."
        )
    destination = arguments.run_directory.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "launch.json").write_text(
        json.dumps(
            {
                "task": task["id"],
                "preset": FULL_METHOD_PRESET,
                "seed": arguments.seed,
                "control_steps": arguments.control_steps,
                "deployment_overrides": changes,
                "cli_overrides": overrides,
                "max_seconds": arguments.max_seconds,
                "initialization": "fresh; no learned weights or optimizer state",
            },
            indent=2,
        )
        + "\n"
    )
    if arguments.entity is not None:
        os.environ["WANDB_ENTITY"] = arguments.entity
    audit = None
    if arguments.numerical_audit:
        from .numerics import NumericalAudit

        audit = NumericalAudit()

    class TrainingLogger(WandbLogger):
        def log(self, data, *args, **kwargs):
            if audit is not None:
                audit.record_losses(data)
            return super().log(data, *args, **kwargs)

    logger = TrainingLogger(
        configuration.logger, exp_seed=arguments.seed, experiment_dir=destination
    )
    import wandb

    run = wandb.run
    if run is None or run.settings.mode != "online" or run.project != arguments.project:
        raise RuntimeError("W&B did not initialize the requested online project.")
    if arguments.entity is not None and run.entity != arguments.entity:
        raise RuntimeError(
            f"Expected W&B entity {arguments.entity!r}, received {run.entity!r}."
        )
    (destination / "wandb.json").write_text(
        json.dumps(
            {
                "id": run.id,
                "entity": run.entity,
                "project": run.project,
                "url": run.url,
                "mode": run.settings.mode,
            },
            indent=2,
        )
        + "\n"
    )
    configuration.env.sim_cfg.video_save_dir = str(destination / "videos")
    configuration.eval.logdir = str(destination / "eval")
    configuration.agent.device = configuration.env.device
    logger.save_config(asdict(configuration))
    logger.add_config({"experiment": asdict(configuration)})
    environment = create_env(configuration.env)
    environment.task_manager.verify_target_object_motion(mode="sim")
    configuration.agent.use_discriminator = environment.use_discriminator
    agent = PPO(
        environment,
        logger=logger,
        agent_cfg=configuration.agent,
        eval_cfg=configuration.eval,
    )
    if audit is not None:
        audit.attach(agent)
    completed = False
    failure = None
    try:
        completed_controls = agent.train(
            stop_requested=(
                (lambda: monotonic() - started >= arguments.max_seconds)
                if arguments.max_seconds is not None
                else None
            )
        )
        if completed_controls <= 0:
            raise RuntimeError(
                "Training produced no positive checkpoint; evaluation is prohibited."
            )
        checkpoint = destination / "models" / str(completed_controls)
        required_files = [
            f"{name}.pt"
            for name, module in agent._checkpoint_models().items()
            if module is not None
        ]
        missing = [name for name in required_files if not (checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete final checkpoint {checkpoint}: missing {missing}."
            )
        if audit is not None and (
            audit.optimizer_updates < 1
            or audit.loss_batches < 1
            or not agent.optimizer.state
        ):
            raise RuntimeError(
                "Numerical smoke did not exercise PPO losses and optimizer updates."
            )
        completed = True
        result = {
            "task": task["id"],
            "completed_control_steps": completed_controls,
            "requested_control_steps": arguments.control_steps,
            "stop_reason": "control_budget"
            if completed_controls == arguments.control_steps
            else "time_budget",
            "elapsed_seconds": monotonic() - started,
            "checkpoint": str(checkpoint),
            "checkpoint_files": required_files,
            "wandb_url": run.url,
        }
        (destination / "training-result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        run.summary.update(
            {
                "training/completed_control_steps": completed_controls,
                "training/stop_reason": result["stop_reason"],
            }
        )
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        if audit is not None:
            audit.save(
                destination / "numerical-audit.json",
                completed=completed,
                failure=failure,
            )
        wandb.finish(exit_code=0 if completed else 1)
