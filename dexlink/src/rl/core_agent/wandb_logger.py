import json
import os
from datetime import datetime
from pathlib import Path

import torch
import wandb

from dexlink.src.rl.config.base import LoggerConfig


class WandbLogger:
    @staticmethod
    def _save_state_dict_atomically(module, model_path: Path) -> None:
        """Expose a checkpoint file only after its complete state is on disk."""
        temporary_path = model_path.with_suffix(f"{model_path.suffix}.tmp")
        torch.save(module.state_dict(), temporary_path)
        os.replace(temporary_path, model_path)

    def __init__(
        self,
        cfg: LoggerConfig,
        exp_seed: int = 0,
        experiment_dir: str | Path | None = None,
    ):
        self.cfg = cfg

        runname = cfg.runname
        if cfg.seed:
            runname = f"{runname}-s{exp_seed}"
        if cfg.timestamp:
            timestamp = datetime.now().strftime("%m_%d-%H_%M")
            runname = f"{runname}-{timestamp}"

        cfg.runname = runname
        experiment_dir = (
            Path(experiment_dir)
            if experiment_dir is not None
            else Path(cfg.logdir) / cfg.project / cfg.group / runname
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = experiment_dir

        if not cfg.disable:
            wandb.init(
                project=cfg.project,
                # entity=cfg.entity,
                name=runname,
                dir=str(experiment_dir),
                group=cfg.group,
            )

        self.log_freq = cfg.log_freq
        self.model_save_freq = cfg.model_save_freq
        self.save_models = cfg.model_save_freq > 0

    def compute_freq(
        self,
        total_train_iters: int,
        desired_num_logs: int = 500,
        desired_model_saves: int = 20,
    ):
        self.log_freq = max(1, total_train_iters // desired_num_logs)
        self.model_save_freq = max(1, total_train_iters // desired_model_saves)
        self.save_models = self.model_save_freq > 0
        print(
            f"Computed log frequency: {self.log_freq}, model save frequency: {self.model_save_freq}"
        )

    def configure_frequencies(self, total_train_iters: int):
        """Resolve logging/save cadence from explicit cfg values or desired totals."""
        explicit_log_freq = self.cfg.log_freq > 0
        explicit_model_save_freq = self.cfg.model_save_freq > 0

        if explicit_log_freq and explicit_model_save_freq:
            self.log_freq = self.cfg.log_freq
            self.model_save_freq = self.cfg.model_save_freq
            self.save_models = self.model_save_freq > 0
            print(
                f"Using explicit log frequency: {self.log_freq}, model save frequency: {self.model_save_freq}"
            )
            return

        self.compute_freq(
            total_train_iters,
            desired_num_logs=self.cfg.desired_num_logs,
            desired_model_saves=self.cfg.desired_model_saves,
        )
        if explicit_log_freq:
            self.log_freq = self.cfg.log_freq
        if explicit_model_save_freq:
            self.model_save_freq = self.cfg.model_save_freq
        self.save_models = self.model_save_freq > 0

    def save_config(self, cfg_dict: dict, name="exp_cfg.json"):
        def _json_fallback(obj):
            # try common structured containers first
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)

        with open(self.experiment_dir / name, "w") as f:
            json.dump(cfg_dict, f, indent=4, default=_json_fallback)

    def add_config(self, cfg_dict: dict):
        if self.cfg.disable:
            return
        wandb.config.update(cfg_dict)

    def load_named_checkpoint_metadata(self, checkpoint_name: str) -> dict | None:
        metadata_path = (
            self.experiment_dir / "models" / checkpoint_name / "metadata.json"
        )
        if not metadata_path.exists():
            return None
        with metadata_path.open("r") as f:
            metadata = json.load(f)
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Expected checkpoint metadata dict at {metadata_path}, "
                f"got {type(metadata).__name__}."
            )
        return metadata

    def save_named_checkpoint(
        self,
        models: dict,
        checkpoint_name: str,
        metadata: dict,
    ) -> None:
        if self.cfg.disable:
            return
        if (
            not checkpoint_name
            or checkpoint_name in {".", ".."}
            or Path(checkpoint_name).name != checkpoint_name
        ):
            raise ValueError(
                "Expected checkpoint_name to be one non-empty path component, "
                f"got {checkpoint_name!r}."
            )
        model_dir = self.experiment_dir / "models" / checkpoint_name
        model_dir.mkdir(parents=True, exist_ok=True)
        for name, module in models.items():
            if module is not None:
                print(f"Saving model '{name}' to named checkpoint {checkpoint_name!r}")
                self._save_state_dict_atomically(
                    module,
                    model_dir / f"{name}.pt",
                )
        with (model_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    def log(
        self,
        data: dict,
        timestep: int,
        train_iters: int,
        force_model_save: bool = False,
        force_log: bool = False,
    ):

        if self.cfg.disable:
            return

        model = data.pop("model", None)  # remove model if present
        if (train_iters % self.log_freq == 0) or force_log:
            wandb.log(data, step=timestep)

        if (
            (train_iters % self.model_save_freq == 0) and self.save_models
        ) or force_model_save:
            # look for 'model' key in data
            if model is not None:
                model_dir = self.experiment_dir / "models" / str(timestep)
                model_dir.mkdir(parents=True, exist_ok=True)
                for name, module in model.items():
                    if module is not None:
                        print(f"Saving model '{name}' at timestep {timestep}")
                        model_path = model_dir / f"{name}.pt"
                        self._save_state_dict_atomically(module, model_path)
