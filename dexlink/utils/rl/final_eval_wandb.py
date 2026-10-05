from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dexlink.src.rl.config.final_eval_wandb import FinalEvalWandbUploadCfg

REQUIRED_EVAL_FILES = (
    "metadata.json",
    "summary_metrics.json",
    "episode_metrics.csv",
)
WANDB_ARTIFACT_NAME_MAX_LENGTH = 128


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object at {path}, got {type(payload).__name__}."
        )
    return payload


def _numeric_summary(path: Path) -> dict[str, float]:
    payload = _load_json_object(path)
    summary: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"Expected non-empty string metric keys in {path}.")
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Expected numeric summary value for {key!r}, got "
                f"{type(value).__name__}."
            )
        summary[key] = float(value)
    if not summary:
        raise ValueError(f"Expected at least one summary metric in {path}.")
    return summary


def _episode_table(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not reader.fieldnames:
            raise ValueError(f"Expected CSV columns in {path}.")
        columns = list(reader.fieldnames)
        rows: list[list[float]] = []
        for row_index, row in enumerate(reader):
            values: list[float] = []
            for column in columns:
                raw = row[column]
                if raw is None or raw == "":
                    raise ValueError(
                        f"Expected numeric value at row {row_index}, column "
                        f"{column!r} in {path}."
                    )
                values.append(float(raw))
            rows.append(values)
    if not rows:
        raise ValueError(f"Expected at least one episode row in {path}.")
    return columns, rows


def _artifact_name(run_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-.")
    if not normalized:
        raise ValueError(f"Could not form a W&B artifact name from {run_name!r}.")
    suffix = "-metrics"
    artifact_name = f"{normalized}{suffix}"
    if len(artifact_name) <= WANDB_ARTIFACT_NAME_MAX_LENGTH:
        return artifact_name

    # Preserve a readable prefix while the digest prevents truncated-name collisions.
    digest = hashlib.sha256(run_name.encode("utf-8")).hexdigest()[:12]
    bounded_suffix = f"-{digest}{suffix}"
    prefix_length = WANDB_ARTIFACT_NAME_MAX_LENGTH - len(bounded_suffix)
    return f"{normalized[:prefix_length].rstrip('-.')}{bounded_suffix}"


def upload_final_eval_metrics(cfg: FinalEvalWandbUploadCfg) -> dict[str, Any]:
    import wandb

    if not cfg.project.strip():
        raise ValueError("Expected a non-empty W&B project.")
    if not cfg.run_name.strip():
        raise ValueError("Expected a non-empty W&B run name.")
    if not cfg.group.strip():
        raise ValueError("Expected a non-empty W&B group.")
    if cfg.run_id is not None and not cfg.run_id.strip():
        raise ValueError("Expected run_id to be non-empty when provided.")
    if not cfg.output_dir.is_dir():
        raise FileNotFoundError(
            f"Evaluation output directory not found: {cfg.output_dir}"
        )
    missing = [
        name for name in REQUIRED_EVAL_FILES if not (cfg.output_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Evaluation output {cfg.output_dir} is missing: {', '.join(missing)}."
        )
    marker_path = cfg.output_dir / "wandb_run.json"
    if marker_path.exists():
        raise FileExistsError(
            f"Evaluation output was already uploaded according to {marker_path}."
        )

    metadata = _load_json_object(cfg.output_dir / "metadata.json")
    summary = _numeric_summary(cfg.output_dir / "summary_metrics.json")
    columns, episode_rows = _episode_table(cfg.output_dir / "episode_metrics.csv")
    wandb_summary = {f"eval/{key}": value for key, value in summary.items()}
    init_kwargs: dict[str, Any] = {
        "project": cfg.project,
        "entity": cfg.entity,
        "name": cfg.run_name,
        "group": cfg.group,
        "job_type": cfg.job_type,
        "config": {
            "evaluation": metadata,
            "evaluation_output_dir": str(cfg.output_dir),
        },
        "mode": cfg.mode,
        "reinit": True,
    }
    if cfg.run_id is not None:
        init_kwargs.update({"id": cfg.run_id, "resume": "must"})
    run = wandb.init(
        **init_kwargs,
    )
    if run is None:
        raise RuntimeError("wandb.init returned None for an enabled upload.")
    try:
        run.log(wandb_summary)
        run.log(
            {
                "eval/episodes": wandb.Table(
                    columns=columns,
                    data=episode_rows,
                )
            }
        )
        artifact = wandb.Artifact(
            _artifact_name(cfg.run_name),
            type="checkpoint_eval_metrics",
            metadata={
                "run_name": cfg.run_name,
                "group": cfg.group,
                "num_episodes": len(episode_rows),
            },
        )
        for filename in REQUIRED_EVAL_FILES + ("summary_metrics.csv",):
            path = cfg.output_dir / filename
            if path.is_file():
                artifact.add_file(str(path), name=filename)
        run.log_artifact(artifact)
        for key, value in wandb_summary.items():
            run.summary[key] = value
        marker = {
            "entity": cfg.entity,
            "group": cfg.group,
            "job_type": cfg.job_type,
            "mode": cfg.mode,
            "name": run.name,
            "project": run.project,
            "run_id": run.id,
            "url": run.url,
            "uploaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "num_episode_rows": len(episode_rows),
            "summary_metric_keys": sorted(summary),
        }
    finally:
        run.finish()
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    return marker
