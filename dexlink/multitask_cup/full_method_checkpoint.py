"""Persistent all-rank checkpoints with strict native actor/critic/AdamW reload."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import torch
import torch.distributed as distributed

from .full_method_config import FullMethodRunConfig
from .full_method_policy import make_full_method_policy
from .learner import parameter_digest
from .task_catalog import file_sha256
from .training_state import (
    PHYSICS_RESUME_LIMITATION,
    atomic_json,
    cpu_tree,
    runtime_state,
)


def save_full_method_checkpoint(
    config: FullMethodRunConfig,
    policy,
    optimizer,
    counters: dict,
    scenes: list,
    exploration,
) -> dict:
    rank = distributed.get_rank()
    if counters["control_step"] <= 0:
        raise ValueError("Full-method checkpoint requires positive training controls.")
    labels = [uuid.uuid4().hex[:12] if rank == 0 else None]
    distributed.broadcast_object_list(labels, src=0)
    directory = (
        Path(config.run_directory)
        / "models"
        / f"{counters['control_step']}-{labels[0]}"
    )
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=False)
    distributed.barrier()
    if not directory.resolve(strict=True).is_relative_to(
        Path(config.run_directory).resolve(strict=True)
    ):
        raise ValueError("Full-method checkpoint escaped its persistent run directory.")
    rank_path = directory / f"rank-{rank}.pt"
    torch.save(runtime_state(scenes, exploration), rank_path.with_suffix(".tmp"))
    rank_path.with_suffix(".tmp").replace(rank_path)
    digests = [None] * config.world_size
    distributed.all_gather_object(digests, parameter_digest(policy))
    if len(set(digests)) != 1:
        raise RuntimeError(
            "Full-method policy differs across synchronized learning ranks."
        )
    if rank == 0:
        state = dict(
            **counters,
            policy=cpu_tree(policy.state_dict()),
            optimizer=cpu_tree(optimizer.state_dict()),
            schema=policy.checkpoint_schema(),
            config_sha256=config.digest(),
            run_id=config.run_id,
        )
        torch.save(state, directory / "shared.tmp")
        (directory / "shared.tmp").replace(directory / "shared.pt")
        loaded = torch.load(
            directory / "shared.pt", map_location="cpu", weights_only=True
        )
        with torch.random.fork_rng(devices=[]):
            restored = make_full_method_policy(
                policy.layout, config.method(), config.architecture
            )
            restored.load_state_dict(loaded["policy"], strict=True)
            restored_optimizer = restored.optimizer()
            restored_optimizer.load_state_dict(loaded["optimizer"])
            if parameter_digest(restored) != digests[0] or not restored_optimizer.state:
                raise RuntimeError(
                    "Native full-method actor/critic/AdamW checkpoint failed strict reload."
                )
            steps = {
                int(value["step"].item()) for value in restored_optimizer.state.values()
            }
            if steps != {counters["optimizer_updates"]}:
                raise RuntimeError(
                    "Restored native AdamW steps disagree with the persistent training counter."
                )
    distributed.barrier()
    result = [None]
    if rank == 0:
        files = {
            f"rank-{index}.pt": file_sha256(directory / f"rank-{index}.pt")
            for index in range(config.world_size)
        }
        files["shared.pt"] = file_sha256(directory / "shared.pt")
        previous = Path(config.run_directory) / "latest.json"
        seal = dict(
            **counters,
            directory=str(directory.resolve()),
            files=files,
            world_size=config.world_size,
            config_sha256=config.digest(),
            parameter_digest=digests[0],
            parent_seal=json.loads(previous.read_text())["seal"]
            if previous.exists()
            else None,
            physics_resume_limitation=PHYSICS_RESUME_LIMITATION,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(directory / "checkpoint.json", seal)
        atomic_json(
            previous,
            dict(
                seal=str(directory / "checkpoint.json"),
                sha256=file_sha256(directory / "checkpoint.json"),
            ),
        )
        result[0] = seal
    distributed.broadcast_object_list(result, src=0)
    return result[0]
