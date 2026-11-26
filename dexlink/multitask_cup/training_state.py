"""Atomic persistent checkpoints and explicit matching-lineage restart."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
import torch.distributed as distributed

from .learner import parameter_digest
from .full_method_config import FullMethodRunConfig
from .task_catalog import file_sha256

PHYSICS_RESUME_LIMITATION = (
    "Physical positions, velocities, control targets, clocks, task RNGs and reset history are restored. "
    "Genesis internal RNG and solver warm-start caches are reinitialized; continuation is not bitwise identical."
)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def prepare_run(config: FullMethodRunConfig, resume: bool):
    root = Path(config.run_directory).resolve()
    if resume:
        if not root.is_dir():
            raise ValueError(
                "Explicit resume requires the original persistent run directory."
            )
    else:
        root.mkdir(parents=True, exist_ok=False)
    lease = (root / "active.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected = {
        "run_id": config.run_id,
        "config_sha256": config.digest(),
        "run_directory": str(root),
        "data_manifest_sha256": file_sha256(Path(config.data) / "manifest.json"),
        "task_bank_sha256": file_sha256(Path(config.data) / "task-bank.pt"),
    }
    identity_path = root / "identity.json"
    if resume and identity_path.exists():
        if json.loads(identity_path.read_text()) != expected:
            raise ValueError(
                "Resume configuration, references or run identity differs from the original run."
            )
    elif resume and any(
        path.name != "active.lock" and not path.name.startswith("identity.json.")
        for path in root.iterdir()
    ):
        raise ValueError(
            "Cannot reconstruct a missing run identity in a directory containing training artifacts."
        )
    else:
        atomic_json(identity_path, expected)
        atomic_json(root / "config.json", asdict(config))
    return lease, expected


def load_latest(config: FullMethodRunConfig) -> tuple[dict, dict] | None:
    pointer_path = Path(config.run_directory) / "latest.json"
    if not pointer_path.exists():
        return None
    pointer = json.loads(pointer_path.read_text())
    seal_path = Path(pointer["seal"])
    root = Path(config.run_directory).resolve(strict=True)
    if (
        not seal_path.resolve(strict=True).is_relative_to(root)
        or file_sha256(seal_path) != pointer["sha256"]
    ):
        raise ValueError(
            "Latest checkpoint pointer is not an intact persistent run-local seal."
        )
    seal = json.loads(seal_path.read_text())
    if (
        seal["control_step"] <= 0
        or seal["config_sha256"] != config.digest()
        or seal["world_size"] != config.world_size
    ):
        raise ValueError(
            "Checkpoint requires positive controls and exact configuration/world-size lineage."
        )
    for filename, expected in seal["files"].items():
        if file_sha256(seal_path.parent / filename) != expected:
            raise ValueError(f"Incomplete or modified checkpoint file: {filename}.")
    model = torch.load(
        seal_path.parent / "shared.pt", map_location="cpu", weights_only=True
    )
    return seal, model


def validate_parent_configuration(parent: dict, current: dict) -> dict:
    """The author permits changing parallel environment count and rollout length on retry."""
    ignored = {
        "run_directory",
        "run_id",
        "parent_checkpoint",
        "parent_checkpoint_sha256",
    }
    before = {key: value for key, value in parent.items() if key not in ignored}
    after = {key: value for key, value in current.items() if key not in ignored}
    changes = {
        "environments_per_task": (
            before.pop("environments_per_task"),
            after.pop("environments_per_task"),
        )
    }
    before["ppo"] = dict(before["ppo"])
    after["ppo"] = dict(after["ppo"])
    changes["rollout_steps"] = (
        before["ppo"].pop("rollout_steps"),
        after["ppo"].pop("rollout_steps"),
    )
    if before != after:
        raise ValueError(
            "Parent import requires matching scientific configuration except approved environment count/rollout changes."
        )
    return changes


def load_parent(config: FullMethodRunConfig) -> tuple[dict, dict]:
    path = Path(config.parent_checkpoint).resolve(strict=True)
    if file_sha256(path) != config.parent_checkpoint_sha256:
        raise ValueError(
            "Parent checkpoint seal SHA256 differs from the declared lineage."
        )
    seal = json.loads(path.read_text())
    if (
        seal["control_step"] <= 0
        or seal["world_size"] != config.world_size
        or Path(seal["directory"]).resolve(strict=True) != path.parent
    ):
        raise ValueError(
            "Parent requires a positive checkpoint, matching world size and canonical directory."
        )
    original = json.loads((path.parent.parent.parent / "config.json").read_text())
    original_digest = hashlib.sha256(
        json.dumps(original, sort_keys=True).encode()
    ).hexdigest()
    if seal["config_sha256"] != original_digest:
        raise ValueError(
            "Parent checkpoint and original configuration hashes disagree."
        )
    changes = validate_parent_configuration(
        original, json.loads(json.dumps(asdict(config)))
    )
    for name, expected in seal["files"].items():
        if file_sha256(path.parent / name) != expected:
            raise ValueError(f"Parent checkpoint file changed: {name}.")
    model = torch.load(path.parent / "shared.pt", map_location="cpu", weights_only=True)
    if (
        model["config_sha256"] != original_digest
        or model["run_id"] != original["run_id"]
        or any(
            model[key] != seal[key]
            for key in (
                "control_step",
                "policy_version",
                "optimizer_updates",
                "aggregate_transitions",
            )
        )
    ):
        raise ValueError("Parent model metadata differs from its positive seal.")
    return seal, {**model, "retry_configuration_changes": changes}


def restore_parent_runtime(scenes: list, exploration, state: dict) -> None:
    """A larger vector batch restarts physical episodes; retain task-local RNG and reset history."""
    if len(scenes) != len(state["scenes"]):
        raise ValueError("Parent geometry/rank layout differs from the current retry.")
    for scene, saved in zip(scenes, state["scenes"]):
        if not torch.equal(scene.task_ids.unique().cpu(), saved["task_ids"].unique()):
            raise ValueError("Parent scene task identities differ from the retry.")
        scene.statistics.load_state_dict(saved["reset_statistics"])
    exploration.load_state_dict(state["exploration"])
    random.setstate(state["python_rng"])
    name, values, position, has_gaussian, cached_gaussian = state["numpy_rng"]
    np.random.set_state(
        (
            name,
            np.array(values, dtype=np.uint32),
            position,
            has_gaussian,
            cached_gaussian,
        )
    )
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])


def runtime_state(scenes: list, exploration) -> dict:
    numpy_state = np.random.get_state()
    return cpu_tree(
        {
            "scenes": [scene.state_dict() for scene in scenes],
            "exploration": exploration.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        }
    )


def restore_runtime(scenes: list, exploration, state: dict) -> None:
    if len(scenes) != len(state["scenes"]):
        raise ValueError(
            "Checkpoint geometry/rank scene count differs from the current layout."
        )
    for scene, saved in zip(scenes, state["scenes"]):
        scene.load_state_dict(saved)
    exploration.load_state_dict(state["exploration"])
    restore_application_rng(state)


def restore_application_rng(state: dict) -> None:
    """Restore application RNGs without touching physical scenes used by independent evaluation."""
    random.setstate(state["python_rng"])
    name, values, position, has_gaussian, cached_gaussian = state["numpy_rng"]
    np.random.set_state(
        (
            name,
            np.array(values, dtype=np.uint32),
            position,
            has_gaussian,
            cached_gaussian,
        )
    )
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])


def save_checkpoint(
    config: FullMethodRunConfig,
    policy,
    optimizer,
    counters: dict,
    scenes: list,
    exploration,
) -> dict:
    rank = distributed.get_rank()
    if counters["control_step"] <= 0:
        raise ValueError("Never save or evaluate an initialization checkpoint.")
    names = [uuid.uuid4().hex[:12] if rank == 0 else None]
    distributed.broadcast_object_list(names, src=0)
    directory = (
        Path(config.run_directory) / "models" / f"{counters['control_step']}-{names[0]}"
    )
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=False)
    distributed.barrier()
    if not directory.resolve(strict=True).is_relative_to(
        Path(config.run_directory).resolve(strict=True)
    ):
        raise ValueError("Checkpoint path escaped the persistent run directory.")
    rank_path = directory / f"rank-{rank}.pt"
    torch.save(runtime_state(scenes, exploration), rank_path.with_suffix(".tmp"))
    rank_path.with_suffix(".tmp").replace(rank_path)
    digests = [None] * config.world_size
    distributed.all_gather_object(digests, parameter_digest(policy))
    if len(set(digests)) != 1:
        raise RuntimeError("Shared policy parameters differ between learning ranks.")
    if rank == 0:
        shared = {
            **counters,
            "policy": cpu_tree(policy.state_dict()),
            "optimizer": cpu_tree(optimizer.state_dict()),
            "schema": policy.checkpoint_schema(),
            "config_sha256": config.digest(),
            "run_id": config.run_id,
        }
        torch.save(shared, directory / "shared.tmp")
        (directory / "shared.tmp").replace(directory / "shared.pt")
        loaded = torch.load(
            directory / "shared.pt", map_location="cpu", weights_only=True
        )
        if (
            loaded["control_step"] != counters["control_step"]
            or not loaded["optimizer"]["state"]
        ):
            raise RuntimeError(
                "Positive checkpoint did not reload with optimizer state."
            )
        with torch.random.fork_rng(devices=[]):
            from .geometry_policy import GeometryPolicySchema, make_geometry_policy

            restored = make_geometry_policy(GeometryPolicySchema(), config.model)
            restored.load_state_dict(loaded["policy"], strict=True)
            restored_optimizer = torch.optim.Adam(
                restored.parameters(), lr=config.ppo.learning_rate
            )
            restored_optimizer.load_state_dict(loaded["optimizer"])
            if parameter_digest(restored) != digests[0] or not restored_optimizer.state:
                raise RuntimeError(
                    "Policy or optimizer checkpoint failed strict reload."
                )
    distributed.barrier()
    result = [None]
    if rank == 0:
        files = {
            f"rank-{index}.pt": file_sha256(directory / f"rank-{index}.pt")
            for index in range(config.world_size)
        }
        files["shared.pt"] = file_sha256(directory / "shared.pt")
        seal = {
            **counters,
            "directory": str(directory.resolve()),
            "files": files,
            "world_size": config.world_size,
            "config_sha256": config.digest(),
            "physics_resume_limitation": PHYSICS_RESUME_LIMITATION,
            "parameter_digest": digests[0],
            "parent_seal": json.loads(
                (Path(config.run_directory) / "latest.json").read_text()
            )["seal"]
            if (Path(config.run_directory) / "latest.json").exists()
            else config.parent_checkpoint or None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(directory / "checkpoint.json", seal)
        atomic_json(
            Path(config.run_directory) / "latest.json",
            {
                "seal": str(directory / "checkpoint.json"),
                "sha256": file_sha256(directory / "checkpoint.json"),
            },
        )
        result[0] = seal
    distributed.broadcast_object_list(result, src=0)
    return result[0]
