"""Reusable scene and rollout rendering utilities."""

from __future__ import annotations


from pathlib import Path


import numpy as np


def load_reference_motion(path: Path) -> dict:
    """Read the two declared reference formats without renaming/converting files."""
    if path.suffix == ".npz":
        archive = np.load(path, allow_pickle=True)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError(f"Expected an NPZ archive at {path}, received {type(archive).__name__}.")
        with archive:
            motion = {key: archive[key] for key in archive.files}
    elif path.suffix == ".npy":
        payload = np.load(path, allow_pickle=True)
        if not isinstance(payload, np.ndarray) or payload.shape != () or not isinstance(payload.item(), dict):
            raise ValueError(f"Expected a scalar-dictionary NPY reference at {path}, received {type(payload).__name__}.")
        motion = payload.item()
    else:
        raise ValueError(f"Expected a native NPZ or scalar-dictionary NPY reference, received {path}.")
    positions = motion["scene_object_pos"]
    if not isinstance(positions, np.ndarray) or positions.ndim != 3 or positions.shape[-1] != 3 or positions.shape[0] < 1:
        raise ValueError(f"Expected scene_object_pos with shape (T, objects, 3), received {getattr(positions, 'shape', None)}.")
    return motion


def _load_episode_from_eval(
    eval_dir: Path, episode_index: int
) -> dict[str, np.ndarray]:
    rollouts_dir = eval_dir / "rollouts"
    chunks = sorted(rollouts_dir.glob("chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"No rollout chunks found in {rollouts_dir}.")
    offset = 0
    for chunk_path in chunks:
        with np.load(chunk_path, allow_pickle=True) as data:
            chunk = {key: data[key] for key in data.files}
        done = np.asarray(chunk["done"])
        if done.ndim != 2:
            raise ValueError(
                f"Expected rollout chunk 'done' with shape (T, N), got {done.shape}."
            )
        count = int(done.shape[1])
        if episode_index < offset + count:
            env_idx = episode_index - offset
            done_indices = np.where(done[:, env_idx])[0]
            length = (
                int(done_indices[0] + 1) if done_indices.size else int(done.shape[0])
            )
            episode = {
                key: np.asarray(value)[:length, env_idx]
                for key, value in chunk.items()
                if key != "start_timestep"
                and np.asarray(value).ndim >= 2
                and np.asarray(value).shape[0] >= length
                and np.asarray(value).shape[1] > env_idx
            }
            if "start_timestep" in chunk:
                episode["start_timestep"] = np.asarray(chunk["start_timestep"])[env_idx]
            episode["length"] = np.asarray(length, dtype=np.int64)
            return episode
        offset += count
    raise ValueError(
        f"Episode index {episode_index} is outside {offset} rollout episodes in {eval_dir}."
    )


def _rollout_motion_from_reference(reference_motion: dict, episode: dict) -> dict:
    motion = dict(reference_motion)
    for key in list(motion):
        if key.startswith("drake_collision_refinement_"):
            motion.pop(key)
    motion["scene_object_pos"] = np.asarray(episode["object_pos"], dtype=np.float32)
    motion["scene_object_quat"] = np.asarray(episode["object_quat"], dtype=np.float32)
    motion["scene_object_arti_vals"] = np.asarray(
        episode["object_arti"], dtype=np.float32
    )
    for side in ("left", "right"):
        key = f"{side}_jvals"
        if key in episode:
            motion[key] = np.asarray(episode[key], dtype=np.float32)
    return motion
