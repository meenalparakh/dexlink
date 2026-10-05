from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from dexlink.src.retargeting.data.oakink_processor_sequence import OakinkProcessorSequence
from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE, ArcticClipSpec, get_clip
from dexlink.src.reference.arctic.runtime import (
    materialize_runtime_sequence,
    materialize_runtime_transfer_sequence,
    materialize_runtime_urdfs,
)
from dexlink.src.reference.arctic.transfer_config import get_transfer

DEXMACHINA_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01, dtype=np.float64)
PART_NAMES = ("top", "bottom")


@dataclass(frozen=True)
class AddAucArgs:
    clip: str | None
    transfer: str | None
    rollout_dir: Path
    workspace: Path
    output_dir: Path | None
    num_vertices_per_part: int
    vertex_seed: int
    bootstrap_samples: int
    bootstrap_seed: int
    overwrite: bool


def parse_args(argv: list[str] | None = None) -> AddAucArgs:
    parser = argparse.ArgumentParser(
        description="Compute the DexMachina ADD-AUC metric for ARCTIC rollout chunks."
    )
    benchmark = parser.add_mutually_exclusive_group(required=True)
    benchmark.add_argument("--clip")
    benchmark.add_argument("--transfer")
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-vertices-per-part", type=int, default=500)
    parser.add_argument("--vertex-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    raw = parser.parse_args(argv)
    if raw.num_vertices_per_part <= 0:
        raise ValueError(
            "Expected --num-vertices-per-part > 0, got " f"{raw.num_vertices_per_part}."
        )
    if raw.bootstrap_samples <= 0:
        raise ValueError(
            f"Expected --bootstrap-samples > 0, got {raw.bootstrap_samples}."
        )
    return AddAucArgs(
        clip=None if raw.clip is None else str(raw.clip),
        transfer=None if raw.transfer is None else str(raw.transfer),
        rollout_dir=raw.rollout_dir.resolve(),
        workspace=raw.workspace.resolve(),
        output_dir=None if raw.output_dir is None else raw.output_dir.resolve(),
        num_vertices_per_part=int(raw.num_vertices_per_part),
        vertex_seed=int(raw.vertex_seed),
        bootstrap_samples=int(raw.bootstrap_samples),
        bootstrap_seed=int(raw.bootstrap_seed),
        overwrite=bool(raw.overwrite),
    )


def compute_auc(distances: np.ndarray, thresholds: np.ndarray) -> float:
    values = np.asarray(distances, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(
            "ADD-AUC expects at least one finite distance, got "
            f"shape {values.shape}."
        )
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if thresholds.size < 2 or np.any(np.diff(thresholds) <= 0.0):
        raise ValueError(
            f"Expected at least two strictly increasing thresholds, got {thresholds}."
        )
    accuracies = np.asarray(
        [np.mean(values < threshold) for threshold in thresholds], dtype=np.float64
    )
    return float(np.trapezoid(accuracies, x=np.linspace(0.0, 1.0, len(thresholds))))


def _part_mesh_path(clip: ArcticClipSpec, workspace: Path, part: str) -> Path:
    asset_dir = clip.object_asset_dir(workspace)
    stl_path = asset_dir / f"{part}_watertight_tiny.stl"
    obj_path = asset_dir / f"{part}_watertight_tiny.obj"
    if stl_path.is_file():
        return stl_path
    if obj_path.is_file():
        return obj_path
    raise FileNotFoundError(
        f"DexMachina {part!r} metric mesh is missing for {clip.clip_id!r}; "
        f"expected {stl_path} or {obj_path}."
    )


def sample_part_vertices(
    clip: ArcticClipSpec,
    workspace: Path,
    *,
    num_vertices: int,
    seed: int,
) -> dict[str, np.ndarray]:
    sampled = {}
    for part in PART_NAMES:
        mesh_path = _part_mesh_path(clip, workspace, part)
        mesh = trimesh.load_mesh(mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Expected Trimesh at {mesh_path}, got {type(mesh)}.")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError(
                f"Expected non-empty vertices shape (V, 3) at {mesh_path}, "
                f"got {vertices.shape}."
            )
        # DexMachina resets np.random.seed(seed) independently for each part.
        rng = np.random.RandomState(seed)
        indices = rng.choice(
            len(vertices), num_vertices, replace=num_vertices > len(vertices)
        )
        sampled[part] = vertices[indices]
    return sampled


def sample_urdf_part_vertices(
    urdf: URDF,
    *,
    num_vertices: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Sample target-asset vertices in each articulated part's local frame."""
    zero_cfg = {
        joint.name: 0.0 for joint in urdf.joints if joint.joint_type != "fixed"
    }
    link_fk = {
        link.name: np.asarray(transform, dtype=np.float64)
        for link, transform in urdf.link_fk(cfg=zero_cfg).items()
    }
    sampled: dict[str, np.ndarray] = {}
    for part in PART_NAMES:
        if part not in link_fk:
            raise ValueError(
                f"Transferred target URDF is missing required link {part!r}; "
                f"got {sorted(link_fk)}."
            )
        part_from_root = np.linalg.inv(link_fk[part])
        vertices_by_visual = []
        for link in urdf.links:
            if link.name != part and not link.name.startswith(f"{part}_"):
                continue
            for visual in link.visuals:
                part_from_visual = part_from_root @ link_fk[link.name] @ visual.origin
                rotation = part_from_visual[:3, :3]
                translation = part_from_visual[:3, 3]
                for mesh in visual.geometry.meshes:
                    vertices = np.asarray(mesh.vertices, dtype=np.float64)
                    vertices_by_visual.append(
                        vertices @ rotation.T + translation[None, :]
                    )
        if not vertices_by_visual:
            raise ValueError(
                f"Transferred target URDF has no visual vertices for part {part!r}."
            )
        vertices = np.concatenate(vertices_by_visual, axis=0)
        rng = np.random.RandomState(seed)
        indices = rng.choice(
            len(vertices), num_vertices, replace=num_vertices > len(vertices)
        )
        sampled[part] = vertices[indices]
    return sampled


def _link_local_transforms(
    urdf: URDF,
    articulation: np.ndarray,
) -> dict[str, np.ndarray]:
    movable = [joint for joint in urdf.joints if joint.joint_type != "fixed"]
    if len(movable) != 1:
        raise ValueError(
            f"DexMachina ARCTIC ADD expects one movable joint, got "
            f"{[joint.name for joint in movable]}."
        )
    values = np.asarray(articulation, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError(f"Expected articulation shape (T, 1), got {values.shape}.")
    fk = urdf.link_fk_batch(cfgs={movable[0].name: values[:, 0]})
    by_name = {link.name: np.asarray(transform) for link, transform in fk.items()}
    missing = [part for part in PART_NAMES if part not in by_name]
    if missing:
        raise ValueError(
            f"ARCTIC URDF is missing ADD part links {missing}; got {sorted(by_name)}."
        )
    return {part: by_name[part] for part in PART_NAMES}


def transform_part_vertices(
    *,
    urdf: URDF,
    part_vertices: dict[str, np.ndarray],
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
    articulation: np.ndarray,
) -> dict[str, np.ndarray]:
    position = np.asarray(position, dtype=np.float64)
    quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64)
    articulation = np.asarray(articulation, dtype=np.float64)
    n_frames = position.shape[0]
    if position.shape != (n_frames, 3):
        raise ValueError(f"Expected position shape (T, 3), got {position.shape}.")
    if quaternion_wxyz.shape != (n_frames, 4):
        raise ValueError(
            f"Expected quaternion shape {(n_frames, 4)}, got {quaternion_wxyz.shape}."
        )
    root_rotation = R.from_quat(np.roll(quaternion_wxyz, -1, axis=-1)).as_matrix()
    link_transform = _link_local_transforms(urdf, articulation)
    transformed = {}
    for part in PART_NAMES:
        local = link_transform[part]
        if local.shape != (n_frames, 4, 4):
            raise ValueError(
                f"Expected {part} FK shape {(n_frames, 4, 4)}, got {local.shape}."
            )
        part_rotation = np.einsum("tij,tjk->tik", root_rotation, local[:, :3, :3])
        part_position = position + np.einsum(
            "tij,tj->ti", root_rotation, local[:, :3, 3]
        )
        transformed[part] = (
            np.einsum("tij,vj->tvi", part_rotation, part_vertices[part])
            + part_position[:, None, :]
        )
    return transformed


def _rollout_chunk_paths(rollout_dir: Path) -> list[Path]:
    chunk_dir = (
        rollout_dir / "rollouts" if (rollout_dir / "rollouts").is_dir() else rollout_dir
    )
    paths = sorted(chunk_dir.glob("chunk_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No chunk_*.npz rollout files found under {chunk_dir}."
        )
    return paths


def load_rollout_episodes(
    rollout_dir: Path,
    *,
    canonical_object_index: int,
    num_scene_objects: int,
    expected_episode_length: int,
) -> list[dict[str, np.ndarray]]:
    if canonical_object_index < 0 or canonical_object_index >= num_scene_objects:
        raise ValueError(
            "Canonical object index must lie inside the scene object array, got "
            f"index {canonical_object_index} for {num_scene_objects} objects."
        )
    if expected_episode_length <= 0:
        raise ValueError(
            f"Expected positive episode length, got {expected_episode_length}."
        )
    episodes = []
    for path in _rollout_chunk_paths(rollout_dir):
        with np.load(path, allow_pickle=False) as data:
            required = (
                "done",
                "action_command_timestep",
                "command_timestep",
                "object_pos",
                "object_quat",
                "object_arti",
            )
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"Rollout chunk {path} is missing fields {missing}.")
            chunk = {key: np.asarray(data[key]) for key in required}
        done = chunk["done"]
        if done.ndim != 2:
            raise ValueError(f"Expected done shape (T, E) in {path}, got {done.shape}.")
        for env_index in range(done.shape[1]):
            done_indices = np.flatnonzero(done[:, env_index])
            length = int(done_indices[0] + 1) if done_indices.size else done.shape[0]
            if length != expected_episode_length:
                raise ValueError(
                    "DexMachina ADD evaluation requires the complete fixed horizon: "
                    f"expected {expected_episode_length} steps, got {length} for "
                    f"environment {env_index} in {path}."
                )
            position = chunk["object_pos"][:length, env_index]
            quaternion = chunk["object_quat"][:length, env_index]
            articulation = chunk["object_arti"][:length, env_index]
            if position.shape != (length, num_scene_objects, 3):
                raise ValueError(
                    "Expected rollout object positions for every scene object with "
                    f"shape {(length, num_scene_objects, 3)}, got {position.shape} "
                    f"in {path}."
                )
            if quaternion.shape != (length, num_scene_objects, 4):
                raise ValueError(
                    "Expected rollout object quaternions with shape "
                    f"{(length, num_scene_objects, 4)}, "
                    f"got {quaternion.shape} in {path}."
                )
            if articulation.ndim != 3 or articulation.shape[:2] != (
                length,
                num_scene_objects,
            ):
                raise ValueError(
                    "Expected rollout object articulation shape "
                    f"({length}, {num_scene_objects}, A), got "
                    f"{articulation.shape} in {path}."
                )
            episodes.append(
                {
                    "reference_timestep": chunk["action_command_timestep"][
                        :length, env_index
                    ],
                    "position": position[:, canonical_object_index],
                    "quaternion": quaternion[:, canonical_object_index],
                    "articulation": articulation[:, canonical_object_index],
                }
            )
    return episodes


def compute_episode_add(
    *,
    urdf: URDF,
    part_vertices: dict[str, np.ndarray],
    reference_motion: dict[str, np.ndarray],
    episode: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    command = np.asarray(episode["reference_timestep"], dtype=np.int64).reshape(-1)
    n_reference = len(reference_motion["position"])
    if np.any(command < 0) or np.any(command >= n_reference):
        raise ValueError(
            f"Command timesteps must be in [0, {n_reference - 1}], got {command}."
        )
    reference_vertices = transform_part_vertices(
        urdf=urdf,
        part_vertices=part_vertices,
        position=reference_motion["position"][command],
        quaternion_wxyz=reference_motion["quaternion"][command],
        articulation=reference_motion["articulation"][command],
    )
    rollout_vertices = transform_part_vertices(
        urdf=urdf,
        part_vertices=part_vertices,
        position=episode["position"],
        quaternion_wxyz=episode["quaternion"],
        articulation=episode["articulation"],
    )
    return {
        part: np.linalg.norm(
            reference_vertices[part] - rollout_vertices[part], axis=-1
        ).mean(axis=-1)
        for part in PART_NAMES
    }


def _bootstrap_ci(
    values: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected finite bootstrap values, got {values}.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarize_add(
    episode_distances: list[dict[str, np.ndarray]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    rows = []
    for episode_index, distances in enumerate(episode_distances):
        row: dict[str, float] = {"episode_index": float(episode_index)}
        for part in PART_NAMES:
            row[f"mean_add_m__{part}"] = float(np.mean(distances[part]))
            row[f"add_auc__{part}"] = compute_auc(
                distances[part], DEXMACHINA_THRESHOLDS_M
            )
        row["mean_add_m__overall"] = float(
            np.mean([row[f"mean_add_m__{part}"] for part in PART_NAMES])
        )
        row["add_auc__overall"] = float(
            np.mean([row[f"add_auc__{part}"] for part in PART_NAMES])
        )
        rows.append(row)

    summary: dict[str, object] = {
        "num_episodes": len(rows),
        "thresholds_m": DEXMACHINA_THRESHOLDS_M.tolist(),
    }
    for metric in (
        "mean_add_m__top",
        "mean_add_m__bottom",
        "mean_add_m__overall",
        "add_auc__top",
        "add_auc__bottom",
        "add_auc__overall",
    ):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        low, high = _bootstrap_ci(
            values, samples=bootstrap_samples, seed=bootstrap_seed
        )
        summary[metric] = {
            "mean": float(values.mean()),
            "ci95_low": low,
            "ci95_high": high,
        }
    pooled_auc = {
        part: compute_auc(
            np.concatenate([item[part] for item in episode_distances]),
            DEXMACHINA_THRESHOLDS_M,
        )
        for part in PART_NAMES
    }
    summary["dexmachina_pooled_add_auc"] = {
        **pooled_auc,
        "overall": float(np.mean(list(pooled_auc.values()))),
    }
    return rows, summary


def _write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    clip = get_clip(args.clip) if args.clip is not None else None
    transfer = get_transfer(args.transfer) if args.transfer is not None else None
    output_dir = args.output_dir or args.rollout_dir / "dexmachina_add_auc"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"ADD-AUC output directory is not empty: {output_dir}. Pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if clip is not None:
        sequence_path = materialize_runtime_sequence(clip, args.workspace)
        expected_episode_length = clip.frame_count
    else:
        if transfer is None:
            raise ValueError("Expected exactly one ARCTIC clip or transfer.")
        sequence_path = materialize_runtime_transfer_sequence(transfer, args.workspace)
        expected_episode_length = transfer.frame_count
    processor = OakinkProcessorSequence(sequence_path)
    reference = processor.retargeted_motion
    scene_object_ids = [str(value) for value in reference["scene_object_ids"]]
    canonical_object_id = str(
        np.asarray(reference["scene_canonical_object_id"], dtype=object).item()
    )
    if canonical_object_id not in scene_object_ids:
        raise ValueError(
            f"Canonical object {canonical_object_id!r} is not in scene objects "
            f"{scene_object_ids}."
        )
    canonical_object_index = scene_object_ids.index(canonical_object_id)
    reference_motion = {
        "position": np.asarray(reference["obj_pos"], dtype=np.float64),
        "quaternion": np.asarray(reference["obj_quat"], dtype=np.float64),
        "articulation": np.asarray(reference["obj_arti"], dtype=np.float64),
    }
    if clip is not None:
        object_urdf, _ = materialize_runtime_urdfs(clip, args.workspace)
    else:
        raw_sequence = np.load(sequence_path, allow_pickle=True)
        if not isinstance(raw_sequence, np.ndarray) or raw_sequence.shape != ():
            raise ValueError(
                f"Expected scalar transfer sequence payload at {sequence_path}, "
                f"got {type(raw_sequence)} with shape "
                f"{getattr(raw_sequence, 'shape', None)}."
            )
        sequence_payload = raw_sequence.item()
        if not isinstance(sequence_payload, dict):
            raise ValueError(
                f"Expected dict transfer payload at {sequence_path}, "
                f"got {type(sequence_payload)}."
            )
        if "target_urdf_path" not in sequence_payload:
            raise KeyError(
                f"Transfer sequence {sequence_path} is missing 'target_urdf_path'."
            )
        object_urdf = Path(str(sequence_payload["target_urdf_path"]))
        if not object_urdf.is_file():
            raise FileNotFoundError(
                f"Transferred target URDF does not exist: {object_urdf}."
            )
    urdf = URDF.load(str(object_urdf))
    part_vertices = (
        sample_part_vertices(
            clip,
            args.workspace,
            num_vertices=args.num_vertices_per_part,
            seed=args.vertex_seed,
        )
        if clip is not None
        else sample_urdf_part_vertices(
            urdf,
            num_vertices=args.num_vertices_per_part,
            seed=args.vertex_seed,
        )
    )
    episodes = load_rollout_episodes(
        args.rollout_dir,
        canonical_object_index=canonical_object_index,
        num_scene_objects=len(scene_object_ids),
        expected_episode_length=expected_episode_length,
    )
    distances = [
        compute_episode_add(
            urdf=urdf,
            part_vertices=part_vertices,
            reference_motion=reference_motion,
            episode=episode,
        )
        for episode in episodes
    ]
    rows, summary = summarize_add(
        distances,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary.update(
        {
            "protocol": "dexmachina_compute_add_v1",
            "clip": None if clip is None else clip.clip_id,
            "transfer": None if transfer is None else transfer.transfer_id,
            "source_clip": None if transfer is None else transfer.source_clip_id,
            "dexmachina_clip": None if clip is None else clip.dexmachina_clip,
            "dexmachina_reported_xhand_add_auc": (
                None if clip is None else clip.dexmachina_xhand_add_auc
            ),
            "dexmachina_reported_xhand_add_auc_sem": (
                None if clip is None else clip.dexmachina_xhand_add_auc_sem
            ),
            "num_vertices_per_part": args.num_vertices_per_part,
            "vertex_seed": args.vertex_seed,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "rollout_dir": str(args.rollout_dir),
            "sequence_path": str(sequence_path),
            "object_urdf": str(object_urdf),
            "scene_object_ids": scene_object_ids,
            "canonical_object_id": canonical_object_id,
            "reference_timestep_field": "action_command_timestep",
        }
    )
    _write_rows(output_dir / "episode_add_auc.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    flat = {
        f"episode_{episode_index:03d}__{part}": values
        for episode_index, episode in enumerate(distances)
        for part, values in episode.items()
    }
    np.savez_compressed(output_dir / "frame_add_distances.npz", **flat)
    print(
        "DexMachina-compatible pooled ADD-AUC: "
        f"{summary['dexmachina_pooled_add_auc']['overall']:.6f}"
    )
    print(f"Wrote ADD-AUC results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
