from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dexlink.src.retargeting.data.process_arctic import ArcticProcessor, class_dict
from dexlink.utils.retargeting.path_utils import get_job_process_name, get_tmp_dir, process_urdf
from dexlink.src.reference.arctic.config import ArcticClipSpec
from dexlink.src.reference.arctic.transfer_config import ArcticTransferSpec


def processor_key(clip: ArcticClipSpec) -> str:
    return f"arctic_{clip.object_name}"


def _write_runtime_urdf(
    *,
    source: Path,
    destination: Path,
    expected_mesh_path: Path | None = None,
) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"ARCTIC source URDF does not exist: {source}.")
    tree = ET.parse(source)
    mesh_elements = tree.getroot().findall(".//mesh")
    if not mesh_elements:
        raise ValueError(f"Expected at least one mesh in ARCTIC URDF {source}.")
    for mesh in mesh_elements:
        filename = mesh.attrib.get("filename")
        if not filename:
            raise ValueError(f"Mesh element in {source} is missing filename.")
        if expected_mesh_path is None:
            raw_mesh_path = Path(filename)
            if raw_mesh_path.is_absolute():
                raise ValueError(
                    f"Expected portable relative mesh paths in {source}, got {filename!r}."
                )
            mesh_path = (source.parent / raw_mesh_path).resolve()
        else:
            if Path(filename).name != expected_mesh_path.name:
                raise ValueError(
                    f"Expected support mesh {expected_mesh_path.name!r} in {source}, "
                    f"got {filename!r}."
                )
            mesh_path = expected_mesh_path.resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"ARCTIC URDF mesh {filename!r} does not exist at {mesh_path}."
            )
        mesh.attrib["filename"] = str(mesh_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def materialize_runtime_urdfs(
    clip: ArcticClipSpec, workspace: Path
) -> tuple[Path, Path]:
    """Recompute absolute mesh paths in process-unique URDF copies."""
    workspace = workspace.resolve()
    job_name = get_job_process_name()
    runtime_dir = get_tmp_dir() / "arctic_evals" / clip.clip_id / job_name
    object_source = (
        clip.object_asset_dir(workspace) / "decomp" / f"{clip.object_name}_decomp.urdf"
    )
    object_output = runtime_dir / (f"{clip.object_name}_decomp_runtime_{job_name}.urdf")
    support_source = clip.support_block_urdf_path(workspace)
    support_mesh = clip.support_block_asset_dir(workspace) / "support_block.obj"
    support_output = runtime_dir / f"support_block_runtime_{job_name}.urdf"
    return (
        _write_runtime_urdf(source=object_source, destination=object_output),
        _write_runtime_urdf(
            source=support_source,
            destination=support_output,
            expected_mesh_path=support_mesh,
        ),
    )


def materialize_runtime_sequence(clip: ArcticClipSpec, workspace: Path) -> Path:
    """Create a process-scoped RL payload containing current-root URDF paths."""
    source = clip.exported_sequence_path(workspace.resolve())
    if not source.is_file():
        raise FileNotFoundError(
            f"ARCTIC RL sequence is missing: {source}. Run assembly first."
        )
    raw = np.load(source, allow_pickle=True)
    if not isinstance(raw, np.ndarray) or raw.shape != ():
        raise ValueError(
            f"Expected scalar ARCTIC sequence payload at {source}, got "
            f"{type(raw)} with shape {getattr(raw, 'shape', None)}."
        )
    payload = raw.item()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload at {source}, got {type(payload)}.")
    payload_clip = str(np.asarray(payload.get("arctic_clip_id"), dtype=object).item())
    if payload_clip != clip.clip_id:
        raise ValueError(
            f"Expected ARCTIC clip id {clip.clip_id!r} in {source}, got {payload_clip!r}."
        )
    scene_ids = np.asarray(payload.get("scene_object_ids"), dtype=object)
    expected_scene_ids = np.asarray(["object", "arctic_support_block"], dtype=object)
    if scene_ids.shape != (2,) or not np.array_equal(scene_ids, expected_scene_ids):
        raise ValueError(
            f"Expected scene ids {expected_scene_ids.tolist()} in {source}, "
            f"got shape={scene_ids.shape}, values={scene_ids.tolist()}."
        )

    object_urdf, support_urdf = materialize_runtime_urdfs(clip, workspace)
    runtime_payload = payload.copy()
    for key in (
        "original_urdf_path",
        "target_urdf_path",
        "target_articulated_urdf_path",
    ):
        if key not in runtime_payload:
            raise KeyError(
                f"ARCTIC sequence {source} is missing required path {key!r}."
            )
        runtime_payload[key] = str(object_urdf)
    runtime_payload["scene_object_urdf_paths"] = np.asarray(
        [str(object_urdf), str(support_urdf)], dtype=object
    )
    if "arctic_demo_path" not in runtime_payload:
        raise KeyError(
            f"ARCTIC sequence {source} is missing required path 'arctic_demo_path'."
        )
    runtime_payload["arctic_demo_path"] = np.array(
        str(clip.processed_demo_path()), dtype=object
    )

    output_dir = get_tmp_dir() / "arctic_evals" / clip.clip_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"sequence_runtime_{get_job_process_name()}.npy"
    np.save(output, runtime_payload, allow_pickle=True)
    return output


def materialize_runtime_transfer_sequence(
    transfer: ArcticTransferSpec, workspace: Path
) -> Path:
    """Create a process-scoped sequence for an assembled geometry transfer."""
    workspace = workspace.resolve()
    source = (
        workspace
        / "transfer_trajectories"
        / transfer.transfer_id
        / "exports"
        / f"{transfer.transfer_id}_oakink_sequence.npy"
    )
    if not source.is_file():
        raise FileNotFoundError(
            f"ARCTIC transfer RL sequence is missing: {source}. Run transfer assembly first."
        )
    raw = np.load(source, allow_pickle=True)
    if not isinstance(raw, np.ndarray) or raw.shape != ():
        raise ValueError(
            f"Expected scalar ARCTIC transfer payload at {source}, got "
            f"{type(raw)} with shape {getattr(raw, 'shape', None)}."
        )
    payload = raw.item()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload at {source}, got {type(payload)}.")
    payload_transfer = str(
        np.asarray(payload.get("arctic_transfer_id"), dtype=object).item()
    )
    if payload_transfer != transfer.transfer_id:
        raise ValueError(
            f"Expected transfer id {transfer.transfer_id!r} in {source}, "
            f"got {payload_transfer!r}."
        )
    scene_ids = np.asarray(payload.get("scene_object_ids"), dtype=object)
    expected_scene_ids = np.asarray(["object", "arctic_support_block"], dtype=object)
    if scene_ids.shape != (2,) or not np.array_equal(scene_ids, expected_scene_ids):
        raise ValueError(
            f"Expected scene ids {expected_scene_ids.tolist()} in {source}, "
            f"got shape={scene_ids.shape}, values={scene_ids.tolist()}."
        )

    required_path_keys = (
        "original_urdf_path",
        "target_urdf_path",
        "target_articulated_urdf_path",
    )
    missing_path_keys = [key for key in required_path_keys if key not in payload]
    if missing_path_keys:
        raise KeyError(
            f"ARCTIC transfer sequence {source} is missing required paths "
            f"{missing_path_keys}."
        )

    target_path = Path(str(payload["target_urdf_path"]))
    support_path = transfer.support_block_urdf_path(workspace)
    if not support_path.is_file():
        raise FileNotFoundError(f"Transfer support URDF does not exist: {support_path}.")
    object_urdf = process_urdf(target_path, append_jobname=True)
    support_urdf = process_urdf(support_path, append_jobname=True)

    runtime_payload = payload.copy()
    for key in required_path_keys:
        runtime_payload[key] = str(object_urdf)
    runtime_payload["scene_object_urdf_paths"] = np.asarray(
        [str(object_urdf), str(support_urdf)], dtype=object
    )
    runtime_payload["arctic_demo_path"] = np.array(
        str(transfer.source_clip.processed_demo_path()), dtype=object
    )

    output_dir = get_tmp_dir() / "arctic_evals" / transfer.transfer_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"sequence_runtime_{get_job_process_name()}.npy"
    np.save(output, runtime_payload, allow_pickle=True)
    return output


def register_processor(clip: ArcticClipSpec, workspace: Path) -> str:
    """Point the legacy ARCTIC processor at the isolated benchmark asset copy."""
    urdf_path, _ = materialize_runtime_urdfs(clip, workspace)
    key = processor_key(clip)
    class_dict[key] = {
        "processor_cls": ArcticProcessor,
        "reference_urdf": str(urdf_path),
        "target_urdf_path": str(urdf_path),
    }
    return key
