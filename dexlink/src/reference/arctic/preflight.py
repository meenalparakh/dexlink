from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from urdfpy import URDF

from dexlink.src.retargeting.data.oakink_processor_sequence import OakinkProcessorSequence
from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE, get_clip
from dexlink.src.reference.arctic.runtime import materialize_runtime_sequence


@dataclass(frozen=True)
class PreflightArgs:
    clip: str
    workspace: Path


def parse_args(argv: list[str] | None = None) -> PreflightArgs:
    parser = argparse.ArgumentParser(
        description="Validate cluster-readable ARCTIC data, runtime paths, and packages."
    )
    parser.add_argument("--clip", required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    raw = parser.parse_args(argv)
    return PreflightArgs(clip=str(raw.clip), workspace=raw.workspace.resolve())


def _npy_header(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Processed ARCTIC sequence does not exist: {path}.")
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in ((2, 0), (3, 0)):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported NPY version {version} in {path}.")
    if shape != () or not dtype.hasobject:
        raise ValueError(
            f"Expected scalar object NPY for processed ARCTIC data, got "
            f"shape={shape}, dtype={dtype} in {path}."
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "npy_version": list(version),
        "shape": list(shape),
        "dtype": str(dtype),
        "fortran_order": bool(fortran_order),
    }


def _package_versions() -> dict[str, str]:
    versions = {}
    required_packages = {
        "numpy": "numpy",
        "torch": "torch",
        "scipy": "scipy",
        "trimesh": "trimesh",
        "urdfpy": "urdfpy",
        "genesis-world": "genesis",
    }
    for distribution, module in required_packages.items():
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Required cluster package {distribution!r} is not installed."
            ) from exc
        importlib.import_module(module)
    return versions


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    clip = get_clip(args.clip)
    processed_header = _npy_header(clip.processed_demo_path())
    runtime_sequence = materialize_runtime_sequence(clip, args.workspace)
    processor = OakinkProcessorSequence(runtime_sequence)
    reference = processor.retargeted_motion
    scene_urdfs = [Path(path) for path in reference["scene_object_urdf_paths"]]
    if len(scene_urdfs) != 2:
        raise ValueError(
            f"Expected two ARCTIC scene URDFs, got {[str(path) for path in scene_urdfs]}."
        )
    urdf_summaries = []
    for path in scene_urdfs:
        if not path.is_file() or not path.is_absolute():
            raise FileNotFoundError(
                f"Expected an absolute, readable runtime URDF path, got {path}."
            )
        urdf = URDF.load(str(path))
        urdf_summaries.append(
            {
                "path": str(path),
                "links": [link.name for link in urdf.links],
                "joints": [joint.name for joint in urdf.joints],
            }
        )
    result = {
        "status": "ok",
        "clip": clip.clip_id,
        "workspace": str(args.workspace),
        "packages": _package_versions(),
        "processed_sequence": processed_header,
        "runtime_sequence": str(runtime_sequence),
        "reference_frames": int(np.asarray(reference["obj_pos"]).shape[0]),
        "scene_urdfs": urdf_summaries,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
