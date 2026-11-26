"""Prepare the included cup tasks for the native conditional multi-task learner."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .catalog import ROOT, sha256, repository_path


def prepare(output: Path) -> None:
    import setuptools  # noqa: F401 - initialize distutils before Torch/Triton
    from dexlink.multitask_cup.full_method_config import FullMethodRunConfig
    from dexlink.multitask_cup.task_data import build_bank

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    template = json.loads(
        (ROOT / "tasks/multitask_cup/manifest.template.json").read_text()
    )
    urdfs = {repository_path(template["robot_urdf"])}
    template["robot_urdf"] = str(repository_path(template["robot_urdf"]))
    template["assets"] = {}
    for row in template["variants"]:
        original = np.load(repository_path(row["reference"]), allow_pickle=True).item()
        payload = original.copy()
        payload["scene_object_urdf_paths"] = np.array(
            [str(repository_path(uri)) for uri in original["scene_object_urdf_paths"]]
        )
        for field in ("target_obj_path", "reference_object_urdf", "target_object_urdf"):
            if field in payload and isinstance(payload[field], str) and payload[field]:
                payload[field] = str(repository_path(payload[field]))
        destination = output / "references" / (row["id"] + ".npy")
        destination.parent.mkdir(exist_ok=True)
        np.save(destination, payload, allow_pickle=True)
        row["reference"] = str(destination)
        row["reference_sha256"] = sha256(destination)
        row["scene_urdfs"] = [str(repository_path(uri)) for uri in row["scene_urdfs"]]
        urdfs.update(Path(uri) for uri in row["scene_urdfs"])
    meshes = set()
    for path in urdfs:
        dependencies = {}
        for mesh in ET.parse(path).findall(".//mesh"):
            dependency = (path.parent / mesh.attrib["filename"]).resolve(strict=True)
            meshes.add(dependency)
            dependencies[str(dependency)] = sha256(dependency)
        template["assets"][str(path)] = dict(
            export_urdf=str(path),
            export_urdf_sha256=sha256(path),
            source_dependencies=dependencies,
        )
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(template, indent=2) + "\n")
    bank = build_bank(manifest, output / "task-bank.pt")
    (output / "bank-validation.json").write_text(json.dumps(bank, indent=2) + "\n")
    asset_manifest = output / "multitask-assets.json"
    asset_manifest.write_text(
        json.dumps(
            {
                "urdfs": {str(path): sha256(path) for path in sorted(urdfs)},
                "meshes": {str(path): sha256(path) for path in sorted(meshes)},
            },
            indent=2,
        )
        + "\n"
    )
    files = {}
    for directory in ("dexlink",):
        for path in (ROOT / directory).rglob("*.py"):
            files[str(path.relative_to(ROOT))] = sha256(path)
    (output / "source-manifest.json").write_text(
        json.dumps(
            {"source_files": files, "asset_manifest_sha256": sha256(asset_manifest)},
            indent=2,
        )
        + "\n"
    )
    config = FullMethodRunConfig(
        data=str(output),
        run_directory=str(ROOT / "outputs/multitask/cup30-seed42"),
        run_id="cup30-seed42",
        task_indices=tuple(range(template["case_count"])),
    )
    (output / "training-config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )
    print(json.dumps(bank))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
