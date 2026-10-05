#!/usr/bin/env python

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConvertGlbConfig:
    input_glb: Path
    output_dir: Path | None = None
    force: bool = False

    @property
    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        return self.input_glb.with_suffix("")

    @property
    def resolved_output_obj(self) -> Path:
        return self.resolved_output_dir / "model.obj"


def _load_scene(input_glb: Path):
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "Converting GLB/GLTF to OBJ requires `trimesh`. Install it in the "
            "active environment before running this script."
        ) from exc

    loaded = trimesh.load(str(input_glb), force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene(loaded)
    elif isinstance(loaded, trimesh.Scene):
        scene = loaded
    else:
        raise TypeError(
            f"Expected trimesh to load {input_glb} as Trimesh or Scene, got {type(loaded)}."
        )

    geometry_count = len(scene.geometry)
    if geometry_count == 0:
        raise ValueError(f"GLB contains no geometry: {input_glb}")
    return scene


def _write_export_payload(payload: object, output_obj: Path) -> None:
    output_dir = output_obj.parent
    if isinstance(payload, dict):
        obj_entries = [
            name for name in payload if Path(str(name)).suffix.lower() == ".obj"
        ]
        if len(obj_entries) != 1:
            raise ValueError(
                "Expected trimesh OBJ export to contain exactly one .obj entry, "
                f"got {obj_entries}."
            )
        obj_entry = obj_entries[0]
        for raw_name, raw_content in payload.items():
            filename = (
                "model.obj" if raw_name == obj_entry else Path(str(raw_name)).name
            )
            _write_payload_file(output_dir / filename, raw_content)
        return

    _write_payload_file(output_obj, payload)


def _write_payload_file(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
        return
    if isinstance(content, bytes):
        path.write_bytes(content)
        return
    raise TypeError(
        f"Expected exported file payload for {path} to be str or bytes, got {type(content)}."
    )


def convert_glb_to_obj(config: ConvertGlbConfig) -> Path:
    input_glb = config.input_glb.resolve()
    if not input_glb.is_file():
        raise FileNotFoundError(f"Input GLB/GLTF does not exist: {input_glb}")
    if input_glb.suffix.lower() not in {".glb", ".gltf"}:
        raise ValueError(f"Expected a .glb or .gltf input path, got: {input_glb}")

    output_dir = config.resolved_output_dir.resolve()
    output_obj = output_dir / "model.obj"
    if output_dir.exists():
        if not config.force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Pass --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = _load_scene(input_glb)
    payload = scene.export(file_type="obj")
    _write_export_payload(payload, output_obj)

    if not output_obj.is_file():
        raise FileNotFoundError(f"Expected OBJ was not written: {output_obj}")
    print(f"Input GLB/GLTF: {input_glb}")
    print(f"Output directory: {output_dir}")
    print(f"Output OBJ: {output_obj}")
    print(f"Scene geometries: {len(scene.geometry)}")
    return output_obj


def parse_args(argv: list[str] | None = None) -> ConvertGlbConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a GLB/GLTF scene to OBJ. By default, /path/to/name.glb is "
            "written to /path/to/name/model.obj, with any material or texture "
            "sidecars emitted into the same /path/to/name directory."
        )
    )
    parser.add_argument("input_glb", type=Path)
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Optional output directory. Defaults to the input path without suffix.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return ConvertGlbConfig(
        input_glb=args.input_glb,
        output_dir=args.output_dir,
        force=args.force,
    )


def main(argv: list[str] | None = None) -> int:
    convert_glb_to_obj(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
