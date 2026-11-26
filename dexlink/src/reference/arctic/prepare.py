from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from dexlink.utils.retargeting.urdf_utils import create_urdf_from_obj
from dexlink.src.reference.arctic.config import (
    DEFAULT_ARCTIC_ROOT,
    DEFAULT_DEXMACHINA_ROOT,
    DEFAULT_WORKSPACE,
    ArcticClipSpec,
    resolve_clips,
)
from dexlink.src.reference.arctic.geometry import write_box_obj

RAW_SUFFIXES = ("mano.npy", "object.npy", "smplx.npy", "egocam.dist.npy")


@dataclass(frozen=True)
class PrepareArgs:
    clips: tuple[str, ...]
    all_clips: bool
    workspace: Path
    dexmachina_root: Path
    arctic_root: Path
    overwrite: bool


def parse_args(argv: list[str] | None = None) -> PrepareArgs:
    parser = argparse.ArgumentParser(
        description="Prepare exact DexMachina ARCTIC clips in an isolated workspace."
    )
    parser.add_argument("--clip", action="append", default=[])
    parser.add_argument("--all-clips", action="store_true")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--dexmachina-root", type=Path, default=DEFAULT_DEXMACHINA_ROOT)
    parser.add_argument("--arctic-root", type=Path, default=DEFAULT_ARCTIC_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    raw = parser.parse_args(argv)
    return PrepareArgs(
        clips=tuple(raw.clip),
        all_clips=bool(raw.all_clips),
        workspace=raw.workspace.resolve(),
        dexmachina_root=raw.dexmachina_root.resolve(),
        arctic_root=raw.arctic_root.resolve(),
        overwrite=bool(raw.overwrite),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path, *, overwrite: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required source file does not exist: {source}")
    if destination.exists() and source.samefile(destination):
        return
    if destination.exists():
        if _sha256(source) == _sha256(destination):
            return
        if not overwrite:
            raise FileExistsError(
                f"Destination differs from source: {destination}. Pass --overwrite "
                "to replace it."
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_asset_dir(source: Path, destination: Path, *, overwrite: bool) -> None:
    if not source.is_dir():
        raise FileNotFoundError(
            f"DexMachina object asset directory is missing: {source}"
        )
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Asset destination already exists: {destination}. Pass --overwrite "
                "to replace it."
            )
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _materialize_absolute_urdf(clip: ArcticClipSpec, workspace: Path) -> Path:
    decomp_dir = clip.object_asset_dir(workspace) / "decomp"
    source = decomp_dir / f"{clip.object_name}_decomp.urdf"
    destination = clip.object_urdf_path(workspace)
    if not source.is_file():
        raise FileNotFoundError(f"Copied DexMachina URDF is missing: {source}")
    tree = ET.parse(source)
    mesh_elements = tree.getroot().findall(".//mesh")
    if not mesh_elements:
        raise ValueError(f"Expected at least one mesh in {source}.")
    for mesh in mesh_elements:
        filename = mesh.attrib.get("filename")
        if not filename:
            raise ValueError(f"Mesh element in {source} is missing filename.")
        mesh_path = (decomp_dir / filename).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"URDF mesh {filename!r} from {source} does not exist at {mesh_path}."
            )
        mesh.attrib["filename"] = str(mesh_path)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def materialize_support_block(clip: ArcticClipSpec, workspace: Path) -> Path:
    asset_dir = clip.support_block_asset_dir(workspace)
    obj_path = write_box_obj(asset_dir / "support_block.obj", clip.support_block_size)
    urdf_path = Path(
        create_urdf_from_obj(
            obj_path=str(obj_path.resolve()),
            save_dir=str(asset_dir),
            scale=1.0,
        )
    ).resolve()
    expected_path = clip.support_block_urdf_path(workspace).resolve()
    if urdf_path != expected_path or not urdf_path.is_file():
        raise FileNotFoundError(
            f"Expected materialized support URDF at {expected_path}, got {urdf_path}."
        )
    return urdf_path


def _array_frame_count(path: Path) -> int:
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if not isinstance(raw, dict):
        raise ValueError(f"Expected processed ARCTIC dict at {path}, got {type(raw)}.")
    world_coord = raw.get("world_coord")
    if not isinstance(world_coord, dict) or "joints.left" not in world_coord:
        raise ValueError(
            f"Processed ARCTIC file {path} is missing world_coord['joints.left']."
        )
    joints = np.asarray(world_coord["joints.left"])
    if joints.ndim != 3 or joints.shape[1:] != (21, 3):
        raise ValueError(
            f"Expected left joints shape (T, 21, 3) in {path}, got {joints.shape}."
        )
    return int(joints.shape[0])


def _validate_raw_sequence(clip: ArcticClipSpec, arctic_root: Path) -> list[Path]:
    raw_dir = arctic_root / "data" / "arctic_data" / "data" / "raw_seqs" / clip.subject
    paths = [raw_dir / f"{clip.sequence_name}.{suffix}" for suffix in RAW_SUFFIXES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Raw ARCTIC inputs for {clip.clip_id!r} are missing: {missing}."
        )
    return paths


def _git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def prepare_clip(clip: ArcticClipSpec, args: PrepareArgs) -> dict[str, object]:
    raw_paths = _validate_raw_sequence(clip, args.arctic_root)
    dex_assets = args.dexmachina_root / "dexmachina" / "assets" / "arctic"
    source_demo = (
        args.arctic_root
        / "outputs"
        / "processed_verts"
        / "seqs"
        / clip.subject
        / f"{clip.sequence_name}.npy"
    )
    if not source_demo.is_file():
        mano_path = raw_paths[0]
        raise FileNotFoundError(
            f"Official ARCTIC processed-vertices export is missing: {source_demo}. "
            "Generate it from the official ARCTIC checkout with:\n"
            f"  cd {args.arctic_root} && python scripts_data/process_seqs.py "
            f"--mano_p {mano_path} --export_verts"
        )
    frame_count = _array_frame_count(source_demo)
    if clip.end_frame > frame_count:
        raise ValueError(
            f"Clip {clip.clip_id!r} ends at frame {clip.end_frame}, but {source_demo} "
            f"contains only {frame_count} frames."
        )

    destination_demo = clip.processed_demo_path()
    destination_asset_dir = clip.object_asset_dir(args.workspace)
    _copy_file(source_demo, destination_demo, overwrite=args.overwrite)
    if not destination_asset_dir.exists() or args.overwrite:
        _copy_asset_dir(
            dex_assets / clip.object_name,
            destination_asset_dir,
            overwrite=args.overwrite,
        )
    urdf_path = _materialize_absolute_urdf(clip, args.workspace)
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Copied object URDF is missing: {urdf_path}")
    support_block_urdf = materialize_support_block(clip, args.workspace)

    return {
        **asdict(clip),
        "frame_count": clip.frame_count,
        "source_processed_demo": str(source_demo),
        "processed_demo": str(destination_demo),
        "processed_demo_sha256": _sha256(destination_demo),
        "object_urdf": str(urdf_path),
        "support_block": {
            "source": "DexMachina envs/base_env.py CARDBOARD_POS/cardbox_size",
            "position": list(clip.support_block_pos),
            "size": list(clip.support_block_size),
            "friction": clip.support_block_friction,
            "urdf": str(support_block_urdf),
        },
        "raw_inputs": [str(path) for path in raw_paths],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    clips = resolve_clips(args.clips, args.all_clips)
    records = [prepare_clip(clip, args) for clip in clips]
    manifest = {
        "protocol": "dexmachina_arctic_main_v1",
        "interval_convention": "start_inclusive_end_exclusive",
        "dexmachina_root": str(args.dexmachina_root),
        "dexmachina_commit": _git_commit(args.dexmachina_root),
        "arctic_root": str(args.arctic_root),
        "workspace": str(args.workspace),
        "clips": records,
    }
    manifest_path = args.workspace / "prepared_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(records)} clip(s); wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
