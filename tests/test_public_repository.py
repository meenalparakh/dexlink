from __future__ import annotations

from pathlib import Path
import subprocess

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_source_has_no_private_machine_or_cluster_paths() -> None:
    forbidden = (
        "/" + "home/",
        "/" + "mnt/",
        "/" + "n/fs/",
        "pvl" + "-exptrack",
        "mp" + "8772",
    )
    inspected_suffixes = {".py", ".sh"}
    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in inspected_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"Found private token {token!r} in {path}"


def test_external_data_contains_no_redistributed_content() -> None:
    external_root = REPOSITORY_ROOT / "external_data"
    allowed_names = {".gitkeep", "README.md"}
    unexpected = [
        path.relative_to(external_root)
        for path in external_root.rglob("*")
        if path.is_file()
        and path.name not in allowed_names
        and subprocess.run(
            ["git", "check-ignore", "--quiet", str(path)],
            cwd=REPOSITORY_ROOT,
            check=False,
        ).returncode
        != 0
    ]
    assert unexpected == []


def test_mano_files_are_not_committed() -> None:
    forbidden_names = {"MANO_LEFT.pkl", "MANO_RIGHT.pkl"}
    committed = {
        path.name for path in REPOSITORY_ROOT.rglob("*") if path.is_file()
    }
    assert committed.isdisjoint(forbidden_names)


def test_every_asset_group_has_explicit_provenance() -> None:
    assets_root = REPOSITORY_ROOT / "assets"
    inventory = yaml.safe_load(
        (assets_root / "PROVENANCE.yaml").read_text(encoding="utf-8")
    )
    declared_paths = {entry["path"] for entry in inventory["asset_groups"]}
    assert len(declared_paths) == len(inventory["asset_groups"])

    actual_paths = {
        path.name
        for path in assets_root.iterdir()
        if path.is_dir() and path.name != "robots"
    }
    actual_paths.update(
        f"robots/{path.name}"
        for path in (assets_root / "robots").iterdir()
        if path.is_dir()
    )
    assert declared_paths == actual_paths

    xhand = next(
        entry for entry in inventory["asset_groups"] if entry["path"] == "robots/xhand"
    )
    assert xhand["license"] == "MIT"
    assert xhand["source_commit"] == "adae5bf620c57723d185b2757ee3ce9656927c20"
    assert (assets_root / xhand["license_file"]).is_file()
    assert (assets_root / xhand["notice_file"]).is_file()


def test_redistributed_xhand_has_dexmachina_license_and_resolved_meshes() -> None:
    xhand_root = REPOSITORY_ROOT / "assets" / "robots" / "xhand"
    license_text = (xhand_root / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Copyright (c) 2025 Mandi Zhao" in license_text
    notice_text = (xhand_root / "NOTICE.md").read_text(encoding="utf-8")
    assert "MandiZhao/dexmachina" in notice_text
    assert "adae5bf620c57723d185b2757ee3ce9656927c20" in notice_text
    assert "68 files" in notice_text

    for urdf_path in xhand_root.glob("*.urdf"):
        text = urdf_path.read_text(encoding="utf-8")
        for raw_reference in text.split('filename="')[1:]:
            mesh_reference = raw_reference.split('"', 1)[0]
            if mesh_reference.startswith("package://"):
                continue
            assert (urdf_path.parent / mesh_reference).resolve().is_file(), (
                urdf_path,
                mesh_reference,
            )


def test_adapted_pytorch3d_helpers_preserve_bsd_attribution() -> None:
    license_path = REPOSITORY_ROOT / "third_party_licenses/PYTORCH3D-BSD.txt"
    license_text = license_path.read_text(encoding="utf-8")
    assert "Copyright (c) Meta Platforms, Inc. and affiliates" in license_text
    assert "Redistribution and use in source and binary forms" in license_text

    helper_path = REPOSITORY_ROOT / "dexlink/utils/retargeting/math_utils.py"
    helper_text = helper_path.read_text(encoding="utf-8")
    assert "third_party_licenses/PYTORCH3D-BSD.txt" in helper_text

    third_party_text = (REPOSITORY_ROOT / "THIRD_PARTY.md").read_text(
        encoding="utf-8"
    )
    assert "third_party_licenses/PYTORCH3D-BSD.txt" in third_party_text


def test_only_xhand_configuration_uses_bundled_robot_assets() -> None:
    config_root = REPOSITORY_ROOT / "dexlink/src/retargeting/data/hand_cfgs"
    for config_path in config_root.glob("*.py"):
        text = config_path.read_text(encoding="utf-8")
        if config_path.name == "xhand.py":
            assert 'get_assets_dir() / "robots"' in text
        else:
            assert 'get_assets_dir() / "robots"' not in text


def test_all_declared_repository_task_assets_exist() -> None:
    for task_path in (REPOSITORY_ROOT / "tasks").rglob("*.yaml"):
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        stack = [task]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in {"mesh_path", "urdf_path"} and isinstance(child, str):
                        path = Path(child)
                        if path.parts and path.parts[0] == "assets":
                            assert (REPOSITORY_ROOT / path).exists(), (
                                task_path,
                                child,
                            )
                    stack.append(child)
            elif isinstance(value, list):
                stack.extend(value)
