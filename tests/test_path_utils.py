from __future__ import annotations

from pathlib import Path

import pytest

from dexlink.utils.retargeting.path_utils import (
    DEXLINK_DATA_ENV,
    DEXLINK_OUTPUTS_ENV,
    DEXLINK_ROOT_ENV,
    get_external_data_dir,
    get_logs_dir,
    get_src,
    process_urdf,
    resolve_project_path,
)


def test_path_roots_are_explicitly_configurable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = tmp_path / "checkout"
    (repository / "dexlink").mkdir(parents=True)
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    monkeypatch.setenv(DEXLINK_ROOT_ENV, str(repository))
    monkeypatch.setenv(DEXLINK_DATA_ENV, str(data))
    monkeypatch.setenv(DEXLINK_OUTPUTS_ENV, str(outputs))

    assert get_src() == repository.resolve()
    assert get_external_data_dir() == data.resolve()
    assert get_logs_dir() == outputs.resolve()
    assert resolve_project_path("external_data/oakink/demo.pkl") == (
        data / "oakink" / "demo.pkl"
    ).resolve()
    assert resolve_project_path("outputs/run/checkpoint.pt") == (
        outputs / "run" / "checkpoint.pt"
    ).resolve()
    assert resolve_project_path("assets/robot.urdf") == (
        repository / "assets" / "robot.urdf"
    ).resolve()


def test_process_urdf_materializes_relative_resources_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = tmp_path / "checkout"
    (repository / "dexlink").mkdir(parents=True)
    outputs = tmp_path / "outputs"
    model_dir = repository / "models"
    model_dir.mkdir()
    mesh_path = model_dir / "cube.obj"
    mesh_path.write_text("v 0 0 0\n", encoding="utf-8")
    urdf_path = model_dir / "cube.urdf"
    urdf_path.write_text(
        '<robot name="cube"><link name="base"><visual><geometry>'
        '<mesh filename="cube.obj"/></geometry></visual></link></robot>',
        encoding="utf-8",
    )
    monkeypatch.setenv(DEXLINK_ROOT_ENV, str(repository))
    monkeypatch.setenv(DEXLINK_OUTPUTS_ENV, str(outputs))

    processed = process_urdf(urdf_path)
    assert processed.is_file()
    rendered = processed.read_text(encoding="utf-8")
    assert f'filename="{mesh_path.resolve()}"' in rendered


def test_process_urdf_rejects_missing_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = tmp_path / "checkout"
    (repository / "dexlink").mkdir(parents=True)
    model_dir = repository / "models"
    model_dir.mkdir()
    urdf_path = model_dir / "broken.urdf"
    urdf_path.write_text(
        '<robot name="broken"><link name="base"><visual><geometry>'
        '<mesh filename="missing.obj"/></geometry></visual></link></robot>',
        encoding="utf-8",
    )
    monkeypatch.setenv(DEXLINK_ROOT_ENV, str(repository))

    with pytest.raises(FileNotFoundError, match="missing.obj"):
        process_urdf(urdf_path)


def test_process_urdf_rejects_non_urdf_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = tmp_path / "checkout"
    (repository / "dexlink").mkdir(parents=True)
    mesh_path = repository / "mesh.obj"
    mesh_path.write_text("v 0 0 0\n", encoding="utf-8")
    monkeypatch.setenv(DEXLINK_ROOT_ENV, str(repository))

    with pytest.raises(ValueError, match=r"Expected a \.urdf file"):
        process_urdf(mesh_path)
