"""Resource installation must validate archives and preserve existing user data."""

from pathlib import Path
import hashlib
import json
import zipfile

import pytest

from dexlink.workflows import resources


def manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, members: dict[str, bytes]
) -> Path:
    archive = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, content in members.items():
            bundle.writestr(name, content)
    root = tmp_path / "repo"
    (root / "resources").mkdir(parents=True)
    document = {
        "schema": "dexlink_resources_v1",
        "assets": {
            "sha256": resources.sha256(archive),
            "files": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in members.items()
            },
        },
    }
    (root / "resources/manifest.json").write_text(json.dumps(document))
    monkeypatch.setattr(resources, "ROOT", root)
    monkeypatch.setenv("DEXLINK_ROOT", str(root))
    (root / "dexlink").mkdir()
    monkeypatch.setenv("DEXLINK_OUTPUTS", str(tmp_path / "outputs"))
    return archive


def test_install_honors_explicit_external_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = manifest(
        tmp_path, monkeypatch, {"external_data/sam3d/cup/model.obj": b"mesh"}
    )
    destination = tmp_path / "external"
    monkeypatch.setenv("DEXLINK_EXTERNAL_DATA", str(destination))
    resources.install("assets", archive)
    assert (destination / "sam3d/cup/model.obj").read_bytes() == b"mesh"
    assert not (resources.ROOT / "external_data").exists()
    resources.install("assets", archive)  # Identical content is safe to verify again.


def test_install_refuses_to_replace_different_existing_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = manifest(tmp_path, monkeypatch, {"assets/fixture.obj": b"new"})
    destination = resources.ROOT / "assets/fixture.obj"
    destination.parent.mkdir()
    destination.write_bytes(b"user data")
    with pytest.raises(FileExistsError, match="different existing content"):
        resources.install("assets", archive)
    assert destination.read_bytes() == b"user data"


def test_install_rejects_escaping_archive_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = manifest(
        tmp_path, monkeypatch, {"assets/fixture.obj": b"safe", "../escaped": b"bad"}
    )
    with pytest.raises(ValueError, match="Unsafe archive member"):
        resources.install("assets", archive)
    assert not (resources.ROOT / "assets").exists()
    assert not (tmp_path / "escaped").exists()


def test_install_rejects_archive_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = manifest(tmp_path, monkeypatch, {"assets/fixture.obj": b"safe"})
    with archive.open("ab") as output:
        output.write(b"unexpected change")
    with pytest.raises(ValueError, match="Archive checksum mismatch"):
        resources.install("assets", archive)
    assert not (resources.ROOT / "assets").exists()


def test_google_drive_asset_download_preserves_verified_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace

    source = manifest(tmp_path, monkeypatch, {"assets/fixture.obj": b"mesh"})
    payload = source.read_bytes()
    specification = resources.resource("assets")
    specification["url"] = "https://drive.google.com/file/d/example/view?usp=sharing"
    monkeypatch.setattr(resources, "resource", lambda identity: specification)

    def download_file(url: str, output: str, *, fuzzy: bool) -> str:
        assert url == specification["url"]
        assert fuzzy is True
        Path(output).write_bytes(payload)
        return output

    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=download_file))
    destination = tmp_path / "downloads/runtime-assets.zip"
    assert resources.download("assets", destination) == destination
    assert destination.read_bytes() == payload
    assert not destination.with_suffix(".zip.partial").exists()
