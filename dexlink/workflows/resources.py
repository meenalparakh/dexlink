"""Download and install the exact runtime assets or positive policy checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
from urllib.parse import urlsplit
import zipfile

from dexlink.utils.retargeting.path_utils import get_logs_dir, resolve_project_path
from .catalog import ROOT, load_catalog, repository_path, sha256


def resource(identity: str) -> dict:
    document = json.loads((ROOT / "resources/manifest.json").read_text())
    if document["schema"] != "dexlink_resources_v1" or identity not in {
        "assets",
        "checkpoints",
    }:
        raise ValueError(
            "Expected assets or checkpoints in a dexlink_resources_v1 manifest."
        )
    return document[identity]


def download(identity: str, destination: Path) -> Path:
    specification = resource(identity)
    if not specification["url"]:
        raise ValueError(
            f"Missing download URL for {identity}; configure resources/manifest.json or provide --archive."
        )
    if destination.exists():
        raise FileExistsError(f"Download destination already exists: {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(
            f"Remove or inspect the previous incomplete download: {temporary}."
        )
    if urlsplit(specification["url"]).hostname == "drive.google.com":
        import gdown

        result = gdown.download(specification["url"], str(temporary), fuzzy=True)
        if result is None:
            raise RuntimeError(
                "Google Drive download failed; check the file sharing settings."
            )
    else:
        with (
            urllib.request.urlopen(specification["url"], timeout=60) as source,
            temporary.open("xb") as output,
        ):
            shutil.copyfileobj(source, output)
    if sha256(temporary) != specification["sha256"]:
        raise ValueError(f"Downloaded archive checksum mismatch: {temporary}.")
    temporary.rename(destination)
    return destination


def install(identity: str, archive: Path) -> None:
    specification = resource(identity)
    if sha256(archive) != specification["sha256"]:
        raise ValueError(f"Archive checksum mismatch for {identity}: {archive}.")
    expected = specification["files"]
    missing = []
    # Validate names and every required file before writing into the checkout.
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len({item.filename for item in members}) != len(members):
            raise ValueError("Archive contains duplicate member names.")
        for item in members:
            path = PurePosixPath(item.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in item.filename
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe archive member: {item.filename}.")
        for name, digest in expected.items():
            if name not in bundle.namelist():
                raise ValueError(f"Missing required archive member: {name}.")
            import hashlib

            if hashlib.sha256(bundle.read(name)).hexdigest() != digest:
                raise ValueError(f"Archive member checksum mismatch: {name}.")
            target = resolve_project_path(name)
            if target.exists():
                if not target.is_file() or sha256(target) != digest:
                    raise FileExistsError(
                        f"Refusing to replace different existing content: {target}."
                    )
            else:
                missing.append(name)
        cache = get_logs_dir() / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="resources-", dir=cache) as staging:
            for name in missing:
                path = Path(staging) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, path.open("xb") as output:
                    shutil.copyfileobj(source, output)
            for name in missing:
                target = resolve_project_path(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise FileExistsError(
                        f"Destination appeared during installation: {target}."
                    )
                (Path(staging) / name).rename(target)
    if identity == "checkpoints":
        for task in load_catalog().values():
            checkpoint = task["checkpoint"]
            if checkpoint is not None:
                if (
                    type(checkpoint["timestep"]) is not int
                    or checkpoint["timestep"] <= 0
                ):
                    raise ValueError(
                        f"Expected a positive checkpoint step for {task['id']}."
                    )
                folder = (
                    repository_path(checkpoint["run_directory"])
                    / "models"
                    / str(checkpoint["timestep"])
                )
                for name, digest in checkpoint["files"].items():
                    if sha256(folder / name) != digest:
                        raise ValueError(
                            f"Learned state differs from the task catalog: {folder / name}."
                        )
    print(
        f"Installed {identity}; verified {len(expected)} files ({len(missing)} newly installed)."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resource", choices=("assets", "checkpoints"))
    parser.add_argument(
        "--archive", type=Path, help="Install a previously downloaded ZIP."
    )
    parser.add_argument(
        "--download-directory", type=Path, default=get_logs_dir() / "downloads"
    )
    args = parser.parse_args(argv)
    archive = args.archive
    if archive is None:
        specification = resource(args.resource)
        archive = args.download_directory / specification["filename"]
        if not archive.exists():
            download(args.resource, archive)
    install(args.resource, archive.resolve(strict=True))


if __name__ == "__main__":
    main()
