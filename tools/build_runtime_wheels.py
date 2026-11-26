"""Rebuild the small, licensed dependency fixes used by the DexLink runtime.

Run from the repository root with setuptools==80.9.0 and wheel==0.45.1.
The source hashes and exact edits below keep these patches reviewable.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request

URDF_REVISION = "5466842899b33bd549e8f9e2a9a987bd5e37373b"
URDF_SOURCES = {
    "setup.py": "3c17a19cb2b66dac3f92ae11259ef440b30f4ef6879b8fc0a52a0a7122997116",
    "urdfpy/urdf.py": "a347891d1d196cf1d8623608c1b0ce4bfeb2e6d274f34b9a9b28c34cdfba196b",
    "urdfpy/utils.py": "23a70c4dc6c915ed6d5ca4009eb0952a003539d94bdec28c78af0ceefb578446",
    "urdfpy/__init__.py": "0afedd918e8e4bd040915ad15804c53accb2790976381bbc97bd376d3027d765",
    "urdfpy/version.py": "5850f3ff67ee62e78bdeceb795e61acea476578d60bc3a13f3c9de0d02db235c",
    "LICENSE": "53afcfa244e268b059a4ee34c9df5802a191059efbfde5774435ce8e0cdbc0c7",
}
PYRENDER_URL = "https://files.pythonhosted.org/packages/28/88/174c28b9d3d03cf6d8edb6f637458f30f1cf1a2bd7a617cbd9dadb1740f6/pyrender-0.1.45-py3-none-any.whl"
PYRENDER_SHA256 = "5cf751d1f21fba4640e830cef3a0b5a95ed0f05677bf92c6b8330056b4023aeb"


def fetch(url: str, path: Path, expected: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as source:
            content = source.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Upstream source hash changed: {url}.")
        path.write_bytes(content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Cached upstream source hash changed: {path}.")


def replace_once(path: Path, before: str, after: str) -> None:
    text = path.read_text()
    if text.count(before) != 1:
        raise ValueError(f"Expected exactly one patch location in {path}: {before!r}.")
    path.write_text(text.replace(before, after))


def build(output: Path, cache: Path) -> None:
    import importlib.metadata

    for package, expected in {"setuptools": "80.9.0", "wheel": "0.45.1"}.items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Rebuilding requires {package}=={expected}.")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, SOURCE_DATE_EPOCH="315532800")
    with tempfile.TemporaryDirectory(prefix="runtime-wheels-", dir=cache) as directory:
        staging = Path(directory)
        source = staging / "urdfpy"
        for name, digest in URDF_SOURCES.items():
            cached = cache / URDF_REVISION / name
            fetch(
                f"https://raw.githubusercontent.com/mmatl/urdfpy/{URDF_REVISION}/{name}",
                cached,
                digest,
            )
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)

        # Match the tested parser source and its modern graph dependency.
        replace_once(source / "setup.py", "networkx==2.2", "networkx==3.3")
        replace_once(source / "setup.py", "    setup_requires = requirements,\n", "")
        replace_once(source / "urdfpy/urdf.py", "astype(np.float)", "astype(float)")
        replace_once(source / "urdfpy/version.py", "0.0.22", "0.0.22+dexlink.1")
        built = staging / "wheels"
        built.mkdir()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(built),
                str(source),
            ],
            check=True,
            env=environment,
        )

        # The renderer's old OpenGL pin conflicts with the required Genesis fork.
        wheel = cache / "pyrender-0.1.45-py3-none-any.whl"
        fetch(PYRENDER_URL, wheel, PYRENDER_SHA256)
        unpacked = staging / "unpacked"
        subprocess.run(
            [sys.executable, "-m", "wheel", "unpack", str(wheel), "-d", str(unpacked)],
            check=True,
        )
        package = unpacked / "pyrender-0.1.45"
        metadata = package / "pyrender-0.1.45.dist-info"
        replace_once(
            metadata / "METADATA", "Version: 0.1.45\n", "Version: 0.1.45+dexlink.1\n"
        )
        replace_once(
            metadata / "METADATA",
            "Requires-Dist: PyOpenGL (==3.1.0)",
            "Requires-Dist: PyOpenGL (>=3.1.4)",
        )
        metadata.rename(package / "pyrender-0.1.45+dexlink.1.dist-info")
        renamed = unpacked / "pyrender-0.1.45+dexlink.1"
        package.rename(renamed)
        subprocess.run(
            [sys.executable, "-m", "wheel", "pack", str(renamed), "-d", str(built)],
            check=True,
            env=environment,
        )
        for path in sorted(built.glob("*.whl")):
            target = output / path.name
            if target.exists() and target.read_bytes() != path.read_bytes():
                raise FileExistsError(
                    f"Refusing to replace a different wheel: {target}."
                )
            shutil.copyfile(path, target)
            print(f"{hashlib.sha256(target.read_bytes()).hexdigest()}  {target.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("third_party/wheels")
    )
    parser.add_argument(
        "--cache-directory", type=Path, default=Path("outputs/cache/runtime-wheels")
    )
    args = parser.parse_args()
    build(args.output_directory.resolve(), args.cache_directory.resolve())


if __name__ == "__main__":
    main()
