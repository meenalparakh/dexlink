"""Portable repository paths and race-free URDF materialization."""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from uuid import uuid4


DEXLINK_ROOT_ENV = "DEXLINK_ROOT"
DEXLINK_DATA_ENV = "DEXLINK_EXTERNAL_DATA"
DEXLINK_OUTPUTS_ENV = "DEXLINK_OUTPUTS"


def get_src() -> Path:
    """Return the DexLink repository root."""

    configured = os.environ.get(DEXLINK_ROOT_ENV)
    root = Path(configured).expanduser() if configured else Path(__file__).parents[3]
    root = root.resolve()
    if not (root / "dexlink").is_dir():
        raise FileNotFoundError(
            f"Expected a DexLink checkout containing 'dexlink/' at {root}. "
            f"Set {DEXLINK_ROOT_ENV} to the checkout root."
        )
    return root


def get_assets_dir() -> Path:
    return get_src() / "assets"


def get_logs_dir() -> Path:
    configured = os.environ.get(DEXLINK_OUTPUTS_ENV)
    return (
        Path(configured).expanduser().resolve()
        if configured
        else get_src() / "outputs"
    )


def get_trajopt_results_dir() -> Path:
    return get_logs_dir() / "retargeting"


def get_tmp_dir() -> Path:
    path = get_logs_dir() / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_external_data_dir() -> Path:
    configured = os.environ.get(DEXLINK_DATA_ENV)
    return (
        Path(configured).expanduser().resolve()
        if configured
        else get_src() / "external_data"
    )


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a repository, external-data, or output path portably.

    Task specifications intentionally store paths relative to the public checkout.
    The two configured mount points retain that stable spelling while allowing the
    large datasets and generated outputs to live elsewhere.
    """

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    parts = candidate.parts
    if parts and parts[0] == "external_data":
        return (get_external_data_dir().joinpath(*parts[1:])).resolve()
    if parts and parts[0] == "outputs":
        return (get_logs_dir().joinpath(*parts[1:])).resolve()
    return (get_src() / candidate).resolve()


def get_abs_cluster_path(path: str | Path) -> Path:
    """Backward-compatible name for portable project-path resolution."""

    return resolve_project_path(path)


def replace_assets_path_in_urdf(urdf_path: str | Path) -> Path:
    """Resolve a URDF path relative to the DexLink checkout."""

    return resolve_project_path(urdf_path)


def process_urdf(urdf_path: str | Path, append_jobname: bool = False) -> Path:
    """Write a cached URDF whose relative resource filenames are absolute."""

    source = resolve_project_path(urdf_path)
    if not source.is_file():
        raise FileNotFoundError(f"Expected an existing URDF file, got {source}.")
    if source.suffix.lower() != ".urdf":
        raise ValueError(f"Expected a .urdf file, got {source}.")
    source_text = source.read_text(encoding="utf-8")
    if not source_text.strip():
        raise ValueError(f"Expected a non-empty URDF file, got {source}.")
    try:
        root = ET.fromstring(source_text)
    except ET.ParseError as error:
        raise ValueError(f"Invalid URDF XML in {source}: {error}.") from error

    # urdfpy needs resource paths that remain valid outside the source directory.
    changed = False
    for element in root.iter():
        filename = element.attrib.get("filename")
        if filename is None or filename.startswith(
            ("package://", "http://", "https://")
        ):
            continue
        resource = Path(filename).expanduser()
        resolved = (
            resource.resolve()
            if resource.is_absolute()
            else (source.parent / resource).resolve()
        )
        if not resolved.exists():
            raise FileNotFoundError(
                f"URDF resource {filename!r} referenced by {source} does not "
                f"exist at {resolved}."
            )
        element.set("filename", str(resolved))
        changed = changed or str(resolved) != filename

    if not changed and not append_jobname:
        return source

    digest = hashlib.sha256(
        f"{source.resolve()}\n{source.stat().st_mtime_ns}\n{source_text}".encode()
    ).hexdigest()[:16]
    suffix = f"_{get_job_process_name()}" if append_jobname else ""
    destination = (
        get_tmp_dir()
        / "processed_urdf"
        / f"{source.stem}_{digest}{suffix}.urdf"
    )
    rendered = ET.tostring(root, encoding="unicode")
    _write_text_atomic(destination, rendered + "\n")
    return destination


def get_job_process_name() -> str:
    """Return a process-unique suffix identifiable in Slurm logs."""

    parts = [
        os.environ[key]
        for key in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID")
        if key in os.environ
    ]
    parts.append(str(os.getpid()))
    return "_".join(parts)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def get_experiments_data_dir() -> str:
    return str(get_external_data_dir())
