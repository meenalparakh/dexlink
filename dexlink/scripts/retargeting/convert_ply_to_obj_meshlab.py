#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

BALL_PIVOTING_FILTER_SCRIPT = """<!DOCTYPE FilterScript>
<FilterScript>
  <filter name="Compute normals for point sets">
    <Param name="K" value="{normal_neighbors}" type="RichInt"/>
    <Param name="smoothIter" value="0" type="RichInt"/>
    <Param name="flipFlag" value="false" type="RichBool"/>
    <Param name="viewPos" x="0" y="0" z="0" type="RichPoint3f"/>
  </filter>
  <filter name="Surface Reconstruction: Ball Pivoting">
    <Param name="BallRadius" value="{ball_radius}" type="RichAbsPerc"/>
    <Param name="Clustering" value="{clustering}" type="RichFloat"/>
    <Param name="CreaseThr" value="{crease_angle}" type="RichFloat"/>
    <Param name="DeleteFaces" value="false" type="RichBool"/>
  </filter>
</FilterScript>
"""

SCREENED_POISSON_FILTER_SCRIPT = """<!DOCTYPE FilterScript>
<FilterScript>
  <filter name="Compute normals for point sets">
    <Param name="K" value="{normal_neighbors}" type="RichInt"/>
    <Param name="smoothIter" value="0" type="RichInt"/>
    <Param name="flipFlag" value="false" type="RichBool"/>
    <Param name="viewPos" x="0" y="0" z="0" type="RichPoint3f"/>
  </filter>
  <filter name="Surface Reconstruction: Screened Poisson">
    <Param name="depth" value="{poisson_depth}" type="RichInt"/>
    <Param name="fulldepth" value="5" type="RichInt"/>
    <Param name="cgdepth" value="0" type="RichInt"/>
    <Param name="scale" value="1.1" type="RichFloat"/>
    <Param name="samplespernode" value="1.5" type="RichFloat"/>
    <Param name="pointweight" value="4" type="RichFloat"/>
    <Param name="iters" value="8" type="RichInt"/>
    <Param name="confidence" value="false" type="RichBool"/>
    <Param name="preclean" value="false" type="RichBool"/>
  </filter>
</FilterScript>
"""


@dataclass(frozen=True)
class ConvertPlyConfig:
    input_ply: Path
    output_obj: Path | None = None
    method: str = "ball_pivoting"
    meshlabserver: str | None = None
    filter_script: Path | None = None
    qt_platform: str = "xcb"
    xvfb_run: str | None = None
    no_xvfb: bool = False
    force: bool = False
    output_mask: tuple[str, ...] = ("vn", "vc")
    normal_neighbors: int = 16
    ball_radius: float = 0.0
    clustering: float = 20.0
    crease_angle: float = 90.0
    poisson_depth: int = 8

    @property
    def resolved_output_obj(self) -> Path:
        if self.output_obj is not None:
            return self.output_obj
        return self.input_ply.with_name(f"{self.input_ply.stem}_meshlab.obj")


def _resolve_meshlabserver(explicit_path: str | None) -> str:
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.is_file():
            raise FileNotFoundError(f"MeshLab server path does not exist: {path}")
        return str(path)

    for candidate in ("meshlabserver", "MeshLabServer"):
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
    raise RuntimeError(
        "Could not find `meshlabserver` on PATH. Install MeshLab and make sure "
        "the CLI is available, or pass --meshlabserver /path/to/meshlabserver."
    )


def _resolve_xvfb_run(explicit_path: str | None, no_xvfb: bool) -> str | None:
    if no_xvfb:
        return None
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.is_file():
            raise FileNotFoundError(f"xvfb-run path does not exist: {path}")
        return str(path)
    return shutil.which("xvfb-run")


def _meshlab_env(qt_platform: str) -> dict[str, str]:
    env = os.environ.copy()
    if qt_platform:
        env["QT_QPA_PLATFORM"] = qt_platform
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
    env.setdefault("MESA_GLSL_VERSION_OVERRIDE", "330")
    return env


def _filter_script_from_config(config: ConvertPlyConfig) -> str:
    if config.normal_neighbors <= 0:
        raise ValueError(
            f"`normal_neighbors` must be positive, got {config.normal_neighbors}."
        )
    if config.ball_radius < 0.0:
        raise ValueError(
            f"`ball_radius` must be non-negative, got {config.ball_radius}."
        )
    if config.clustering <= 0.0:
        raise ValueError(f"`clustering` must be positive, got {config.clustering}.")
    if config.crease_angle <= 0.0:
        raise ValueError(f"`crease_angle` must be positive, got {config.crease_angle}.")
    if config.poisson_depth <= 0:
        raise ValueError(
            f"`poisson_depth` must be positive, got {config.poisson_depth}."
        )

    if config.filter_script is not None:
        if not config.filter_script.is_file():
            raise FileNotFoundError(
                f"MeshLab filter script does not exist: {config.filter_script}"
            )
        return config.filter_script.read_text(encoding="utf-8")

    if config.method == "ball_pivoting":
        return BALL_PIVOTING_FILTER_SCRIPT.format(
            normal_neighbors=config.normal_neighbors,
            ball_radius=config.ball_radius,
            clustering=config.clustering,
            crease_angle=config.crease_angle,
        )
    if config.method == "screened_poisson":
        return SCREENED_POISSON_FILTER_SCRIPT.format(
            normal_neighbors=config.normal_neighbors,
            poisson_depth=config.poisson_depth,
        )
    raise ValueError(
        f"Expected method 'ball_pivoting' or 'screened_poisson', got {config.method!r}."
    )


def _validate_output_mask(output_mask: tuple[str, ...]) -> None:
    valid_tokens = {
        "vc",
        "vf",
        "vq",
        "vn",
        "vt",
        "fc",
        "ff",
        "fq",
        "fn",
        "wt",
    }
    for token in output_mask:
        if token not in valid_tokens:
            raise ValueError(
                f"Invalid MeshLab output mask token {token!r}. "
                f"Expected one of {sorted(valid_tokens)}."
            )


def convert_ply_to_obj(config: ConvertPlyConfig) -> Path:
    input_ply = config.input_ply.resolve()
    if not input_ply.is_file():
        raise FileNotFoundError(f"Input PLY does not exist: {input_ply}")
    if input_ply.suffix.lower() != ".ply":
        raise ValueError(f"Expected a .ply input file, got: {input_ply}")

    output_obj = config.resolved_output_obj.resolve()
    if output_obj.suffix.lower() != ".obj":
        raise ValueError(f"Expected a .obj output path, got: {output_obj}")
    if output_obj.exists() and not config.force:
        raise FileExistsError(
            f"Output OBJ already exists: {output_obj}. Pass --force to overwrite it."
        )
    output_obj.parent.mkdir(parents=True, exist_ok=True)

    meshlabserver = _resolve_meshlabserver(config.meshlabserver)
    xvfb_run = _resolve_xvfb_run(config.xvfb_run, config.no_xvfb)
    filter_script = _filter_script_from_config(config)
    _validate_output_mask(config.output_mask)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".mlx",
        encoding="utf-8",
        delete=False,
    ) as tmp:
        tmp.write(filter_script)
        mlx_path = Path(tmp.name)

    try:
        cmd = [
            meshlabserver,
            "-i",
            str(input_ply),
            "-o",
            str(output_obj),
            "-m",
            *config.output_mask,
            "-s",
            str(mlx_path),
        ]
        if xvfb_run is not None:
            cmd = [
                xvfb_run,
                "-a",
                "-s",
                "-screen 0 1280x1024x24 +extension GLX +render",
            ] + cmd
        else:
            print(
                "Warning: `xvfb-run` was not found. MeshLab may fail with GLEW/OpenGL "
                "errors unless this shell already has a working display/OpenGL context."
            )
        print("Running:", " ".join(cmd))
        env = _meshlab_env(config.qt_platform)
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "MeshLab failed while converting the PLY. If the error mentions "
                "GLEW, OpenGL, or a Qt platform plugin, install/use xvfb-run with "
                "Mesa software rendering. On Ubuntu, try: "
                "`sudo apt-get install xvfb libgl1-mesa-dri libglx-mesa0`, then rerun "
                "this script. You can also pass --xvfb-run /path/to/xvfb-run or "
                "--no-xvfb if this shell has a working display."
            ) from exc
    finally:
        mlx_path.unlink(missing_ok=True)

    if not output_obj.is_file():
        raise FileNotFoundError(
            f"MeshLab did not create the expected OBJ: {output_obj}"
        )
    print(f"Saved OBJ mesh to {output_obj}")
    return output_obj


def parse_args(argv: list[str] | None = None) -> ConvertPlyConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a PLY point cloud or mesh to OBJ through MeshLab. "
            "Ball Pivoting is the default because it preserves open boundaries "
            "better than Poisson reconstruction for cup-like objects."
        )
    )
    parser.add_argument("input_ply", type=Path)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument(
        "--method",
        choices=("ball_pivoting", "screened_poisson"),
        default="ball_pivoting",
    )
    parser.add_argument("--meshlabserver", default=None, type=str)
    parser.add_argument("--filter-script", default=None, type=Path)
    parser.add_argument(
        "--qt-platform",
        default="",
        type=str,
        help=(
            "Optional QT_QPA_PLATFORM override. Leave empty when using xvfb-run; "
            "try xcb, offscreen, or minimal only for debugging platform issues."
        ),
    )
    parser.add_argument("--xvfb-run", default=None, type=str)
    parser.add_argument("--no-xvfb", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-mask",
        nargs="+",
        default=("vn", "vc"),
        help=(
            "MeshLab output mask tokens passed after -m. Defaults to vertex normals "
            "and vertex colors: vn vc."
        ),
    )
    parser.add_argument("--normal-neighbors", default=16, type=int)
    parser.add_argument("--ball-radius", default=0.0, type=float)
    parser.add_argument("--clustering", default=20.0, type=float)
    parser.add_argument("--crease-angle", default=90.0, type=float)
    parser.add_argument("--poisson-depth", default=8, type=int)
    args = parser.parse_args(argv)
    return ConvertPlyConfig(
        input_ply=args.input_ply,
        output_obj=args.output,
        method=args.method,
        meshlabserver=args.meshlabserver,
        filter_script=args.filter_script,
        qt_platform=args.qt_platform,
        xvfb_run=args.xvfb_run,
        no_xvfb=args.no_xvfb,
        force=args.force,
        output_mask=tuple(args.output_mask),
        normal_neighbors=args.normal_neighbors,
        ball_radius=args.ball_radius,
        clustering=args.clustering,
        crease_angle=args.crease_angle,
        poisson_depth=args.poisson_depth,
    )


def main(argv: list[str] | None = None) -> int:
    convert_ply_to_obj(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
