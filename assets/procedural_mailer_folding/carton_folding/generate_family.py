#!/usr/bin/env python3
"""Generate a reproducible, collision-screened family of folding cartons.

All lengths are in metres. Rejected candidates and their reasons are recorded;
only parameter-feasible candidates passing the sampled geometry checker are
published as box_0000/, box_0001/, ... . This is not dynamics certification.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from build_carton import Parameters, generate
from kinematics import Model
from motion import Motion
from validate_carton import validate


def make_family(
    out: Path,
    count: int = 10,
    seed: int = 0,
    ranges: dict[str, tuple[float, float]] | None = None,
    backend: str = "urdfpy",
    samples: int = 401,
) -> dict:
    if count < 1 or samples < 2:
        raise ValueError("count must be positive and samples must be >= 2")
    ranges = ranges or {
        "width": (0.05, 0.16),
        "depth": (0.04, 0.12),
        "height": (0.08, 0.24),
        "thickness": (0.0006, 0.0016),
    }
    for key, (low, high) in ranges.items():
        if not np.isfinite([low, high]).all() or not 0 < low <= high:
            raise ValueError(f"Invalid {key} range: {low}, {high}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Never silently overwrite an earlier generated family.
    if (out / "manifest.json").exists() or any(out.glob("box_*")):
        raise ValueError(
            f"{out} already contains a family; choose an empty output directory"
        )
    rng = np.random.default_rng(seed)
    accepted, rejected = [], []
    max_attempts = max(100, 30 * count)
    for attempt in range(max_attempts):
        raw = {key: float(rng.uniform(*bounds)) for key, bounds in ranges.items()}
        try:
            params = Parameters(**raw)
            params.validate()
        except (ValueError, TypeError) as exc:
            rejected.append({"parameters": raw, "reason": str(exc)})
            continue
        with tempfile.TemporaryDirectory(prefix="_candidate_", dir=out) as temp:
            directory = Path(temp)
            urdf, metadata = generate(params, directory)
            # Missing optional libraries are not parameter rejection reasons.
            model = Model(urdf, backend)
            motion = Motion(metadata)
            report = validate(model, motion, samples)
            if not report["passed"]:
                rejected.append(
                    {
                        "parameters": raw,
                        "reason": "sampled_validation_failed",
                        "collision_pairs": report["collision_pairs"],
                        "static_errors": report["static_errors"],
                        "joint_limit_errors": report["joint_limit_errors"],
                    }
                )
                continue
            (directory / "validation.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            (directory / "configuration.json").write_text(
                json.dumps(raw, indent=2) + "\n"
            )
            name = f"box_{len(accepted):04d}"
            shutil.copytree(directory, out / name)
            accepted.append(
                {
                    "directory": name,
                    "parameters": asdict(params.resolved()),
                    "derived": params.derived(),
                    "validation_passed": True,
                    "num_configurations": report["num_configurations"],
                }
            )
            print(
                f'{name}: {raw["width"]*1000:.1f} x {raw["depth"]*1000:.1f} x '
                f'{raw["height"]*1000:.1f} mm, t={raw["thickness"]*1000:.2f} mm',
                flush=True,
            )
        if len(accepted) == count:
            break
    manifest = {
        "seed": seed,
        "ranges_m": ranges,
        "requested_count": count,
        "accepted_count": len(accepted),
        "backend": backend,
        "samples_requested_per_box": samples,
        "accepted": accepted,
        "rejected": rejected,
        "limitations": "Sampled rigid geometry only, not contact dynamics or continuous collision proof.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if len(accepted) != count:
        raise ValueError(
            f"Only {len(accepted)}/{count} feasible boxes after {max_attempts} attempts; "
            "inspect manifest.json and relax the parameter ranges"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("family"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=401)
    parser.add_argument("--backend", choices=("urdfpy", "reference"), default="urdfpy")
    for name, bounds in {
        "width": (0.05, 0.16),
        "depth": (0.04, 0.12),
        "height": (0.08, 0.24),
        "thickness": (0.0006, 0.0016),
    }.items():
        parser.add_argument(
            "--" + name + "-range",
            type=float,
            nargs=2,
            default=bounds,
            metavar=("MIN", "MAX"),
        )
    args = parser.parse_args()
    ranges = {
        name: tuple(getattr(args, name + "_range"))
        for name in ("width", "depth", "height", "thickness")
    }
    try:
        make_family(args.out, args.count, args.seed, ranges, args.backend, args.samples)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
