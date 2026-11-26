#!/usr/bin/env python3
"""Seeded procedural mailer families, accepting only collision-checked candidates."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
from build_mailer import Parameters, generate
from kinematics import Model
from motion import Motion
from validate_mailer import validate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("family"))
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--samples", type=int, default=1001)
    ap.add_argument("--backend", choices=("urdfpy", "reference"), default="urdfpy")
    defaults = {
        "width": (0.15, 0.32),
        "depth": (0.11, 0.26),
        "height": (0.024, 0.070),
        "thickness": (0.0007, 0.0025),
    }
    for key, default in defaults.items():
        ap.add_argument(
            "--" + key + "-range",
            nargs=2,
            type=float,
            default=default,
            metavar=("MIN", "MAX"),
        )
    a = ap.parse_args()
    if a.count < 1 or a.samples < 2:
        ap.error("count >= 1 and samples >= 2 are required")
    if a.out.exists():
        ap.error("Choose a new output directory; existing families are not overwritten")
    ranges = {k: getattr(a, k + "_range") for k in defaults}
    if any(
        not np.all(np.isfinite(v)) or v[0] <= 0 or v[1] < v[0] for v in ranges.values()
    ):
        ap.error("Require finite 0 < MIN <= MAX ranges")
    # Missing urdfpy is an environment failure, not a rejected geometry candidate.
    if a.backend == "urdfpy":
        try:
            import urdfpy
        except (ImportError, AttributeError) as exc:
            ap.exit(
                2,
                f"urdfpy import failed: {exc}. See README, or explicitly select --backend reference.\n",
            )
    a.out.mkdir(parents=True)
    rng = np.random.default_rng(a.seed)
    accepted = []
    rejected = []
    for attempt in range(max(100, 50 * a.count)):
        p = Parameters(**{k: float(rng.uniform(*v)) for k, v in ranges.items()})
        out = a.out / f"box_{len(accepted):04d}"
        try:
            path, _ = generate(p, out)
            report = validate(
                Model(path, a.backend), Motion(path.with_suffix(".json")), a.samples
            )
            if not report["passed"]:
                raise ValueError(
                    f"Validation: errors={report['static_errors']}; pairs={report['collision_pairs']}; limits={report['joint_limit_errors']}"
                )
        except (ValueError, ArithmeticError) as exc:
            rejected.append(
                dict(attempt=attempt, configuration=asdict(p), reason=str(exc))
            )
            if out.exists():
                shutil.rmtree(out)
            continue
        (out / "configuration.json").write_text(json.dumps(asdict(p), indent=2) + "\n")
        (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        accepted.append(
            dict(
                path=str(out.relative_to(a.out)),
                parameters=asdict(p),
                configurations_checked=report["sampled_configurations"],
            )
        )
        print(f"Accepted {len(accepted)}/{a.count}: {out}", flush=True)
        if len(accepted) == a.count:
            break
    manifest = dict(
        seed=a.seed,
        ranges=ranges,
        backend=a.backend,
        accepted=accepted,
        rejected=rejected,
        requested_count=a.count,
        complete=len(accepted) == a.count,
    )
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if len(accepted) != a.count:
        ap.exit(
            1,
            f"Only {len(accepted)} of {a.count} feasible models found; see manifest.\n",
        )


if __name__ == "__main__":
    main()
