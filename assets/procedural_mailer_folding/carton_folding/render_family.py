#!/usr/bin/env python3
"""Render a contact sheet for a selection of procedural cartons."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from kinematics import Model
from motion import Motion
from PIL import Image, ImageDraw
from rendering import BG, INK, MUTED, font, scene_image


def gallery(paths: list[Path], output: Path, backend: str = "urdfpy") -> None:
    if not paths:
        raise ValueError("At least one URDF is required")
    cols, tw, th = min(3, len(paths)), 450, 390
    rows = math.ceil(len(paths) / cols)
    canvas = Image.new("RGB", (cols * tw, rows * th + 135), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (26, 22), "ONE GENERATOR / DIFFERENT DIMENSIONS", font=font(26, True), fill=INK
    )
    draw.text(
        (26, 62),
        "Flaps, hinge margins and tuck motions are recomputed for each box.",
        font=font(16),
        fill=MUTED,
    )
    for i, path in enumerate(paths):
        model = Model(path, backend)
        motion = Motion(path.with_suffix(".json"))
        x, y = (i % cols) * tw, 100 + (i // cols) * th
        scene = scene_image(model, motion.keyframes[-1], tw, th - 65)
        canvas.paste(scene, (x, y))
        p = motion.p
        name = path.parent.name.replace("_", " ").title()
        draw.text((x + 25, y + th - 57), name, font=font(18, True), fill=INK)
        dims = " x ".join(f"{p[k]*1000:g}" for k in ("width", "depth", "height"))
        draw.text(
            (x + 25, y + th - 29),
            f'{dims} mm  |  board {p["thickness"]*1000:g} mm',
            font=font(14),
            fill=MUTED,
        )
    draw.text(
        (26, canvas.height - 27),
        f"Nominal fold-line dimensions. Views independently fitted, not to a common scale. FK: {backend}.",
        font=font(13),
        fill=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("gallery.png"))
    parser.add_argument("--backend", choices=("urdfpy", "reference"), default="urdfpy")
    args = parser.parse_args()
    try:
        gallery(args.models, args.out, args.backend)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
