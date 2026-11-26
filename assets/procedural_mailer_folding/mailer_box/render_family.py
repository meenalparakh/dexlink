#!/usr/bin/env python3
"""Gallery of procedural mailers. Each tile uses its own fitted camera."""
import argparse
from pathlib import Path

from kinematics import Model
from motion import Motion
from PIL import Image, ImageDraw
from rendering import BG, INK, MUTED, font, scene_image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("urdfs", type=Path, nargs="+")
    ap.add_argument("--backend", choices=("urdfpy", "reference"), default="urdfpy")
    ap.add_argument("--out", type=Path, default=Path("gallery.png"))
    a = ap.parse_args()
    cols = min(3, len(a.urdfs))
    rows = (len(a.urdfs) + cols - 1) // cols
    tw, th = 480, 392
    im = Image.new("RGB", (cols * tw, rows * th + 130), BG)
    dr = ImageDraw.Draw(im)
    dr.text(
        (28, 20), "ONE GENERATOR / DIFFERENT MAILERS", font=font(29, True), fill=INK
    )
    dr.text(
        (28, 65),
        "Panels, folded margins, apertures and insertion poses regenerate from dimensions.",
        font=font(16),
        fill=MUTED,
    )
    for i, path in enumerate(a.urdfs):
        m = Model(path, a.backend)
        mo = Motion(path.with_suffix(".json"))
        p = mo.p
        x, y = (i % cols) * tw, 100 + (i // cols) * th
        im.paste(scene_image(m, mo.evaluate(mo.open_time)[0], tw, th - 74), (x, y))
        dr.text(
            (x + 24, y + th - 64),
            path.parent.name.replace("_", " ").title(),
            font=font(19, True),
            fill=INK,
        )
        dims = " x ".join(f"{p[n]*1000:g}" for n in ("width", "depth", "height"))
        dr.text(
            (x + 24, y + th - 35),
            f'{dims} mm / {p["thickness"]*1000:g} mm board',
            font=font(14),
            fill=MUTED,
        )
    dr.text(
        (28, im.height - 25),
        f"Nominal fold-line dimensions. Independently fitted views, not a common scale. FK: {a.backend}.",
        font=font(13),
        fill=MUTED,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    im.save(a.out)


if __name__ == "__main__":
    main()
