"""Shared-palette GIF output without frame-to-frame palette flicker."""

from pathlib import Path

import numpy as np
from PIL import Image


def save_gif(
    frames: list[Image.Image], path: Path, fps: float, max_width: int = 840
) -> None:
    if not frames:
        raise ValueError("Cannot save an empty animation")
    # A shared palette avoids frame-to-frame palette flicker and color drift.
    resized = []
    for frame in frames:
        if frame.width > max_width:
            frame = frame.resize(
                (max_width, round(frame.height * max_width / frame.width)),
                Image.Resampling.LANCZOS,
            )
        resized.append(frame)
    palette_samples = Image.new("RGB", (resized[0].width, resized[0].height * 4))
    for k, idx in enumerate(np.linspace(0, len(resized) - 1, 4).astype(int)):
        palette_samples.paste(resized[idx], (0, k * resized[idx].height))
    palette = palette_samples.quantize(colors=128)
    encoded = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in resized
    ]
    encoded[0].save(
        path,
        save_all=True,
        append_images=encoded[1:],
        loop=0,
        duration=[
            max(10, 10 * (round((i + 1) * 100 / fps) - round(i * 100 / fps)))
            for i in range(len(encoded))
        ],
        disposal=2,
        optimize=False,
    )
