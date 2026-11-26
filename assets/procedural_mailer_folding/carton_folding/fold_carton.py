#!/usr/bin/env python3
"""Animate a folding carton using urdfpy FK, with optional native urdfpy viewer.

Default: render GIF, MP4, cross-section GIF, stills, and joint trajectory.
No OpenGL is needed except for --native-viewer. The optional --backend reference
is an explicitly different backend; it is not an emulation of urdfpy.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
from kinematics import Model
from motion import Motion
from PIL import Image
from rendering import contact_sheet, flat_net_svg, full_frame, section_frame

ROOT = Path(__file__).resolve().parent


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=ROOT / "carton.urdf")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--backend", choices=("urdfpy", "reference"), default="urdfpy")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--snapshots-only", action="store_true")
    parser.add_argument("--no-mp4", action="store_true")
    parser.add_argument("--native-viewer", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0 or args.width < 800 or args.height < 600:
        parser.error("Require fps > 0, width >= 800, and height >= 600")
    try:
        model = Model(args.urdf, args.backend)
        motion = Motion(args.metadata or args.urdf.with_suffix(".json"))
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    if args.native_viewer:
        if model.robot is None:
            parser.error("--native-viewer requires --backend urdfpy")
        # urdfpy's native viewer plays the trajectory forward AND backward.
        times, cfgs, _ = motion.sample(60)
        model.robot.animate(
            cfg_trajectory={j: [q[j] for q in cfgs] for j in model.joint_names},
            loop_time=2 * motion.total,
        )
        return
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    motion.save(out / "trajectory.json", fps=120)
    (out / "assembled_configuration.json").write_text(
        json.dumps(motion.keyframes[-1], indent=2) + "\n"
    )
    contact_sheet(model, motion, out / "overview.png")
    flat_net_svg(model, motion, out / "flat_net.svg")
    full_frame(model, motion, 0, args.width, args.height).save(out / "unfolded.png")
    full_frame(model, motion, motion.total, args.width, args.height).save(
        out / "assembled.png"
    )
    section_frame(model, motion, motion.bounds[13]).save(out / "insertion_entry.png")
    section_frame(model, motion, motion.total).save(out / "insertion_seated.png")
    generated = [
        "flat_net.svg",
        "overview.png",
        "unfolded.png",
        "assembled.png",
        "insertion_entry.png",
        "insertion_seated.png",
        "trajectory.json",
        "assembled_configuration.json",
    ]
    if not args.snapshots_only:
        times, _, _ = motion.sample(args.fps)
        writer = None
        if not args.no_mp4:
            try:
                import imageio.v2 as imageio

                writer = imageio.get_writer(
                    str(out / "folding.mp4"),
                    fps=args.fps,
                    codec="libx264",
                    quality=8,
                    macro_block_size=2,
                    ffmpeg_params=["-pix_fmt", "yuv420p"],
                )
            except (ImportError, RuntimeError, OSError) as exc:
                print(f"MP4 unavailable; continuing with GIF: {exc}", file=sys.stderr)
        frames = []
        try:
            for i, t in enumerate(times):
                frame = full_frame(model, motion, float(t), args.width, args.height)
                gif_width = min(720, frame.width)
                frames.append(
                    frame.resize(
                        (gif_width, round(frame.height * gif_width / frame.width)),
                        Image.Resampling.LANCZOS,
                    )
                )
                if writer is not None:
                    writer.append_data(np.array(frame))
                if i % 50 == 0:
                    print(f"Rendering {i+1}/{len(times)} frames", flush=True)
        finally:
            if writer is not None:
                writer.close()
        save_gif(frames, out / "folding.gif", args.fps)
        generated.append("folding.gif")
        if writer is not None:
            generated.append("folding.mp4")
        del frames
        # Show the alignment hold, the whole top insertion, and the seated hold.
        section_times = np.r_[
            np.full(round(args.fps * 0.7), motion.bounds[13]),
            np.linspace(
                motion.bounds[13], motion.bounds[14], round(args.fps * 2.4) + 1
            ),
            np.full(round(args.fps * 1.2), motion.bounds[14]),
        ]
        section_frames = [section_frame(model, motion, float(t)) for t in section_times]
        save_gif(section_frames, out / "insertion_section.gif", args.fps)
        generated.append("insertion_section.gif")
    provenance = {
        "fk_backend": model.backend,
        "urdfpy_executed": model.robot is not None,
        "python": sys.version,
        "rendering": "Software orthographic z-buffer; no OpenGL required",
        "generated": generated,
        "notice": "This is a kinematic visualization, not a contact-dynamics simulation.",
    }
    for package in ("numpy", "Pillow", "urdfpy", "networkx", "trimesh"):
        try:
            provenance[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            provenance[package] = None
    (out / "render_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved to {out.resolve()} (FK backend: {model.backend})")


if __name__ == "__main__":
    main()
