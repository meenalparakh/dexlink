#!/usr/bin/env python3
"""Visualize the articulated mailer from actual URDF geometry and joint angles.
The default backend calls the installed urdfpy package. The independent
--backend reference is explicit and never silently substituted for urdfpy.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import numpy as np
from animation_io import save_gif
from kinematics import Model
from motion import Motion
from PIL import Image
from rendering import flat_net_svg, full_frame, overview, section_frame

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", type=Path, default=ROOT / "mailer.urdf")
    ap.add_argument("--backend", choices=("urdfpy", "reference"), default="urdfpy")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "output")
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--snapshots-only", action="store_true")
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument("--native-viewer", action="store_true")
    args = ap.parse_args()
    if (
        not math.isfinite(args.fps)
        or args.fps <= 0
        or args.width < 900
        or args.height < 650
    ):
        ap.error("Require a positive finite fps, width >= 900 and height >= 650")
    try:
        model = Model(args.urdf, args.backend)
        motion = Motion(args.urdf.with_suffix(".json"))
        if set(model.joint_names) != set(motion.zero):
            raise ValueError("URDF joint names and metadata do not match")
    except (RuntimeError, ValueError, OSError) as exc:
        ap.exit(2, f"Error: {exc}\n")
    if args.native_viewer:
        if model.robot is None:
            ap.error("--native-viewer requires --backend urdfpy")
        ts, qs, _ = motion.sample(60)
        model.robot.animate(
            cfg_trajectory={j: [q[j] for q in qs] for j in model.joint_names},
            loop_time=2 * motion.total,
        )
        return
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    motion.save(out / "trajectory.json", 120)
    times, qs, _ = motion.sample(120)
    np.savez_compressed(
        out / "trajectory.npz",
        time=times,
        joint_names=list(motion.zero),
        positions=[[q[j] for j in motion.zero] for q in qs],
    )
    for name, time in [
        ("open_configuration", motion.open_time),
        ("closed_configuration", motion.total),
    ]:
        (out / f"{name}.json").write_text(
            json.dumps(motion.evaluate(time)[0], indent=2) + "\n"
        )
    overview(model, motion, out / "overview.png")
    flat_net_svg(model, motion, out / "flat_net.svg")
    for name, time, caption in [
        ("unfolded", 0, None),
        ("open", motion.open_time, "Mailer assembled / lid open"),
        ("closed", motion.total, None),
    ]:
        full_frame(model, motion, time, args.width, args.height, caption).save(
            out / f"{name}.png"
        )
    for kind in ("base", "front", "wing"):
        section_frame(model, motion, motion.total, kind).save(
            out / f"{kind}_section.png"
        )
    generated = [
        "trajectory.json",
        "trajectory.npz",
        "open_configuration.json",
        "closed_configuration.json",
        "overview.png",
        "flat_net.svg",
        "unfolded.png",
        "open.png",
        "closed.png",
        "base_section.png",
        "front_section.png",
        "wing_section.png",
    ]
    if not args.snapshots_only:
        ts, _, _ = motion.sample(args.fps)
        writer = None
        frames = []
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
                print(f"MP4 unavailable; still generating GIF: {exc}", file=sys.stderr)
        try:
            for i, t in enumerate(ts):
                frame = full_frame(model, motion, float(t), args.width, args.height)
                frames.append(
                    frame.resize(
                        (720, round(frame.height * 720 / frame.width)),
                        Image.Resampling.LANCZOS,
                    )
                )
                if writer is not None:
                    writer.append_data(np.asarray(frame))
                if i % 60 == 0:
                    print(f"Frame {i+1}/{len(ts)}", flush=True)
        finally:
            if writer is not None:
                writer.close()
        save_gif(frames, out / "folding.gif", args.fps)
        generated.append("folding.gif")
        del frames
        if writer is not None:
            generated.append("folding.mp4")
        for kind, k0, k1 in [("base", 7, 8), ("front", 14, 15), ("wing", 13, 15)]:
            a, b = motion.bounds[k0], motion.bounds[k1]
            tt = np.r_[
                np.full(round(args.fps * 0.8), a),
                np.linspace(a, b, round(args.fps * 3.0) + 1),
                np.full(round(args.fps * 1.2), b),
            ]
            ff = [section_frame(model, motion, float(t), kind) for t in tt]
            save_gif(ff, out / f"{kind}_insertion.gif", args.fps)
            generated.append(f"{kind}_insertion.gif")
    provenance = {
        "fk_backend": model.backend,
        "urdfpy_executed": model.robot is not None,
        "python": sys.version,
        "rendering": "Orthographic software z-buffer using actual URDF geometry",
        "generated": generated,
        "notice": "Kinematic preview only, not a contact-dynamics simulation.",
    }
    for pkg in ("numpy", "Pillow", "urdfpy", "networkx", "trimesh", "imageio"):
        try:
            provenance[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            provenance[pkg] = None
    (out / "render_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved {out.resolve()} / FK backend {model.backend}")


if __name__ == "__main__":
    main()
