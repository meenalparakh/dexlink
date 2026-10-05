#!/usr/bin/env python
"""Render exact terminal grasp states saved by grasp-refinement evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dexlink.utils.retargeting.render_utils import render_centered_four_views
from dexlink.visualizations.rl.render_paper_retargeting_figures import (
    _build_bimanual_frame_instances,
    _load_exported_npy_payload,
)


REFERENCE_ROOT_NAME = "grasp_refinement_sequences"


@dataclass(frozen=True)
class TerminalRenderArgs:
    cohort_csv: Path
    references_root: Path
    output_dir: Path
    urdf_cache_root: Path
    max_terminal_drift_m: float
    view_width: int
    view_height: int


def parse_args() -> TerminalRenderArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--references-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--urdf-cache-root", type=Path, required=True)
    parser.add_argument("--max-terminal-drift-m", type=float, default=0.5)
    parser.add_argument("--view-width", type=int, default=360)
    parser.add_argument("--view-height", type=int, default=270)
    parsed = parser.parse_args()
    if parsed.max_terminal_drift_m <= 0.0:
        raise ValueError(
            "Expected max_terminal_drift_m > 0, got "
            f"{parsed.max_terminal_drift_m}."
        )
    if parsed.view_width <= 0 or parsed.view_height <= 0:
        raise ValueError(
            "Expected positive view dimensions, got "
            f"{parsed.view_width}x{parsed.view_height}."
        )
    return TerminalRenderArgs(
        cohort_csv=parsed.cohort_csv.resolve(),
        references_root=parsed.references_root.resolve(),
        output_dir=parsed.output_dir.resolve(),
        urdf_cache_root=parsed.urdf_cache_root.resolve(),
        max_terminal_drift_m=float(parsed.max_terminal_drift_m),
        view_width=int(parsed.view_width),
        view_height=int(parsed.view_height),
    )


def _manifest_records(references_root: Path) -> dict[int, dict[str, object]]:
    path = references_root / "grasp_refinement_manifest.json"
    manifest = json.loads(path.read_text())
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Expected records list in {path}.")
    result: dict[int, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("index"), int):
            raise ValueError(f"Malformed manifest record: {record!r}.")
        index = int(record["index"])
        if index in result:
            raise ValueError(f"Duplicate manifest index {index}.")
        result[index] = record
    return result


def _local_reference_path(
    record: dict[str, object], references_root: Path
) -> Path:
    reference = record.get("reference_path")
    if not isinstance(reference, str):
        raise ValueError(f"Manifest record lacks reference_path: {record!r}.")
    parts = Path(reference).parts
    try:
        root_index = parts.index(REFERENCE_ROOT_NAME)
    except ValueError as exc:
        raise ValueError(
            f"Expected {REFERENCE_ROOT_NAME!r} in reference path {reference!r}."
        ) from exc
    local_path = references_root.joinpath(*parts[root_index + 1 :])
    if not local_path.is_file():
        raise FileNotFoundError(f"Reference not found: {local_path}")
    return local_path


def _terminal_index(done: np.ndarray) -> int:
    if done.ndim != 2 or done.shape[1] < 1:
        raise ValueError(f"Expected done shape (T, E) with E >= 1, got {done.shape}.")
    indices = np.flatnonzero(done[:, 0])
    if indices.size != 1:
        raise ValueError(
            "Expected exactly one terminal transition for rendered environment zero, "
            f"got indices {indices.tolist()}."
        )
    return int(indices[0])


def _terminal_motion(reference_path: Path, rollout_path: Path) -> dict:
    motion = _load_exported_npy_payload(reference_path)
    with np.load(rollout_path, allow_pickle=False) as rollout:
        terminal_idx = _terminal_index(np.asarray(rollout["done"], dtype=bool))
        object_pos = np.asarray(rollout["object_pos"], dtype=np.float32)
        object_quat = np.asarray(rollout["object_quat"], dtype=np.float32)
        object_arti = np.asarray(rollout["object_arti"], dtype=np.float32)
        if object_pos.ndim != 4 or object_pos.shape[2:] != (1, 3):
            raise ValueError(
                "Expected single-object position rollout shape (T, E, 1, 3), got "
                f"{object_pos.shape}."
            )
        if object_quat.shape != (*object_pos.shape[:3], 4):
            raise ValueError(
                "Expected object quaternion shape (T, E, 1, 4), got "
                f"{object_quat.shape}."
            )
        if object_arti.ndim != 4 or object_arti.shape[:3] != object_pos.shape[:3]:
            raise ValueError(
                "Expected object articulation shape (T, E, 1, A), got "
                f"{object_arti.shape}."
            )

        motion["retargeted_obj_pos"] = object_pos[terminal_idx, 0, 0][None]
        motion["retargeted_obj_quat"] = object_quat[terminal_idx, 0, 0][None]
        motion["retargeted_obj_arti"] = object_arti[terminal_idx, 0, 0][None]
        for side in ("left", "right"):
            key = f"{side}_jvals"
            rollout_key = f"{side}_jvals"
            if rollout_key in rollout.files:
                joint_values = np.asarray(rollout[rollout_key], dtype=np.float32)
                if joint_values.ndim != 3:
                    raise ValueError(
                        f"Expected {rollout_key} shape (T, E, J), got "
                        f"{joint_values.shape}."
                    )
                motion[key] = joint_values[terminal_idx, 0][None]
    return motion


def _add_title(image: Image.Image, title: str) -> Image.Image:
    title_height = 32
    canvas = Image.new(
        "RGB", (image.width, image.height + title_height), color=(17, 24, 39)
    )
    canvas.paste(image, (0, title_height))
    ImageDraw.Draw(canvas).text((10, 8), title, fill=(255, 255, 255))
    return canvas


def main() -> int:
    args = parse_args()
    records = _manifest_records(args.references_root)
    with args.cohort_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No cohort rows found in {args.cohort_csv}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.urdf_cache_root.mkdir(parents=True, exist_ok=True)
    rendered_rows: list[dict[str, object]] = []
    for row in rows:
        terminal_drift_m = float(row["terminal_drift_m"])
        if terminal_drift_m > args.max_terminal_drift_m:
            continue
        index = int(row["index"])
        if index not in records:
            raise KeyError(f"Cohort index {index} is absent from the manifest.")
        reference_path = _local_reference_path(records[index], args.references_root)
        rollout_path = (
            args.cohort_csv.parent / row["run_name"] / "rollouts" / "chunk_0000.npz"
        )
        if not rollout_path.is_file():
            raise FileNotFoundError(f"Rollout not found: {rollout_path}")

        motion = _terminal_motion(reference_path, rollout_path)
        instances = _build_bimanual_frame_instances(
            motion=motion,
            frame_idx=0,
            mesh_type="visual",
            urdf_cache_root=args.urdf_cache_root,
        )
        image = render_centered_four_views(
            instances,
            image_width=args.view_width,
            image_height=args.view_height,
            camera_distance_m=0.55,
            camera_elevation_m=0.25,
        )
        success = float(row["grasp_stability_success"]) >= 1.0
        title = (
            f"{index:03d} {row['category']} | "
            f"{'PASS' if success else 'FAIL'} | terminal drift "
            f"{terminal_drift_m * 1000.0:.1f} mm"
        )
        image = _add_title(image, title)
        output_path = args.output_dir / f"{index:03d}_{row['category']}_terminal.png"
        image.save(output_path)
        rendered_rows.append(
            {
                "index": index,
                "category": row["category"],
                "template_id": row["template_id"],
                "success": int(success),
                "terminal_drift_m": terminal_drift_m,
                "render_path": str(output_path),
            }
        )
        print(f"[terminal-render] {output_path}", flush=True)

    manifest_path = args.output_dir / "terminal_render_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rendered_rows[0]))
        writer.writeheader()
        writer.writerows(rendered_rows)
    print(
        f"[terminal-render] rendered {len(rendered_rows)}/{len(rows)} states with "
        f"drift <= {args.max_terminal_drift_m:.3f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
