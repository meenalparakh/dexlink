from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from urdfpy import URDF

from dexlink.utils.retargeting.path_utils import get_assets_dir, process_urdf
from dexlink.src.reference.arctic.assemble_transfer_motion import (
    _target_first_frame_bounds,
)
from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE
from dexlink.src.reference.arctic.transfer_config import (
    ARCTIC_TRANSFER_SPECS,
    ArcticTransferSpec,
    get_transfer,
)


@dataclass(frozen=True)
class SupportAudit:
    transfer_id: str
    support_size_m: list[float]
    support_position_m: list[float]
    target_first_frame_bounds_m: list[list[float]]
    initial_vertical_gap_m: float
    footprint_overlap_fraction: float
    first_lift_frame: int | None
    early_hand_support_intrusion_max_m: float
    lifted_hand_support_intrusion_max_m: float
    intrusion_frame_count: int
    intrusion_frames_over_0_5mm: list[int]


def _load_sequence(transfer: ArcticTransferSpec, workspace: Path) -> dict[str, np.ndarray]:
    path = (
        workspace
        / "transfer_trajectories"
        / transfer.transfer_id
        / "exports"
        / f"{transfer.transfer_id}_oakink_sequence.npy"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Transfer sequence does not exist: {path}.")
    raw = np.load(path, allow_pickle=True)
    if not isinstance(raw, np.ndarray) or raw.shape != ():
        raise ValueError(
            f"Expected scalar transfer payload at {path}, got shape "
            f"{getattr(raw, 'shape', None)}."
        )
    payload = raw.item()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict transfer payload at {path}.")
    return payload


def _footprint_overlap_fraction(
    target_bounds: np.ndarray, support_position: np.ndarray, support_size: np.ndarray
) -> float:
    support_low = support_position[:2] - 0.5 * support_size[:2]
    support_high = support_position[:2] + 0.5 * support_size[:2]
    overlap = np.maximum(
        np.minimum(target_bounds[1, :2], support_high)
        - np.maximum(target_bounds[0, :2], support_low),
        0.0,
    )
    support_area = float(np.prod(support_size[:2]))
    return float(np.prod(overlap) / support_area)


def _hand_support_intrusion_by_frame(
    payload: dict[str, np.ndarray],
    support_position: np.ndarray,
    support_size: np.ndarray,
) -> np.ndarray:
    support_low = support_position - 0.5 * support_size
    support_high = support_position + 0.5 * support_size
    frame_count = int(np.asarray(payload["reference_obj_pos"]).shape[0])
    maximum_depth = np.zeros(frame_count, dtype=np.float64)

    for side in ("left", "right"):
        names = np.asarray(payload[f"{side}_jnames"], dtype=object)
        values = np.asarray(payload[f"{side}_jvals"], dtype=np.float64)
        if values.shape != (frame_count, names.size):
            raise ValueError(
                f"Expected {side} joint values shape {(frame_count, names.size)}, "
                f"got {values.shape}."
            )
        urdf_path = process_urdf(
            get_assets_dir()
            / "robots"
            / "xhand"
            / f"xhand_{side}_6dof_abs_obj.urdf"
        )
        hand = URDF.load(str(urdf_path))
        for frame_index, frame_values in enumerate(values):
            config = dict(
                zip(names.tolist(), frame_values.astype(float).tolist(), strict=True)
            )
            mesh_fk = hand.collision_trimesh_fk(cfg=config)
            for mesh, transform in mesh_fk.items():
                vertices = np.asarray(mesh.vertices, dtype=np.float64)
                vertices_h = np.concatenate(
                    [vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)],
                    axis=1,
                )
                points = (np.asarray(transform) @ vertices_h.T).T[:, :3]
                inside = np.logical_and(
                    points > support_low[None, :], points < support_high[None, :]
                ).all(axis=1)
                if not inside.any():
                    continue
                inside_points = points[inside]
                face_distances = np.minimum(
                    inside_points - support_low[None, :],
                    support_high[None, :] - inside_points,
                )
                maximum_depth[frame_index] = max(
                    maximum_depth[frame_index],
                    float(face_distances.min(axis=1).max()),
                )
    return maximum_depth


def audit_one(transfer: ArcticTransferSpec, workspace: Path) -> SupportAudit:
    payload = _load_sequence(transfer, workspace)
    support_positions = np.asarray(payload["scene_object_pos"][:, 1], dtype=np.float64)
    if support_positions.ndim != 2 or support_positions.shape[1] != 3:
        raise ValueError(
            f"Expected support positions shape (T, 3), got {support_positions.shape}."
        )
    if not np.allclose(support_positions, support_positions[0], atol=1.0e-8):
        raise ValueError(f"Support moves in transfer {transfer.transfer_id!r}.")
    support_position = support_positions[0]
    support_size = np.asarray(transfer.support_block_size, dtype=np.float64)
    target_path = process_urdf(Path(str(payload["target_urdf_path"])))
    target_bounds = _target_first_frame_bounds(payload, target_path)
    support_top = support_position[2] + 0.5 * support_size[2]
    vertical_gap = float(target_bounds[0, 2] - support_top)

    object_positions = np.asarray(payload["reference_obj_pos"], dtype=np.float64)
    lifted = object_positions[:, 2] > object_positions[0, 2] + 0.02
    lift_indices = np.flatnonzero(lifted)
    intrusion = _hand_support_intrusion_by_frame(
        payload, support_position, support_size
    )
    return SupportAudit(
        transfer_id=transfer.transfer_id,
        support_size_m=support_size.tolist(),
        support_position_m=support_position.tolist(),
        target_first_frame_bounds_m=target_bounds.tolist(),
        initial_vertical_gap_m=vertical_gap,
        footprint_overlap_fraction=_footprint_overlap_fraction(
            target_bounds, support_position, support_size
        ),
        first_lift_frame=None if lift_indices.size == 0 else int(lift_indices[0]),
        early_hand_support_intrusion_max_m=float(intrusion[:10].max()),
        lifted_hand_support_intrusion_max_m=(
            float(intrusion[lifted].max()) if lifted.any() else 0.0
        ),
        intrusion_frame_count=int(np.count_nonzero(intrusion > 5.0e-4)),
        intrusion_frames_over_0_5mm=np.flatnonzero(intrusion > 5.0e-4)
        .astype(int)
        .tolist(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit support placement and hand intrusion for ARCTIC transfers."
    )
    parser.add_argument(
        "--transfer",
        action="append",
        choices=tuple(ARCTIC_TRANSFER_SPECS),
        dest="transfers",
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/arctic_evals/transfer_support_audit.json"),
    )
    raw = parser.parse_args(argv)
    transfers = tuple(raw.transfers or ARCTIC_TRANSFER_SPECS)
    results = [
        audit_one(get_transfer(transfer_id), raw.workspace.resolve())
        for transfer_id in transfers
    ]
    payload = {
        "protocol": (
            "First-frame posed target bounds; exact axis-aligned support overlap; "
            "collision-mesh hand vertices tested against the support interior."
        ),
        "results": [asdict(result) for result in results],
    }
    output = raw.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
