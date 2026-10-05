from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# These are the time-indexed fields in the OakInk sequence export consumed by RL.
# Listing them explicitly prevents a coincidentally matching metadata array from
# being extended silently.
TIME_INDEXED_FIELDS = {
    "left_mano_verts",
    "right_mano_verts",
    "left_mano_joints",
    "right_mano_joints",
    "left_reference_keypoint_pos",
    "right_reference_keypoint_pos",
    "left_human_reference_keypoint_pos",
    "right_human_reference_keypoint_pos",
    "left_human_reference_keypoint_valid",
    "right_human_reference_keypoint_valid",
    "reference_obj_pos",
    "reference_obj_quat",
    "reference_obj_rot",
    "reference_obj_arti",
    "left_jvals",
    "right_jvals",
    "retargeted_obj_pos",
    "retargeted_obj_quat",
    "retargeted_obj_rot",
    "retargeted_obj_arti",
    "scene_object_pos",
    "scene_object_quat",
    "scene_object_rot",
    "scene_object_arti_vals",
    "scene_object_active_mask",
}

REFINEMENT_TIME_INDEXED_FIELDS = {
    "drake_collision_refinement_right_jvals",
    "drake_collision_refinement_jvals",
    "drake_collision_refinement_unresolved_frame_mask",
    "drake_collision_refinement_valid_frame_mask",
    "drake_collision_refinement_invalid_frame_mask",
    "drake_collision_refinement_final_collision_violation",
    "drake_collision_refinement_acceptance_collision_violation",
    "drake_collision_refinement_radius_bound_violation",
}

REFINEMENT_INDEX_FIELDS = {
    "drake_collision_refinement_unresolved_frame_indices",
    "drake_collision_refinement_valid_frame_indices",
    "drake_collision_refinement_invalid_frame_indices",
}


def extend_terminal_hold(
    payload: dict[str, object],
    *,
    final_phase_name: str,
    expected_current_frames: int,
    target_frames: int,
) -> dict[str, object]:
    """Extend a final hold phase by repeating the exact terminal scene state."""
    if target_frames <= expected_current_frames:
        raise ValueError(
            f"Expected target_frames > {expected_current_frames}, got {target_frames}."
        )
    phase_names = np.asarray(payload["phase_names"])
    phase_lengths = np.asarray(payload["phase_lengths"])
    if phase_names.ndim != 1 or phase_lengths.shape != phase_names.shape:
        raise ValueError(
            "Expected matching one-dimensional phase_names and phase_lengths, got "
            f"{phase_names.shape} and {phase_lengths.shape}."
        )
    if str(phase_names[-1]) != final_phase_name:
        raise ValueError(
            f"Expected final phase {final_phase_name!r}, got {phase_names[-1]!r}."
        )
    if int(phase_lengths[-1]) != expected_current_frames:
        raise ValueError(
            f"Expected {expected_current_frames} existing frames in final phase "
            f"{final_phase_name!r}, got {int(phase_lengths[-1])}."
        )

    num_steps = int(phase_lengths.sum())
    discovered_time_fields = {
        key
        for key, value in payload.items()
        if isinstance(value, np.ndarray)
        and value.ndim >= 1
        and value.shape[0] == num_steps
    }
    if not TIME_INDEXED_FIELDS.issubset(discovered_time_fields):
        raise ValueError(
            "Missing time-indexed OakInk sequence fields. Expected at least "
            f"{sorted(TIME_INDEXED_FIELDS)}, got {sorted(discovered_time_fields)}."
        )
    has_refinement = "drake_collision_refinement_end_frame" in payload
    optional_time_fields = (
        REFINEMENT_TIME_INDEXED_FIELDS if has_refinement else set()
    )
    known_time_fields = TIME_INDEXED_FIELDS | optional_time_fields
    unexpected_time_fields = discovered_time_fields - (
        known_time_fields | REFINEMENT_INDEX_FIELDS
    )
    if unexpected_time_fields:
        raise ValueError(
            "Unexpected time-indexed OakInk sequence fields: "
            f"{sorted(unexpected_time_fields)}."
        )
    if has_refinement:
        missing_refinement_fields = (
            REFINEMENT_TIME_INDEXED_FIELDS | REFINEMENT_INDEX_FIELDS
        ) - payload.keys()
        if missing_refinement_fields:
            raise ValueError(
                "Collision-refined sequence is missing refinement fields: "
                f"{sorted(missing_refinement_fields)}."
            )
        refinement_end = int(
            np.asarray(payload["drake_collision_refinement_end_frame"]).item()
        )
        if refinement_end != num_steps:
            raise ValueError(
                f"Expected collision refinement to end at frame {num_steps}, "
                f"got {refinement_end}."
            )

    extra_frames = target_frames - expected_current_frames
    extended = dict(payload)
    for key in sorted(known_time_fields):
        value = np.asarray(payload[key])
        extended[key] = np.concatenate(
            (value, np.repeat(value[-1:], extra_frames, axis=0)), axis=0
        )
    extended_lengths = phase_lengths.copy()
    extended_lengths[-1] = target_frames
    extended["phase_lengths"] = extended_lengths
    if has_refinement:
        extended["drake_collision_refinement_end_frame"] = np.asarray(
            num_steps + extra_frames,
            dtype=np.asarray(payload["drake_collision_refinement_end_frame"]).dtype,
        )
        for state in ("unresolved", "valid", "invalid"):
            mask_key = f"drake_collision_refinement_{state}_frame_mask"
            index_key = f"drake_collision_refinement_{state}_frame_indices"
            extended[index_key] = np.flatnonzero(extended[mask_key]).astype(
                np.asarray(payload[index_key]).dtype,
                copy=False,
            )
    return extended


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend the final hold phase of an exported OakInk sequence."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-phase-name", required=True)
    parser.add_argument("--expected-current-frames", type=int, required=True)
    parser.add_argument("--target-frames", type=int, required=True)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output paths must differ to preserve the source.")
    raw = np.load(args.input, allow_pickle=True)
    if raw.shape != ():
        raise ValueError(
            f"Expected a scalar object array in {args.input}, got shape {raw.shape}."
        )
    payload = raw.item()
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a dictionary payload in {args.input}, got {type(payload).__name__}."
        )
    extended = extend_terminal_hold(
        payload,
        final_phase_name=args.final_phase_name,
        expected_current_frames=args.expected_current_frames,
        target_frames=args.target_frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, extended, allow_pickle=True)
    print(
        f"Extended {args.input} from {sum(payload['phase_lengths'])} to "
        f"{sum(extended['phase_lengths'])} frames and wrote {args.output}."
    )


if __name__ == "__main__":
    main()
