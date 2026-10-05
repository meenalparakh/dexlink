from __future__ import annotations

import numpy as np


def build_hand_bps_jpos_dict(
    *,
    demo,
    hand: str,
    robot_cfg: dict,
    hand_jpos_dict: dict[str, np.ndarray],
    frame_idx: int,
) -> dict[str, np.ndarray]:
    human_robot_kpt_map = robot_cfg.get("human_robot_kpt_map")
    if human_robot_kpt_map is None:
        return {name: np.asarray(value) for name, value in hand_jpos_dict.items()}

    if not hasattr(demo, "get_human_hand_kpt_positions"):
        raise ValueError(
            "robot_cfg defines human_robot_kpt_map, but the demo processor does not "
            "provide get_human_hand_kpt_positions()."
        )

    requested_names = tuple(human_robot_kpt_map.keys())
    surface_kpt_names = [name for name in requested_names if name not in hand_jpos_dict]
    surface_kpt_pos = demo.get_human_hand_kpt_positions(
        hand,
        names=surface_kpt_names,
        frame_idx=frame_idx,
    )

    bps_jpos_dict = {}
    for name in requested_names:
        if name in hand_jpos_dict:
            value = hand_jpos_dict[name]
        elif name in surface_kpt_pos:
            value = surface_kpt_pos[name]
        else:
            raise ValueError(
                f"Could not resolve BPS keypoint {name!r}. Expected it in the demo "
                "joint dict or get_human_hand_kpt_positions()."
            )
        point = np.asarray(value)
        if point.shape != (3,):
            raise ValueError(
                f"Expected BPS keypoint {name!r} to have shape (3,), "
                f"got {point.shape}."
            )
        bps_jpos_dict[name] = point

    return bps_jpos_dict
