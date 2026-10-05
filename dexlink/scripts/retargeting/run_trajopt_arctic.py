#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dexlink.scripts.retargeting.retarget_probe_utils import (
    RetargetSolveTiming,
    export_retarget_probe_objs,
    write_retarget_timing_json,
)
from dexlink.src.retargeting.config.trajopt_final import (
    config_to_dict,
    extract_overrides,
    load_experiment_config,
)
from dexlink.src.retargeting.data.hand_cfgs.inspire import INSPIRE_CFGs
from dexlink.src.retargeting.data.hand_cfgs.xhand import XHAND_CFGs
from dexlink.src.retargeting.data.process_arctic import get_processor_cfg
from dexlink.src.retargeting.optimization.hand_bps import build_hand_bps_jpos_dict
from dexlink.src.retargeting.optimization.motion_sequence_optimizer import DrakeSequenceOptimizer
from dexlink.src.retargeting.optimization.optimizer_final import DrakeOptimizer, load_q_solution_from_npz
from dexlink.utils.retargeting.path_utils import process_urdf


def _normalize_robot_model(robot_model: str) -> str:
    aliases = {
        "inspire": "inspire",
        "inspire_hand": "inspire",
        "xhand": "xhand",
    }
    value = str(robot_model).lower()
    if value not in aliases:
        raise ValueError(
            f"Unsupported robot_model {robot_model!r}; use 'inspire_hand' or 'xhand'."
        )
    return aliases[value]


def get_robot_spec(side: str, robot_model: str):
    robot_model = _normalize_robot_model(robot_model)
    if robot_model == "inspire":
        cfgs = INSPIRE_CFGs
    elif robot_model == "xhand":
        cfgs = XHAND_CFGs
    else:
        raise ValueError(
            f"Unsupported robot_model {robot_model!r}; use 'inspire_hand' or 'xhand'."
        )
    left_robot_urdf = process_urdf(cfgs["left"]["urdf_path"])
    right_robot_urdf = process_urdf(cfgs["right"]["urdf_path"])

    if side == "left":
        return left_robot_urdf, cfgs["left"]
    if side == "right":
        return right_robot_urdf, cfgs["right"]
    raise ValueError(f"Unsupported side {side!r}; use 'left' or 'right'.")


def _normalize_hand(hand: str) -> str:
    aliases = {
        "left": "left",
        "l": "left",
        "right": "right",
        "r": "right",
    }
    value = str(hand).lower()
    if value not in aliases:
        raise ValueError(f"Unsupported hand {hand!r}; use 'left' or 'right'.")
    return aliases[value]


def _resolve_local_timestep(arctic_cfg) -> int:
    start = arctic_cfg.start_timestep
    end = arctic_cfg.end_timestep
    skip = arctic_cfg.skip
    timestep = arctic_cfg.timestep

    if skip <= 0:
        raise ValueError(f"`arctic.skip` must be > 0, got {skip}.")
    if timestep < start:
        raise ValueError(
            f"`arctic.timestep` ({timestep}) must be >= `arctic.start_timestep` ({start})."
        )
    if end != -1 and timestep >= end:
        raise ValueError(
            f"`arctic.timestep` ({timestep}) must be < `arctic.end_timestep` ({end})."
        )
    if (timestep - start) % skip != 0:
        raise ValueError(
            f"`arctic.timestep` ({timestep}) is not aligned with "
            f"`arctic.start_timestep` ({start}) and `arctic.skip` ({skip})."
        )
    return (timestep - start) // skip


def _get_hand_data(demo, hand: str, local_timestep: int):
    joint_attr = f"relevant_{hand}_joint_pos"
    verts_attr = f"{hand}_hand_verts"

    joint_pos = getattr(demo, joint_attr)
    hand_verts = getattr(demo, verts_attr)
    if not joint_pos:
        raise ValueError(f"No joint positions found for hand={hand!r}.")
    if hand_verts is None:
        raise ValueError(f"No hand vertices found for hand={hand!r}.")

    jpos_dict = {key: value[local_timestep] for key, value in joint_pos.items()}
    reference_pcd_dict = {f"{hand}_verts": hand_verts[local_timestep]}
    return jpos_dict, reference_pcd_dict


def _object_articulation_for_frame(
    demo,
    local_timestep: int,
) -> tuple[list[str], np.ndarray]:
    motion = demo.object_reference_motion
    joint_names = [str(name) for name in motion.get("joint_names", [])]
    obj_arti = motion.get("obj_arti")
    if obj_arti is None:
        return joint_names, np.zeros((len(joint_names),), dtype=np.float32)
    values = np.asarray(obj_arti, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(
            f"Expected ARCTIC obj_arti shape (T, n_joints), got {values.shape}."
        )
    if local_timestep < 0 or local_timestep >= values.shape[0]:
        raise ValueError(
            f"Local timestep {local_timestep} is out of bounds for obj_arti with "
            f"{values.shape[0]} frames."
        )
    if values.shape[1] != len(joint_names):
        raise ValueError(
            f"Expected obj_arti width to match joint_names, got {values.shape[1]} "
            f"values and {len(joint_names)} names."
        )
    return joint_names, values[local_timestep]


def _motion_metadata(arctic_cfg, export_dir: Path) -> dict[str, object]:
    return {
        "source_dataset": "arctic",
        "arctic_demo_path": str(arctic_cfg.demo_path),
        "arctic_target_object": arctic_cfg.target_object,
        "arctic_timestep": arctic_cfg.timestep,
        "arctic_start_timestep": arctic_cfg.start_timestep,
        "arctic_end_timestep": arctic_cfg.end_timestep,
        "arctic_skip": arctic_cfg.skip,
        "arctic_hand": arctic_cfg.hand,
        "arctic_run_dir": str(export_dir),
        "arctic_target_obj_path": arctic_cfg.target_obj_path,
        "arctic_target_transform_info": arctic_cfg.target_transform_info,
        "arctic_target_scale": arctic_cfg.target_scale,
        "traj_robot_model": arctic_cfg.robot_model,
        "retarget_only": True,
    }


def _write_retarget_probes(
    *,
    export_dir: Path,
    drake_optimizer: DrakeOptimizer,
    target_obj_path: Path,
    target_pose: np.ndarray,
    robot_urdf_path: str | Path,
    q_solution: np.ndarray,
) -> None:
    timing_history = getattr(drake_optimizer, "solve_timing_history", [])
    main_solve_time_sec = (
        sum(float(item["main_solve_time_sec"]) for item in timing_history)
        if timing_history
        else None
    )
    obj_paths = export_retarget_probe_objs(
        output_dir=export_dir,
        target_obj_path=target_obj_path,
        target_pose=target_pose,
        robot_urdf_path=robot_urdf_path,
        dof_names=drake_optimizer.dof_names,
        q_solution=q_solution,
    )
    timing_path = write_retarget_timing_json(
        output_dir=export_dir,
        timing=RetargetSolveTiming(
            main_solve_time_sec=main_solve_time_sec,
            solve_attempts=len(timing_history),
            eval_count=(
                int(getattr(drake_optimizer.logger, "eval_count"))
                if getattr(drake_optimizer, "logger", None) is not None
                else None
            ),
        ),
        extra={
            "solve_history": timing_history,
            "probe_objs": obj_paths,
        },
    )
    print(f"Saved retarget timing/probe files to {timing_path.parent}")


def _try_write_retarget_probes(**kwargs) -> None:
    export_dir = Path(kwargs["export_dir"])
    try:
        _write_retarget_probes(**kwargs)
    except Exception as exc:
        error_path = export_dir / "retgt_probe_error.json"
        error_path.write_text(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"WARNING: failed to save retarget probe OBJs: {exc}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the legacy Arctic Drake trajectory optimizer with config overrides."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="arctic_default",
        help="Preset name from dexlink.src.retargeting.config.trajopt_final.PRESETS",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config and exit.",
    )
    parser.add_argument(
        "--retarget-only",
        action="store_true",
        help="Export a static retargeted-template motion for the selected ARCTIC frame.",
    )
    parser.add_argument(
        "--retarget-only-frames",
        type=int,
        default=20,
        help="Number of repeated frames to export for the selected ARCTIC template.",
    )
    parser.add_argument(
        "--save-retarget-probes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save retgt_timing.json plus target/hand/overlay OBJ probes after "
            "retargeting. This is a sidecar diagnostic and does not affect optimization."
        ),
    )
    args, unknown = parser.parse_known_args(argv)

    overrides = extract_overrides(unknown)
    print(f"Applying config overrides: {overrides}")
    cfg = load_experiment_config(preset=args.preset, overrides=overrides)

    arctic_cfg = cfg.arctic
    opt_cfg = cfg.trajopt
    arctic_cfg.robot_model = _normalize_robot_model(arctic_cfg.robot_model)

    if args.print_config:
        from pprint import pprint

        pprint(config_to_dict(cfg))
        return

    hand = _normalize_hand(arctic_cfg.hand)
    local_timestep = _resolve_local_timestep(arctic_cfg)

    processor_cfg = get_processor_cfg(arctic_cfg.target_object)
    demo_class = processor_cfg["processor_cls"]
    reference_urdf = processor_cfg["reference_urdf"]
    target_urdf_input = processor_cfg["target_urdf_path"]

    print(f"Reference urdf: {reference_urdf}")
    print(f"Target urdf: {target_urdf_input}")

    demo = demo_class(
        original_demo_path=str(arctic_cfg.demo_path),
        target_object=arctic_cfg.target_object,
        start_timestep=arctic_cfg.start_timestep,
        end_timestep=arctic_cfg.end_timestep,
        skip=arctic_cfg.skip,
        target_urdf=target_urdf_input,
        reference_urdf=reference_urdf,
        target_obj_path=arctic_cfg.target_obj_path,
        target_transform_info=arctic_cfg.target_transform_info,
        target_scale=arctic_cfg.target_scale,
        run_dir=cfg.run_dir,
    )

    original_asset_obj, ori_pose = demo.get_trimesh_at_timestep(
        timestep=local_timestep,
        obj_type="original",
        mesh_type=arctic_cfg.mesh_type,
    )
    target_asset_obj, tgt_pose = demo.get_trimesh_at_timestep(
        timestep=local_timestep,
        obj_type="target",
        mesh_type=arctic_cfg.mesh_type,
    )

    export_dir = Path(cfg.run_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    original_obj_path = (
        export_dir
        / f"{arctic_cfg.target_object}_{hand}_t{arctic_cfg.timestep:04d}_original.obj"
    )
    target_obj_path = (
        export_dir
        / f"{arctic_cfg.target_object}_{hand}_t{arctic_cfg.timestep:04d}_target.obj"
    )
    original_asset_obj.copy().export(original_obj_path)
    target_asset_obj.copy().export(target_obj_path)

    jpos_dict, reference_pcd_dict = _get_hand_data(demo, hand, local_timestep)
    robot_urdf_path, robot_cfg = get_robot_spec(
        side=hand, robot_model=arctic_cfg.robot_model
    )
    hand_bps_jpos_dict = build_hand_bps_jpos_dict(
        demo=demo,
        hand=hand,
        robot_cfg=robot_cfg,
        hand_jpos_dict=jpos_dict,
        frame_idx=local_timestep,
    )

    drake_optimizer = DrakeOptimizer(
        cfg=opt_cfg,
        reference_obj_path=original_obj_path,
        target_obj_path=target_obj_path,
        robot_urdf_path=robot_urdf_path,
        robot_cfg=robot_cfg,
        ori_pose=ori_pose,
        tgt_pose=tgt_pose,
        hand_jpos_dict=jpos_dict,
        hand_bps_jpos_dict=hand_bps_jpos_dict,
        optimization_run_dir=export_dir,
    )

    if opt_cfg.initial_solution_npz is not None:
        q_initial = load_q_solution_from_npz(
            opt_cfg.initial_solution_npz,
            drake_optimizer.dof_names,
            hand_side=hand,
            frame_idx=opt_cfg.initial_solution_frame,
        )
        q_solution = drake_optimizer.refine_solution(q_initial)
    else:
        q_solution = drake_optimizer.optimize()

    if opt_cfg.visualize:
        drake_optimizer.plant_with_visualizer(
            q_robot=q_solution,
            pcd_dict=reference_pcd_dict,
            pause=opt_cfg.pause_after_visualize,
        )

    sequence_optimizer = DrakeSequenceOptimizer(
        seq_optcfg=cfg.trajopt_seq,
        cfg=opt_cfg,
        reference_obj_path=original_obj_path,
        target_obj_path=target_obj_path,
        robot_urdf_path=robot_urdf_path,
        robot_cfg=robot_cfg,
        ori_pose=ori_pose,
        tgt_pose=tgt_pose,
        hand_jpos_dict=jpos_dict,
        hand_bps_jpos_dict=hand_bps_jpos_dict,
        optimization_run_dir=export_dir,
    )
    object_arti_names, object_arti_vals = _object_articulation_for_frame(
        demo,
        local_timestep,
    )
    sequence_motion_path = export_dir / "sequence_motion_lineseq.npz"
    saved_motion_path = sequence_optimizer.export_static_template_motion(
        q_solution,
        sequence_motion_path,
        hold_frames=args.retarget_only_frames,
        object_name=demo.target_obj_name,
        object_urdf_path=demo.target_obj_urdf_path,
        object_arti_names=object_arti_names,
        object_arti_vals=object_arti_vals,
        hand_name=arctic_cfg.robot_model,
        motion_metadata=_motion_metadata(arctic_cfg, export_dir),
    )
    print(f"Saved ARCTIC retargeted template motion to {saved_motion_path}")
    if args.save_retarget_probes:
        _try_write_retarget_probes(
            export_dir=export_dir,
            drake_optimizer=drake_optimizer,
            target_obj_path=target_obj_path,
            target_pose=tgt_pose,
            robot_urdf_path=robot_urdf_path,
            q_solution=q_solution,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
