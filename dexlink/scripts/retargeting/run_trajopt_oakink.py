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
from dexlink.src.retargeting.optimization.hand_bps import build_hand_bps_jpos_dict
from dexlink.src.retargeting.optimization.motion_sequence_optimizer import DrakeSequenceOptimizer
from dexlink.src.retargeting.optimization.oakink_processor import OakInkProcessor
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
    else:
        cfgs = XHAND_CFGs
    left_robot_urdf = process_urdf(
        cfgs["left"]["urdf_path"], append_jobname=True
    )
    right_robot_urdf = process_urdf(
        cfgs["right"]["urdf_path"], append_jobname=True
    )

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


def _get_hand_data(demo, hand: str):
    joint_attr = f"relevant_{hand}_joint_pos"
    verts_attr = f"{hand}_hand_verts"

    joint_pos = getattr(demo, joint_attr)
    hand_verts = getattr(demo, verts_attr)
    if not joint_pos:
        raise ValueError(f"No joint positions found for hand={hand!r}.")
    if hand_verts is None:
        raise ValueError(f"No hand vertices found for hand={hand!r}.")

    jpos_dict = {key: value[0] for key, value in joint_pos.items()}
    reference_pcd_dict = {f"{hand}_verts": hand_verts[0]}
    return jpos_dict, reference_pcd_dict


def _save_sequence_result(result: dict, export_dir: Path) -> Path:
    out_path = export_dir / "sequence_result.npz"
    payload = {
        "control_points": np.asarray(result["control_points"], dtype=float),
    }
    if result.get("duration") is not None:
        payload["duration"] = np.asarray([float(result["duration"])], dtype=float)
    if result.get("n_reach_frames") is not None:
        payload["n_reach_frames"] = np.asarray(
            [int(result["n_reach_frames"])], dtype=int
        )
    if result.get("n_lift_frames") is not None:
        payload["n_lift_frames"] = np.asarray([int(result["n_lift_frames"])], dtype=int)
    if result.get("contains_lift") is not None:
        payload["contains_lift"] = np.asarray(
            [bool(result["contains_lift"])], dtype=bool
        )
    np.savez(out_path, **payload)
    return out_path


def _sequence_motion_stem(opt_cfg) -> str:
    refinement_tags = []
    if opt_cfg.initial_solution_npz is not None:
        refinement_tags.append("loaded")
    if opt_cfg.non_penetration_refinement:
        refinement_tags.append("nonpen")
    if opt_cfg.sequential_finger_refinement:
        refinement_tags.append("seqfinger")
    if opt_cfg.sequential_line_search_refinement:
        refinement_tags.append("lineseq")
    if opt_cfg.contact_target_refinement:
        refinement_tags.append("contacttarget")

    if not refinement_tags:
        return "sequence_motion"
    return "sequence_motion_" + "_".join(refinement_tags)


def _motion_metadata(
    oakink_cfg,
    demo: OakInkProcessor,
    export_dir: Path,
    target_obj_path: Path,
    target_bps_obj_path: Path | None,
    *,
    target_articulated_urdf_path: Path | None = None,
    retarget_only: bool | None = None,
    articulation_retarget_mode: str | None = None,
    articulation_sample_id: str | None = None,
) -> dict[str, object]:
    metadata = {
        "oakink_demo_path": str(oakink_cfg.demo_path),
        "oakink_target_demo_path": oakink_cfg.target_demo_path,
        "oakink_target_obj_path": str(target_obj_path),
        "oakink_requested_target_obj_path": oakink_cfg.target_obj_path,
        "oakink_target_bps_obj_path": (
            None if target_bps_obj_path is None else str(target_bps_obj_path)
        ),
        "oakink_target_articulated_urdf_path": (
            None
            if target_articulated_urdf_path is None
            else str(target_articulated_urdf_path)
        ),
        "oakink_requested_target_bps_obj_path": oakink_cfg.target_bps_obj_path,
        "oakink_target_transform_info": oakink_cfg.target_transform_info,
        "oakink_target_scale": oakink_cfg.target_scale,
        "oakink_run_dir": str(export_dir),
        "oakink_hand_side": oakink_cfg.oakink_hand_side,
        "oakink_start_timestep": oakink_cfg.start_timestep,
        "oakink_end_timestep": oakink_cfg.end_timestep,
        "oakink_skip": oakink_cfg.skip,
        "traj_robot_model": oakink_cfg.robot_model,
    }
    if retarget_only is not None:
        metadata["retarget_only"] = retarget_only
    if articulation_retarget_mode is not None:
        metadata["articulation_retarget_mode"] = articulation_retarget_mode
    if articulation_sample_id is not None:
        metadata["articulation_sample_id"] = articulation_sample_id
    return metadata


def _parse_object_articulation_values(
    value: str | None,
) -> tuple[tuple[str, ...], np.ndarray | None]:
    if value is None:
        return (), None
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(
            "--object-articulation-joint-values-json must be a non-empty JSON object."
        )
    joint_names = tuple(str(name) for name in parsed)
    joint_values = np.asarray(
        [float(parsed[name]) for name in joint_names],
        dtype=np.float32,
    )
    return joint_names, joint_values


def _export_q_trace_motion(
    sequence_optimizer: DrakeSequenceOptimizer,
    q_trace: list[tuple[str, np.ndarray]],
    *,
    sequence_source: str,
    export_dir: Path,
    sequence_motion_stem: str,
    demo: OakInkProcessor,
    oakink_cfg,
    target_obj_path: Path,
    target_bps_obj_path: Path | None,
    target_articulated_urdf_path: Path | None,
    retarget_only: bool,
    retarget_only_frames: int,
    object_arti_names: tuple[str, ...],
    object_arti_vals: np.ndarray | None,
    articulation_retarget_mode: str | None,
    articulation_sample_id: str | None,
) -> list[Path]:
    saved_paths = []
    for trace_name, q_trace_solution in q_trace:
        motion_path = export_dir / f"{sequence_motion_stem}__{trace_name}.npz"
        metadata = _motion_metadata(
            oakink_cfg,
            demo,
            export_dir,
            target_obj_path,
            target_bps_obj_path,
            target_articulated_urdf_path=target_articulated_urdf_path,
            retarget_only=retarget_only,
            articulation_retarget_mode=articulation_retarget_mode,
            articulation_sample_id=articulation_sample_id,
        )
        metadata["trajopt_trace_name"] = trace_name
        if retarget_only:
            saved_paths.append(
                sequence_optimizer.export_static_template_motion(
                    q_trace_solution,
                    motion_path,
                    hold_frames=retarget_only_frames,
                    object_name=demo.target_obj_name,
                    object_urdf_path=(
                        target_articulated_urdf_path
                        if object_arti_names
                        else demo.target_obj_urdf_path
                    ),
                    object_arti_names=object_arti_names,
                    object_arti_vals=object_arti_vals,
                    hand_name=oakink_cfg.robot_model,
                    motion_metadata=metadata,
                )
            )
            continue

        if sequence_source == "optimize":
            sequence_result = sequence_optimizer.optimize_seq(q_trace_solution)
        elif sequence_source == "single_step":
            sequence_result = sequence_optimizer.construct_unoptimized_trajectory(
                q_trace_solution
            )
        else:
            raise ValueError(
                f"Unsupported trajopt_seq.sequence_source {sequence_source!r}; "
                "use 'optimize' or 'single_step'."
            )
        saved_paths.append(
            sequence_optimizer.export_trajectory(
                sequence_result,
                motion_path,
                object_name=demo.target_obj_name,
                object_urdf_path=(
                    target_articulated_urdf_path
                    if object_arti_names
                    else demo.target_obj_urdf_path
                ),
                hand_name=oakink_cfg.robot_model,
                motion_metadata=metadata,
            )
        )
    return saved_paths


def _preview_target_transform(
    demo: OakInkProcessor,
    drake_optimizer: DrakeOptimizer,
    reference_pcd_dict: dict[str, np.ndarray],
    pause: bool,
) -> None:
    if (demo.target_transform is None) and (demo.target_scale == 1.0):
        return
    print(
        "Previewing target mesh preprocessing before optimization in Drake visualizer."
    )
    q_preview = np.zeros(
        drake_optimizer.plant_f.num_positions(drake_optimizer.hand_model), dtype=float
    )
    drake_optimizer.plant_with_visualizer(
        q_robot=q_preview,
        pcd_dict=reference_pcd_dict,
        pause=pause,
    )


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
        description="Run DrakeOptimizer on an OakInk grasp with config overrides."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="oakink_default",
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
        help=(
            "Skip sequence construction and instead export a static retargeted-template "
            "motion without appending the default lift sequence."
        ),
    )
    parser.add_argument(
        "--retarget-only-frames",
        type=int,
        default=20,
        help="Number of repeated frames to export for --retarget-only.",
    )
    parser.add_argument(
        "--object-articulation-joint-values-json",
        type=str,
        default=None,
        help=(
            "JSON object mapping articulated target joint names to the values used "
            "when materializing the target OBJ/BPS OBJ for this retargeting sample."
        ),
    )
    parser.add_argument("--articulation-retarget-mode", type=str, default=None)
    parser.add_argument("--articulation-sample-id", type=str, default=None)
    parser.add_argument(
        "--save-retarget-probes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save retgt_timing.json plus target/hand/overlay OBJ probes after "
            "retargeting. This is a sidecar diagnostic and does not affect optimization."
        ),
    )
    parser.add_argument(
        "--target-articulated-urdf-path",
        type=Path,
        default=None,
        help=(
            "Articulated target URDF whose joint names match "
            "--object-articulation-joint-values-json. Retargeting may still use a "
            "baked OBJ target, but exported motion stores this URDF for downstream FK."
        ),
    )
    args, unknown = parser.parse_known_args(argv)

    overrides = extract_overrides(unknown)
    print(f"Applying config overrides: {overrides}")
    cfg = load_experiment_config(preset=args.preset, overrides=overrides)

    oakink_cfg = cfg.oakink
    opt_cfg = cfg.trajopt
    seq_opt_cfg = cfg.trajopt_seq
    oakink_cfg.robot_model = _normalize_robot_model(oakink_cfg.robot_model)

    if args.print_config:
        from pprint import pprint

        pprint(config_to_dict(cfg))
        return

    if oakink_cfg.demo_path is None:
        raise ValueError("`oakink.demo_path` is required to run OakInk trajopt.")

    hand = _normalize_hand(oakink_cfg.hand)
    object_arti_names, object_arti_vals = _parse_object_articulation_values(
        args.object_articulation_joint_values_json
    )

    demo = OakInkProcessor(
        original_demo_path=str(oakink_cfg.demo_path),
        run_dir=cfg.run_dir,
        target_demo_path=oakink_cfg.target_demo_path,
        target_obj_path=oakink_cfg.target_obj_path,
        target_transform_info=oakink_cfg.target_transform_info,
        target_scale=oakink_cfg.target_scale,
        oakink_hand_side=oakink_cfg.oakink_hand_side,
        oakink_smplx_model_path=oakink_cfg.oakink_smplx_model_path,
        oakink_mano_assets_root=oakink_cfg.oakink_mano_assets_root,
    )

    export_dir = Path(cfg.run_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    original_obj_path = Path(demo.export_object_obj(obj_type="original"))
    target_obj_path = Path(demo.export_object_obj(obj_type="target"))
    target_bps_obj_path = None
    if oakink_cfg.target_bps_obj_path is not None:
        target_bps_obj_path = Path(
            demo.materialize_target_bps_mesh(oakink_cfg.target_bps_obj_path)
        )
    target_articulated_urdf_path = args.target_articulated_urdf_path
    if object_arti_names and target_articulated_urdf_path is None:
        raise ValueError(
            "--target-articulated-urdf-path is required when "
            "--object-articulation-joint-values-json is provided."
        )
    object_motion_urdf_path = (
        target_articulated_urdf_path if object_arti_names else demo.target_obj_urdf_path
    )
    ori_pose = demo.get_object_base_pose(obj_type="original")
    tgt_pose = demo.get_object_base_pose(obj_type="target")

    jpos_dict, reference_pcd_dict = _get_hand_data(demo, hand)
    robot_urdf_path, robot_cfg = get_robot_spec(
        side=hand, robot_model=oakink_cfg.robot_model
    )
    hand_bps_jpos_dict = build_hand_bps_jpos_dict(
        demo=demo,
        hand=hand,
        robot_cfg=robot_cfg,
        hand_jpos_dict=jpos_dict,
        frame_idx=0,
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
        target_bps_obj_path=target_bps_obj_path,
    )
    # target_obj_path = Path(drake_optimizer.target_obj_path)
    _preview_target_transform(
        demo,
        drake_optimizer,
        reference_pcd_dict,
        pause=opt_cfg.pause_after_visualize,
    )

    if opt_cfg.initial_solution_npz is not None:
        q_initial = load_q_solution_from_npz(
            opt_cfg.initial_solution_npz,
            drake_optimizer.dof_names,
            hand_side=hand,
            frame_idx=opt_cfg.initial_solution_frame,
        )
        q_trace = [
            ("loaded_initial", np.asarray(q_initial, dtype=float).reshape(-1).copy())
        ]
        q_trace.extend(drake_optimizer.refine_solution_trace(q_initial))
    else:
        q_trace = drake_optimizer.optimize_with_trace()
    q_solution = q_trace[-1][1]

    if opt_cfg.visualize:
        drake_optimizer.plant_with_visualizer(
            q_robot=q_solution,
            pcd_dict=reference_pcd_dict,
            pause=opt_cfg.pause_after_visualize,
        )

    sequence_motion_stem = _sequence_motion_stem(opt_cfg)
    sequence_motion_path = export_dir / f"{sequence_motion_stem}.npz"
    sequence_optimizer = DrakeSequenceOptimizer(
        seq_optcfg=seq_opt_cfg,
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
        target_bps_obj_path=target_bps_obj_path,
    )
    sequence_source = str(seq_opt_cfg.sequence_source).lower()
    if args.retarget_only:
        saved_motion_path = sequence_optimizer.export_static_template_motion(
            q_solution,
            sequence_motion_path,
            hold_frames=args.retarget_only_frames,
            object_name=demo.target_obj_name,
            object_urdf_path=object_motion_urdf_path,
            object_arti_names=object_arti_names,
            object_arti_vals=object_arti_vals,
            hand_name=oakink_cfg.robot_model,
            motion_metadata=_motion_metadata(
                oakink_cfg,
                demo,
                export_dir,
                target_obj_path,
                target_bps_obj_path,
                target_articulated_urdf_path=target_articulated_urdf_path,
                retarget_only=True,
                articulation_retarget_mode=args.articulation_retarget_mode,
                articulation_sample_id=args.articulation_sample_id,
            ),
        )
        print(f"Saved retarget-only final motion to {saved_motion_path}")
        saved_motion_paths = _export_q_trace_motion(
            sequence_optimizer,
            q_trace,
            sequence_source=sequence_source,
            export_dir=export_dir,
            sequence_motion_stem=sequence_motion_stem,
            demo=demo,
            oakink_cfg=oakink_cfg,
            target_obj_path=target_obj_path,
            target_bps_obj_path=target_bps_obj_path,
            target_articulated_urdf_path=target_articulated_urdf_path,
            retarget_only=True,
            retarget_only_frames=args.retarget_only_frames,
            object_arti_names=object_arti_names,
            object_arti_vals=object_arti_vals,
            articulation_retarget_mode=args.articulation_retarget_mode,
            articulation_sample_id=args.articulation_sample_id,
        )
        print(f"Saved retarget-only motions to {saved_motion_paths}")
        if args.save_retarget_probes:
            _try_write_retarget_probes(
                export_dir=export_dir,
                drake_optimizer=drake_optimizer,
                target_obj_path=target_obj_path,
                target_pose=tgt_pose,
                robot_urdf_path=robot_urdf_path,
                q_solution=q_solution,
            )
        return

    if sequence_source == "optimize":
        sequence_result = sequence_optimizer.optimize_seq(q_solution)
    elif sequence_source == "single_step":
        sequence_result = sequence_optimizer.construct_unoptimized_trajectory(
            q_solution
        )
    else:
        raise ValueError(
            "Unsupported trajopt_seq.sequence_source "
            f"{seq_opt_cfg.sequence_source!r}; use 'optimize' or 'single_step'."
        )
    saved_sequence_path = _save_sequence_result(sequence_result, export_dir)
    print(f"Saved sequence optimization result to {saved_sequence_path}")
    saved_motion_path = sequence_optimizer.export_trajectory(
        sequence_result,
        sequence_motion_path,
        object_name=demo.target_obj_name,
        object_urdf_path=object_motion_urdf_path,
        hand_name=oakink_cfg.robot_model,
        motion_metadata=_motion_metadata(
            oakink_cfg,
            demo,
            export_dir,
            target_obj_path,
            target_bps_obj_path,
            target_articulated_urdf_path=target_articulated_urdf_path,
            retarget_only=False,
            articulation_retarget_mode=args.articulation_retarget_mode,
            articulation_sample_id=args.articulation_sample_id,
        ),
    )
    print(f"Saved sequence motion to {saved_motion_path}")
    saved_trace_motion_paths = _export_q_trace_motion(
        sequence_optimizer,
        q_trace,
        sequence_source=sequence_source,
        export_dir=export_dir,
        sequence_motion_stem=sequence_motion_stem,
        demo=demo,
        oakink_cfg=oakink_cfg,
        target_obj_path=target_obj_path,
        target_bps_obj_path=target_bps_obj_path,
        target_articulated_urdf_path=target_articulated_urdf_path,
        retarget_only=False,
        retarget_only_frames=args.retarget_only_frames,
        object_arti_names=object_arti_names,
        object_arti_vals=object_arti_vals,
        articulation_retarget_mode=args.articulation_retarget_mode,
        articulation_sample_id=args.articulation_sample_id,
    )
    print(f"Saved trace sequence motions to {saved_trace_motion_paths}")

    if args.save_retarget_probes:
        _try_write_retarget_probes(
            export_dir=export_dir,
            drake_optimizer=drake_optimizer,
            target_obj_path=target_obj_path,
            target_pose=tgt_pose,
            robot_urdf_path=robot_urdf_path,
            q_solution=q_solution,
        )

    if opt_cfg.visualize:
        sequence_optimizer.visualize_trajectory(
            sequence_result,
            pcd_dict=reference_pcd_dict,
            pause=opt_cfg.pause_after_visualize,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
