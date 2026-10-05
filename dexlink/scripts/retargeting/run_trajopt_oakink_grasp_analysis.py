#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexlink.scripts.retargeting.run_trajopt_oakink import (
    _get_hand_data,
    _normalize_hand,
    get_robot_spec,
)
from dexlink.src.retargeting.config.trajopt_final import (
    config_to_dict,
    extract_overrides,
    load_experiment_config,
)
from dexlink.src.retargeting.optimization.grasp_pose_analysis import (
    evaluate_grasp_pose_costs,
    save_grasp_pose_snapshot,
    solve_stage1_grasp_pose,
)
from dexlink.src.retargeting.optimization.hand_bps import build_hand_bps_jpos_dict
from dexlink.src.retargeting.optimization.oakink_processor import OakInkProcessor
from dexlink.src.retargeting.optimization.optimizer_final import DrakeOptimizer


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run only the stage-one OakInk grasp optimizer and export cost analysis."
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
        "--num-nearby",
        type=int,
        default=500,
        help="Number of nearby samples around the optimized grasp.",
    )
    parser.add_argument(
        "--num-random",
        type=int,
        default=500,
        help="Number of random joint configurations to evaluate.",
    )
    parser.add_argument(
        "--nearby-std",
        type=float,
        default=0.05,
        help="Stddev for nearby sampling around the optimized grasp.",
    )
    parser.add_argument(
        "--analysis-seed",
        type=int,
        default=42,
        help="Random seed for nearby/random sampling.",
    )
    return parser.parse_known_args(argv)


def main(argv=None):
    args, unknown = _parse_args(argv)
    overrides = extract_overrides(unknown)
    print(f"Applying config overrides: {overrides}")
    cfg = load_experiment_config(preset=args.preset, overrides=overrides)

    if args.print_config:
        from pprint import pprint

        pprint(config_to_dict(cfg))
        return

    oakink_cfg = cfg.oakink
    opt_cfg = cfg.trajopt
    if oakink_cfg.demo_path is None:
        raise ValueError("`oakink.demo_path` is required to run OakInk trajopt.")

    hand = _normalize_hand(oakink_cfg.hand)
    demo = OakInkProcessor(
        original_demo_path=str(oakink_cfg.demo_path),
        run_dir=cfg.run_dir,
        target_demo_path=oakink_cfg.target_demo_path,
        target_obj_path=oakink_cfg.target_obj_path,
        oakink_hand_side=oakink_cfg.oakink_hand_side,
        oakink_smplx_model_path=oakink_cfg.oakink_smplx_model_path,
        oakink_mano_assets_root=oakink_cfg.oakink_mano_assets_root,
    )

    export_dir = Path(cfg.run_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    original_obj_path = Path(demo.export_object_obj(obj_type="original"))
    target_obj_path = Path(demo.export_object_obj(obj_type="target"))
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

    optimizer = DrakeOptimizer(
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
    stage1_result = solve_stage1_grasp_pose(optimizer)

    snapshot_path = save_grasp_pose_snapshot(
        optimizer=optimizer,
        q_solution=stage1_result["q_solution"],
        q_solution_raw=stage1_result["q_solution_raw"],
        rotation_params=stage1_result["rotation_params"],
        save_path=export_dir / "stage1_grasp_pose_snapshot.npz",
    )
    analysis_path = export_dir / "stage1_grasp_pose_cost_analysis.npz"
    evaluate_grasp_pose_costs(
        optimizer,
        q_center=stage1_result["q_solution"],
        rotation_params=stage1_result["rotation_params"],
        num_nearby=args.num_nearby,
        num_random=args.num_random,
        nearby_std=args.nearby_std,
        seed=args.analysis_seed,
        save_path=analysis_path,
    )
    print(f"Saved stage-one grasp snapshot to {snapshot_path}")
    print(f"Saved stage-one grasp analysis to {analysis_path}")

    if opt_cfg.visualize:
        optimizer.plant_with_visualizer(
            q_robot=stage1_result["q_solution"],
            pcd_dict=reference_pcd_dict,
            pause=opt_cfg.pause_after_visualize,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
