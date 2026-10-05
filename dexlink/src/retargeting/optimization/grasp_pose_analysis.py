from __future__ import annotations

from pathlib import Path
from time import time
from typing import TYPE_CHECKING

import numpy as np
from pydrake.all import SnoptSolver, Solve
from pydrake.solvers import SolverOptions

from dexlink.src.retargeting.optimization.utils_final import (
    compute_hand_cfg_vecs,
    compute_object_bps_differentiable,
)

if TYPE_CHECKING:
    from dexlink.src.retargeting.optimization.optimizer_final import DrakeOptimizer


PAIR_KEYS = (
    "total_cost",
    "hand_cfg_cost",
    "hand_object_bps_cost",
    "rotation_cost",
    "bps_distance_cost",
    "bps_direction_cost",
    "bps_normal_cost",
    "bps_penetration_cost",
    "bps_feature_cost",
)


def compute_cost_breakdown(
    optimizer: DrakeOptimizer,
    q_robot: np.ndarray,
    rotation_params: np.ndarray | None = None,
) -> dict[str, float]:
    q_robot = np.asarray(q_robot, dtype=float).reshape(-1)

    if rotation_params is not None:
        rotation_params = np.asarray(rotation_params, dtype=float).reshape(-1)
        rotation_matrix = optimizer.angle_axis_to_rotation_matrix(rotation_params)
        rotation_cost = float(np.sum(rotation_params[:2] ** 2))
    else:
        rotation_matrix = None
        rotation_cost = 0.0

    jpos_dict = optimizer.compute_robot_joint_pos(q_robot)

    hand_cfg_cost = 0.0
    if optimizer.cfg.hand_cfg_weight > 0:
        computed_hand_vec_dict = compute_hand_cfg_vecs(jpos_dict)
        hand_cfg_cost = float(
            optimizer.hand_cfg_cost(computed_hand_vec_dict, rotation_matrix)
        )

    hand_object_bps_cost = 0.0
    bps_logs = {
        "distance_cost": 0.0,
        "direction_cost": 0.0,
        "normal_cost": 0.0,
        "penetration_cost": 0.0,
        "feature_cost": 0.0,
    }
    if optimizer.cfg.hand_object_bps_weight > 0:
        hand_jpos_array = optimizer._computed_hand_bps_points(jpos_dict)
        computed_hand_rel_bps = compute_object_bps_differentiable(
            optimizer.target_pcd,
            hand_jpos_array,
            beta=optimizer.cfg.bps_beta,
            normalize=False,
            include_other_features=optimizer.cfg.bps_nocs_features,
        )
        hand_object_bps_cost, bps_logs = optimizer.hand_object_bps_cost(
            computed_hand_rel_bps, rotation_matrix
        )
        hand_object_bps_cost = float(hand_object_bps_cost)
        bps_logs = {key: float(value) for key, value in bps_logs.items()}

    total_cost = float(
        np.exp(hand_cfg_cost * optimizer.cfg.hand_cfg_weight)
        * hand_object_bps_cost
        * np.exp(rotation_cost)
    )
    return {
        "total_cost": total_cost,
        "hand_cfg_cost": hand_cfg_cost,
        "hand_object_bps_cost": hand_object_bps_cost,
        "rotation_cost": rotation_cost,
        "bps_distance_cost": bps_logs["distance_cost"],
        "bps_direction_cost": bps_logs["direction_cost"],
        "bps_normal_cost": bps_logs["normal_cost"],
        "bps_penetration_cost": bps_logs["penetration_cost"],
        "bps_feature_cost": bps_logs["feature_cost"],
    }


def solve_stage1_grasp_pose(
    optimizer: DrakeOptimizer,
    *,
    seed: int | None = None,
    min_optimizer_iterations: int | None = None,
) -> dict[str, object]:
    if seed is not None:
        optimizer.cfg.seed = int(seed)
    if min_optimizer_iterations is None:
        min_optimizer_iterations = int(optimizer.cfg.min_optimizer_iterations)

    result = None
    q_sol_raw = None
    rotation_sol = None
    while True:
        np.random.seed(optimizer.cfg.seed)
        prog = optimizer.prog

        if optimizer.cfg.rotational_freedom:
            q_and_rot = np.concatenate([optimizer.q, optimizer.rotation_var])
        else:
            q_and_rot = optimizer.q

        prog.AddCost(lambda x: optimizer.compute_costs(x), q_and_rot)
        if optimizer.cfg.penetration_constraint:
            optimizer.add_penetration_constraint()

        solver = SnoptSolver()
        solver_id = solver.solver_id()
        solver_tolerance = optimizer.cfg.solver_tolerance
        prog.SetSolverOption(solver_id, "Feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Major feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Major optimality tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Minor feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Minor optimality tolerance", solver_tolerance)
        solver_options = SolverOptions()

        prog.SetInitialGuess(
            optimizer.q,
            np.random.uniform(
                low=-0.1, high=0.1, size=optimizer.plant_f.num_positions()
            ),
        )
        if optimizer.cfg.rotational_freedom:
            prog.SetInitialGuess(
                optimizer.rotation_var,
                np.random.uniform(low=-0.1, high=0.1, size=3),
            )

        solve_start_time = time()
        result = Solve(prog, solver_options=solver_options)
        solve_elapsed_sec = time() - solve_start_time
        optimizer.logger.print_counts()
        optimizer.logger.dump(optimizer.run_dir)
        print(
            f"[stage1_grasp_pose] "
            f"is_success={result.is_success()} solve_time_sec={solve_elapsed_sec:.3f}"
        )

        q_sol_raw = np.asarray(result.GetSolution(optimizer.q), dtype=float)
        rotation_sol = (
            None
            if not optimizer.cfg.rotational_freedom
            else np.asarray(result.GetSolution(optimizer.rotation_var), dtype=float)
        )

        if optimizer.logger.eval_count < min_optimizer_iterations:
            print(
                f"Optimization finished with only {optimizer.logger.eval_count} evaluations. "
                "Retrying with a different random seed..."
            )
            optimizer.logger.clear()
            optimizer.cfg.seed += 1
            continue
        break

    q_sol_refined = optimizer.non_penetration_refinement(q_sol_raw)
    q_sol_refined = np.asarray(
        optimizer.contact_target_refinement(q_sol_refined),
        dtype=float,
    )
    return {
        "result": result,
        "q_solution_raw": q_sol_raw,
        "q_solution": q_sol_refined,
        "rotation_params": rotation_sol,
        "success": bool(result.is_success()) if result is not None else False,
        "eval_count": int(optimizer.logger.eval_count),
    }


def save_grasp_pose_snapshot(
    *,
    optimizer: DrakeOptimizer,
    q_solution: np.ndarray,
    q_solution_raw: np.ndarray | None = None,
    rotation_params: np.ndarray | None = None,
    save_path: str | Path,
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    breakdown = compute_cost_breakdown(
        optimizer,
        q_solution,
        rotation_params=rotation_params,
    )
    payload = {
        "q_solution": np.asarray(q_solution, dtype=float),
        "q_solution_raw": (
            np.asarray(q_solution_raw, dtype=float)
            if q_solution_raw is not None
            else np.asarray(q_solution, dtype=float)
        ),
        "dof_names": np.asarray(optimizer.dof_names, dtype="<U128"),
        **{key: np.asarray([value], dtype=float) for key, value in breakdown.items()},
    }
    if rotation_params is not None:
        payload["rotation_params"] = np.asarray(rotation_params, dtype=float)
    np.savez(save_path, **payload)
    print(f"[grasp_pose_analysis] saved snapshot: {save_path}")
    return save_path


def sample_grasp_pose_configurations(
    *,
    q_center: np.ndarray,
    lower_limit: np.ndarray,
    upper_limit: np.ndarray,
    num_nearby: int,
    num_random: int,
    nearby_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    q_center = np.asarray(q_center, dtype=float).reshape(-1)
    lower_limit = np.asarray(lower_limit, dtype=float).reshape(-1)
    upper_limit = np.asarray(upper_limit, dtype=float).reshape(-1)
    rng = np.random.default_rng(seed)

    samples = [q_center[None, :]]
    sample_types = ["optimized"]

    if num_nearby > 0:
        nearby = q_center[None, :] + rng.normal(
            scale=float(nearby_std), size=(int(num_nearby), q_center.shape[0])
        )
        nearby = np.clip(nearby, lower_limit, upper_limit)
        samples.append(nearby)
        sample_types.extend(["nearby"] * int(num_nearby))

    if num_random > 0:
        random_samples = rng.uniform(
            low=lower_limit, high=upper_limit, size=(int(num_random), q_center.shape[0])
        )
        samples.append(random_samples)
        sample_types.extend(["random"] * int(num_random))

    return np.vstack(samples), np.asarray(sample_types, dtype="<U16")


def evaluate_grasp_pose_costs(
    optimizer: DrakeOptimizer,
    *,
    q_center: np.ndarray,
    rotation_params: np.ndarray | None,
    num_nearby: int,
    num_random: int,
    nearby_std: float,
    seed: int,
    save_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    lower_limit = optimizer.plant_f.GetPositionLowerLimits()
    upper_limit = optimizer.plant_f.GetPositionUpperLimits()
    q_samples, sample_types = sample_grasp_pose_configurations(
        q_center=q_center,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        num_nearby=num_nearby,
        num_random=num_random,
        nearby_std=nearby_std,
        seed=seed,
    )

    results: dict[str, list[float] | np.ndarray] = {
        "q_samples": q_samples,
        "sample_type": sample_types,
        "q_l2_to_optimized": np.linalg.norm(q_samples - q_center[None, :], axis=1),
    }
    for key in PAIR_KEYS:
        results[key] = []

    for q_sample in q_samples:
        breakdown = compute_cost_breakdown(
            optimizer,
            q_sample,
            rotation_params=rotation_params,
        )
        for key in PAIR_KEYS:
            results[key].append(float(breakdown[key]))

    packed = {
        key: (np.asarray(value) if isinstance(value, list) else value)
        for key, value in results.items()
    }
    packed["optimized_q"] = np.asarray(q_center, dtype=float)
    if rotation_params is not None:
        packed["rotation_params"] = np.asarray(rotation_params, dtype=float)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, **packed)
        print(f"[grasp_pose_analysis] saved: {save_path}")

    return packed
