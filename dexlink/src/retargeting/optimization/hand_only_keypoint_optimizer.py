from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from pydrake.all import MathematicalProgram, SnoptSolver, Solve

from dexlink.src.retargeting.optimization.fingertip_ik_optimizer import (
    FINGERTIP_NAMES,
    DrakeFingertipIKOptimizer,
    FingertipIKConfig,
)
from dexlink.src.retargeting.optimization.utils_final import compute_hand_cfg_vecs


@dataclass(frozen=True)
class HandOnlyKeypointIKConfig(FingertipIKConfig):
    """Configuration for hand-cfg minimization under a fingertip-RMSE constraint."""

    fingertip_tolerance_m: float = 0.005

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fingertip_tolerance_m <= 0.0:
            raise ValueError(
                "Expected fingertip_tolerance_m > 0, got "
                f"{self.fingertip_tolerance_m}."
            )


@dataclass(frozen=True)
class HandOnlyKeypointIKAttempt:
    restart_index: int
    solver_success: bool
    solution_result: str
    solve_time_sec: float
    objective: float
    fingertip_mse_m2: float
    fingertip_rmse_m: float
    maximum_fingertip_error_m: float
    fingertip_constraint_satisfied: bool
    hand_cfg_cost: float


@dataclass(frozen=True)
class HandOnlyKeypointIKResult:
    q_solution: np.ndarray
    achieved_fingertips: np.ndarray
    target_fingertips: np.ndarray
    per_tip_error_m: np.ndarray
    mse_m2: float
    rmse_m: float
    maximum_fingertip_error_m: float
    fingertip_constraint_satisfied: bool
    hand_cfg_cost: float
    objective: float
    solver_success: bool
    best_restart_index: int
    attempts: tuple[HandOnlyKeypointIKAttempt, ...]


def hand_configuration_cost(
    reference_vectors: dict[str, tuple[list[tuple[str, ...]], np.ndarray]],
    computed_vectors: dict[str, tuple[list[tuple[str, ...]], np.ndarray]],
) -> float:
    """Compute the same unrotated hand-configuration cost as the full objective."""

    reference_pairs, reference_pair_vectors = reference_vectors["pairs"]
    computed_pairs, computed_pair_vectors = computed_vectors["pairs"]
    reference_triplets, reference_triplet_vectors = reference_vectors["triplets"]
    computed_triplets, computed_triplet_vectors = computed_vectors["triplets"]
    if reference_pairs != computed_pairs:
        raise ValueError("Reference and computed hand-cfg pair definitions differ.")
    if reference_triplets != computed_triplets:
        raise ValueError("Reference and computed hand-cfg triplet definitions differ.")
    if reference_pair_vectors.shape != computed_pair_vectors.shape:
        raise ValueError(
            "Reference and computed hand-cfg pair-vector shapes differ: "
            f"{reference_pair_vectors.shape} and {computed_pair_vectors.shape}."
        )
    if reference_triplet_vectors.shape != computed_triplet_vectors.shape:
        raise ValueError(
            "Reference and computed hand-cfg triplet-vector shapes differ: "
            f"{reference_triplet_vectors.shape} and "
            f"{computed_triplet_vectors.shape}."
        )

    # Match normalized directions with cosine distance and metric vectors with MSE.
    pair_costs = []
    for pair_index, (_, _, mode) in enumerate(reference_pairs):
        reference = reference_pair_vectors[pair_index]
        computed = computed_pair_vectors[pair_index]
        if mode == "normalize":
            pair_costs.append(2.0 * (1.0 - np.dot(reference, computed)))
        elif mode == "unnormalize":
            difference = reference - computed
            pair_costs.append(np.dot(difference, difference))
        else:
            raise ValueError(f"Unknown hand-cfg pair mode {mode!r}.")

    triplet_costs = []
    for triplet_index, (_, _, _, mode) in enumerate(reference_triplets):
        reference = reference_triplet_vectors[triplet_index]
        computed = computed_triplet_vectors[triplet_index]
        if mode == "normalize":
            triplet_costs.append(2.0 * (1.0 - np.dot(reference, computed)))
        elif mode == "unnormalize":
            difference = reference - computed
            triplet_costs.append(np.dot(difference, difference))
        else:
            raise ValueError(f"Unknown hand-cfg triplet mode {mode!r}.")
    return np.mean(pair_costs) + np.mean(triplet_costs)


class DrakeHandOnlyKeypointIKOptimizer(DrakeFingertipIKOptimizer):
    """Minimize hand configuration subject to object-agnostic fingertip IK."""

    def __init__(
        self,
        *,
        config: HandOnlyKeypointIKConfig,
        robot_urdf_path: str | Path,
        robot_cfg: dict[str, Any],
        human_keypoints: dict[str, np.ndarray],
    ) -> None:
        expected_names = tuple(robot_cfg["human_robot_link_map"])
        if tuple(human_keypoints) != expected_names:
            raise ValueError(
                "Expected human_keypoints to follow the robot hand-cfg ordering "
                f"{list(expected_names)}, got {list(human_keypoints)}."
            )
        self.hand_only_config = config
        self.human_keypoints = human_keypoints
        self.reference_hand_cfg = compute_hand_cfg_vecs(human_keypoints)
        fingertip_targets = {name: human_keypoints[name] for name in FINGERTIP_NAMES}
        super().__init__(
            config=config,
            robot_urdf_path=robot_urdf_path,
            robot_cfg=robot_cfg,
            fingertip_targets=fingertip_targets,
        )
        self.hand_cfg_names = expected_names

    def hand_cfg_cost(self, q: np.ndarray):
        robot_keypoints = self.robot_keypoints(q, self.hand_cfg_names)
        computed_hand_cfg = compute_hand_cfg_vecs(robot_keypoints)
        return hand_configuration_cost(self.reference_hand_cfg, computed_hand_cfg)

    def _make_program(self) -> tuple[MathematicalProgram, np.ndarray]:
        program = MathematicalProgram()
        q = program.NewContinuousVariables(self.plant_f.num_positions(), "q")
        lower, upper = self._position_bounds()
        program.AddBoundingBoxConstraint(lower, upper, q)
        program.AddCost(self.hand_cfg_cost, q)

        # Bound the same scalar error optimized by the exact-fingertip baseline,
        # then use the remaining freedom to preserve the human hand configuration.
        squared_tolerance = self.hand_only_config.fingertip_tolerance_m**2
        program.AddConstraint(
            lambda values: np.asarray([self.fingertip_mse(values)]),
            [0.0],
            [squared_tolerance],
            q,
        )

        solver_id = SnoptSolver().solver_id()
        tolerance = self.hand_only_config.solver_tolerance
        for option in (
            "Feasibility tolerance",
            "Major feasibility tolerance",
            "Major optimality tolerance",
            "Minor feasibility tolerance",
            "Minor optimality tolerance",
        ):
            program.SetSolverOption(solver_id, option, tolerance)
        program.SetSolverOption(
            solver_id,
            "Major iterations limit",
            self.hand_only_config.max_iterations,
        )
        return program, q

    def _fingertip_warm_start(self, restart_index: int) -> np.ndarray:
        """Minimize fingertip MSE before starting the constrained solve."""

        program, q = DrakeFingertipIKOptimizer._make_program(self)
        program.SetInitialGuess(q, self._initial_guess(restart_index))
        result = Solve(program)
        q_solution = np.asarray(result.GetSolution(q), dtype=float).reshape(-1)
        if not np.all(np.isfinite(q_solution)):
            raise RuntimeError(
                f"Fingertip warm start {restart_index} returned non-finite q."
            )
        return q_solution

    def solve(self) -> HandOnlyKeypointIKResult:
        attempts: list[HandOnlyKeypointIKAttempt] = []
        candidates: list[
            tuple[float, np.ndarray, np.ndarray, float, float, bool, int]
        ] = []
        for restart_index in range(self.hand_only_config.num_restarts):
            program, q = self._make_program()
            program.SetInitialGuess(q, self._fingertip_warm_start(restart_index))
            solve_start = perf_counter()
            solve_result = Solve(program)
            solve_time_sec = perf_counter() - solve_start
            q_solution = np.asarray(solve_result.GetSolution(q), dtype=float).reshape(
                -1
            )
            achieved = np.asarray(
                self.robot_keypoint_array(q_solution, FINGERTIP_NAMES), dtype=float
            )
            errors = achieved - self.target_fingertip_array
            fingertip_mse_m2 = float(np.mean(errors * errors))
            fingertip_rmse_m = float(np.sqrt(fingertip_mse_m2))
            per_tip_error_m = np.linalg.norm(errors, axis=1)
            maximum_fingertip_error_m = float(per_tip_error_m.max())
            fingertip_constraint_satisfied = bool(
                fingertip_rmse_m <= self.hand_only_config.fingertip_tolerance_m + 1e-6
            )
            hand_cfg_cost = float(self.hand_cfg_cost(q_solution))
            objective = hand_cfg_cost
            if not np.isfinite(objective) or not np.all(np.isfinite(q_solution)):
                raise RuntimeError(
                    f"SNOPT restart {restart_index} returned a non-finite solution."
                )
            solver_success = bool(solve_result.is_success())
            attempts.append(
                HandOnlyKeypointIKAttempt(
                    restart_index=restart_index,
                    solver_success=solver_success,
                    solution_result=str(solve_result.get_solution_result()),
                    solve_time_sec=float(solve_time_sec),
                    objective=objective,
                    fingertip_mse_m2=fingertip_mse_m2,
                    fingertip_rmse_m=fingertip_rmse_m,
                    maximum_fingertip_error_m=maximum_fingertip_error_m,
                    fingertip_constraint_satisfied=fingertip_constraint_satisfied,
                    hand_cfg_cost=hand_cfg_cost,
                )
            )
            if solver_success and fingertip_constraint_satisfied:
                candidates.append(
                    (
                        objective,
                        q_solution,
                        achieved,
                        fingertip_mse_m2,
                        hand_cfg_cost,
                        solver_success,
                        restart_index,
                    )
                )
            print(
                "[hand_only_keypoint_ik] "
                f"restart={restart_index} success={solver_success} "
                f"constraint_satisfied={fingertip_constraint_satisfied} "
                f"hand_cfg_cost={hand_cfg_cost:.8g} "
                f"fingertip_rmse_m={fingertip_rmse_m:.8f} "
                f"max_tip_error_m={maximum_fingertip_error_m:.8f} "
                f"solve_time_sec={solve_time_sec:.3f}"
            )

        if not candidates:
            rmse_values = [attempt.fingertip_rmse_m for attempt in attempts]
            raise RuntimeError(
                "No SNOPT restart returned a solver-successful solution satisfying "
                f"the {self.hand_only_config.fingertip_tolerance_m:g} m fingertip "
                f"RMSE constraint. Fingertip RMSE values: {rmse_values}."
            )

        (
            objective,
            q_solution,
            achieved,
            fingertip_mse_m2,
            hand_cfg_cost,
            solver_success,
            best_restart_index,
        ) = min(candidates, key=lambda candidate: candidate[0])
        per_tip_error_m = np.linalg.norm(achieved - self.target_fingertip_array, axis=1)
        maximum_fingertip_error_m = float(per_tip_error_m.max())
        return HandOnlyKeypointIKResult(
            q_solution=q_solution,
            achieved_fingertips=achieved,
            target_fingertips=self.target_fingertip_array.copy(),
            per_tip_error_m=per_tip_error_m,
            mse_m2=float(fingertip_mse_m2),
            rmse_m=float(np.sqrt(fingertip_mse_m2)),
            maximum_fingertip_error_m=maximum_fingertip_error_m,
            fingertip_constraint_satisfied=True,
            hand_cfg_cost=float(hand_cfg_cost),
            objective=float(objective),
            solver_success=solver_success,
            best_restart_index=best_restart_index,
            attempts=tuple(attempts),
        )
