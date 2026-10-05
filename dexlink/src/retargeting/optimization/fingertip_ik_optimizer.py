from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from pydrake.all import MathematicalProgram, RobotDiagramBuilder, SnoptSolver, Solve

from dexlink.utils.retargeting.path_utils import process_urdf

FINGERTIP_NAMES = (
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "little_tip",
)


@dataclass(frozen=True)
class FingertipIKConfig:
    """Configuration for the object-agnostic XHand fingertip IK baseline."""

    solver_tolerance: float = 1e-6
    max_iterations: int = 500
    num_restarts: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.solver_tolerance <= 0.0:
            raise ValueError(
                "Expected solver_tolerance > 0, got " f"{self.solver_tolerance}."
            )
        if self.max_iterations <= 0:
            raise ValueError(f"Expected max_iterations > 0, got {self.max_iterations}.")
        if self.num_restarts <= 0:
            raise ValueError(f"Expected num_restarts > 0, got {self.num_restarts}.")


@dataclass(frozen=True)
class FingertipIKAttempt:
    restart_index: int
    solver_success: bool
    solution_result: str
    solve_time_sec: float
    mse_m2: float
    rmse_m: float


@dataclass(frozen=True)
class FingertipIKResult:
    q_solution: np.ndarray
    achieved_fingertips: np.ndarray
    target_fingertips: np.ndarray
    per_tip_error_m: np.ndarray
    mse_m2: float
    rmse_m: float
    solver_success: bool
    best_restart_index: int
    attempts: tuple[FingertipIKAttempt, ...]


@dataclass(frozen=True)
class FingertipMotionExportSpec:
    output_path: Path
    active_side: str
    hold_frames: int
    object_pose: np.ndarray
    object_name: str
    object_urdf_path: Path
    hand_name: str
    human_skeleton_keypoints: dict[str, np.ndarray]
    human_bps_keypoints: dict[str, np.ndarray]
    metadata: dict[str, Any]
    retarget_objective: str = "exact_fingertips"


def _validate_keypoints(
    keypoints: dict[str, np.ndarray],
    *,
    expected_names: tuple[str, ...] | None,
    field: str,
) -> dict[str, np.ndarray]:
    if not isinstance(keypoints, dict) or not keypoints:
        raise ValueError(f"Expected {field} to be a non-empty dictionary.")
    if expected_names is not None and tuple(keypoints) != expected_names:
        raise ValueError(
            f"Expected {field} keys {list(expected_names)}, got {list(keypoints)}."
        )
    validated: dict[str, np.ndarray] = {}
    for name, value in keypoints.items():
        point = np.asarray(value, dtype=float)
        if point.shape != (3,):
            raise ValueError(
                f"Expected {field}[{name!r}] shape (3,), got {point.shape}."
            )
        if not np.all(np.isfinite(point)):
            raise ValueError(
                f"Expected finite {field}[{name!r}], got {point.tolist()}."
            )
        validated[name] = point
    return validated


class DrakeFingertipIKOptimizer:
    """Fit only the five XHand fingertip positions with Drake and SNOPT.

    The plant contains the robot hand only. Object geometry, hand-shape vectors,
    BPS descriptors, collision constraints, and contact objectives are deliberately
    absent so this remains a clean fingertip-position baseline.
    """

    def __init__(
        self,
        *,
        config: FingertipIKConfig,
        robot_urdf_path: str | Path,
        robot_cfg: dict[str, Any],
        fingertip_targets: dict[str, np.ndarray],
    ) -> None:
        self.config = config
        # Every Slurm task gets its own processed URDF. Reusing the shared
        # ``*_cluster.urdf`` path lets concurrent tasks read while another task
        # atomically replaces that file, which can surface as a stale NFS handle.
        self.robot_urdf_path = Path(process_urdf(robot_urdf_path, append_jobname=True))
        self.robot_cfg = robot_cfg
        self.fingertip_targets = _validate_keypoints(
            fingertip_targets,
            expected_names=FINGERTIP_NAMES,
            field="fingertip_targets",
        )
        self.target_fingertip_array = np.stack(
            [self.fingertip_targets[name] for name in FINGERTIP_NAMES], axis=0
        )
        self._build_plant()
        self._validate_robot_maps()

    def _build_plant(self) -> None:
        builder = RobotDiagramBuilder(time_step=0.001)
        plant = builder.plant()
        parser = builder.parser()
        print(f"Loading fingertip-IK robot URDF from: {self.robot_urdf_path}")
        (hand_model,) = parser.AddModels(self.robot_urdf_path)
        world_frame = plant.world_frame()
        try:
            root_frame = plant.GetFrameByName("root", hand_model)
        except RuntimeError as exc:
            raise ValueError(
                f"Expected robot URDF {self.robot_urdf_path} to define frame 'root'."
            ) from exc
        plant.WeldFrames(world_frame, root_frame)

        diagram = builder.Build()
        root_context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(root_context)
        plant_ad = plant.ToAutoDiffXd()

        self.plant_f = plant
        self.diagram_f = diagram
        self.root_context_f = root_context
        self.context_f = plant_context
        self.plant_ad = plant_ad
        self.context_ad = plant_ad.CreateDefaultContext()
        self.hand_model = hand_model
        self.dof_names = list(
            plant.GetPositionNames(
                add_model_instance_prefix=False,
                always_add_suffix=False,
            )
        )
        if len(self.dof_names) != plant.num_positions():
            raise ValueError(
                "Expected one Drake position name per robot position, got "
                f"{len(self.dof_names)} names and {plant.num_positions()} positions."
            )

    def _validate_robot_maps(self) -> None:
        link_map = self.robot_cfg.get("human_robot_link_map")
        kpt_map = self.robot_cfg.get("human_robot_kpt_map")
        if not isinstance(link_map, dict) or not isinstance(kpt_map, dict):
            raise ValueError(
                "Expected robot_cfg to define human_robot_link_map and "
                "human_robot_kpt_map dictionaries."
            )
        missing_tips = [name for name in FINGERTIP_NAMES if name not in link_map]
        if missing_tips:
            raise ValueError(
                "Robot semantic link map is missing fingertip names: "
                f"{missing_tips}."
            )
        for semantic_name, frame_name in {**link_map, **kpt_map}.items():
            try:
                self.plant_f.GetFrameByName(str(frame_name), self.hand_model)
            except RuntimeError as exc:
                raise ValueError(
                    f"Robot keypoint {semantic_name!r} maps to missing frame "
                    f"{frame_name!r} in {self.robot_urdf_path}."
                ) from exc

    def _plant_context(self, q: np.ndarray):
        if q.dtype == float:
            return self.plant_f, self.context_f
        return self.plant_ad, self.context_ad

    def robot_keypoints(
        self,
        q: np.ndarray,
        names: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        q = np.asarray(q)
        if q.shape != (self.plant_f.num_positions(),):
            raise ValueError(
                f"Expected q shape ({self.plant_f.num_positions()},), got {q.shape}."
            )
        plant, context = self._plant_context(q)
        plant.SetPositions(context, q)
        world_frame = plant.world_frame()
        frame_map = {
            **self.robot_cfg["human_robot_link_map"],
            **self.robot_cfg["human_robot_kpt_map"],
        }
        missing_names = [name for name in names if name not in frame_map]
        if missing_names:
            raise ValueError(
                f"Robot keypoint maps are missing semantic names {missing_names}."
            )
        positions: dict[str, np.ndarray] = {}
        for name in names:
            frame = plant.GetFrameByName(str(frame_map[name]), self.hand_model)
            positions[name] = plant.CalcRelativeTransform(
                context, world_frame, frame
            ).translation()
        return positions

    def robot_keypoint_array(
        self,
        q: np.ndarray,
        names: tuple[str, ...],
    ) -> np.ndarray:
        positions = self.robot_keypoints(q, names)
        return np.stack([positions[name] for name in names], axis=0)

    def fingertip_mse(self, q: np.ndarray):
        achieved = self.robot_keypoint_array(q, FINGERTIP_NAMES)
        error = achieved - self.target_fingertip_array
        return np.mean(error * error)

    def _position_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.maximum(
            self.plant_f.GetPositionLowerLimits(),
            np.full(self.plant_f.num_positions(), -2.0 * np.pi),
        )
        upper = np.minimum(
            self.plant_f.GetPositionUpperLimits(),
            np.full(self.plant_f.num_positions(), 2.0 * np.pi),
        )
        if np.any(lower > upper):
            bad = np.flatnonzero(lower > upper).tolist()
            raise ValueError(f"Invalid Drake joint bounds at indices {bad}.")
        return lower, upper

    def _open_initial_guess(self) -> np.ndarray:
        q = np.zeros(self.plant_f.num_positions(), dtype=float)
        dof_to_index = {name: idx for idx, name in enumerate(self.dof_names)}
        open_qpos = self.robot_cfg.get("open_qpos")
        if not isinstance(open_qpos, dict):
            raise ValueError("Expected robot_cfg['open_qpos'] to be a dictionary.")
        missing_open_joints = [name for name in open_qpos if name not in dof_to_index]
        if missing_open_joints:
            raise ValueError(
                "Open XHand configuration references joints absent from Drake: "
                f"{missing_open_joints}."
            )
        for name, value in open_qpos.items():
            q[dof_to_index[name]] = float(value)
        return q

    def _translation_indices(self) -> tuple[int, int, int]:
        indices = []
        for axis in "xyz":
            matches = [
                idx
                for idx, name in enumerate(self.dof_names)
                if name.endswith(f"forearm_t{axis}_link_joint")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one forearm t{axis} joint, got indices "
                    f"{matches} from {self.dof_names}."
                )
            indices.append(matches[0])
        return tuple(indices)  # type: ignore[return-value]

    def _initial_guess(self, restart_index: int) -> np.ndarray:
        q = self._open_initial_guess()
        lower, upper = self._position_bounds()
        if restart_index > 0:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.config.seed, restart_index])
            )
            q += rng.normal(scale=0.25, size=q.shape)
            q = np.clip(q, lower, upper)

        achieved = np.asarray(
            self.robot_keypoint_array(q, FINGERTIP_NAMES), dtype=float
        )
        translation_delta = np.mean(self.target_fingertip_array, axis=0) - np.mean(
            achieved, axis=0
        )
        for idx, delta in zip(self._translation_indices(), translation_delta):
            q[idx] += float(delta)
        return np.clip(q, lower, upper)

    def _make_program(self) -> tuple[MathematicalProgram, np.ndarray]:
        prog = MathematicalProgram()
        q = prog.NewContinuousVariables(self.plant_f.num_positions(), "q")
        lower, upper = self._position_bounds()
        prog.AddBoundingBoxConstraint(lower, upper, q)
        prog.AddCost(self.fingertip_mse, q)

        solver_id = SnoptSolver().solver_id()
        tolerance = self.config.solver_tolerance
        for option in (
            "Feasibility tolerance",
            "Major feasibility tolerance",
            "Major optimality tolerance",
            "Minor feasibility tolerance",
            "Minor optimality tolerance",
        ):
            prog.SetSolverOption(solver_id, option, tolerance)
        prog.SetSolverOption(
            solver_id,
            "Major iterations limit",
            self.config.max_iterations,
        )
        return prog, q

    def solve(self) -> FingertipIKResult:
        attempts: list[FingertipIKAttempt] = []
        candidates: list[tuple[float, np.ndarray, np.ndarray, bool, int]] = []
        for restart_index in range(self.config.num_restarts):
            prog, q = self._make_program()
            prog.SetInitialGuess(q, self._initial_guess(restart_index))
            solve_start = perf_counter()
            result = Solve(prog)
            solve_time_sec = perf_counter() - solve_start
            q_solution = np.asarray(result.GetSolution(q), dtype=float).reshape(-1)
            achieved = np.asarray(
                self.robot_keypoint_array(q_solution, FINGERTIP_NAMES), dtype=float
            )
            errors = achieved - self.target_fingertip_array
            mse_m2 = float(np.mean(errors * errors))
            rmse_m = float(np.sqrt(mse_m2))
            if not np.isfinite(mse_m2) or not np.all(np.isfinite(q_solution)):
                raise RuntimeError(
                    f"SNOPT restart {restart_index} returned a non-finite solution."
                )
            attempt = FingertipIKAttempt(
                restart_index=restart_index,
                solver_success=bool(result.is_success()),
                solution_result=str(result.get_solution_result()),
                solve_time_sec=float(solve_time_sec),
                mse_m2=mse_m2,
                rmse_m=rmse_m,
            )
            attempts.append(attempt)
            candidates.append(
                (
                    mse_m2,
                    q_solution,
                    achieved,
                    bool(result.is_success()),
                    restart_index,
                )
            )
            print(
                "[fingertip_ik] "
                f"restart={restart_index} success={result.is_success()} "
                f"rmse_m={rmse_m:.8f} solve_time_sec={solve_time_sec:.3f}"
            )

        mse_m2, q_solution, achieved, solver_success, best_restart_index = min(
            candidates, key=lambda candidate: candidate[0]
        )
        per_tip_error_m = np.linalg.norm(achieved - self.target_fingertip_array, axis=1)
        return FingertipIKResult(
            q_solution=q_solution,
            achieved_fingertips=achieved,
            target_fingertips=self.target_fingertip_array.copy(),
            per_tip_error_m=per_tip_error_m,
            mse_m2=float(mse_m2),
            rmse_m=float(np.sqrt(mse_m2)),
            solver_success=solver_success,
            best_restart_index=best_restart_index,
            attempts=tuple(attempts),
        )

    @staticmethod
    def _infer_side_from_dofs(dof_names: list[str]) -> str:
        first_name = dof_names[0]
        if first_name.startswith("R_") or first_name.startswith("right_"):
            return "right"
        if first_name.startswith("L_") or first_name.startswith("left_"):
            return "left"
        raise ValueError(f"Could not infer hand side from dof name {first_name!r}.")

    @staticmethod
    def _mirror_joint_name(joint_name: str) -> str:
        for source, target in (
            ("R_", "L_"),
            ("L_", "R_"),
            ("right_", "left_"),
            ("left_", "right_"),
        ):
            if joint_name.startswith(source):
                return target + joint_name[len(source) :]
        raise ValueError(f"Cannot mirror side-specific joint name {joint_name!r}.")

    def _inactive_q(self, active_q: np.ndarray) -> np.ndarray:
        inactive = np.zeros_like(active_q, dtype=float)
        open_qpos = self.robot_cfg["open_qpos"]
        for idx, joint_name in enumerate(self.dof_names):
            if joint_name.endswith("forearm_tx_link_joint"):
                inactive[idx] = 0.0
            elif joint_name.endswith("forearm_ty_link_joint"):
                inactive[idx] = 0.35
            elif joint_name.endswith("forearm_tz_link_joint"):
                inactive[idx] = active_q[idx]
            elif joint_name.endswith(
                (
                    "forearm_roll_link_joint",
                    "forearm_pitch_link_joint",
                    "forearm_yaw_link_joint",
                )
            ):
                inactive[idx] = 0.0
            else:
                inactive[idx] = float(open_qpos[joint_name])
        lower, upper = self._position_bounds()
        return np.clip(inactive, lower, upper)

    def export_static_motion(
        self,
        result: FingertipIKResult,
        spec: FingertipMotionExportSpec,
    ) -> Path:
        if spec.active_side not in ("left", "right"):
            raise ValueError(
                f"Expected active_side 'left' or 'right', got {spec.active_side!r}."
            )
        inferred_side = self._infer_side_from_dofs(self.dof_names)
        if spec.active_side != inferred_side:
            raise ValueError(
                f"Export side {spec.active_side!r} disagrees with robot dofs "
                f"for {inferred_side!r}."
            )
        if spec.hold_frames < 1:
            raise ValueError(f"Expected hold_frames >= 1, got {spec.hold_frames}.")
        object_pose = np.asarray(spec.object_pose, dtype=float)
        if object_pose.shape != (4, 4):
            raise ValueError(
                f"Expected object_pose shape (4, 4), got {object_pose.shape}."
            )
        skeleton = _validate_keypoints(
            spec.human_skeleton_keypoints,
            expected_names=None,
            field="human_skeleton_keypoints",
        )
        bps = _validate_keypoints(
            spec.human_bps_keypoints,
            expected_names=None,
            field="human_bps_keypoints",
        )
        expected_skeleton_names = tuple(self.robot_cfg["human_robot_link_map"])
        expected_bps_names = tuple(self.robot_cfg["human_robot_kpt_map"])
        if tuple(skeleton) != expected_skeleton_names:
            raise ValueError(
                "Human skeleton keypoint ordering does not match robot_cfg: "
                f"expected {list(expected_skeleton_names)}, got {list(skeleton)}."
            )
        if tuple(bps) != expected_bps_names:
            raise ValueError(
                "Human BPS keypoint ordering does not match robot_cfg: "
                f"expected {list(expected_bps_names)}, got {list(bps)}."
            )

        from scipy.spatial.transform import Rotation

        n_frames = spec.hold_frames
        active_q = np.repeat(result.q_solution[None, :], n_frames, axis=0).astype(
            np.float32
        )
        inactive_q = np.repeat(
            self._inactive_q(result.q_solution)[None, :], n_frames, axis=0
        ).astype(np.float32)
        active_names = np.asarray(self.dof_names, dtype=object)
        inactive_names = np.asarray(
            [self._mirror_joint_name(name) for name in self.dof_names],
            dtype=object,
        )
        skeleton_names = tuple(skeleton)
        bps_names = tuple(bps)
        human_skeleton = np.repeat(
            np.stack([skeleton[name] for name in skeleton_names])[None, :, :],
            n_frames,
            axis=0,
        ).astype(np.float32)
        human_bps = np.repeat(
            np.stack([bps[name] for name in bps_names])[None, :, :],
            n_frames,
            axis=0,
        ).astype(np.float32)
        robot_skeleton = np.repeat(
            np.asarray(
                self.robot_keypoint_array(result.q_solution, skeleton_names),
                dtype=float,
            )[None, :, :],
            n_frames,
            axis=0,
        ).astype(np.float32)
        robot_bps = np.repeat(
            np.asarray(
                self.robot_keypoint_array(result.q_solution, bps_names),
                dtype=float,
            )[None, :, :],
            n_frames,
            axis=0,
        ).astype(np.float32)

        inactive_side = "left" if spec.active_side == "right" else "right"
        object_quat_wxyz = np.roll(
            Rotation.from_matrix(object_pose[:3, :3]).as_quat(), 1
        ).astype(np.float32)
        payload: dict[str, np.ndarray] = {
            "asset_pos": np.repeat(
                object_pose[:3, 3][None, :], n_frames, axis=0
            ).astype(np.float32),
            "asset_quat": np.repeat(object_quat_wxyz[None, :], n_frames, axis=0),
            "asset_arti_vals": np.zeros((n_frames, 0), dtype=np.float32),
            "asset_arti_names": np.asarray([], dtype=object),
            "traj_hand_side": np.asarray(spec.active_side, dtype=object),
            "traj_hand_name": np.asarray(spec.hand_name, dtype=object),
            "traj_object_name": np.asarray(spec.object_name, dtype=object),
            "traj_object_urdf_path": np.asarray(
                str(spec.object_urdf_path), dtype=object
            ),
            "exact_timesteps": np.arange(n_frames, dtype=np.int64),
            "n_reach_frames": np.asarray([0], dtype=np.int64),
            "n_lift_frames": np.asarray([0], dtype=np.int64),
            "is_active_left": np.asarray([spec.active_side == "left"], dtype=bool),
            "is_active_right": np.asarray([spec.active_side == "right"], dtype=bool),
            "human_skeleton_keypoint_names": np.asarray(skeleton_names, dtype=object),
            "human_bps_keypoint_names": np.asarray(bps_names, dtype=object),
            f"{spec.active_side}_human_skeleton_keypoint_pos": human_skeleton,
            f"{spec.active_side}_human_bps_keypoint_pos": human_bps,
            f"{spec.active_side}_robot_skeleton_keypoint_pos": robot_skeleton,
            f"{spec.active_side}_robot_bps_keypoint_pos": robot_bps,
            f"{spec.active_side}_human_skeleton_keypoint_valid": np.ones(
                (n_frames,), dtype=bool
            ),
            f"{spec.active_side}_human_bps_keypoint_valid": np.ones(
                (n_frames,), dtype=bool
            ),
            f"{inactive_side}_human_skeleton_keypoint_pos": np.full_like(
                human_skeleton, np.nan
            ),
            f"{inactive_side}_human_bps_keypoint_pos": np.full_like(human_bps, np.nan),
            f"{inactive_side}_robot_skeleton_keypoint_pos": np.full_like(
                robot_skeleton, np.nan
            ),
            f"{inactive_side}_robot_bps_keypoint_pos": np.full_like(robot_bps, np.nan),
            f"{inactive_side}_human_skeleton_keypoint_valid": np.zeros(
                (n_frames,), dtype=bool
            ),
            f"{inactive_side}_human_bps_keypoint_valid": np.zeros(
                (n_frames,), dtype=bool
            ),
            "retarget_objective": np.asarray(spec.retarget_objective, dtype=object),
            "fingertip_ik_mse_m2": np.asarray(result.mse_m2, dtype=np.float64),
            "fingertip_ik_rmse_m": np.asarray(result.rmse_m, dtype=np.float64),
        }
        if spec.active_side == "right":
            payload.update(
                {
                    "right_joint_names": active_names,
                    "right_joint_vals": active_q,
                    "left_joint_names": inactive_names,
                    "left_joint_vals": inactive_q,
                }
            )
        else:
            payload.update(
                {
                    "left_joint_names": active_names,
                    "left_joint_vals": active_q,
                    "right_joint_names": inactive_names,
                    "right_joint_vals": inactive_q,
                }
            )
        for key, value in spec.metadata.items():
            if key in payload:
                raise ValueError(
                    f"Motion metadata key {key!r} would overwrite a core field."
                )
            payload[key] = np.asarray(value, dtype=object)

        spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(spec.output_path, **payload)
        return spec.output_path
