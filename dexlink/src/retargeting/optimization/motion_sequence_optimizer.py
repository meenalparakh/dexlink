from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydrake.all import (
    KinematicTrajectoryOptimization,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MinimumDistanceLowerBoundConstraint,
    RigidTransform,
    RobotDiagramBuilder,
    Role,
    SnoptSolver,
    Solve,
)
from pydrake.geometry import Rgba, Sphere
from scipy.spatial.transform import Rotation as R

from dexlink.src.retargeting.optimization.optimizer_final import DrakeOptimizer
from dexlink.src.retargeting.optimization.utils_final import (
    compute_hand_cfg_vecs,
    compute_object_bps_differentiable,
)

if TYPE_CHECKING:
    from dexlink.src.retargeting.config.trajopt_final import TrajOptSeqCfg


class DrakeSequenceOptimizer(DrakeOptimizer):
    def __init__(self, seq_optcfg: TrajOptSeqCfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_optcfg = seq_optcfg
        self.open_cfg = self.robot_cfg["open_qpos"]

    def initialize_prog(self):
        # override to do nothing since we construct the optimization problem in optimize_seq
        pass

    def back_direction(self, q_solution):
        jpos_dict = self.compute_robot_joint_pos(q_solution)
        computed_hand_vec_dict = compute_hand_cfg_vecs(jpos_dict)
        palm_vec = computed_hand_vec_dict["triplets"][1][0]
        norm = np.linalg.norm(palm_vec)
        if norm < 1e-4:
            return np.array([0.0, 0.0, -1.0])
        return -palm_vec / norm

    def get_root_translation_indices(self) -> np.ndarray:
        root_indices = self.get_root_position_indices()
        root_names = np.asarray(
            [self.dof_names[idx] for idx in root_indices], dtype=object
        )
        translation_mask = np.asarray(
            ["_tx_" in name or "_ty_" in name or "_tz_" in name for name in root_names],
            dtype=bool,
        )
        return root_indices[translation_mask]

    def _clip_to_joint_limits(self, q: np.ndarray) -> np.ndarray:
        lower_limit = self.plant_f.GetPositionLowerLimits()
        upper_limit = self.plant_f.GetPositionUpperLimits()
        clipped = np.clip(np.asarray(q, dtype=float), lower_limit, upper_limit)
        return clipped

    def _make_open_preshape(self, q_solution: np.ndarray) -> np.ndarray:
        q_start = np.asarray(q_solution, dtype=float).copy()
        dof_to_index = {name: idx for idx, name in enumerate(self.dof_names)}
        for joint_name, value in self.open_cfg.items():
            if joint_name in dof_to_index:
                q_start[dof_to_index[joint_name]] = float(value)

        root_translation_indices = self.get_root_translation_indices()
        q_start[root_translation_indices] = q_solution[
            root_translation_indices
        ] + 0.08 * self.back_direction(q_solution)
        return self._clip_to_joint_limits(q_start)

    def _get_wrist_tz_index(self) -> int:
        tz_indices = [
            idx
            for idx, name in enumerate(self.dof_names)
            if "_tz_" in name or name.endswith("forearm_tz_link_joint")
        ]
        if len(tz_indices) != 1:
            raise ValueError(
                f"Expected exactly one wrist tz joint, found indices {tz_indices}."
            )
        return int(tz_indices[0])

    def _build_lift_segment(
        self,
        q_final: np.ndarray,
        lift_steps: int,
        lift_height: float = 0.30,
    ) -> np.ndarray:
        if lift_steps < 1:
            raise ValueError(f"Expected lift_steps >= 1, got {lift_steps}.")

        q_final = np.asarray(q_final, dtype=float).reshape(-1)
        lift_segment = np.repeat(q_final[None, :], lift_steps, axis=0)
        tz_idx = self._get_wrist_tz_index()
        z_step = lift_height / float(lift_steps)
        lift_segment[:, tz_idx] += z_step * np.arange(1, lift_steps + 1, dtype=float)
        return self._clip_to_joint_limits(lift_segment)

    def _sample_non_penetrating_candidate(
        self,
        target_q: np.ndarray,
        reference_q: np.ndarray,
        rng: np.random.Generator,
        noise_scale: float,
        max_attempts: int = 60,
    ) -> np.ndarray:
        candidate = self._clip_to_joint_limits(target_q)
        if not self.check_penetration(candidate):
            return candidate

        for _ in range(max_attempts):
            proposal = target_q + rng.normal(scale=noise_scale, size=target_q.shape)
            proposal = 0.8 * proposal + 0.2 * reference_q
            proposal = self._clip_to_joint_limits(proposal)
            if not self.check_penetration(proposal):
                return proposal

        return self._clip_to_joint_limits(reference_q)

    def construct_initial_traj(self, q_solution):
        # q_solution is the final grasp configuration of the hand
        # that needs to be attained. however, it may be colliding/penetrating with the object
        # so via optimization, we mimic the action of reaching for the object
        # and grasping it, which should yield a more natural trajectory and also resolve penetrations
        # the hand starts with a more open configuration, and then gradually closes to reach the final grasp configuration
        # the trajectory should be initialized as follows:
        # 1. given the q_solution, get the wrist pose
        # 2. the start point should be the wrist pose moved back (opposite the palm direction) by 0.2m from the solution
        # 3. then have 30 (self.seq_optcfg.n_control_pts) steps interpolated from the start to the q_solution
        #    at each step, we use the gaussian sampling around the interpolated configuration,
        #    and check for penetration constraint. if violated, we sample again.
        # 4. now a sequence of non-penetrating configurations from the open configuration to the final grasp configuration
        #    is obtained, which can be used as the initial trajectory for optimization
        q_solution = np.asarray(q_solution, dtype=float).reshape(-1)
        n_control_pts = int(self.seq_optcfg.n_control_pts)
        if n_control_pts < 2:
            raise ValueError("Trajectory optimization needs at least 2 control points.")

        q_start = self._make_open_preshape(q_solution)
        init_traj = np.zeros((n_control_pts, self.plant_f.num_positions()), dtype=float)
        rng = np.random.default_rng(self.seq_optcfg.seed)
        init_traj[0] = self._sample_non_penetrating_candidate(
            q_start,
            q_start,
            rng,
            noise_scale=0.01,
        )

        for ctrl_idx, alpha in enumerate(
            np.linspace(0.0, 1.0, n_control_pts)[1:], start=1
        ):
            interp_q = (1.0 - alpha) * q_start + alpha * q_solution
            reference_q = init_traj[ctrl_idx - 1]
            noise_scale = max(0.0025, 0.03 * (1.0 - alpha))
            init_traj[ctrl_idx] = self._sample_non_penetrating_candidate(
                interp_q,
                reference_q,
                rng,
                noise_scale=noise_scale,
            )

        return init_traj

    def check_penetration(self, q):
        q = np.asarray(q, dtype=float).reshape(-1)
        signed_distances = self._eval_signed_distances_for_pairs(q)
        return bool(np.any(signed_distances < self.cfg.penetration_bound))

    # def _surface_contact_cost(self, q_var):
    #     contact_joint_names = [
    #         "thumb_tip",
    #         "index_tip",
    #         "middle_tip",
    #         "ring_tip",
    #         "little_tip",
    #         "thumb_proximal",
    #         "index_proximal",
    #         "middle_proximal",
    #         "ring_proximal",
    #         "little_proximal",
    #     ]
    #     jpos_dict = self.compute_robot_joint_pos(q_var)
    #     contact_points = np.stack(
    #         [jpos_dict[joint_name] for joint_name in contact_joint_names if joint_name in jpos_dict],
    #         axis=0,
    #     )
    #     object_points = self.target_pcd[:, :3]
    #     diff = contact_points[:, None, :] - object_points[None, :, :]
    #     sq_dists = np.sum(diff * diff, axis=-1)
    #     weights = np.exp(-50.0 * sq_dists)
    #     weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-9)
    #     expected_sq_dists = np.sum(weights * sq_dists, axis=1)
    #     return np.sum(expected_sq_dists)

    @staticmethod
    def _quadratic_tracking_cost(x, q_nominal):
        diff = x - q_nominal
        return np.sum(diff * diff)

    @staticmethod
    def _smoothness_cost(x):
        n_q = x.shape[0] // 2
        diff = x[:n_q] - x[n_q:]
        return np.sum(diff * diff)

    @staticmethod
    def _second_difference_cost(x):
        n_q = x.shape[0] // 3
        accel = x[:n_q] - 2.0 * x[n_q : 2 * n_q] + x[2 * n_q :]
        return np.sum(accel * accel)

    def _final_knot_tolerances(self, q_solution: np.ndarray) -> np.ndarray:
        tolerance = np.full_like(q_solution, 0.15, dtype=float)
        root_indices = self.get_root_position_indices()
        tolerance[root_indices] = 0.12
        root_translation_indices = self.get_root_translation_indices()
        tolerance[root_translation_indices] = 0.03
        return tolerance

    def _terminal_hand_objective(self, q_var, rotation_var=None):
        if rotation_var is not None:
            rotation_matrix = self.angle_axis_to_rotation_matrix(rotation_var)
        else:
            rotation_matrix = None

        jpos_dict = self.compute_robot_joint_pos(q_var)

        hand_jpos_array = self._computed_hand_bps_points(jpos_dict)
        computed_hand_rel_bps = compute_object_bps_differentiable(
            self.target_pcd,
            hand_jpos_array,
            beta=self.cfg.bps_beta,
            normalize=False,
            include_other_features=self.cfg.bps_nocs_features,
        )
        self.cfg.distance_weight = self.seq_optcfg.distance_cost_weight
        hand_object_bps_cost, _ = self.hand_object_bps_cost(
            computed_hand_rel_bps,
            rotation_matrix,
        )
        return (
            hand_object_bps_cost
            * self.seq_optcfg.bps_cost_weight
            # np.exp(hand_cfg_cost * self.cfg.hand_cfg_weight)
            # * hand_object_bps_cost
            # * np.exp(rotation_cost)
        )

    def _duration_bounds(self) -> tuple[float, float]:
        n_control_pts = max(2, int(self.seq_optcfg.n_control_pts))
        min_duration = 0.5
        max_duration = max(1.0, 0.15 * (n_control_pts - 1))
        return min_duration, max_duration

    def _control_point_path_samples(self) -> np.ndarray:
        n_control_pts = max(2, int(self.seq_optcfg.n_control_pts))
        return np.linspace(0.0, 1.0, n_control_pts)

    def _infer_active_side(self) -> str:
        first_name = str(self.dof_names[0])
        if first_name.startswith("R_") or first_name.startswith("right_"):
            return "right"
        if first_name.startswith("L_") or first_name.startswith("left_"):
            return "left"
        raise ValueError(f"Could not infer active side from dof name {first_name!r}.")

    def _mirror_joint_name(self, joint_name: str) -> str:
        replacements = [
            ("R_", "L_"),
            ("L_", "R_"),
            ("right_", "left_"),
            ("left_", "right_"),
        ]
        for src, dst in replacements:
            if joint_name.startswith(src):
                return dst + joint_name[len(src) :]
        return joint_name

    def _inactive_side_template(self, active_q: np.ndarray) -> np.ndarray:
        template = np.zeros_like(active_q, dtype=float)
        for idx, joint_name in enumerate(self.dof_names):
            lower = self.plant_f.GetPositionLowerLimits()[idx]
            upper = self.plant_f.GetPositionUpperLimits()[idx]
            if joint_name.endswith("forearm_ty_link_joint"):
                template[idx] = 0.35
            elif joint_name.endswith("forearm_tz_link_joint"):
                template[idx] = np.clip(active_q[idx], lower, upper)
            elif joint_name.endswith(
                (
                    "forearm_roll_link_joint",
                    "forearm_pitch_link_joint",
                    "forearm_yaw_link_joint",
                )
            ):
                template[idx] = 0.0
            else:
                template[idx] = self.open_cfg.get(joint_name, 0.0)
            template[idx] = np.clip(template[idx], lower, upper)
        return template

    def _sample_q_trajectory(
        self,
        traj: dict | np.ndarray,
        time_step: float = 0.05,
    ) -> np.ndarray:
        if isinstance(traj, dict):
            drake_traj = traj.get("trajectory")
            control_points = traj.get("control_points")
        else:
            drake_traj = None
            control_points = traj

        if drake_traj is not None:
            times = np.append(
                np.arange(drake_traj.start_time(), drake_traj.end_time(), time_step),
                drake_traj.end_time(),
            )
            return np.stack(
                [
                    np.asarray(drake_traj.value(float(t))).reshape(-1)[
                        : self.plant_f.num_positions()
                    ]
                    for t in times
                ],
                axis=0,
            )

        control_points = np.asarray(control_points, dtype=float)
        if control_points.ndim != 2:
            raise ValueError(
                f"Expected control_points to be 2D, got shape {control_points.shape}."
            )
        return control_points

    def _keypoint_export_payload(
        self,
        *,
        active_side: str,
        active_joint_vals: np.ndarray,
        n_frames: int,
        human_keypoints_valid: bool,
    ) -> dict[str, np.ndarray]:
        reference_payload = self.reference_keypoint_payload()
        skeleton_names = tuple(
            reference_payload["human_skeleton_keypoint_names"].tolist()
        )
        bps_names = tuple(reference_payload["human_bps_keypoint_names"].tolist())

        human_skeleton = np.repeat(
            reference_payload["human_skeleton_keypoint_pos"][None, :, :],
            n_frames,
            axis=0,
        ).astype(np.float32)
        human_bps = np.repeat(
            reference_payload["human_bps_keypoint_pos"][None, :, :],
            n_frames,
            axis=0,
        ).astype(np.float32)
        robot_skeleton = np.stack(
            [self.robot_keypoint_array(q, skeleton_names) for q in active_joint_vals],
            axis=0,
        ).astype(np.float32)
        robot_bps = np.stack(
            [self.robot_keypoint_array(q, bps_names) for q in active_joint_vals],
            axis=0,
        ).astype(np.float32)

        inactive_side = "left" if active_side == "right" else "right"
        payload = {
            "human_skeleton_keypoint_names": reference_payload[
                "human_skeleton_keypoint_names"
            ],
            "human_bps_keypoint_names": reference_payload["human_bps_keypoint_names"],
            f"{active_side}_human_skeleton_keypoint_pos": human_skeleton,
            f"{active_side}_human_bps_keypoint_pos": human_bps,
            f"{active_side}_robot_skeleton_keypoint_pos": robot_skeleton,
            f"{active_side}_robot_bps_keypoint_pos": robot_bps,
            f"{active_side}_human_skeleton_keypoint_valid": np.ones(
                (n_frames,), dtype=bool
            )
            * bool(human_keypoints_valid),
            f"{active_side}_human_bps_keypoint_valid": np.ones((n_frames,), dtype=bool)
            * bool(human_keypoints_valid),
        }
        for side in (inactive_side,):
            payload[f"{side}_human_skeleton_keypoint_pos"] = np.full_like(
                human_skeleton,
                np.nan,
            )
            payload[f"{side}_human_bps_keypoint_pos"] = np.full_like(human_bps, np.nan)
            payload[f"{side}_robot_skeleton_keypoint_pos"] = np.full_like(
                robot_skeleton,
                np.nan,
            )
            payload[f"{side}_robot_bps_keypoint_pos"] = np.full_like(robot_bps, np.nan)
            payload[f"{side}_human_skeleton_keypoint_valid"] = np.zeros(
                (n_frames,), dtype=bool
            )
            payload[f"{side}_human_bps_keypoint_valid"] = np.zeros(
                (n_frames,), dtype=bool
            )
        return payload

    def optimize_seq(self, q_solution):
        # construct the initial trajectory
        q_solution = np.asarray(q_solution, dtype=float).reshape(-1)
        init_traj = self.construct_initial_traj(q_solution)

        # construct mathematical optimization problem for trajectory
        n_q = self.plant_f.num_positions()
        n_control_pts = int(self.seq_optcfg.n_control_pts)
        lower_limit = self.plant_f.GetPositionLowerLimits()
        upper_limit = self.plant_f.GetPositionUpperLimits()

        trajopt = KinematicTrajectoryOptimization(
            n_q,
            num_control_points=n_control_pts,
        )
        prog = trajopt.get_mutable_prog()
        control_points = trajopt.control_points()
        rotation_var = None
        if self.seq_optcfg.bps_cost_weight > 0.0 and self.cfg.rotational_freedom:
            rotation_var = prog.NewContinuousVariables(3, "terminal_rotation_var")
            prog.AddBoundingBoxConstraint(
                -np.pi * np.ones(3),
                np.pi * np.ones(3),
                rotation_var,
            )

        trajopt.AddPositionBounds(lower_limit, upper_limit)
        trajopt.AddVelocityBounds(
            self.plant_f.GetVelocityLowerLimits(),
            self.plant_f.GetVelocityUpperLimits(),
        )
        min_duration, max_duration = self._duration_bounds()
        trajopt.AddDurationConstraint(min_duration, max_duration)

        prog.AddBoundingBoxConstraint(init_traj[0], init_traj[0], control_points[:, 0])
        # a small cost of last control point to be close to the solution.
        prog.AddCost(
            lambda x: self._quadratic_tracking_cost(x, q_solution)
            * self.seq_optcfg.closeness_cost_weight,
            control_points[:, -1],
        )

        if self.cfg.penetration_constraint_mode == "minimum_distance_lower_bound":
            non_penetration_constraint = MinimumDistanceLowerBoundConstraint(
                self.plant_f,
                bound=self.cfg.penetration_bound,
                plant_context=self.plant_context_f,
                influence_distance_offset=self.cfg.influence_distance_offset,
            )
            for s in self._control_point_path_samples():
                trajopt.AddPathPositionConstraint(
                    constraint=non_penetration_constraint,
                    s=float(s),
                )
        else:
            for ctrl_idx in range(n_control_pts):
                self.add_penetration_constraint(
                    prog=prog,
                    q=control_points[:, ctrl_idx],
                )

        for idx in range(n_control_pts - 1):
            prog.AddCost(
                lambda x: 5.0 * self._smoothness_cost(x),
                np.concatenate([control_points[:, idx], control_points[:, idx + 1]]),
            )

        # add the hand config and bps cost to the final knot
        if rotation_var is None:
            prog.AddCost(
                lambda x: self._terminal_hand_objective(x),
                control_points[:, -1],
            )
        else:
            prog.AddCost(
                lambda x: self._terminal_hand_objective(
                    x[:n_q],
                    x[n_q:],
                ),
                np.concatenate([control_points[:, -1], rotation_var]),
            )

        solver = SnoptSolver()
        solver_id = solver.solver_id()
        solver_tolerance = self.seq_optcfg.solver_tolerance
        prog.SetSolverOption(solver_id, "Feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Major feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Major optimality tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Minor feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Minor optimality tolerance", solver_tolerance)
        prog.SetInitialGuess(control_points, init_traj.T)
        if rotation_var is not None:
            prog.SetInitialGuess(rotation_var, np.array([1e-6, 0.0, 0.0]))

        result = Solve(prog)

        control_point_solution = result.GetSolution(control_points).T
        bspline_traj = trajopt.ReconstructTrajectory(result)
        return {
            "success": result.is_success(),
            "control_points": control_point_solution,
            "trajectory": bspline_traj,
            "duration": float(result.GetSolution(trajopt.duration())),
            "basis": trajopt.basis(),
            "rotation_var": (
                None
                if rotation_var is None
                else np.asarray(result.GetSolution(rotation_var), dtype=float)
            ),
        }

    def visualize_trajectory(
        self,
        traj,
        pcd_dict=None,
        max_n_pts=200,
        marker_radius=0.005,
        pause=True,
        time_step=0.05,
    ):
        meshcat = self.get_meshcat()
        meshcat.Delete()
        meshcat.DeleteAddedControls()
        pcd_dict = {} if pcd_dict is None else dict(pcd_dict)

        robot_builder = RobotDiagramBuilder(time_step=0.001)

        plant = robot_builder.plant()
        parser = robot_builder.parser()
        builder = robot_builder.builder()
        scene_graph = robot_builder.scene_graph()
        (hand_model,) = parser.AddModels(self.robot_urdf_path)
        (object_model,) = parser.AddModels(self.target_obj_path)

        # weld robot to the world
        W = plant.world_frame()
        plant.WeldFrames(W, plant.GetFrameByName("root", hand_model))

        # Weld using the model body's frame (not arbitrary frame index ordering).
        asset_body_idx = plant.GetBodyIndices(object_model)[0]
        asset_frame = plant.get_body(asset_body_idx).body_frame()

        pose = self.tgt_pose.copy()

        X_WA = RigidTransform(pose)
        plant.WeldFrames(W, asset_frame, X_WA)
        plant.Finalize()

        visualizer = MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration),
        )
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)
        visualizer_context = visualizer.GetMyContextFromRoot(context)

        pcd_dict.update(
            {
                "reference_pcd": self.reference_pcd[..., :3],
                "target_pcd": self.target_pcd[..., :3],
            }
        )

        def add_markers(prefix, count, color):
            for i in range(count):
                meshcat.SetObject(
                    f"{prefix}/{i}",
                    Sphere(marker_radius),
                    rgba=color,
                )

        def set_markers(prefix, points, indices):
            for i, p_idx in enumerate(indices):
                meshcat.SetTransform(
                    f"{prefix}/{i}",
                    RigidTransform(points[p_idx]),
                )

        # Reference demo point clouds (human + object).
        for pts_name, pts_array in pcd_dict.items():
            rgb = np.random.random(3)
            add_markers(
                f"reference/{pts_name}", max_n_pts, Rgba(rgb[0], rgb[1], rgb[2], 0.9)
            )
            # sample max_n_pts from the pts_array
            sampled_indices = np.random.choice(
                len(pts_array), size=max_n_pts, replace=(len(pts_array) < max_n_pts)
            )
            set_markers(f"reference/{pts_name}", pts_array, sampled_indices)

        if isinstance(traj, dict):
            drake_traj = traj.get("trajectory")
            control_points = traj.get("control_points")
            duration = traj.get("duration")
        else:
            drake_traj = traj
            control_points = None
            duration = None

        q_samples = None
        if drake_traj is not None:
            times = np.append(
                np.arange(drake_traj.start_time(), drake_traj.end_time(), time_step),
                drake_traj.end_time(),
            )
            q_samples = [
                np.asarray(drake_traj.value(float(t))).reshape(-1)[
                    : plant.num_positions()
                ]
                for t in times
            ]
        elif control_points is not None:
            control_points = np.asarray(control_points, dtype=float)
            if control_points.ndim != 2:
                raise ValueError(
                    f"`control_points` must be 2D for visualization, got {control_points.shape}."
                )
            q_samples = [control_points[idx] for idx in range(control_points.shape[0])]
            if duration is None:
                duration = time_step * max(0, len(q_samples) - 1)
            times = np.linspace(0.0, float(duration), len(q_samples))
        else:
            raise ValueError(
                "Expected `traj` to be either a Drake trajectory or a result dict "
                "containing `trajectory` or `control_points`."
            )

        meshcat.StartRecording()
        visualizer.StartRecording(False)

        for t, q_sample in zip(times, q_samples):
            context.SetTime(float(t))
            plant.SetPositions(plant_context, np.asarray(q_sample, dtype=float))
            visualizer.ForcedPublish(visualizer_context)

        visualizer.StopRecording()
        visualizer.PublishRecording()
        meshcat.PublishRecording()

        visualizer.ForcedPublish(visualizer_context)
        if pause:
            input("Press Enter to continue...")
        return meshcat

    def construct_trajectory(
        self,
        q_trajectory,
        *,
        contains_lift: bool = False,
        n_reach_frames: int | None = None,
    ):
        q_trajectory = np.asarray(q_trajectory, dtype=float)
        if q_trajectory.ndim != 2:
            raise ValueError(
                f"Expected q_trajectory to have shape (T, nq), got {q_trajectory.shape}."
            )

        lift_height = 0.30
        if contains_lift:
            if n_reach_frames is None:
                raise ValueError(
                    "`n_reach_frames` is required when `contains_lift` is True."
                )
            n_reach = int(n_reach_frames)
            if n_reach < 1 or n_reach >= q_trajectory.shape[0]:
                raise ValueError(
                    f"Expected 1 <= n_reach_frames < {q_trajectory.shape[0]}, got {n_reach}."
                )
            full_q = q_trajectory
            lift_steps = full_q.shape[0] - n_reach
        else:
            n_reach = q_trajectory.shape[0]
            lift_steps = max(10, n_reach // 2)
            lift_segment = self._build_lift_segment(
                q_trajectory[-1],
                lift_steps=lift_steps,
                lift_height=lift_height,
            )
            full_q = np.concatenate([q_trajectory, lift_segment], axis=0)

        if lift_steps < 1:
            raise ValueError(
                f"Expected at least one lift frame, got lift_steps={lift_steps}."
            )

        obj_pos = np.repeat(self.tgt_pose[:3, 3][None, :], full_q.shape[0], axis=0)
        obj_quat_single = np.roll(
            R.from_matrix(self.tgt_pose[:3, :3]).as_quat(), 1
        ).astype(float)
        obj_quat = np.repeat(obj_quat_single[None, :], full_q.shape[0], axis=0)
        obj_rot_single = R.from_matrix(self.tgt_pose[:3, :3]).as_rotvec().astype(float)
        obj_rot = np.repeat(obj_rot_single[None, :], full_q.shape[0], axis=0)
        lift_offset = np.linspace(0.0, lift_height, lift_steps + 1, dtype=float)[1:]
        obj_pos[n_reach:, 2] += lift_offset

        joint_names = []
        obj_arti = np.zeros((full_q.shape[0], 0), dtype=float)
        return {
            "q_trajectory": full_q,
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "obj_rot": obj_rot,
            "obj_arti": obj_arti,
            "joint_names": joint_names,
            "n_reach_frames": n_reach,
            "n_lift_frames": lift_steps,
        }

    def export_trajectory(
        self,
        traj_dict,
        export_npy_path,
        object_name: str | None = None,
        object_urdf_path: str | Path | None = None,
        hand_name: str | None = None,
        motion_metadata: dict[str, str | int | None] | None = None,
    ):
        q_samples = self._sample_q_trajectory(traj_dict)
        contains_lift = False
        n_reach_frames = None
        if isinstance(traj_dict, dict):
            contains_lift = bool(traj_dict.get("contains_lift", False))
            if "n_reach_frames" in traj_dict:
                n_reach_frames = int(traj_dict["n_reach_frames"])
        motion = self.construct_trajectory(
            q_samples,
            contains_lift=contains_lift,
            n_reach_frames=n_reach_frames,
        )

        active_side = self._infer_active_side()
        active_joint_names = np.asarray(self.dof_names, dtype=object)
        inactive_joint_names = np.asarray(
            [self._mirror_joint_name(name) for name in self.dof_names], dtype=object
        )
        inactive_joint_vals = np.repeat(
            self._inactive_side_template(motion["q_trajectory"][0])[None, :],
            motion["q_trajectory"].shape[0],
            axis=0,
        )

        payload = {
            "asset_pos": motion["obj_pos"].astype(np.float32),
            "asset_quat": motion["obj_quat"].astype(np.float32),
            "asset_arti_vals": motion["obj_arti"].astype(np.float32),
            "asset_arti_names": np.asarray(motion["joint_names"], dtype=object),
            "traj_hand_side": np.array(active_side, dtype=object),
            "traj_hand_name": np.array(
                hand_name if hand_name is not None else active_side, dtype=object
            ),
            "traj_object_name": np.array(
                object_name if object_name is not None else "oakink_object",
                dtype=object,
            ),
            "traj_object_urdf_path": np.array(
                "" if object_urdf_path is None else str(Path(object_urdf_path)),
                dtype=object,
            ),
            "exact_timesteps": np.arange(
                motion["q_trajectory"].shape[0], dtype=np.int64
            ),
            "n_reach_frames": np.array([motion["n_reach_frames"]], dtype=np.int64),
            "n_lift_frames": np.array([motion["n_lift_frames"]], dtype=np.int64),
            "is_active_left": np.array([active_side == "left"], dtype=bool),
            "is_active_right": np.array([active_side == "right"], dtype=bool),
        }
        if motion_metadata is not None:
            for key, value in motion_metadata.items():
                payload[key] = np.array(value, dtype=object)

        if active_side == "right":
            payload["right_joint_names"] = active_joint_names
            payload["right_joint_vals"] = motion["q_trajectory"].astype(np.float32)
            payload["left_joint_names"] = inactive_joint_names
            payload["left_joint_vals"] = inactive_joint_vals.astype(np.float32)
        else:
            payload["left_joint_names"] = active_joint_names
            payload["left_joint_vals"] = motion["q_trajectory"].astype(np.float32)
            payload["right_joint_names"] = inactive_joint_names
            payload["right_joint_vals"] = inactive_joint_vals.astype(np.float32)

        payload.update(
            self._keypoint_export_payload(
                active_side=active_side,
                active_joint_vals=motion["q_trajectory"].astype(np.float32),
                n_frames=motion["q_trajectory"].shape[0],
                human_keypoints_valid=False,
            )
        )

        export_npy_path = Path(export_npy_path)
        export_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(export_npy_path, **payload)
        return export_npy_path

    def export_static_template_motion(
        self,
        q_solution,
        export_npy_path,
        *,
        hold_frames: int = 20,
        object_name: str | None = None,
        object_urdf_path: str | Path | None = None,
        object_arti_names: list[str] | tuple[str, ...] | None = None,
        object_arti_vals: np.ndarray | None = None,
        hand_name: str | None = None,
        motion_metadata: dict[str, str | int | None] | None = None,
    ):
        q_solution = np.asarray(q_solution, dtype=float).reshape(-1)
        if q_solution.shape[0] != self.plant_f.num_positions():
            raise ValueError(
                "Expected q_solution to match the plant position dimension "
                f"{self.plant_f.num_positions()}, got {q_solution.shape[0]}."
            )
        if hold_frames < 1:
            raise ValueError(f"Expected hold_frames >= 1, got {hold_frames}.")

        active_side = self._infer_active_side()
        active_joint_names = np.asarray(self.dof_names, dtype=object)
        inactive_joint_names = np.asarray(
            [self._mirror_joint_name(name) for name in self.dof_names], dtype=object
        )
        inactive_joint_vals_single = self._inactive_side_template(q_solution)
        inactive_joint_vals = np.repeat(
            inactive_joint_vals_single[None, :],
            hold_frames,
            axis=0,
        )
        active_joint_vals = np.repeat(
            q_solution[None, :],
            hold_frames,
            axis=0,
        ).astype(np.float32)

        obj_pos = np.repeat(
            self.tgt_pose[:3, 3][None, :],
            hold_frames,
            axis=0,
        ).astype(np.float32)
        obj_quat_single = np.roll(
            R.from_matrix(self.tgt_pose[:3, :3]).as_quat(),
            1,
        ).astype(np.float32)
        obj_quat = np.repeat(obj_quat_single[None, :], hold_frames, axis=0)
        if object_arti_names is None:
            object_arti_names = ()
        joint_names = np.asarray(list(object_arti_names), dtype=object)
        if object_arti_vals is None:
            obj_arti_single = np.zeros((len(joint_names),), dtype=np.float32)
        else:
            obj_arti_single = np.asarray(object_arti_vals, dtype=np.float32).reshape(-1)
            if obj_arti_single.shape[0] != len(joint_names):
                raise ValueError(
                    "Expected object_arti_vals length to match object_arti_names, "
                    f"got {obj_arti_single.shape[0]} values and "
                    f"{len(joint_names)} names."
                )
        obj_arti = np.repeat(obj_arti_single[None, :], hold_frames, axis=0)

        payload = {
            "asset_pos": obj_pos,
            "asset_quat": obj_quat,
            "asset_arti_vals": obj_arti,
            "asset_arti_names": joint_names,
            "traj_hand_side": np.array(active_side, dtype=object),
            "traj_hand_name": np.array(
                hand_name if hand_name is not None else active_side, dtype=object
            ),
            "traj_object_name": np.array(
                object_name if object_name is not None else "oakink_object",
                dtype=object,
            ),
            "traj_object_urdf_path": np.array(
                "" if object_urdf_path is None else str(Path(object_urdf_path)),
                dtype=object,
            ),
            "exact_timesteps": np.arange(hold_frames, dtype=np.int64),
            "n_reach_frames": np.array([0], dtype=np.int64),
            "n_lift_frames": np.array([0], dtype=np.int64),
            "is_active_left": np.array([active_side == "left"], dtype=bool),
            "is_active_right": np.array([active_side == "right"], dtype=bool),
        }
        if motion_metadata is not None:
            for key, value in motion_metadata.items():
                payload[key] = np.array(value, dtype=object)

        if active_side == "right":
            payload["right_joint_names"] = active_joint_names
            payload["right_joint_vals"] = active_joint_vals
            payload["left_joint_names"] = inactive_joint_names
            payload["left_joint_vals"] = inactive_joint_vals.astype(np.float32)
        else:
            payload["left_joint_names"] = active_joint_names
            payload["left_joint_vals"] = active_joint_vals
            payload["right_joint_names"] = inactive_joint_names
            payload["right_joint_vals"] = inactive_joint_vals.astype(np.float32)

        payload.update(
            self._keypoint_export_payload(
                active_side=active_side,
                active_joint_vals=active_joint_vals,
                n_frames=hold_frames,
                human_keypoints_valid=True,
            )
        )

        export_npy_path = Path(export_npy_path)
        export_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(export_npy_path, **payload)
        return export_npy_path

    def construct_unoptimized_trajectory(self, q_solution):
        # construct a trajectory, based on interpolation in joint space
        # starting from an open configuration to the final grasp configuration, without any optimization or penetration checking
        # and adding a lift at the end
        q_solution = np.asarray(q_solution, dtype=float).reshape(-1)
        if q_solution.shape[0] != self.plant_f.num_positions():
            raise ValueError(
                "Expected q_solution to match the plant position dimension "
                f"{self.plant_f.num_positions()}, got {q_solution.shape[0]}."
            )

        n_reach = int(self.seq_optcfg.n_control_pts)
        if n_reach < 2:
            raise ValueError(
                "Unoptimized trajectory construction needs at least 2 reach frames."
            )

        q_start = self._make_open_preshape(q_solution)
        approach = np.stack(
            [
                self._clip_to_joint_limits((1.0 - alpha) * q_start + alpha * q_solution)
                for alpha in np.linspace(0.0, 1.0, n_reach)
            ],
            axis=0,
        )
        n_lift = max(10, n_reach // 2)
        lift_segment = self._build_lift_segment(
            q_solution,
            lift_steps=n_lift,
        )
        control_points = np.concatenate([approach, lift_segment], axis=0)
        duration = 0.05 * float(control_points.shape[0] - 1)
        return {
            "success": True,
            "control_points": control_points,
            "trajectory": None,
            "duration": duration,
            "basis": None,
            "rotation_var": None,
            "contains_lift": True,
            "n_reach_frames": n_reach,
            "n_lift_frames": n_lift,
        }
