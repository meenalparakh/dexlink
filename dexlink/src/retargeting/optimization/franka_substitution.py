from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import trimesh
from pydrake.all import (
    MathematicalProgram,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    RigidTransform,
    RobotDiagramBuilder,
    Role,
    SnoptSolver,
    StartMeshcat,
)
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from dexlink.src.retargeting.data.hand_cfgs.xhand import XHAND_CFGs
from dexlink.utils.retargeting.math_utils import get_X
from dexlink.utils.retargeting.path_utils import process_urdf

if not hasattr(np, "product"):
    np.product = np.prod


@dataclass
class FrankaSubstitutionConfig:
    end_effector_link_name: str = "right_hand_link"
    max_position_error_m: float = 0.1
    max_rotation_error_fro: float = 0.35
    position_cost_weight: float = 2_000.0
    rotation_cost_weight: float = 300.0
    smoothness_cost_weight: float = 0.0
    smooth_joint_limit: float = 0.9
    solver_tolerance: float = 1e-3
    enable_sequential_warm_start: bool = True
    warm_start_max_position_error_m: float = 0.10
    warm_start_max_rotation_error_fro: float = 0.50
    warm_start_position_cost_weight: float = 2_000.0
    warm_start_rotation_cost_weight: float = 200.0
    warm_start_regularization_weight: float = 5.0
    warm_start_solver_tolerance: float = 1e-3
    enable_two_stage_solve: bool = True
    stage2_max_position_error_m: float = 0.02
    stage2_max_rotation_error_fro: float = 0.10
    stage2_position_cost_weight: float = 3_000.0
    stage2_rotation_cost_weight: float = 500.0
    stage2_smoothness_cost_weight: float = 0.0
    stage2_solver_tolerance: float = 1e-5
    initial_guess: np.ndarray = field(
        default_factory=lambda: np.asarray(
            [0.0, 0, -0.0, -2.0, 0.0, 2.0, 0.0],
            dtype=float,
        )
    )
    max_joint_step: np.ndarray = field(
        default_factory=lambda: np.asarray(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=float,
        )
    )
    stage2_max_joint_step: np.ndarray = field(
        default_factory=lambda: np.asarray(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=float,
        )
    )
    franka_base_pose: np.ndarray = field(
        default_factory=lambda: np.asarray(
            [
                #     [0.0, -1.0, 0.0, -0.5],
                #     [1.0, 0.0, 0.0, 0.0],
                #     [0.0, 0.0, 1.0, 0.0],
                #     [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, -0.4],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
    )
    object_raised_by: float = (
        0.05  # same as how much the franka arm base is lower than the object base.
    )
    search_robot_base_xy_grid: bool = True
    robot_base_xy_grid: tuple[tuple[float, float], ...] = (
        (-0.6, -0.4),
        (-0.6, 0.4),
        (-0.4, -0.4),
        (-0.4, 0.4),
    )
    robot_base_object_center_distance_xy: float = 0.5
    export_in_robot_base_frame: bool = False


class FrankaSubstitution:
    def __init__(
        self,
        provided_traj_info: str | Path | Mapping[str, Any],
        franka_xhand_urdf: str | Path,
        cfg: FrankaSubstitutionConfig | None = None,
    ):
        self.provided_traj_info = provided_traj_info
        self.franka_xhand_urdf = process_urdf(franka_xhand_urdf)
        self.cfg = FrankaSubstitutionConfig() if cfg is None else cfg

        self._traj_payload = self._load_provided_trajectory(provided_traj_info)
        self.active_side = self._infer_active_side(self._traj_payload)
        if self.active_side != "right":
            raise ValueError(
                "Initial Franka substitution only supports right-hand xhand "
                f"trajectories, got active_side={self.active_side!r}."
            )

        self.source_xhand_urdf = process_urdf(XHAND_CFGs[self.active_side]["urdf_path"])
        self.active_joint_names, self.active_joint_values = (
            self._extract_active_joint_data(
                self._traj_payload,
                self.active_side,
            )
        )
        self.n_frames = int(self.active_joint_values.shape[0])
        self._validate_source_trajectory()

        self.reference_wrist_poses: np.ndarray | None = None
        self.prog: MathematicalProgram | None = None
        self.q_arm: np.ndarray | None = None
        self.result = None
        self.arm_trajectory: np.ndarray | None = None
        self.warm_start_trajectory: np.ndarray | None = None
        self.export_payload: dict[str, Any] | None = None
        self.meshcat = None
        self.initial_robot_base_pose_world = self.cfg.franka_base_pose.copy()

        self.build_source_robot()

        z = self.get_demo_base() - self.cfg.object_raised_by
        print(
            f"Inferred demo base height z={z:.3f} from trajectory payload; using this to set Franka base pose."
        )
        self.robot_base_pose = self.cfg.franka_base_pose.copy()
        self.robot_base_pose[2, 3] = z
        self.initial_robot_base_pose_world = self.robot_base_pose.copy()

        self.build_plant()
        self.compute_fk()

    def get_demo_base(self) -> float:
        asset_pos = np.asarray(self._traj_payload.get("asset_pos"), dtype=float)
        asset_quat = np.asarray(self._traj_payload.get("asset_quat"), dtype=float)

        if asset_pos.ndim != 2 or asset_pos.shape[1] != 3:
            raise ValueError(
                f"Expected `asset_pos` to have shape (T, 3), got {asset_pos.shape}."
            )
        if asset_quat.ndim != 2 or asset_quat.shape[1] != 4:
            raise ValueError(
                f"Expected `asset_quat` to have shape (T, 4) in wxyz format, got {asset_quat.shape}."
            )

        first_obj_pos = asset_pos[0]
        first_obj_quat = asset_quat[0]
        base_pose = get_X(first_obj_pos, first_obj_quat)

        target_obj_path = self._traj_payload.get("oakink_target_obj_path")
        if target_obj_path is not None:
            target_obj_path = self._extract_scalar(target_obj_path)
            if target_obj_path not in ("", "None", None):
                target_mesh = trimesh.load_mesh(str(Path(target_obj_path)))
                if isinstance(target_mesh, trimesh.Scene):
                    mesh_geometry = tuple(target_mesh.geometry.values())
                    if len(mesh_geometry) == 0:
                        raise ValueError(
                            f"Expected mesh geometry in oakink_target_obj_path={target_obj_path!r}, got an empty scene."
                        )
                    target_mesh = trimesh.util.concatenate(
                        [geometry.copy() for geometry in mesh_geometry]
                    )
                if not isinstance(target_mesh, trimesh.Trimesh):
                    raise ValueError(
                        f"Expected oakink_target_obj_path={target_obj_path!r} to load as a mesh, got "
                        f"{type(target_mesh).__name__}."
                    )
                target_mesh = target_mesh.copy()
                target_mesh.apply_transform(base_pose)
                return float(target_mesh.bounds[0, 2])

        object_urdf_path = self._traj_payload.get("traj_object_urdf_path")
        if object_urdf_path is None:
            raise KeyError(
                "Trajectory payload is missing both `oakink_target_obj_path` and "
                "`traj_object_urdf_path`, so the demo object base height cannot be computed."
            )
        object_urdf_path = process_urdf(str(self._extract_scalar(object_urdf_path)))

        asset_arti_vals = np.asarray(
            self._traj_payload.get("asset_arti_vals"), dtype=float
        )
        asset_arti_names = np.asarray(
            self._traj_payload.get("asset_arti_names", np.asarray([], dtype=object)),
            dtype=object,
        ).reshape(-1)
        if asset_arti_vals.ndim != 2:
            raise ValueError(
                f"Expected `asset_arti_vals` to have shape (T, n_joint), got {asset_arti_vals.shape}."
            )
        if asset_arti_vals.shape[0] != asset_pos.shape[0]:
            raise ValueError(
                f"`asset_arti_vals` has {asset_arti_vals.shape[0]} frames but `asset_pos` has "
                f"{asset_pos.shape[0]} frames."
            )
        if asset_arti_vals.shape[1] != asset_arti_names.shape[0]:
            raise ValueError(
                f"`asset_arti_vals` has {asset_arti_vals.shape[1]} articulation columns but "
                f"`asset_arti_names` has {asset_arti_names.shape[0]} names."
            )

        object_urdf = URDF.load(str(Path(object_urdf_path)))
        first_obj_arti = asset_arti_vals[0].reshape(-1)
        joint_dict = dict(zip(asset_arti_names.tolist(), first_obj_arti.tolist()))

        fk = object_urdf.visual_trimesh_fk(joint_dict)
        if len(fk) == 0:
            fk = object_urdf.collision_trimesh_fk(joint_dict)
        if len(fk) == 0:
            raise ValueError(
                f"No visual or collision meshes were found in object URDF {object_urdf_path}."
            )

        mesh_pieces = []
        for mesh, pose in fk.items():
            transformed_mesh = mesh.copy()
            transformed_mesh.apply_transform(pose)
            mesh_pieces.append(transformed_mesh)
        combined_mesh = trimesh.util.concatenate(mesh_pieces)
        combined_mesh.apply_transform(base_pose)
        return float(combined_mesh.bounds[0, 2])

    @staticmethod
    def _load_provided_trajectory(
        provided_traj_info: str | Path | Mapping[str, Any]
    ) -> dict[str, Any]:
        if isinstance(provided_traj_info, (str, Path)):
            traj_path = Path(provided_traj_info)
            if traj_path.suffix != ".npz":
                raise ValueError(
                    "Expected provided trajectory to be an .npz file exported by "
                    f"the motion sequence optimizer, got {traj_path}."
                )
            with np.load(traj_path, allow_pickle=True) as data:
                return {key: data[key] for key in data.files}

        if not isinstance(provided_traj_info, Mapping):
            raise TypeError(
                "provided_traj_info must be either a path to a saved .npz trajectory "
                f"or a mapping payload, got {type(provided_traj_info).__name__}."
            )
        return dict(provided_traj_info)

    @staticmethod
    def _extract_scalar(value: Any) -> Any:
        if isinstance(value, np.ndarray) and value.shape == ():
            return value.item()
        return value

    def _infer_active_side(self, payload: Mapping[str, Any]) -> str:
        if "traj_hand_side" in payload:
            side = str(self._extract_scalar(payload["traj_hand_side"]))
            if side in {"left", "right"}:
                return side

        if "is_active_right" in payload and bool(
            np.asarray(payload["is_active_right"]).reshape(-1)[0]
        ):
            return "right"
        if "is_active_left" in payload and bool(
            np.asarray(payload["is_active_left"]).reshape(-1)[0]
        ):
            return "left"

        raise ValueError(
            "Could not infer active hand side from trajectory payload. Expected "
            "`traj_hand_side` or `is_active_left` / `is_active_right`."
        )

    @staticmethod
    def _extract_active_joint_data(
        payload: Mapping[str, Any], active_side: str
    ) -> tuple[np.ndarray, np.ndarray]:
        joint_name_key = f"{active_side}_joint_names"
        joint_val_key = f"{active_side}_joint_vals"
        if joint_name_key not in payload or joint_val_key not in payload:
            raise KeyError(
                f"Trajectory payload must contain {joint_name_key!r} and "
                f"{joint_val_key!r}."
            )

        joint_names = np.asarray(payload[joint_name_key], dtype=object).reshape(-1)
        joint_values = np.asarray(payload[joint_val_key], dtype=float)
        if joint_values.ndim != 2:
            raise ValueError(
                f"Expected {joint_val_key} to have shape (T, nq), got {joint_values.shape}."
            )
        if joint_values.shape[1] != joint_names.shape[0]:
            raise ValueError(
                f"Joint name/value mismatch: {joint_names.shape[0]} names for "
                f"{joint_values.shape[1]} trajectory columns."
            )
        return joint_names, joint_values

    def _validate_source_trajectory(self) -> None:
        expected_root_joint_names = [
            "R_forearm_tx_link_joint",
            "R_forearm_ty_link_joint",
            "R_forearm_tz_link_joint",
            "R_forearm_roll_link_joint",
            "R_forearm_pitch_link_joint",
            "R_forearm_yaw_link_joint",
        ]
        missing_root_joints = [
            joint_name
            for joint_name in expected_root_joint_names
            if joint_name not in set(self.active_joint_names.tolist())
        ]
        if missing_root_joints:
            raise ValueError(
                "Provided trajectory does not look like a floating xhand motion. "
                f"Missing expected 6-DoF joints: {missing_root_joints}."
                f"Active joint names from trajectory: {self.active_joint_names.tolist()}."
            )

    def _get_root_joint_indices(self) -> dict[str, int]:
        joint_name_to_index = {
            str(name): idx for idx, name in enumerate(self.active_joint_names.tolist())
        }
        root_joint_names = {
            "tx": "R_forearm_tx_link_joint",
            "ty": "R_forearm_ty_link_joint",
            "tz": "R_forearm_tz_link_joint",
            "roll": "R_forearm_roll_link_joint",
            "pitch": "R_forearm_pitch_link_joint",
            "yaw": "R_forearm_yaw_link_joint",
        }
        return {
            key: joint_name_to_index[joint_name]
            for key, joint_name in root_joint_names.items()
        }

    def _source_wrist_pose_from_joint_values(
        self, joint_values: np.ndarray
    ) -> np.ndarray:
        joint_cfg = {
            str(joint_name): float(joint_values[traj_col])
            for traj_col, joint_name in enumerate(self.active_joint_names.tolist())
        }
        fk_links = self.source_robot.link_fk(cfg=joint_cfg)
        for link, transform in fk_links.items():
            if link.name == self.cfg.end_effector_link_name:
                return np.asarray(transform, dtype=float)
        raise ValueError(
            f"urdfpy FK did not return a transform for end-effector link "
            f"{self.cfg.end_effector_link_name!r}."
        )

    def _set_robot_base_xy(self, x: float, y: float) -> None:
        updated_pose = np.asarray(self.robot_base_pose, dtype=float).copy()
        updated_pose[0, 3] = float(x)
        updated_pose[1, 3] = float(y)
        self.robot_base_pose = updated_pose
        self.initial_robot_base_pose_world = updated_pose.copy()
        self.build_plant()

    def _robot_base_xy_candidates(self) -> list[tuple[float, float]]:
        configured_xy = (
            float(self.cfg.franka_base_pose[0, 3]),
            float(self.cfg.franka_base_pose[1, 3]),
        )
        if not self.cfg.search_robot_base_xy_grid:
            return [configured_xy]

        candidates = [(float(x), float(y)) for x, y in self.cfg.robot_base_xy_grid]
        if configured_xy not in candidates:
            candidates.insert(0, configured_xy)
        return candidates

    @staticmethod
    def _transform_positions_by_matrix(
        positions: np.ndarray,
        transform: np.ndarray,
    ) -> np.ndarray:
        positions = np.asarray(positions, dtype=float)
        if positions.shape[-1] != 3:
            raise ValueError(
                f"Expected positions to end in dimension 3, got {positions.shape}."
            )
        flat_positions = positions.reshape(-1, 3)
        flat_h = np.concatenate(
            [flat_positions, np.ones((flat_positions.shape[0], 1), dtype=float)],
            axis=1,
        )
        transformed = (transform @ flat_h.T).T[:, :3]
        return transformed.reshape(positions.shape)

    @staticmethod
    def _transform_quat_wxyz_by_matrix(
        quat_wxyz: np.ndarray,
        transform: np.ndarray,
    ) -> np.ndarray:
        quat_wxyz = np.asarray(quat_wxyz, dtype=float)
        if quat_wxyz.shape[-1] != 4:
            raise ValueError(
                f"Expected quaternions to end in dimension 4, got {quat_wxyz.shape}."
            )
        world_to_target = R.from_matrix(transform[:3, :3])
        quat_xyzw = np.roll(quat_wxyz.reshape(-1, 4), -1, axis=1)
        rotated_xyzw = (world_to_target * R.from_quat(quat_xyzw)).as_quat()
        return np.roll(rotated_xyzw, 1, axis=1).reshape(quat_wxyz.shape)

    def _export_payload_in_robot_base_frame(
        self,
        export_payload: dict[str, Any],
    ) -> dict[str, Any]:
        world_to_robot = np.linalg.inv(
            np.asarray(self.initial_robot_base_pose_world, dtype=float)
        )
        transformed_payload = dict(export_payload)

        if "asset_pos" in transformed_payload:
            transformed_payload["asset_pos"] = self._transform_positions_by_matrix(
                np.asarray(transformed_payload["asset_pos"], dtype=float),
                world_to_robot,
            ).astype(np.float32)
        if "asset_quat" in transformed_payload:
            transformed_payload["asset_quat"] = self._transform_quat_wxyz_by_matrix(
                np.asarray(transformed_payload["asset_quat"], dtype=float),
                world_to_robot,
            ).astype(np.float32)

        transformed_reference_wrist_poses = np.einsum(
            "ij,tjk->tik",
            world_to_robot,
            np.asarray(self.reference_wrist_poses, dtype=float),
        ).astype(np.float32)
        transformed_payload["reference_wrist_poses"] = transformed_reference_wrist_poses
        transformed_payload["traj_robot_base_pose"] = np.eye(4, dtype=np.float32)
        transformed_payload["traj_robot_base_pose_original_world"] = np.asarray(
            self.initial_robot_base_pose_world,
            dtype=np.float32,
        )
        transformed_payload["traj_exported_in_robot_base_frame"] = np.asarray(
            [True],
            dtype=bool,
        )
        return transformed_payload

    def build_source_robot(self) -> None:
        self.source_robot = URDF.load(str(Path(self.source_xhand_urdf)))

        source_link_names = {link.name for link in self.source_robot.links}
        if self.cfg.end_effector_link_name not in source_link_names:
            raise ValueError(
                f"End-effector link {self.cfg.end_effector_link_name!r} is not present "
                f"in source URDF {self.source_xhand_urdf}."
            )

        self.source_actuated_joint_names = [
            joint.name for joint in self.source_robot.actuated_joints
        ]
        missing_traj_joints = [
            joint_name
            for joint_name in self.active_joint_names.tolist()
            if joint_name not in self.source_actuated_joint_names
        ]
        if missing_traj_joints:
            raise ValueError(
                "Provided trajectory contains joints that are not actuated joints in "
                f"the source xhand URDF: {missing_traj_joints}."
            )

    def compute_fk(self) -> np.ndarray:
        reference_poses = []
        for frame_idx in range(self.n_frames):
            joint_cfg = {
                str(joint_name): float(self.active_joint_values[frame_idx, traj_col])
                for traj_col, joint_name in enumerate(self.active_joint_names.tolist())
            }
            fk_links = self.source_robot.link_fk(cfg=joint_cfg)

            target_link_transform = None
            for link, transform in fk_links.items():
                if link.name == self.cfg.end_effector_link_name:
                    target_link_transform = transform
                    break

            if target_link_transform is None:
                raise ValueError(
                    f"urdfpy FK did not return a transform for end-effector link "
                    f"{self.cfg.end_effector_link_name!r}."
                )
            reference_poses.append(np.asarray(target_link_transform, dtype=float))

        self.reference_wrist_poses = np.asarray(reference_poses, dtype=float)
        return self.reference_wrist_poses

    def build_plant(self) -> None:
        builder = RobotDiagramBuilder(time_step=0.001)
        plant_f = builder.plant()
        parser = builder.parser()
        (franka_model,) = parser.AddModels(self.franka_xhand_urdf)

        world_frame = plant_f.world_frame()
        plant_f.WeldFrames(
            world_frame,
            plant_f.GetFrameByName("panda_link0", franka_model),
            X_FM=RigidTransform(self.robot_base_pose),
        )

        diagram_f = builder.Build()
        context_f = diagram_f.CreateDefaultContext()
        plant_context_f = plant_f.GetMyContextFromRoot(context_f)

        plant_ad = plant_f.ToAutoDiffXd()
        context_ad = plant_ad.CreateDefaultContext()

        model_name = plant_f.GetModelInstanceName(franka_model)
        franka_model_ad = plant_ad.GetModelInstanceByName(model_name)

        self.plant_f = plant_f
        self.diagram_f = diagram_f
        self.context_f = context_f
        self.plant_context_f = plant_context_f
        self.world_frame_f = world_frame
        self.franka_model = franka_model

        self.plant_ad = plant_ad
        self.context_ad = context_ad
        self.world_frame_ad = plant_ad.world_frame()
        self.franka_model_ad = franka_model_ad
        self.end_effector_frame_f = plant_f.GetFrameByName(
            self.cfg.end_effector_link_name,
            franka_model,
        )
        self.end_effector_frame_ad = plant_ad.GetFrameByName(
            self.cfg.end_effector_link_name,
            franka_model_ad,
        )

        self.franka_position_names = np.asarray(
            plant_f.GetPositionNames(
                add_model_instance_prefix=False,
                always_add_suffix=False,
            ),
            dtype=object,
        )
        self.franka_joint_name_to_index = {
            str(name): idx
            for idx, name in enumerate(self.franka_position_names.tolist())
        }
        active_joint_index_map = {
            str(name): idx for idx, name in enumerate(self.active_joint_names.tolist())
        }

        self.arm_joint_names = np.asarray(
            [
                name
                for name in self.franka_position_names.tolist()
                if str(name).startswith("panda_joint")
            ],
            dtype=object,
        )
        if self.arm_joint_names.shape[0] != 7:
            raise ValueError(
                "Expected Franka plant to expose 7 panda joints, got "
                f"{self.arm_joint_names.shape[0]} from {self.franka_position_names.tolist()}."
            )
        self.arm_joint_indices = np.asarray(
            [
                self.franka_joint_name_to_index[str(name)]
                for name in self.arm_joint_names.tolist()
            ],
            dtype=int,
        )

        self.finger_joint_names = np.asarray(
            [
                name
                for name in self.active_joint_names.tolist()
                if "forearm_" not in str(name)
            ],
            dtype=object,
        )
        missing_finger_joints = [
            str(name)
            for name in self.finger_joint_names.tolist()
            if str(name) not in self.franka_joint_name_to_index
        ]
        if missing_finger_joints:
            raise ValueError(
                "The Franka+XHand URDF is missing finger joints from the source xhand "
                f"trajectory: {missing_finger_joints}."
            )
        self.finger_joint_indices = np.asarray(
            [
                self.franka_joint_name_to_index[str(name)]
                for name in self.finger_joint_names.tolist()
            ],
            dtype=int,
        )
        self.finger_trajectory = np.asarray(
            [
                self.active_joint_values[:, active_joint_index_map[str(name)]]
                for name in self.finger_joint_names.tolist()
            ],
            dtype=float,
        ).T

        lower_limits = plant_f.GetPositionLowerLimits()
        upper_limits = plant_f.GetPositionUpperLimits()
        self.arm_lower_limits = np.asarray(
            lower_limits[self.arm_joint_indices], dtype=float
        )
        self.arm_upper_limits = np.asarray(
            upper_limits[self.arm_joint_indices], dtype=float
        )

        # update the limits to be within the smooth_joint_limit fraction of the full joint range, to encourage smoother motions and avoid hitting hard limits.
        joint_ranges = self.arm_upper_limits - self.arm_lower_limits
        mid_points = (self.arm_upper_limits + self.arm_lower_limits) / 2
        limit_margin = self.cfg.smooth_joint_limit * joint_ranges / 2
        self.arm_lower_limits = np.maximum(
            self.arm_lower_limits, mid_points - limit_margin
        )
        self.arm_upper_limits = np.minimum(
            self.arm_upper_limits, mid_points + limit_margin
        )

        print(
            f"Franka arm joint limits inferred from plant: lower={self.arm_lower_limits}, "
            f"upper={self.arm_upper_limits}."
        )

    def get_meshcat(self):
        if self.meshcat is None:
            self.meshcat = StartMeshcat()
        return self.meshcat

    def init_prog(self, initial_guess: np.ndarray | None = None) -> None:
        self.prog = MathematicalProgram()
        self.q_arm = self.prog.NewContinuousVariables(
            self.arm_joint_names.shape[0], self.n_frames, "panda_q"
        )
        self.prog.AddBoundingBoxConstraint(
            self.arm_lower_limits[:, None].repeat(self.n_frames, axis=1),
            self.arm_upper_limits[:, None].repeat(self.n_frames, axis=1),
            self.q_arm,
        )

        tiled_initial_guess = self._prepare_trajectory_initial_guess(initial_guess)
        self.prog.SetInitialGuess(self.q_arm, tiled_initial_guess)

    def _prepare_trajectory_initial_guess(
        self,
        initial_guess: np.ndarray | None,
    ) -> np.ndarray:
        if initial_guess is None:
            arm_seed = np.asarray(self.cfg.initial_guess, dtype=float).reshape(-1)
            if arm_seed.shape[0] != self.arm_joint_names.shape[0]:
                raise ValueError(
                    "Expected initial_guess to have 7 entries for the Franka arm, got "
                    f"{arm_seed.shape[0]}."
                )
            tiled_initial_guess = np.repeat(arm_seed[:, None], self.n_frames, axis=1)
        else:
            tiled_initial_guess = np.asarray(initial_guess, dtype=float)
            expected_shape = (self.arm_joint_names.shape[0], self.n_frames)
            if tiled_initial_guess.shape != expected_shape:
                raise ValueError(
                    "Expected trajectory initial guess to have shape "
                    f"{expected_shape}, got {tiled_initial_guess.shape}."
                )
        return np.clip(
            tiled_initial_guess,
            self.arm_lower_limits[:, None],
            self.arm_upper_limits[:, None],
        )

    def _compose_full_franka_q(self, arm_q: np.ndarray, frame_idx: int) -> np.ndarray:
        full_q = np.zeros(self.plant_f.num_positions(self.franka_model), dtype=object)
        full_q[self.arm_joint_indices] = arm_q
        full_q[self.finger_joint_indices] = self.finger_trajectory[frame_idx]
        return full_q

    def _end_effector_terms(self, arm_q: np.ndarray, frame_idx: int) -> tuple[Any, Any]:
        q_full = self._compose_full_franka_q(arm_q, frame_idx)
        self.plant_ad.SetPositions(self.context_ad, self.franka_model_ad, q_full)
        X_WE = self.plant_ad.CalcRelativeTransform(
            self.context_ad,
            self.world_frame_ad,
            self.end_effector_frame_ad,
        )
        target_pose = self.reference_wrist_poses[frame_idx]
        translation_error = X_WE.translation() - target_pose[:3, 3]
        rotation_error = X_WE.rotation().matrix() - target_pose[:3, :3]
        return translation_error, rotation_error

    def _end_effector_terms_float(
        self, arm_q: np.ndarray, frame_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        q_full = np.asarray(self._compose_full_franka_q(arm_q, frame_idx), dtype=float)
        self.plant_f.SetPositions(self.plant_context_f, self.franka_model, q_full)
        X_WE = self.plant_f.CalcRelativeTransform(
            self.plant_context_f,
            self.world_frame_f,
            self.end_effector_frame_f,
        )
        target_pose = self.reference_wrist_poses[frame_idx]
        translation_error = (
            np.asarray(X_WE.translation(), dtype=float) - target_pose[:3, 3]
        )
        rotation_error = (
            np.asarray(X_WE.rotation().matrix(), dtype=float) - target_pose[:3, :3]
        )
        return translation_error, rotation_error

    def _tracking_cost(
        self,
        arm_q: np.ndarray,
        frame_idx: int,
        position_cost_weight: float,
        rotation_cost_weight: float,
    ) -> Any:
        translation_error, rotation_error = self._end_effector_terms(arm_q, frame_idx)
        translation_cost = np.sum(translation_error * translation_error)
        rotation_cost = np.sum(rotation_error * rotation_error)
        return (
            position_cost_weight * translation_cost
            + rotation_cost_weight * rotation_cost
        )

    def _translation_constraint_value(
        self, arm_q: np.ndarray, frame_idx: int
    ) -> np.ndarray:
        translation_error, _ = self._end_effector_terms(arm_q, frame_idx)
        return np.asarray([np.sum(translation_error * translation_error)])

    def _rotation_constraint_value(
        self, arm_q: np.ndarray, frame_idx: int
    ) -> np.ndarray:
        _, rotation_error = self._end_effector_terms(arm_q, frame_idx)
        return np.asarray([np.sum(rotation_error * rotation_error)])

    @staticmethod
    def _smoothness_cost(pair_q: np.ndarray) -> Any:
        n_joints = pair_q.shape[0] // 2
        delta = pair_q[n_joints:] - pair_q[:n_joints]
        return np.sum(delta * delta)

    def add_costs(
        self,
        position_cost_weight: float,
        rotation_cost_weight: float,
        smoothness_cost_weight: float,
    ) -> None:
        for frame_idx in range(self.n_frames):
            self.prog.AddCost(
                lambda x, current_frame=frame_idx: self._tracking_cost(
                    x,
                    current_frame,
                    position_cost_weight,
                    rotation_cost_weight,
                ),
                self.q_arm[:, frame_idx],
            )

        for frame_idx in range(self.n_frames - 1):
            self.prog.AddCost(
                lambda x: smoothness_cost_weight * self._smoothness_cost(x),
                np.concatenate(
                    [self.q_arm[:, frame_idx], self.q_arm[:, frame_idx + 1]]
                ),
            )

    def add_constraints(
        self,
        max_position_error_m: float,
        max_rotation_error_fro: float,
        max_joint_step: np.ndarray,
    ) -> None:
        position_bound_sq = float(max_position_error_m**2)
        rotation_bound_sq = float(max_rotation_error_fro**2)
        for frame_idx in range(self.n_frames):
            self.prog.AddConstraint(
                lambda x, current_frame=frame_idx: self._translation_constraint_value(
                    x, current_frame
                ),
                [0.0],
                [position_bound_sq],
                self.q_arm[:, frame_idx],
            )
            self.prog.AddConstraint(
                lambda x, current_frame=frame_idx: self._rotation_constraint_value(
                    x, current_frame
                ),
                [0.0],
                [rotation_bound_sq],
                self.q_arm[:, frame_idx],
            )

        max_joint_step = np.asarray(max_joint_step, dtype=float).reshape(-1)
        if max_joint_step.shape[0] != self.arm_joint_names.shape[0]:
            raise ValueError(
                "Expected max_joint_step to provide one limit per panda joint, got "
                f"{max_joint_step.shape[0]} entries."
            )

        delta_matrix = np.hstack(
            [
                -np.eye(self.arm_joint_names.shape[0]),
                np.eye(self.arm_joint_names.shape[0]),
            ]
        )
        for frame_idx in range(self.n_frames - 1):
            decision_pair = np.concatenate(
                [self.q_arm[:, frame_idx], self.q_arm[:, frame_idx + 1]]
            )
            self.prog.AddLinearConstraint(
                delta_matrix,
                -max_joint_step,
                max_joint_step,
                decision_pair,
            )

    def _configure_snopt(
        self, prog: MathematicalProgram, solver_tolerance: float
    ) -> SnoptSolver:
        solver = SnoptSolver()
        solver_id = solver.solver_id()
        prog.SetSolverOption(solver_id, "Feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(
            solver_id,
            "Major feasibility tolerance",
            solver_tolerance,
        )
        prog.SetSolverOption(
            solver_id,
            "Major optimality tolerance",
            solver_tolerance,
        )
        return solver

    def _solve_trajectory_problem(
        self,
        *,
        max_position_error_m: float,
        max_rotation_error_fro: float,
        position_cost_weight: float,
        rotation_cost_weight: float,
        smoothness_cost_weight: float,
        max_joint_step: np.ndarray,
        solver_tolerance: float,
        initial_guess: np.ndarray | None,
    ):
        self.init_prog(initial_guess=initial_guess)
        self.add_costs(
            position_cost_weight=position_cost_weight,
            rotation_cost_weight=rotation_cost_weight,
            smoothness_cost_weight=smoothness_cost_weight,
        )
        self.add_constraints(
            max_position_error_m=max_position_error_m,
            max_rotation_error_fro=max_rotation_error_fro,
            max_joint_step=max_joint_step,
        )
        solver = self._configure_snopt(self.prog, solver_tolerance)
        result = solver.Solve(self.prog)
        arm_trajectory = np.asarray(result.GetSolution(self.q_arm), dtype=float)
        return result, arm_trajectory

    def _solve_warm_start_frame(
        self,
        frame_idx: int,
        seed_q: np.ndarray,
    ) -> np.ndarray:
        warm_seed = np.asarray(seed_q, dtype=float).reshape(-1)
        if warm_seed.shape[0] != self.arm_joint_names.shape[0]:
            raise ValueError(
                "Expected warm-start seed to have 7 entries for the Franka arm, got "
                f"{warm_seed.shape[0]}."
            )

        prog = MathematicalProgram()
        q = prog.NewContinuousVariables(self.arm_joint_names.shape[0], "panda_q_warm")
        prog.AddBoundingBoxConstraint(self.arm_lower_limits, self.arm_upper_limits, q)
        prog.SetInitialGuess(
            q, np.clip(warm_seed, self.arm_lower_limits, self.arm_upper_limits)
        )
        prog.AddCost(
            lambda x, current_frame=frame_idx: self._tracking_cost(
                x,
                current_frame,
                self.cfg.warm_start_position_cost_weight,
                self.cfg.warm_start_rotation_cost_weight,
            ),
            q,
        )
        prog.AddCost(
            lambda x, target=warm_seed: self.cfg.warm_start_regularization_weight
            * np.sum((x - target) * (x - target)),
            q,
        )
        prog.AddConstraint(
            lambda x, current_frame=frame_idx: self._translation_constraint_value(
                x, current_frame
            ),
            [0.0],
            [float(self.cfg.warm_start_max_position_error_m**2)],
            q,
        )
        prog.AddConstraint(
            lambda x, current_frame=frame_idx: self._rotation_constraint_value(
                x, current_frame
            ),
            [0.0],
            [float(self.cfg.warm_start_max_rotation_error_fro**2)],
            q,
        )

        solver = self._configure_snopt(prog, self.cfg.warm_start_solver_tolerance)
        result = solver.Solve(prog)
        if not result.is_success():
            pass
            # raise RuntimeError(
            #     "Sequential warm-start IK failed at frame "
            #     f"{frame_idx} with solver status {result.get_solution_result()}."
            # )
        return np.asarray(result.GetSolution(q), dtype=float)

    def _build_warm_start_trajectory(self) -> np.ndarray:
        if not self.cfg.enable_sequential_warm_start:
            warm_start = self._prepare_trajectory_initial_guess(None)
            self.warm_start_trajectory = warm_start.T
            return warm_start

        warm_start = np.zeros(
            (self.arm_joint_names.shape[0], self.n_frames), dtype=float
        )
        prev_q = np.asarray(self.cfg.initial_guess, dtype=float).reshape(-1)
        for frame_idx in range(self.n_frames):
            prev_q = self._solve_warm_start_frame(frame_idx, prev_q)
            warm_start[:, frame_idx] = prev_q
        self.warm_start_trajectory = warm_start.T
        return warm_start

    def _evaluate_trajectory_quality(
        self,
        arm_trajectory: np.ndarray,
        *,
        max_position_error_m: float,
        max_rotation_error_fro: float,
        max_joint_step: np.ndarray,
    ) -> dict[str, float | bool]:
        arm_trajectory = np.asarray(arm_trajectory, dtype=float)
        expected_shape = (self.arm_joint_names.shape[0], self.n_frames)
        if arm_trajectory.shape != expected_shape:
            raise ValueError(
                f"Expected arm_trajectory to have shape {expected_shape}, got {arm_trajectory.shape}."
            )

        translation_error_sq = np.zeros(self.n_frames, dtype=float)
        rotation_error_sq = np.zeros(self.n_frames, dtype=float)
        for frame_idx in range(self.n_frames):
            q_frame = arm_trajectory[:, frame_idx]
            translation_error, rotation_error = self._end_effector_terms_float(
                q_frame, frame_idx
            )
            translation_error_sq[frame_idx] = float(
                np.sum(translation_error * translation_error)
            )
            rotation_error_sq[frame_idx] = float(
                np.sum(rotation_error * rotation_error)
            )

        max_joint_step = np.asarray(max_joint_step, dtype=float).reshape(-1)
        if max_joint_step.shape[0] != self.arm_joint_names.shape[0]:
            raise ValueError(
                "Expected max_joint_step to provide one limit per panda joint, got "
                f"{max_joint_step.shape[0]} entries."
            )

        if self.n_frames > 1:
            step_violation = (
                np.abs(np.diff(arm_trajectory, axis=1)) - max_joint_step[:, None]
            )
            max_joint_step_violation = float(np.max(step_violation))
        else:
            max_joint_step_violation = 0.0

        position_bound_sq = float(max_position_error_m**2)
        rotation_bound_sq = float(max_rotation_error_fro**2)
        max_position_violation = float(np.max(translation_error_sq - position_bound_sq))
        max_rotation_violation = float(np.max(rotation_error_sq - rotation_bound_sq))
        finite_solution = bool(np.all(np.isfinite(arm_trajectory)))

        return {
            "finite_solution": finite_solution,
            "max_position_error_m": float(np.sqrt(np.max(translation_error_sq))),
            "max_rotation_error_fro": float(np.sqrt(np.max(rotation_error_sq))),
            "max_position_violation_sq": max(0.0, max_position_violation),
            "max_rotation_violation_sq": max(0.0, max_rotation_violation),
            "max_joint_step_violation": max(0.0, max_joint_step_violation),
            "within_constraints": bool(
                finite_solution
                and max_position_violation <= 0.0
                and max_rotation_violation <= 0.0
                and max_joint_step_violation <= 0.0
            ),
        }

    def solve(self):
        last_result = None
        last_arm_trajectory = None
        last_quality = None
        candidate_xy_list = self._robot_base_xy_candidates()

        for candidate_x, candidate_y in candidate_xy_list:
            print(
                f"Trying Franka base XY candidate ({candidate_x:.3f}, {candidate_y:.3f})"
            )
            self._set_robot_base_xy(candidate_x, candidate_y)

            initial_guess = self._build_warm_start_trajectory()
            result, arm_trajectory = self._solve_trajectory_problem(
                max_position_error_m=self.cfg.max_position_error_m,
                max_rotation_error_fro=self.cfg.max_rotation_error_fro,
                position_cost_weight=self.cfg.position_cost_weight,
                rotation_cost_weight=self.cfg.rotation_cost_weight,
                smoothness_cost_weight=self.cfg.smoothness_cost_weight,
                max_joint_step=self.cfg.max_joint_step,
                solver_tolerance=self.cfg.solver_tolerance,
                initial_guess=initial_guess,
            )

            stage_quality = self._evaluate_trajectory_quality(
                arm_trajectory,
                max_position_error_m=self.cfg.max_position_error_m,
                max_rotation_error_fro=self.cfg.max_rotation_error_fro,
                max_joint_step=self.cfg.max_joint_step,
            )
            if not stage_quality["finite_solution"]:
                print(
                    "Candidate produced non-finite stage-1 trajectory; trying next base XY."
                )
                last_result = result
                last_arm_trajectory = arm_trajectory
                last_quality = stage_quality
                continue

            if self.cfg.enable_two_stage_solve:
                result, arm_trajectory = self._solve_trajectory_problem(
                    max_position_error_m=self.cfg.stage2_max_position_error_m,
                    max_rotation_error_fro=self.cfg.stage2_max_rotation_error_fro,
                    position_cost_weight=self.cfg.stage2_position_cost_weight,
                    rotation_cost_weight=self.cfg.stage2_rotation_cost_weight,
                    smoothness_cost_weight=self.cfg.stage2_smoothness_cost_weight,
                    max_joint_step=self.cfg.stage2_max_joint_step,
                    solver_tolerance=self.cfg.stage2_solver_tolerance,
                    initial_guess=arm_trajectory,
                )
                final_quality = self._evaluate_trajectory_quality(
                    arm_trajectory,
                    max_position_error_m=self.cfg.stage2_max_position_error_m,
                    max_rotation_error_fro=self.cfg.stage2_max_rotation_error_fro,
                    max_joint_step=self.cfg.stage2_max_joint_step,
                )
            else:
                final_quality = stage_quality

            last_result = result
            last_arm_trajectory = arm_trajectory
            last_quality = final_quality

            if result.is_success() and final_quality["within_constraints"]:
                print(
                    f"Selected Franka base XY candidate ({candidate_x:.3f}, {candidate_y:.3f})."
                )
                self.result = result
                self.arm_trajectory = arm_trajectory.T
                return self.result

            print(
                "Candidate failed quality check: "
                f"solver_success={result.is_success()} "
                f"within_constraints={final_quality['within_constraints']}."
            )

        if last_result is None or last_arm_trajectory is None or last_quality is None:
            raise RuntimeError(
                "Franka substitution did not evaluate any base XY candidates."
            )

        self.result = last_result
        self.arm_trajectory = last_arm_trajectory.T
        raise RuntimeError(
            "Franka substitution failed for all base XY candidates. Last candidate had "
            f"solver_success={last_result.is_success()} "
            f"within_constraints={last_quality['within_constraints']} "
            f"max_position_error_m={last_quality['max_position_error_m']:.6f} "
            f"max_rotation_error_fro={last_quality['max_rotation_error_fro']:.6f}."
        )

    def visualize_trajectory(
        self,
        arm_trajectory: np.ndarray | None = None,
        pause: bool = True,
        time_step: float = 0.05,
    ):
        if arm_trajectory is None:
            if self.arm_trajectory is None:
                raise RuntimeError(
                    "Call solve() before visualize_trajectory(), or pass an explicit arm_trajectory."
                )
            arm_trajectory = self.arm_trajectory

        arm_trajectory = np.asarray(arm_trajectory, dtype=float)
        if arm_trajectory.shape != (self.n_frames, self.arm_joint_names.shape[0]):
            raise ValueError(
                "Expected arm_trajectory to have shape "
                f"({self.n_frames}, {self.arm_joint_names.shape[0]}), got {arm_trajectory.shape}."
            )

        meshcat = self.get_meshcat()
        meshcat.Delete()
        meshcat.DeleteAddedControls()

        builder = RobotDiagramBuilder(time_step=0.001)
        plant = builder.plant()
        parser = builder.parser()
        diagram_builder = builder.builder()
        scene_graph = builder.scene_graph()

        (source_model,) = parser.AddModels(self.source_xhand_urdf)
        (franka_model,) = parser.AddModels(self.franka_xhand_urdf)

        world_frame = plant.world_frame()
        plant.WeldFrames(world_frame, plant.GetFrameByName("root", source_model))
        plant.WeldFrames(
            world_frame,
            plant.GetFrameByName("panda_link0", franka_model),
            X_FM=RigidTransform(self.robot_base_pose),
        )
        plant.Finalize()

        visualizer = MeshcatVisualizer.AddToBuilder(
            diagram_builder,
            scene_graph,
            meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration),
        )
        diagram = diagram_builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)
        visualizer_context = visualizer.GetMyContextFromRoot(context)

        source_position_names = np.asarray(
            plant.GetPositionNames(
                source_model,
                add_model_instance_prefix=False,
                always_add_suffix=False,
            ),
            dtype=object,
        )
        source_joint_name_to_index = {
            str(name): idx for idx, name in enumerate(source_position_names.tolist())
        }

        visualizer.StartRecording(False)
        meshcat.StartRecording()

        times = np.arange(self.n_frames, dtype=float) * float(time_step)
        for frame_idx, t in enumerate(times):
            q_source = np.zeros(plant.num_positions(source_model), dtype=float)
            for traj_col, joint_name in enumerate(self.active_joint_names.tolist()):
                if joint_name not in source_joint_name_to_index:
                    raise ValueError(
                        f"Joint {joint_name!r} is not present in the source xhand visualization plant."
                    )
                q_source[source_joint_name_to_index[joint_name]] = (
                    self.active_joint_values[frame_idx, traj_col]
                )

            q_franka = np.asarray(
                self._compose_full_franka_q(arm_trajectory[frame_idx], frame_idx),
                dtype=float,
            )

            context.SetTime(float(t))
            plant.SetPositions(plant_context, source_model, q_source)
            plant.SetPositions(plant_context, franka_model, q_franka)
            visualizer.ForcedPublish(visualizer_context)

        visualizer.StopRecording()
        visualizer.PublishRecording()
        meshcat.PublishRecording()
        visualizer.ForcedPublish(visualizer_context)

        if pause:
            input("Press Enter to continue...")
        return meshcat

    def export_info(self, export_path: str | Path | None = None) -> dict[str, Any]:
        if self.arm_trajectory is None or self.result is None:
            raise RuntimeError("Call solve() before export_info().")

        export_payload = dict(self._traj_payload)
        export_payload["traj_robot_base_pose_original_world"] = np.asarray(
            self.initial_robot_base_pose_world,
            dtype=np.float32,
        )
        export_payload["traj_hand_name"] = np.array("franka_xhand", dtype=object)
        export_payload["traj_robot_model"] = np.array("franka_xhand", dtype=object)
        export_payload["traj_hand_urdf_path"] = np.array(
            str(Path(self.franka_xhand_urdf)), dtype=object
        )
        export_payload["traj_robot_base_pose"] = np.asarray(
            self.robot_base_pose,
            dtype=np.float32,
        )
        export_payload["traj_object_raised_by"] = np.asarray(
            [self.cfg.object_raised_by],
            dtype=np.float32,
        )
        export_payload["traj_exported_in_robot_base_frame"] = np.asarray(
            [False],
            dtype=bool,
        )

        export_payload["franka_joint_names"] = np.asarray(
            self.arm_joint_names, dtype=object
        )
        export_payload["franka_joint_vals"] = self.arm_trajectory.astype(np.float32)
        export_payload["reference_wrist_poses"] = self.reference_wrist_poses.astype(
            np.float32
        )
        quality = self._evaluate_trajectory_quality(
            self.arm_trajectory.T,
            max_position_error_m=(
                self.cfg.stage2_max_position_error_m
                if self.cfg.enable_two_stage_solve
                else self.cfg.max_position_error_m
            ),
            max_rotation_error_fro=(
                self.cfg.stage2_max_rotation_error_fro
                if self.cfg.enable_two_stage_solve
                else self.cfg.max_rotation_error_fro
            ),
            max_joint_step=(
                self.cfg.stage2_max_joint_step
                if self.cfg.enable_two_stage_solve
                else self.cfg.max_joint_step
            ),
        )
        export_payload["franka_substitution_success"] = np.array(
            [bool(quality["within_constraints"])],
            dtype=bool,
        )
        export_payload["franka_substitution_solver_success"] = np.array(
            [bool(self.result.is_success())],
            dtype=bool,
        )
        export_payload["franka_substitution_max_position_error_m"] = np.array(
            [quality["max_position_error_m"]],
            dtype=np.float32,
        )
        export_payload["franka_substitution_max_rotation_error_fro"] = np.array(
            [quality["max_rotation_error_fro"]],
            dtype=np.float32,
        )
        export_payload["franka_substitution_max_position_violation_sq"] = np.array(
            [quality["max_position_violation_sq"]],
            dtype=np.float32,
        )
        export_payload["franka_substitution_max_rotation_violation_sq"] = np.array(
            [quality["max_rotation_violation_sq"]],
            dtype=np.float32,
        )
        export_payload["franka_substitution_max_joint_step_violation"] = np.array(
            [quality["max_joint_step_violation"]],
            dtype=np.float32,
        )

        if self.cfg.export_in_robot_base_frame:
            export_payload = self._export_payload_in_robot_base_frame(export_payload)

        active_joint_names = np.concatenate(
            [self.arm_joint_names, self.finger_joint_names]
        ).astype(object)
        active_joint_vals = np.concatenate(
            [self.arm_trajectory, self.finger_trajectory],
            axis=1,
        ).astype(np.float32)
        export_payload[f"{self.active_side}_joint_names"] = active_joint_names
        export_payload[f"{self.active_side}_joint_vals"] = active_joint_vals

        self.export_payload = export_payload
        if export_path is not None:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(export_path, **export_payload)
        return export_payload
