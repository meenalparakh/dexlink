from __future__ import annotations

import importlib.util
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

import numpy as np
import trimesh
from pydrake.all import (
    DistanceConstraint,
    MathematicalProgram,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MinimumDistanceLowerBoundConstraint,
    RigidTransform,
    RobotDiagramBuilder,
    Role,
    SnoptSolver,
    Solve,
    StartMeshcat,
)
from pydrake.geometry import Rgba, Sphere
from pydrake.solvers import SolverOptions
from urdfpy import URDF

from dexlink.src.retargeting.optimization.utils_final import (
    CostTraceLogger,
    compute_hand_cfg_vecs,
    compute_object_bps_differentiable,
    sample_pts_from_obj,
)
from dexlink.utils.retargeting.path_utils import process_urdf, resolve_project_path

if TYPE_CHECKING:
    from dexlink.src.retargeting.config.trajopt_final import TrajoptCfg


def _npz_scalar(value):
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.shape == (1,):
        return array[0].item()
    raise ValueError(f"Expected scalar metadata, got shape {array.shape}.")


def load_q_solution_from_npz(
    npz_path: str | Path,
    dof_names: list[str],
    *,
    hand_side: str | None = None,
    frame_idx: int = -1,
) -> np.ndarray:
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    expected_dof_names = list(dof_names)

    if "q_solution" in data.files and "dof_names" in data.files:
        source_q = np.asarray(data["q_solution"], dtype=float).reshape(-1)
        source_dof_names = [str(name) for name in data["dof_names"].tolist()]
    elif {
        "left_joint_names",
        "left_joint_vals",
        "right_joint_names",
        "right_joint_vals",
    }.issubset(data.files):
        if hand_side is None:
            if "traj_hand_side" not in data.files:
                raise ValueError(
                    f"{npz_path} is sequence_motion format but has no "
                    "'traj_hand_side'. Pass hand_side explicitly."
                )
            hand_side = str(_npz_scalar(data["traj_hand_side"]))
        if hand_side not in {"left", "right"}:
            raise ValueError(
                f"Expected hand_side 'left' or 'right', got {hand_side!r}."
            )
        source_dof_names = [
            str(name) for name in data[f"{hand_side}_joint_names"].tolist()
        ]
        source_q_traj = np.asarray(data[f"{hand_side}_joint_vals"], dtype=float)
        if source_q_traj.ndim != 2:
            raise ValueError(
                f"Expected {hand_side}_joint_vals to have shape (T, nq), "
                f"got {source_q_traj.shape}."
            )
        source_q = source_q_traj[int(frame_idx)].reshape(-1)
    else:
        raise ValueError(
            f"Expected {npz_path} to contain either q_solution+dof_names or "
            "sequence_motion fields left/right_joint_names and left/right_joint_vals."
        )

    if len(source_dof_names) != source_q.shape[0]:
        raise ValueError(
            f"Expected len(source_dof_names)==len(source_q), got "
            f"{len(source_dof_names)} and {source_q.shape[0]}."
        )
    if set(source_dof_names) != set(expected_dof_names):
        missing = [name for name in expected_dof_names if name not in source_dof_names]
        extra = [name for name in source_dof_names if name not in expected_dof_names]
        raise ValueError(
            f"Initial solution dof names do not match optimizer dofs. "
            f"Missing: {missing}. Extra: {extra}."
        )

    source_name_to_idx = {name: idx for idx, name in enumerate(source_dof_names)}
    return np.asarray(
        [source_q[source_name_to_idx[name]] for name in expected_dof_names],
        dtype=float,
    )


class DrakeOptimizer:
    def __init__(
        self,
        cfg: TrajoptCfg,
        reference_obj_path,
        target_obj_path,
        robot_urdf_path,
        robot_cfg,  # an example: dexlink.src.rl.sim.hand_cfgs.inspire:INSPIRE_LEFT_CFG
        ori_pose,
        tgt_pose,
        hand_jpos_dict,
        optimization_run_dir,
        hand_bps_jpos_dict=None,
        target_bps_obj_path=None,
    ):
        self.reference_obj_path = resolve_project_path(reference_obj_path)
        if self.reference_obj_path.suffix.lower() not in {".obj", ".ply"}:
            raise ValueError(
                "Expected reference_obj_path to be an OBJ or PLY mesh, got "
                f"{self.reference_obj_path}."
            )
        if not self.reference_obj_path.is_file():
            raise FileNotFoundError(
                f"Reference object mesh does not exist: {self.reference_obj_path}."
            )
        self.target_obj_path = target_obj_path
        self.target_bps_obj_path = (
            self.target_obj_path if target_bps_obj_path is None else target_bps_obj_path
        )
        self.robot_urdf_path = process_urdf(robot_urdf_path)
        self.robot_cfg = robot_cfg
        self.ori_pose = ori_pose  # (4, 4)
        self.tgt_pose = tgt_pose  # (4, 4)
        self.hand_jpos_dict = self._validate_jpos_dict(
            hand_jpos_dict,
            "hand_jpos_dict",
        )
        if hand_bps_jpos_dict is None:
            if "human_robot_kpt_map" in robot_cfg:
                missing = [
                    name
                    for name in robot_cfg["human_robot_kpt_map"]
                    if name not in self.hand_jpos_dict
                ]
                if missing:
                    raise ValueError(
                        "robot_cfg defines human_robot_kpt_map with keypoints that "
                        "are absent from hand_jpos_dict. Pass hand_bps_jpos_dict "
                        f"for BPS computation. Missing keys: {missing}."
                    )
                hand_bps_jpos_dict = {
                    name: self.hand_jpos_dict[name]
                    for name in robot_cfg["human_robot_kpt_map"]
                }
            else:
                hand_bps_jpos_dict = self.hand_jpos_dict
        self.hand_bps_jpos_dict = self._validate_jpos_dict(
            hand_bps_jpos_dict,
            "hand_bps_jpos_dict",
        )
        self.hand_bps_jpos_names = tuple(self.hand_bps_jpos_dict.keys())
        self.meshcat = None
        self.cfg = cfg

        self.vis_server = None
        self._robot_urdfpy = None
        self._target_obj_urdfpy = None
        self._target_contact_mesh = None
        self.run_dir = Path(optimization_run_dir)
        self.logger = CostTraceLogger(
            enabled=self.cfg.log_cost_terms,
            plot=self.cfg.plot_cost_terms,
            every=self.cfg.cost_log_every,
        )
        self._last_reference_bps_beta = None

        self.build_plant()
        self.initialize_prog()
        self.construct_gt()

    @staticmethod
    def _is_drake_invalid_mesh_inertia_error(exc: RuntimeError) -> bool:
        message = str(exc)
        return (
            "CalcSpatialInertia()" in message
            and "calculated volume of a triangle surface mesh" in message
            and "1e-14" in message
        ) or "Spatial inertia fails SpatialInertia::IsPhysicallyValid()" in message

    @staticmethod
    def _load_make_connected_solid_mesh():
        script_path = (
            Path(__file__).resolve().parents[2]
            / "scripts/trajopt/make_connected_watertight_mesh.py"
        )
        if not script_path.exists():
            raise FileNotFoundError(
                f"Solid mesh repair script does not exist: {script_path}"
            )
        spec = importlib.util.spec_from_file_location(
            "dexlink_make_connected_watertight_mesh",
            script_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load solid mesh repair script: {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "make_connected_solid_mesh"):
            raise AttributeError(
                f"{script_path} does not define `make_connected_solid_mesh`."
            )
        return module.make_connected_solid_mesh

    def _repair_target_obj_for_drake_inertia(self) -> str:
        input_path = Path(self.target_obj_path)
        if not input_path.exists():
            raise FileNotFoundError(
                f"Cannot repair target mesh because it does not exist: {input_path}"
            )
        output_path = input_path.with_name(f"{input_path.stem}_solid_watertight.obj")
        print(
            "Drake rejected target mesh due to invalid spatial inertia. "
            f"Building connected solid mesh: {output_path}"
        )
        make_connected_solid_mesh = self._load_make_connected_solid_mesh()
        repaired_mesh = make_connected_solid_mesh(
            input_path=input_path,
            output_path=output_path,
            pitch_divisions=300,
            auto_coarsen=True,
            export=True,
        )
        repaired_volume = abs(float(repaired_mesh.volume))
        if repaired_volume < 1e-14:
            raise ValueError(
                "Solid mesh repair produced a near-zero-volume mesh: "
                f"{output_path}, volume={repaired_volume}."
            )
        old_target_obj_path = self.target_obj_path
        self.target_obj_path = str(output_path)
        if self.target_bps_obj_path == old_target_obj_path:
            self.target_bps_obj_path = self.target_obj_path
        print(f"Retrying Drake target load with repaired mesh: {self.target_obj_path}")
        return self.target_obj_path

    @staticmethod
    def _validate_jpos_dict(jpos_dict, arg_name: str) -> dict[str, np.ndarray]:
        if not isinstance(jpos_dict, dict) or len(jpos_dict) == 0:
            raise ValueError(f"Expected {arg_name} to be a non-empty dict.")
        validated = {}
        for name, value in jpos_dict.items():
            point = np.asarray(value)
            if point.shape != (3,):
                raise ValueError(
                    f"Expected {arg_name}[{name!r}] to have shape (3,), "
                    f"got {point.shape}."
                )
            validated[str(name)] = point
        return validated

    def _reference_hand_bps_points(self) -> np.ndarray:
        return np.stack(
            [self.hand_bps_jpos_dict[name] for name in self.hand_bps_jpos_names],
            axis=0,
        )

    def _computed_hand_bps_points(
        self,
        jpos_dict: dict[str, np.ndarray],
    ) -> np.ndarray:
        missing_names = [
            name for name in self.hand_bps_jpos_names if name not in jpos_dict
        ]
        if missing_names:
            raise ValueError(
                "Robot joint position dict is missing BPS keypoints: "
                f"{missing_names}. Check robot_cfg['human_robot_kpt_map'] and the URDF "
                "dummy links used for trajopt."
            )
        return np.stack([jpos_dict[name] for name in self.hand_bps_jpos_names], axis=0)

    @staticmethod
    def _ordered_keypoint_array(
        keypoint_dict: dict[str, np.ndarray],
        names: tuple[str, ...],
        *,
        dict_name: str,
    ) -> np.ndarray:
        missing_names = [name for name in names if name not in keypoint_dict]
        if missing_names:
            raise ValueError(f"{dict_name} is missing keypoints: {missing_names}.")
        return np.stack(
            [np.asarray(keypoint_dict[name], dtype=np.float32) for name in names],
            axis=0,
        )

    def reference_keypoint_payload(self) -> dict[str, np.ndarray]:
        skeleton_names = tuple(self.robot_cfg["human_robot_link_map"].keys())
        bps_names = tuple(self.hand_bps_jpos_names)
        return {
            "human_skeleton_keypoint_names": np.asarray(skeleton_names, dtype=object),
            "human_bps_keypoint_names": np.asarray(bps_names, dtype=object),
            "human_skeleton_keypoint_pos": self._ordered_keypoint_array(
                self.hand_jpos_dict,
                skeleton_names,
                dict_name="hand_jpos_dict",
            ),
            "human_bps_keypoint_pos": self._ordered_keypoint_array(
                self.hand_bps_jpos_dict,
                bps_names,
                dict_name="hand_bps_jpos_dict",
            ),
        }

    # def robot_keypoint_array(
    #     self,
    #     q_var: np.ndarray,
    #     keypoint_names: tuple[str, ...],
    # ) -> np.ndarray:
    #     jpos_dict = self.compute_robot_joint_pos(q_var)
    #     return self._ordered_keypoint_array(
    #         jpos_dict,
    #         keypoint_names,
    #         dict_name="robot FK keypoint dict",
    #     )

    @staticmethod
    def _solution_vector_to_float(q_var: np.ndarray) -> np.ndarray:
        values = []
        for value in np.asarray(q_var, dtype=object).reshape(-1):
            if hasattr(value, "value"):
                value = value.value()
            values.append(float(value))
        return np.asarray(values, dtype=np.float32)

    def robot_keypoint_array(
        self,
        q_var: np.ndarray,
        keypoint_names: tuple[str, ...],
    ) -> np.ndarray:
        q_solution = self._solution_vector_to_float(q_var)
        if q_solution.shape[0] != len(self.dof_names):
            raise ValueError(
                f"Expected q_solution length {len(self.dof_names)}, "
                f"got {q_solution.shape[0]}."
            )

        if self._robot_urdfpy is None:
            self._robot_urdfpy = URDF.load(str(process_urdf(self.robot_urdf_path)))

        frame_map = dict(self.robot_cfg["human_robot_link_map"])
        frame_map.update(self.robot_cfg.get("human_robot_kpt_map", {}))

        missing_mappings = [name for name in keypoint_names if name not in frame_map]
        if missing_mappings:
            raise ValueError(
                f"robot_cfg is missing FK frame mappings for keypoints: {missing_mappings}."
            )

        joint_cfg = {
            name: float(value)
            for name, value in zip(self.dof_names, q_solution.tolist())
        }
        link_fk = self._robot_urdfpy.link_fk(cfg=joint_cfg)
        link_fk_by_name = {
            link.name: np.asarray(transform, dtype=np.float32)
            for link, transform in link_fk.items()
        }

        keypoints = {}
        missing_links = []
        for human_name in keypoint_names:
            link_name = frame_map[human_name]
            if link_name not in link_fk_by_name:
                missing_links.append(link_name)
                continue
            keypoints[human_name] = link_fk_by_name[link_name][:3, 3]

        if missing_links:
            raise ValueError(
                "URDF FK did not return robot keypoint links required for export: "
                f"{missing_links}."
            )

        return self._ordered_keypoint_array(
            keypoints,
            keypoint_names,
            dict_name="robot FK keypoint dict",
        )

    def _target_mesh_for_contact_refinement(self) -> trimesh.Trimesh:
        if self._target_contact_mesh is not None:
            return self._target_contact_mesh

        mesh_or_scene = trimesh.load(self.target_obj_path, force="scene")
        if isinstance(mesh_or_scene, trimesh.Scene):
            if len(mesh_or_scene.geometry) == 0:
                raise ValueError(
                    f"No geometry found in target mesh file: {self.target_obj_path}"
                )
            mesh = mesh_or_scene.dump(concatenate=True)
        else:
            mesh = mesh_or_scene
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected a Trimesh, got {type(mesh).__name__}.")
        if mesh.faces is None or len(mesh.faces) == 0:
            raise ValueError(f"Target mesh has no faces: {self.target_obj_path}")
        mesh.apply_transform(self.tgt_pose)
        self._target_contact_mesh = mesh
        return mesh

    def _robot_frame_name_for_human_keypoint(self, human_keypoint_name: str) -> str:
        human_robot_kpt_map = self.robot_cfg.get("human_robot_kpt_map")
        if human_robot_kpt_map is None:
            human_robot_kpt_map = self.robot_cfg["human_robot_link_map"]
        if human_keypoint_name in human_robot_kpt_map:
            return human_robot_kpt_map[human_keypoint_name]
        if human_keypoint_name in self.robot_cfg["human_robot_link_map"]:
            return self.robot_cfg["human_robot_link_map"][human_keypoint_name]
        raise KeyError(
            f"No robot frame mapping for human keypoint {human_keypoint_name!r}."
        )

    def _compute_robot_keypoint_frames(
        self,
        q_var,
        human_keypoint_names: tuple[str, ...],
    ):
        plant, context = self.get_plant_context(q_var)
        plant.SetPositions(context, q_var)
        world_frame = plant.GetFrameByName(
            self.W_frame.name(),
            self.W_frame.model_instance(),
        )
        transforms = {}
        for human_keypoint_name in human_keypoint_names:
            robot_frame_name = self._robot_frame_name_for_human_keypoint(
                human_keypoint_name
            )
            try:
                keypoint_frame = plant.GetFrameByName(robot_frame_name, self.hand_model)
            except RuntimeError as exc:
                raise ValueError(
                    "robot_cfg maps human keypoint "
                    f"{human_keypoint_name!r} to robot frame {robot_frame_name!r}, "
                    f"but that frame is missing from {self.robot_urdf_path}."
                ) from exc
            transforms[human_keypoint_name] = plant.CalcRelativeTransform(
                context,
                world_frame,
                keypoint_frame,
            )
        return transforms

    @staticmethod
    def _finger_aliases(finger_name: str) -> tuple[str, ...]:
        aliases = {
            "wrist": ("wrist", "hand_link"),
            "thumb": ("thumb",),
            "index": ("index",),
            "mid": ("mid", "middle"),
            "ring": ("ring",),
            "pinky": ("pinky", "little"),
        }
        if finger_name not in aliases:
            raise ValueError(
                f"Unknown finger name {finger_name!r}. Expected one of "
                f"{list(aliases)}."
            )
        return aliases[finger_name]

    def _build_finger_hand_obj_geom_pairs(
        self,
    ) -> dict[str, list[tuple[object, object]]]:
        finger_joint_maps = self.robot_cfg.get("finger_joint_maps")
        if finger_joint_maps is None:
            return {}

        human_robot_link_map = self.robot_cfg["human_robot_link_map"]
        human_robot_kpt_map = self.robot_cfg.get("human_robot_kpt_map", {})
        finger_geom_pairs = {}
        for finger_name in finger_joint_maps:
            aliases = self._finger_aliases(finger_name)
            mapped_link_names = {
                robot_link_name
                for human_name, robot_link_name in {
                    **human_robot_link_map,
                    **human_robot_kpt_map,
                }.items()
                if any(alias in human_name for alias in aliases)
            }
            matching_hand_geom_ids = [
                geom_id
                for geom_id, body_name in self.hand_geom_id_to_body_name.items()
                if body_name in mapped_link_names
                or any(alias in body_name for alias in aliases)
            ]
            finger_geom_pairs[finger_name] = [
                (hand_geom_id, obj_geom_id)
                for hand_geom_id in matching_hand_geom_ids
                for obj_geom_id in self.obj_geom_ids
            ]
        return finger_geom_pairs

    def get_meshcat(self):
        if self.meshcat is None:
            self.meshcat = StartMeshcat()
        return self.meshcat

    def build_plant(self):
        builder = RobotDiagramBuilder(time_step=0.001)
        plant_f = builder.plant()
        parser = builder.parser()
        builder.scene_graph()

        print(f"Loading robot URDF from: {self.robot_urdf_path}")
        print(f"Loading target object URDF from: {self.target_obj_path}")

        (hand_model,) = parser.AddModels(
            process_urdf(self.robot_urdf_path, append_jobname=True)
        )
        try:
            (object_model,) = parser.AddModels(self.target_obj_path)
        except RuntimeError as exc:
            if not self._is_drake_invalid_mesh_inertia_error(exc):
                raise
            repaired_target_obj_path = self._repair_target_obj_for_drake_inertia()
            (object_model,) = parser.AddModels(repaired_target_obj_path)

        # weld robot to the world
        W = plant_f.world_frame()
        plant_f.WeldFrames(W, plant_f.GetFrameByName("root", hand_model))

        # Weld using the model body's frame (not arbitrary frame index ordering).
        asset_body_idx = plant_f.GetBodyIndices(object_model)[0]
        print(f"Asset body index: {asset_body_idx}")
        asset_frame = plant_f.get_body(asset_body_idx).body_frame()
        X_WA = RigidTransform(self.tgt_pose)
        plant_f.WeldFrames(W, asset_frame, X_WA)

        diagram = builder.Build()
        context_f = diagram.CreateDefaultContext()
        plant_context_f = plant_f.GetMyContextFromRoot(context_f)

        plant_ad = plant_f.ToAutoDiffXd()
        context_ad = plant_ad.CreateDefaultContext()

        self.plant_f = plant_f
        self.diagram_f = diagram
        self.context_f = context_f
        self.plant_context_f = plant_context_f
        self.plant_ad = plant_ad
        self.context_ad = context_ad
        self.W_frame = W
        self.hand_model = hand_model
        self.object_model = object_model

        self.dof_names = plant_f.GetPositionNames(
            add_model_instance_prefix=False, always_add_suffix=False
        )

        self.hand_body_ids = self.plant_f.GetBodyIndices(self.hand_model)
        self.object_body_ids = self.plant_f.GetBodyIndices(self.object_model)

        print(f"Hand body ids: {self.hand_body_ids}")
        print(f"Object body ids: {self.object_body_ids}")

        self.hand_geom_ids = []
        self.hand_geom_id_to_body_name = {}
        self.hand_body_name_to_geom_ids = {}
        for body_id in self.hand_body_ids:
            body = self.plant_f.get_body(body_id)
            geom_id = self.plant_f.GetCollisionGeometriesForBody(body)
            self.hand_geom_ids.extend(geom_id)
            self.hand_body_name_to_geom_ids[body.name()] = list(geom_id)
            for hand_geom_id in geom_id:
                self.hand_geom_id_to_body_name[hand_geom_id] = body.name()

        print(f"Hand geom ids: {self.hand_geom_ids}")

        self.obj_geom_ids = []
        for body_id in self.object_body_ids:
            body = self.plant_f.get_body(body_id)
            geom_id = self.plant_f.GetCollisionGeometriesForBody(body)
            self.obj_geom_ids.extend(geom_id)
        print(f"Object geom ids: {self.obj_geom_ids}")
        self.hand_obj_geom_pairs = [
            (hand_geom_id, obj_geom_id)
            for hand_geom_id in self.hand_geom_ids
            for obj_geom_id in self.obj_geom_ids
        ]
        print(f"Num hand-object geometry pairs: {len(self.hand_obj_geom_pairs)}")
        self.finger_hand_obj_geom_pairs = self._build_finger_hand_obj_geom_pairs()
        for finger_name, geom_pairs in self.finger_hand_obj_geom_pairs.items():
            print(f"Num {finger_name} hand-object geometry pairs: {len(geom_pairs)}")

        print(f"Num dofs: {len(self.dof_names)}")
        print(self.dof_names)

    def plant_with_visualizer(
        self,
        q_robot,
        pcd_dict={},
        max_n_pts=200,
        marker_radius=0.005,
        pause=True,
        center_demo=True,
    ):
        meshcat = self.get_meshcat()
        meshcat.Delete()
        meshcat.DeleteAddedControls()

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

        # Optimized hand configuration.
        q_robot = np.asarray(q_robot).reshape(-1)

        assert plant.num_positions(hand_model) == len(
            q_robot
        ), f"Expected {plant.num_positions(hand_model)} joint positions, got {len(q_robot)}"
        plant.SetPositions(
            plant_context, hand_model, q_robot[: plant.num_positions(hand_model)]
        )

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

        visualizer.ForcedPublish(visualizer_context)
        if pause:
            input("Press Enter to continue...")
        return meshcat

    def construct_gt(self):
        self.ref_dict = compute_hand_cfg_vecs(self.hand_jpos_dict)
        self.reference_pcd = sample_pts_from_obj(
            self.reference_obj_path,
            self.ori_pose,
            self.cfg.num_pts_sample,
            return_nocs=self.cfg.bps_nocs_features,
        )
        self.target_pcd = sample_pts_from_obj(
            self.target_bps_obj_path,
            self.tgt_pose,
            self.cfg.num_pts_sample,
            return_nocs=self.cfg.bps_nocs_features,
        )

        self._refresh_reference_hand_bps(force=True)

    def _refresh_reference_hand_bps(self, force=False):
        if (not force) and self._last_reference_bps_beta == self.cfg.bps_beta:
            return
        hand_jpos_array = self._reference_hand_bps_points()

        self.reference_hand_rel_bps = compute_object_bps_differentiable(
            self.reference_pcd,
            hand_jpos_array,
            beta=self.cfg.bps_beta,
            normalize=False,
            include_other_features=self.cfg.bps_nocs_features,
            apply_penetration_correction=False,
        )
        self._last_reference_bps_beta = float(self.cfg.bps_beta)
        print(f"Number of basis points used: {len(self.reference_hand_rel_bps)}")

    def update_beta(self, eval_count):
        if self.cfg.bps_beta_range is None:
            return
        start_beta, end_beta = self.cfg.bps_beta_range
        if start_beta <= 0 or end_beta <= 0:
            raise ValueError(
                "bps_beta_range must contain positive values for log-space annealing."
            )
        anneal_steps = max(1, int(self.cfg.bps_beta_anneal_steps))
        t = min(max(eval_count - 1, 0) / anneal_steps, 1.0)
        log_beta = (1.0 - t) * np.log(start_beta) + t * np.log(end_beta)
        self.cfg.bps_beta = float(np.exp(log_beta))
        if eval_count == 1 or eval_count % self.cfg.bps_beta_update_every == 0:
            self._refresh_reference_hand_bps()

    def get_plant_context(self, q_var):
        if q_var.dtype == float:
            return self.plant_f, self.plant_context_f
        return self.plant_ad, self.context_ad

    def compute_robot_joint_pos(self, q_var):
        plant, context = self.get_plant_context(q_var)
        plant.SetPositions(context, q_var)

        # Compute hand-cfg link keypoints and the denser hand-object BPS keypoints.
        human_robot_link_map = self.robot_cfg["human_robot_link_map"]
        human_robot_kpt_map = self.robot_cfg.get("human_robot_kpt_map")
        if human_robot_kpt_map is None:
            human_robot_kpt_map = human_robot_link_map
        frame_map = {**human_robot_link_map, **human_robot_kpt_map}
        jpos_dict = {}

        world_frame = plant.GetFrameByName(
            self.W_frame.name(),
            self.W_frame.model_instance(),
        )

        for human_link_name, robot_link_name in frame_map.items():
            try:
                joint_frame = plant.GetFrameByName(robot_link_name, self.hand_model)
            except RuntimeError as exc:
                raise ValueError(
                    "robot_cfg maps human keypoint "
                    f"{human_link_name!r} to robot frame {robot_link_name!r}, but "
                    f"that frame is missing from {self.robot_urdf_path}."
                ) from exc
            jpos_rel_world = plant.CalcRelativeTransform(
                context, world_frame, joint_frame
            ).translation()

            jpos_dict[human_link_name] = jpos_rel_world
        return jpos_dict

    @staticmethod
    def _hand_keypoint_entity(name: str) -> str:
        if name == "wrist" or "palm" in name:
            return "palm"
        if name.startswith("thumb"):
            return "thumb"
        if name.startswith("index"):
            return "index"
        if name.startswith("middle"):
            return "mid"
        if name.startswith("ring"):
            return "ring"
        if name.startswith("little"):
            return "pinky"
        raise ValueError(f"Cannot assign hand keypoint {name!r} to an entity.")

    @staticmethod
    def _stage_entity(stage_name: str) -> str:
        if stage_name == "wrist":
            return "palm"
        if stage_name in {"thumb", "index", "mid", "ring", "pinky"}:
            return stage_name
        raise ValueError(f"Cannot assign refinement stage {stage_name!r} to an entity.")

    def _stage_included_entities(self, stage_name: str) -> tuple[str, ...]:
        included_entities = []
        for current_stage_name in self.cfg.sequential_finger_refinement_order:
            entity = self._stage_entity(current_stage_name)
            if entity not in included_entities:
                included_entities.append(entity)
            if current_stage_name == stage_name:
                return tuple(included_entities)
        raise ValueError(
            f"Refinement stage {stage_name!r} is not in "
            "cfg.sequential_finger_refinement_order."
        )

    def _hand_cfg_pair_in_entities(
        self,
        pair: tuple[str, str, str],
        included_entities: tuple[str, ...],
    ) -> bool:
        joint1, joint2, _ = pair
        return (
            self._hand_keypoint_entity(joint1) in included_entities
            and self._hand_keypoint_entity(joint2) in included_entities
        )

    def _hand_cfg_triplet_in_entities(
        self,
        triplet: tuple[str, str, str, str],
        included_entities: tuple[str, ...],
    ) -> bool:
        joint1, joint2, joint3, _ = triplet
        return (
            self._hand_keypoint_entity(joint1) in included_entities
            and self._hand_keypoint_entity(joint2) in included_entities
            and self._hand_keypoint_entity(joint3) in included_entities
        )

    def hand_cfg_cost(
        self,
        computed_dict,
        rotational_params=None,
        included_entities: tuple[str, ...] | None = None,
    ):

        pair_costs = []
        for pair_idx, (_, _, mode) in enumerate(self.ref_dict["pairs"][0]):
            pair = self.ref_dict["pairs"][0][pair_idx]
            if included_entities is not None and not self._hand_cfg_pair_in_entities(
                pair,
                included_entities,
            ):
                continue
            ref_vec = self.ref_dict["pairs"][1][pair_idx]

            if rotational_params is not None:
                ref_vec = (rotational_params @ ref_vec.reshape(3, 1)).reshape(3)

            comp_vec = computed_dict["pairs"][1][pair_idx]
            if mode == "normalize":
                cost = 2 * (1 - np.dot(ref_vec, comp_vec))
            else:
                diff = ref_vec - comp_vec
                cost = np.dot(diff, diff)
            pair_costs.append(cost)

        triplet_costs = []
        for tripled_idx, (_, _, _, mode) in enumerate(self.ref_dict["triplets"][0]):
            triplet = self.ref_dict["triplets"][0][tripled_idx]
            if (
                included_entities is not None
                and not self._hand_cfg_triplet_in_entities(
                    triplet,
                    included_entities,
                )
            ):
                continue
            ref_vec = self.ref_dict["triplets"][1][tripled_idx]
            comp_vec = computed_dict["triplets"][1][tripled_idx]
            if rotational_params is not None:
                ref_vec = (rotational_params @ ref_vec.reshape(3, 1)).reshape(3)
            if mode == "normalize":
                cost = 2 * (1 - np.dot(ref_vec, comp_vec))
            else:
                diff = ref_vec - comp_vec
                cost = np.dot(diff, diff)
            triplet_costs.append(cost)

        total_cost = 0.0
        if pair_costs:
            total_cost = total_cost + np.mean(pair_costs)
        if triplet_costs:
            total_cost = total_cost + np.mean(triplet_costs)
        return total_cost

    def hand_object_bps_cost(
        self,
        computed_hand_rel_bps,
        rotational_params=None,
        reference_hand_rel_bps=None,
    ):
        if reference_hand_rel_bps is None:
            reference_hand_rel_bps = self.reference_hand_rel_bps
        comp_displacement = computed_hand_rel_bps[..., :3]
        comp_normal = computed_hand_rel_bps[..., 3:6]
        comp_features = computed_hand_rel_bps[..., 6:]

        ref_displacement = reference_hand_rel_bps[..., :3]
        ref_normal = reference_hand_rel_bps[..., 3:6]
        ref_features = reference_hand_rel_bps[..., 6:]
        if computed_hand_rel_bps.shape != reference_hand_rel_bps.shape:
            raise ValueError(
                "Expected computed and reference hand-object BPS to have the same "
                f"shape, got {computed_hand_rel_bps.shape} and "
                f"{reference_hand_rel_bps.shape}."
            )

        comp_distance_sq = np.sum(comp_displacement**2, axis=-1) + 1e-4
        ref_distance_sq = np.sum(ref_displacement**2, axis=-1) + 1e-4

        comp_direction = comp_displacement / np.sqrt(comp_distance_sq)[..., None]
        ref_direction = ref_displacement / np.sqrt(ref_distance_sq)[..., None]

        if rotational_params is not None:
            # convert the ref_direction and ref_normal to the rotated frame
            ref_direction = (rotational_params @ ref_direction.reshape(-1, 3).T).T
            ref_normal = (rotational_params @ ref_normal.reshape(-1, 3).T).T

        distance_cost = np.abs(1 - comp_distance_sq / ref_distance_sq).sum()
        direction_cost = 1 - (comp_direction * ref_direction).sum(axis=-1)
        direction_cost = direction_cost.sum()

        nrm_diff = 1 - (comp_normal * ref_normal).sum(axis=-1)
        normal_cost = nrm_diff * np.exp(-ref_distance_sq / (2 * (0.05**2)))
        normal_cost = normal_cost.sum()

        penetration_dot = np.sum(comp_direction * comp_normal, axis=-1)
        penetration_cost = np.where(
            penetration_dot > 0.0,
            (penetration_dot + 1.0) / (np.sqrt(ref_distance_sq) + 0.01),
            0.0,
        ).sum()

        feature_cost = np.abs(comp_features - ref_features).sum()

        total_cost = np.exp(
            # self.cfg.bps_beta
            self.cfg.bps_exp_coeff
            * (
                distance_cost * self.cfg.distance_weight
                + direction_cost
                + normal_cost
                # + penetration_cost
                + feature_cost
            )
        )

        # total_cost = distance_cost * self.cfg.distance_weight + direction_cost + normal_cost + feature_cost
        logs = {
            "distance_cost": distance_cost,
            "direction_cost": direction_cost,
            "normal_cost": normal_cost,
            "penetration_cost": penetration_cost,
            "feature_cost": feature_cost,
        }

        return total_cost, logs

    def initialize_prog(self):
        num_positions = self.plant_f.num_positions()

        prog = MathematicalProgram()
        q = prog.NewContinuousVariables(num_positions, "q")

        lower_limit = self.plant_f.GetPositionLowerLimits()
        _lower_limit = np.ones(num_positions) * (-2 * np.pi)
        lower_limit = np.maximum(lower_limit, _lower_limit)
        upper_limit = self.plant_f.GetPositionUpperLimits()
        _upper_limit = np.ones(num_positions) * (2 * np.pi)
        upper_limit = np.minimum(upper_limit, _upper_limit)
        prog.AddBoundingBoxConstraint(
            lower_limit,
            upper_limit,
            q,
        )

        self.prog = prog
        self.q = q

        if self.cfg.rotational_freedom:
            # angle axis format for rotating the bps and hand cfg direction vectors
            self.rotation_var = prog.NewContinuousVariables(3, "rotation_var")
            prog.AddBoundingBoxConstraint(
                -np.pi * np.ones(3), np.pi * np.ones(3), self.rotation_var
            )

    def angle_axis_to_rotation_matrix(self, angle_axis):
        angle_sq = np.dot(angle_axis, angle_axis)
        angle_sq_value = (
            angle_sq.value() if hasattr(angle_sq, "value") else float(angle_sq)
        )
        K = np.array(
            [
                [0, -angle_axis[2], angle_axis[1]],
                [angle_axis[2], 0, -angle_axis[0]],
                [-angle_axis[1], angle_axis[0], 0],
            ]
        )
        if angle_sq_value < 1e-8:
            sin_over_angle = 1.0 - angle_sq / 6.0 + (angle_sq * angle_sq) / 120.0
            one_minus_cos_over_angle_sq = (
                0.5 - angle_sq / 24.0 + (angle_sq * angle_sq) / 720.0
            )
        else:
            angle = np.sqrt(angle_sq)
            sin_over_angle = np.sin(angle) / angle
            one_minus_cos_over_angle_sq = (1.0 - np.cos(angle)) / angle_sq
        R = np.eye(3) + sin_over_angle * K + one_minus_cos_over_angle_sq * (K @ K)
        return R

    def get_root_position_indices(self):
        root_names = ["tx", "ty", "tz", "roll", "pitch", "yaw"]
        root_names = [f"forearm_{name}" for name in root_names]
        root_indices = []

        for idx, dof_name in enumerate(self.dof_names):
            if any([root_name in dof_name for root_name in root_names]):
                root_indices.append(idx)

        return np.asarray(root_indices, dtype=int)

    def get_finger_base_indices(self):
        # get the human_robot_link_map values that contain "base"
        human_robot_link_map = self.robot_cfg["human_robot_link_map"]
        base_link_names = [
            v for k, v in human_robot_link_map.items() if "base" in k.lower()
        ]
        robot_urdf_inst = URDF.load(self.robot_urdf_path)
        # get the joint names corresponding to the base_link_names
        joints = [j for j in robot_urdf_inst.joints if j.joint_type != "fixed"]
        link_to_joint = {j.child: j.name for j in joints}
        base_joint_names = [link_to_joint[link_name] for link_name in base_link_names]

        base_indices = [
            self.dof_names.index(joint_name) for joint_name in base_joint_names
        ]
        return np.asarray(base_indices, dtype=int)

    def get_finger_joint_indices(self, finger_name: str) -> np.ndarray:
        finger_joint_maps = self.robot_cfg.get("finger_joint_maps")
        if finger_joint_maps is None:
            raise ValueError(
                "robot_cfg must define 'finger_joint_maps' to use "
                "sequential_finger_refinement."
            )
        if finger_name not in finger_joint_maps:
            raise ValueError(
                f"robot_cfg['finger_joint_maps'] has no entry for {finger_name!r}. "
                f"Available fingers: {list(finger_joint_maps)}."
            )

        joint_names = finger_joint_maps[finger_name]
        missing_joint_names = [
            joint_name for joint_name in joint_names if joint_name not in self.dof_names
        ]
        if missing_joint_names:
            raise ValueError(
                f"Finger {finger_name!r} maps to joints absent from Drake dofs: "
                f"{missing_joint_names}. Available dofs: {self.dof_names}."
            )
        return np.asarray(
            [self.dof_names.index(joint_name) for joint_name in joint_names],
            dtype=int,
        )

    def _robot_urdf(self):
        if self._robot_urdfpy is None:
            self._robot_urdfpy = URDF.load(self.robot_urdf_path)
        return self._robot_urdfpy

    def get_finger_joint_child_link_names(self, finger_name: str) -> tuple[str, ...]:
        if finger_name == "wrist":
            wrist_link_name = self.robot_cfg.get("wrist_link_name")
            if wrist_link_name is None:
                raise ValueError(
                    "robot_cfg must define 'wrist_link_name' to refine the wrist."
                )
            return (wrist_link_name,)

        finger_joint_maps = self.robot_cfg.get("finger_joint_maps")
        if finger_joint_maps is None:
            raise ValueError(
                "robot_cfg must define 'finger_joint_maps' to use "
                "sequential_finger_refinement."
            )
        if finger_name not in finger_joint_maps:
            raise ValueError(
                f"robot_cfg['finger_joint_maps'] has no entry for {finger_name!r}. "
                f"Available fingers: {list(finger_joint_maps)}."
            )

        joint_to_child_link = {
            joint.name: joint.child
            for joint in self._robot_urdf().joints
            if joint.joint_type != "fixed"
        }
        missing_joint_names = [
            joint_name
            for joint_name in finger_joint_maps[finger_name]
            if joint_name not in joint_to_child_link
        ]
        if missing_joint_names:
            raise ValueError(
                f"Finger {finger_name!r} maps to joints absent from the URDF: "
                f"{missing_joint_names}."
            )
        return tuple(
            joint_to_child_link[joint_name]
            for joint_name in finger_joint_maps[finger_name]
        )

    def compute_costs(self, q_var):

        if self.cfg.rotational_freedom:
            q_robot = q_var[: self.plant_f.num_positions()]
            rotation_params = q_var[self.plant_f.num_positions() :]
            _rotation_params = self.angle_axis_to_rotation_matrix(rotation_params)
            rotation_cost = np.sum(rotation_params[:2] ** 2)
        else:
            q_robot = q_var
            _rotation_params = None
            rotation_cost = 0.0

        eval_idx = self.logger.increment()
        self.update_beta(eval_idx)
        jpos_dict = self.compute_robot_joint_pos(q_robot)

        total_cost = 0.0
        hand_cfg_cost = 0.0
        hand_object_bps_cost = 0.0

        # hand_cfg cost
        hand_cfg_cost = 0.0
        if self.cfg.hand_cfg_weight > 0:
            computed_hand_vec_dict = compute_hand_cfg_vecs(jpos_dict)
            hand_cfg_cost = self.hand_cfg_cost(computed_hand_vec_dict, _rotation_params)
            # total_cost += self.cfg.hand_cfg_weight * hand_cfg_cost

        # hand-object bps cost
        hand_object_bps_cost = 0.0
        if self.cfg.hand_object_bps_weight > 0:
            hand_jpos_array = self._computed_hand_bps_points(jpos_dict)
            computed_hand_rel_bps = compute_object_bps_differentiable(
                self.target_pcd,
                hand_jpos_array,
                beta=self.cfg.bps_beta,
                normalize=False,
                include_other_features=self.cfg.bps_nocs_features,
            )
            hand_object_bps_cost, hand_object_logs = self.hand_object_bps_cost(
                computed_hand_rel_bps, _rotation_params
            )
            # total_cost += self.cfg.hand_object_bps_weight * hand_object_bps_cost
        else:
            hand_object_logs = {}

        total_cost = (
            np.exp(hand_cfg_cost * self.cfg.hand_cfg_weight)
            * hand_object_bps_cost
            * np.exp(rotation_cost)
        )

        self.logger.record(
            eval_idx=eval_idx,
            total_cost=total_cost,
            hand_cfg_cost=hand_cfg_cost,
            hand_object_bps_cost=hand_object_bps_cost,
            # add hand-object bps cost logs
            **{f"bps_{k}": v for k, v in hand_object_logs.items()},
            bps_beta=self.cfg.bps_beta,
        )

        return total_cost

    def _bps_indices_for_entities(
        self,
        included_entities: tuple[str, ...],
    ) -> np.ndarray:
        indices = [
            idx
            for idx, keypoint_name in enumerate(self.hand_bps_jpos_names)
            if self._hand_keypoint_entity(keypoint_name) in included_entities
        ]
        if len(indices) == 0:
            raise ValueError(
                f"No hand-object BPS keypoints found for entities {included_entities}."
            )
        return np.asarray(indices, dtype=int)

    def compute_costs_for_entities(
        self,
        q_var,
        included_entities: tuple[str, ...],
    ):
        if self.cfg.rotational_freedom:
            q_robot = q_var[: self.plant_f.num_positions()]
            rotation_params = q_var[self.plant_f.num_positions() :]
            _rotation_params = self.angle_axis_to_rotation_matrix(rotation_params)
            rotation_cost = np.sum(rotation_params[:2] ** 2)
        else:
            q_robot = q_var
            _rotation_params = None
            rotation_cost = 0.0

        jpos_dict = self.compute_robot_joint_pos(q_robot)
        hand_cfg_cost = 0.0
        if self.cfg.hand_cfg_weight > 0:
            computed_hand_vec_dict = compute_hand_cfg_vecs(jpos_dict)
            hand_cfg_cost = self.hand_cfg_cost(
                computed_hand_vec_dict,
                _rotation_params,
                included_entities=included_entities,
            )

        hand_object_bps_cost = 0.0
        if self.cfg.hand_object_bps_weight > 0:
            bps_indices = self._bps_indices_for_entities(included_entities)
            hand_jpos_array = self._computed_hand_bps_points(jpos_dict)[bps_indices]
            computed_hand_rel_bps = compute_object_bps_differentiable(
                self.target_pcd,
                hand_jpos_array,
                beta=self.cfg.bps_beta,
                normalize=False,
                include_other_features=self.cfg.bps_nocs_features,
            )
            hand_object_bps_cost, _ = self.hand_object_bps_cost(
                computed_hand_rel_bps,
                _rotation_params,
                reference_hand_rel_bps=self.reference_hand_rel_bps[bps_indices],
            )

        return (
            np.exp(hand_cfg_cost * self.cfg.hand_cfg_weight)
            * hand_object_bps_cost
            * np.exp(rotation_cost)
        )

    def _eval_signed_distances_for_geom_pairs(self, q_var, geom_pairs):
        plant, context = self.get_plant_context(q_var)
        plant.SetPositions(context, q_var)
        query_port = plant.get_geometry_query_input_port()
        if not query_port.HasValue(context):
            raise RuntimeError(
                "Cannot evaluate geometry query input port. Ensure MultibodyPlant is "
                "connected to SceneGraph and the stored context is from the same diagram."
            )

        query_object = query_port.Eval(context)
        distances = np.empty(len(geom_pairs), dtype=q_var.dtype)
        for pair_idx, (link_id, obj_id) in enumerate(geom_pairs):
            signed_distance_pair = query_object.ComputeSignedDistancePairClosestPoints(
                link_id, obj_id
            )
            distances[pair_idx] = signed_distance_pair.distance
        return distances

    def _eval_signed_distances_for_pairs(self, q_var):
        return self._eval_signed_distances_for_geom_pairs(
            q_var,
            self.hand_obj_geom_pairs,
        )

    def add_penetration_constraint_distance_pairs(self, prog=None, q=None):
        prog = self.prog if prog is None else prog
        q = self.q if q is None else q
        self.add_distance_constraints_for_geom_pairs(
            self.hand_obj_geom_pairs,
            distance_lower=self.cfg.penetration_bound,
            distance_upper=self.cfg.penetration_distance_upper,
            prog=prog,
            q=q,
        )

    def add_distance_constraints_for_geom_pairs(
        self,
        geom_pairs,
        *,
        distance_lower: float,
        distance_upper: float,
        prog=None,
        q=None,
    ):
        prog = self.prog if prog is None else prog
        q = self.q if q is None else q
        for link_id, obj_id in geom_pairs:
            distance_constraint = DistanceConstraint(
                self.plant_f,
                geometry_pair=(link_id, obj_id),
                plant_context=self.plant_context_f,
                distance_lower=distance_lower,
                distance_upper=distance_upper,
            )
            prog.AddConstraint(distance_constraint, q)

    def add_penetration_constraint_signed_distance_pairs(self, prog=None, q=None):
        prog = self.prog if prog is None else prog
        q = self.q if q is None else q
        if len(self.hand_obj_geom_pairs) == 0:
            return
        self.add_penetration_constraint_signed_distance_pairs_for_geom_pairs(
            self.hand_obj_geom_pairs,
            prog=prog,
            q=q,
        )

    def add_penetration_constraint_signed_distance_pairs_for_geom_pairs(
        self,
        geom_pairs,
        prog=None,
        q=None,
    ):
        prog = self.prog if prog is None else prog
        q = self.q if q is None else q
        if len(geom_pairs) == 0:
            return
        lb = np.full(len(geom_pairs), self.cfg.penetration_bound)
        ub = np.full(len(geom_pairs), self.cfg.penetration_distance_upper)
        prog.AddConstraint(
            lambda x: self._eval_signed_distances_for_geom_pairs(x, geom_pairs),
            lb,
            ub,
            q,
        )

    def add_penetration_constraint(self, prog=None, q=None):
        prog = self.prog if prog is None else prog
        q = self.q if q is None else q
        mode = self.cfg.penetration_constraint_mode
        if mode == "distance_constraint":
            self.add_penetration_constraint_distance_pairs(prog=prog, q=q)
            return
        if mode == "signed_distance_pairs":
            self.add_penetration_constraint_signed_distance_pairs(prog=prog, q=q)
            return
        if mode == "minimum_distance_lower_bound":
            penetration_constraint = MinimumDistanceLowerBoundConstraint(
                self.plant_f,
                bound=self.cfg.penetration_bound,
                plant_context=self.plant_context_f,
                influence_distance_offset=self.cfg.influence_distance_offset,
            )
            prog.AddConstraint(penetration_constraint, q)
            return
        raise ValueError(
            f"Unknown penetration_constraint_mode='{mode}'. Expected one of "
            "'distance_constraint', 'signed_distance_pairs', "
            "'minimum_distance_lower_bound'."
        )

    def _optimize(self, seed):

        np.random.seed(seed)
        solver_tolerance = self.cfg.solver_tolerance
        prog = self.prog

        if self.cfg.rotational_freedom:
            q = np.concatenate([self.q, self.rotation_var])
        else:
            q = self.q

        prog.AddCost(lambda x: self.compute_costs(x), q)

        if self.cfg.penetration_constraint:
            self.add_penetration_constraint()

        solver = SnoptSolver()
        solver_id = solver.solver_id()
        prog.SetSolverOption(solver_id, "Feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Major feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Major optimality tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Minor feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Minor optimality tolerance", solver_tolerance)
        solver_options = SolverOptions()

        prog.SetInitialGuess(
            self.q,
            np.random.uniform(low=-0.1, high=0.1, size=self.plant_f.num_positions()),
        )
        if self.cfg.rotational_freedom:
            prog.SetInitialGuess(
                self.rotation_var,
                np.random.uniform(low=-0.1, high=0.1, size=3),
            )

        solve_start_time = time()
        result = Solve(prog, solver_options=solver_options)
        solve_elapsed_sec = time() - solve_start_time
        self.last_solve_timing = {
            "main_solve_time_sec": float(solve_elapsed_sec),
            "is_success": bool(result.is_success()),
            "eval_count": int(self.logger.eval_count),
        }
        self.solve_timing_history = [
            *getattr(self, "solve_timing_history", []),
            self.last_solve_timing,
        ]
        self.logger.print_counts()
        self.logger.dump(self.run_dir)
        print(
            f"[single_step_ik] "
            f"is_success={result.is_success()} solve_time_sec={solve_elapsed_sec:.3f}"
        )

        q_sol = result.GetSolution(self.q)

        return q_sol

    def optimize(self):
        while True:
            q_sol = self._optimize(seed=self.cfg.seed)
            if self.logger.eval_count < self.cfg.min_optimizer_iterations:
                # optimization likely failed. try again with a different seed.
                print(
                    f"Optimization finished with only {self.logger.eval_count} evaluations. "
                    "Retrying with a different random seed..."
                )
                self.logger.clear()
                self.cfg.seed += 1
            else:
                break

        return self.refine_solution(q_sol)

    def optimize_with_trace(self) -> list[tuple[str, np.ndarray]]:
        while True:
            q_sol = self._optimize(seed=self.cfg.seed)
            if self.logger.eval_count < self.cfg.min_optimizer_iterations:
                print(
                    f"Optimization finished with only {self.logger.eval_count} evaluations. "
                    "Retrying with a different random seed..."
                )
                self.logger.clear()
                self.cfg.seed += 1
            else:
                break

        trace = [("main_solve", np.asarray(q_sol, dtype=float).reshape(-1).copy())]
        trace.extend(self.refine_solution_trace(q_sol))
        return trace

    def refine_solution(self, q_sol: np.ndarray) -> np.ndarray:
        trace = self.refine_solution_trace(q_sol)
        if len(trace) == 0:
            return np.asarray(q_sol, dtype=float).reshape(-1)
        return trace[-1][1]

    def refine_solution_trace(self, q_sol: np.ndarray) -> list[tuple[str, np.ndarray]]:
        q_sol = np.asarray(q_sol, dtype=float).reshape(-1)
        if q_sol.shape != (self.plant_f.num_positions(),):
            raise ValueError(
                "Expected q_sol to have shape "
                f"({self.plant_f.num_positions()},), got {q_sol.shape}."
            )
        trace = []
        q_current = self.non_penetration_refinement(q_sol)
        if self.cfg.non_penetration_refinement:
            trace.append(("nonpen", np.asarray(q_current, dtype=float).copy()))

        if self.cfg.sequential_finger_refinement:
            q_current = self.sequential_finger_refinement(q_current)
            trace.append(("seqfinger", np.asarray(q_current, dtype=float).copy()))

        if self.cfg.sequential_line_search_refinement:
            q_current = self.sequential_line_search_refinement(q_current, q_sol)
            trace.append(("lineseq", np.asarray(q_current, dtype=float).copy()))

        q_current = self.contact_target_refinement(q_current)
        if self.cfg.contact_target_refinement:
            trace.append(("contacttarget", np.asarray(q_current, dtype=float).copy()))
        return trace

    def _sequential_finger_nominal_weight(self, finger_name: str) -> float:
        if finger_name == "wrist":
            return self.cfg.sequential_wrist_nominal_weight
        return self.cfg.sequential_finger_nominal_weight

    def _sequential_finger_refinement_cost(
        self,
        q_var,
        q_nominal: np.ndarray,
        finger_name: str,
    ):
        q_error = q_var - q_nominal
        nominal_cost = np.dot(q_error, q_error)
        total_cost = self._sequential_finger_nominal_weight(finger_name) * nominal_cost
        if self.cfg.sequential_finger_use_masked_cost:
            included_entities = self._stage_included_entities(finger_name)
            total_cost = total_cost + self.compute_costs_for_entities(
                q_var,
                included_entities,
            )
        return total_cost

    def _finger_child_link_geom_pairs(
        self,
        child_link_name: str,
    ) -> list[tuple[object, object]]:
        hand_geom_ids = self.hand_body_name_to_geom_ids.get(child_link_name)
        if hand_geom_ids is None:
            raise ValueError(
                f"Child link {child_link_name!r} is not a hand body in Drake. "
                f"Available hand bodies: {list(self.hand_body_name_to_geom_ids)}."
            )
        if len(hand_geom_ids) == 0:
            raise ValueError(
                f"Child link {child_link_name!r} has no collision geometries."
            )
        return [
            (hand_geom_id, obj_geom_id)
            for hand_geom_id in hand_geom_ids
            for obj_geom_id in self.obj_geom_ids
        ]

    def _finger_hand_geom_ids(self, finger_name: str) -> list[object]:
        geom_pairs = self.finger_hand_obj_geom_pairs.get(finger_name)
        if geom_pairs is None:
            raise ValueError(
                f"No hand-object geometry pairs found for stage {finger_name!r}."
            )
        hand_geom_ids = []
        for hand_geom_id, _ in geom_pairs:
            if hand_geom_id not in hand_geom_ids:
                hand_geom_ids.append(hand_geom_id)
        return hand_geom_ids

    @staticmethod
    def _hand_self_geom_pairs(
        active_geom_ids: list[object],
        fixed_geom_ids: list[object],
    ) -> list[tuple[object, object]]:
        return [
            (active_geom_id, fixed_geom_id)
            for active_geom_id in active_geom_ids
            for fixed_geom_id in fixed_geom_ids
            if active_geom_id != fixed_geom_id
        ]

    def _robot_parent_link_for_frame(
        self,
        frame_name: str,
        candidate_link_names: tuple[str, ...],
    ) -> str:
        if frame_name in candidate_link_names:
            return frame_name

        child_to_parent_link = {
            joint.child: joint.parent for joint in self._robot_urdf().joints
        }
        current_link_name = frame_name
        visited_link_names = set()
        while current_link_name in child_to_parent_link:
            if current_link_name in visited_link_names:
                raise ValueError(
                    f"Cycle detected while walking URDF parents from {frame_name!r}."
                )
            visited_link_names.add(current_link_name)
            current_link_name = child_to_parent_link[current_link_name]
            if current_link_name in candidate_link_names:
                return current_link_name

        raise ValueError(
            f"Could not map robot frame {frame_name!r} to any candidate contact link "
            f"{candidate_link_names}."
        )

    def _demo_contact_child_link_names(self, finger_name: str) -> tuple[str, ...]:
        stage_entity = self._stage_entity(finger_name)
        candidate_link_names = self.get_finger_joint_child_link_names(finger_name)
        reference_object_points = np.asarray(self.reference_pcd[..., :3], dtype=float)
        if reference_object_points.ndim != 2 or reference_object_points.shape[1] != 3:
            raise ValueError(
                "Expected reference_pcd[..., :3] to have shape (N, 3), got "
                f"{reference_object_points.shape}."
            )

        contact_child_link_names = []
        for keypoint_name in self.hand_bps_jpos_names:
            if self._hand_keypoint_entity(keypoint_name) != stage_entity:
                continue
            keypoint = np.asarray(self.hand_bps_jpos_dict[keypoint_name], dtype=float)
            distances = np.linalg.norm(
                reference_object_points - keypoint[None, :], axis=1
            )
            if float(np.min(distances)) >= self.cfg.sequential_finger_contact_distance:
                continue

            robot_frame_name = self._robot_frame_name_for_human_keypoint(keypoint_name)
            child_link_name = self._robot_parent_link_for_frame(
                robot_frame_name,
                candidate_link_names,
            )
            if child_link_name not in contact_child_link_names:
                contact_child_link_names.append(child_link_name)
        return tuple(contact_child_link_names)

    def _select_finger_contact_geom_pairs(
        self,
        q_var: np.ndarray,
        finger_name: str,
    ) -> list[tuple[object, object]]:
        contact_geom_pairs = []
        child_link_names = self._demo_contact_child_link_names(finger_name)
        for child_link_name in child_link_names:
            geom_pairs = self._finger_child_link_geom_pairs(child_link_name)
            distances = np.asarray(
                self._eval_signed_distances_for_geom_pairs(q_var, geom_pairs),
                dtype=float,
            )
            closest_pair_idx = int(np.argmin(distances))
            contact_geom_pairs.append(geom_pairs[closest_pair_idx])
        return contact_geom_pairs

    def _distance_bound_violation(
        self,
        q_var: np.ndarray,
        geom_pairs: list[tuple[object, object]],
        *,
        distance_lower: float,
        distance_upper: float,
    ) -> float:
        if len(geom_pairs) == 0:
            return 0.0
        distances = np.asarray(
            self._eval_signed_distances_for_geom_pairs(q_var, geom_pairs),
            dtype=float,
        )
        lower_violation = np.maximum(distance_lower - distances, 0.0)
        upper_violation = np.maximum(distances - distance_upper, 0.0)
        return float(
            np.dot(lower_violation, lower_violation)
            + np.dot(
                upper_violation,
                upper_violation,
            )
        )

    def _line_search_constraint_violation(
        self,
        q_var: np.ndarray,
        object_geom_pairs: list[tuple[object, object]],
        inter_stage_geom_pairs: list[tuple[object, object]],
        contact_geom_pairs: list[tuple[object, object]],
    ) -> float:
        return (
            self._distance_bound_violation(
                q_var,
                object_geom_pairs,
                distance_lower=self.cfg.sequential_finger_penetration_bound,
                distance_upper=self.cfg.penetration_distance_upper,
            )
            + self._distance_bound_violation(
                q_var,
                inter_stage_geom_pairs,
                distance_lower=self.cfg.sequential_inter_stage_penetration_bound,
                distance_upper=self.cfg.penetration_distance_upper,
            )
            + self._distance_bound_violation(
                q_var,
                contact_geom_pairs,
                distance_lower=0.0,
                distance_upper=self.cfg.sequential_finger_contact_tolerance,
            )
        )

    def _line_search_joint_values(
        self,
        q_current: np.ndarray,
        q_nominal: np.ndarray,
        joint_idx: int,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
    ) -> np.ndarray:
        n_samples = int(self.cfg.sequential_line_search_samples)
        radius = float(self.cfg.sequential_line_search_radius)
        if n_samples < 2:
            raise ValueError(
                "sequential_line_search_samples must be >= 2, got " f"{n_samples}."
            )
        if radius <= 0.0:
            raise ValueError(
                f"sequential_line_search_radius must be > 0, got {radius}."
            )

        lower_limit = float(lower_limits[joint_idx])
        upper_limit = float(upper_limits[joint_idx])
        if lower_limit > upper_limit:
            raise ValueError(
                f"Joint {self.dof_names[joint_idx]!r} has invalid limits "
                f"[{lower_limit}, {upper_limit}]."
            )
        lower = max(lower_limit, float(q_current[joint_idx]) - radius)
        upper = min(upper_limit, float(q_current[joint_idx]) + radius)
        if lower > upper:
            raise ValueError(
                f"Line search interval is empty for joint {self.dof_names[joint_idx]!r}: "
                f"[{lower}, {upper}]."
            )

        values = np.concatenate(
            [
                np.linspace(lower, upper, n_samples),
                np.asarray(
                    [
                        np.clip(q_current[joint_idx], lower_limit, upper_limit),
                        np.clip(q_nominal[joint_idx], lower_limit, upper_limit),
                    ],
                    dtype=float,
                ),
            ]
        )
        return np.unique(values)

    def _line_search_single_joint(
        self,
        q_current: np.ndarray,
        q_nominal: np.ndarray,
        joint_idx: int,
        object_geom_pairs: list[tuple[object, object]],
        inter_stage_geom_pairs: list[tuple[object, object]],
        contact_geom_pairs: list[tuple[object, object]],
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        best_q = None
        best_key = None
        for joint_value in self._line_search_joint_values(
            q_current,
            q_nominal,
            joint_idx,
            lower_limits,
            upper_limits,
        ):
            q_candidate = q_current.copy()
            q_candidate[joint_idx] = joint_value
            violation = self._line_search_constraint_violation(
                q_candidate,
                object_geom_pairs,
                inter_stage_geom_pairs,
                contact_geom_pairs,
            )
            q_error = q_candidate - q_nominal
            nominal_error = float(np.dot(q_error, q_error))
            key = (violation > 1e-12, violation, nominal_error)
            if best_key is None or key < best_key:
                best_key = key
                best_q = q_candidate

        if best_q is None or best_key is None:
            raise RuntimeError(
                f"No line-search candidates evaluated for joint "
                f"{self.dof_names[joint_idx]!r}."
            )
        return best_q, float(best_key[1])

    def sequential_line_search_refinement(
        self,
        q_solved: np.ndarray,
        q_nominal: np.ndarray,
    ) -> np.ndarray:
        if not self.cfg.sequential_line_search_refinement:
            return q_solved

        q_current = np.asarray(q_solved, dtype=float).reshape(-1)
        q_nominal = np.asarray(q_nominal, dtype=float).reshape(-1)
        expected_shape = (self.plant_f.num_positions(),)
        if q_current.shape != expected_shape:
            raise ValueError(
                f"Expected q_solved shape {expected_shape}, got {q_current.shape}."
            )
        if q_nominal.shape != expected_shape:
            raise ValueError(
                f"Expected q_nominal shape {expected_shape}, got {q_nominal.shape}."
            )
        n_passes = int(self.cfg.sequential_line_search_passes)
        if n_passes < 1:
            raise ValueError(
                f"sequential_line_search_passes must be >= 1, got {n_passes}."
            )

        lower_limits = self.plant_f.GetPositionLowerLimits()
        upper_limits = self.plant_f.GetPositionUpperLimits()
        previous_stage_geom_ids = []
        for finger_name in self.cfg.sequential_finger_refinement_order:
            finger_joint_indices = self.get_finger_joint_indices(finger_name)
            object_geom_pairs = self.finger_hand_obj_geom_pairs.get(finger_name)
            if object_geom_pairs is None or len(object_geom_pairs) == 0:
                raise ValueError(
                    f"No hand-object collision geometry pairs found for stage "
                    f"{finger_name!r}."
                )
            active_stage_geom_ids = self._finger_hand_geom_ids(finger_name)
            inter_stage_geom_pairs = self._hand_self_geom_pairs(
                active_stage_geom_ids,
                previous_stage_geom_ids,
            )
            contact_geom_pairs = self._select_finger_contact_geom_pairs(
                q_current,
                finger_name,
            )
            print(
                f"[sequential_line_search:{finger_name}] "
                f"num_joints={len(finger_joint_indices)} "
                f"num_geom_pairs={len(object_geom_pairs)} "
                f"num_inter_stage_pairs={len(inter_stage_geom_pairs)} "
                f"num_contact_pairs={len(contact_geom_pairs)}"
            )

            for _ in range(n_passes):
                for joint_idx in finger_joint_indices:
                    q_current, _ = self._line_search_single_joint(
                        q_current,
                        q_nominal,
                        int(joint_idx),
                        object_geom_pairs,
                        inter_stage_geom_pairs,
                        contact_geom_pairs,
                        lower_limits,
                        upper_limits,
                    )

            final_violation = self._line_search_constraint_violation(
                q_current,
                object_geom_pairs,
                inter_stage_geom_pairs,
                contact_geom_pairs,
            )
            print(
                f"[sequential_line_search:{finger_name}] "
                f"final_constraint_violation={final_violation:.6e}"
            )
            for active_geom_id in active_stage_geom_ids:
                if active_geom_id not in previous_stage_geom_ids:
                    previous_stage_geom_ids.append(active_geom_id)
        return q_current

    def sequential_finger_refinement(self, q_solved):
        if not self.cfg.sequential_finger_refinement:
            return q_solved

        q_current = np.asarray(q_solved, dtype=float).reshape(-1)
        if q_current.shape != (self.plant_f.num_positions(),):
            raise ValueError(
                "Expected q_solved to have shape "
                f"({self.plant_f.num_positions()},), got {q_current.shape}."
            )
        if not self.finger_hand_obj_geom_pairs:
            raise ValueError(
                "sequential_finger_refinement requires robot_cfg['finger_joint_maps'] "
                "and per-finger collision geometry pairs."
            )

        previous_stage_geom_ids = []
        for finger_name in self.cfg.sequential_finger_refinement_order:
            included_entities = self._stage_included_entities(finger_name)
            if self.cfg.sequential_finger_use_masked_cost:
                print(
                    f"Refining placement of {finger_name} joints with cost entities "
                    f"{included_entities}."
                )
            else:
                print(
                    f"Refining placement of {finger_name} joints with nominal-only "
                    "cost plus constraints."
                )

            finger_joint_indices = self.get_finger_joint_indices(finger_name)
            geom_pairs = self.finger_hand_obj_geom_pairs.get(finger_name)
            if geom_pairs is None or len(geom_pairs) == 0:
                raise ValueError(
                    f"No hand-object collision geometry pairs found for finger "
                    f"{finger_name!r}. Check the finger aliases and URDF collision "
                    "body names."
                )
            active_stage_geom_ids = self._finger_hand_geom_ids(finger_name)
            inter_stage_geom_pairs = self._hand_self_geom_pairs(
                active_stage_geom_ids,
                previous_stage_geom_ids,
            )
            contact_geom_pairs = self._select_finger_contact_geom_pairs(
                q_current,
                finger_name,
            )

            fixed_indices = np.setdiff1d(
                np.arange(self.plant_f.num_positions(), dtype=int),
                finger_joint_indices,
            )
            prog = MathematicalProgram()
            q_refined = prog.NewContinuousVariables(
                self.plant_f.num_positions(),
                f"q_{finger_name}_refined",
            )

            lower_limit = self.plant_f.GetPositionLowerLimits()
            upper_limit = self.plant_f.GetPositionUpperLimits()
            prog.AddBoundingBoxConstraint(lower_limit, upper_limit, q_refined)
            prog.AddBoundingBoxConstraint(
                q_current[fixed_indices],
                q_current[fixed_indices],
                q_refined[fixed_indices],
            )
            prog.AddCost(
                lambda x, q_nominal=q_current.copy(), active_name=finger_name: (
                    self._sequential_finger_refinement_cost(
                        x,
                        q_nominal,
                        active_name,
                    )
                ),
                q_refined,
            )
            self.add_distance_constraints_for_geom_pairs(
                geom_pairs,
                distance_lower=self.cfg.sequential_finger_penetration_bound,
                distance_upper=self.cfg.penetration_distance_upper,
                prog=prog,
                q=q_refined,
            )
            if len(inter_stage_geom_pairs) > 0:
                self.add_distance_constraints_for_geom_pairs(
                    inter_stage_geom_pairs,
                    distance_lower=self.cfg.sequential_inter_stage_penetration_bound,
                    distance_upper=self.cfg.penetration_distance_upper,
                    prog=prog,
                    q=q_refined,
                )
            if len(contact_geom_pairs) > 0:
                self.add_distance_constraints_for_geom_pairs(
                    contact_geom_pairs,
                    distance_lower=0.0,
                    distance_upper=self.cfg.sequential_finger_contact_tolerance,
                    prog=prog,
                    q=q_refined,
                )
            prog.SetInitialGuess(q_refined, q_current)

            solver = SnoptSolver()
            solver_id = solver.solver_id()
            solver_tolerance = self.cfg.solver_tolerance
            prog.SetSolverOption(solver_id, "Feasibility tolerance", solver_tolerance)
            prog.SetSolverOption(
                solver_id, "Major feasibility tolerance", solver_tolerance
            )
            prog.SetSolverOption(
                solver_id, "Major optimality tolerance", solver_tolerance
            )
            prog.SetSolverOption(
                solver_id, "Minor feasibility tolerance", solver_tolerance
            )
            prog.SetSolverOption(
                solver_id, "Minor optimality tolerance", solver_tolerance
            )

            refine_start_time = time()
            result = Solve(prog)
            refine_elapsed_sec = time() - refine_start_time
            print(
                f"[sequential_finger_refinement:{finger_name}] "
                f"is_success={result.is_success()} "
                f"solve_time_sec={refine_elapsed_sec:.3f} "
                f"num_joints={len(finger_joint_indices)} "
                f"num_geom_pairs={len(geom_pairs)} "
                f"num_inter_stage_pairs={len(inter_stage_geom_pairs)} "
                f"num_contact_pairs={len(contact_geom_pairs)}"
            )

            # if result.is_success():
            q_current = result.GetSolution(q_refined)
            for active_geom_id in active_stage_geom_ids:
                if active_geom_id not in previous_stage_geom_ids:
                    previous_stage_geom_ids.append(active_geom_id)
        return q_current

    def _extract_contact_targets(self, q_solved: np.ndarray) -> dict[str, np.ndarray]:
        jpos_dict = self.compute_robot_joint_pos(q_solved)
        hand_bps_points = self._computed_hand_bps_points(jpos_dict)
        mesh = self._target_mesh_for_contact_refinement()
        closest_points, distances, face_indices = trimesh.proximity.closest_point(
            mesh,
            hand_bps_points,
        )
        contact_mask = distances < float(self.cfg.contact_target_distance)
        contact_indices = np.nonzero(contact_mask)[0]
        if len(contact_indices) == 0:
            return {
                "names": np.asarray([], dtype=object),
                "target_positions": np.zeros((0, 3), dtype=float),
                "target_normals": np.zeros((0, 3), dtype=float),
                "local_robot_normals": np.zeros((0, 3), dtype=float),
            }

        contact_names = tuple(self.hand_bps_jpos_names[idx] for idx in contact_indices)
        target_positions = np.asarray(closest_points[contact_indices], dtype=float)
        face_indices = np.asarray(face_indices[contact_indices], dtype=np.int64)
        target_normals = np.asarray(mesh.face_normals[face_indices], dtype=float)
        target_norms = np.linalg.norm(target_normals, axis=1)
        if np.any(target_norms <= 1e-10):
            bad_names = [
                contact_names[idx] for idx in np.nonzero(target_norms <= 1e-10)[0]
            ]
            raise ValueError(
                f"Target object has near-zero face normals for contact kpts: {bad_names}."
            )
        target_normals = target_normals / target_norms[:, None]

        transforms = self._compute_robot_keypoint_frames(q_solved, contact_names)
        local_robot_normals = []
        for contact_name, target_normal in zip(contact_names, target_normals):
            rotation = np.asarray(transforms[contact_name].rotation().matrix())
            desired_robot_normal = -target_normal
            local_robot_normals.append(rotation.T @ desired_robot_normal)

        return {
            "names": np.asarray(contact_names, dtype=object),
            "target_positions": target_positions,
            "target_normals": target_normals,
            "local_robot_normals": np.stack(local_robot_normals, axis=0),
        }

    def _contact_target_cost(
        self,
        q_var,
        contact_names: tuple[str, ...],
        target_positions: np.ndarray,
        target_normals: np.ndarray,
        local_robot_normals: np.ndarray,
        q_nominal: np.ndarray,
    ):
        transforms = self._compute_robot_keypoint_frames(q_var, contact_names)
        position_cost = 0.0
        normal_cost = 0.0
        for contact_idx, contact_name in enumerate(contact_names):
            transform = transforms[contact_name]
            position = transform.translation()
            position_error = position - target_positions[contact_idx]
            position_cost = position_cost + np.dot(position_error, position_error)

            rotation = transform.rotation().matrix()
            robot_normal = rotation @ local_robot_normals[contact_idx]
            desired_robot_normal = -target_normals[contact_idx]
            normal_cost = normal_cost + (
                1.0 - np.dot(robot_normal, desired_robot_normal)
            )

        q_error = q_var - q_nominal
        nominal_cost = np.dot(q_error, q_error)
        return (
            self.cfg.contact_target_position_weight * position_cost
            + self.cfg.contact_target_normal_weight * normal_cost
            + self.cfg.contact_target_nominal_weight * nominal_cost
        )

    def contact_target_refinement(self, q_solved):
        if not self.cfg.contact_target_refinement:
            return q_solved

        q_solved = np.asarray(q_solved, dtype=float).reshape(-1)
        contact_targets = self._extract_contact_targets(q_solved)
        contact_names = tuple(str(name) for name in contact_targets["names"])
        if len(contact_names) == 0:
            print(
                "[contact_target_refinement] no BPS keypoints within "
                f"{self.cfg.contact_target_distance:.4f}m of the object."
            )
            return q_solved

        print(
            "[contact_target_refinement] optimizing contact kpts: "
            f"{list(contact_names)}"
        )
        prog = MathematicalProgram()
        q_refined = prog.NewContinuousVariables(
            self.plant_f.num_positions(),
            "q_contact_refined",
        )
        lower_limit = self.plant_f.GetPositionLowerLimits()
        upper_limit = self.plant_f.GetPositionUpperLimits()
        prog.AddBoundingBoxConstraint(lower_limit, upper_limit, q_refined)
        prog.AddCost(
            lambda x: self._contact_target_cost(
                x,
                contact_names,
                contact_targets["target_positions"],
                contact_targets["target_normals"],
                contact_targets["local_robot_normals"],
                q_solved,
            ),
            q_refined,
        )
        if self.cfg.penetration_constraint:
            self.add_penetration_constraint(prog=prog, q=q_refined)
        prog.SetInitialGuess(q_refined, q_solved)

        solver = SnoptSolver()
        solver_id = solver.solver_id()
        solver_tolerance = self.cfg.solver_tolerance
        prog.SetSolverOption(solver_id, "Feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Major feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Major optimality tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Minor feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Minor optimality tolerance", solver_tolerance)

        refine_start_time = time()
        result = Solve(prog)
        refine_elapsed_sec = time() - refine_start_time
        print(
            "[contact_target_refinement] "
            f"is_success={result.is_success()} solve_time_sec={refine_elapsed_sec:.3f}"
        )

        if not result.is_success():
            return q_solved
        return result.GetSolution(q_refined)

    def non_penetration_refinement(self, q_solved):
        if not self.cfg.non_penetration_refinement:
            return q_solved

        q_solved = np.asarray(q_solved, dtype=float).reshape(-1)
        root_indices = self.get_root_position_indices()
        finger_base_indices = self.get_finger_base_indices()
        to_optimize_indices = np.concatenate([root_indices, finger_base_indices])

        print(f"Root position indices: {root_indices}")
        print(f"Finger base indices: {finger_base_indices}")
        fixed_indices = np.setdiff1d(
            np.arange(self.plant_f.num_positions(), dtype=int),
            to_optimize_indices,
        )

        prog = MathematicalProgram()
        q_refined = prog.NewContinuousVariables(
            self.plant_f.num_positions(), "q_refined"
        )

        lower_limit = self.plant_f.GetPositionLowerLimits()
        upper_limit = self.plant_f.GetPositionUpperLimits()
        prog.AddBoundingBoxConstraint(lower_limit, upper_limit, q_refined)
        prog.AddBoundingBoxConstraint(
            q_solved[fixed_indices], q_solved[fixed_indices], q_refined[fixed_indices]
        )

        # q_nominal = q_solved[to_optimize_indices].copy()
        # prog.AddCost(
        #     # lambda x: np.dot(x - q_nominal, x - q_nominal),
        #     lambda x: np.abs(x - q_nominal).sum(),
        #     q_refined[to_optimize_indices],
        # )
        prog.AddCost(lambda x: self.compute_costs(x), q_refined)

        self.add_penetration_constraint(prog=prog, q=q_refined)

        prog.SetInitialGuess(q_refined, q_solved)

        solver = SnoptSolver()
        solver_id = solver.solver_id()
        solver_tolerance = self.cfg.solver_tolerance
        prog.SetSolverOption(solver_id, "Feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Major feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Major optimality tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Minor feasibility tolerance", solver_tolerance)
        prog.SetSolverOption(solver_id, "Minor optimality tolerance", solver_tolerance)

        refine_start_time = time()
        result = Solve(prog)
        refine_elapsed_sec = time() - refine_start_time
        print(
            "[non_penetration_refinement] "
            f"is_success={result.is_success()} solve_time_sec={refine_elapsed_sec:.3f}"
        )

        if not result.is_success():
            return q_solved
        return result.GetSolution(q_refined)
