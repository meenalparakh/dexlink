from __future__ import annotations

from typing import TYPE_CHECKING, Any

import genesis as gs
import numpy as np
import torch

from dexlink.src.rl.sim.art_object import ArtObject, get_object_cfg
from dexlink.src.rl.sim.robot import Hand, get_hand_specific_cfg
from dexlink.src.retargeting.data.demo_loader import is_oakink_dataset, load_demo_processor
from dexlink.src.retargeting.data.oakink_processor_grasp import OakinkProcessorGrasp
from dexlink.utils.retargeting.path_utils import get_logs_dir

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import SceneCfg
    from dexlink.src.rl.sim.env import Env
    from dexlink.src.retargeting.data.exported_oakink_processor import ExportedOakInkProcessor
    from dexlink.src.retargeting.data.process_arctic import ArcticProcessor, RandomTrajProcessor


class SceneManager:
    def __init__(self, cfg: SceneCfg, env: Env):
        self.cfg = cfg
        self.env = env
        self.device = env.device
        self.video_save_freq = self.cfg.video_save_freq
        self.video_save_length = self.cfg.video_save_length
        self.logs_dir = get_logs_dir() / self.cfg.video_save_dir

        self.record = (self.video_save_freq > 0) and (not self.cfg.show_viewer)
        self.is_recording = False
        self.scene_entities: dict[str, Any] = {}
        self.object_only = self.cfg.object_only_scene
        self.hand_side_available = {"left": False, "right": False}
        # self.use_force_sensor = self.cfg.use_force_sensor and (
        #     not self.cfg.object_only_scene
        # )

        self.original_demo: (
            ArcticProcessor | ExportedOakInkProcessor | RandomTrajProcessor
        )
        self._env_random_force: np.ndarray | None = None

    def initialize(self):
        self.original_demo = load_demo_processor(
            self.cfg,
            start_timestep=self.cfg.start_timestep,
            end_timestep=self.cfg.end_timestep,
        )
        self.hand_side_available = self._resolve_hand_side_availability()
        self.cfg.apply_random_force = self.cfg.apply_random_force and isinstance(
            self.original_demo, OakinkProcessorGrasp
        )
        if is_oakink_dataset(self.cfg.demo_dataset):
            print(
                f"Using {self.cfg.demo_dataset} demo from: {self.cfg.oakink_exported_path}"
            )
        elif self.cfg.demo_dataset == "arctic":
            print(f"Using arctic demo from: {self.cfg.original_demo_path}")
        else:
            print(
                f"Using random trajectory results from: {self.original_demo.robot_obj_traj_path}"
            )
        self.total_steps = self.original_demo.n_frames
        self.camera_lookat = self._compute_camera_lookat()

        scene_cfg = self.get_scene_cfg()
        gs.init(backend=gs.cpu if self.env.device == "cpu" else gs.cuda)

        self.scene = gs.Scene(**scene_cfg)

        self.initialize_cameras()
        self.mov_cam = self.cfg.camera_mov

        _plane = self.scene.add_entity(
            gs.morphs.Plane(pos=(0, 0, -0.05)),
        )
        self.scene_entities["plane"] = _plane

        if self.cfg.demo_dataset == "oakink_lift":
            support_height = float(self.cfg.lift_support_height)
            support_surface = gs.surfaces.Rough(
                roughness=0.8, color=(0.58, 0.50, 0.42, 1.0)
            )
            self.scene_entities["platform"] = self.scene.add_entity(
                gs.morphs.Box(
                    pos=(0.0, 0.0, support_height / 2.0),
                    size=(0.8, 1.2, support_height),
                    fixed=True,
                    collision=True,
                ),
                surface=support_surface,
            )
            self.scene_entities["block"] = self.scene.add_entity(
                gs.morphs.Box(
                    pos=(0.0, 0.0, support_height + 0.05),
                    size=(0.1, 0.1, 0.1),
                    fixed=True,
                    collision=True,
                ),
                surface=gs.surfaces.Rough(roughness=0.9, color=(0.8, 0.8, 0.8, 1.0)),
            )

        if not is_oakink_dataset(self.cfg.demo_dataset):
            cardbox_size = (0.1, 0.2, 0.1)
            if "notebook" in self.cfg.target_object:
                print("Adding a SMALLER cardboard box for notebook")
                cardbox_size = (0.1, 0.15, 0.1)

            cardbox_surface = gs.surfaces.Rough(
                roughness=0.6, color=(167 / 255.0, 134 / 255.0, 103 / 255.0, 1.0)
            )
            self.scene_entities["platform"] = self.scene.add_entity(
                gs.morphs.Box(
                    pos=(0, -0.08, 0.90),
                    size=cardbox_size,
                    fixed=True,
                    collision=True,
                ),
                surface=cardbox_surface,
            )
        self.object_cfg = get_object_cfg(
            reference_demo=self.original_demo,
            urdf_path=self.original_demo.target_obj_urdf_path,
            obj_name=self.original_demo.target_obj_name,
            object_fixed=self.cfg.object_fixed,
        )

        self.scene_entities["object"] = ArtObject(
            scene=self.scene, obj_cfg=self.object_cfg, num_envs=self.cfg.num_envs
        )

        if not self.object_only:
            self.hand_cfgs = get_hand_specific_cfg(self.cfg.hand_name)
            robot_base_pose = getattr(self.original_demo, "robot_base_pose", None)
            if self.has_hand("left"):
                self.scene_entities["left"] = Hand(
                    scene=self.scene,
                    robo_cfg=self.hand_cfgs["left"],
                    num_envs=self.cfg.num_envs,
                    device=self.device,
                    obs_cfg=self.env.cfg.obs_cfg.hand_cfg,
                    env=self.env,
                    name="left",
                    base_pose=robot_base_pose,
                )
            if self.has_hand("right"):
                self.scene_entities["right"] = Hand(
                    scene=self.scene,
                    robo_cfg=self.hand_cfgs["right"],
                    num_envs=self.cfg.num_envs,
                    device=self.device,
                    obs_cfg=self.env.cfg.obs_cfg.hand_cfg,
                    env=self.env,
                    name="right",
                    base_pose=robot_base_pose,
                )

            self.original_demo.process_retargeted_motion(
                left_hand=self.scene_entities.get("left"),
                right_hand=self.scene_entities.get("right"),
            )

        # # not being used currently. the contact information is being
        # # directly read from the ground truth simulator information
        # # the obs config directly controls whether contact info is taken into account

        # if self.use_force_sensor:
        #     self.hand_link_names = {
        #         "left": self.hand_cfgs["left"]["all_relevant_links"],
        #         "right": self.hand_cfgs["right"]["all_relevant_links"],
        #     }
        #     left_link_indices = [
        #         self.scene_entities["left"].robot.get_link(name).idx_local
        #         for name in self.hand_link_names["left"]
        #     ]
        #     right_link_indices = [
        #         self.scene_entities["right"].robot.get_link(name).idx_local
        #         for name in self.hand_link_names["right"]
        #     ]
        #     self.hand_link_indices = dict(
        #         left=left_link_indices, right=right_link_indices
        #     )

        #     hand_sensors = dict(left=[], right=[])
        #     for side in ["left", "right"]:
        #         robot_idx = self.scene_entities[side].robot.idx
        #         for link_idx in self.hand_link_indices[side]:
        #             hand_sensors[side].append(
        #                 self.scene.add_sensor(
        #                     gs.sensors.ContactForce(
        #                         entity_idx=robot_idx, link_idx_local=link_idx
        #                     )
        #                 )
        #             )
        #     self.hand_sensors = hand_sensors

        else:
            self.hand_sensors = None

        print(f"BUILDING SCENE WITH {self.cfg.num_envs} ENVS...")
        self.scene.build(
            n_envs=self.cfg.num_envs,
            env_spacing=(3.0, 3.0),
            n_envs_per_row=int(np.sqrt(self.cfg.num_envs)) + 1,
        )

        # note that gains for the object are being set to 0.0
        self.scene_entities["object"].set_gains()
        if not self.object_only:
            for side in self.active_hand_sides:
                self.scene_entities[side].set_gains()

        if self.cfg.apply_random_force:
            self.resample_random_force(env_ids=None)

    def resample_random_force(self, env_ids: np.ndarray | list[int] | None = None):
        if not self.cfg.apply_random_force:
            return

        if self._env_random_force is None:
            self._env_random_force = np.zeros(
                (self.env.num_envs, 1, 3), dtype=np.float32
            )

        if env_ids is None:
            env_ids_np = np.arange(self.env.num_envs, dtype=np.int64)
        else:
            env_ids_np = np.asarray(env_ids, dtype=np.int64).reshape(-1)
            if env_ids_np.size == 0:
                return

        f_mag = self.cfg.random_force_mag
        self._env_random_force[env_ids_np] = np.random.uniform(
            -f_mag, f_mag, size=(len(env_ids_np), 1, 3)
        ).astype(np.float32)

    def _resolve_hand_side_availability(self) -> dict[str, bool]:
        if self.object_only:
            return {"left": False, "right": False}
        availability = getattr(self.original_demo, "hand_side_available", None)
        if availability is None:
            return {"left": True, "right": True}
        return {
            "left": bool(availability.get("left", False)),
            "right": bool(availability.get("right", False)),
        }

    def has_hand(self, side: str) -> bool:
        return bool(self.hand_side_available.get(side, False))

    @property
    def active_hand_sides(self) -> list[str]:
        return [side for side in ("left", "right") if self.has_hand(side)]

    def get_scene_cfg(self):
        cfg = self.cfg

        return dict(
            sim_options=gs.options.SimOptions(
                dt=cfg.dt,
                # dt=0.005,
                # dt=1 / 100.0,
                substeps=4,
                gravity=(0, 0, cfg.gravity),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_lookat=self.camera_lookat,
                camera_fov=cfg.camera_fov,
                max_FPS=cfg.max_FPS,
            ),
            show_viewer=cfg.show_viewer,
            vis_options=gs.options.VisOptions(
                # show_world_frame=True,
                show_world_frame=False,
            ),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=False,
                enable_joint_limit=cfg.enable_joint_limit,
                max_collision_pairs=cfg.max_collision_pairs,
                batch_links_info=cfg.batch_links_info,
                batch_dofs_info=cfg.batch_dofs_info,
            ),
            profiling_options=gs.options.ProfilingOptions(
                show_FPS=False,
            ),
        )

    def read_contact_sensors(self):
        if self.hand_sensors is None:
            return None

    def _compute_camera_lookat(self):
        # motion = self.original_demo.retargeted_motion
        # if motion is None:
        motion = self.original_demo.object_reference_motion

        obj_pos = motion.get("obj_pos", None)
        if obj_pos is None or len(obj_pos) == 0:
            return self.cfg.camera_look_at

        center = np.asarray(obj_pos).mean(axis=0)
        return tuple(float(v) for v in center.tolist())

    def initialize_cameras(self):
        cfg = self.cfg
        self.cam_theta = cfg.camera_theta
        lookat = self.camera_lookat
        self.cam = self.scene.add_camera(
            res=cfg.resolution,
            pos=(
                cfg.camera_radius * np.sin(self.cam_theta),
                cfg.camera_radius * np.cos(self.cam_theta),
                cfg.camera_ht,
            ),
            lookat=lookat,
            fov=cfg.camera_fov,
            GUI=False,
        )

    def video_recording_hook(self, timestep: int):
        """Simple recording hook that rotates camera and records clip.

        This mirrors prior behavior but centralizes it for easy experimentation.
        """
        # assert (
        #     timestep + 1
        # ) == self.env.env_step_counter, (
        #     f"timestep {timestep} != env step counter {self.env.env_step_counter}"
        # )

        save_dir = self.logs_dir
        save_freq = self.video_save_freq
        save_length = self.video_save_length

        if (save_freq <= 0) or (self.cam is None) or (not self.record):
            return
        if (
            (timestep % save_freq == 0)
            and (not self.is_recording)
            # and ((timestep // save_freq) > 0)
        ):
            # if (timestep % save_freq == 0) and (not self.is_recording):
            self.cam.start_recording()
            self.is_recording = True
        if self.is_recording and (timestep % save_freq == save_length):
            _start = timestep - save_length
            fname = f"{save_dir}/{_start}_{timestep}.mp4"
            self.cam.stop_recording(save_to_filename=fname, fps=30)
            self.is_recording = False
        if self.is_recording:
            dtheta = (2 * np.pi) / (save_length if save_length > 0 else 1)
            if not self.mov_cam:
                dtheta = 0.0
            self.cam_theta = (self.cam_theta + dtheta) % (2 * np.pi)
            self.cam.set_pose(
                pos=(
                    self.cfg.camera_radius * np.sin(self.cam_theta),
                    self.cfg.camera_radius * np.cos(self.cam_theta),
                    self.cfg.camera_ht,
                ),
                lookat=self.camera_lookat,
            )
            img, _, _, _ = self.cam.render(depth=True, segmentation=True)
            return img

    def verify_target_object_motion(self, mode="sim"):

        if not self.cfg.save_reference_video:
            return

        # available modes: 'sim', 'meshcat'
        # run only after env has been created
        reference = self.original_demo.object_reference_motion
        retargeted_motion = self.original_demo.retargeted_motion

        pos = torch.from_numpy(reference["obj_pos"]).to(self.env.device)
        quat = torch.from_numpy(reference["obj_quat"]).to(self.env.device)
        if reference["obj_arti"] is None:
            n_frames = pos.shape[0]
            n_obj_joints = len(self.scene_entities["object"].dof_map)
            arti = torch.zeros((n_frames, n_obj_joints), dtype=torch.float32).to(
                self.env.device
            )
        else:
            arti = torch.from_numpy(reference["obj_arti"]).to(self.env.device)
        if retargeted_motion:
            reset_mgr = self.env.reset_manager
            pos = reset_mgr.frame_pos.to(self.env.device)
            quat = reset_mgr.frame_quat.to(self.env.device)
            arti = (
                None
                if reset_mgr.frame_arti is None
                else reset_mgr.frame_arti.to(self.env.device)
            )

            hand_jvals = {
                side: getattr(reset_mgr, f"frame_{side}_jvals").to(self.env.device)
                for side in self.active_hand_sides
            }
        else:
            hand_jvals = {}

        if mode == "sim":
            asset: ArtObject = self.scene_entities["object"]

            while True:
                self.is_recording = True
                self.cam.start_recording()

                for frame_idx in range(self.total_steps):
                    _pos = pos[frame_idx]
                    _quat = quat[frame_idx]
                    _arti = None if arti is None else arti[frame_idx]

                    asset.set_state(
                        pos=_pos,
                        quat=_quat,
                        jpos=_arti,
                        env_idx=np.arange(self.env.num_envs),
                    )

                    for side in self.active_hand_sides:
                        robot: Hand = self.scene_entities[side]
                        hand_jpos = (
                            hand_jvals[side][frame_idx]
                            .unsqueeze(0)
                            .repeat(self.env.num_envs, 1)
                        )
                        robot.set_robot_state(
                            jpos=hand_jpos,
                            env_idx=np.arange(self.env.num_envs),
                        )

                    self.scene.step()

                    self.video_recording_hook(timestep=0)

                fname = f"{self.logs_dir}/reference_motion.mp4"
                self.cam.stop_recording(save_to_filename=fname, fps=30)
                self.is_recording = False

                # replay = input("Replay? (y/n): ")
                replay = "n"
                if replay.lower() != "y":
                    break

        elif mode == "meshcat":
            pass

    def step(self):
        if self.cfg.apply_random_force:
            # if self._env_random_force is None:
            self.resample_random_force(env_ids=None)
            link_idx = self.scene_entities["object"].object.base_link_idx
            self.scene.sim.rigid_solver.apply_links_external_force(
                force=self._env_random_force, links_idx=link_idx
            )

        self.scene.step()
