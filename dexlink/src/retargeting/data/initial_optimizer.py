from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dexlink.src.rl.sim.robot import Hand
from dexlink.utils.retargeting.math_utils import (
    matrix_from_quat,
    quat_error_magnitude,
    quat_mul,
)
from dexlink.utils.retargeting.mesh_utils import compute_bps_efficient, pcd_distance
from dexlink.utils.retargeting.path_utils import get_tmp_dir
from dexlink.utils.retargeting.vis_server import VisAnimator

if TYPE_CHECKING:
    from dexlink.src.rl.sim.env import Env


class InitialOptimizer:
    """
    Simple Optimization for obtaining good initialization ///////
    """

    def __init__(self, env: Env):
        self.env = env
        self.cmd_mgr = env.commands_manager
        self.rew_mgr = env.reward_manager
        self.disc_mgr = (
            env.disc_observation_manager
            if getattr(env, "use_discriminator", False)
            else None
        )
        self.obs_mgr = env.observation_manager
        self.device = env.device
        self.demo = env.scene_manager.original_demo
        self._params: Dict[str, torch.Tensor] = {}
        self._initialized = False

        self.ref_left = None
        self.ref_right = None
        self.ref_obj_pos = None
        self.ref_obj_quat = None
        self.ref_obj_arti = None
        self.ref_obj_pcd = None
        self.ref_left_bps = None
        self.ref_right_bps = None
        self.ref_disc = None
        self.horizon = None
        self.start_idx = 0
        self.dt = 1.0
        self.fk_consistency_weight = 1.0
        self.hand_cfg_weight = 1.0
        self.loss_weights = {
            "hand_left": 1.0,
            "hand_right": 1.0,
            "left_fk_consistency": self.fk_consistency_weight,
            "right_fk_consistency": self.fk_consistency_weight,
            "left_hand_cfg": self.hand_cfg_weight,
            "right_hand_cfg": self.hand_cfg_weight,
            "obj_pose": 1.0,
            "obj_pcd": 1.0,
            "hand_obj_left": 1.0,
            "hand_obj_right": 1.0,
            "disc": 1.0,
        }

        # Reuse the same pair set as the RL reward when available.
        self.hand_cfg_pairs = getattr(self.rew_mgr, "pairs_to_optimize", None) or [
            ("wrist", "thumb_tip", "normalize"),
            ("wrist", "index_tip", "normalize"),
            ("wrist", "middle_tip", "normalize"),
            ("wrist", "ring_tip", "normalize"),
            ("wrist", "little_tip", "normalize"),
            ("thumb_base", "thumb_tip", "normalize"),
            ("index_base", "index_tip", "normalize"),
            ("middle_base", "middle_tip", "normalize"),
            ("ring_base", "ring_tip", "normalize"),
            ("little_base", "little_tip", "normalize"),
            ("thumb_tip", "index_tip", "unnormalize"),
            ("thumb_tip", "middle_tip", "unnormalize"),
            ("thumb_tip", "ring_tip", "unnormalize"),
            ("thumb_tip", "little_tip", "unnormalize"),
        ]

    def initialize_parameters(self):
        """
        1. start hand config, corresponding to the first frame
        2. joint velocities corresponding to each frame transition
        3. object initial position, orientation, articulation
        4. object linear and angular velocity and articulation velocity
        """
        # the reference demo is already truncated with cfg.start_timestep and cfg.end_timestep
        start_idx = 0
        horizon = self.env.scene_manager.total_steps
        self._initialize_parameters(start_idx=start_idx, horizon=horizon)

    def forward_simulation(self):
        """
        simple forward simulation to obtain the observations
        not the real env simulation

        use the velocities to compute the configurations for all time steps
        starting from the initial configurations

        """
        if not self._initialized:
            self.initialize_parameters()

        traj = {}
        traj["obj_pos"] = self._integrate(
            self._params["obj_pos0"], self._params["obj_linear"]
        )
        traj["obj_arti"] = self._integrate(
            self._params["obj_arti0"], self._params["obj_arti_vel"]
        )
        traj["obj_quat"] = self._integrate_quaternions(
            self._params["obj_quat0"], self._params["obj_angular"]
        )

        if self.ref_left is not None:
            traj["left_jpos"] = self._integrate(
                self._params["left_jpos0"], self._params["left_jpos_vel"]
            )
            left_hand: Hand = self.env.scene_manager.scene_entities["left"]
            traj["left_link_pos"] = self._integrate(
                self._params["left_link_pos0"], self._params["left_link_pos_vel"]
            )
            traj["left_link_pos_fk"] = self._robot_link_positions_from_jpos(
                traj["left_jpos"], left_hand
            )
        if self.ref_right is not None:
            traj["right_jpos"] = self._integrate(
                self._params["right_jpos0"], self._params["right_jpos_vel"]
            )
            right_hand: Hand = self.env.scene_manager.scene_entities["right"]
            traj["right_link_pos"] = self._integrate(
                self._params["right_link_pos0"], self._params["right_link_pos_vel"]
            )
            traj["right_link_pos_fk"] = self._robot_link_positions_from_jpos(
                traj["right_jpos"], right_hand
            )
        return traj

    def compute_loss(
        self,
        enabled_losses: set[str] | None = None,
        loss_weights: dict[str, float] | None = None,
    ):
        """
        compute the following losses
        1. hand-robot configuration matching loss (using important hand keypoints)
        2. hand-object relative bps loss
        3. object bps representation matching loss
        4. discriminator feature loss (that is, compute disc features, and directly
           match them to the demo features, instead of using the network to compute logits)
        """
        traj = self.forward_simulation()
        losses = {}

        if self.ref_left is not None:
            losses["hand_left"] = F.mse_loss(traj["left_link_pos"], self.ref_left)
            losses["left_fk_consistency"] = self.fk_consistency_weight * F.mse_loss(
                traj["left_link_pos"], traj["left_link_pos_fk"]
            )
            left_hand: Hand = self.env.scene_manager.scene_entities["left"]
            losses["left_hand_cfg"] = (
                self.hand_cfg_weight
                * self._hand_cfg_similarity_loss(
                    pred_pos=traj["left_link_pos"],
                    ref_pos=self.ref_left,
                    joint_names=left_hand.human_joint_names,
                )
            )
        if self.ref_right is not None:
            losses["hand_right"] = F.mse_loss(traj["right_link_pos"], self.ref_right)
            losses["right_fk_consistency"] = self.fk_consistency_weight * F.mse_loss(
                traj["right_link_pos"], traj["right_link_pos_fk"]
            )
            right_hand: Hand = self.env.scene_manager.scene_entities["right"]
            losses["right_hand_cfg"] = (
                self.hand_cfg_weight
                * self._hand_cfg_similarity_loss(
                    pred_pos=traj["right_link_pos"],
                    ref_pos=self.ref_right,
                    joint_names=right_hand.human_joint_names,
                )
            )

        pos_loss = F.mse_loss(traj["obj_pos"], self.ref_obj_pos)
        arti_loss = F.mse_loss(traj["obj_arti"], self.ref_obj_arti)
        pred_quat = F.normalize(traj["obj_quat"], dim=-1)
        ref_quat = F.normalize(self.ref_obj_quat, dim=-1)
        quat_loss = quat_error_magnitude(pred_quat, ref_quat).mean()
        losses["obj_pose"] = pos_loss + arti_loss + quat_loss

        pred_pcd = self.obs_mgr.compute_object_pcd_obs(
            pos=traj["obj_pos"],
            quat=traj["obj_quat"],
            arti=traj["obj_arti"],
            normals=True,
        )
        losses["obj_pcd"] = pcd_distance(
            pred_pcd, self.ref_obj_pcd, metric="bps", bps_n_pts=128
        )

        if self.ref_left_bps is not None and "left_link_pos" in traj:
            pred_left_bps = compute_bps_efficient(pred_pcd, traj["left_link_pos_fk"])
            losses["hand_obj_left"] = F.mse_loss(pred_left_bps, self.ref_left_bps)
        if self.ref_right_bps is not None and "right_link_pos" in traj:
            pred_right_bps = compute_bps_efficient(pred_pcd, traj["right_link_pos_fk"])
            losses["hand_obj_right"] = F.mse_loss(pred_right_bps, self.ref_right_bps)

        if self.disc_mgr is not None and self.ref_disc is not None:
            pred_disc = self._compute_disc_features(traj)
            losses["disc"] = F.mse_loss(pred_disc, self.ref_disc)

        if enabled_losses is None:
            enabled_losses = set(losses.keys())

        weights = self.loss_weights.copy()
        if loss_weights:
            weights.update(loss_weights)

        total = torch.tensor(0.0, device=self.device)
        weighted = {}
        for name, val in losses.items():
            if name not in enabled_losses:
                continue
            weight = weights.get(name, 1.0)
            weighted[name] = val * weight
            total = total + weighted[name]

        weighted["total"] = total
        return weighted

    def _hand_cfg_similarity_loss(
        self, pred_pos: torch.Tensor, ref_pos: torch.Tensor, joint_names: list[str]
    ) -> torch.Tensor:
        """Direction-vector hand configuration loss (same pair set as RewardManager)."""
        # pred_pos/ref_pos: (T, K, 3) where K corresponds to joint_names order.
        name_to_idx = {name: i for i, name in enumerate(joint_names)}

        pred_vecs = []
        ref_vecs = []
        for j1, j2, mode in self.hand_cfg_pairs:
            if (j1 not in name_to_idx) or (j2 not in name_to_idx):
                continue
            i1 = name_to_idx[j1]
            i2 = name_to_idx[j2]
            pv = pred_pos[:, i2] - pred_pos[:, i1]
            rv = ref_pos[:, i2] - ref_pos[:, i1]
            if mode == "normalize":
                pv = pv / (pv.norm(dim=-1, keepdim=True) + 1e-8)
                rv = rv / (rv.norm(dim=-1, keepdim=True) + 1e-8)
            pred_vecs.append(pv)
            ref_vecs.append(rv)

        if len(pred_vecs) == 0:
            return torch.zeros((), device=pred_pos.device, dtype=pred_pos.dtype)

        pred_stack = torch.stack(pred_vecs, dim=0)  # (M, T, 3)
        ref_stack = torch.stack(ref_vecs, dim=0)  # (M, T, 3)
        return ((pred_stack - ref_stack) ** 2).sum(dim=-1).mean()

    def fk(self, jvals: np.ndarray, robot_joints, robot_inst) -> dict:
        """
        jvals: (n_joints,) numpy array of joint values
        returns: dict of link name to pose (4x4 numpy array)
        """
        jmap = {j.name: jvals[:, i] for i, j in enumerate(robot_joints)}
        link_poses = robot_inst.link_fk_batch(cfgs=jmap)

        link_poses = {link.name: poses for link, poses in link_poses.items()}
        return link_poses

    def _robot_link_positions_from_jpos(
        self, joint_values: torch.Tensor, hand: Hand
    ) -> torch.Tensor:
        """Non-differentiable URDF FK used only to regularize joint/link consistency."""
        jvals_np = joint_values.detach().cpu().numpy()
        fk_results = self.fk(jvals_np, hand.joints, hand.urdf_inst)
        link_names = hand.hand_mapped_link_names
        link_positions = np.zeros((self.horizon, len(link_names), 3), dtype=np.float32)
        for idx, link_name in enumerate(link_names):
            link_positions[:, idx, :] = fk_results[link_name][:, :3, 3]
        return torch.from_numpy(link_positions).to(self.device)

    def optimize_scipy(
        self,
        verbose: bool = True,
        print_every: int = 10,
        enabled_losses: set[str] | None = None,
        loss_weights: dict[str, float] | None = None,
    ):
        """
        main optimization loop, using scipy
        """
        try:
            from scipy.optimize import minimize
        except Exception as exc:
            raise RuntimeError("scipy is required for optimize_scipy") from exc

        if not self._initialized:
            self.initialize_parameters()

        flat, shapes = self._flatten_params()

        eval_count = 0

        def objective(flat_vals):
            nonlocal eval_count
            self._unflatten_params(flat_vals, shapes)
            with torch.no_grad():
                loss_val = float(
                    self.compute_loss(
                        enabled_losses=enabled_losses, loss_weights=loss_weights
                    )["total"]
                    .mean()
                    .cpu()
                    .item()
                )
            eval_count += 1
            if verbose and (eval_count % max(1, print_every) == 0):
                print(f"[scipy] eval={eval_count} loss={loss_val:.6f}")
            return loss_val

        options = {"disp": True} if verbose else {}
        res = minimize(objective, flat, method="L-BFGS-B", options=options)
        self._unflatten_params(res.x, shapes)
        return res

    def optimize_torch(
        self,
        n_steps: int = 200,
        lr: float = 1e-2,
        log_wandb: bool = False,
        wandb_project: str = "initial_optimizer",
        wandb_runname: str | None = None,
        wandb_group: str | None = "initial_optimizer",
        wandb_config: dict | None = None,
        log_every: int = 10,
        finish_wandb: bool = False,
        enabled_losses: set[str] | None = None,
        loss_weights: dict[str, float] | None = None,
    ):
        """
        main optimization loop, using pytorch optimizer
        """
        # batch size is the number of initial samples optimized
        # independently.
        if not self._initialized:
            self.initialize_parameters()

        params = [p for p in self._params.values() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)

        wandb = None
        if log_wandb:
            try:
                import wandb as _wandb  # type: ignore

                wandb = _wandb
            except Exception:
                wandb = None

        if wandb is not None:
            cfg = {
                "n_steps": n_steps,
                "lr": lr,
                "horizon": self.horizon,
                "dt": self.dt,
                "fk_consistency_weight": self.fk_consistency_weight,
                "hand_cfg_weight": getattr(self, "hand_cfg_weight", None),
                "device": str(self.device),
            }
            if wandb_config:
                cfg.update(wandb_config)
            if wandb.run is None:
                wandb.init(
                    project=wandb_project,
                    name=wandb_runname,
                    group=wandb_group,
                    config=cfg,
                    reinit=False,
                )

        best = float("inf")
        best_state = None
        for step in tqdm(range(n_steps)):
            optimizer.zero_grad(set_to_none=True)
            losses = self.compute_loss(
                enabled_losses=enabled_losses, loss_weights=loss_weights
            )
            loss = losses["total"]
            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.mean()
            loss.backward()
            optimizer.step()

            if loss.item() < best:
                best = loss.item()
                best_state = {k: v.detach().clone() for k, v in self._params.items()}

            if wandb is not None and (
                step % max(1, log_every) == 0 or step == n_steps - 1
            ):
                log_dict: dict[str, float] = {}
                for k, v in losses.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    if v.numel() == 1:
                        log_dict[f"loss/{k}"] = float(v.detach().cpu().item())
                    else:
                        vv = v.detach()
                        log_dict[f"loss/{k}_mean"] = float(vv.mean().cpu().item())
                        log_dict[f"loss/{k}_max"] = float(vv.max().cpu().item())
                log_dict["opt/step"] = step
                log_dict["opt/best_total"] = best
                wandb.log(log_dict, step=step)

        if best_state is not None:
            self._params = best_state

        if wandb is not None and finish_wandb and (wandb.run is not None):
            wandb.finish()
        return losses

    def visualize(self):
        losses = self.compute_loss()
        return {k: float(v.detach().cpu().item()) for k, v in losses.items()}

    def optimize_staged(
        self,
        stage_steps: tuple[int, int, int] = (200, 200, 200),
        stage_lrs: tuple[float, float, float] = (1e-2, 1e-2, 1e-2),
        log_wandb: bool = False,
        wandb_project: str = "initial_optimizer",
        wandb_runname: str | None = None,
        wandb_group: str | None = "initial_optimizer",
        log_every: int = 10,
    ):
        """Run three stages: hand config, object pose, then hand-object relative."""
        stage_names = ["hand_cfg", "obj_pose", "hand_obj_rel"]
        stages = [
            {
                "enabled": {
                    "left_hand_cfg",
                    "right_hand_cfg",
                    "left_fk_consistency",
                    "right_fk_consistency",
                }
            },
            {"enabled": {"obj_pose"}},
            {
                "enabled": {
                    "hand_obj_left",
                    "hand_obj_right",
                    "left_fk_consistency",
                    "right_fk_consistency",
                }
            },
        ]

        losses = None
        for idx, cfg in enumerate(stages):
            runname = (
                f"{wandb_runname}_stage{idx+1}_{stage_names[idx]}"
                if wandb_runname
                else None
            )
            losses = self.optimize_torch(
                n_steps=stage_steps[idx],
                lr=stage_lrs[idx],
                log_wandb=log_wandb,
                wandb_project=wandb_project,
                wandb_runname=runname,
                wandb_group=wandb_group,
                log_every=log_every,
                enabled_losses=cfg["enabled"],
            )
        return losses

    def save_optimized(self, path: str | None = None) -> str:
        """Save optimized parameters and trajectories to a .npz file."""
        traj = self.forward_simulation()
        data = {
            "obj_pos": traj["obj_pos"].detach().cpu().numpy(),
            "obj_quat": traj["obj_quat"].detach().cpu().numpy(),
            "obj_arti": traj["obj_arti"].detach().cpu().numpy(),
        }
        if "left_jpos" in traj:
            data["left_jpos"] = traj["left_jpos"].detach().cpu().numpy()
            data["left_link_pos"] = traj["left_link_pos"].detach().cpu().numpy()
        if "right_jpos" in traj:
            data["right_jpos"] = traj["right_jpos"].detach().cpu().numpy()
            data["right_link_pos"] = traj["right_link_pos"].detach().cpu().numpy()

        out_path = path or str(get_tmp_dir() / "initial_optimizer_state.npz")
        np.savez(out_path, **data)
        return out_path

    def visualize_with_animator(self, port_vis: int = 6000):
        """Visualize reference vs optimized hand link trajectories."""
        traj = self.forward_simulation()
        vis = VisAnimator(port_vis=port_vis)

        object_entity = self.env.scene_manager.scene_entities.get("object", None)
        if object_entity is not None:
            vis.initialize_urdf(
                urdf_inst=object_entity.urdf_inst,
                name="reference/object_urdf",
                mesh_type="visual",
            )
            vis.initialize_urdf(
                urdf_inst=object_entity.urdf_inst,
                name="optimized/object_urdf",
                mesh_type="visual",
            )

        if self.ref_left is not None:
            left_hand: Hand = self.env.scene_manager.scene_entities["left"]
            ref = self.ref_left.detach().cpu().numpy()
            pred = traj["left_link_pos"].detach().cpu().numpy()
            vis.initialize_points(
                mesh_vertices=ref[0],
                mesh_colors=np.tile(np.array([[1.0, 0.5, 0.2]]), (ref.shape[1], 1)),
                posed=False,
                num_points=ref.shape[1],
                name="reference/left_hand",
            )
            vis.initialize_points(
                mesh_vertices=pred[0],
                mesh_colors=np.tile(np.array([[0.2, 0.6, 1.0]]), (pred.shape[1], 1)),
                posed=False,
                num_points=pred.shape[1],
                name="optimized/left_hand",
            )
            vis.initialize_urdf(
                urdf_inst=left_hand.urdf_inst,
                name="optimized/left_urdf",
                mesh_type="visual",
            )

        if self.ref_right is not None:
            right_hand: Hand = self.env.scene_manager.scene_entities["right"]
            ref = self.ref_right.detach().cpu().numpy()
            pred = traj["right_link_pos"].detach().cpu().numpy()
            vis.initialize_points(
                mesh_vertices=ref[0],
                mesh_colors=np.tile(np.array([[1.0, 0.5, 0.2]]), (ref.shape[1], 1)),
                posed=False,
                num_points=ref.shape[1],
                name="reference/right_hand",
            )
            vis.initialize_points(
                mesh_vertices=pred[0],
                mesh_colors=np.tile(np.array([[0.2, 0.6, 1.0]]), (pred.shape[1], 1)),
                posed=False,
                num_points=pred.shape[1],
                name="optimized/right_hand",
            )
            vis.initialize_urdf(
                urdf_inst=right_hand.urdf_inst,
                name="optimized/right_urdf",
                mesh_type="visual",
            )

        vis.initialize()

        if object_entity is not None:
            ref_obj_pos = self.ref_obj_pos.detach().cpu().numpy()
            ref_obj_quat = self.ref_obj_quat.detach().cpu()
            ref_obj_rot = matrix_from_quat(ref_obj_quat).numpy()
            ref_obj_arti = self.ref_obj_arti.detach().cpu().numpy()

            opt_obj_pos = traj["obj_pos"].detach().cpu().numpy()
            opt_obj_quat = traj["obj_quat"].detach().cpu()
            opt_obj_rot = matrix_from_quat(opt_obj_quat).numpy()
            opt_obj_arti = traj["obj_arti"].detach().cpu().numpy()

            obj_joint_names = [j.name for j in object_entity.joints]

        for t in range(self.horizon):
            pts_dict = {}
            urdf_dict = {}

            if self.ref_left is not None:
                pts_dict["reference/left_hand"] = (
                    self.ref_left[t].detach().cpu().numpy()
                )
                pts_dict["optimized/left_hand"] = (
                    traj["left_link_pos"][t].detach().cpu().numpy()
                )
                left_hand = self.env.scene_manager.scene_entities["left"]
                left_jpos = traj["left_jpos"][t].detach().cpu().numpy()
                left_joint_dict = {
                    j.name: left_jpos[i] for i, j in enumerate(left_hand.joints)
                }
                urdf_dict["optimized/left_urdf"] = {
                    "pos": np.zeros(3),
                    "rot": np.eye(3),
                    "joints": left_joint_dict,
                }

            if self.ref_right is not None:
                pts_dict["reference/right_hand"] = (
                    self.ref_right[t].detach().cpu().numpy()
                )
                pts_dict["optimized/right_hand"] = (
                    traj["right_link_pos"][t].detach().cpu().numpy()
                )
                right_hand = self.env.scene_manager.scene_entities["right"]
                right_jpos = traj["right_jpos"][t].detach().cpu().numpy()
                right_joint_dict = {
                    j.name: right_jpos[i] for i, j in enumerate(right_hand.joints)
                }
                urdf_dict["optimized/right_urdf"] = {
                    "pos": np.zeros(3),
                    "rot": np.eye(3),
                    "joints": right_joint_dict,
                }

            if object_entity is not None:
                ref_joint_dict = {}
                opt_joint_dict = {}
                if len(obj_joint_names) > 0:
                    ref_joint_dict = {
                        name: float(ref_obj_arti[t, i])
                        for i, name in enumerate(obj_joint_names)
                    }
                    opt_joint_dict = {
                        name: float(opt_obj_arti[t, i])
                        for i, name in enumerate(obj_joint_names)
                    }
                urdf_dict["reference/object_urdf"] = {
                    "pos": ref_obj_pos[t],
                    "rot": ref_obj_rot[t],
                    "joints": ref_joint_dict,
                }
                urdf_dict["optimized/object_urdf"] = {
                    "pos": opt_obj_pos[t],
                    "rot": opt_obj_rot[t],
                    "joints": opt_joint_dict,
                }

            vis.add_frame(pts_dict=pts_dict, urdf_dict=urdf_dict)

        vis.play()

    def _initialize_parameters(self, start_idx: int, horizon: int):
        self.start_idx = start_idx
        self.horizon = horizon

        ref = self.demo
        obj = ref.object_reference_motion
        obj_pos = torch.from_numpy(obj["obj_pos"]).float()
        obj_quat = torch.from_numpy(obj["obj_quat"]).float()
        obj_arti = torch.from_numpy(obj["obj_arti"]).float()

        self.ref_obj_pos = obj_pos.to(self.device)
        self.ref_obj_quat = obj_quat.to(self.device)
        self.ref_obj_arti = obj_arti.to(self.device)

        # left = ref.world_coord.get("joints.left", None)
        # right = ref.world_coord.get("joints.right", None)
        if "left" in self.env.scene_manager.scene_entities:
            # arrange it as per the order of the robot env
            left_robot: Hand = self.env.scene_manager.scene_entities["left"]
            human_joint_names = left_robot.human_joint_names
            joint_indices = [ref.hand_joint_map[name] for name in human_joint_names]
            self.left_robot_inst = left_robot.urdf_inst
            self.left_joints = [
                j for j in self.left_robot_inst.joints if (j.joint_type != "fixed")
            ]
            left = ref.world_coord["joints.left"][:, joint_indices, :]
            self.ref_left = torch.from_numpy(left).float().to(self.device)

        if "right" in self.env.scene_manager.scene_entities:
            right_robot: Hand = self.env.scene_manager.scene_entities["right"]
            human_joint_names = right_robot.human_joint_names
            joint_indices = [ref.hand_joint_map[name] for name in human_joint_names]
            self.right_robot_inst = right_robot.urdf_inst
            self.right_joints = [
                j for j in self.right_robot_inst.joints if (j.joint_type != "fixed")
            ]
            right = ref.world_coord["joints.right"][:, joint_indices, :]
            self.ref_right = torch.from_numpy(right).float().to(self.device)

        command_pcd_idx = self.cmd_mgr.obs_command_terms.index("pcd")
        self.ref_obj_pcd = self.cmd_mgr.obs_commands_buffer[command_pcd_idx].reshape(
            self.horizon, -1, 6
        )
        self.ref_obj_pcd = self.ref_obj_pcd.to(self.device)
        # assert self.ref_obj_pcd.shape[1] == self.cmd_mgr.

        if self.ref_left is not None:
            self.ref_left_bps = compute_bps_efficient(self.ref_obj_pcd, self.ref_left)
        if self.ref_right is not None:
            self.ref_right_bps = compute_bps_efficient(self.ref_obj_pcd, self.ref_right)

        # if self.disc_mgr is not None:
        if False:  # for now disable discriminator loss
            self.ref_disc = self._compute_disc_features(
                {
                    "left_pos": self.ref_left,
                    "right_pos": self.ref_right,
                    "obj_pos": self.ref_obj_pos,
                    "obj_quat": self.ref_obj_quat,
                    "obj_arti": self.ref_obj_arti,
                }
            )

        self._params = {
            "obj_pos0": torch.nn.Parameter(
                torch.zeros_like(self.ref_obj_pos[:1].detach())
            ),
            "obj_linear": torch.nn.Parameter(
                torch.zeros((horizon - 1, 3), device=self.device)
            ),
            "obj_quat0": torch.nn.Parameter(
                torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device
                )
            ),
            "obj_angular": torch.nn.Parameter(
                torch.zeros((horizon - 1, 3), device=self.device)
            ),
            "obj_arti0": torch.nn.Parameter(
                torch.zeros_like(self.ref_obj_arti[:1].detach())
            ),
            "obj_arti_vel": torch.nn.Parameter(
                torch.zeros(
                    (horizon - 1, self.ref_obj_arti.shape[-1]), device=self.device
                )
            ),
        }

        if self.ref_left is not None:
            left_hand: Hand = self.env.scene_manager.scene_entities["left"]
            n_left_joints = len(left_hand.dof_map)
            n_left_links = self.ref_left.shape[1]
            self._params.update(
                {
                    "left_jpos0": torch.nn.Parameter(
                        torch.zeros((1, n_left_joints), device=self.device)
                    ),
                    "left_jpos_vel": torch.nn.Parameter(
                        torch.zeros((horizon - 1, n_left_joints), device=self.device)
                    ),
                    "left_link_pos0": torch.nn.Parameter(
                        self.ref_left[:1].detach().clone()
                    ),
                    "left_link_pos_vel": torch.nn.Parameter(
                        torch.zeros((horizon - 1, n_left_links, 3), device=self.device)
                    ),
                }
            )
        if self.ref_right is not None:
            right_hand: Hand = self.env.scene_manager.scene_entities["right"]
            n_right_joints = len(right_hand.dof_map)
            n_right_links = self.ref_right.shape[1]
            self._params.update(
                {
                    "right_jpos0": torch.nn.Parameter(
                        torch.zeros((1, n_right_joints), device=self.device)
                    ),
                    "right_jpos_vel": torch.nn.Parameter(
                        torch.zeros((horizon - 1, n_right_joints), device=self.device)
                    ),
                    "right_link_pos0": torch.nn.Parameter(
                        self.ref_right[:1].detach().clone()
                    ),
                    "right_link_pos_vel": torch.nn.Parameter(
                        torch.zeros((horizon - 1, n_right_links, 3), device=self.device)
                    ),
                }
            )

        self._initialized = True

    def _integrate(self, x0: torch.Tensor, vel: torch.Tensor):
        if vel.numel() == 0:
            return x0
        steps = torch.cumsum(vel * self.dt, dim=0)
        return torch.cat([x0, x0 + steps], dim=0)

    def _integrate_quaternions(
        self, quat0: torch.Tensor, angular_vel: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate quaternions using angular velocity via exponential map.
        quat0: initial quaternion [1, 4] in format [w, x, y, z]
        angular_vel: angular velocity vectors [T-1, 3]
        Returns: integrated quaternions [T, 4]
        """
        if angular_vel.numel() == 0:
            return quat0

        quats = [quat0[0]]  # Start with initial quaternion

        for t in range(angular_vel.shape[0]):
            omega = angular_vel[t]  # [3]
            omega_norm = torch.norm(omega)

            if omega_norm < 1e-6:
                # No rotation, quaternion stays the same
                quats.append(quats[-1])
            else:
                # Exponential map (wxyz): q_delta = [cos(||omega||*dt/2), sin(||omega||*dt/2) * omega/||omega||]
                angle = omega_norm * self.dt
                axis = omega / omega_norm
                half_angle = angle / 2.0
                q_delta = torch.cat(
                    [torch.cos(half_angle).unsqueeze(0), torch.sin(half_angle) * axis]
                )  # (4,)
                q_current = quats[-1].unsqueeze(0)  # (1, 4)
                q_new = quat_mul(q_delta.unsqueeze(0), q_current).squeeze(0)
                quats.append(F.normalize(q_new, dim=-1))

        return torch.stack(quats, dim=0)

    # Quaternion multiplication is handled by dexlink.utils.retargeting.math_utils.quat_mul (wxyz).

    def _compute_disc_features(self, traj: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.disc_mgr is None:
            return torch.zeros((self.horizon - 1, 1), device=self.device)

        named_obs = {
            "obj_pos": traj["obj_pos"],
            "obj_quat": traj["obj_quat"],
            "obj_arti": traj["obj_arti"],
        }
        if traj.get("left_link_pos", None) is not None:
            named_obs["left_lpos"] = traj["left_link_pos"]
        if traj.get("right_link_pos", None) is not None:
            named_obs["right_lpos"] = traj["right_link_pos"]

        disc_terms = []
        for t in range(1, self.horizon):
            current = {k: v[t : t + 1] for k, v in named_obs.items()}
            prev = {k: v[t - 1 : t] for k, v in named_obs.items()}
            disc = self.disc_mgr.compute_disc_obs(
                current_named_obs=current, prev_named_obs=prev
            )
            disc_terms.append(disc)
        return torch.cat(disc_terms, dim=0)

    def _flatten_params(self) -> Tuple[np.ndarray, Dict[str, Tuple[int, ...]]]:
        flat_list = []
        shapes = {}
        for k, v in self._params.items():
            shapes[k] = tuple(v.shape)
            flat_list.append(v.detach().cpu().numpy().reshape(-1))
        return np.concatenate(flat_list), shapes

    def _unflatten_params(
        self, flat_vals: np.ndarray, shapes: Dict[str, Tuple[int, ...]]
    ):
        offset = 0
        for k, shape in shapes.items():
            size = int(np.prod(shape))
            arr = flat_vals[offset : offset + size].reshape(shape)
            with torch.no_grad():
                self._params[k].copy_(torch.from_numpy(arr).to(self.device))
            offset += size

    def visualize_result(self):
        self.visualize_with_animator()
