import os

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import numpy as np
from scipy.optimize import approx_fprime, minimize
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from dexlink.src.rl.sim.robot import get_hand_specific_cfg
from dexlink.src.retargeting.data.process_arctic import ArcticProcessor
from dexlink.utils.retargeting.path_utils import get_tmp_dir, process_urdf
from dexlink.utils.retargeting.vis_server import VisAnimator

os.environ["QT_QPA_PLATFORM"] = "offscreen"


class Optimizer:
    def __init__(self, original_demo_path: str, hand_cfg: dict, side: str = "right"):
        self.original_demo = ArcticProcessor(
            original_demo_path, target_object="arctic_laptop"
        )
        self.side = side

        start_frame = 0
        end_frame = 100

        self.side_acro = "l" if side == "left" else "r"
        self.side_acro_cap = "L" if side == "left" else "R"
        self.reference_joint_pos = self.original_demo.data["world_coord"][
            f"joints.{self.side}"
        ][
            start_frame:end_frame
        ]  # (n_frames, 21, 3)
        self.reference_kpts_pos = self.original_demo.data["world_coord"][
            f"verts.{self.side}"
        ][
            start_frame:end_frame
        ]  # (n_frames, n_kpts, 3)
        self.reference_wrist_pos = self.original_demo.data["params"][
            f"trans_{self.side_acro}"
        ][
            start_frame:end_frame
        ]  # (n_frames, 3)
        # in axis-angle format
        self.reference_wrist_rot = self.original_demo.data["params"][
            f"rot_{self.side_acro}"
        ][
            start_frame:end_frame
        ]  # (n_frames, 3)

        self.n_frames = self.reference_joint_pos.shape[0]

        # compute the joint positions relative to wrist
        # for t in range(self.n_frames):
        #     wrist_pos = self.reference_wrist_pos[t, :]  # (1, 3)
        #     wrist_aa = self.reference_wrist_rot[t, :]  # (1, 3)
        #     # convert to rotation matrix
        #     wrist_rot = R.from_rotvec(wrist_aa).as_matrix()  # (3, 3)
        #     wrist_pose = np.eye(4)
        #     wrist_pose[:3, :3] = wrist_rot
        #     wrist_pose[:3, 3] = wrist_pos
        #     # compute inverse
        #     wrist_pose_inv = np.linalg.inv(wrist_pose)
        #     # compute joint positions relative to wrist
        #     jpos = self.reference_joint_pos[t, :, :]  # (21, 3)
        #     jpos_homo = np.concatenate([jpos, np.ones((jpos.shape[0], 1))], axis=1)  # (21, 4)
        #     jpos_homo_wrist = wrist_pose_inv @ jpos_homo.T  # (4, 21)
        #     jpos_homo_wrist = jpos_homo_wrist[:3, :].T  # (21, 3)
        #     # compute_verts relative to wrist
        #     vpos = self.reference_kpts_pos[t, :, :]  # (n_kpts, 3)
        #     vpos_homo = np.concatenate([vpos, np.ones((vpos.shape[0], 1))], axis=1)  # (n_kpts, 4)
        #     vpos_homo_wrist = wrist_pose_inv @ vpos_homo.T  # (4, n_kpts)
        #     vpos_homo_wrist = vpos_homo_wrist[:3, :].T  # (n_kpts, 3)

        #     self.reference_kpts_pos[t, :, :] = vpos_homo_wrist
        #     self.reference_joint_pos[t, :, :] = jpos_homo_wrist

        self.hand_joint_map = {
            "wrist": 0,
            "thumb_tip": 16,
            "index_tip": 17,
            "middle_tip": 18,
            "ring_tip": 19,
            "little_tip": 20,
            "thumb_base": 14,
            "index_base": 1,
            "middle_base": 4,
            "ring_base": 10,
            "little_base": 7,
        }

        self.robot_link_map = {
            "wrist": "base_link",
            "thumb_tip": f"{self.side_acro_cap}_TH_Tip",
            "index_tip": f"{self.side_acro_cap}_FF_Tip",
            "middle_tip": f"{self.side_acro_cap}_MF_Tip",
            "ring_tip": f"{self.side_acro_cap}_RF_Tip",
            "little_tip": f"{self.side_acro_cap}_LF_Tip",
            "thumb_base": f"{self.side_acro_cap}_TH_L2",
            "index_base": f"{self.side_acro_cap}_FF_L1",
            "middle_base": f"{self.side_acro_cap}_MF_L1",
            "ring_base": f"{self.side_acro_cap}_RF_L1",
            "little_base": f"{self.side_acro_cap}_LF_L1",
        }

        self.pairs_to_optimize = [
            ("wrist", "thumb_tip", "normalize"),
            ("wrist", "index_tip", "normalize"),
            ("wrist", "middle_tip", "normalize"),
            ("wrist", "ring_tip", "normalize"),
            ("wrist", "little_tip", "normalize"),
            # now wrt to finger joints
            ("thumb_base", "thumb_tip", "normalize"),
            ("index_base", "index_tip", "normalize"),
            ("middle_base", "middle_tip", "normalize"),
            ("ring_base", "ring_tip", "normalize"),
            ("little_base", "little_tip", "normalize"),
            # thumb to fingers (absolute)
            ("thumb_tip", "index_tip", "unnormalize"),
            ("thumb_tip", "middle_tip", "unnormalize"),
            ("thumb_tip", "ring_tip", "unnormalize"),
            ("thumb_tip", "little_tip", "unnormalize"),
        ]

        # thumb, index, middle, ring, little joints in human hands
        # finger_tip_indices = [16, 17, 18, 19, 20]
        # finger_joint_indices = [14, 1, 4, 10, 7]
        # thumb_wrist_index = 13

        self.hand_cfg = hand_cfg

        hand_urdf = str(process_urdf(hand_cfg[self.side]["urdf_path"]))
        self.robot_inst = URDF.load(hand_urdf)

        self.robot_root_joints = [
            # f"{self.side_acro_cap}_forearm_{x}_link_joint" for x in ['tx', 'ty', 'tz', 'roll', 'pitch', 'yaw']
        ]
        # their values are always zero

        self.robot_joints = [
            j
            for j in self.robot_inst.joints
            if (j.joint_type != "fixed") and (j.name not in self.robot_root_joints)
        ]

        def _get_limit(limit_obj, attr, default):
            if limit_obj is None:
                return default
            val = getattr(limit_obj, attr, None)
            if val is None or np.isnan(val):
                return default
            return float(val)

        self.lower_bounds = np.array(
            [_get_limit(j.limit, "lower", -np.inf) for j in self.robot_joints],
            dtype=np.float64,
        )
        self.upper_bounds = np.array(
            [_get_limit(j.limit, "upper", np.inf) for j in self.robot_joints],
            dtype=np.float64,
        )
        self.optimized_direction_vectors = None
        self.frame_losses = []
        self.total_losses = []

        # to optimize: joint values for all frames
        self.reference_direction_vectors = (
            self.construct_hand_direction_vectors()
        )  # (n_pairs, n_frames, 3)
        self.to_optimize_joint_values = np.random.rand(
            self.n_frames, len(self.robot_joints)
        )

    def fk(self, jvals: np.ndarray) -> dict:
        """
        jvals: (n_joints,) numpy array of joint values
        returns: dict of link name to pose (4x4 numpy array)
        """
        jmap = {j.name: jvals[:, i] for i, j in enumerate(self.robot_joints)}
        jmap.update(
            {jname: np.zeros((jvals.shape[0],)) for jname in self.robot_root_joints}
        )
        jmap[f"{self.side_acro_cap}_forearm_yaw_link_joint"] += (
            np.pi / 2.0
        )  # add offset to roll joint
        link_poses = self.robot_inst.link_fk_batch(cfgs=jmap)

        link_poses = {link.name: poses for link, poses in link_poses.items()}
        return link_poses

    def compute_robot_direction_vectors(self, link_poses: dict):
        # as per the pairs to optimize, compute a vector for each pair
        vectors = []
        for joint1, joint2, mode in self.pairs_to_optimize:
            pos1 = link_poses[self.robot_link_map[joint1]][:, :3, 3]
            pos2 = link_poses[self.robot_link_map[joint2]][:, :3, 3]
            vec = pos2 - pos1
            if mode == "normalize":
                norm = np.linalg.norm(vec, axis=-1, keepdims=True)
                vec = vec / (norm + 1e-8)
            vectors.append(vec)

        return np.stack(vectors)

    def construct_hand_direction_vectors(self):
        # use the reference joint positions to construct direction vectors
        vectors = []
        for joint1, joint2, mode in self.pairs_to_optimize:
            pos1 = self.reference_joint_pos[:, self.hand_joint_map[joint1], :]
            pos2 = self.reference_joint_pos[:, self.hand_joint_map[joint2], :]
            vec = pos2 - pos1
            if mode == "normalize":
                vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
            vectors.append(vec)
        return np.array(vectors)

    def _reference_joint_world_positions(self) -> np.ndarray:
        world_positions = np.zeros_like(self.reference_joint_pos)
        for t in range(self.n_frames):
            rot = R.from_rotvec(self.reference_wrist_rot[t]).as_matrix()
            rel = self.reference_joint_pos[t]
            world_positions[t] = (rot @ rel.T).T + self.reference_wrist_pos[t]
        return world_positions

    def _robot_link_positions(self, joint_values: np.ndarray):
        fk_results = self.fk(joint_values)
        link_names = list(dict.fromkeys(self.robot_link_map.values()))
        link_positions = np.zeros((self.n_frames, len(link_names), 3))
        for idx, link_name in enumerate(link_names):
            link_positions[:, idx, :] = fk_results[link_name][:, :3, 3]
        return link_positions, link_names

    def _sample_within_bounds(self):
        low = np.where(np.isfinite(self.lower_bounds), self.lower_bounds, -1.0)
        high = np.where(np.isfinite(self.upper_bounds), self.upper_bounds, 1.0)
        return np.random.uniform(
            low[None, :], high[None, :], size=(self.n_frames, len(self.robot_joints))
        )

    def smoothness_loss(self, joint_values: np.ndarray):
        # joint_values: (n_frames, n_joints)
        assert joint_values.shape == (self.n_frames, len(self.robot_joints))
        if joint_values.shape[0] < 2:
            return 0.0
        diff = joint_values[1:, :] - joint_values[:-1, :]
        loss = np.mean(diff**2)
        return loss

    def _compute_direction_loss(self, robot_vectors, reference_vectors):
        # robot_vectors/reference_vectors: (n_pairs, n_frames, 3)
        diff = robot_vectors - reference_vectors
        return np.mean(np.sum(diff**2, axis=-1), axis=0)  # per frame avg over pairs

    def compute_wrist_loss(self, fk_results):
        # the robot wrist and the reference wrist should be close
        robot_wrist_pos = fk_results[self.robot_link_map["wrist"]][
            :, :3, 3
        ]  # (n_frames, 3)
        human_wrist_joint_pos = self.reference_joint_pos[
            :, self.hand_joint_map["wrist"], :
        ]  # (n_frames, 3)
        diff = robot_wrist_pos - human_wrist_joint_pos
        return np.mean(np.sum(diff**2, axis=-1), axis=0)  # per frame avg over pairs

    def _evaluate_loss(
        self, joint_values, reference_vectors, smoothness_weight, wrist_weight
    ):
        fk_results = self.fk(joint_values)
        robot_vecs = self.compute_robot_direction_vectors(fk_results)
        dir_loss_per_frame = self._compute_direction_loss(robot_vecs, reference_vectors)
        smooth_loss = smoothness_weight * self.smoothness_loss(joint_values)
        wrist_loss = wrist_weight * self.compute_wrist_loss(fk_results)
        total_loss = dir_loss_per_frame.mean() + smooth_loss + wrist_loss
        return total_loss, dir_loss_per_frame, smooth_loss, robot_vecs

    def optimize_scipy(
        self,
        smoothness_weight: float = 5e-2,
        wrist_weight: float = 1.0,
        method: str = "Powell",
        num_warm_starts: int = 5,
        warm_start_noise: float = 0.05,
        options: dict | None = None,
        verbose: bool = True,
    ):
        """Gradient-free optimization variant using SciPy's minimize."""

        n_frames = self.n_frames
        n_joints = len(self.robot_joints)
        reference_vecs = self.reference_direction_vectors

        init = self._warm_start_initialization(
            reference_vecs,
            smoothness_weight,
            wrist_weight=wrist_weight,
            num_candidates=num_warm_starts,
            noise_scale=warm_start_noise,
        )
        init = np.clip(init, self.lower_bounds, self.upper_bounds)

        bounds = None
        if method in {"L-BFGS-B", "TNC", "SLSQP"}:
            bounds = []
            for _ in range(n_frames):
                for low, high in zip(self.lower_bounds, self.upper_bounds):
                    bounds.append(
                        (
                            low if np.isfinite(low) else None,
                            high if np.isfinite(high) else None,
                        )
                    )

        self.scipy_objective_history = []

        def objective(flat_vals):
            joint_vals = flat_vals.reshape(n_frames, n_joints)
            joint_vals = np.clip(joint_vals, self.lower_bounds, self.upper_bounds)
            total, dir_loss, smooth_loss, robot_vecs = self._evaluate_loss(
                joint_vals, reference_vecs, smoothness_weight, wrist_weight
            )
            self.scipy_objective_history.append(total)
            self._scipy_last = (joint_vals.copy(), dir_loss, smooth_loss, robot_vecs)
            if verbose and len(self.scipy_objective_history) % 10 == 0:
                print(
                    f"[SciPy-{method}] eval {len(self.scipy_objective_history)} loss={total:.6f}"
                )
            return total

        res = minimize(
            objective,
            init.ravel(),
            method=method,
            bounds=bounds,
            options=options or {},
        )

        final_vals = res.x.reshape(n_frames, n_joints)
        final_vals = np.clip(final_vals, self.lower_bounds, self.upper_bounds)
        self.to_optimize_joint_values = final_vals
        self.optimized_direction_vectors = self._scipy_last[3]
        self.total_losses = self.scipy_objective_history
        self.frame_losses = [self._scipy_last[1]]
        self.smoothness_losses = [self._scipy_last[2]]
        return final_vals, res

    def _warm_start_initialization(
        self,
        reference_vecs,
        smoothness_weight,
        wrist_weight,
        num_candidates: int = 5,
        noise_scale: float = 0.05,
    ) -> np.ndarray:
        candidates = []
        bounds_span = self.upper_bounds - self.lower_bounds
        bounds_span = np.where(np.isfinite(bounds_span), bounds_span, 1.0)

        # default_qpos = np.array(
        #     self.hand_cfg[self.side].get("default_qpos", []), dtype=np.float64
        # )
        # if default_qpos.shape[0] == n_joints:
        #     base = np.tile(default_qpos[None, :], (self.n_frames, 1))
        #     candidates.append(base)
        #     if noise_scale > 0:
        #         noise = noise_scale * np.random.randn(self.n_frames, n_joints) * bounds_span
        #         candidates.append(base + noise)

        candidates.append(self.to_optimize_joint_values.copy())

        while len(candidates) < max(1, num_candidates):
            candidates.append(self._sample_within_bounds())

        best_loss = np.inf
        best_candidate = None
        self.warm_start_losses = []
        for cand in candidates:
            cand = np.clip(cand, self.lower_bounds, self.upper_bounds)
            loss, _, _, _ = self._evaluate_loss(
                cand, reference_vecs, smoothness_weight, wrist_weight
            )
            self.warm_start_losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_candidate = cand

        if best_candidate is None:
            best_candidate = candidates[0]

        return best_candidate.copy()

    def optimize(
        self,
        n_iters: int = 500,
        lr: float = 0.05,
        smoothness_weight: float = 5e-2,
        wrist_weight: float = 1.0,
        verbose: bool = True,
        num_warm_starts: int = 5,
        warm_start_noise: float = 0.05,
    ):
        n_frames = self.n_frames
        n_joints = len(self.robot_joints)
        reference_vecs = self.reference_direction_vectors
        joint_values = self._warm_start_initialization(
            reference_vecs,
            smoothness_weight,
            wrist_weight=wrist_weight,
            num_candidates=num_warm_starts,
            noise_scale=warm_start_noise,
        )
        best_loss = np.inf
        best_joint_values = joint_values.copy()
        self.total_losses = []
        self.frame_losses = []
        self.smoothness_losses = []

        def loss_wrapper(flat_joint_vals):
            reshaped = flat_joint_vals.reshape(n_frames, n_joints)
            total, _, _, _ = self._evaluate_loss(
                reshaped, reference_vecs, smoothness_weight, wrist_weight=wrist_weight
            )
            return total

        for it in range(n_iters):
            total_loss, dir_loss_per_frame, smooth_loss, robot_vecs = (
                self._evaluate_loss(
                    joint_values,
                    reference_vecs,
                    smoothness_weight,
                    wrist_weight=wrist_weight,
                )
            )

            self.total_losses.append(total_loss)
            self.frame_losses.append(dir_loss_per_frame)
            self.smoothness_losses.append(smooth_loss)

            if total_loss < best_loss:
                best_loss = total_loss
                best_joint_values = joint_values.copy()
                self.optimized_direction_vectors = robot_vecs.copy()

            grads_flat = approx_fprime(joint_values.ravel(), loss_wrapper, 1e-3)
            grads = grads_flat.reshape(n_frames, n_joints)

            joint_values -= lr * grads
            joint_values = np.clip(joint_values, self.lower_bounds, self.upper_bounds)

            if verbose and (it % max(1, n_iters // 10) == 0 or it == n_iters - 1):
                print(
                    f"Iter {it + 1}/{n_iters}: loss={total_loss:.6f}, smooth={smooth_loss:.6f}"
                )

        self.to_optimize_joint_values = best_joint_values
        return best_joint_values

    def visualize_optimized_results(self):
        if self.optimized_direction_vectors is None:
            raise RuntimeError(
                "Run optimize() before calling visualize_optimized_results()."
            )

        if plt is None:
            raise RuntimeError("matplotlib is required for visualize_optimized_results")

        plt.figure(figsize=(10, 4))
        plt.plot(
            np.arange(len(self.total_losses)), self.total_losses, label="total loss"
        )
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Optimization Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(10, 4))
        for idx, pair in enumerate(self.pairs_to_optimize):
            pair_loss = np.mean(
                (
                    self.optimized_direction_vectors[idx]
                    - self.reference_direction_vectors[idx]
                )
                ** 2,
                axis=-1,
            )
            plt.plot(pair_loss, label=f"{pair[0]}->{pair[1]} {pair[2]}")
        plt.xlabel("Frame")
        plt.ylabel("MSE")
        plt.title("Direction Error per Pair")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(get_tmp_dir() / "optimization_loss_curve.png")

    def visualize_with_animator(
        self, joint_values: np.ndarray | None = None, port_vis: int = 6000
    ):
        if joint_values is None:
            joint_values = self.to_optimize_joint_values

        if joint_values.shape != (self.n_frames, len(self.robot_joints)):
            raise ValueError(
                f"joint_values should have shape {(self.n_frames, len(self.robot_joints))}, got {joint_values.shape}"
            )

        # human_positions = self._reference_joint_world_positions()
        # use relative positions to wrist
        human_positions = self.reference_kpts_pos
        robot_positions, _ = self._robot_link_positions(joint_values)
        robot_joint_dict = [
            {j.name: joint_values[t, i] for i, j in enumerate(self.robot_joints)}
            for t in range(self.n_frames)
        ]
        # # add root joints
        # for t in range(self.n_frames):
        #     for jname in self.robot_root_joints:
        #         robot_joint_dict[t][jname] = 0.0
        #     robot_joint_dict[t][f"{self.side_acro_cap}_forearm_yaw_link_joint"] += (
        #         np.pi / 2.0
        #     )  # add offset to roll joint

        human_color = np.array([1.0, 0.5, 0.2])
        robot_color = np.array([0.2, 0.6, 1.0])

        vis = VisAnimator()
        vis.initialize_points(
            mesh_vertices=human_positions[0],
            mesh_colors=np.tile(human_color[None, :], (human_positions.shape[1], 1)),
            posed=False,
            num_points=human_positions.shape[1],
            name="reference/human",
        )
        vis.initialize_urdf(
            urdf_inst=self.robot_inst,
            name="reference/robot_urdf",
            mesh_type="visual",
        )
        vis.initialize_points(
            mesh_vertices=robot_positions[0],
            mesh_colors=np.tile(robot_color[None, :], (robot_positions.shape[1], 1)),
            posed=False,
            num_points=robot_positions.shape[1],
            name="reference/robot",
        )
        vis.initialize()

        for frame_idx in range(self.n_frames):
            vis.add_frame(
                pts_dict={
                    "reference/human": human_positions[frame_idx],
                    "reference/robot": robot_positions[frame_idx],
                },
                urdf_dict={
                    "reference/robot_urdf": {
                        "pos": np.zeros(3),
                        "rot": np.eye(3),
                        "joints": robot_joint_dict[frame_idx],
                    }
                },
            )

        vis.play()


if __name__ == "__main__":
    from dexlink.utils.retargeting.path_utils import get_external_data_dir

    original_demo_path: str = str(
        get_external_data_dir()
        / "original_processed_verts"
        / "seqs"
        / "s01"
        / "laptop_use_02.npy"
    )
    hand_cfg = get_hand_specific_cfg("inspire_hand")

    optimizer = Optimizer(original_demo_path, hand_cfg, side="right")
    # best_jvals = optimizer.optimize(n_iters=1000, lr=0.01, smoothness_weight=1e-3, verbose=True, num_warm_starts=100)
    # optimizer.visualize_optimized_results()

    # using scipy optimizer
    # best_jvals, res = optimizer.optimize_scipy(
    #     smoothness_weight=0.5,
    #     wrist_weight=1.0,
    #     # method="L-BFGS-B",
    #     method="Powell",
    #     num_warm_starts=50,
    #     warm_start_noise=0.1,
    #     options={"maxiter": 100, "disp": False},
    #     verbose=True,
    # )

    # # dump the best joint values
    # np.save(get_tmp_dir() / "optimized_hand_joint_values.npy", best_jvals)

    # load the joint values and visualize
    best_jvals = np.load(get_tmp_dir() / "optimized_hand_joint_values.npy")
    optimizer.visualize_with_animator(joint_values=best_jvals)
