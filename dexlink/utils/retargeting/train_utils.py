import random

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def merge_configs(*parts):
    cfg = {}
    for part in parts:
        cfg.update(part)
    return cfg


def run_meta(name: str, group: str | None = None):
    return {"logger.runname": name, "logger.group": group or name}


def add_noise_to_jpos(jpos, noise_scale, wrist_idx):
    # jpos are (21, 3) arrays, denoting xyz of the finger joints.
    # the noise is added as follows:
    # - two types of noise:
    # - (a) hand translation noise: noise_scale[0]
    # - (b) wrist rotation noise: noise_scale[1]
    # - first center the jpos by the wrist position, then compute the rotation matrix from the small noise
    #   and apply the transformation to the centered jpos, then add back the wrist position.
    #   and the same translation noise to all joints.
    if len(noise_scale) != 2:
        raise ValueError(f"Expected noise_scale length 2, got {len(noise_scale)}")

    trans_scale, rot_scale = float(noise_scale[0]), float(noise_scale[1])

    if torch.is_tensor(jpos):
        if jpos.shape[-1] != 3:
            raise ValueError(f"Expected last dimension 3, got {jpos.shape}")
        if not (-jpos.shape[-2] <= wrist_idx < jpos.shape[-2]):
            raise IndexError(
                f"wrist_idx {wrist_idx} is out of bounds for shape {jpos.shape}"
            )

        wrist = jpos[..., wrist_idx : wrist_idx + 1, :]  # (..., 1, 3)
        centered = jpos - wrist

        orig_shape = centered.shape
        flat = centered.reshape(-1, orig_shape[-2], 3)  # (B, J, 3)
        B = flat.shape[0]
        device = jpos.device
        dtype = jpos.dtype

        rotvec = torch.randn(B, 3, device=device, dtype=dtype) * rot_scale
        theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp_min(1e-9)
        axis = rotvec / theta

        kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
        K = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        K[:, 0, 1] = -kz
        K[:, 0, 2] = ky
        K[:, 1, 0] = kz
        K[:, 1, 2] = -kx
        K[:, 2, 0] = -ky
        K[:, 2, 1] = kx

        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        sin_t = torch.sin(theta).unsqueeze(-1)
        cos_t = torch.cos(theta).unsqueeze(-1)
        rotm = eye + sin_t * K + (1.0 - cos_t) * (K @ K)

        rotated = flat @ rotm.transpose(1, 2)
        rotated = rotated.reshape(orig_shape)

        trans = (torch.randn(B, 3, device=device, dtype=dtype) * trans_scale).reshape(
            orig_shape[:-2] + (1, 3)
        )

        return rotated + wrist + trans

    jpos = np.asarray(jpos)
    if jpos.shape[-1] != 3:
        raise ValueError(f"Expected last dimension 3, got {jpos.shape}")
    if not (-jpos.shape[-2] <= wrist_idx < jpos.shape[-2]):
        raise IndexError(
            f"wrist_idx {wrist_idx} is out of bounds for shape {jpos.shape}"
        )

    wrist = jpos[..., wrist_idx : wrist_idx + 1, :]  # (..., 1, 3)
    centered = jpos - wrist

    orig_shape = centered.shape
    flat = centered.reshape(-1, orig_shape[-2], 3)  # (B, J, 3)
    B = flat.shape[0]

    rotvec = np.random.randn(B, 3) * rot_scale
    rotm = R.from_rotvec(rotvec).as_matrix()  # (B, 3, 3)
    rotated = flat @ np.swapaxes(rotm, -1, -2)
    rotated = rotated.reshape(orig_shape)

    trans = (np.random.randn(B, 3) * trans_scale).reshape(orig_shape[:-2] + (1, 3))

    return rotated + wrist + trans


if __name__ == "__main__":
    from pathlib import Path

    from dexlink.src.retargeting.data.process_arctic import get_processor_cfg
    from dexlink.utils.retargeting.path_utils import get_external_data_dir
    from dexlink.utils.retargeting.vis_server import VizServer

    set_seed(0)

    # Demo ARCTIC file (hardcoded for quick local debugging).
    demo_path = (
        get_external_data_dir()
        / "original_processed_verts"
        / "seqs"
        / "s01"
        / "laptop_use_02.npy"
    )
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo path not found: {demo_path}")

    target_object = "arctic_laptop"
    processor_cfg = get_processor_cfg(target_object)
    processor_cls = processor_cfg["processor_cls"]
    reference_urdf = processor_cfg["reference_urdf"]
    target_urdf = processor_cfg["target_urdf_path"]

    processor = processor_cls(
        str(demo_path),
        target_object,
        start_timestep=0,
        end_timestep=-1,
        skip=1,
        reference_urdf=reference_urdf,
        target_urdf=target_urdf,
        retargeted_results=None,
    )

    # Pull one frame of real ARCTIC hand joints as (21, 3) jpos.
    frame_idx = min(100, processor.n_frames - 1)
    jpos = np.asarray(processor.world_coord["joints.left"][frame_idx], dtype=np.float32)
    wrist_idx = processor.hand_joint_map["wrist"]
    print(
        f"Loaded jpos from {Path(demo_path).name}, frame={frame_idx}, shape={jpos.shape}"
    )

    vis = VizServer(port_vis=6000)
    vis.view_pcd(
        jpos,
        colors=np.tile(np.array([[0.1, 0.3, 0.9]]), (jpos.shape[0], 1)),
        name="jpos_original",
        size=0.01,
    )
    wrist_original = jpos[wrist_idx : wrist_idx + 1]
    vis.view_pcd(
        wrist_original,
        colors=np.array([[0.0, 1.0, 1.0]]),
        name="wrist_original",
        size=0.03,
    )

    for _ in range(10):
        noisy_jpos = add_noise_to_jpos(
            jpos=jpos,
            noise_scale=(0.02, 0.1),  # (translation_std_m, rotation_std_rad)
            wrist_idx=wrist_idx,
        )

        vis.view_pcd(
            noisy_jpos,
            colors=np.tile(np.array([[0.9, 0.2, 0.2]]), (noisy_jpos.shape[0], 1)),
            name="jpos_noisy",
            size=0.01,
        )
        wrist_noisy = noisy_jpos[wrist_idx : wrist_idx + 1]

        vis.view_pcd(
            wrist_noisy,
            colors=np.array([[1.0, 1.0, 0.0]]),
            name="wrist_noisy",
            size=0.03,
        )
        input(
            "Visualizing original (blue) vs noisy (red) joints on meshcat (port 6000). Press Enter to exit..."
        )
