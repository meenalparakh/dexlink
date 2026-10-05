from __future__ import annotations

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from typing import TYPE_CHECKING

import numpy as np

from dexlink.src.rl.world.real_utils.control import prepended_sys_path, pushd, repo_root

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import RealCfg


@dataclass
class RealSenseCameraBundle:
    camera: object
    calibration: dict[str, np.ndarray]


def _load_realsense_camera_class():
    deployment_root = repo_root() / "real_deployment" / "deoxys_xhand"
    module_path = deployment_root / "utils" / "camera_utils.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Failed to find RealSense camera module at {module_path}."
        )
    module_spec = spec_from_file_location("dexlink_real_camera_utils", module_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_path}.")
    module = module_from_spec(module_spec)
    with prepended_sys_path(deployment_root, repo_root()), pushd(deployment_root):
        module_spec.loader.exec_module(module)
    if not hasattr(module, "RealSenseCamera"):
        raise AttributeError(f"Expected {module_path} to define RealSenseCamera.")
    return module.RealSenseCamera


def _camera_intrinsics_matrix(camera: object, cam: str) -> np.ndarray:
    intr = camera.get_intrinsics(cam)
    return np.array(
        [
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _get_camera_calibration(camera: object) -> dict[str, np.ndarray]:
    color_intrinsics = _camera_intrinsics_matrix(camera, cam="color")
    raw_depth_intrinsics = _camera_intrinsics_matrix(camera, cam="depth")
    aligned_depth_intrinsics = color_intrinsics.copy()
    return {
        "color_intrinsics": color_intrinsics,
        "raw_depth_intrinsics": raw_depth_intrinsics,
        "aligned_depth_intrinsics": aligned_depth_intrinsics,
    }


def _log_camera_resolution(camera: object, prefix: str) -> None:
    if not hasattr(camera, "get_stream_resolution"):
        return
    color_w, color_h = camera.get_stream_resolution("color")
    depth_w, depth_h = camera.get_stream_resolution("depth")
    print(
        f"{prefix} color resolution: {color_w}x{color_h}; "
        f"depth resolution: {depth_w}x{depth_h}"
    )


def _log_camera_info(camera: object, prefix: str) -> None:
    color_intr = camera.get_intrinsics("color")
    depth_intr = camera.get_intrinsics("depth")
    _log_camera_resolution(camera, prefix=prefix)
    print(
        f"{prefix} color intrinsics: "
        f"fx={color_intr['fx']:.3f}, fy={color_intr['fy']:.3f}, "
        f"cx={color_intr['cx']:.3f}, cy={color_intr['cy']:.3f}"
    )
    print(
        f"{prefix} depth intrinsics: "
        f"fx={depth_intr['fx']:.3f}, fy={depth_intr['fy']:.3f}, "
        f"cx={depth_intr['cx']:.3f}, cy={depth_intr['cy']:.3f}"
    )


def _make_camera(real_cfg: "RealCfg"):
    RealSenseCamera = _load_realsense_camera_class()
    if bool(real_cfg.camera_query_resolution):
        return RealSenseCamera(img_w=None, img_h=None, fps=int(real_cfg.camera_fps))
    return RealSenseCamera(
        img_w=int(real_cfg.camera_img_w),
        img_h=int(real_cfg.camera_img_h),
        fps=int(real_cfg.camera_fps),
    )


def create_realsense_camera_bundle(
    real_cfg: "RealCfg",
    *,
    log_prefix: str = "RealSense camera",
) -> RealSenseCameraBundle:
    camera = _make_camera(real_cfg)
    calibration = _get_camera_calibration(camera)
    _log_camera_info(camera, prefix=log_prefix)
    return RealSenseCameraBundle(camera=camera, calibration=calibration)


def create_realsense_camera(real_cfg: "RealCfg"):
    bundle = create_realsense_camera_bundle(real_cfg)
    return bundle.camera
