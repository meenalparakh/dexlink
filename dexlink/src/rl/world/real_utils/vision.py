from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from dexlink.utils.retargeting.math_utils import depth_to_pointcloud


@dataclass
class StaticSceneCalibration:
    base_pose_cam: np.ndarray
    world_from_cam: np.ndarray
    table_plane_world: np.ndarray
    num_frames: int
    wrist_marker_pose_wrist: np.ndarray | None = None
    wrist_marker_delta_pose_wrist: np.ndarray | None = None


def get_aruco_dictionary(dict_name: str) -> cv2.aruco.Dictionary:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is not available in this environment.")
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"Unknown ArUco dictionary name: {dict_name!r}.")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def detect_marker_poses(
    *,
    color_image_rgb: np.ndarray,
    intrinsics: np.ndarray,
    aruco_dict: cv2.aruco.Dictionary,
    marker_sizes_m: dict[int, float],
) -> dict[int, np.ndarray]:
    if color_image_rgb.ndim != 3 or color_image_rgb.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB image of shape (H, W, 3), got {color_image_rgb.shape}."
        )
    gray = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2GRAY)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    solvepnp_flag = (
        cv2.SOLVEPNP_IPPE_SQUARE
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
        else cv2.SOLVEPNP_ITERATIVE
    )
    poses: dict[int, np.ndarray] = {}
    for marker_idx, marker_id in enumerate(ids.reshape(-1).tolist()):
        if int(marker_id) not in marker_sizes_m:
            continue
        half_size = float(marker_sizes_m[int(marker_id)]) / 2.0
        object_corners = np.array(
            [
                [-half_size, half_size, 0.0],
                [half_size, half_size, 0.0],
                [half_size, -half_size, 0.0],
                [-half_size, -half_size, 0.0],
            ],
            dtype=np.float32,
        )
        image_corners = corners[marker_idx].reshape(4, 2).astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(
            object_corners,
            image_corners,
            intrinsics,
            dist_coeffs,
            flags=solvepnp_flag,
        )
        if not success:
            continue
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.from_rotvec(rvec.reshape(3)).as_matrix().astype(np.float32)
        pose[:3, 3] = tvec.reshape(3).astype(np.float32)
        poses[int(marker_id)] = pose
    return poses


def make_object_mask(
    *,
    base_color: np.ndarray,
    base_depth: np.ndarray,
    current_color: np.ndarray,
    current_depth: np.ndarray,
    depth_delta_threshold_m: float,
    rgb_delta_threshold: float,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    if base_color.shape != current_color.shape:
        raise ValueError(
            "Base color and current color shape mismatch: "
            f"{base_color.shape} vs {current_color.shape}."
        )
    if base_depth.shape != current_depth.shape:
        raise ValueError(
            "Base depth and current depth shape mismatch: "
            f"{base_depth.shape} vs {current_depth.shape}."
        )
    valid_current = np.isfinite(current_depth) & (current_depth > min_depth_m)
    valid_current &= current_depth < max_depth_m
    valid_base = np.isfinite(base_depth) & (base_depth > min_depth_m)
    delta = base_depth - current_depth
    color_delta = np.linalg.norm(
        current_color.astype(np.float32) - base_color.astype(np.float32), axis=-1
    )
    depth_mask = valid_current & valid_base & (delta > depth_delta_threshold_m)
    color_mask = color_delta > rgb_delta_threshold
    return depth_mask | color_mask


def erode_object_mask(
    mask: np.ndarray, kernel_size: int, iterations: int
) -> np.ndarray:
    if iterations == 0:
        return mask.astype(bool)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8) * 255, kernel, iterations=iterations)
    return eroded > 0


def masked_point_cloud(
    *,
    depth_image: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    max_points: int,
) -> np.ndarray:
    if mask.shape != depth_image.shape:
        raise ValueError(f"Expected mask shape {depth_image.shape}, got {mask.shape}.")
    masked_depth = np.where(mask, depth_image, 0.0).astype(np.float32)
    points = depth_to_pointcloud(masked_depth, intrinsics)
    if points.shape[0] == 0:
        return points.astype(np.float32)
    if points.shape[0] > max_points:
        sample_idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[sample_idx]
    return points.astype(np.float32)


def masked_point_cloud_with_rgb(
    *,
    depth_image: np.ndarray,
    color_image_rgb: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if mask.shape != depth_image.shape:
        raise ValueError(f"Expected mask shape {depth_image.shape}, got {mask.shape}.")
    if color_image_rgb.ndim != 3 or color_image_rgb.shape[2] != 3:
        raise ValueError(
            f"Expected color_image_rgb shape (H, W, 3), got {color_image_rgb.shape}."
        )
    if color_image_rgb.shape[:2] != depth_image.shape:
        raise ValueError(
            "Expected color_image_rgb spatial shape to match depth_image: "
            f"{color_image_rgb.shape[:2]} vs {depth_image.shape}."
        )
    masked_depth = np.where(mask, depth_image, 0.0).astype(np.float32)
    points = depth_to_pointcloud(masked_depth, intrinsics)
    colors = color_image_rgb.reshape(-1, 3)[mask.reshape(-1)].astype(np.float32) / 255.0
    if points.shape[0] != colors.shape[0]:
        raise RuntimeError(
            "Point/color count mismatch after RGB-D point-cloud construction: "
            f"{points.shape[0]} points vs {colors.shape[0]} colors."
        )
    if points.shape[0] == 0:
        return points.astype(np.float32), colors.astype(np.float32)
    if points.shape[0] > max_points:
        sample_idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[sample_idx]
        colors = colors[sample_idx]
    return points.astype(np.float32), colors.astype(np.float32)


def full_point_cloud(
    *,
    depth_image: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    intrinsics: np.ndarray,
    max_points: int,
) -> np.ndarray:
    valid_mask = np.isfinite(depth_image) & (depth_image > min_depth_m)
    valid_mask &= depth_image < max_depth_m
    return masked_point_cloud(
        depth_image=depth_image,
        mask=valid_mask,
        intrinsics=intrinsics,
        max_points=max_points,
    )


def full_point_cloud_with_rgb(
    *,
    depth_image: np.ndarray,
    color_image_rgb: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    intrinsics: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid_mask = np.isfinite(depth_image) & (depth_image > min_depth_m)
    valid_mask &= depth_image < max_depth_m
    return masked_point_cloud_with_rgb(
        depth_image=depth_image,
        color_image_rgb=color_image_rgb,
        mask=valid_mask,
        intrinsics=intrinsics,
        max_points=max_points,
    )


def transform_points_homogeneous(
    points: np.ndarray, target_T_source: np.ndarray
) -> np.ndarray:
    if points.shape[0] == 0:
        return points.astype(np.float32)
    homogeneous = np.concatenate(
        [points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    transformed = (target_T_source @ homogeneous.T).T[:, :3]
    return transformed.astype(np.float32)


def filter_world_points_x_min(points: np.ndarray, x_min: float) -> np.ndarray:
    if points.shape[0] == 0:
        return points.astype(np.float32)
    return points[points[:, 0] >= x_min].astype(np.float32)


def fit_table_plane(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        raise ValueError("Cannot fit a table plane to an empty point cloud.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane_model, _ = pcd.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000,
    )
    plane = np.asarray(plane_model, dtype=np.float32)
    normal = plane[:3]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 0.0:
        raise ValueError("Estimated table plane had zero normal.")
    plane /= normal_norm
    if plane[2] < 0.0:
        plane *= -1.0
    if plane[2] < 0.9:
        raise ValueError(
            f"Estimated table plane normal is not close to +z: {plane[:3]}."
        )
    return plane


def horizontal_plane_from_z(z_height_world: float) -> np.ndarray:
    return np.array([0.0, 0.0, 1.0, -float(z_height_world)], dtype=np.float32)


def remove_points_near_or_below_plane(
    points: np.ndarray, plane_world: np.ndarray, height_threshold_m: float
) -> np.ndarray:
    if points.shape[0] == 0:
        return points.astype(np.float32)
    signed_distance = points @ plane_world[:3] + plane_world[3]
    return points[signed_distance >= height_threshold_m].astype(np.float32)


def remove_points_near_reference_pcd(
    *,
    points: np.ndarray,
    reference_points: np.ndarray,
    exclusion_radius_m: float,
) -> np.ndarray:
    if points.shape[0] == 0 or reference_points.shape[0] == 0:
        return points.astype(np.float32)
    query_pcd = o3d.geometry.PointCloud()
    query_pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    reference_pcd = o3d.geometry.PointCloud()
    reference_pcd.points = o3d.utility.Vector3dVector(
        reference_points.astype(np.float64)
    )
    distances = np.asarray(query_pcd.compute_point_cloud_distance(reference_pcd))
    return points[distances >= exclusion_radius_m].astype(np.float32)


def keep_largest_cluster(
    *,
    points: np.ndarray,
    eps_m: float,
    min_points: int,
) -> np.ndarray:
    if points.shape[0] == 0:
        return points.astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    labels = np.asarray(
        pcd.cluster_dbscan(eps=eps_m, min_points=min_points, print_progress=False)
    )
    valid = labels >= 0
    if not np.any(valid):
        return points.astype(np.float32)
    largest_label = int(np.bincount(labels[valid]).argmax())
    return points[labels == largest_label].astype(np.float32)


def desired_world_from_base_marker() -> np.ndarray:
    desired = np.eye(4, dtype=np.float32)
    desired[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return desired


def estimate_static_scene_calibration(
    *,
    base_poses_cam: list[np.ndarray],
    scene_pcds_cam: list[np.ndarray],
    world_x_min: float,
) -> StaticSceneCalibration:
    if len(base_poses_cam) == 0:
        raise ValueError("No base poses available for static scene calibration.")
    translations = np.stack([pose[:3, 3] for pose in base_poses_cam], axis=0)
    mean_translation = np.mean(translations, axis=0).astype(np.float32)
    rotations = R.from_matrix(
        np.stack([pose[:3, :3] for pose in base_poses_cam], axis=0)
    )
    mean_rotation = rotations.mean().as_matrix().astype(np.float32)
    base_pose_cam = np.eye(4, dtype=np.float32)
    base_pose_cam[:3, :3] = mean_rotation
    base_pose_cam[:3, 3] = mean_translation

    world_from_cam = (
        desired_world_from_base_marker() @ np.linalg.inv(base_pose_cam)
    ).astype(np.float32)

    table_plane_world = horizontal_plane_from_z(
        z_height_world=float(desired_world_from_base_marker()[2, 3])
    )
    return StaticSceneCalibration(
        base_pose_cam=base_pose_cam,
        world_from_cam=world_from_cam,
        table_plane_world=table_plane_world,
        num_frames=len(base_poses_cam),
    )
