from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


@dataclass
class WorldCalibrationObservation:
    frame_idx: int
    wrist_pose_base: np.ndarray
    marker_pose_cam: np.ndarray


@dataclass
class WorldCalibrationSolution:
    base_pose_cam: np.ndarray
    wrist_marker_pose_wrist: np.ndarray
    wrist_marker_delta_pose_wrist: np.ndarray
    per_observation_position_errors_m: np.ndarray
    per_observation_rotation_errors_deg: np.ndarray


class WorldCalibrationSolver:
    def __init__(
        self,
        *,
        initial_wrist_marker_pose_wrist: np.ndarray,
        marker_delta_translation_sigma_m: float,
        marker_delta_rotation_sigma_deg: float,
        marker_delta_regularization_weight: float = 1.0,
        translation_weight: float = 1.0,
        rotation_weight: float = 1.0,
    ) -> None:
        self.initial_wrist_marker_pose_wrist = np.asarray(
            initial_wrist_marker_pose_wrist, dtype=np.float64
        )
        if self.initial_wrist_marker_pose_wrist.shape != (4, 4):
            raise ValueError(
                "Expected initial_wrist_marker_pose_wrist shape (4, 4), got "
                f"{self.initial_wrist_marker_pose_wrist.shape}."
            )
        if marker_delta_translation_sigma_m <= 0.0:
            raise ValueError(
                "Expected marker_delta_translation_sigma_m > 0, got "
                f"{marker_delta_translation_sigma_m}."
            )
        if marker_delta_rotation_sigma_deg <= 0.0:
            raise ValueError(
                "Expected marker_delta_rotation_sigma_deg > 0, got "
                f"{marker_delta_rotation_sigma_deg}."
            )
        self.marker_delta_translation_sigma_m = float(marker_delta_translation_sigma_m)
        self.marker_delta_rotation_sigma_rad = np.deg2rad(
            float(marker_delta_rotation_sigma_deg)
        )
        if marker_delta_regularization_weight < 0.0:
            raise ValueError(
                "Expected marker_delta_regularization_weight >= 0, got "
                f"{marker_delta_regularization_weight}."
            )
        self.marker_delta_regularization_weight = float(
            marker_delta_regularization_weight
        )
        self.translation_weight = float(translation_weight)
        self.rotation_weight = float(rotation_weight)

    def solve(
        self,
        observations: list[WorldCalibrationObservation],
        *,
        initial_base_pose_cam: np.ndarray,
    ) -> WorldCalibrationSolution:
        if len(observations) == 0:
            raise ValueError("Expected at least one calibration observation.")
        initial_base_pose_cam = np.asarray(initial_base_pose_cam, dtype=np.float64)
        if initial_base_pose_cam.shape != (4, 4):
            raise ValueError(
                f"Expected initial_base_pose_cam shape (4, 4), got {initial_base_pose_cam.shape}."
            )
        x0 = np.concatenate(
            [
                self._pose_to_params(initial_base_pose_cam),
                np.zeros(6, dtype=np.float64),
            ]
        )
        result = least_squares(
            self._residuals,
            x0=x0,
            args=(observations,),
            method="trf",
        )
        if not result.success:
            raise RuntimeError(
                f"World calibration optimization failed: {result.message}"
            )
        base_pose_cam, delta_pose_wrist = self._unpack_solution(result.x)
        wrist_marker_pose_wrist = (
            delta_pose_wrist @ self.initial_wrist_marker_pose_wrist
        ).astype(np.float32)
        pos_errors: list[float] = []
        rot_errors_deg: list[float] = []
        for obs in observations:
            pred_marker_pose_cam = (
                base_pose_cam @ obs.wrist_pose_base @ wrist_marker_pose_wrist
            )
            pos_errors.append(
                float(
                    np.linalg.norm(
                        pred_marker_pose_cam[:3, 3] - obs.marker_pose_cam[:3, 3]
                    )
                )
            )
            rel_rot = R.from_matrix(obs.marker_pose_cam[:3, :3]).inv() * R.from_matrix(
                pred_marker_pose_cam[:3, :3]
            )
            rot_errors_deg.append(float(np.rad2deg(rel_rot.magnitude())))
        return WorldCalibrationSolution(
            base_pose_cam=base_pose_cam.astype(np.float32),
            wrist_marker_pose_wrist=wrist_marker_pose_wrist.astype(np.float32),
            wrist_marker_delta_pose_wrist=delta_pose_wrist.astype(np.float32),
            per_observation_position_errors_m=np.asarray(pos_errors, dtype=np.float32),
            per_observation_rotation_errors_deg=np.asarray(
                rot_errors_deg, dtype=np.float32
            ),
        )

    def predict_marker_pose_cam(
        self,
        *,
        base_pose_cam: np.ndarray,
        wrist_pose_base: np.ndarray,
        wrist_marker_pose_wrist: np.ndarray,
    ) -> np.ndarray:
        base_pose_cam = np.asarray(base_pose_cam, dtype=np.float64)
        wrist_pose_base = np.asarray(wrist_pose_base, dtype=np.float64)
        wrist_marker_pose_wrist = np.asarray(wrist_marker_pose_wrist, dtype=np.float64)
        return (base_pose_cam @ wrist_pose_base @ wrist_marker_pose_wrist).astype(
            np.float32
        )

    def _residuals(
        self,
        x: np.ndarray,
        observations: list[WorldCalibrationObservation],
    ) -> np.ndarray:
        base_pose_cam, delta_pose_wrist = self._unpack_solution(x)
        wrist_marker_pose_wrist = (
            delta_pose_wrist @ self.initial_wrist_marker_pose_wrist
        )
        residuals: list[np.ndarray] = []
        for obs in observations:
            pred_marker_pose_cam = (
                base_pose_cam @ obs.wrist_pose_base @ wrist_marker_pose_wrist
            )
            residuals.append(
                self.translation_weight
                * (pred_marker_pose_cam[:3, 3] - obs.marker_pose_cam[:3, 3])
            )
            rel_rot = R.from_matrix(obs.marker_pose_cam[:3, :3]).inv() * R.from_matrix(
                pred_marker_pose_cam[:3, :3]
            )
            residuals.append(self.rotation_weight * rel_rot.as_rotvec())
        residuals.append(
            self.marker_delta_regularization_weight
            * (delta_pose_wrist[:3, 3] / self.marker_delta_translation_sigma_m)
        )
        residuals.append(
            self.marker_delta_regularization_weight
            * (
                R.from_matrix(delta_pose_wrist[:3, :3]).as_rotvec()
                / self.marker_delta_rotation_sigma_rad
            )
        )
        return np.concatenate(residuals, axis=0)

    def _unpack_solution(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x.shape != (12,):
            raise ValueError(
                f"Expected optimization variable shape (12,), got {x.shape}."
            )
        return self._params_to_pose(x[:6]), self._params_to_pose(x[6:])

    def _pose_to_params(self, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64)
        return np.concatenate(
            [pose[:3, 3], R.from_matrix(pose[:3, :3]).as_rotvec()], axis=0
        )

    def _params_to_pose(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (6,):
            raise ValueError(f"Expected pose params shape (6,), got {params.shape}.")
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = params[:3]
        pose[:3, :3] = R.from_rotvec(params[3:]).as_matrix()
        return pose
