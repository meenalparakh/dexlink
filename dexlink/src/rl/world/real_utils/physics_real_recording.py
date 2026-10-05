"""Recording/logging helpers for the real backend."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dexlink.src.rl.world.real_utils.physics_real_mixin_base import PhysicsRealMixinBase


class PhysicsRealRecordingMixin(PhysicsRealMixinBase):
    def _append_recording_frame(self, *, force_capture: bool) -> None:
        frame = self._latest_camera_frame(force_capture=force_capture)
        color = np.asarray(frame["color"])
        if color.ndim != 3 or color.shape[-1] != 3:
            raise ValueError(
                f"Expected camera color frame shape (H, W, 3), got {color.shape}."
            )
        self._recorded_rgb_frames.append(color.copy())
        self._recording_last_frame_time = float(frame["timestamp"])

    def _start_recording_impl(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._recorded_rgb_frames = []
        self._recording_last_frame_time = None
        self.is_recording = True
        self._append_recording_frame(force_capture=True)

    def _stop_recording_impl(self, save_to_filename: str, fps: int = 30) -> None:
        if not self.is_recording:
            return
        self.is_recording = False
        if len(self._recorded_rgb_frames) == 0:
            raise RuntimeError("No recorded RGB frames are available to write.")
        first_frame = self._recorded_rgb_frames[0]
        if first_frame.ndim != 3 or first_frame.shape[-1] != 3:
            raise ValueError(
                f"Expected recorded RGB frame shape (H, W, 3), got {first_frame.shape}."
            )
        height, width = first_frame.shape[:2]
        output_path = Path(save_to_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}.")
        try:
            for frame_rgb in self._recorded_rgb_frames:
                if frame_rgb.shape[:2] != (height, width):
                    raise ValueError(
                        "Recorded RGB frame resolution changed during recording: "
                        f"expected {(height, width)}, got {frame_rgb.shape[:2]}."
                    )
                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        self._recorded_rgb_frames = []
        self._recording_last_frame_time = None

    def _video_recording_hook_impl(self, timestep: int) -> None:
        del timestep
        if not self.is_recording:
            return
        frame = self._latest_camera_frame(force_capture=False)
        frame_timestamp = float(frame["timestamp"])
        if self._recording_last_frame_time is None:
            self._append_recording_frame(force_capture=False)
            return
        if (
            frame_timestamp - self._recording_last_frame_time
            < self._recording_frame_interval_s
        ):
            return
        self._recorded_rgb_frames.append(np.asarray(frame["color"]).copy())
        self._recording_last_frame_time = frame_timestamp
