"""Physical parameters for the threaded cap and normally closed spring clip."""

from __future__ import annotations

from dataclasses import dataclass
import math

SCREW_ROBOT = "v3_11_threaded_cap_assembly"
SCREW_JOINTS = ("cap_x", "cap_y", "cap_z", "cap_roll", "cap_pitch", "cap_yaw")

CLIP_ROBOT = "v3_11_spring_clip_assembly"
CLIP_JOINT = "clip_hinge"

NUT_ROBOT = "v3_11_captured_thread_nut"
NUT_JOINTS = ("nut_x", "nut_y", "nut_z", "nut_roll", "nut_pitch", "nut_yaw")


@dataclass(frozen=True)
class NutThreadConfig:
    pitch: float = 0.008
    engaged_turns: float = 3.0
    initial_nut_height: float = 0.105
    capture_radius: float = 0.003
    capture_height: float = 0.0015
    capture_tilt: float = 0.10
    capture_phase: float = 0.20
    thread_friction_torque: float = 0.05
    thread_damping: float = 0.005
    coordinate_armature: float = 1e-6

    @property
    def travel(self) -> float:
        return self.pitch * self.engaged_turns

    def validate(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Expected finite positive nut-thread {name}, got {value}."
                )
        if self.capture_phase >= math.pi or self.capture_height >= self.pitch / 2:
            raise ValueError(
                "Nut capture tolerances must be smaller than half a thread pitch/turn."
            )


@dataclass(frozen=True)
class ScrewCapConfig:
    pitch: float = 0.018
    engaged_turns: float = 1.0
    initial_cap_height: float = 0.205
    thread_friction_torque: float = 0.08
    thread_damping: float = 0.005
    release_angle_tolerance: float = 0.003
    release_position_tolerance: float = 0.0002
    coordinate_link_mass: float = 1e-6
    coordinate_link_inertia: float = 1e-9
    coordinate_armature: float = 0.0

    @property
    def thread_end_angle(self) -> float:
        return 2 * math.pi * self.engaged_turns

    @property
    def thread_travel(self) -> float:
        return self.pitch * self.engaged_turns

    def validate(self) -> None:
        for name, value in vars(self).items():
            if name == "coordinate_armature":
                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"Expected finite non-negative screw coordinate_armature, got {value}."
                    )
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Expected finite positive screw {name}, got {value}.")
        if self.release_angle_tolerance >= self.thread_end_angle:
            raise ValueError(
                "Screw release tolerance must be smaller than the engaged angle."
            )


@dataclass(frozen=True)
class SpringClipConfig:
    stiffness: float = 3.0
    damping: float = 0.12
    friction_torque: float = 0.002
    maximum_opening: float = math.radians(40)
    handle_offset_x: float = -0.14
    handle_offset_z: float = 0.13

    def validate(self) -> None:
        for name in (
            "stiffness",
            "damping",
            "friction_torque",
            "maximum_opening",
            "handle_offset_z",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Expected finite positive spring clip {name}, got {value}."
                )
        if not math.isfinite(self.handle_offset_x) or self.handle_offset_x >= 0:
            raise ValueError("Expected clip handles behind the hinge (negative X).")
