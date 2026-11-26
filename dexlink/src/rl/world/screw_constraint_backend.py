"""Native Genesis equality activation for the finite screw thread.

Genesis exposes joint-equality construction but no per-environment activation
setter. This backend kernel changes only the constraint type; positions and
velocities remain entirely owned by the rigid solver.
"""

import genesis as gs
import gstaichi as ti
import numpy as np
import math
from typing import TYPE_CHECKING

from dexlink.src.rl.config.passive_mechanisms import (
    NUT_JOINTS,
    NutThreadConfig,
    ScrewCapConfig,
)

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity

THREAD_EQUALITIES = (
    "thread_lock_cap_x",
    "thread_lock_cap_y",
    "thread_lock_cap_roll",
    "thread_lock_cap_pitch",
    "thread_pitch",
)

NUT_EQUALITIES = (
    "nut_lock_x",
    "nut_lock_y",
    "nut_lock_roll",
    "nut_lock_pitch",
    "nut_helical_pitch",
)


def add_nut_constraints(entity: "RigidEntity", config: NutThreadConfig) -> None:
    if entity.is_built or entity.equalities:
        raise ValueError("Add nut constraints once, before scene.build().")
    config.validate()
    for name, coordinate in zip(
        NUT_EQUALITIES,
        (NUT_JOINTS[0], NUT_JOINTS[1], NUT_JOINTS[3], NUT_JOINTS[4], NUT_JOINTS[5]),
        strict=True,
    ):
        data = np.zeros(11)
        data[1] = 2 * math.pi / config.pitch if coordinate == "nut_yaw" else 0.0
        entity._add_equality(
            name=name,
            type=gs.EQUALITY_TYPE.JOINT,
            objs_name=(coordinate, "nut_z"),
            data=data,
            sol_params=np.array([0.02, 1.0, 0.9999, 0.9999, 0.001, 0.5, 2.0]),
        )


@ti.kernel(pure=gs.use_pure)
def set_thread_phase_origin(
    eq_data: ti.template(),
    equality_index: ti.i32,
    environment_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    origins: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for index in range(environment_indices.shape[0]):
        eq_data[equality_index, environment_indices[index]][0] = origins[index]


def add_thread_constraints(entity: "RigidEntity", config: ScrewCapConfig) -> None:
    """Register solver rows before scene.build(), without URDF mimic FK.

    URDF mimic would keep overriding the cap's free coordinates in geometry FK
    after disengagement. Registering equalities only with the physics backend
    keeps the reference, observation and released-cap geometry consistent.
    """
    if entity.is_built or entity.equalities:
        raise ValueError("Add screw constraints once, before scene.build().")
    config.validate()
    for name, coordinate in zip(
        THREAD_EQUALITIES,
        ("cap_x", "cap_y", "cap_roll", "cap_pitch", "cap_yaw"),
        strict=True,
    ):
        data = np.zeros(11)
        data[1] = 2 * math.pi / config.pitch if coordinate == "cap_yaw" else 0.0
        entity._add_equality(
            name=name,
            type=gs.EQUALITY_TYPE.JOINT,
            objs_name=(coordinate, "cap_z"),
            data=data,
            sol_params=np.array([0.02, 1.0, 0.9999, 0.9999, 0.001, 0.5, 2.0]),
        )


@ti.kernel(pure=gs.use_pure)
def set_thread_engagement(
    eq_type: ti.template(),
    equality_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    environment_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    engaged: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for i, j in ti.ndrange(equality_indices.shape[0], environment_indices.shape[0]):
        eq_type[equality_indices[i], environment_indices[j]] = (
            gs.ti_int(gs.EQUALITY_TYPE.JOINT) if engaged[j] else gs.ti_int(-1)
        )
