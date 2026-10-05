"""Task manager for references that keep two objects active at the same time."""

from __future__ import annotations

import numpy as np

from dexlink.src.rl.multi_object.task_manager import MultiObjectTaskManager


class DualActiveTaskManager(MultiObjectTaskManager):
    """Accept references whose active mask names more than one object per frame.

    The default manager refuses them: it derives a single active object index per
    frame and raises as soon as a frame names two. That index is only ever used
    for equality, to group reference timesteps that belong to the same part of
    the task, and to pick a reset target, so a frame with two active objects can
    report the lower of the two without changing what either consumer does.

    Nothing else about the manager changes. The object tracking reward already
    reads the mask itself and averages over whichever objects it names.
    """

    def _derive_active_object_index(self, active_mask: np.ndarray) -> np.ndarray:
        if active_mask.ndim != 2:
            raise ValueError(
                f"Expected scene_object_active_mask shape (T, N), got {active_mask.shape}."
            )
        active_index = np.full(active_mask.shape[0], -1, dtype=np.int64)
        for frame_idx, frame_mask in enumerate(active_mask):
            active = np.flatnonzero(frame_mask)
            if active.shape[0] > 0:
                active_index[frame_idx] = int(active[0])
        return active_index
