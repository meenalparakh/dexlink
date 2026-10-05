from __future__ import annotations

from pathlib import Path

import numpy as np

from dexlink.src.retargeting.data.oakink_processor_grasp import OakinkProcessorGrasp
from dexlink.src.retargeting.data.oakink_processor_lift import OakinkProcessorLift
from dexlink.src.retargeting.data.oakink_processor_sequence import OakinkProcessorSequence


class ExportedOakInkProcessor:
    def __new__(
        cls,
        exported_npy_path: str | Path,
        *args,
        **kwargs,
    ):
        raw = np.load(exported_npy_path, allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.shape == ():
            payload = raw.item()
        else:
            payload = raw

        if not isinstance(payload, dict):
            raise ValueError(
                f"Expected exported OakInk payload dict in {exported_npy_path}, got {type(payload)}."
            )

        demo_kind = str(payload.get("demo_kind", "grasp")).lower()
        if demo_kind == "lift":
            return OakinkProcessorLift(exported_npy_path, *args, **kwargs)
        if demo_kind == "sequence":
            return OakinkProcessorSequence(exported_npy_path, *args, **kwargs)
        return OakinkProcessorGrasp(exported_npy_path, *args, **kwargs)
