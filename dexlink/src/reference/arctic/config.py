from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dexlink.utils.retargeting.path_utils import get_external_data_dir, get_logs_dir

DEFAULT_WORKSPACE = get_logs_dir() / "arctic_evals"
DEFAULT_DEXMACHINA_ROOT = get_external_data_dir() / "dexmachina"
DEFAULT_ARCTIC_ROOT = get_external_data_dir() / "arctic"
DEXMACHINA_SUPPORT_BLOCK_POS = (0.0, -0.08, 0.90)
DEXMACHINA_SUPPORT_BLOCK_SIZE = (0.20, 0.20, 0.10)
DEXMACHINA_NOTEBOOK_SUPPORT_BLOCK_SIZE = (0.15, 0.15, 0.10)
DEXMACHINA_SUPPORT_BLOCK_FRICTION = 0.1
# Derived from the posed first-frame notebook mesh bounds in the frozen reference.
CORRECTED_NOTEBOOK_SUPPORT_BLOCK_POS = (
    0.109930725,
    -0.107861435,
    0.90183614,
)
CORRECTED_NOTEBOOK_SUPPORT_BLOCK_SIZE = (0.10, 0.10, 0.10)


@dataclass(frozen=True)
class ArcticClipSpec:
    """Exact ARCTIC interval used by the DexMachina main comparison."""

    clip_id: str
    dexmachina_clip: str
    object_name: str
    subject: str
    sequence_name: str
    start_frame: int
    end_frame: int
    dexmachina_xhand_add_auc: float | None
    dexmachina_xhand_add_auc_sem: float | None
    retargeting_source_clip_id: str | None = None
    support_block_pos_override: tuple[float, float, float] | None = None
    support_block_size_override: tuple[float, float, float] | None = None
    support_block_asset_name_override: str | None = None

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError(
                f"Invalid [start, end) interval for {self.clip_id!r}: "
                f"[{self.start_frame}, {self.end_frame})."
            )
        if self.support_block_pos_override is not None and len(
            self.support_block_pos_override
        ) != 3:
            raise ValueError(
                "Expected support_block_pos_override with three xyz values, got "
                f"{self.support_block_pos_override}."
            )
        if self.support_block_size_override is not None and (
            len(self.support_block_size_override) != 3
            or any(side <= 0.0 for side in self.support_block_size_override)
        ):
            raise ValueError(
                "Expected a positive support_block_size_override with three xyz "
                f"values, got {self.support_block_size_override}."
            )

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def raw_sequence_stem(self) -> str:
        return f"{self.subject}/{self.sequence_name}"

    def processed_demo_path(self) -> Path:
        """Resolve the shared processed sequence from the current project root."""
        return (
            get_external_data_dir()
            / "arctic_eval_clips"
            / self.subject
            / f"{self.sequence_name}.npy"
        )

    def object_asset_dir(self, workspace: Path) -> Path:
        return workspace / "assets" / "arctic" / self.object_name

    def object_urdf_path(self, workspace: Path) -> Path:
        return (
            self.object_asset_dir(workspace)
            / "decomp"
            / f"{self.object_name}_decomp_abs.urdf"
        )

    @property
    def support_block_pos(self) -> tuple[float, float, float]:
        if self.support_block_pos_override is not None:
            return self.support_block_pos_override
        return DEXMACHINA_SUPPORT_BLOCK_POS

    @property
    def support_block_size(self) -> tuple[float, float, float]:
        if self.support_block_size_override is not None:
            return self.support_block_size_override
        if self.object_name == "notebook":
            return DEXMACHINA_NOTEBOOK_SUPPORT_BLOCK_SIZE
        return DEXMACHINA_SUPPORT_BLOCK_SIZE

    @property
    def support_block_friction(self) -> float:
        return DEXMACHINA_SUPPORT_BLOCK_FRICTION

    def support_block_asset_dir(self, workspace: Path) -> Path:
        size_name = self.support_block_asset_name_override
        if size_name is None:
            size_name = "notebook" if self.object_name == "notebook" else "standard"
        return workspace / "assets" / "arctic" / "support_blocks" / size_name

    def support_block_urdf_path(self, workspace: Path) -> Path:
        return self.support_block_asset_dir(workspace) / "simple_object.urdf"

    def trajectory_dir(self, workspace: Path) -> Path:
        return workspace / "trajectories" / self.clip_id

    def dense_motion_path(self, workspace: Path) -> Path:
        return self.trajectory_dir(workspace) / "dense_retargeted_motion.npz"

    def exported_sequence_path(self, workspace: Path) -> Path:
        return self.trajectory_dir(workspace) / "arctic_sequence_exported.npy"

    @property
    def retargeting_clip_id(self) -> str:
        return self.retargeting_source_clip_id or self.clip_id


_BENCHMARK_CLIPS = (
    ArcticClipSpec(
        "ketchup-100",
        "ketchup30-130-s01-u01",
        "ketchup",
        "s01",
        "ketchup_use_01",
        30,
        130,
        0.903838,
        0.002077,
    ),
    ArcticClipSpec(
        "box-200",
        "box30-230-s01-u01",
        "box",
        "s01",
        "box_use_01",
        30,
        230,
        0.865255,
        0.014366,
    ),
    ArcticClipSpec(
        "mixer-170",
        "mixer30-200-s01-u01",
        "mixer",
        "s01",
        "mixer_use_01",
        30,
        200,
        0.898528,
        0.005272,
    ),
    ArcticClipSpec(
        "ketchup-300",
        "ketchup40-340-s01-u02",
        "ketchup",
        "s01",
        "ketchup_use_02",
        40,
        340,
        0.723543,
        0.044032,
    ),
    ArcticClipSpec(
        "mixer-300",
        "mixer40-340-s01-u01",
        "mixer",
        "s01",
        "mixer_use_01",
        40,
        340,
        0.662344,
        0.069503,
    ),
    ArcticClipSpec(
        "notebook-300",
        "notebook40-340-s02-u02",
        "notebook",
        "s02",
        "notebook_use_02",
        40,
        340,
        0.890477,
        0.012263,
    ),
    ArcticClipSpec(
        "waffleiron-300",
        "waffleiron40-340-s01-u01",
        "waffleiron",
        "s01",
        "waffleiron_use_01",
        40,
        340,
        0.803234,
        0.012737,
    ),
)

_SMOKE_CLIPS = (
    ArcticClipSpec(
        "box-20-smoke",
        "not_reported",
        "box",
        "s01",
        "box_use_01",
        30,
        50,
        None,
        None,
    ),
)

_SCENE_VARIANT_CLIPS = (
    ArcticClipSpec(
        "notebook-300-centered-support",
        "notebook40-340-s02-u02-centered-support",
        "notebook",
        "s02",
        "notebook_use_02",
        40,
        340,
        None,
        None,
        retargeting_source_clip_id="notebook-300",
        support_block_pos_override=CORRECTED_NOTEBOOK_SUPPORT_BLOCK_POS,
        support_block_size_override=CORRECTED_NOTEBOOK_SUPPORT_BLOCK_SIZE,
        support_block_asset_name_override="notebook_centered_010",
    ),
)

BENCHMARK_CLIPS = {clip.clip_id: clip for clip in _BENCHMARK_CLIPS}
SMOKE_CLIPS = {clip.clip_id: clip for clip in _SMOKE_CLIPS}
SCENE_VARIANT_CLIPS = {clip.clip_id: clip for clip in _SCENE_VARIANT_CLIPS}
ARCTIC_CLIPS = {**BENCHMARK_CLIPS, **SMOKE_CLIPS, **SCENE_VARIANT_CLIPS}
if len(ARCTIC_CLIPS) != (
    len(_BENCHMARK_CLIPS) + len(_SMOKE_CLIPS) + len(_SCENE_VARIANT_CLIPS)
):
    raise ValueError("ARCTIC clip ids must be unique.")


def get_clip(clip_id: str) -> ArcticClipSpec:
    try:
        return ARCTIC_CLIPS[clip_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown ARCTIC clip {clip_id!r}; expected one of {sorted(ARCTIC_CLIPS)}."
        ) from exc


def resolve_clips(
    clip_ids: tuple[str, ...], all_clips: bool
) -> tuple[ArcticClipSpec, ...]:
    if all_clips and clip_ids:
        raise ValueError("Use either --clip or --all-clips, not both.")
    if all_clips:
        return tuple(_BENCHMARK_CLIPS)
    if not clip_ids:
        raise ValueError("Specify at least one --clip, or pass --all-clips.")
    return tuple(get_clip(clip_id) for clip_id in clip_ids)
