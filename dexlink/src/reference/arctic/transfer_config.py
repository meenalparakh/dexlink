from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dexlink.src.retargeting.data.process_arctic import get_processor_cfg
from dexlink.src.reference.arctic.config import ArcticClipSpec, get_clip


@dataclass(frozen=True)
class ArcticTransferSpec:
    """One dense ARCTIC clip retargeted to a different articulated asset."""

    transfer_id: str
    source_clip_id: str
    target_object: str
    support_clip_id: str
    support_block_size: tuple[float, float, float]
    support_block_xy_offset: tuple[float, float]
    description: str

    @property
    def source_clip(self) -> ArcticClipSpec:
        return get_clip(self.source_clip_id)

    @property
    def frame_count(self) -> int:
        return self.source_clip.frame_count

    @property
    def support_clip(self) -> ArcticClipSpec:
        """Support geometry follows the target category, not the source clip."""
        return get_clip(self.support_clip_id)

    def frame_output_dir(self, workspace: Path, timestep: int, side: str) -> Path:
        return (
            workspace
            / "transfer_retargeted"
            / self.transfer_id
            / f"t{timestep:04d}"
            / side
        )

    def frame_motion_path(self, workspace: Path, timestep: int, side: str) -> Path:
        return self.frame_output_dir(workspace, timestep, side) / "sequence_motion_lineseq.npz"

    def support_block_asset_dir(self, workspace: Path) -> Path:
        return (
            workspace
            / "assets"
            / "arctic"
            / "transfer_support_blocks"
            / self.transfer_id
        )

    def support_block_urdf_path(self, workspace: Path) -> Path:
        return self.support_block_asset_dir(workspace) / "simple_object.urdf"

    def validate(self) -> None:
        # Resolve both identifiers eagerly so malformed manifests fail before submission.
        self.source_clip
        self.support_clip
        get_processor_cfg(self.target_object)
        if len(self.support_block_size) != 3 or any(
            side <= 0.0 for side in self.support_block_size
        ):
            raise ValueError(
                f"Expected a positive xyz support size for {self.transfer_id!r}, "
                f"got {self.support_block_size}."
            )
        if len(self.support_block_xy_offset) != 2:
            raise ValueError(
                f"Expected an xy support offset for {self.transfer_id!r}, "
                f"got {self.support_block_xy_offset}."
            )


_TRANSFER_SPECS = (
    ArcticTransferSpec(
        transfer_id="box-200_proc-box-1",
        source_clip_id="box-200",
        target_object="proc_box_1",
        support_clip_id="box-200",
        support_block_size=(0.10, 0.10, 0.10),
        support_block_xy_offset=(0.0, -0.01),
        description="DexMachina box-200 clip retargeted to procedural box variation 1",
    ),
    ArcticTransferSpec(
        transfer_id="notebook-300_proc-notebook-1",
        source_clip_id="notebook-300",
        target_object="proc_notebook_1",
        support_clip_id="notebook-300",
        support_block_size=(0.10, 0.10, 0.10),
        support_block_xy_offset=(0.0, 0.0),
        description=(
            "DexMachina notebook-300 clip retargeted to procedural notebook variation 1"
        ),
    ),
    ArcticTransferSpec(
        transfer_id="waffleiron-300_proc-notebook-1",
        source_clip_id="waffleiron-300",
        target_object="cross_waffleiron_to_proc_notebook_1",
        support_clip_id="notebook-300",
        support_block_size=(0.075, 0.075, 0.10),
        support_block_xy_offset=(0.0, 0.0),
        description=(
            "DexMachina waffleiron-300 motion and contacts transferred to procedural "
            "notebook variation 1"
        ),
    ),
    ArcticTransferSpec(
        transfer_id="ketchup-300_proc-ketchup-2",
        source_clip_id="ketchup-300",
        target_object="proc_ketchup_2",
        support_clip_id="ketchup-300",
        support_block_size=(0.10, 0.10, 0.10),
        support_block_xy_offset=(0.0, 0.0),
        description=(
            "DexMachina ketchup-300 clip retargeted to procedural ketchup "
            "variation 2"
        ),
    ),
)

ARCTIC_TRANSFER_SPECS = {spec.transfer_id: spec for spec in _TRANSFER_SPECS}
if len(ARCTIC_TRANSFER_SPECS) != len(_TRANSFER_SPECS):
    raise ValueError("ARCTIC transfer ids must be unique.")


def get_transfer(transfer_id: str) -> ArcticTransferSpec:
    try:
        spec = ARCTIC_TRANSFER_SPECS[transfer_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown ARCTIC transfer {transfer_id!r}; expected one of "
            f"{sorted(ARCTIC_TRANSFER_SPECS)}."
        ) from exc
    spec.validate()
    return spec


def resolve_transfers(
    transfer_ids: tuple[str, ...], all_transfers: bool
) -> tuple[ArcticTransferSpec, ...]:
    if all_transfers and transfer_ids:
        raise ValueError("Use either --transfer or --all-transfers, not both.")
    if all_transfers:
        transfers = _TRANSFER_SPECS
    elif transfer_ids:
        transfers = tuple(get_transfer(transfer_id) for transfer_id in transfer_ids)
    else:
        raise ValueError("Specify at least one --transfer, or pass --all-transfers.")
    for transfer in transfers:
        transfer.validate()
    return transfers
