from __future__ import annotations

from pathlib import Path

import pytest

from dexlink.utils.retargeting.path_utils import DEXLINK_OUTPUTS_ENV
from dexlink.visualizations.retargeting.render_oakink_last_frame_views import (
    _resolve_robot_urdf_path,
)
from dexlink.visualizations.rl.rollout_rendering import (
    CameraSpec,
    localize_path,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_xhand_visualization_resolves_bundled_robot_without_other_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(DEXLINK_OUTPUTS_ENV, str(tmp_path / "outputs"))
    for side in ("left", "right"):
        urdf_path = _resolve_robot_urdf_path("xhand", side)
        assert urdf_path.is_file()
        assert urdf_path.is_relative_to(tmp_path / "outputs")
        assert "assets/robots/xhand" in urdf_path.read_text(encoding="utf-8")


def test_rollout_rendering_uses_portable_repository_paths() -> None:
    camera = CameraSpec(direction=(-1.0, 1.0, 0.3), margin=1.2)
    assert camera.margin == 1.2
    assert localize_path("assets/PROVENANCE.yaml") == (
        REPOSITORY_ROOT / "assets" / "PROVENANCE.yaml"
    )
