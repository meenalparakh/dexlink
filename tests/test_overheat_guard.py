from __future__ import annotations

import sys
from types import ModuleType

import pytest

from dexlink.utils.rl.overheat import load_overheat_guard


def test_disabled_overheat_guard_does_not_require_site_module() -> None:
    assert load_overheat_guard(False) is None


def test_enabled_overheat_guard_requires_explicit_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "check_overheat", raising=False)
    with pytest.raises(ModuleNotFoundError, match="agent.check_overheat=True"):
        load_overheat_guard(True)


def test_enabled_overheat_guard_validates_and_returns_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("check_overheat")
    module.pause_needed = lambda: False
    module.pause = lambda: None
    monkeypatch.setitem(sys.modules, "check_overheat", module)

    guard = load_overheat_guard(True)

    assert guard is module
    assert guard.pause_needed() is False
