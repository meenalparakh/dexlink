"""Optional, site-neutral thermal guard integration for long RL runs."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol, cast


class OverheatGuard(Protocol):
    """Interface expected from an optional ``check_overheat`` package."""

    def pause_needed(self) -> bool: ...

    def pause(self) -> None: ...


def load_overheat_guard(enabled: bool) -> OverheatGuard | None:
    """Load an installed thermal guard only when explicitly enabled.

    DexLink does not bundle or assume a cluster-specific module path. Sites that
    need this hook can provide an importable ``check_overheat`` module exposing
    ``pause_needed`` and ``pause``.
    """
    if not enabled:
        return None

    try:
        module = import_module("check_overheat")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "agent.check_overheat=True requires an importable check_overheat "
            "module. Install the site-specific package or set "
            "agent.check_overheat=False."
        ) from exc

    for function_name in ("pause_needed", "pause"):
        function = getattr(module, function_name, None)
        if not callable(function):
            raise TypeError(
                "Expected check_overheat to expose callable functions "
                f"pause_needed() and pause(); {function_name!r} is missing."
            )
    return cast(OverheatGuard, module)
