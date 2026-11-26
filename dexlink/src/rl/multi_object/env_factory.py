from __future__ import annotations

from dexlink.src.rl.config.base import EnvConfig
from dexlink.src.rl.dual_active.registry import is_dual_active_exported_path
from dexlink.src.rl.world.env import Env


def create_env(cfg: EnvConfig) -> Env:
    env_class = str(cfg.env_class)
    if env_class == "library_rigid":
        if cfg.backend != "sim":
            raise ValueError("library_rigid requires the simulation backend.")
        from dexlink.src.rl.library.environment import LibraryEnv
        return LibraryEnv(cfg)
    if env_class == "single_object":
        return Env(cfg)
    if env_class == "multi_object":
        if cfg.backend == "real":
            from dexlink.src.rl.multi_object.real_env import RealMultiObjectEnv

            return RealMultiObjectEnv(cfg)
        # A task whose two hands each support an object of their own opts in by
        # name, so it gets the derived classes without every other task needing
        # to say anything. The table is in
        # `dexlink.src.rl.dual_active.registry`; nothing else is in it.
        if is_dual_active_exported_path(
            str(getattr(cfg.task_cfg, "oakink_exported_path", "") or "")
        ):
            from dexlink.src.rl.dual_active.env import DualActiveEnv

            return DualActiveEnv(cfg)
        from dexlink.src.rl.multi_object.env import MultiObjectEnv

        return MultiObjectEnv(cfg)
    # The same variant, asked for outright rather than by name.
    if env_class == "dual_active_object":
        if cfg.backend == "real":
            raise ValueError(
                "dual_active_object currently supports only cfg.backend='sim'. "
                f"Got {cfg.backend!r}."
            )
        from dexlink.src.rl.dual_active.env import DualActiveEnv

        return DualActiveEnv(cfg)
    raise ValueError(
        "Expected env.env_class to be 'single_object', 'multi_object' or "
        f"'library_rigid' or 'dual_active_object', got {env_class!r}."
    )
