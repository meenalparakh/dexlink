from __future__ import annotations

from dexlink.src.rl.multi_object.env import MultiObjectEnv


class RealMultiObjectEnv(MultiObjectEnv):
    """Multi-object env with the real-world backend.

    The policy-facing managers remain the multi-object RL managers. Only the
    physics backend is swapped so checkpoint observation/action terms stay
    aligned with the training env.
    """

    def _create_physics_manager(self):
        if self.cfg.backend != "real":
            return super()._create_physics_manager()

        from dexlink.src.rl.multi_object.physics_real import MultiObjectPhysicsReal

        return MultiObjectPhysicsReal(self.cfg.real_cfg, self)
