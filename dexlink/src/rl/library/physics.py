"""Bound Genesis memory while failing explicitly on contact-buffer saturation."""

from __future__ import annotations

import json
from pathlib import Path
import torch

from dexlink.src.rl.multi_object.physics_sim import MultiObjectPhysicsSim


class NovelPhysicsSim(MultiObjectPhysicsSim):
    def __init__(self, sim_cfg, env) -> None:
        super().__init__(sim_cfg, env)
        self.maximum_observed_contacts = 0
        self.maximum_observed_broad_pairs = 0
        self.finite_state_output: Path | None = None
        self.validation_context = "initialization"

    def validate_native_states(self, output: Path) -> None:
        """Capture a native failure before the environment resets its observations."""
        self.finite_state_output = output

    def step(self) -> None:
        super().step()
        if self.finite_state_output is not None:
            solver = self.scene.sim.rigid_solver
            invalid = {}
            for name, values in {
                "qpos": solver.get_qpos(),
                "dof_velocity": solver.get_dofs_velocity(),
            }.items():
                mask = ~torch.isfinite(values)
                if bool(mask.any()):
                    rows = torch.where(mask.any(dim=1))[0]
                    invalid[name] = {
                        "env_indices": rows.cpu().tolist(),
                        "values": str(values[rows[:8]].cpu()),
                    }
            if invalid:
                self.finite_state_output.parent.mkdir(parents=True, exist_ok=True)
                self.finite_state_output.write_text(
                    json.dumps(
                        {
                            "context": self.validation_context,
                            "invalid_native_states": invalid,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                raise RuntimeError(
                    f"Non-finite native physics state before reset: {invalid}"
                )
        # These diagnostics are specific to the pinned Genesis solver. Never
        # silently drop contact pairs to make a larger batch fit in GPU memory.
        collider = self.scene.sim.rigid_solver.collider
        state, info = collider._collider_state, collider._collider_info
        contacts = int(state.n_contacts.to_torch().max())
        broad_pairs = int(state.n_broad_pairs.to_torch().max())
        contact_capacity = int(info._max_contact_pairs[None])
        broad_capacity = int(info._max_collision_pairs_broad[None])
        self.maximum_observed_contacts = max(self.maximum_observed_contacts, contacts)
        self.maximum_observed_broad_pairs = max(
            self.maximum_observed_broad_pairs, broad_pairs
        )
        if contacts >= contact_capacity or broad_pairs >= broad_capacity:
            raise RuntimeError(
                f"Collision-buffer capacity reached: contacts={contacts}/{contact_capacity}, "
                f"broad_pairs={broad_pairs}/{broad_capacity}. Increase max_collision_pairs "
                "and reduce the environment batch if needed; contacts must not be dropped."
            )
