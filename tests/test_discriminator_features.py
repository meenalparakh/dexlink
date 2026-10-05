from __future__ import annotations

import torch

from dexlink.src.rl.sim.disc_manager import _bps_distance_vec as sim_bps_distance
from dexlink.src.rl.world.disc_manager import _bps_distance_vec as world_bps_distance


def test_duplicate_environment_bps_features_match_expected_deltas() -> None:
    first = torch.tensor([[[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]]])
    second = torch.tensor([[[1.5, 1.0, 4.0, 0.4, 0.1, 0.8]]])
    expected = second - first

    assert torch.allclose(sim_bps_distance(first, second), expected)
    assert torch.allclose(world_bps_distance(first, second), expected)
