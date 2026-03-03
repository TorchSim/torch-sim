"""Tests for the particle life force function."""

import torch

from torch_sim.models.particle_life import particle_life_pair_force


def _dummy_z(n: int) -> torch.Tensor:
    return torch.ones(n, dtype=torch.long)


def test_inner_region_repulsive() -> None:
    """For dr < beta the force is negative (repulsive)."""
    dr = torch.tensor([0.1, 0.2])
    z = _dummy_z(2)
    f = particle_life_pair_force(dr, z, z, A=1.0, beta=0.3, sigma=1.0)
    assert (f < 0).all()


def test_zero_beyond_sigma() -> None:
    """Force is zero at and beyond sigma."""
    dr = torch.tensor([1.0, 1.5])
    z = _dummy_z(2)
    f = particle_life_pair_force(dr, z, z, A=1.0, beta=0.3, sigma=1.0)
    assert (f == 0.0).all()


def test_amplitude_scaling() -> None:
    """Outer-region force scales with A."""
    dr = torch.tensor([0.6])  # between beta and sigma
    z = _dummy_z(1)
    f1 = particle_life_pair_force(dr, z, z, A=1.0, beta=0.3, sigma=1.0)
    f2 = particle_life_pair_force(dr, z, z, A=3.0, beta=0.3, sigma=1.0)
    torch.testing.assert_close(f2, 3.0 * f1)
