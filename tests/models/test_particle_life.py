"""Tests for the particle life force function and wrapped model."""

import torch

import torch_sim as ts
from torch_sim.models.particle_life import ParticleLifeModel, particle_life_pair_force


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


def test_particle_life_model_evaluation(si_double_sim_state: ts.SimState) -> None:
    """ParticleLifeModel (wrapped PairForcesModel) evaluates correctly."""
    model = ParticleLifeModel(
        A=1.0,
        beta=0.3,
        sigma=5.26,
        cutoff=5.26,
        dtype=si_double_sim_state.dtype,
        compute_stress=True,
    )
    results = model(si_double_sim_state)
    assert "energy" in results
    assert "forces" in results
    assert "stress" in results
    assert results["energy"].shape == (si_double_sim_state.n_systems,)
    assert results["forces"].shape == (si_double_sim_state.n_atoms, 3)
    assert results["stress"].shape == (si_double_sim_state.n_systems, 3, 3)
    # Energy should be zeros for PairForcesModel
    assert torch.allclose(
        results["energy"],
        torch.zeros(si_double_sim_state.n_systems, dtype=si_double_sim_state.dtype),
    )


def test_particle_life_model_force_conservation(
    si_double_sim_state: ts.SimState,
) -> None:
    """ParticleLifeModel forces sum to zero (Newton's third law)."""
    model = ParticleLifeModel(
        A=1.0,
        beta=0.3,
        sigma=5.26,
        cutoff=5.26,
        dtype=torch.float64,
    )
    results = model(si_double_sim_state)
    for sys_idx in range(si_double_sim_state.n_systems):
        mask = si_double_sim_state.system_idx == sys_idx
        assert torch.allclose(
            results["forces"][mask].sum(dim=0),
            torch.zeros(3, dtype=torch.float64),
            atol=1e-10,
        )
