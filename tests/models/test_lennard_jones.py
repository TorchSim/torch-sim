"""Tests for the Lennard-Jones pair functions and wrapped model."""

import torch

import torch_sim as ts
from torch_sim.models.lennard_jones import (
    LennardJonesModel,
    lennard_jones_pair,
    lennard_jones_pair_force,
)


def _dummy_z(n: int) -> torch.Tensor:
    return torch.ones(n, dtype=torch.long)


def test_lennard_jones_pair_minimum() -> None:
    """Minimum of LJ is at r = 2^(1/6) * sigma."""
    dr = torch.linspace(0.9, 1.5, 500)
    z = _dummy_z(len(dr))
    energies = lennard_jones_pair(dr, z, z, sigma=1.0, epsilon=1.0)
    min_r = dr[energies.argmin()]
    assert abs(min_r.item() - 2 ** (1 / 6)) < 0.01


def test_lennard_jones_pair_energy_at_minimum() -> None:
    """Energy at minimum equals -epsilon."""
    r_min = torch.tensor([2 ** (1 / 6)])
    z = _dummy_z(1)
    e = lennard_jones_pair(r_min, z, z, sigma=1.0, epsilon=2.0)
    torch.testing.assert_close(e, torch.tensor([-2.0]), rtol=1e-5, atol=1e-5)


def test_lennard_jones_pair_epsilon_scaling() -> None:
    """Energy scales linearly with epsilon."""
    dr = torch.tensor([1.5])
    z = _dummy_z(1)
    e1 = lennard_jones_pair(dr, z, z, sigma=1.0, epsilon=1.0)
    e2 = lennard_jones_pair(dr, z, z, sigma=1.0, epsilon=3.0)
    torch.testing.assert_close(e2, 3.0 * e1)


def test_lennard_jones_pair_repulsive_core() -> None:
    """The potential is strongly repulsive at short distances."""
    z = _dummy_z(1)
    e_close = lennard_jones_pair(torch.tensor([0.5]), z, z)
    e_far = lennard_jones_pair(torch.tensor([2.0]), z, z)
    assert e_close > e_far
    assert e_close > 0  # Repulsive
    assert e_far < 0  # Attractive


def test_lennard_jones_pair_zero_distance() -> None:
    """The function handles zero distances gracefully."""
    dr = torch.zeros(2)
    z = _dummy_z(2)
    energy = lennard_jones_pair(dr, z, z)
    assert not torch.isnan(energy).any()
    assert not torch.isinf(energy).any()


def test_lennard_jones_pair_force_scaling() -> None:
    """Force scales linearly with epsilon."""
    dr = torch.tensor([1.5])
    f1 = lennard_jones_pair_force(dr, sigma=1.0, epsilon=1.0)
    f2 = lennard_jones_pair_force(dr, sigma=1.0, epsilon=2.0)
    torch.testing.assert_close(f2, 2.0 * f1)


def test_lennard_jones_pair_force_repulsive_core() -> None:
    """Force is repulsive at short distances and attractive at long distances."""
    f_close = lennard_jones_pair_force(torch.tensor([0.5]))
    f_far = lennard_jones_pair_force(torch.tensor([2.0]))
    assert f_close > 0  # Repulsive
    assert f_far < 0  # Attractive
    assert abs(f_close) > abs(f_far)


def test_lennard_jones_pair_force_zero_distance() -> None:
    """The force function handles zero distances gracefully."""
    dr = torch.zeros(2)
    force = lennard_jones_pair_force(dr)
    assert not torch.isnan(force).any()
    assert not torch.isinf(force).any()


def test_lennard_jones_force_energy_consistency() -> None:
    """Force is consistent with the energy gradient."""
    dr = torch.linspace(0.8, 2.0, 100, requires_grad=True)
    z = _dummy_z(len(dr))

    force_direct = lennard_jones_pair_force(dr)

    energy = lennard_jones_pair(dr, z, z)
    force_from_grad = -torch.autograd.grad(energy.sum(), dr, create_graph=True)[0]

    torch.testing.assert_close(force_direct, force_from_grad, rtol=1e-4, atol=1e-4)


def test_lennard_jones_model_evaluation(si_double_sim_state: ts.SimState) -> None:
    """LennardJonesModel (wrapped PairPotentialModel) evaluates correctly."""
    model = LennardJonesModel(
        sigma=3.405,
        epsilon=0.0104,
        cutoff=2.5 * 3.405,
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=True,
    )
    results = model(si_double_sim_state)
    assert "energy" in results
    assert "forces" in results
    assert "stress" in results
    assert results["energy"].shape == (si_double_sim_state.n_systems,)
    assert results["forces"].shape == (si_double_sim_state.n_atoms, 3)
    assert results["stress"].shape == (si_double_sim_state.n_systems, 3, 3)


def test_lennard_jones_model_force_conservation(
    si_double_sim_state: ts.SimState,
) -> None:
    """LennardJonesModel forces sum to zero (Newton's third law)."""
    model = LennardJonesModel(
        sigma=3.405,
        epsilon=0.0104,
        cutoff=2.5 * 3.405,
        dtype=torch.float64,
        compute_forces=True,
    )
    results = model(si_double_sim_state)
    for sys_idx in range(si_double_sim_state.n_systems):
        mask = si_double_sim_state.system_idx == sys_idx
        assert torch.allclose(
            results["forces"][mask].sum(dim=0),
            torch.zeros(3, dtype=torch.float64),
            atol=1e-10,
        )
