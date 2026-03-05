"""Tests for the Morse pair functions and wrapped model."""

import torch

import torch_sim as ts
from torch_sim.models.morse import MorseModel, morse_pair, morse_pair_force


def _dummy_z(n: int) -> torch.Tensor:
    return torch.ones(n, dtype=torch.long)


def test_morse_pair_minimum_at_sigma() -> None:
    """Morse minimum is at r = sigma."""
    dr = torch.linspace(0.5, 2.0, 500)
    z = _dummy_z(len(dr))
    energies = morse_pair(dr, z, z, sigma=1.0, epsilon=5.0, alpha=5.0)
    min_r = dr[energies.argmin()]
    assert abs(min_r.item() - 1.0) < 0.01


def test_morse_pair_energy_at_minimum() -> None:
    """Morse energy at minimum equals -epsilon."""
    dr = torch.tensor([1.0])
    z = _dummy_z(1)
    e = morse_pair(dr, z, z, sigma=1.0, epsilon=5.0, alpha=5.0)
    torch.testing.assert_close(e, torch.tensor([-5.0]), rtol=1e-5, atol=1e-5)


def test_morse_pair_scaling() -> None:
    """Energy scales linearly with epsilon."""
    dr = torch.tensor([1.5])
    z = _dummy_z(1)
    e1 = morse_pair(dr, z, z, epsilon=1.0)
    e2 = morse_pair(dr, z, z, epsilon=2.0)
    torch.testing.assert_close(e2, 2.0 * e1, rtol=1e-5, atol=1e-5)


def test_morse_pair_force_scaling() -> None:
    """Force scales linearly with epsilon."""
    dr = torch.tensor([1.5])
    z = _dummy_z(1)
    f1 = morse_pair_force(dr, z, z, epsilon=1.0)
    f2 = morse_pair_force(dr, z, z, epsilon=2.0)
    torch.testing.assert_close(f2, 2.0 * f1)


def test_morse_force_energy_consistency() -> None:
    """Force is consistent with the energy gradient."""
    dr = torch.linspace(0.8, 2.0, 100, requires_grad=True)
    z = _dummy_z(len(dr))

    force_direct = morse_pair_force(dr, z, z)

    energy = morse_pair(dr, z, z)
    force_from_grad = -torch.autograd.grad(energy.sum(), dr, create_graph=True)[0]

    torch.testing.assert_close(force_direct, force_from_grad, rtol=1e-4, atol=1e-4)


def test_morse_alpha_effect() -> None:
    """Larger alpha values make the potential well narrower."""
    dr = torch.linspace(0.8, 1.2, 100)
    z = _dummy_z(len(dr))

    energy1 = morse_pair(dr, z, z, alpha=5.0)
    energy2 = morse_pair(dr, z, z, alpha=10.0)

    def get_well_width(energy: torch.Tensor) -> torch.Tensor:
        min_e = torch.min(energy)
        half_e = min_e / 2
        mask = energy < half_e
        return dr[mask].max() - dr[mask].min()

    assert get_well_width(energy2) < get_well_width(energy1)


def test_morse_model_evaluation(si_double_sim_state: ts.SimState) -> None:
    """MorseModel (wrapped PairPotentialModel) evaluates correctly."""
    model = MorseModel(
        sigma=2.55,
        epsilon=0.436,
        alpha=1.359,
        cutoff=6.0,
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


def test_morse_model_force_conservation(si_double_sim_state: ts.SimState) -> None:
    """MorseModel forces sum to zero (Newton's third law)."""
    model = MorseModel(
        sigma=2.55,
        epsilon=0.436,
        alpha=1.359,
        cutoff=6.0,
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
