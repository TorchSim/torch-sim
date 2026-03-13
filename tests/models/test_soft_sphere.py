"""Tests for the soft sphere pair functions, wrapped model, and multi-species models."""

import pytest
import torch

import torch_sim as ts
from torch_sim.models.soft_sphere import (
    MultiSoftSpherePairFn,
    SoftSphereModel,
    SoftSphereMultiModel,
    soft_sphere_pair,
)


def _dummy_z(n: int) -> torch.Tensor:
    return torch.ones(n, dtype=torch.long)


def test_soft_sphere_zero_beyond_sigma() -> None:
    """Soft-sphere energy is zero for r >= sigma."""
    dr = torch.tensor([1.0, 1.5, 2.0])
    z = _dummy_z(3)
    e = soft_sphere_pair(dr, z, z, sigma=1.0)
    assert e[1] == 0.0
    assert e[2] == 0.0


def test_soft_sphere_repulsive_only() -> None:
    """Soft-sphere energies are non-negative for r < sigma."""
    dr = torch.linspace(0.1, 0.99, 50)
    z = _dummy_z(len(dr))
    e = soft_sphere_pair(dr, z, z, sigma=1.0, epsilon=1.0, alpha=2.0)
    assert (e >= 0).all()


@pytest.mark.parametrize(
    ("distance", "sigma", "epsilon", "alpha", "expected"),
    [
        (0.5, 1.0, 1.0, 2.0, 0.125),  # distance < sigma
        (1.0, 1.0, 1.0, 2.0, 0.0),  # distance = sigma
        (1.5, 1.0, 1.0, 2.0, 0.0),  # distance > sigma
    ],
)
def test_soft_sphere_pair_single(
    distance: float, sigma: float, epsilon: float, alpha: float, expected: float
) -> None:
    """Test the soft sphere pair calculation for single values."""
    dr = torch.tensor([distance])
    z = _dummy_z(1)
    energy = soft_sphere_pair(dr, z, z, sigma=sigma, epsilon=epsilon, alpha=alpha)
    torch.testing.assert_close(energy, torch.tensor([expected]))


def _make_mss(
    sigma: float = 1.0, epsilon: float = 1.0, alpha: float = 2.0
) -> MultiSoftSpherePairFn:
    """Two-species MultiSoftSpherePairFn with uniform parameters."""
    n = 2
    return MultiSoftSpherePairFn(
        atomic_numbers=torch.tensor([18, 36]),
        sigma_matrix=torch.full((n, n), sigma),
        epsilon_matrix=torch.full((n, n), epsilon),
        alpha_matrix=torch.full((n, n), alpha),
    )


def test_multi_soft_sphere_zero_beyond_sigma() -> None:
    """Energy is zero for r >= sigma."""
    fn = _make_mss(sigma=1.0)
    dr = torch.tensor([1.0, 1.5])
    zi = zj = torch.tensor([18, 36])
    e = fn(dr, zi, zj)
    assert (e == 0.0).all()


def test_multi_soft_sphere_repulsive_only() -> None:
    """Energy is non-negative for r < sigma."""
    fn = _make_mss(sigma=2.0, epsilon=1.0, alpha=2.0)
    dr = torch.linspace(0.1, 1.99, 20)
    zi = zj = torch.full((20,), 18, dtype=torch.long)
    assert (fn(dr, zi, zj) >= 0).all()


def test_multi_soft_sphere_species_lookup() -> None:
    """Different species pairs use the correct off-diagonal parameters."""
    sigma_matrix = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    epsilon_matrix = torch.ones(2, 2)
    alpha_matrix = torch.full((2, 2), 2.0)
    fn = MultiSoftSpherePairFn(
        atomic_numbers=torch.tensor([18, 36]),
        sigma_matrix=sigma_matrix,
        epsilon_matrix=epsilon_matrix,
        alpha_matrix=alpha_matrix,
    )
    dr = torch.tensor([0.5])
    zi_same = torch.tensor([18])
    zj_same = torch.tensor([18])
    zi_cross = torch.tensor([18])
    zj_cross = torch.tensor([36])
    e_same = fn(dr, zi_same, zj_same)  # sigma=1.0, r=0.5 < sigma → non-zero
    e_cross = fn(dr, zi_cross, zj_cross)  # sigma=2.0, r=0.5 < sigma → non-zero
    # cross pair has larger sigma so (1 - r/sigma) is larger → higher energy
    assert e_cross > e_same


def test_multi_soft_sphere_alpha_matrix_default() -> None:
    """Omitting alpha_matrix defaults to 2.0 for all pairs."""
    fn_default = MultiSoftSpherePairFn(
        atomic_numbers=torch.tensor([18, 36]),
        sigma_matrix=torch.full((2, 2), 1.0),
        epsilon_matrix=torch.full((2, 2), 1.0),
    )
    fn_explicit = _make_mss(sigma=1.0, epsilon=1.0, alpha=2.0)
    dr = torch.tensor([0.5])
    zi = zj = torch.tensor([18])
    torch.testing.assert_close(fn_default(dr, zi, zj), fn_explicit(dr, zi, zj))


def test_multi_soft_sphere_bad_matrix_shape_raises() -> None:
    with pytest.raises(ValueError, match="sigma_matrix"):
        MultiSoftSpherePairFn(
            atomic_numbers=torch.tensor([18, 36]),
            sigma_matrix=torch.ones(3, 3),  # wrong shape
            epsilon_matrix=torch.ones(2, 2),
        )


def test_multispecies_initialization_defaults() -> None:
    """Multi-species model initializes with default parameters."""
    model = SoftSphereMultiModel(atomic_numbers=torch.tensor([0, 1]), dtype=torch.float32)
    assert model.sigma_matrix.shape == (2, 2)
    assert model.epsilon_matrix.shape == (2, 2)
    assert model.alpha_matrix.shape == (2, 2)


def test_multispecies_initialization_custom() -> None:
    """Multi-species model stores custom parameter matrices."""
    sigma_matrix = torch.tensor([[1.0, 1.5], [1.5, 2.0]], dtype=torch.float64)
    epsilon_matrix = torch.tensor([[1.0, 0.5], [0.5, 1.5]], dtype=torch.float64)
    alpha_matrix = torch.tensor([[2.0, 3.0], [3.0, 4.0]], dtype=torch.float64)

    model = SoftSphereMultiModel(
        atomic_numbers=torch.tensor([0, 1]),
        sigma_matrix=sigma_matrix,
        epsilon_matrix=epsilon_matrix,
        alpha_matrix=alpha_matrix,
        cutoff=3.0,
        dtype=torch.float64,
    )

    assert torch.allclose(model.sigma_matrix, sigma_matrix)
    assert torch.allclose(model.epsilon_matrix, epsilon_matrix)
    assert torch.allclose(model.alpha_matrix, alpha_matrix)
    assert model.cutoff.item() == 3.0


def test_multispecies_matrix_validation() -> None:
    """Incorrectly sized matrices raise ValueError."""
    sigma_matrix = torch.tensor([[1.0, 1.5], [1.5, 2.0]])
    epsilon_matrix = torch.tensor([[1.0, 0.5], [0.5, 1.5]])

    with pytest.raises(ValueError, match="sigma_matrix must have shape"):
        SoftSphereMultiModel(
            atomic_numbers=torch.tensor([0, 1, 2]),
            sigma_matrix=sigma_matrix,
            epsilon_matrix=epsilon_matrix,
        )


@pytest.mark.parametrize(
    ("matrix_name", "matrix"),
    [
        ("sigma_matrix", torch.tensor([[1.0, 1.5], [2.0, 2.0]])),
        ("epsilon_matrix", torch.tensor([[1.0, 0.5], [0.7, 1.5]])),
        ("alpha_matrix", torch.tensor([[2.0, 3.0], [4.0, 4.0]])),
    ],
)
def test_matrix_symmetry_validation(matrix_name: str, matrix: torch.Tensor) -> None:
    """Parameter matrices are validated for symmetry."""
    symmetric_matrix = torch.tensor([[1.0, 1.5], [1.5, 2.0]])
    params = {
        "atomic_numbers": torch.tensor([0, 1]),
        "sigma_matrix": symmetric_matrix,
        "epsilon_matrix": symmetric_matrix,
        "alpha_matrix": symmetric_matrix,
    }
    params[matrix_name] = matrix

    with pytest.raises(ValueError, match="is not symmetric"):
        SoftSphereMultiModel(**params)


def test_multispecies_cutoff_default() -> None:
    """Default cutoff is the maximum sigma value."""
    sigma_matrix = torch.tensor([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5], [2.0, 2.5, 3.0]])
    model = SoftSphereMultiModel(
        atomic_numbers=torch.tensor([0, 1, 2]), sigma_matrix=sigma_matrix
    )
    assert model.cutoff.item() == 3.0


def test_multispecies_evaluation() -> None:
    """Multi-species model evaluates without error on a small system."""
    sigma_matrix = torch.tensor([[1.0, 0.8], [0.8, 0.6]], dtype=torch.float64)
    epsilon_matrix = torch.tensor([[1.0, 0.5], [0.5, 2.0]], dtype=torch.float64)

    model = SoftSphereMultiModel(
        atomic_numbers=torch.tensor([0, 1]),
        sigma_matrix=sigma_matrix,
        epsilon_matrix=epsilon_matrix,
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=True,
    )

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]],
        dtype=torch.float64,
    )
    cell = torch.eye(3, dtype=torch.float64) * 2.0
    state = ts.SimState(
        positions=positions,
        cell=cell,
        pbc=True,
        masses=torch.ones(4, dtype=torch.float64),
        atomic_numbers=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )
    results = model(state)
    assert "energy" in results
    assert "forces" in results
    assert "stress" in results


def test_soft_sphere_model_evaluation(si_double_sim_state: ts.SimState) -> None:
    """SoftSphereModel (wrapped PairPotentialModel) evaluates correctly."""
    model = SoftSphereModel(
        sigma=5.0,
        epsilon=0.0104,
        alpha=2.0,
        cutoff=5.0,
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


def test_soft_sphere_model_force_conservation(
    si_double_sim_state: ts.SimState,
) -> None:
    """SoftSphereModel forces sum to zero (Newton's third law)."""
    model = SoftSphereModel(
        sigma=5.0,
        epsilon=0.0104,
        alpha=2.0,
        cutoff=5.0,
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
