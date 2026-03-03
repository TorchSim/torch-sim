"""Tests for the general pair potential model and standard pair functions."""

import functools

import pytest
import torch

import torch_sim as ts
from tests.conftest import DEVICE, DTYPE
from tests.models.conftest import make_validate_model_outputs_test
from torch_sim.models.interface import ModelInterface
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.models.morse import MorseModel
from torch_sim.models.pair_potential import (
    MultiSoftSpherePairFn,
    PairForcesModel,
    PairPotentialModel,
    full_to_half_list,
    lj_pair,
    morse_pair,
    particle_life_pair_force,
    soft_sphere_pair,
)
from torch_sim.models.soft_sphere import SoftSphereModel


# Argon LJ parameters
LJ_SIGMA = 3.405
LJ_EPSILON = 0.0104
LJ_CUTOFF = 2.5 * LJ_SIGMA


@pytest.fixture
def lj_model_pp() -> PairPotentialModel:
    return PairPotentialModel(
        pair_fn=functools.partial(lj_pair, sigma=LJ_SIGMA, epsilon=LJ_EPSILON),
        cutoff=LJ_CUTOFF,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
        per_atom_energies=True,
        per_atom_stresses=True,
    )


@pytest.fixture
def particle_life_model() -> PairForcesModel:
    return PairForcesModel(
        force_fn=functools.partial(particle_life_pair_force, A=1.0, beta=0.3, sigma=5.26),
        cutoff=5.26,
        dtype=DTYPE,
        compute_stress=True,
        per_atom_stresses=True,
    )


# Interface validation via factory
test_pair_potential_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="lj_model_pp", device=DEVICE, dtype=DTYPE
)

test_pair_forces_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="particle_life_model", device=DEVICE, dtype=DTYPE
)


def _dummy_z(n: int) -> torch.Tensor:
    return torch.ones(n, dtype=torch.long)


def test_lj_pair_minimum() -> None:
    """Minimum of LJ is at r = 2^(1/6) * sigma."""
    dr = torch.linspace(0.9, 1.5, 500)
    z = _dummy_z(len(dr))
    energies = lj_pair(dr, z, z, sigma=1.0, epsilon=1.0)
    min_r = dr[energies.argmin()]
    assert abs(min_r.item() - 2 ** (1 / 6)) < 0.01


def test_lj_pair_energy_at_minimum() -> None:
    """Energy at minimum equals -epsilon."""
    r_min = torch.tensor([2 ** (1 / 6)])
    z = _dummy_z(1)
    e = lj_pair(r_min, z, z, sigma=1.0, epsilon=2.0)
    torch.testing.assert_close(e, torch.tensor([-2.0]), rtol=1e-5, atol=1e-5)


def test_lj_pair_epsilon_scaling() -> None:
    dr = torch.tensor([1.5])
    z = _dummy_z(1)
    e1 = lj_pair(dr, z, z, sigma=1.0, epsilon=1.0)
    e2 = lj_pair(dr, z, z, sigma=1.0, epsilon=3.0)
    torch.testing.assert_close(e2, 3.0 * e1)


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


def test_particle_life_force_inner() -> None:
    """For dr < beta the force is negative (repulsive)."""
    dr = torch.tensor([0.1, 0.2])
    z = _dummy_z(2)
    f = particle_life_pair_force(dr, z, z, A=1.0, beta=0.3, sigma=1.0)
    assert (f < 0).all()


def test_particle_life_force_zero_beyond_sigma() -> None:
    dr = torch.tensor([1.0, 1.5])
    z = _dummy_z(2)
    f = particle_life_pair_force(dr, z, z, A=1.0, beta=0.3, sigma=1.0)
    assert f[0] == 0.0
    assert f[1] == 0.0


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


def _build_potential_model_pair(name: str) -> tuple[PairPotentialModel, ModelInterface]:
    """Return (PairPotentialModel, reference_model) for a named potential."""
    if name == "lj-half":
        pp = PairPotentialModel(
            pair_fn=functools.partial(lj_pair, sigma=LJ_SIGMA, epsilon=LJ_EPSILON),
            cutoff=LJ_CUTOFF,
            dtype=DTYPE,
            compute_forces=True,
            compute_stress=True,
            per_atom_energies=True,
            per_atom_stresses=True,
            reduce_to_half_list=True,
        )
        ref = LennardJonesModel(
            sigma=LJ_SIGMA,
            epsilon=LJ_EPSILON,
            cutoff=LJ_CUTOFF,
            dtype=DTYPE,
            compute_forces=True,
            compute_stress=True,
            per_atom_energies=True,
            per_atom_stresses=True,
        )
        return pp, ref
    if name == "lj-full":
        pp = PairPotentialModel(
            pair_fn=functools.partial(lj_pair, sigma=LJ_SIGMA, epsilon=LJ_EPSILON),
            cutoff=LJ_CUTOFF,
            dtype=DTYPE,
            compute_forces=True,
            compute_stress=True,
            per_atom_energies=True,
            per_atom_stresses=True,
            reduce_to_half_list=False,
        )
        ref = LennardJonesModel(
            sigma=LJ_SIGMA,
            epsilon=LJ_EPSILON,
            cutoff=LJ_CUTOFF,
            dtype=DTYPE,
            compute_forces=True,
            compute_stress=True,
            per_atom_energies=True,
            per_atom_stresses=True,
        )
        return pp, ref
    if name == "morse":
        sigma, epsilon, alpha, cutoff = 4.0, 5.0, 5.0, 5.0
        pp = PairPotentialModel(
            pair_fn=functools.partial(
                morse_pair, sigma=sigma, epsilon=epsilon, alpha=alpha
            ),
            cutoff=cutoff,
            dtype=DTYPE,
            compute_forces=True,
            compute_stress=True,
        )
        ref = MorseModel(
            sigma=sigma,
            epsilon=epsilon,
            alpha=alpha,
            cutoff=cutoff,
            dtype=DTYPE,
            compute_forces=True,
            compute_stress=True,
        )
        return pp, ref
    if name == "soft_sphere":
        sigma, epsilon, alpha = 5, 0.0104, 2.0
        pp = PairPotentialModel(
            pair_fn=functools.partial(
                soft_sphere_pair, sigma=sigma, epsilon=epsilon, alpha=alpha
            ),
            cutoff=sigma,
            dtype=DTYPE,
            compute_forces=True,
            compute_stress=True,
        )
        ref = SoftSphereModel(
            sigma=sigma,
            epsilon=epsilon,
            alpha=alpha,
            cutoff=sigma,
            dtype=DTYPE,
            compute_forces=True,
            compute_stress=True,
        )
        return pp, ref

    raise ValueError(f"Unknown potential: {name}")


@pytest.mark.parametrize("potential", ["lj-half", "lj-full", "morse", "soft_sphere"])
def test_potential_matches_reference(
    mixed_double_sim_state: ts.SimState,
    potential: str,
) -> None:
    """PairPotentialModel matches the dedicated reference model."""
    model_pp, model_ref = _build_potential_model_pair(potential)
    out_pp = model_pp(mixed_double_sim_state)
    out_ref = model_ref(mixed_double_sim_state)

    assert (out_pp["energy"] != 0).all()

    for key in out_pp:
        torch.testing.assert_close(out_pp[key], out_ref[key], rtol=1e-4, atol=1e-5)


def test_full_to_half_list_removes_duplicates() -> None:
    """i < j mask halves a symmetric full neighbor list."""
    # 3-atom full list: (0,1),(1,0),(0,2),(2,0),(1,2),(2,1)
    mapping = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])
    system_mapping = torch.zeros(6, dtype=torch.long)
    shifts_idx = torch.zeros(6, 3)
    m, _s, _sh = full_to_half_list(mapping, system_mapping, shifts_idx)
    assert m.shape[1] == 3
    assert (m[0] < m[1]).all()


def test_full_to_half_list_preserves_system_and_shifts() -> None:
    mapping = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    system_mapping = torch.tensor([0, 0, 1, 1])
    shifts_idx = torch.tensor(
        [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32
    )
    m, s, sh = full_to_half_list(mapping, system_mapping, shifts_idx)
    assert m.shape[1] == 2
    assert s.tolist() == [0, 1]
    # shifts for kept pairs (0→1) and (2→3)
    assert sh[0].tolist() == [1, 0, 0]
    assert sh[1].tolist() == [0, 1, 0]


@pytest.mark.parametrize("key", ["energy", "forces", "stress", "stresses"])
def test_half_list_matches_full(si_double_sim_state: ts.SimState, key: str) -> None:
    """reduce_to_half_list=True gives the same result as the default full list."""
    fn = functools.partial(lj_pair, sigma=LJ_SIGMA, epsilon=LJ_EPSILON)
    needs_forces = key in ("forces", "stress", "stresses")
    needs_stress = key in ("stress", "stresses")
    common = dict(
        pair_fn=fn,
        cutoff=LJ_CUTOFF,
        dtype=DTYPE,
        compute_forces=needs_forces,
        compute_stress=needs_stress,
        per_atom_stresses=(key == "stresses"),
    )
    model_full = PairPotentialModel(**common)
    model_half = PairPotentialModel(**common, reduce_to_half_list=True)
    out_full = model_full(si_double_sim_state)
    out_half = model_half(si_double_sim_state)
    torch.testing.assert_close(out_half[key], out_full[key], rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("potential", ["lj", "morse", "soft_sphere"])
def test_autograd_force_fn_matches_potential_model(
    si_double_sim_state: ts.SimState, potential: str
) -> None:
    """PairForcesModel with -dV/dr force fn matches PairPotentialModel forces/stress."""
    if potential == "lj":
        pair_fn = functools.partial(lj_pair, sigma=LJ_SIGMA, epsilon=LJ_EPSILON)
        cutoff = LJ_CUTOFF
    elif potential == "morse":
        pair_fn = functools.partial(morse_pair, sigma=4.0, epsilon=5.0, alpha=5.0)
        cutoff = 5.0
    else:
        pair_fn = functools.partial(soft_sphere_pair, sigma=5, epsilon=0.0104, alpha=2.0)
        cutoff = 5.0

    def force_fn(dr: torch.Tensor, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        dr_g = dr.requires_grad_()
        e = pair_fn(dr_g, zi, zj)
        (dv_dr,) = torch.autograd.grad(e.sum(), dr_g)
        return -dv_dr

    model_pp = PairPotentialModel(
        pair_fn=pair_fn,
        cutoff=cutoff,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
        per_atom_stresses=True,
    )
    model_pf = PairForcesModel(
        force_fn=force_fn,
        cutoff=cutoff,
        dtype=DTYPE,
        compute_stress=True,
        per_atom_stresses=True,
    )
    out_pp = model_pp(si_double_sim_state)
    out_pf = model_pf(si_double_sim_state)

    assert (out_pp["forces"] != 0.0).all()

    for key in ("forces", "stress", "stresses"):
        torch.testing.assert_close(out_pp[key], out_pf[key], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("key", ["forces", "stress", "stresses"])
def test_forces_model_half_list_matches_full(
    si_double_sim_state: ts.SimState, key: str
) -> None:
    """PairForcesModel: reduce_to_half_list gives the same result as full list."""
    fn = functools.partial(particle_life_pair_force, A=1.0, beta=0.3, sigma=5.26)
    needs_stress = key in ("stress", "stresses")
    common = dict(
        force_fn=fn,
        cutoff=5.26,
        dtype=DTYPE,
        compute_stress=needs_stress,
        per_atom_stresses=(key == "stresses"),
    )
    model_full = PairForcesModel(**common)
    model_half = PairForcesModel(**common, reduce_to_half_list=True)
    out_full = model_full(si_double_sim_state)
    out_half = model_half(si_double_sim_state)
    torch.testing.assert_close(out_half[key], out_full[key], rtol=1e-10, atol=1e-13)
