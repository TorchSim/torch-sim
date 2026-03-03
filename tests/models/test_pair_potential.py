"""Tests for the general pair potential model and pair forces model."""

import functools

import pytest
import torch
from ase.build import bulk

import torch_sim as ts
from tests.conftest import DEVICE, DTYPE
from tests.models.conftest import make_validate_model_outputs_test
from torch_sim.models.lennard_jones import LennardJonesModel, lennard_jones_pair
from torch_sim.models.morse import morse_pair
from torch_sim.models.pair_potential import (
    PairForcesModel,
    PairPotentialModel,
    full_to_half_list,
)
from torch_sim.models.particle_life import particle_life_pair_force
from torch_sim.models.soft_sphere import soft_sphere_pair


# Argon LJ parameters
LJ_SIGMA = 3.405
LJ_EPSILON = 0.0104
LJ_CUTOFF = 2.5 * LJ_SIGMA


@pytest.fixture
def lj_model_pp() -> PairPotentialModel:
    return PairPotentialModel(
        pair_fn=functools.partial(lennard_jones_pair, sigma=LJ_SIGMA, epsilon=LJ_EPSILON),
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
    fn = functools.partial(lennard_jones_pair, sigma=LJ_SIGMA, epsilon=LJ_EPSILON)
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
        pair_fn = functools.partial(
            lennard_jones_pair, sigma=LJ_SIGMA, epsilon=LJ_EPSILON
        )
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


def test_force_conservation(
    lj_model_pp: PairPotentialModel, si_double_sim_state: ts.SimState
) -> None:
    """Forces sum to zero (Newton's third law)."""
    out = lj_model_pp(si_double_sim_state)
    for sys_idx in range(si_double_sim_state.n_systems):
        mask = si_double_sim_state.system_idx == sys_idx
        assert torch.allclose(
            out["forces"][mask].sum(dim=0),
            torch.zeros(3, dtype=DTYPE),
            atol=1e-10,
        )


def test_stress_tensor_symmetry(
    lj_model_pp: PairPotentialModel, si_double_sim_state: ts.SimState
) -> None:
    """Stress tensor is symmetric."""
    out = lj_model_pp(si_double_sim_state)
    for i in range(si_double_sim_state.n_systems):
        stress = out["stress"][i]
        assert torch.allclose(stress, stress.T, atol=1e-10)


def test_multi_system(ar_double_sim_state: ts.SimState) -> None:
    """Multi-system batched evaluation matches single-system evaluation."""
    model = LennardJonesModel(
        sigma=3.405,
        epsilon=0.0104,
        cutoff=2.5 * 3.405,
        dtype=torch.float64,
        device=DEVICE,
        compute_forces=True,
        compute_stress=True,
    )
    out = model(ar_double_sim_state)

    assert out["energy"].shape == (ar_double_sim_state.n_systems,)
    # Both systems are identical, so energies should match
    torch.testing.assert_close(out["energy"][0], out["energy"][1], rtol=1e-10, atol=1e-10)


def test_unwrapped_positions_consistency() -> None:
    """Wrapped and unwrapped positions give identical results."""
    ar_atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat([2, 2, 2])
    cell = torch.tensor(ar_atoms.get_cell().array, dtype=torch.float64, device=DEVICE)

    state_wrapped = ts.io.atoms_to_state(ar_atoms, DEVICE, torch.float64)

    positions_unwrapped = state_wrapped.positions.clone()
    n_atoms = positions_unwrapped.shape[0]
    positions_unwrapped[: n_atoms // 2] += cell[0]
    positions_unwrapped[n_atoms // 4 : n_atoms // 2] -= cell[1]

    state_unwrapped = ts.SimState.from_state(state_wrapped, positions=positions_unwrapped)

    model = LennardJonesModel(
        sigma=3.405,
        epsilon=0.0104,
        cutoff=2.5 * 3.405,
        dtype=torch.float64,
        device=DEVICE,
        compute_forces=True,
        compute_stress=True,
    )

    results_wrapped = model(state_wrapped)
    results_unwrapped = model(state_unwrapped)

    for key in ("energy", "forces", "stress"):
        torch.testing.assert_close(
            results_wrapped[key], results_unwrapped[key], rtol=1e-10, atol=1e-10
        )
