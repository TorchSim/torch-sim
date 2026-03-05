"""Tests for the general pair potential model and pair forces model."""

import functools

import pytest
import torch
from ase.build import bulk

import torch_sim as ts
from tests.conftest import DEVICE, DTYPE
from tests.models.conftest import make_validate_model_outputs_test
from torch_sim import io
from torch_sim.models.lennard_jones import LennardJonesModel, lennard_jones_pair
from torch_sim.models.morse import morse_pair
from torch_sim.models.pair_potential import (
    PairForcesModel,
    PairPotentialModel,
    full_to_half_list,
)
from torch_sim.models.particle_life import particle_life_pair_force
from torch_sim.models.soft_sphere import soft_sphere_pair
from torch_sim.neighbors import torch_nl_n2


# BMHTF (Born-Meyer-Huggins-Tosi-Fumi) potential for NaCl
# Na-Cl interaction parameters
BMHTF_A = 20.3548
BMHTF_B = 3.1546
BMHTF_C = 674.4793
BMHTF_D = 837.0770
BMHTF_SIGMA = 2.755
BMHTF_CUTOFF = 10.0


def bmhtf_pair(
    dr: torch.Tensor,
    zi: torch.Tensor,  # noqa: ARG001
    zj: torch.Tensor,  # noqa: ARG001
    A: float,
    B: float,
    C: float,
    D: float,
    sigma: float,
) -> torch.Tensor:
    """Born-Meyer-Huggins-Tosi-Fumi (BMHTF) potential for ionic crystals."""
    exp_term = A * torch.exp(B * (sigma - dr))
    r6_term = C / dr.pow(6)
    r8_term = D / dr.pow(8)
    energy = exp_term - r6_term - r8_term
    return torch.where(dr > 0, energy, torch.zeros_like(energy))


@pytest.fixture
def nacl_sim_state() -> ts.SimState:
    """NaCl structure for BMHTF potential tests."""
    nacl_atoms = bulk("NaCl", "rocksalt", a=5.64)
    return io.atoms_to_state(nacl_atoms, device=DEVICE, dtype=DTYPE)


@pytest.fixture
def bmhtf_model_pp() -> PairPotentialModel:
    """BMHTF model using PairPotentialModel to test general case."""
    return PairPotentialModel(
        pair_fn=functools.partial(
            bmhtf_pair,
            A=BMHTF_A,
            B=BMHTF_B,
            C=BMHTF_C,
            D=BMHTF_D,
            sigma=BMHTF_SIGMA,
        ),
        cutoff=BMHTF_CUTOFF,
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
    model_fixture_name="bmhtf_model_pp", device=DEVICE, dtype=DTYPE
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
    # Argon LJ parameters
    sigma = 3.405
    epsilon = 0.0104
    cutoff = 2.5 * sigma
    fn = functools.partial(lennard_jones_pair, sigma=sigma, epsilon=epsilon)
    needs_forces = key in ("forces", "stress", "stresses")
    needs_stress = key in ("stress", "stresses")
    common = dict(
        pair_fn=fn,
        cutoff=cutoff,
        dtype=si_double_sim_state.dtype,
        compute_forces=needs_forces,
        compute_stress=needs_stress,
        per_atom_stresses=(key == "stresses"),
    )
    model_full = PairPotentialModel(**common)
    model_half = PairPotentialModel(**common, reduce_to_half_list=True)
    out_full = model_full(si_double_sim_state)
    out_half = model_half(si_double_sim_state)
    torch.testing.assert_close(out_half[key], out_full[key], rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("potential", ["bmhtf", "morse", "soft_sphere"])
def test_autograd_force_fn_matches_potential_model(
    nacl_sim_state: ts.SimState,
    si_double_sim_state: ts.SimState,
    potential: str,
) -> None:
    """PairForcesModel with -dV/dr force fn matches PairPotentialModel forces/stress."""
    # Use NaCl for BMHTF, si_double for others
    sim_state = nacl_sim_state if potential == "bmhtf" else si_double_sim_state
    if potential == "bmhtf":
        pair_fn = functools.partial(
            bmhtf_pair,
            A=BMHTF_A,
            B=BMHTF_B,
            C=BMHTF_C,
            D=BMHTF_D,
            sigma=BMHTF_SIGMA,
        )
        cutoff = BMHTF_CUTOFF
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
        dtype=sim_state.dtype,
        compute_forces=True,
        compute_stress=True,
        per_atom_stresses=True,
    )
    model_pf = PairForcesModel(
        force_fn=force_fn,
        cutoff=cutoff,
        dtype=sim_state.dtype,
        compute_stress=True,
        per_atom_stresses=True,
    )
    out_pp = model_pp(sim_state)
    out_pf = model_pf(sim_state)

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
    bmhtf_model_pp: PairPotentialModel, nacl_sim_state: ts.SimState
) -> None:
    """Forces sum to zero (Newton's third law)."""
    out = bmhtf_model_pp(nacl_sim_state)
    for sys_idx in range(nacl_sim_state.n_systems):
        mask = nacl_sim_state.system_idx == sys_idx
        assert torch.allclose(
            out["forces"][mask].sum(dim=0),
            torch.zeros(3, dtype=nacl_sim_state.dtype),
            atol=1e-10,
        )


def test_stress_tensor_symmetry(
    bmhtf_model_pp: PairPotentialModel, nacl_sim_state: ts.SimState
) -> None:
    """Stress tensor is symmetric."""
    out = bmhtf_model_pp(nacl_sim_state)
    for i in range(nacl_sim_state.n_systems):
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


def test_retain_graph_allows_param_grad(nacl_sim_state: ts.SimState) -> None:
    """With retain_graph=True, energy graph survives force computation so we can
    differentiate energy w.r.t. model parameters (e.g. A, B, C, D)."""
    A = torch.tensor(BMHTF_A, dtype=nacl_sim_state.dtype, requires_grad=True)
    pair_fn = functools.partial(
        bmhtf_pair,
        A=A,
        B=BMHTF_B,
        C=BMHTF_C,
        D=BMHTF_D,
        sigma=BMHTF_SIGMA,
    )
    model = PairPotentialModel(
        pair_fn=pair_fn,
        cutoff=BMHTF_CUTOFF,
        dtype=nacl_sim_state.dtype,
        compute_forces=True,
        neighbor_list_fn=torch_nl_n2,
        retain_graph=True,
    )
    out = model(nacl_sim_state)
    assert out["forces"] is not None
    (grad,) = torch.autograd.grad(out["energy"].sum(), A)
    assert grad.shape == A.shape
    assert grad.abs() > 0


def test_no_retain_graph_frees_graph(nacl_sim_state: ts.SimState) -> None:
    """Without retain_graph, differentiating energy w.r.t. parameters after force
    computation raises because the graph has been freed."""
    A = torch.tensor(BMHTF_A, dtype=nacl_sim_state.dtype, requires_grad=True)
    pair_fn = functools.partial(
        bmhtf_pair,
        A=A,
        B=BMHTF_B,
        C=BMHTF_C,
        D=BMHTF_D,
        sigma=BMHTF_SIGMA,
    )
    model = PairPotentialModel(
        pair_fn=pair_fn,
        cutoff=BMHTF_CUTOFF,
        dtype=nacl_sim_state.dtype,
        compute_forces=True,
        neighbor_list_fn=torch_nl_n2,
        retain_graph=False,
    )
    out = model(nacl_sim_state)
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(out["energy"].sum(), A)
