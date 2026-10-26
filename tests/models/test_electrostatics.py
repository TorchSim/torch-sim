"""Tests for the electrostatics ModelInterface wrappers."""

from __future__ import annotations

import math
import traceback

import pytest
import torch
from ase.build import bulk

import torch_sim as ts
from tests.conftest import DEVICE, DTYPE
from tests.models.conftest import make_validate_model_outputs_test
from torch_sim.units import UnitConversion


try:
    from torch_sim.models.electrostatics import DSFCoulombModel, EwaldModel, PMEModel

    _IMPORT_ERROR: str | None = None
except (ImportError, OSError, RuntimeError):
    _IMPORT_ERROR = traceback.format_exc()

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None, reason=f"nvalchemiops not installed: {_IMPORT_ERROR}"
)


def _make_charged_state(
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
    repeat: int = 1,
) -> ts.SimState:
    """Build a small NaCl-like state with alternating +1/-1 site charges."""
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64, cubic=True).repeat(repeat)
    state = ts.io.atoms_to_state(atoms, device, dtype)
    n = state.n_atoms
    charges = torch.empty(n, dtype=dtype, device=device)
    charges[::2] = 1.0
    charges[1::2] = -1.0
    state._atom_extras["partial_charges"] = charges  # noqa: SLF001
    return state


@pytest.fixture
def dsf_model() -> DSFCoulombModel:
    return DSFCoulombModel(cutoff=8.0, alpha=0.2, device=DEVICE, dtype=DTYPE)


@pytest.fixture
def ewald_model() -> EwaldModel:
    return EwaldModel(cutoff=8.0, accuracy=1e-6, device=DEVICE, dtype=DTYPE)


@pytest.fixture
def pme_model() -> PMEModel:
    return PMEModel(cutoff=8.0, accuracy=1e-6, device=DEVICE, dtype=DTYPE)


def _add_partial_charges(state: ts.SimState) -> ts.SimState:
    """Inject alternating +/-0.5 site charges into a state."""
    n = state.n_atoms
    charges = torch.zeros(n, dtype=state.positions.dtype, device=state.device)
    charges[::2] = 0.5
    charges[1::2] = -0.5
    state._atom_extras["partial_charges"] = charges  # noqa: SLF001
    return state


test_dsf_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="dsf_model",
    device=DEVICE,
    dtype=DTYPE,
    state_modifiers=[_add_partial_charges],
)
test_ewald_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="ewald_model",
    device=DEVICE,
    dtype=DTYPE,
    state_modifiers=[_add_partial_charges],
)
test_pme_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="pme_model",
    device=DEVICE,
    dtype=DTYPE,
    state_modifiers=[_add_partial_charges],
)


def test_dsf_nonzero_energy() -> None:
    """Charged system should produce nonzero electrostatic energy."""
    model = DSFCoulombModel(cutoff=8.0, alpha=0.2, device=DEVICE, dtype=DTYPE)
    state = _make_charged_state()
    out = model(state)
    assert out["energy"].abs().item() > 0


@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        pytest.param(DSFCoulombModel, {"cutoff": 8.0, "alpha": 0.2}, id="dsf"),
        pytest.param(EwaldModel, {"cutoff": 8.0, "accuracy": 1e-6}, id="ewald"),
        pytest.param(
            PMEModel,
            {"cutoff": 8.0, "accuracy": 1e-6, "mesh_spacing": 1.0},
            id="pme",
        ),
    ],
)
def test_electrostatics_stress_matches_finite_strain_sign(
    model_cls: type[DSFCoulombModel | EwaldModel | PMEModel],
    kwargs: dict[str, float],
) -> None:
    """Electrostatic stress should match dE/dstrain/V, not the virial sign."""
    row_cell = torch.tensor(
        [[5.2, 0.3, 0.1], [0.2, 5.6, 0.4], [0.15, 0.35, 6.1]],
        dtype=DTYPE,
        device=DEVICE,
    )
    positions = torch.tensor(
        [[0.4, 0.5, 0.6], [1.9, 1.4, 2.3], [3.1, 2.6, 1.7], [4.0, 3.4, 4.2]],
        dtype=DTYPE,
        device=DEVICE,
    )
    charges = torch.tensor([0.8, -0.7, 0.4, -0.5], dtype=DTYPE, device=DEVICE)
    state = ts.SimState(
        positions=positions,
        masses=torch.ones(4, dtype=DTYPE, device=DEVICE),
        cell=row_cell.mT.unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.tensor([11, 17, 11, 17], dtype=torch.int64, device=DEVICE),
    )
    state._atom_extras["partial_charges"] = charges  # noqa: SLF001

    model = model_cls(
        **kwargs,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )

    stress = model(state)["stress"][0]
    volume = state.volume[0]
    frac_positions = torch.linalg.solve(row_cell.mT, positions.mT).mT
    identity = torch.eye(3, dtype=DTYPE, device=DEVICE)

    def strained_energy(strain: torch.Tensor) -> torch.Tensor:
        strained_row_cell = row_cell @ (identity + strain)
        strained_state = ts.SimState(
            positions=frac_positions @ strained_row_cell,
            masses=state.masses,
            cell=strained_row_cell.mT.unsqueeze(0),
            pbc=state.pbc,
            atomic_numbers=state.atomic_numbers,
            system_idx=state.system_idx,
        )
        strained_state._atom_extras["partial_charges"] = charges  # noqa: SLF001
        return model(strained_state)["energy"][0]

    step = 1e-3
    finite_diff_stress = torch.zeros((3, 3), dtype=DTYPE, device=DEVICE)
    for idx_i in range(3):
        for idx_j in range(idx_i, 3):
            strain = torch.zeros((3, 3), dtype=DTYPE, device=DEVICE)
            if idx_i == idx_j:
                strain[idx_i, idx_i] = step
                energy_plus = strained_energy(strain)
                strain[idx_i, idx_i] = -step
                energy_minus = strained_energy(strain)
            else:
                strain[idx_i, idx_j] = 0.5 * step
                strain[idx_j, idx_i] = 0.5 * step
                energy_plus = strained_energy(strain)
                strain[idx_i, idx_j] = -0.5 * step
                strain[idx_j, idx_i] = -0.5 * step
                energy_minus = strained_energy(strain)
            stress_component = (energy_plus - energy_minus) / (2 * step * volume)
            finite_diff_stress[idx_i, idx_j] = stress_component
            finite_diff_stress[idx_j, idx_i] = stress_component

    torch.testing.assert_close(
        stress,
        finite_diff_stress,
        rtol=5e-3,
        atol=5e-7,
    )


def test_ewald_pme_energy_agreement() -> None:
    """Ewald and PME should give the same converged Coulomb energy."""
    state = _make_charged_state()
    ewald = EwaldModel(cutoff=8.0, accuracy=1e-6, device=DEVICE, dtype=DTYPE)
    pme = PMEModel(cutoff=8.0, accuracy=1e-6, device=DEVICE, dtype=DTYPE)
    torch.testing.assert_close(
        ewald(state)["energy"], pme(state)["energy"], atol=1e-3, rtol=1e-3
    )


def _slab_reference_energy_forces(
    positions: torch.Tensor, charges: torch.Tensor, box: float, n_max: int = 100
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference slab energy and forces from a direct Coulomb sum over x and y images.

    Images within a circle of radius n_max * box are summed directly, and the
    cells outside it are added analytically as point dipoles.
    """
    positions = positions.clone().requires_grad_()
    # In-plane image shifts within the circle
    idx = torch.arange(-n_max, n_max + 1, dtype=positions.dtype, device=DEVICE)
    # The z shift is always zero because a slab has no images along z
    shifts = torch.cartesian_prod(idx, idx, torch.zeros_like(idx[:1])) * box
    shifts = shifts[shifts.norm(dim=-1) <= n_max * box]
    # Pair distances and charge products over all images
    pair_vectors = positions[:, None] - positions[None, :] + shifts[:, None, None]
    dist_sq = pair_vectors.pow(2).sum(dim=-1)
    pair_charges = (charges[:, None] * charges[None, :]).expand_as(dist_sq)
    # Direct sum without the i = j, n = 0 terms
    is_pair = dist_sq > 0
    energy = 0.5 * (pair_charges[is_pair] * dist_sq[is_pair].rsqrt()).sum()
    # Dipole term of the cells outside the circle
    dipole_sq = (charges[:, None] * positions).sum(dim=0).pow(2)
    energy = energy + math.pi * (dipole_sq[2] - 0.5 * dipole_sq[:2].sum()) / (
        n_max * box**3
    )
    # e^2 / Angstrom to eV
    energy = energy * UnitConversion.e2_per_Ang_to_eV
    forces = -torch.autograd.grad(energy, positions)[0]
    return energy.detach(), forces


@pytest.mark.parametrize(
    "model_cls",
    [pytest.param(EwaldModel, id="ewald"), pytest.param(PMEModel, id="pme")],
)
def test_slab_matches_2d_lattice_sum(model_cls: type[EwaldModel | PMEModel]) -> None:
    """A slab state should match the direct Coulomb sum over its 2D lattice."""
    box = 6.0
    positions = torch.tensor(
        [[0.3, 0.4, 8.8], [3.1, 2.7, 11.1], [1.4, 4.2, 10.6], [4.4, 1.0, 9.3]],
        dtype=DTYPE,
        device=DEVICE,
    )
    charges = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=DTYPE, device=DEVICE)
    state = ts.SimState(
        positions=positions,
        masses=torch.ones(4, dtype=DTYPE, device=DEVICE),
        cell=torch.diag(
            torch.tensor([box, box, 20.0], dtype=DTYPE, device=DEVICE)
        ).unsqueeze(0),
        pbc=[True, True, False],
        atomic_numbers=torch.tensor([11, 17, 11, 17], dtype=torch.int64, device=DEVICE),
    )
    state._atom_extras["partial_charges"] = charges  # noqa: SLF001

    model = model_cls(device=DEVICE, dtype=DTYPE)
    out = model(state)
    energy, forces = _slab_reference_energy_forces(positions, charges, box)
    torch.testing.assert_close(out["energy"][0], energy, atol=1e-4, rtol=0)
    torch.testing.assert_close(out["forces"], forces, atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "model_cls",
    [pytest.param(EwaldModel, id="ewald"), pytest.param(PMEModel, id="pme")],
)
def test_bulk_matches_madelung_energy(model_cls: type[EwaldModel | PMEModel]) -> None:
    """A 64-atom NaCl crystal should match its Madelung energy at the default cutoff."""
    state = _make_charged_state(repeat=2)
    madelung = 1.7475645946  # rocksalt, per ion pair at the nearest-neighbor distance
    nearest_neighbor = 5.64 / 2
    energy = -state.n_atoms / 2 * madelung / nearest_neighbor
    energy *= UnitConversion.e2_per_Ang_to_eV
    out = model_cls(device=DEVICE, dtype=DTYPE)(state)
    torch.testing.assert_close(
        out["energy"][0],
        torch.tensor(energy, dtype=DTYPE, device=DEVICE),
        atol=2e-4,
        rtol=0,
    )


def test_sum_model_lj_plus_dsf() -> None:
    """LJ + DSF should be additive through SumModel."""
    from torch_sim.models.interface import SumModel
    from torch_sim.models.lennard_jones import LennardJonesModel

    lj = LennardJonesModel(
        sigma=2.8, epsilon=0.01, cutoff=7.0, device=DEVICE, dtype=DTYPE
    )
    dsf = DSFCoulombModel(cutoff=8.0, alpha=0.2, device=DEVICE, dtype=DTYPE)
    combined = SumModel(lj, dsf)
    state = _make_charged_state()
    lj_out = lj(state)
    dsf_out = dsf(state)
    sum_out = combined(state)
    torch.testing.assert_close(sum_out["energy"], lj_out["energy"] + dsf_out["energy"])
    torch.testing.assert_close(sum_out["forces"], lj_out["forces"] + dsf_out["forces"])
