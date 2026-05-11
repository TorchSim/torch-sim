"""Tests for the electrostatics ModelInterface wrappers."""

import traceback  # noqa: I001

import pytest
import torch
from ase.build import bulk

import torch_sim as ts
from tests.conftest import DEVICE, DTYPE
from tests.models.conftest import make_validate_model_outputs_test

try:
    from torch_sim.models.electrostatics import DSFCoulombModel, EwaldModel, PMEModel
except (ImportError, OSError, RuntimeError):
    pytest.skip(
        f"nvalchemiops not installed: {traceback.format_exc()}",
        allow_module_level=True,
    )


def _make_charged_state(
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
) -> ts.SimState:
    """Build a small NaCl-like state with alternating +1/-1 site charges."""
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64, cubic=True)
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


def test_ewald_pme_energy_agreement() -> None:
    """Ewald and PME should give the same converged Coulomb energy."""
    state = _make_charged_state()
    ewald = EwaldModel(cutoff=8.0, accuracy=1e-6, device=DEVICE, dtype=DTYPE)
    pme = PMEModel(cutoff=8.0, accuracy=1e-6, device=DEVICE, dtype=DTYPE)
    torch.testing.assert_close(
        ewald(state)["energy"], pme(state)["energy"], atol=1e-3, rtol=1e-3
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
