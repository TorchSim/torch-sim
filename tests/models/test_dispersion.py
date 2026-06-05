"""Tests for the D3DispersionModel wrapper."""

from __future__ import annotations

import traceback

import pytest
import torch

import torch_sim as ts
from tests.conftest import DEVICE, DTYPE
from tests.models.conftest import make_validate_model_outputs_test


try:
    from nvalchemiops.torch.interactions.dispersion import D3Parameters

    from torch_sim.models.dispersion import D3DispersionModel

    _IMPORT_ERROR: str | None = None
except (ImportError, OSError, RuntimeError):
    _IMPORT_ERROR = traceback.format_exc()

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None, reason=f"nvalchemiops not installed: {_IMPORT_ERROR}"
)


def _make_d3_params(device: torch.device = DEVICE) -> D3Parameters:
    """Build minimal D3 reference parameters for testing (elements up to Fe=26)."""
    max_z = 26
    mesh = 5
    return D3Parameters(
        rcov=torch.rand(max_z + 1, device=device),
        r4r2=torch.rand(max_z + 1, device=device),
        c6ab=torch.rand(max_z + 1, max_z + 1, mesh, mesh, device=device),
        cn_ref=torch.rand(max_z + 1, max_z + 1, mesh, mesh, device=device),
    )


# BJ damping parameters from
# https://github.com/dftd3/simple-dftd3/blob/main/assets/parameters.toml
PBE_BJ = {"a1": 0.4289, "s8": 0.7875, "a2": 4.4407, "s6": 1.0}
R2SCAN_BJ = {"a1": 0.49484001, "s8": 0.78981345, "a2": 5.73083694, "s6": 1.0}


@pytest.fixture
def d3_model_pbe() -> D3DispersionModel:
    return D3DispersionModel(
        **PBE_BJ,
        d3_params=_make_d3_params(),
        cutoff=12.0,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )


@pytest.fixture
def d3_model_r2scan() -> D3DispersionModel:
    return D3DispersionModel(
        **R2SCAN_BJ,
        d3_params=_make_d3_params(),
        cutoff=12.0,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )


def test_d3_stress_matches_finite_strain_sign() -> None:
    """Stress should match dE/dstrain/V, not the opposite virial sign."""
    row_cell = torch.tensor(
        [[4.2, 0.3, 0.1], [0.2, 4.8, 0.4], [0.15, 0.35, 5.1]],
        dtype=DTYPE,
        device=DEVICE,
    )
    positions = torch.tensor(
        [[0.4, 0.5, 0.6], [1.9, 1.4, 2.3], [3.1, 2.6, 1.7]],
        dtype=DTYPE,
        device=DEVICE,
    )
    state = ts.SimState(
        positions=positions,
        masses=torch.ones(3, dtype=DTYPE, device=DEVICE),
        cell=row_cell.mT.unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.tensor([6, 8, 14], dtype=torch.int64, device=DEVICE),
    )

    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(1234)
    max_z = 14
    mesh = 5
    rcov = torch.rand(max_z + 1, generator=gen, device=DEVICE) + 0.5
    r4r2 = torch.rand(max_z + 1, generator=gen, device=DEVICE) + 0.5
    c6ab = 20.0 * (
        torch.rand(
            max_z + 1,
            max_z + 1,
            mesh,
            mesh,
            generator=gen,
            device=DEVICE,
        )
        + 0.1
    )
    c6ab = 0.5 * (c6ab + c6ab.permute(1, 0, 3, 2))
    cn_ref = 4.0 * torch.rand(
        max_z + 1,
        max_z + 1,
        mesh,
        mesh,
        generator=gen,
        device=DEVICE,
    )
    cn_ref = 0.5 * (cn_ref + cn_ref.permute(1, 0, 3, 2))

    model = D3DispersionModel(
        **PBE_BJ,
        d3_params=D3Parameters(rcov=rcov, r4r2=r4r2, c6ab=c6ab, cn_ref=cn_ref),
        cutoff=6.0,
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


test_d3_pbe_outputs = make_validate_model_outputs_test(
    model_fixture_name="d3_model_pbe", device=DEVICE, dtype=DTYPE
)

test_d3_r2scan_outputs = make_validate_model_outputs_test(
    model_fixture_name="d3_model_r2scan", device=DEVICE, dtype=DTYPE
)
