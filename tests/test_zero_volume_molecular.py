"""Tests for zero-volume (open / molecular) and finite non-periodic box setups.

See https://github.com/TorchSim/torch-sim/issues/549
"""

from collections.abc import Callable

import pytest
import torch

import torch_sim as ts
from tests.conftest import DEVICE, DTYPE
from tests.test_neighbors import _all_nl_backends
from torch_sim import neighbors, transforms
from torch_sim.autobatching import calculate_memory_scalers
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.testing import SIMSTATE_MOLECULE_GENERATORS


def _dimer_non_periodic(
    *,
    cell: torch.Tensor,
) -> ts.SimState:
    """Two Ar atoms along x; cell shape (1, 3, 3), pbc=False."""
    return ts.SimState(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [3.4, 0.0, 0.0]],
            dtype=DTYPE,
            device=DEVICE,
        ),
        masses=torch.full((2,), 39.948, dtype=DTYPE, device=DEVICE),
        cell=cell.to(dtype=DTYPE, device=DEVICE),
        pbc=torch.tensor([False, False, False], device=DEVICE),
        atomic_numbers=torch.full((2,), 18, dtype=torch.long, device=DEVICE),
        system_idx=torch.zeros(2, dtype=torch.long, device=DEVICE),
    )


@pytest.mark.parametrize("sim_state_name", tuple(SIMSTATE_MOLECULE_GENERATORS.keys()))
def test_molecular_generators_zero_volume_non_periodic(sim_state_name: str) -> None:
    """ASE-derived molecules use a zero cell and open boundaries."""
    state = SIMSTATE_MOLECULE_GENERATORS[sim_state_name](DEVICE, DTYPE)
    assert not state.pbc.any()
    assert state.cell.abs().sum() == 0
    assert state.volume.item() == 0.0


def test_zero_volume_wrap_and_mic_unchanged(benzene_sim_state: ts.SimState) -> None:
    """Singular/zero cells must not wrap positions or apply MIC."""
    state = benzene_sim_state.clone()
    displaced = state.positions + 100.0
    state.positions = displaced
    assert torch.allclose(state.wrap_positions, displaced)
    dr = torch.tensor([[10.0, 0.0, 0.0]], dtype=DTYPE, device=DEVICE)
    mic = transforms.minimum_image_displacement(dr=dr, cell=state.cell[0], pbc=state.pbc)
    torch.testing.assert_close(mic, dr)


def test_nve_zero_cell_remains_open(lj_model: LennardJonesModel) -> None:
    """Zero-volume cells have open boundaries (no reflecting faces)."""
    zero_cell = torch.zeros(1, 3, 3, dtype=DTYPE, device=DEVICE)
    state = _dimer_non_periodic(cell=zero_cell)
    kT = torch.tensor(500.0, dtype=DTYPE, device=DEVICE)
    md = ts.nve_init(state, lj_model, kT=kT)
    for _ in range(500):
        md = ts.nve_step(md, lj_model, dt=torch.tensor(0.005, dtype=DTYPE, device=DEVICE))
    span = md.positions.max() - md.positions.min()
    assert span.max().item() > 8.0


def test_memory_scalers_ignore_zero_volume(benzene_sim_state: ts.SimState) -> None:
    """Autobatching density uses atomic bounding box, not det(cell)=0."""
    scalers = calculate_memory_scalers(benzene_sim_state, "n_atoms_x_density")
    bbox = (
        benzene_sim_state.positions.max(dim=0).values
        - benzene_sim_state.positions.min(dim=0).values
        + 2.0
    )
    expected = benzene_sim_state.n_atoms**2 / (bbox.prod().item() / 1000)
    assert scalers[0] == pytest.approx(expected)


@pytest.mark.parametrize("nl_implementation", _all_nl_backends())
def test_neighbor_lists_zero_cell_molecule(
    nl_implementation: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    benzene_sim_state: ts.SimState,
) -> None:
    """Neighbor backends must handle zero cell + pbc=False (open molecule)."""
    cutoff = 5.0
    mapping, sys_map, shifts = nl_implementation(
        benzene_sim_state.positions,
        benzene_sim_state.cell,
        benzene_sim_state.pbc,
        cutoff,
        benzene_sim_state.system_idx,
    )
    assert mapping.shape[0] == 2
    assert mapping.shape[1] > 0
    assert (shifts == 0).all()
    assert sys_map.numel() == mapping.shape[1]


def test_pbc_wrap_singular_cell_even_if_pbc_true() -> None:
    """Zero cell is non-invertible: wrapping is skipped even if pbc flags are set."""
    cell = torch.zeros(1, 3, 3, dtype=DTYPE, device=DEVICE)
    pbc = torch.tensor([[True, True, True]], device=DEVICE)
    positions = torch.tensor([[5.0, 5.0, 5.0]], dtype=DTYPE, device=DEVICE)
    system_idx = torch.zeros(1, dtype=torch.long, device=DEVICE)
    wrapped, shifts = transforms.pbc_wrap_batched_and_get_lattice_shifts(
        positions, cell, system_idx, pbc
    )
    torch.testing.assert_close(wrapped, positions)
    assert (shifts == 0).all()


@pytest.mark.skipif(
    not neighbors.ALCHEMIOPS_AVAILABLE,
    reason="nvalchemiops is not installed",
)
def test_alchemiops_nominal_cell_only_for_nl(benzene_sim_state: ts.SimState) -> None:
    """Alchemiops may substitute a nominal cell internally; SimState cell stays zero."""
    cell_before = benzene_sim_state.cell.clone()
    neighbors.alchemiops_nl_cell_list(
        benzene_sim_state.positions,
        benzene_sim_state.cell,
        benzene_sim_state.pbc,
        5.0,
        benzene_sim_state.system_idx,
    )
    torch.testing.assert_close(benzene_sim_state.cell, cell_before)
