"""Tests for default elastic reflection on non-periodic axes (finite cell)."""

import pytest
import torch

import torch_sim as ts
from tests.conftest import DEVICE, DTYPE
from torch_sim import transforms
from torch_sim.integrators.md import position_step
from torch_sim.models.lennard_jones import LennardJonesModel


def _dimer_in_box(cell_length: float) -> ts.SimState:
    return ts.SimState(
        positions=torch.tensor(
            [[1.0, 1.0, 1.0], [4.4, 1.0, 1.0]],
            dtype=DTYPE,
            device=DEVICE,
        ),
        masses=torch.full((2,), 39.948, dtype=DTYPE, device=DEVICE),
        cell=torch.eye(3, dtype=DTYPE, device=DEVICE).unsqueeze(0) * cell_length,
        pbc=torch.tensor([False, False, False], device=DEVICE),
        atomic_numbers=torch.full((2,), 18, dtype=torch.long, device=DEVICE),
        system_idx=torch.zeros(2, dtype=torch.long, device=DEVICE),
    )


def test_apply_reflect_returns_new_tensors() -> None:
    """Boundary application returns reflected tensors without mutating inputs."""
    positions = torch.tensor([[11.0, 1.0, 1.0]], dtype=DTYPE, device=DEVICE)
    momenta = torch.tensor([[10.0, 0.0, 0.0]], dtype=DTYPE, device=DEVICE)
    masses = torch.tensor([1.0], dtype=DTYPE, device=DEVICE)
    cell = torch.eye(3, dtype=DTYPE, device=DEVICE).unsqueeze(0) * 10.0
    system_idx = torch.zeros(1, dtype=torch.long, device=DEVICE)
    pbc = torch.tensor([False, False, False], device=DEVICE)
    pos_in, mom_in = positions.clone(), momenta.clone()
    out_pos, out_mom = transforms.apply_nonperiodic_reflecting_boundaries(
        positions, cell, system_idx, pbc, momenta=momenta, masses=masses
    )
    assert positions[0, 0].item() == 11.0
    assert momenta[0, 0].item() == 10.0
    assert out_pos[0, 0].item() == pytest.approx(9.0)
    assert out_mom is not None
    assert out_mom[0, 0].item() == pytest.approx(-10.0)
    torch.testing.assert_close(positions, pos_in)
    torch.testing.assert_close(momenta, mom_in)


def test_reflect_skips_zero_cell() -> None:
    """Singular cells do not reflect."""
    positions = torch.tensor([[50.0, 0.0, 0.0]], dtype=DTYPE, device=DEVICE)
    cell = torch.zeros(1, 3, 3, dtype=DTYPE, device=DEVICE)
    system_idx = torch.zeros(1, dtype=torch.long, device=DEVICE)
    pbc = torch.tensor([False, False, False], device=DEVICE)
    out_pos, _ = transforms.apply_nonperiodic_reflecting_boundaries(
        positions, cell, system_idx, pbc
    )
    assert out_pos[0, 0].item() == 50.0


def test_position_step_assigns_reflected_momenta(lj_model: LennardJonesModel) -> None:
    """MD position_step must assign bounced momenta, not only positions."""
    sim = _dimer_in_box(10.0)
    md = ts.nve_init(sim, lj_model, kT=torch.tensor(0.0, dtype=DTYPE, device=DEVICE))
    md.positions = torch.tensor(
        [[9.99, 5.0, 5.0], [4.4, 1.0, 1.0]], dtype=DTYPE, device=DEVICE
    )
    md.momenta = torch.tensor(
        [[200.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=DTYPE, device=DEVICE
    )
    md.forces = torch.zeros_like(md.momenta)
    mom_before = md.momenta.clone()
    position_step(md, dt=torch.tensor(0.05, dtype=DTYPE, device=DEVICE))
    assert md.positions[0, 0].item() < 10.0
    assert md.positions[0, 0].item() > 0.0
    assert md.momenta[0, 0].item() < 0.0
    assert md.momenta is not mom_before


def test_simstate_position_step_does_not_touch_momenta() -> None:
    """SimState boundary path reflects positions only (no momenta on base state)."""
    sim = _dimer_in_box(10.0)
    sim.positions = torch.tensor(
        [[9.5, 5.0, 5.0], [4.4, 1.0, 1.0]], dtype=DTYPE, device=DEVICE
    )
    trial = sim.positions.clone()
    trial[0, 0] = 11.0
    sim.set_constrained_positions(trial)
    assert sim.positions[0, 0].item() == pytest.approx(9.0)


def test_nve_finite_box_confines_atoms(lj_model: LennardJonesModel) -> None:
    """Finite cell + pbc=False keeps atoms inside the box."""
    state = _dimer_in_box(8.0)
    kT = torch.tensor(500.0, dtype=DTYPE, device=DEVICE)
    md = ts.nve_init(state, lj_model, kT=kT)
    for _ in range(500):
        md = ts.nve_step(md, lj_model, dt=torch.tensor(0.005, dtype=DTYPE, device=DEVICE))
    assert md.positions.min().item() >= -1e-6
    assert md.positions.max().item() <= 8.0 + 1e-6


def test_mixed_pbc_reflects_only_nonperiodic_axes() -> None:
    """Periodic x wraps; non-periodic y reflects at cell faces."""
    cell = torch.diag(
        torch.tensor([10.0, 5.0, 10.0], dtype=DTYPE, device=DEVICE)
    ).unsqueeze(0)
    positions = torch.tensor([[15.0, -1.0, 3.0]], dtype=DTYPE, device=DEVICE)
    system_idx = torch.zeros(1, dtype=torch.long, device=DEVICE)
    pbc = torch.tensor([True, False, True], device=DEVICE)
    wrapped = transforms.pbc_wrap_batched(positions, cell, system_idx, pbc)
    assert wrapped[0, 0].item() == pytest.approx(5.0)
    out_pos, _ = transforms.apply_nonperiodic_reflecting_boundaries(
        wrapped, cell, system_idx, pbc
    )
    assert out_pos[0, 1].item() == pytest.approx(1.0)
