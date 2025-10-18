import torch

import torch_sim as ts
from tests.conftest import DTYPE
from torch_sim.constraints import FixAtoms, FixCom
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.transforms import unwrap_positions
from torch_sim.units import MetalUnits


def test_fix_com_nvt_langevin(
    ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel
):
    n_steps = 1000
    dt = torch.tensor(0.001, dtype=DTYPE)
    kT = torch.tensor(300, dtype=DTYPE) * MetalUnits.temperature

    dofs_before = ar_double_sim_state.calc_dof()
    ar_double_sim_state.constraints = [FixCom()]
    assert torch.allclose(ar_double_sim_state.calc_dof(), dofs_before - 3)

    state = ts.nvt_langevin_init(
        state=ar_double_sim_state, model=lj_model, kT=kT, seed=42
    )
    positions = []
    system_masses = torch.zeros((state.n_systems, 1), dtype=DTYPE).scatter_add_(
        0,
        state.system_idx.unsqueeze(-1).expand(-1, 1),
        state.masses.unsqueeze(-1),
    )
    for _step in range(n_steps):
        state = ts.nvt_langevin_step(model=lj_model, state=state, dt=dt, kT=kT)
        positions.append(state.positions.clone())
    traj_positions = torch.stack(positions)

    unwrapped_positions = unwrap_positions(
        traj_positions, ar_double_sim_state.cell, state.system_idx
    )
    coms = torch.zeros((n_steps, state.n_systems, 3), dtype=DTYPE).scatter_add_(
        1,
        state.system_idx[None, :, None].expand(n_steps, -1, 3),
        state.masses.unsqueeze(-1) * unwrapped_positions,
    )
    coms /= system_masses
    coms_drift = coms - coms[0]
    assert torch.allclose(coms_drift, torch.zeros_like(coms_drift), atol=1e-4)


def test_fix_atoms_nvt_langevin(
    ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel
):
    n_steps = 1000
    dt = torch.tensor(0.001, dtype=DTYPE)
    kT = torch.tensor(300, dtype=DTYPE) * MetalUnits.temperature

    dofs_before = ar_double_sim_state.calc_dof()
    ar_double_sim_state.constraints = [
        FixAtoms(indices=torch.tensor([0, 1], dtype=torch.long))
    ]
    assert torch.allclose(
        ar_double_sim_state.calc_dof(), dofs_before - torch.tensor([6, 0])
    )
    state = ts.nvt_langevin_init(
        state=ar_double_sim_state, model=lj_model, kT=kT, seed=42
    )
    positions = []
    for _step in range(n_steps):
        state = ts.nvt_langevin_step(model=lj_model, state=state, dt=dt, kT=kT)
        positions.append(state.positions.clone())
    traj_positions = torch.stack(positions)

    unwrapped_positions = unwrap_positions(
        traj_positions, ar_double_sim_state.cell, state.system_idx
    )
    diff_positions = unwrapped_positions - unwrapped_positions[0]
    assert torch.max(diff_positions[:, :2]) < 1e-8
    assert torch.max(diff_positions[:, 2:]) > 1e-2
