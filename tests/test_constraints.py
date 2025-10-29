from typing import get_args

import pytest
import torch

import torch_sim as ts
from tests.conftest import DTYPE
from torch_sim.constraints import FixAtoms, FixCom
from torch_sim.models.interface import ModelInterface
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.optimizers import FireFlavor
from torch_sim.transforms import get_centers_of_mass, unwrap_positions
from torch_sim.units import MetalUnits


def test_fix_com(ar_supercell_sim_state: ts.SimState, lj_model: LennardJonesModel):
    """Test adjustment of positions and momenta with FixCom constraint."""
    ar_supercell_sim_state.add_constraints([FixCom()])
    initial_positions = ar_supercell_sim_state.positions.clone()
    ar_supercell_sim_state.set_positions(initial_positions + 0.5)
    assert torch.allclose(ar_supercell_sim_state.positions, initial_positions, atol=1e-8)

    ar_supercell_mdstate = ts.nve_init(
        state=ar_supercell_sim_state,
        model=lj_model,
        kT=torch.tensor(10.0, dtype=DTYPE),
        seed=42,
    )
    ar_supercell_mdstate.set_momenta(torch.randn_like(ar_supercell_mdstate.momenta) * 0.1)
    assert torch.allclose(
        ar_supercell_mdstate.momenta.mean(dim=0), torch.zeros(3, dtype=DTYPE), atol=1e-8
    )


def test_fix_atoms(ar_supercell_sim_state: ts.SimState, lj_model: LennardJonesModel):
    """Test adjustment of positions and momenta with FixAtoms constraint."""
    indices_to_fix = torch.tensor([0, 5, 10], dtype=torch.long)
    ar_supercell_sim_state.add_constraints([FixAtoms(indices=indices_to_fix)])
    initial_positions = ar_supercell_sim_state.positions.clone()
    # displacement = torch.randn_like(ar_supercell_sim_state.positions) * 0.5
    displacement = 0.5
    ar_supercell_sim_state.set_positions(initial_positions + displacement)
    assert torch.allclose(
        ar_supercell_sim_state.positions[indices_to_fix],
        initial_positions[indices_to_fix],
        atol=1e-8,
    )
    # Check that other positions have changed
    unfixed_indices = torch.tensor(
        [i for i in range(ar_supercell_sim_state.n_atoms) if i not in indices_to_fix],
        dtype=torch.long,
    )
    assert not torch.allclose(
        ar_supercell_sim_state.positions[unfixed_indices],
        initial_positions[unfixed_indices],
        atol=1e-8,
    )

    ar_supercell_mdstate = ts.nve_init(
        state=ar_supercell_sim_state,
        model=lj_model,
        kT=torch.tensor(10.0, dtype=DTYPE),
        seed=42,
    )
    ar_supercell_mdstate.set_momenta(torch.randn_like(ar_supercell_mdstate.momenta) * 0.1)
    assert torch.allclose(
        ar_supercell_mdstate.momenta[indices_to_fix],
        torch.zeros_like(ar_supercell_mdstate.momenta[indices_to_fix]),
        atol=1e-8,
    )


def test_fix_com_nvt_langevin(cu_sim_state: ts.SimState, lj_model: LennardJonesModel):
    """Test FixCom constraint in NVT Langevin dynamics."""
    n_steps = 1000
    dt = torch.tensor(0.001, dtype=DTYPE)
    kT = torch.tensor(300, dtype=DTYPE) * MetalUnits.temperature

    dofs_before = cu_sim_state.calc_dof()
    cu_sim_state.constraints = [FixCom()]
    assert torch.allclose(cu_sim_state.calc_dof(), dofs_before - 3)

    state = ts.nvt_langevin_init(state=cu_sim_state, model=lj_model, kT=kT, seed=42)
    positions = []
    system_masses = torch.zeros((state.n_systems, 1), dtype=DTYPE).scatter_add_(
        0,
        state.system_idx.unsqueeze(-1).expand(-1, 1),
        state.masses.unsqueeze(-1),
    )
    temperatures = []
    for _step in range(n_steps):
        state = ts.nvt_langevin_step(model=lj_model, state=state, dt=dt, kT=kT)
        positions.append(state.positions.clone())
        temp = ts.calc_kT(
            masses=state.masses,
            momenta=state.momenta,
            system_idx=state.system_idx,
            dof_per_system=state.calc_dof(),
        )
        temperatures.append(temp / MetalUnits.temperature)
    temperatures = torch.stack(temperatures)

    traj_positions = torch.stack(positions)

    # unwrapped_positions = unwrap_positions(
    #     traj_positions, ar_double_sim_state.cell, state.system_idx
    # )
    coms = torch.zeros((n_steps, state.n_systems, 3), dtype=DTYPE).scatter_add_(
        1,
        state.system_idx[None, :, None].expand(n_steps, -1, 3),
        state.masses.unsqueeze(-1) * traj_positions,
    )
    coms /= system_masses
    coms_drift = coms - coms[0]
    assert torch.allclose(coms_drift, torch.zeros_like(coms_drift), atol=1e-6)
    assert (torch.mean(temperatures[len(temperatures) // 2 :]) - 300) / 300 < 0.30


def test_fix_atoms_nvt_langevin(cu_sim_state: ts.SimState, lj_model: LennardJonesModel):
    """Test FixAtoms constraint in NVT Langevin dynamics."""
    n_steps = 1000
    dt = torch.tensor(0.001, dtype=DTYPE)
    kT = torch.tensor(300, dtype=DTYPE) * MetalUnits.temperature

    dofs_before = cu_sim_state.calc_dof()
    cu_sim_state.constraints = [FixAtoms(indices=torch.tensor([0, 1], dtype=torch.long))]
    assert torch.allclose(cu_sim_state.calc_dof(), dofs_before - torch.tensor([6]))
    state = ts.nvt_langevin_init(state=cu_sim_state, model=lj_model, kT=kT, seed=42)
    positions = []
    temperatures = []
    for _step in range(n_steps):
        state = ts.nvt_langevin_step(model=lj_model, state=state, dt=dt, kT=kT)
        positions.append(state.positions.clone())
        temp = ts.calc_kT(
            masses=state.masses,
            momenta=state.momenta,
            system_idx=state.system_idx,
            dof_per_system=state.calc_dof(),
        )
        temperatures.append(temp / MetalUnits.temperature)
    temperatures = torch.stack(temperatures)
    traj_positions = torch.stack(positions)

    unwrapped_positions = unwrap_positions(
        traj_positions, cu_sim_state.cell, state.system_idx
    )
    diff_positions = unwrapped_positions - unwrapped_positions[0]
    assert torch.max(diff_positions[:, :2]) < 1e-8
    assert torch.max(diff_positions[:, 2:]) > 1e-3
    assert (torch.mean(temperatures[len(temperatures) // 2 :]) - 300) / 300 < 0.30


def test_state_manipulation_with_constraints(ar_double_sim_state: ts.SimState):
    """Test that constraints are properly propagated during state manipulation."""
    # Set up constraints on the original state
    ar_double_sim_state.add_constraints(
        [FixAtoms(indices=torch.tensor([0, 1])), FixCom()]
    )

    # Extract individual systems from the double system state
    first_system = ar_double_sim_state[0]
    second_system = ar_double_sim_state[1]
    concatenated_state = ts.concatenate_states(
        [first_system, first_system, second_system]
    )

    # Verify constraint propagation to subsystems
    assert len(first_system.constraints) == 2
    assert len(second_system.constraints) == 2
    assert len(concatenated_state.constraints) == 2

    # Verify FixAtoms constraint indices are correctly mapped
    assert torch.all(first_system.constraints[0].indices == torch.tensor([0, 1]))
    assert torch.all(second_system.constraints[0].indices == torch.tensor([]))
    assert torch.all(
        concatenated_state.constraints[0].indices == torch.tensor([0, 1, 32, 33])
    )

    # Verify FixCom constraint system masks
    assert torch.all(
        concatenated_state.constraints[1].system_idx == torch.tensor([0, 1, 2])
    )

    # Test constraint propagation after splitting concatenated state
    split_systems = concatenated_state.split()
    assert len(split_systems[0].constraints) == 2
    assert torch.all(split_systems[0].constraints[0].indices == torch.tensor([0, 1]))
    assert torch.all(split_systems[1].constraints[0].indices == torch.tensor([0, 1]))
    assert torch.all(
        split_systems[2].constraints[0].indices == torch.tensor([], dtype=torch.long)
    )

    # Test constraint manipulation with different configurations
    ar_double_sim_state.constraints = []
    ar_double_sim_state.add_constraints([FixCom()])
    isolated_system = ar_double_sim_state[0]
    assert torch.all(
        isolated_system.constraints[0].system_idx == torch.tensor([0], dtype=torch.long)
    )

    # Test concatenation with mixed constraint states
    isolated_system.constraints = []
    mixed_concatenated_state = ts.concatenate_states(
        [isolated_system, ar_double_sim_state, isolated_system]
    )
    assert torch.all(
        mixed_concatenated_state.constraints[0].system_idx == torch.tensor([1, 2])
    )


def test_fix_com_gradient_descent_optimization(
    ar_supercell_sim_state: ts.SimState, lj_model: ModelInterface
) -> None:
    """Test FixCom constraint in Gradient Descent optimization."""
    # Add some random displacement to positions
    perturbed_positions = (
        ar_supercell_sim_state.positions
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    ar_supercell_sim_state.positions = perturbed_positions
    initial_state = ar_supercell_sim_state
    ar_supercell_sim_state.add_constraints(FixCom())

    initial_coms = get_centers_of_mass(
        positions=initial_state.positions,
        masses=initial_state.masses,
        system_idx=initial_state.system_idx,
        n_systems=initial_state.n_systems,
    )

    # Initialize Gradient Descent optimizer
    state = ts.gradient_descent_init(
        state=ar_supercell_sim_state, model=lj_model, lr=0.01
    )

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    while abs(energies[-2] - energies[-1]) > 1e-6:
        state = ts.gradient_descent_step(state=state, model=lj_model, pos_lr=0.01)
        energies.append(state.energy.item())

    final_coms = get_centers_of_mass(
        positions=state.positions,
        masses=state.masses,
        system_idx=state.system_idx,
        n_systems=initial_state.n_systems,
    )

    assert torch.allclose(final_coms, initial_coms, atol=1e-4)
    assert not torch.allclose(state.positions, initial_state.positions)


def test_fix_atoms_gradient_descent_optimization(
    ar_supercell_sim_state: ts.SimState, lj_model: ModelInterface
) -> None:
    """Test FixAtoms constraint in Gradient Descent optimization."""
    # Add some random displacement to positions
    perturbed_positions = (
        ar_supercell_sim_state.positions
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    ar_supercell_sim_state.positions = perturbed_positions
    initial_state = ar_supercell_sim_state
    initial_state.add_constraints(FixAtoms(indices=[0]))
    initial_position = initial_state.positions[0].clone()

    # Initialize Gradient Descent optimizer
    state = ts.gradient_descent_init(
        state=ar_supercell_sim_state, model=lj_model, lr=0.01
    )

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    while abs(energies[-2] - energies[-1]) > 1e-6:
        state = ts.gradient_descent_step(state=state, model=lj_model, pos_lr=0.01)
        energies.append(state.energy.item())

    final_position = state.positions[0]

    assert torch.allclose(final_position, initial_position, atol=1e-5)
    assert not torch.allclose(state.positions, initial_state.positions)


@pytest.mark.parametrize("fire_flavor", get_args(FireFlavor))
def test_test_atoms_fire_optimization(
    ar_supercell_sim_state: ts.SimState, lj_model: ModelInterface, fire_flavor: FireFlavor
) -> None:
    """Test FixAtoms constraint in FIRE optimization."""
    # Add some random displacement to positions
    # Create a fresh copy for each test run to avoid interference

    current_positions = (
        ar_supercell_sim_state.positions.clone()
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    current_sim_state = ts.SimState(
        positions=current_positions,
        masses=ar_supercell_sim_state.masses.clone(),
        cell=ar_supercell_sim_state.cell.clone(),
        pbc=ar_supercell_sim_state.pbc,
        atomic_numbers=ar_supercell_sim_state.atomic_numbers.clone(),
        system_idx=ar_supercell_sim_state.system_idx.clone(),
    )
    indices = torch.tensor([0, 2], dtype=torch.long)
    current_sim_state.add_constraints(FixAtoms(indices=indices))

    # Initialize FIRE optimizer
    state = ts.fire_init(
        current_sim_state, lj_model, fire_flavor=fire_flavor, dt_start=0.1
    )
    initial_position = state.positions[indices].clone()

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    max_steps = 1000  # Add max step to prevent infinite loop
    steps_taken = 0
    while abs(energies[-2] - energies[-1]) > 1e-6 and steps_taken < max_steps:
        state = ts.fire_step(state=state, model=lj_model, dt_max=0.3)
        energies.append(state.energy.item())
        steps_taken += 1

    final_position = state.positions[indices]

    assert torch.allclose(final_position, initial_position, atol=1e-5)


@pytest.mark.parametrize("fire_flavor", get_args(FireFlavor))
def test_fix_com_fire_optimization(
    ar_supercell_sim_state: ts.SimState, lj_model: ModelInterface, fire_flavor: FireFlavor
) -> None:
    """Test FixCom constraint in FIRE optimization."""
    # Add some random displacement to positions
    # Create a fresh copy for each test run to avoid interference

    current_positions = (
        ar_supercell_sim_state.positions.clone()
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    current_sim_state = ts.SimState(
        positions=current_positions,
        masses=ar_supercell_sim_state.masses.clone(),
        cell=ar_supercell_sim_state.cell.clone(),
        pbc=ar_supercell_sim_state.pbc,
        atomic_numbers=ar_supercell_sim_state.atomic_numbers.clone(),
        system_idx=ar_supercell_sim_state.system_idx.clone(),
    )
    current_sim_state.add_constraints(FixCom())

    # Initialize FIRE optimizer
    state = ts.fire_init(
        current_sim_state, lj_model, fire_flavor=fire_flavor, dt_start=0.1
    )
    initial_com = get_centers_of_mass(
        positions=state.positions,
        masses=state.masses,
        system_idx=state.system_idx,
        n_systems=state.n_systems,
    )

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    max_steps = 1000  # Add max step to prevent infinite loop
    steps_taken = 0
    while abs(energies[-2] - energies[-1]) > 1e-6 and steps_taken < max_steps:
        state = ts.fire_step(state=state, model=lj_model, dt_max=0.3)
        energies.append(state.energy.item())
        steps_taken += 1

    final_com = get_centers_of_mass(
        positions=state.positions,
        masses=state.masses,
        system_idx=state.system_idx,
        n_systems=state.n_systems,
    )

    assert torch.allclose(final_com, initial_com, atol=1e-4)
