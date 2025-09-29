import pytest
import torch

import torch_sim as ts
from torch_sim.integrators import (
    NPTLangevinState,
    calculate_momenta,
    npt_langevin,
    nve,
    nvt_langevin,
    nvt_nose_hoover,
    nvt_nose_hoover_invariant,
)
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.quantities import calc_kT
from torch_sim.state import concatenate_states
from torch_sim.units import MetalUnits


def test_calculate_momenta_basic(device: torch.device):
    """Test basic functionality of calculate_momenta."""
    seed = 42
    dtype = torch.float64

    # Create test inputs for 3 systems with 2 atoms each
    n_atoms = 8
    positions = torch.randn(n_atoms, 3, dtype=dtype, device=device)
    masses = torch.rand(n_atoms, dtype=dtype, device=device) + 0.5
    system_idx = torch.tensor(
        [0, 0, 1, 1, 2, 2, 3, 3], device=device
    )  # 3 systems with 2 atoms each
    kT = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=dtype, device=device)

    # Run the function
    momenta = calculate_momenta(positions, masses, system_idx, kT, seed=seed)

    # Basic checks
    assert momenta.shape == positions.shape
    assert momenta.dtype == dtype
    assert momenta.device == device

    # Check that each system has zero center of mass momentum
    for b in range(4):
        system_mask = system_idx == b
        system_momenta = momenta[system_mask]
        com_momentum = torch.mean(system_momenta, dim=0)
        assert torch.allclose(
            com_momentum, torch.zeros(3, dtype=dtype, device=device), atol=1e-10
        )


def test_calculate_momenta_single_atoms(device: torch.device):
    """Test that calculate_momenta preserves momentum for systems with single atoms."""
    seed = 42
    dtype = torch.float64

    # Create test inputs with some systems having single atoms
    positions = torch.randn(5, 3, dtype=dtype, device=device)
    masses = torch.rand(5, dtype=dtype, device=device) + 0.5
    system_idx = torch.tensor(
        [0, 1, 1, 2, 3], device=device
    )  # systems 0, 2, and 3 have single atoms
    kT = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=dtype, device=device)

    # Generate momenta and save the raw values before COM correction
    generator = torch.Generator(device=device).manual_seed(seed)
    raw_momenta = torch.randn(
        positions.shape, device=device, dtype=dtype, generator=generator
    ) * torch.sqrt(masses * kT[system_idx]).unsqueeze(-1)

    # Run the function
    momenta = calculate_momenta(positions, masses, system_idx, kT, seed=seed)

    # Check that single-atom systems have unchanged momenta
    for b in [0, 2, 3]:  # Single atom systems
        system_mask = system_idx == b
        # The momentum should be exactly the same as the raw value for single atoms
        assert torch.allclose(momenta[system_mask], raw_momenta[system_mask])

    # Check that multi-atom systems have zero COM
    for b in [1]:  # Multi-atom systems
        system_mask = system_idx == b
        system_momenta = momenta[system_mask]
        com_momentum = torch.mean(system_momenta, dim=0)
        assert torch.allclose(
            com_momentum, torch.zeros(3, dtype=dtype, device=device), atol=1e-10
        )


def test_npt_langevin(ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel):
    dtype = torch.float64
    n_steps = 200
    dt = torch.tensor(0.001, dtype=dtype)
    kT = torch.tensor(100.0, dtype=dtype) * MetalUnits.temperature
    external_pressure = torch.tensor(0.0, dtype=dtype) * MetalUnits.pressure

    # Initialize integrator
    init_fn, update_fn = npt_langevin(
        model=lj_model,
        dt=dt,
        kT=kT,
        external_pressure=external_pressure,
        alpha=40 * dt,
    )

    # Run dynamics for several steps
    state = init_fn(state=ar_double_sim_state, seed=42)
    energies = []
    temperatures = []
    for _step in range(n_steps):
        state = update_fn(state=state)

        # Calculate instantaneous temperature from kinetic energy
        temp = calc_kT(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        )
        energies.append(state.energy)
        temperatures.append(temp / MetalUnits.temperature)

    # Convert temperatures list to tensor
    temperatures_tensor = torch.stack(temperatures)
    temperatures_list = [t.tolist() for t in temperatures_tensor.T]

    energies_tensor = torch.stack(energies)
    energies_list = [t.tolist() for t in energies_tensor.T]

    # Basic sanity checks
    assert len(energies_list[0]) == n_steps
    assert len(temperatures_list[0]) == n_steps

    # Check temperature is roughly maintained for each trajectory
    mean_temps = torch.mean(temperatures_tensor, dim=0)  # Mean temp for each trajectory
    for mean_temp in mean_temps:
        assert (
            abs(mean_temp - kT.item() / MetalUnits.temperature) < 150.0
        )  # Allow for thermal fluctuations

    # Check energy is stable for each trajectory
    for traj in energies_list:
        energy_std = torch.tensor(traj).std()
        assert energy_std < 1.0  # Adjust threshold as needed

    # Check positions and momenta have correct shapes
    n_atoms = 8

    # Verify the two systems remain distinct
    pos_diff = torch.norm(
        state.positions[:n_atoms].mean(0) - state.positions[n_atoms:].mean(0)
    )
    assert pos_diff > 0.0001  # Systems should remain separated


def test_npt_langevin_multi_kt(
    ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel
):
    dtype = torch.float64
    n_steps = 200
    dt = torch.tensor(0.001, dtype=dtype)
    kT = torch.tensor([300, 10_000], dtype=dtype) * MetalUnits.temperature
    external_pressure = torch.tensor(0, dtype=dtype) * MetalUnits.pressure

    # Initialize integrator
    init_fn, update_fn = npt_langevin(
        model=lj_model,
        dt=dt,
        kT=kT,
        external_pressure=external_pressure,
        alpha=40 * dt,
    )

    # Run dynamics for several steps
    state = init_fn(state=ar_double_sim_state, seed=42)
    energies = []
    temperatures = []
    for _step in range(n_steps):
        state = update_fn(state=state)

        # Calculate instantaneous temperature from kinetic energy
        temp = calc_kT(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        )
        energies.append(state.energy)
        temperatures.append(temp / MetalUnits.temperature)

    # Convert temperatures list to tensor
    temperatures_tensor = torch.stack(temperatures)
    temperatures_list = [t.tolist() for t in temperatures_tensor.T]

    energies_tensor = torch.stack(energies)
    energies_list = [t.tolist() for t in energies_tensor.T]

    # Basic sanity checks
    assert len(energies_list[0]) == n_steps
    assert len(temperatures_list[0]) == n_steps

    # Check temperature is roughly maintained for each trajectory
    mean_temps = torch.mean(temperatures_tensor, dim=0)  # Mean temp for each trajectory
    assert torch.allclose(mean_temps, kT / MetalUnits.temperature, rtol=0.5)


def test_nvt_langevin(ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel):
    dtype = torch.float64
    n_steps = 100
    dt = torch.tensor(0.001, dtype=dtype)
    kT = torch.tensor(300, dtype=dtype) * MetalUnits.temperature

    # Initialize integrator
    init_fn, update_fn = nvt_langevin(
        model=lj_model,
        dt=dt,
        kT=kT,
    )

    # Run dynamics for several steps
    state = init_fn(state=ar_double_sim_state, seed=42)
    energies = []
    temperatures = []
    for _step in range(n_steps):
        state = update_fn(state=state)

        # Calculate instantaneous temperature from kinetic energy
        temp = calc_kT(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        )
        energies.append(state.energy)
        temperatures.append(temp / MetalUnits.temperature)

    # Convert temperatures list to tensor
    temperatures_tensor = torch.stack(temperatures)
    temperatures_list = [t.tolist() for t in temperatures_tensor.T]

    energies_tensor = torch.stack(energies)
    energies_list = [t.tolist() for t in energies_tensor.T]

    # Basic sanity checks
    assert len(energies_list[0]) == n_steps
    assert len(temperatures_list[0]) == n_steps

    # Check temperature is roughly maintained for each trajectory
    mean_temps = torch.mean(temperatures_tensor, dim=0)  # Mean temp for each trajectory
    for mean_temp in mean_temps:
        assert (
            abs(mean_temp - kT.item() / MetalUnits.temperature) < 100.0
        )  # Allow for thermal fluctuations

    # Check energy is stable for each trajectory
    for traj in energies_list:
        energy_std = torch.tensor(traj).std()
        assert energy_std < 1.0  # Adjust threshold as needed

    # Check positions and momenta have correct shapes
    n_atoms = 8

    # Verify the two systems remain distinct
    pos_diff = torch.norm(
        state.positions[:n_atoms].mean(0) - state.positions[n_atoms:].mean(0)
    )
    assert pos_diff > 0.0001  # Systems should remain separated


def test_nvt_langevin_multi_kt(
    ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel
):
    dtype = torch.float64
    n_steps = 200
    dt = torch.tensor(0.001, dtype=dtype)
    kT = torch.tensor([300, 10_000], dtype=dtype) * MetalUnits.temperature

    # Initialize integrator
    init_fn, update_fn = nvt_langevin(
        model=lj_model,
        dt=dt,
        kT=kT,
    )

    # Run dynamics for several steps
    state = init_fn(state=ar_double_sim_state, seed=42)
    energies = []
    temperatures = []
    for _step in range(n_steps):
        state = update_fn(state=state)

        # Calculate instantaneous temperature from kinetic energy
        temp = calc_kT(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        )
        energies.append(state.energy)
        temperatures.append(temp / MetalUnits.temperature)

    # Convert temperatures list to tensor
    temperatures_tensor = torch.stack(temperatures)
    temperatures_list = [t.tolist() for t in temperatures_tensor.T]

    energies_tensor = torch.stack(energies)
    energies_list = [t.tolist() for t in energies_tensor.T]

    # Basic sanity checks
    assert len(energies_list[0]) == n_steps
    assert len(temperatures_list[0]) == n_steps

    # Check temperature is roughly maintained for each trajectory
    mean_temps = torch.mean(temperatures_tensor, dim=0)  # Mean temp for each trajectory
    assert torch.allclose(mean_temps, kT / MetalUnits.temperature, rtol=0.5)


def test_nvt_nose_hoover(ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel):
    dtype = torch.float64
    n_steps = 100
    dt = torch.tensor(0.001, dtype=dtype)
    kT = torch.tensor(300, dtype=dtype) * MetalUnits.temperature

    # Initialize integrator
    init_fn, update_fn = nvt_nose_hoover(
        model=lj_model,
        dt=dt,
        kT=kT,
    )

    # Run dynamics for several steps
    state = init_fn(state=ar_double_sim_state, seed=42)
    energies = []
    temperatures = []
    invariants = []
    for _step in range(n_steps):
        state = update_fn(state=state)

        # Calculate instantaneous temperature from kinetic energy
        temp = calc_kT(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        )
        energies.append(state.energy)
        temperatures.append(temp / MetalUnits.temperature)
        invariants.append(nvt_nose_hoover_invariant(state, kT))

    # Convert temperatures list to tensor
    temperatures_tensor = torch.stack(temperatures)
    temperatures_list = [t.tolist() for t in temperatures_tensor.T]

    energies_tensor = torch.stack(energies)
    energies_list = [t.tolist() for t in energies_tensor.T]

    invariants_tensor = torch.stack(invariants)

    # Basic sanity checks
    assert len(energies_list[0]) == n_steps
    assert len(temperatures_list[0]) == n_steps

    # Check temperature is roughly maintained for each trajectory
    mean_temps = torch.mean(temperatures_tensor, dim=0)  # Mean temp for each trajectory
    for mean_temp in mean_temps:
        assert (
            abs(mean_temp - kT.item() / MetalUnits.temperature) < 100.0
        )  # Allow for thermal fluctuations

    # Check energy is stable for each trajectory
    for traj in energies_list:
        energy_std = torch.tensor(traj).std()
        assert energy_std < 1.0  # Adjust threshold as needed

    # Check invariant conservation (should be roughly constant)
    for traj_idx in range(invariants_tensor.shape[1]):
        invariant_traj = invariants_tensor[:, traj_idx]
        invariant_std = invariant_traj.std()
        # Allow for some drift but should be relatively stable
        # Less than 10% relative variation
        assert invariant_std / invariant_traj.mean() < 0.1

    # Check positions and momenta have correct shapes
    n_atoms = 8

    # Verify the two systems remain distinct
    pos_diff = torch.norm(
        state.positions[:n_atoms].mean(0) - state.positions[n_atoms:].mean(0)
    )
    assert pos_diff > 0.0001  # Systems should remain separated


def test_nvt_nose_hoover_multi_equivalent_to_single(
    mixed_double_sim_state: ts.SimState, lj_model: LennardJonesModel
):
    """Test that nvt_nose_hoover with multiple identical kT values behaves like
    running different single kT, assuming same initial state
    (most importantly same momenta)."""
    dtype = torch.float64
    n_steps = 100
    dt = torch.tensor(0.001, dtype=dtype)
    kT = torch.tensor(300, dtype=dtype) * MetalUnits.temperature

    # Initialize integrator
    init_fn, update_fn = nvt_nose_hoover(
        model=lj_model,
        dt=dt,
        kT=kT,
    )
    final_temperatures = []
    initial_momenta = []
    # Run dynamics for several steps
    for i in range(mixed_double_sim_state.n_systems):
        state = init_fn(state=mixed_double_sim_state[i], seed=42)
        initial_momenta.append(state.momenta.clone())
        for _step in range(n_steps):
            state = update_fn(state=state)

            # Calculate instantaneous temperature from kinetic energy
        temp = calc_kT(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        )
        final_temperatures.append(temp / MetalUnits.temperature)

    initial_momenta_tensor = torch.concat(initial_momenta)
    final_temperatures = torch.concat(final_temperatures)
    state = init_fn(state=mixed_double_sim_state, seed=42, momenta=initial_momenta_tensor)
    for _step in range(n_steps):
        state = update_fn(state=state)

        # Calculate instantaneous temperature from kinetic energy
    temp = calc_kT(
        masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
    )

    assert torch.allclose(final_temperatures, temp / MetalUnits.temperature)


def test_nvt_nose_hoover_multi_kt(
    ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel
):
    dtype = torch.float64
    n_steps = 200
    dt = torch.tensor(0.001, dtype=dtype)
    kT = torch.tensor([300, 10_000], dtype=dtype) * MetalUnits.temperature

    # Initialize integrator
    init_fn, update_fn = nvt_nose_hoover(
        model=lj_model,
        dt=dt,
        kT=kT,
    )

    # Run dynamics for several steps
    state = init_fn(state=ar_double_sim_state, seed=42)
    energies = []
    temperatures = []
    invariants = []
    for _step in range(n_steps):
        state = update_fn(state=state)

        # Calculate instantaneous temperature from kinetic energy
        temp = calc_kT(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        )
        energies.append(state.energy)
        temperatures.append(temp / MetalUnits.temperature)
        invariants.append(nvt_nose_hoover_invariant(state, kT))

    # Convert temperatures list to tensor
    temperatures_tensor = torch.stack(temperatures)
    temperatures_list = [t.tolist() for t in temperatures_tensor.T]

    energies_tensor = torch.stack(energies)
    energies_list = [t.tolist() for t in energies_tensor.T]

    invariants_tensor = torch.stack(invariants)

    # Basic sanity checks
    assert len(energies_list[0]) == n_steps
    assert len(temperatures_list[0]) == n_steps

    # Check temperature is roughly maintained for each trajectory
    mean_temps = torch.mean(temperatures_tensor, dim=0)  # Mean temp for each trajectory
    assert torch.allclose(mean_temps, kT / MetalUnits.temperature, rtol=0.5)

    # Check invariant conservation for each system
    for traj_idx in range(invariants_tensor.shape[1]):
        invariant_traj = invariants_tensor[:, traj_idx]
        invariant_std = invariant_traj.std()
        # Allow for some drift but should be relatively stable
        # Less than 10% relative variation
        assert invariant_std / invariant_traj.mean() < 0.1


def test_nve(ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel):
    dtype = torch.float64
    n_steps = 100
    dt = torch.tensor(0.001, dtype=dtype)
    kT = torch.tensor(100.0, dtype=dtype) * MetalUnits.temperature

    # Initialize integrator
    nve_init, nve_update = nve(model=lj_model, dt=dt, kT=kT)
    state = nve_init(state=ar_double_sim_state, seed=42)

    # Run dynamics for several steps
    energies = []
    for _step in range(n_steps):
        state = nve_update(state=state, dt=dt)

        energies.append(state.energy)

    energies_tensor = torch.stack(energies)

    # assert conservation of energy
    assert torch.allclose(energies_tensor[:, 0], energies_tensor[0, 0], atol=1e-4)
    assert torch.allclose(energies_tensor[:, 1], energies_tensor[0, 1], atol=1e-4)


@pytest.mark.parametrize(
    "sim_state_fixture_name", ["casio3_sim_state", "ar_supercell_sim_state"]
)
def test_compare_single_vs_batched_integrators(
    sim_state_fixture_name: str,
    request: pytest.FixtureRequest,
    lj_model: LennardJonesModel,
) -> None:
    """Test NVE single vs batched for a tilted cell to verify PBC wrapping.

    NOTE: added triclinic cell after https://github.com/Radical-AI/torch-sim/issues/171.
    Although the addition doesn't fail if we do not add the changes suggested in issue.
    """
    sim_state = request.getfixturevalue(sim_state_fixture_name)
    n_steps = 100

    initial_states = {
        "single": sim_state,
        "batched": concatenate_states([sim_state, sim_state]),
    }

    final_states = {}
    for state_name, state in initial_states.items():
        # Initialize integrator
        kT = torch.tensor(100.0) * MetalUnits.temperature
        dt = torch.tensor(0.001)  # Small timestep for stability

        nve_init, nve_update = nve(model=lj_model, dt=dt, kT=kT)
        # Initialize momenta (even if zero) and get forces
        state = nve_init(state=state, seed=42)  # kT is ignored if momenta are set below
        # Ensure momenta start at zero AFTER init which might randomize them based on kT
        state.momenta = torch.zeros_like(state.momenta)  # Start from rest

        for _step in range(n_steps):
            state = nve_update(state=state, dt=dt)

        final_states[state_name] = state

    # Check energy conservation
    single_state = final_states["single"]
    batched_state_0 = final_states["batched"][0]
    batched_state_1 = final_states["batched"][1]

    # Compare single state results with each part of the batched state
    for final_state in [batched_state_0, batched_state_1]:
        # Check positions first - most likely to fail with incorrect PBC
        torch.testing.assert_close(single_state.positions, final_state.positions)
        # Check other state components
        torch.testing.assert_close(single_state.momenta, final_state.momenta)
        torch.testing.assert_close(single_state.forces, final_state.forces)
        torch.testing.assert_close(single_state.masses, final_state.masses)
        torch.testing.assert_close(single_state.cell, final_state.cell)
        torch.testing.assert_close(single_state.energy, final_state.energy)


def test_compute_cell_force_atoms_per_system():
    """Test that compute_cell_force correctly scales by number of atoms per system.

    Covers fix in https://github.com/Radical-AI/torch-sim/pull/153."""
    from torch_sim.integrators.npt import _compute_cell_force

    # Setup minimal state with two systems having 8:1 atom ratio
    s1, s2 = torch.zeros(8, dtype=torch.long), torch.ones(64, dtype=torch.long)

    state = NPTLangevinState(
        positions=torch.zeros((72, 3)),
        velocities=torch.zeros((72, 3)),
        energy=torch.zeros(2),
        forces=torch.zeros((72, 3)),
        masses=torch.ones(72),
        cell=torch.eye(3).repeat(2, 1, 1),
        pbc=True,
        system_idx=torch.cat([s1, s2]),
        atomic_numbers=torch.ones(72, dtype=torch.long),
        stress=torch.zeros((2, 3, 3)),
        reference_cell=torch.eye(3).repeat(2, 1, 1),
        cell_positions=torch.ones((2, 3, 3)),
        cell_velocities=torch.zeros((2, 3, 3)),
        cell_masses=torch.ones(2),
    )

    # Get forces and compare ratio
    cell_force = _compute_cell_force(state, torch.tensor(0.0), torch.tensor([1.0, 1.0]))
    force_ratio = (
        torch.diagonal(cell_force[1]).mean() / torch.diagonal(cell_force[0]).mean()
    )

    # Force ratio should match atom ratio (8:1) with the fix
    assert abs(force_ratio - 8.0) / 8.0 < 0.1
