from typing import Any

import torch

from torch_sim.integrators import calculate_momenta, npt_langevin, nve, nvt_langevin
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.quantities import calc_kT
from torch_sim.state import SimState, concatenate_states
from torch_sim.units import MetalUnits


def test_calculate_momenta_basic(device: torch.device):
    """Test basic functionality of calculate_momenta."""
    seed = 42
    dtype = torch.float64

    # Create test inputs for 3 batches with 2 atoms each
    n_atoms = 8
    positions = torch.randn(n_atoms, 3, dtype=dtype, device=device)
    masses = torch.rand(n_atoms, dtype=dtype, device=device) + 0.5
    batch = torch.tensor(
        [0, 0, 1, 1, 2, 2, 3, 3], device=device
    )  # 3 batches with 2 atoms each
    kT = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=dtype, device=device)

    # Run the function
    momenta = calculate_momenta(positions, masses, batch, kT, seed=seed)

    # Basic checks
    assert momenta.shape == positions.shape
    assert momenta.dtype == dtype
    assert momenta.device == device

    # Check that each batch has zero center of mass momentum
    for b in range(4):
        batch_mask = batch == b
        batch_momenta = momenta[batch_mask]
        com_momentum = torch.mean(batch_momenta, dim=0)
        assert torch.allclose(
            com_momentum, torch.zeros(3, dtype=dtype, device=device), atol=1e-10
        )


def test_calculate_momenta_single_atoms(device: torch.device):
    """Test that calculate_momenta preserves momentum for batches with single atoms."""
    seed = 42
    dtype = torch.float64

    # Create test inputs with some batches having single atoms
    positions = torch.randn(5, 3, dtype=dtype, device=device)
    masses = torch.rand(5, dtype=dtype, device=device) + 0.5
    batch = torch.tensor(
        [0, 1, 1, 2, 3], device=device
    )  # Batches 0, 2, and 3 have single atoms
    kT = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=dtype, device=device)

    # Generate momenta and save the raw values before COM correction
    generator = torch.Generator(device=device).manual_seed(seed)
    raw_momenta = torch.randn(
        positions.shape, device=device, dtype=dtype, generator=generator
    ) * torch.sqrt(masses * kT[batch]).unsqueeze(-1)

    # Run the function
    momenta = calculate_momenta(positions, masses, batch, kT, seed=seed)

    # Check that single-atom batches have unchanged momenta
    for b in [0, 2, 3]:  # Single atom batches
        batch_mask = batch == b
        # The momentum should be exactly the same as the raw value for single atoms
        assert torch.allclose(momenta[batch_mask], raw_momenta[batch_mask])

    # Check that multi-atom batches have zero COM
    for b in [1]:  # Multi-atom batches
        batch_mask = batch == b
        batch_momenta = momenta[batch_mask]
        com_momentum = torch.mean(batch_momenta, dim=0)
        assert torch.allclose(
            com_momentum, torch.zeros(3, dtype=dtype, device=device), atol=1e-10
        )


def test_npt_langevin(ar_double_sim_state: SimState, lj_model: LennardJonesModel):
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
        temp = calc_kT(state.momenta, state.masses, batch=state.batch)
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
    ar_double_sim_state: SimState, lj_model: LennardJonesModel
):
    dtype = torch.float64
    n_steps = 200
    dt = torch.tensor(0.001, dtype=dtype)
    kT = torch.tensor([300.0, 10000.0], dtype=dtype) * MetalUnits.temperature
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
        temp = calc_kT(state.momenta, state.masses, batch=state.batch)
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


def test_nvt_langevin(ar_double_sim_state: SimState, lj_model: LennardJonesModel):
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
        temp = calc_kT(state.momenta, state.masses, batch=state.batch)
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
    ar_double_sim_state: SimState, lj_model: LennardJonesModel
):
    dtype = torch.float64
    n_steps = 200
    dt = torch.tensor(0.001, dtype=dtype)
    kT = torch.tensor([300.0, 10000.0], dtype=dtype) * MetalUnits.temperature

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
        temp = calc_kT(state.momenta, state.masses, batch=state.batch)
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


def test_nve(ar_double_sim_state: SimState, lj_model: LennardJonesModel):
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


def test_compare_single_vs_batched_integrators(
    ar_supercell_sim_state: SimState, lj_model: Any
) -> None:
    """Test that single and batched integrators give the same results."""
    initial_states = {
        "single": ar_supercell_sim_state,
        "batched": concatenate_states([ar_supercell_sim_state, ar_supercell_sim_state]),
    }

    final_states = {}
    for state_name, state in initial_states.items():
        # Initialize integrator
        kT = torch.tensor(100.0) * MetalUnits.temperature
        dt = torch.tensor(0.001)  # Small timestep for stability

        nve_init, nve_update = nve(model=lj_model, dt=dt, kT=kT)
        state = nve_init(state=state, seed=42)
        state.momenta = torch.zeros_like(state.momenta)

        for _step in range(100):
            state = nve_update(state=state, dt=dt)

        final_states[state_name] = state

    # Check energy conservation
    ar_single_state = final_states["single"]
    ar_batched_state_0 = final_states["batched"][0]
    ar_batched_state_1 = final_states["batched"][1]

    for final_state in [ar_batched_state_0, ar_batched_state_1]:
        assert torch.allclose(ar_single_state.positions, final_state.positions)
        assert torch.allclose(ar_single_state.momenta, final_state.momenta)
        assert torch.allclose(ar_single_state.forces, final_state.forces)
        assert torch.allclose(ar_single_state.masses, final_state.masses)
        assert torch.allclose(ar_single_state.cell, final_state.cell)
