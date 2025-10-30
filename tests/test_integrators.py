import pytest
import torch

import torch_sim as ts
from tests.conftest import DEVICE, DTYPE
from torch_sim.integrators import calculate_momenta
from torch_sim.integrators.npt import _compute_cell_force
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.units import MetalUnits


def test_calculate_momenta_basic():
    """Test basic functionality of calculate_momenta."""
    seed = 42

    # Create test inputs for 3 systems with 2 atoms each
    n_atoms = 8
    positions = torch.randn(n_atoms, 3, dtype=DTYPE, device=DEVICE)
    masses = torch.rand(n_atoms, dtype=DTYPE, device=DEVICE) + 0.5
    system_idx = torch.tensor(
        [0, 0, 1, 1, 2, 2, 3, 3], device=DEVICE
    )  # 3 systems with 2 atoms each
    kT = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=DTYPE, device=DEVICE)

    # Run the function
    momenta = calculate_momenta(positions, masses, system_idx, kT, seed=seed)

    # Basic checks
    assert momenta.shape == positions.shape
    assert momenta.dtype == DTYPE
    assert momenta.device == DEVICE

    # Check that each system has zero center of mass momentum
    for sys_idx in range(4):
        system_mask = system_idx == sys_idx
        system_momenta = momenta[system_mask]
        com_momentum = torch.mean(system_momenta, dim=0)
        assert torch.allclose(
            com_momentum, torch.zeros(3, dtype=DTYPE, device=DEVICE), atol=1e-10
        )


def test_calculate_momenta_single_atoms():
    """Test that calculate_momenta preserves momentum for systems with single atoms."""
    seed = 42

    # Create test inputs with some systems having single atoms
    positions = torch.randn(5, 3, dtype=DTYPE, device=DEVICE)
    masses = torch.rand(5, dtype=DTYPE, device=DEVICE) + 0.5
    system_idx = torch.tensor(
        [0, 1, 1, 2, 3], device=DEVICE
    )  # systems 0, 2, and 3 have single atoms
    kT = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=DTYPE, device=DEVICE)

    # Generate momenta and save the raw values before COM correction
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    raw_momenta = torch.randn(
        positions.shape, device=DEVICE, dtype=DTYPE, generator=generator
    ) * torch.sqrt(masses * kT[system_idx]).unsqueeze(-1)

    # Run the function
    momenta = calculate_momenta(positions, masses, system_idx, kT, seed=seed)

    # Check that single-atom systems have unchanged momenta
    for sys_idx in (0, 2, 3):  # Single atom systems
        system_mask = system_idx == sys_idx
        # The momentum should be exactly the same as the raw value for single atoms
        assert torch.allclose(momenta[system_mask], raw_momenta[system_mask])

    # Check that multi-atom systems have zero COM
    for sys_idx in (1,):  # Multi-atom systems
        system_mask = system_idx == sys_idx
        system_momenta = momenta[system_mask]
        com_momentum = torch.mean(system_momenta, dim=0)
        assert torch.allclose(
            com_momentum, torch.zeros(3, dtype=DTYPE, device=DEVICE), atol=1e-10
        )


def test_npt_langevin(
    ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel
) -> None:
    n_steps = 200
    dt = torch.tensor(0.001, dtype=DTYPE)
    kT = torch.tensor(100.0, dtype=DTYPE) * MetalUnits.temperature
    external_pressure = torch.tensor(0.0, dtype=DTYPE) * MetalUnits.pressure
    alpha = 40 * dt
    cell_alpha = alpha
    b_tau = 1 / (1000 * dt)

    # Initialize integrator using new direct API
    state = ts.npt_langevin_init(
        state=ar_double_sim_state, model=lj_model, dt=dt, kT=kT, alpha=alpha, seed=42
    )

    # Run dynamics for several steps
    energies = []
    temperatures = []
    for _step in range(n_steps):
        state = ts.npt_langevin_step(
            state=state,
            model=lj_model,
            dt=dt,
            kT=kT,
            external_pressure=external_pressure,
            alpha=alpha,
            cell_alpha=cell_alpha,
            b_tau=b_tau,
        )

        # Calculate instantaneous temperature from kinetic energy
        temp = ts.calc_kT(
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
    n_steps = 200
    dt = torch.tensor(0.001, dtype=DTYPE)
    kT = torch.tensor([300, 10_000], dtype=DTYPE) * MetalUnits.temperature
    external_pressure = torch.tensor(0, dtype=DTYPE) * MetalUnits.pressure
    alpha = 40 * dt
    cell_alpha = alpha
    b_tau = 1 / (1000 * dt)

    # Initialize integrator using new direct API
    state = ts.npt_langevin_init(
        state=ar_double_sim_state, model=lj_model, dt=dt, kT=kT, alpha=alpha, seed=42
    )

    # Run dynamics for several steps
    energies = []
    temperatures = []
    for _step in range(n_steps):
        state = ts.npt_langevin_step(
            state=state,
            model=lj_model,
            dt=dt,
            kT=kT,
            external_pressure=external_pressure,
            alpha=alpha,
            cell_alpha=cell_alpha,
            b_tau=b_tau,
        )

        # Calculate instantaneous temperature from kinetic energy
        temp = ts.calc_kT(
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
    n_steps = 100
    dt = torch.tensor(0.001, dtype=DTYPE)
    kT = torch.tensor(300, dtype=DTYPE) * MetalUnits.temperature

    # Initialize integrator
    state = ts.nvt_langevin_init(
        state=ar_double_sim_state, model=lj_model, kT=kT, seed=42
    )
    energies = []
    temperatures = []
    for _step in range(n_steps):
        state = ts.nvt_langevin_step(model=lj_model, state=state, dt=dt, kT=kT)

        # Calculate instantaneous temperature from kinetic energy
        temp = ts.calc_kT(
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
    n_steps = 200
    dt = torch.tensor(0.001, dtype=DTYPE)
    kT = torch.tensor([300, 10_000], dtype=DTYPE) * MetalUnits.temperature

    # Initialize integrator
    state = ts.nvt_langevin_init(
        state=ar_double_sim_state, model=lj_model, kT=kT, seed=42
    )
    energies = []
    temperatures = []
    for _step in range(n_steps):
        state = ts.nvt_langevin_step(model=lj_model, state=state, dt=dt, kT=kT)

        # Calculate instantaneous temperature from kinetic energy
        temp = ts.calc_kT(
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

    # Run dynamics for several steps
    state = ts.nvt_nose_hoover_init(
        state=ar_double_sim_state, model=lj_model, dt=dt, kT=kT, seed=42
    )
    energies = []
    temperatures = []
    invariants = []
    for _step in range(n_steps):
        state = ts.nvt_nose_hoover_step(
            state=state,
            model=lj_model,
            dt=dt,
            kT=kT,
        )

        # Calculate instantaneous temperature from kinetic energy
        temp = ts.calc_kT(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        )
        energies.append(state.energy)
        temperatures.append(temp / MetalUnits.temperature)
        invariants.append(ts.nvt_nose_hoover_invariant(state, kT))

    # Convert temperatures list to tensor
    temperatures_tensor = torch.stack(temperatures)
    temperatures_list = [t.tolist() for t in temperatures_tensor.T]
    assert torch.allclose(
        temperatures_tensor[-1],
        torch.tensor([299.9910, 299.6800], dtype=dtype),
    )

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

    final_temperatures = []
    initial_momenta = []
    # Run dynamics for several steps
    for i in range(mixed_double_sim_state.n_systems):
        state = ts.nvt_nose_hoover_init(
            state=mixed_double_sim_state[i], model=lj_model, dt=dt, kT=kT, seed=42
        )
        initial_momenta.append(state.momenta.clone())
        for _step in range(n_steps):
            state = ts.nvt_nose_hoover_step(
                state=state,
                model=lj_model,
                dt=dt,
                kT=kT,
            )

            # Calculate instantaneous temperature from kinetic energy
        temp = ts.calc_kT(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        )
        final_temperatures.append(temp / MetalUnits.temperature)

    initial_momenta_tensor = torch.concat(initial_momenta)
    final_temperatures = torch.concat(final_temperatures)
    state = ts.nvt_nose_hoover_init(
        state=mixed_double_sim_state,
        model=lj_model,
        dt=dt,
        kT=kT,
        seed=42,
        momenta=initial_momenta_tensor,
    )
    for _step in range(n_steps):
        state = ts.nvt_nose_hoover_step(state=state, model=lj_model, dt=dt, kT=kT)

        # Calculate instantaneous temperature from kinetic energy
    temp = ts.calc_kT(
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

    # Run dynamics for several steps
    state = ts.nvt_nose_hoover_init(
        state=ar_double_sim_state, model=lj_model, dt=dt, kT=kT, seed=42
    )
    energies = []
    temperatures = []
    invariants = []
    for _step in range(n_steps):
        state = ts.nvt_nose_hoover_step(state=state, model=lj_model, dt=dt, kT=kT)

        # Calculate instantaneous temperature from kinetic energy
        temp = ts.calc_kT(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        )
        energies.append(state.energy)
        temperatures.append(temp / MetalUnits.temperature)
        invariants.append(ts.nvt_nose_hoover_invariant(state, kT))

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


def test_nvt_csvr(ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel):
    n_steps = 100
    dt = torch.tensor(0.001, dtype=DTYPE)
    kT = torch.tensor(300, dtype=DTYPE) * MetalUnits.temperature

    # Initialize integrator
    state = ts.nvt_csvr_init(state=ar_double_sim_state, model=lj_model, kT=kT, seed=42)
    energies = []
    temperatures = []
    for _step in range(n_steps):
        state = ts.nvt_csvr_step(model=lj_model, state=state, dt=dt, kT=kT)

        # Calculate instantaneous temperature from kinetic energy
        temp = ts.calc_kT(
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


def test_npt_nose_hoover(ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel):
    dtype = torch.float64
    n_steps = 100
    dt = torch.tensor(0.001, dtype=dtype)
    kT = torch.tensor(300, dtype=dtype) * MetalUnits.temperature
    external_pressure = torch.tensor(0.0, dtype=dtype) * MetalUnits.pressure

    # Run dynamics for several steps
    state = ts.npt_nose_hoover_init(
        state=ar_double_sim_state,
        model=lj_model,
        dt=dt,
        kT=kT,
        external_pressure=external_pressure,
        seed=42,
    )
    energies = []
    temperatures = []
    invariants = []
    for _step in range(n_steps):
        state = ts.npt_nose_hoover_step(
            state=state,
            model=lj_model,
            dt=dt,
            kT=kT,
            external_pressure=external_pressure,
        )

        # Calculate instantaneous temperature from kinetic energy
        temp = ts.calc_kT(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        )
        energies.append(state.energy)
        temperatures.append(temp / MetalUnits.temperature)
        invariants.append(ts.npt_nose_hoover_invariant(state, kT, external_pressure))

    # Convert temperatures list to tensor
    temperatures_tensor = torch.stack(temperatures)
    temperatures_list = [t.tolist() for t in temperatures_tensor.T]
    assert torch.allclose(
        temperatures_tensor[-1],
        torch.tensor([297.8602, 297.5306], dtype=dtype),
    )

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

    # Check energy is stable for each trajectory (NPT allows energy fluctuations)
    for traj in energies_list:
        energy_std = torch.tensor(traj).std()
        assert energy_std < 2.0  # Allow more fluctuation than NVT due to volume changes

    # Check invariant conservation (should be roughly constant)
    for traj_idx in range(invariants_tensor.shape[1]):
        invariant_traj = invariants_tensor[:, traj_idx]
        invariant_std = invariant_traj.std()
        # Allow for some drift but should be relatively stable
        # Less than 15% relative variation (more lenient than NVT)
        assert invariant_std / invariant_traj.mean() < 0.15

    # Check positions and momenta have correct shapes
    n_atoms = 8

    # Verify the two systems remain distinct
    pos_diff = torch.norm(
        state.positions[:n_atoms].mean(0) - state.positions[n_atoms:].mean(0)
    )
    assert pos_diff > 0.0001  # Systems should remain separated


def test_npt_nose_hoover_multi_equivalent_to_single(
    mixed_double_sim_state: ts.SimState, lj_model: LennardJonesModel
):
    """Test that nvt_nose_hoover with multiple identical kT values behaves like
    running different single kT, assuming same initial state
    (most importantly same momenta)."""
    dtype = torch.float64
    n_steps = 100
    dt = torch.tensor(0.001, dtype=dtype)
    kT = torch.tensor(300, dtype=dtype) * MetalUnits.temperature
    external_pressure = torch.tensor(0.0, dtype=dtype) * MetalUnits.pressure

    final_temperatures = []
    initial_momenta = []
    # Run dynamics for several steps
    for i in range(mixed_double_sim_state.n_systems):
        state = ts.npt_nose_hoover_init(
            state=mixed_double_sim_state[i],
            model=lj_model,
            dt=dt,
            kT=kT,
            external_pressure=external_pressure,
            seed=42,
        )
        initial_momenta.append(state.momenta.clone())
        for _step in range(n_steps):
            state = ts.npt_nose_hoover_step(
                state=state,
                model=lj_model,
                dt=dt,
                kT=kT,
                external_pressure=external_pressure,
            )

            # Calculate instantaneous temperature from kinetic energy
        temp = ts.calc_kT(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        )
        final_temperatures.append(temp / MetalUnits.temperature)

    initial_momenta_tensor = torch.concat(initial_momenta)
    final_temperatures = torch.concat(final_temperatures)
    state = ts.npt_nose_hoover_init(
        state=mixed_double_sim_state,
        model=lj_model,
        dt=dt,
        kT=kT,
        external_pressure=external_pressure,
        seed=42,
        momenta=initial_momenta_tensor,
    )
    for _step in range(n_steps):
        state = ts.npt_nose_hoover_step(
            state=state, model=lj_model, dt=dt, kT=kT, external_pressure=external_pressure
        )

        # Calculate instantaneous temperature from kinetic energy
    temp = ts.calc_kT(
        masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
    )

    assert torch.allclose(final_temperatures, temp / MetalUnits.temperature)


def test_npt_nose_hoover_multi_kt(
    ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel
):
    dtype = torch.float64
    n_steps = 200
    dt = torch.tensor(0.001, dtype=dtype)
    kT = torch.tensor([300, 10_000], dtype=dtype) * MetalUnits.temperature
    external_pressure = torch.tensor(0.0, dtype=dtype) * MetalUnits.pressure

    # Run dynamics for several steps
    state = ts.npt_nose_hoover_init(
        state=ar_double_sim_state,
        model=lj_model,
        dt=dt,
        kT=kT,
        external_pressure=external_pressure,
        seed=42,
    )
    energies = []
    temperatures = []
    invariants = []
    for _step in range(n_steps):
        state = ts.npt_nose_hoover_step(
            state=state,
            model=lj_model,
            dt=dt,
            kT=kT,
            external_pressure=external_pressure,
        )

        # Calculate instantaneous temperature from kinetic energy
        temp = ts.calc_kT(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        )
        energies.append(state.energy)
        temperatures.append(temp / MetalUnits.temperature)
        invariants.append(ts.npt_nose_hoover_invariant(state, kT, external_pressure))

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
        # Less than 15% relative variation (more lenient than NVT)
        assert invariant_std / invariant_traj.mean() < 0.15


def test_nve(ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel):
    n_steps = 100
    dt = torch.tensor(0.001, dtype=DTYPE)
    kT = torch.tensor(100.0, dtype=DTYPE) * MetalUnits.temperature

    # Initialize integrator
    state = ts.nve_init(state=ar_double_sim_state, model=lj_model, kT=kT, seed=42)

    # Run dynamics for several steps
    energies = []
    for _step in range(n_steps):
        state = ts.nve_step(state=state, model=lj_model, dt=dt)

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

    NOTE: added triclinic cell after #171.
    Although the addition doesn't fail if we do not add the changes suggested in issue.
    """
    sim_state = request.getfixturevalue(sim_state_fixture_name)
    n_steps = 100

    initial_states = {
        "single": sim_state,
        "batched": ts.concatenate_states([sim_state, sim_state]),
    }

    final_states = {}
    for state_name, state in initial_states.items():
        # Initialize integrator
        kT = torch.tensor(100.0) * MetalUnits.temperature
        dt = torch.tensor(0.001)  # Small timestep for stability

        # Initialize momenta (even if zero) and get forces
        state = ts.nve_init(
            state=state, model=lj_model, kT=kT, seed=42
        )  # kT is ignored if momenta are set below
        # Ensure momenta start at zero AFTER init which might randomize them based on kT
        state.momenta = torch.zeros_like(state.momenta)  # Start from rest

        for _step in range(n_steps):
            state = ts.nve_step(state=state, model=lj_model, dt=dt)

        final_states[state_name] = state

    # Check energy conservation
    single_state = final_states["single"]
    batched_state_0 = final_states["batched"][0]
    batched_state_1 = final_states["batched"][1]

    # Compare single state results with each part of the batched state
    for final_state in (batched_state_0, batched_state_1):
        # Check positions first - most likely to fail with incorrect PBC
        torch.testing.assert_close(single_state.positions, final_state.positions)
        # Check other state components
        torch.testing.assert_close(single_state.momenta, final_state.momenta)
        torch.testing.assert_close(single_state.forces, final_state.forces)
        torch.testing.assert_close(single_state.masses, final_state.masses)
        torch.testing.assert_close(single_state.cell, final_state.cell)
        torch.testing.assert_close(single_state.energy, final_state.energy)


def test_compute_cell_force_atoms_per_system():
    """Test that compute_cell_force correctly scales by number of atoms per system."""

    # Setup minimal state with two systems having 8:1 atom ratio
    s1, s2 = torch.zeros(8, dtype=torch.long), torch.ones(64, dtype=torch.long)

    state = ts.NPTLangevinState(
        positions=torch.zeros((72, 3)),
        momenta=torch.zeros((72, 3)),
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
