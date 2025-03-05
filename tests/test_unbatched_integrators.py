from typing import Any

import torch

from torchsim.quantities import kinetic_energy, temperature
from torchsim.state import BaseState
from torchsim.unbatched_integrators import (
    MDState,
    nve,
    nvt_langevin,
    nvt_nose_hoover,
    nvt_nose_hoover_invariant,
)
from torchsim.units import MetalUnits
from torchsim.utils import calculate_momenta


def test_nve_integrator(
    ar_fcc_base_state: BaseState, unbatched_lj_calculator: Any
) -> None:
    """Test NVE integration conserves energy."""
    # Initialize integrator
    kT = torch.tensor(100.0) * MetalUnits.temperature  # Temperature in K
    dt = torch.tensor(0.001) * MetalUnits.time  # Small timestep for stability

    nve_init, nve_update = nve(model=unbatched_lj_calculator, dt=dt, kT=kT)

    # Remove batch dimension from cell
    ar_fcc_base_state.cell = ar_fcc_base_state.cell.squeeze(0)

    state = nve_init(state=ar_fcc_base_state)

    # Run several steps
    energies = torch.zeros(20)
    for step in range(1000):
        state = nve_update(state, dt)
        if step % 50 == 0:
            total_energy = state.energy + kinetic_energy(state.momenta, state.masses)
            energies[step // 50] = total_energy

    # Check energy conservation
    energy_drift = torch.abs(energies - energies[1]) / torch.abs(energies[1])
    assert torch.all(energy_drift < 0.05), "Energy should be conserved in NVE"


def test_nvt_langevin_integrator(
    ar_fcc_base_state: BaseState, unbatched_lj_calculator: Any
) -> None:
    """Test Langevin thermostat maintains target temperature."""
    # Initialize integrator
    target_temp = torch.tensor(100.0) * MetalUnits.temperature
    dt = torch.tensor(0.001) * MetalUnits.time
    gamma = torch.tensor(10.0) / MetalUnits.time  # Friction coefficient

    langevin_init, langevin_update = nvt_langevin(
        model=unbatched_lj_calculator, dt=dt, kT=target_temp, gamma=gamma
    )

    # Remove batch dimension from cell
    ar_fcc_base_state.cell = ar_fcc_base_state.cell.squeeze(0)

    state = langevin_init(state=ar_fcc_base_state, seed=42)
    # Run equilibration
    temperatures = torch.zeros(500)
    for step in range(500):
        state = langevin_update(state, target_temp)
        temp = temperature(state.momenta, state.masses) / MetalUnits.temperature
        temperatures[step] = temp

    average_temperature = torch.mean(temperatures)
    # Check temperature control
    assert 120 > average_temperature > 80, "Temperature should be maintained"


def test_nvt_nose_hoover_integrator(
    ar_fcc_base_state: BaseState, unbatched_lj_calculator: Any
) -> None:
    """Test Nose-Hoover chain thermostat maintains temperature."""
    # Initialize integrator
    target_temp = torch.tensor(100.0) * MetalUnits.temperature
    dt = torch.tensor(0.001) * MetalUnits.time

    ar_fcc_base_state.cell = ar_fcc_base_state.cell.squeeze(0)

    nvt_init, nvt_update = nvt_nose_hoover(
        model=unbatched_lj_calculator,
        dt=dt,
        kT=target_temp,
        chain_length=3,
        chain_steps=3,
        sy_steps=3,
    )

    state = nvt_init(state=ar_fcc_base_state, seed=42)

    # Run equilibration
    temperatures = torch.zeros(500)
    for step in range(500):
        state = nvt_update(state, target_temp)
        temp = temperature(state.momenta, state.masses) / MetalUnits.temperature
        temperatures[step] = temp

    average_temperature = torch.mean(temperatures)
    # Check temperature control
    assert 120 > average_temperature > 80, "Temperature should be maintained"

    # Check chain properties
    assert hasattr(state, "chain"), "Should have chain thermostat"
    assert hasattr(state.chain, "positions"), "Chain should have positions"
    assert hasattr(state.chain, "momenta"), "Chain should have momenta"
    assert state.chain.positions.shape[0] == 3, "Should have 3 chain thermostats"


def test_integrator_state_properties(
    ar_fcc_base_state: BaseState, unbatched_lj_calculator: Any
) -> None:
    """Test that all integrators preserve state properties."""
    device = ar_fcc_base_state.positions.device
    dtype = ar_fcc_base_state.positions.dtype

    momenta = calculate_momenta(
        ar_fcc_base_state.positions,
        ar_fcc_base_state.masses,
        100.0 * MetalUnits.temperature,
        device,
        dtype,
    )
    md_state = MDState(
        positions=ar_fcc_base_state.positions,
        momenta=momenta,
        masses=ar_fcc_base_state.masses,
        cell=ar_fcc_base_state.cell.squeeze(0),
        pbc=ar_fcc_base_state.pbc,
        forces=torch.zeros_like(ar_fcc_base_state.positions),
        energy=torch.tensor(0.0),
        atomic_numbers=ar_fcc_base_state.atomic_numbers,
    )

    for integrator in [nve, nvt_langevin, nvt_nose_hoover]:
        init_fn, update_fn = integrator(
            model=unbatched_lj_calculator,
            dt=torch.tensor(0.001) * MetalUnits.time,
            kT=torch.tensor(100.0) * MetalUnits.temperature,
        )
        state = init_fn(state=md_state)

        # Check basic state properties
        assert hasattr(state, "positions"), "Should have positions"
        assert hasattr(state, "momenta"), "Should have momenta"
        assert hasattr(state, "forces"), "Should have forces"
        assert hasattr(state, "masses"), "Should have masses"
        assert hasattr(state, "cell"), "Should have cell"
        assert hasattr(state, "pbc"), "Should have PBC flag"

        # Check tensor shapes
        n_atoms = len(state.masses)
        assert state.positions.shape == (n_atoms, 3)
        assert state.momenta.shape == (n_atoms, 3)
        assert state.forces.shape == (n_atoms, 3)
        assert state.masses.shape == (n_atoms,)
        assert state.cell.shape == (3, 3)

        assert torch.allclose(md_state.momenta, state.momenta)


def test_nvt_nose_hoover_invariant(
    ar_fcc_base_state: BaseState, unbatched_lj_calculator: Any
) -> None:
    """Test Nose-Hoover chain thermostat maintains temperature."""
    # Initialize integrator
    target_temp = torch.tensor(100.0) * MetalUnits.temperature
    dt = torch.tensor(0.001) * MetalUnits.time

    ar_fcc_base_state.cell = ar_fcc_base_state.cell.squeeze(0)

    nvt_init, nvt_update = nvt_nose_hoover(
        model=unbatched_lj_calculator,
        dt=dt,
        kT=target_temp,
        chain_length=3,
        chain_steps=3,
        sy_steps=3,
    )

    state = nvt_init(state=ar_fcc_base_state, seed=42)

    # Run equilibration
    invariant = torch.zeros(500)
    for step in range(500):
        state = nvt_update(state, target_temp)
        invariant[step] = nvt_nose_hoover_invariant(state, target_temp)

    # Check energy conservation
    invariant_drift = torch.abs(invariant - invariant[0]) / torch.abs(invariant[0])
    assert torch.all(invariant_drift < 0.05), "invariant should be conserved"
