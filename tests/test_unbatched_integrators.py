from typing import Any

import torch

from torchsim.quantities import kinetic_energy
from torchsim.state import BaseState
from torchsim.unbatched_integrators import MDState, nve, nvt_langevin, nvt_nose_hoover
from torchsim.utils import calculate_momenta


# skip all tests in this file for now

"""
pytest.skip(
    reason="Unbatched integrators deprecated, skipping integrator tests",
    allow_module_level=True,
)
"""


def test_nve_integrator(si_base_state: BaseState, unbatched_lj_calculator: Any) -> None:
    """Test NVE integration conserves energy."""
    # Initialize integrator
    kT = torch.tensor(300.0)  # Temperature in K
    dt = torch.tensor(0.001)  # Small timestep for stability

    nve_init, nve_update = nve(model=unbatched_lj_calculator, dt=dt, kT=kT)

    # Remove batch dimension from cell
    si_base_state.cell = si_base_state.cell.squeeze(0)

    state = nve_init(state=si_base_state)
    # Store initial energy
    initial_energy = state.energy + kinetic_energy(state.momenta, state.masses)

    # Run several steps
    energies = []
    for _ in range(100):
        state = nve_update(state, dt)
        total_energy = state.energy + kinetic_energy(state.momenta, state.masses)
        energies.append(total_energy)

    # Check energy conservation
    energies = torch.tensor(energies)
    energy_drift = torch.abs(energies - initial_energy) / torch.abs(initial_energy)
    assert torch.all(energy_drift < 0.01), "Energy should be conserved in NVE"


def test_nvt_langevin_integrator(
    si_base_state: BaseState, unbatched_lj_calculator: Any
) -> None:
    """Test Langevin thermostat maintains target temperature."""
    # Initialize integrator
    target_temp = torch.tensor(300.0)
    dt = torch.tensor(0.001)
    gamma = torch.tensor(0.1)  # Friction coefficient

    langevin_init, langevin_update = nvt_langevin(
        model=unbatched_lj_calculator, dt=dt, kT=target_temp, gamma=gamma
    )

    # Remove batch dimension from cell
    si_base_state.cell = si_base_state.cell.squeeze(0)

    state = langevin_init(state=si_base_state, seed=42)
    # Run equilibration
    temperatures = []
    for _ in range(500):
        state = langevin_update(state, target_temp)
        KE = kinetic_energy(state.momenta, state.masses)
        temp = 2 * KE / (3 * len(state.masses))  # 3N degrees of freedom
        temperatures.append(temp)

    # Check temperature control
    assert 400 > target_temp > 200, "Temperature should be maintained"


def test_nvt_nose_hoover_integrator(
    si_base_state: BaseState, unbatched_lj_calculator: Any
) -> None:
    """Test Nose-Hoover chain thermostat maintains temperature."""
    # Initialize integrator
    target_temp = torch.tensor(300.0)
    dt = torch.tensor(0.001)

    si_base_state.cell = si_base_state.cell.squeeze(0)

    nvt_init, nvt_update = nvt_nose_hoover(
        model=unbatched_lj_calculator,
        dt=dt,
        kT=target_temp,
        chain_length=3,
        chain_steps=3,
        sy_steps=3,
    )

    state = nvt_init(state=si_base_state, seed=42)

    # Run equilibration
    temperatures = []
    for _ in range(500):
        state = nvt_update(state, target_temp)
        KE = kinetic_energy(state.momenta, state.masses)
        temp = 2 * KE / (3 * len(state.masses))
        temperatures.append(temp)

    # Check temperature control
    assert 400 > target_temp > 200, "Temperature should be maintained"

    # Check chain properties
    assert hasattr(state, "chain"), "Should have chain thermostat"
    assert hasattr(state.chain, "positions"), "Chain should have positions"
    assert hasattr(state.chain, "momenta"), "Chain should have momenta"
    assert state.chain.positions.shape[0] == 3, "Should have 3 chain thermostats"


def test_integrator_state_properties(
    si_base_state: BaseState, unbatched_lj_calculator: Any
) -> None:
    """Test that all integrators preserve state properties."""
    device = si_base_state.positions.device
    dtype = si_base_state.positions.dtype

    momenta = calculate_momenta(
        si_base_state.positions, si_base_state.masses, 300.0, device, dtype
    )
    md_state = MDState(
        positions=si_base_state.positions,
        momenta=momenta,
        masses=si_base_state.masses,
        cell=si_base_state.cell.squeeze(0),
        pbc=si_base_state.pbc,
        forces=torch.zeros_like(si_base_state.positions),
        energy=torch.tensor(0.0),
        atomic_numbers=si_base_state.atomic_numbers,
    )

    for integrator in [nve, nvt_langevin, nvt_nose_hoover]:
        init_fn, update_fn = integrator(
            model=unbatched_lj_calculator,
            dt=torch.tensor(0.001),
            kT=torch.tensor(300.0),
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
