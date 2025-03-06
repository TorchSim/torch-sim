from typing import Any

import pytest
import torch

from torchsim.integrators import batched_initialize_momenta, nve, nvt_langevin
from torchsim.models.lennard_jones import LennardJonesModel
from torchsim.quantities import temperature
from torchsim.state import BaseState, concatenate_states, slice_substate
from torchsim.unbatched_integrators import MDState, initialize_momenta
from torchsim.units import MetalUnits


@pytest.fixture
def ar_double_base_state(ar_base_state: BaseState) -> BaseState:
    """Create a basic state from ar_fcc_base_state."""
    batch = torch.repeat_interleave(torch.arange(2), ar_base_state.positions.shape[0])

    return BaseState(
        positions=torch.cat([ar_base_state.positions, ar_base_state.positions], dim=0),
        cell=torch.cat([ar_base_state.cell, ar_base_state.cell], dim=0),
        masses=torch.cat([ar_base_state.masses, ar_base_state.masses], dim=0),
        atomic_numbers=torch.cat(
            [ar_base_state.atomic_numbers, ar_base_state.atomic_numbers], dim=0
        ),
        batch=batch,
        pbc=ar_base_state.pbc,
    )


def batched_initialize_momenta_loop(
    positions: torch.Tensor,  # shape: (n_batches, n_atoms_per_batch, 3)
    masses: torch.Tensor,  # shape: (n_batches, n_atoms_per_batch)
    kT: torch.Tensor,  # shape: (n_batches,)
    seeds: torch.Tensor,  # shape: (n_batches,)
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Initialize momenta for batched molecular dynamics.

    Args:
        positions: Tensor of atomic positions with shape
            (n_batches, n_atoms_per_batch, 3).
        masses: Tensor of atomic masses with shape
            (n_batches, n_atoms_per_batch).
        kT: Tensor of temperature values in energy units for
            each batch with shape (n_batches,).
        seeds: Tensor of random seeds for each batch with shape (n_batches,).
        device: The device on which to allocate the tensors (e.g., 'cpu' or 'cuda').
        dtype: The data type of the tensors (e.g., torch.float32).

    Returns:
        momenta: Tensor of initialized momenta with shape
            (n_batches, n_atoms_per_batch, 3).
    """
    n_batches = positions.shape[0]
    n_atoms_per_batch = positions.shape[1]

    # Initialize momenta tensor
    momenta = torch.zeros((n_batches, n_atoms_per_batch, 3), device=device, dtype=dtype)

    # Create a generator for each batch using the provided seeds
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]

    # Generate random momenta for each batch
    for i in range(n_batches):
        # Generate random velocities from normal distribution
        batch_momenta = torch.randn(
            (n_atoms_per_batch, 3), device=device, dtype=dtype, generator=generators[i]
        )

        # Scale by sqrt(mass * kT)
        mass_factors = torch.sqrt(masses[i]).unsqueeze(-1)
        kT_factor = torch.sqrt(kT[i])
        batch_momenta *= mass_factors * kT_factor

        # Remove center of mass motion if more than one atom
        if n_atoms_per_batch > 1:
            mean_momentum = torch.mean(batch_momenta, dim=0, keepdim=True)
            batch_momenta = batch_momenta - mean_momentum

        momenta[i] = batch_momenta

    return momenta


def test_batched_initialize_momenta_loop():
    # Set random seed for reproducibility
    seed = 42

    device = torch.device("cpu")
    dtype = torch.float64

    n_batches = 3
    n_atoms_per_batch = 4

    # Create test inputs
    positions = torch.randn(n_batches, n_atoms_per_batch, 3, device=device, dtype=dtype)
    masses = torch.rand(n_batches, n_atoms_per_batch, device=device, dtype=dtype) + 0.5
    kT = torch.tensor([0.1, 0.2, 0.3], device=device, dtype=dtype)
    seeds = torch.arange(seed, seed + n_batches, dtype=torch.int64)

    # Run non-batched version first
    unbatched_momenta = []
    for i in range(n_batches):
        state = MDState(
            positions=positions[i],
            momenta=torch.zeros_like(positions[i]),
            masses=masses[i],
            forces=torch.zeros_like(positions[i]),
            energy=torch.zeros(1, device=device, dtype=dtype),
            atomic_numbers=torch.ones(
                n_atoms_per_batch, device=device, dtype=torch.int64
            ),
            cell=torch.eye(3, device=device, dtype=dtype),
            pbc=False,
        )
        state = initialize_momenta(state, kT[i], device, dtype, seed=int(seeds[i]))
        unbatched_momenta.append(state.momenta)

    unbatched_momenta = torch.stack(unbatched_momenta)

    # Run batched version
    batched_momenta = batched_initialize_momenta_loop(
        positions=positions,
        masses=masses,
        kT=kT,
        seeds=seeds,  # seeds before device and dtype
        device=device,
        dtype=dtype,
    )

    assert torch.allclose(batched_momenta, unbatched_momenta, rtol=1e-6)


def test_batched_initialize_momenta():
    seed = 42
    device = torch.device("cpu")
    dtype = torch.float64

    n_batches = 3
    n_atoms_per_batch = 4

    # Create test inputs
    positions = torch.randn(n_batches, n_atoms_per_batch, 3, device=device, dtype=dtype)
    masses = torch.rand(n_batches, n_atoms_per_batch, device=device, dtype=dtype) + 0.5
    kT = torch.tensor([0.1, 0.2, 0.3], device=device, dtype=dtype)
    seeds = torch.arange(seed, seed + n_batches, dtype=torch.int64)

    # Run non-batched version first
    unbatched_momenta = []
    for batch_idx in range(n_batches):
        # Use corresponding seed for each batch

        state = MDState(
            positions=positions[batch_idx],
            momenta=torch.zeros_like(positions[batch_idx]),
            masses=masses[batch_idx],
            forces=torch.zeros_like(positions[batch_idx]),
            energy=torch.zeros(1, device=device, dtype=dtype),
            atomic_numbers=torch.ones(
                n_atoms_per_batch, device=device, dtype=torch.int64
            ),
            cell=torch.eye(3, device=device, dtype=dtype),
            pbc=False,
        )
        state = initialize_momenta(
            state, kT[batch_idx], device, dtype, seed=int(seeds[batch_idx])
        )
        unbatched_momenta.append(state.momenta)

    unbatched_momenta = torch.stack(unbatched_momenta)

    # Run batched version
    batched_momenta = batched_initialize_momenta(
        positions, masses, kT, seeds, device, dtype
    )

    assert torch.allclose(batched_momenta, unbatched_momenta, rtol=1e-6)

    # Verify center of mass momentum is zero for each batch
    for batch_idx in range(n_batches):
        com_momentum = torch.mean(batched_momenta[batch_idx], dim=0)
        assert torch.allclose(
            com_momentum, torch.zeros(3, device=device, dtype=dtype), atol=1e-10
        )


def test_nvt_langevin(
    ar_double_base_state: BaseState,
    lj_calculator: LennardJonesModel,
    device: torch.device,
) -> None:
    dtype = torch.float64
    n_steps = 100
    dt = torch.tensor(0.001, device=device, dtype=dtype)
    kT = torch.tensor(100.0, device=device, dtype=dtype) * MetalUnits.temperature

    # Initialize integrator
    init_fn, update_fn = nvt_langevin(
        model=lj_calculator,
        dt=dt,
        kT=kT,
    )

    # Run dynamics for several steps
    state = init_fn(state=ar_double_base_state, seed=42)
    energies = []
    temperatures = []
    for _ in range(n_steps):
        state = update_fn(state=state)

        # Calculate instantaneous temperature from kinetic energy
        temp = temperature(state.momenta, state.masses, batch=state.batch)

        energies.append(state.energy)
        temperatures.append(temp * 11606)

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
            abs(mean_temp - kT.item() * 11606) < 150.0
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


def test_nve(
    ar_double_base_state: BaseState,
    lj_calculator: LennardJonesModel,
    device: torch.device,
):
    dtype = torch.float64
    n_steps = 100
    dt = torch.tensor(0.001, device=device, dtype=dtype)
    kT = torch.tensor(100.0, device=device, dtype=dtype) * MetalUnits.temperature

    # Initialize integrator
    nve_init, nve_update = nve(model=lj_calculator, dt=dt, kT=kT)
    state = nve_init(state=ar_double_base_state, seed=42)

    # Run dynamics for several steps
    energies = []
    for _ in range(n_steps):
        state = nve_update(state=state, dt=dt)

        energies.append(state.energy)

    energies_tensor = torch.stack(energies)

    # assert conservation of energy
    assert torch.allclose(energies_tensor[:, 0], energies_tensor[0, 0], atol=1e-4)
    assert torch.allclose(energies_tensor[:, 1], energies_tensor[0, 1], atol=1e-4)


def test_compare_single_vs_batched_integrators(
    ar_base_state: BaseState, lj_calculator: Any
) -> None:
    """Test that single and batched integrators give the same results."""

    initial_states = {
        "single": ar_base_state,
        "batched": concatenate_states([ar_base_state, ar_base_state]),
    }

    final_states = {}
    for state_name, state in initial_states.items():
        # Initialize integrator
        kT = torch.tensor(100.0) * MetalUnits.temperature
        dt = torch.tensor(0.001)  # Small timestep for stability

        nve_init, nve_update = nve(model=lj_calculator, dt=dt, kT=kT)
        state = nve_init(state=state, seed=42)
        state.momenta = torch.zeros_like(state.momenta)

        for _ in range(100):
            state = nve_update(state=state, dt=dt)

        final_states[state_name] = state

    # Check energy conservation
    ar_single_state = final_states["single"]
    ar_batched_state_0 = slice_substate(final_states["batched"], 0)
    ar_batched_state_1 = slice_substate(final_states["batched"], 1)

    for final_state in [ar_batched_state_0, ar_batched_state_1]:
        assert torch.allclose(ar_single_state.positions, final_state.positions)
        assert torch.allclose(ar_single_state.momenta, final_state.momenta)
        assert torch.allclose(ar_single_state.forces, final_state.forces)
        assert torch.allclose(ar_single_state.masses, final_state.masses)
        assert torch.allclose(ar_single_state.cell, final_state.cell)
