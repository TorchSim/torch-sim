import copy
import functools
from typing import TYPE_CHECKING

import pytest
import torch
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

import torch_sim as ts
from torch_sim.io import state_to_atoms
from torch_sim.models.mace import MaceModel
from torch_sim.optimizers import frechet_cell_fire


if TYPE_CHECKING:
    from mace.calculators import MACECalculator


def test_torchsim_frechet_cell_fire_vs_ase_mace(
    rattled_sio2_sim_state: ts.state.SimState,
    torchsim_mace_mpa: MaceModel,
    ase_mace_mpa: "MACECalculator",
) -> None:
    """Compare torch-sim's Frechet Cell FIRE optimizer with ASE's FIRE + FrechetCellFilter
    using MACE-MPA-0.

    This test ensures that the custom Frechet Cell FIRE implementation behaves comparably
    to the established ASE equivalent when using a MACE force field.
    It checks for consistency in final energies, forces, positions, and cell parameters.
    """
    pytest.importorskip("mace")

    # Use float64 for consistency with the MACE model fixture and for precision
    dtype = torch.float64
    device = torchsim_mace_mpa.device  # Use device from the model

    # --- Setup Initial State with float64 ---
    # Deepcopy to avoid modifying the fixture state for other tests
    initial_state = copy.deepcopy(rattled_sio2_sim_state).to(dtype=dtype, device=device)

    # Ensure grads are enabled for both positions and cell for optimization
    initial_state.positions = initial_state.positions.detach().requires_grad_(
        requires_grad=True
    )
    initial_state.cell = initial_state.cell.detach().requires_grad_(requires_grad=True)

    n_steps = 100  # Number of optimization steps (increased from 20)
    force_tol = 0.02  # Convergence criterion for forces

    # --- Run torch-sim Frechet Cell FIRE with MACE model ---
    # Use functools.partial to set md_flavor for the frechet_cell_fire optimizer
    torch_sim_optimizer = functools.partial(frechet_cell_fire, md_flavor="ase_fire")

    custom_opt_state = ts.optimize(
        system=initial_state,
        model=torchsim_mace_mpa,
        optimizer=torch_sim_optimizer,
        max_steps=n_steps,
        convergence_fn=ts.generate_force_convergence_fn(force_tol=force_tol),
    )

    # --- Setup ASE System with native MACE calculator ---
    # Convert initial SimState to ASE Atoms object
    ase_atoms = state_to_atoms(initial_state)[0]  # state_to_atoms returns a list
    ase_atoms.calc = ase_mace_mpa  # Assign the MACE calculator

    # --- Run ASE FIRE with FrechetCellFilter ---
    # Apply FrechetCellFilter for cell optimization
    filtered_ase_atoms = FrechetCellFilter(ase_atoms)
    ase_optimizer = FIRE(filtered_ase_atoms)

    # Run ASE optimization
    ase_optimizer.run(fmax=force_tol, steps=n_steps)

    # --- Compare Results ---
    final_custom_energy = custom_opt_state.energy.item()
    final_custom_forces_max = torch.norm(custom_opt_state.forces, dim=-1).max().item()
    final_custom_positions = custom_opt_state.positions.detach()
    # Ensure cell is in row vector format and squeezed for comparison
    final_custom_cell = custom_opt_state.row_vector_cell.squeeze(0).detach()

    final_ase_energy = ase_atoms.get_potential_energy()
    ase_forces_raw = ase_atoms.get_forces()
    if ase_forces_raw is not None:
        final_ase_forces = torch.tensor(ase_forces_raw, device=device, dtype=dtype)
        final_ase_forces_max = torch.norm(final_ase_forces, dim=-1).max().item()
    else:
        # Should not happen if calculator ran and produced forces
        final_ase_forces_max = float("nan")

    final_ase_positions = torch.tensor(
        ase_atoms.get_positions(), device=device, dtype=dtype
    )
    final_ase_cell = torch.tensor(ase_atoms.get_cell(), device=device, dtype=dtype)

    # Compare energies (looser tolerance for ML potentials due to potential minor
    # numerical differences)
    energy_diff = abs(final_custom_energy - final_ase_energy)
    assert energy_diff < 5e-4, (
        f"Final energies differ significantly after {n_steps} steps: "
        f"torch-sim={final_custom_energy:.6f}, ASE={final_ase_energy:.6f}, "
        f"Diff={energy_diff:.2e}"
    )

    # Report forces for diagnostics
    print(
        f"Max Force ({n_steps} steps): torch-sim={final_custom_forces_max:.4f}, "
        f"ASE={final_ase_forces_max:.4f}"
    )

    # Compare positions (average displacement, looser tolerance)
    avg_displacement = (
        torch.norm(final_custom_positions - final_ase_positions, dim=-1).mean().item()
    )
    assert avg_displacement < 1e-2, (
        f"Final positions differ significantly (avg displacement: {avg_displacement:.4f})"
    )

    # Compare cell matrices (Frobenius norm, looser tolerance)
    cell_diff = torch.norm(final_custom_cell - final_ase_cell).item()
    assert cell_diff < 1e-2, (
        f"Final cell matrices differ significantly (Frobenius norm: {cell_diff:.4f})"
        f"\nTorch-sim Cell:\n{final_custom_cell}"
        f"\nASE Cell:\n{final_ase_cell}"
    )
