import copy
import random

import numpy as np
import pytest
import torch
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

import torch_sim as ts
from torch_sim.io import state_to_atoms
from torch_sim.optimizers import frechet_cell_fire
from torch_sim.state import SimState


pytest.importorskip("mace")
from mace.calculators import MACECalculator
from mace.calculators.foundations_models import mace_mp

from torch_sim.models.mace import MaceModel


# Seed everything
torch.manual_seed(123)
rng = np.random.default_rng(123)
random.seed(123)


@pytest.fixture
def torchsim_mace_mp_model() -> MaceModel:
    """Provides a MACE MP model instance for the optimizer tests."""
    # Use float64 for potentially higher precision needed in optimization
    dtype = torch.float64
    mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
    mace_model_raw = mace_mp(
        model=mace_checkpoint_url,
        return_raw_model=True,
        default_dtype=str(dtype).split(".")[-1],
    )
    return MaceModel(
        model=mace_model_raw,
        device="cpu",
        dtype=dtype,
        compute_forces=True,
        compute_stress=True,
    )


@pytest.fixture
def ase_mace_mp_calculator() -> MACECalculator:
    """Provides an ASE MACECalculator instance using mace_mp."""
    # Ensure dtype matches the one used in the torchsim fixture (float64)
    mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
    dtype_str = str(torch.float64).split(".")[-1]
    # Use the mace_mp function to get the ASE calculator directly
    return mace_mp(
        model=mace_checkpoint_url,
        device="cpu",
        default_dtype=dtype_str,
        dispersion=False,
    )


def test_unit_cell_frechet_fire_vs_ase(
    rattled_sio2_sim_state: SimState,
    torchsim_mace_mp_model: MaceModel,
    ase_mace_mp_calculator: MACECalculator,
) -> None:
    """Compare Frechet Cell FIRE optimizer with
    ASE's FIRE + FrechetCellFilter using MACE."""

    # Use float64 for consistency with the MACE model fixture
    dtype = torch.float64
    device = torchsim_mace_mp_model.device

    # --- Setup Initial State with float64 ---
    initial_state = copy.deepcopy(rattled_sio2_sim_state).to(dtype=dtype, device=device)

    # Ensure grads are enabled for both positions and cell
    initial_state.positions = initial_state.positions.detach().requires_grad_(
        requires_grad=True
    )
    initial_state.cell = initial_state.cell.detach().requires_grad_(requires_grad=True)

    n_steps = 20
    force_tol = 0.02

    # --- Run Custom Frechet Cell FIRE with MACE model ---

    custom_state = ts.optimize(
        system=initial_state,
        model=torchsim_mace_mp_model,
        optimizer=frechet_cell_fire,
        max_steps=n_steps,
        convergence_fn=ts.generate_force_convergence_fn(force_tol=force_tol),
    )
    # --- Setup ASE System with native MACE calculator ---
    ase_atoms = state_to_atoms(initial_state)[0]
    # Use the native ASE MACE calculator fixture, not the removed TorchCalculator
    ase_atoms.calc = ase_mace_mp_calculator

    # --- Run ASE FIRE with FrechetCellFilter ---
    filtered_atoms = FrechetCellFilter(ase_atoms)
    ase_opt = FIRE(filtered_atoms)

    ase_opt.run(fmax=force_tol, steps=n_steps)

    # --- Compare Results (between custom_state and ase_sim_state) ---
    final_custom_energy = custom_state.energy.item()
    final_custom_forces_max = torch.norm(custom_state.forces, dim=-1).max().item()

    final_custom_pos = custom_state.positions.detach()
    final_custom_cell = custom_state.row_vector_cell.squeeze(0).detach()

    final_ase_energy = ase_atoms.get_potential_energy()
    final_ase_forces = torch.tensor(ase_atoms.get_forces(), device=device, dtype=dtype)

    if final_ase_forces is not None:
        final_ase_forces_max = torch.norm(final_ase_forces, dim=-1).max().item()
    else:
        final_ase_forces_max = float("nan")

    final_ase_pos = torch.tensor(ase_atoms.get_positions(), device=device, dtype=dtype)
    final_ase_cell = torch.tensor(ase_atoms.get_cell(), device=device, dtype=dtype)

    # Compare energies (use looser tolerance for ML potential comparison)
    assert abs(final_custom_energy - final_ase_energy) < 5e-2, (
        f"Final energies differ significantly after {n_steps} steps: "
        f"Custom={final_custom_energy:.6f}, ASE_State={final_ase_energy:.6f}"
    )

    # Compare forces (report)
    print(
        f"Max Force ({n_steps} steps): Custom={final_custom_forces_max:.4f}, "
        f"ASE_State={final_ase_forces_max:.4f}"
    )

    # Compare positions (looser tolerance)
    pos_diff = torch.norm(final_custom_pos - final_ase_pos, dim=-1).mean().item()
    assert pos_diff < 1.0, (
        f"Final positions differ significantly (avg displacement: {pos_diff:.4f})"
    )

    # Compare cell matrices (looser tolerance)
    cell_diff = torch.norm(final_custom_cell - final_ase_cell).item()
    assert cell_diff < 1.0, (
        f"Final cell matrices differ significantly (Frobenius norm: {cell_diff:.4f})"
        f"\nCustom Cell:\n{final_custom_cell}"
        f"\nASE_State Cell:\n{final_ase_cell}"
    )
