"""Hybrid swap Monte Carlo simulation."""

# /// script
# dependencies = ["mace-torch>=0.3.12", "pymatgen>=2025.2.18"]
# ///
from dataclasses import dataclass

import torch
from mace.calculators.foundations_models import mace_mp
from pymatgen.core import Structure

import torch_sim as ts
from torch_sim.integrators.md import MDState
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.units import MetalUnits as Units


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
kT = 1000 * Units.temperature

# Option 1: Load the raw model from the downloaded model
loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    return_raw_model=True,
    default_dtype=str(dtype).removeprefix("torch."),
    device=str(device),
)

# Option 2: Load from local file (comment out Option 1 to use this)
# loaded_model = torch.load("path/to/model.pt", map_location=device)

model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=False,
    dtype=dtype,
    enable_cueq=False,
)

# %%
lattice = [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
species = ["Cu", "Cu", "Cu", "Zr", "Cu", "Zr", "Zr", "Zr"]
coords = [
    [0.0, 0.0, 0.0],
    [0.25, 0.25, 0.25],
    [0.0, 0.5, 0.5],
    [0.25, 0.75, 0.75],
    [0.5, 0.0, 0.5],
    [0.75, 0.25, 0.75],
    [0.5, 0.5, 0.0],
    [0.75, 0.75, 0.25],
]
structure = Structure(lattice, species, coords)

state = ts.io.structures_to_state([structure], device=device, dtype=dtype)


# %%
@dataclass
class HybridSwapMCState(ts.SwapMCState, MDState):
    """State for Monte Carlo simulations.

    Attributes:
        energy: Energy of the system
        last_swap: Last swap attempted
    """

    _atom_attributes = (
        ts.SwapMCState._atom_attributes | MDState._atom_attributes  # noqa: SLF001
    )
    _system_attributes = (
        ts.SwapMCState._system_attributes | MDState._system_attributes  # noqa: SLF001
    )


md_state = ts.nvt_langevin_init(state=state, model=model, kT=torch.tensor(kT), seed=42)

swap_state = ts.swap_mc_init(state=md_state, model=model)
hybrid_state = HybridSwapMCState(
    **vars(md_state),
    last_permutation=torch.arange(
        md_state.n_atoms, device=md_state.device, dtype=torch.long
    ),
)

rng = torch.Generator(device=device)
rng.manual_seed(42)

n_steps = 100
dt = torch.tensor(0.002)
for step in range(n_steps):
    if step % 10 == 0:
        hybrid_state = ts.swap_mc_step(state=hybrid_state, model=model, kT=kT, rng=rng)
    else:
        hybrid_state = ts.nvt_langevin_step(
            model=model, state=hybrid_state, dt=dt, kT=torch.tensor(kT)
        )
