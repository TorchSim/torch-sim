"""Hybrid swap Monte Carlo simulation."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
#     "pymatgen>=2025.2.18",
# ]
# ///

from dataclasses import dataclass

import torch
from mace.calculators.foundations_models import mace_mp
from pymatgen.core import Structure

import torch_sim as ts
from torch_sim.integrators import MDState, nvt_langevin
from torch_sim.models.mace import MaceModel
from torch_sim.monte_carlo import swap_monte_carlo
from torch_sim.units import MetalUnits


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

kT = 1000 * MetalUnits.temperature

# Option 1: Load the raw model from the downloaded model
mace_checkpoint_url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# Option 2: Load from local file (comment out Option 1 to use this)
# MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
# loaded_model = torch.load(MODEL_PATH, map_location=device)

model = MaceModel(
    model=loaded_model,
    device=device,
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
class HybridSwapMCState(MDState):
    """State for Monte Carlo simulations.

    Attributes:
        energy: Energy of the system
        last_swap: Last swap attempted
    """

    last_permutation: torch.Tensor


nvt_init, nvt_step = nvt_langevin(model=model, dt=0.002, kT=kT)
md_state = nvt_init(state, seed=42)

swap_init, swap_step = swap_monte_carlo(model=model, kT=kT, seed=42)
swap_state = swap_init(md_state)
hybrid_state = HybridSwapMCState(
    **vars(md_state),
    last_permutation=torch.zeros(
        md_state.n_batches, device=md_state.device, dtype=torch.bool
    ),
)

generator = torch.Generator(device=device)
generator.manual_seed(42)

n_steps = 100
for step in range(n_steps):
    if step % 10 == 0:
        hybrid_state = swap_step(hybrid_state, kT=torch.tensor(kT), generator=generator)
    else:
        hybrid_state = nvt_step(hybrid_state, dt=torch.tensor(0.002), kT=torch.tensor(kT))
