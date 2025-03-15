# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
#     "moyopy>=0.4.1",
#     "pymatgen>=2025.2.18",
# ]
# ///
"""Demo of the amorphous-to-crystalline (A2C) algorithm for a-Si, ported to torchsim from
jax-md https://github.com/jax-md/jax-md/blob/main/jax_md/a2c/a2c_workflow.py.
"""

import os
import time
from collections import defaultdict

import torch
from mace.calculators.foundations_models import mace_mp
from moyopy import MoyoDataset, SpaceGroupType
from moyopy.interface import MoyoAdapter
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Element, Structure
from tqdm import tqdm

from torch_sim.models.mace import MaceModel, UnbatchedMaceModel
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.quantities import temperature
from torch_sim.io import state_to_structures, structures_to_state
from torch_sim.transforms import get_fractional_coordinates
from torch_sim.unbatched.unbatched_integrators import (
    NVTNoseHooverState,
    nvt_nose_hoover,
    nvt_nose_hoover_invariant,
)
from torch_sim.units import MetalUnits as Units
from torch_sim.utils.a2c import (
    get_subcells_to_crystallize,
    get_target_temperature,
    get_unit_cell_relaxed_structure_batched,
    random_packed_structure,
    subcells_to_structures,
)


"""
# Example of how to use random_packed_structure_multi
from torch_sim.utils.a2c import random_packed_structure_multi

comp = Composition("Fe80B20")
cell = torch.tensor(
    [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=dtype, device=device
)
structure_multi = random_packed_structure_multi(
    composition=comp,
    cell=cell,
    auto_diameter=True,
    device=device,
    dtype=dtype,
    max_iter=100,
)
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model"
raw_model = mace_mp(model=mace_checkpoint_url, return_raw_model=True)
PERIODIC = True

# Define system and model
comp = Composition("Si64")
cell = torch.tensor(
    [[11.1, 0.0, 0.0], [0.0, 11.1, 0.0], [0.0, 0.0, 11.1]], dtype=dtype, device=device
)
atomic_numbers = [Element(el).Z for el in comp.get_el_amt_dict()] * int(comp.num_atoms)

atomic_numbers = torch.tensor(atomic_numbers, device=device, dtype=torch.int)
atomic_masses = [Element(el).atomic_mass for el in comp.get_el_amt_dict()] * int(
    comp.num_atoms
)
species = [Element.from_Z(Z).symbol for Z in atomic_numbers]

model = UnbatchedMaceModel(
    model=raw_model,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    periodic=PERIODIC,
    compute_force=True,
    compute_stress=False,  # We don't need stress for MD
    dtype=dtype,
    enable_cueq=False,
)
# Workflow starts here
structure = random_packed_structure(
    composition=comp,
    cell=cell,
    auto_diameter=True,
    device=device,
    dtype=dtype,
    max_iter=100,
)

# Relax structure in batches of 6
batch_size = 1 if os.getenv("CI") else 6
max_optim_steps = (
    1 if os.getenv("CI") else 100
)  # Number of optimization steps for unit cell relaxation

# MD parameters
equi_steps = 25 if os.getenv("CI") else 2500  # MD steps for melt equilibration
cool_steps = 25 if os.getenv("CI") else 2500  # MD steps for quenching equilibration
final_steps = (
    25 if os.getenv("CI") else 2500
)  # MD steps for amorphous phase equilibration
T_high = 2000  # Melt temperature
T_low = 300  # Quench to this temperature
dt = 0.002 * Units.time  # time step = 2fs
tau = 40 * dt  # oscillation period in Nose-Hoover thermostat
simulation_steps = equi_steps + cool_steps + final_steps


nvt_nose_hoover_init, nvt_nose_hoover_update = nvt_nose_hoover(
    model=model,
    kT=T_high * Units.temperature,
    dt=dt,
)

StateDict = {
    "positions": structure.positions,
    "masses": torch.tensor(atomic_masses, device=device, dtype=dtype),
    "cell": cell,
    "pbc": PERIODIC,
    "atomic_numbers": atomic_numbers,
}
state = nvt_nose_hoover_init(StateDict)

logger = {
    "T": torch.zeros((simulation_steps, 1), device=device, dtype=dtype),
    "H": torch.zeros((simulation_steps, 1), device=device, dtype=dtype),
}


def step_fn(
    step: int, state: NVTNoseHooverState, logger: dict
) -> tuple[NVTNoseHooverState, dict]:
    """Step function for NVT-MD with Nose-Hoover thermostat."""
    current_temp = get_target_temperature(step, equi_steps, cool_steps, T_high, T_low)
    logger["T"][step] = (
        temperature(masses=state.masses, momenta=state.momenta) / Units.temperature
    )
    logger["H"][step] = nvt_nose_hoover_invariant(
        state, kT=current_temp * Units.temperature
    ).item()
    state = nvt_nose_hoover_update(state, kT=current_temp * Units.temperature)
    return state, logger


# Run NVT-MD with the melt-quench-equilibrate temperature profile
for step in range(simulation_steps):
    state, logger = step_fn(step, state, logger)
    temp, invariant = logger["T"][step].item(), logger["H"][step].item()
    print(f"Step {step}: Temperature: {temp:.4f} K: H: {invariant:.4f} eV")

print(
    f"Amorphous structure is ready: positions\n = "
    f"{state.positions}\ncell\n = {state.cell}\nspecies = {species}"
)

# Convert positions to fractional coordinates
fractional_positions = get_fractional_coordinates(
    positions=state.positions, cell=state.cell
)

# Get subcells to crystallize
subcells = get_subcells_to_crystallize(
    fractional_positions=fractional_positions,
    species=species,
    d_frac=0.2 if os.getenv("CI") else 0.1,
    n_min=2,
    n_max=8,
)
print(f"Created {len(subcells)} subcells from a-Si")

# To save time in this example, we (i) keep only the "cubic" subcells where a==b==c, and
# (ii) keep if number of atoms in the subcell is 2, 4 or 8. This reduces the number of
# subcells to relax from approx. 80k to around 160.
subcells = [
    subcell
    for subcell in subcells
    if torch.all((subcell[2] - subcell[1]) == (subcell[2] - subcell[1])[0])
    and subcell[0].shape[0] in (2, 4, 8)
]
print(f"Subcells kept for this example: {len(subcells)}")

candidate_structures = subcells_to_structures(
    candidates=subcells,
    fractional_positions=fractional_positions,
    cell=state.cell,
    species=species,
)

pymatgen_struct_list = [
    Structure(
        lattice=struct[1].detach().cpu().numpy(),
        species=struct[2],
        coords=struct[0].detach().cpu().numpy(),
        coords_are_cartesian=False,
    )
    for struct in candidate_structures
]

start_time = time.perf_counter()
# Create a batched model
model = MaceModel(
    model=raw_model,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    periodic=PERIODIC,
    compute_force=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

pymatgen_relaxed_struct_list = []
# Process structures in batches of 4
for i in tqdm(range(0, len(pymatgen_struct_list), batch_size)):
    batch_structs = pymatgen_struct_list[i : i + batch_size]
    # Combine structures into a single batched state
    batch_state = structures_to_state(batch_structs, device=device, dtype=dtype)

    final_state, logger, final_energy, final_pressure = (
        get_unit_cell_relaxed_structure_batched(
            state=batch_state,
            model=model,
            max_iter=max_optim_steps,
        )
    )

    final_struct_list = state_to_structures(final_state)

    # NOTE: Possible OOM, so we don't store the logger
    # relaxed_structures.append((pymatgen_struct, logger, final_energy, final_pressure))
    for i, final_struct in enumerate(final_struct_list):
        pymatgen_relaxed_struct_list.append(
            (final_struct, final_energy[i], final_pressure[i])
        )

lowest_e_struct = sorted(
    pymatgen_relaxed_struct_list, key=lambda x: x[-2] / x[0].num_sites
)[0]
spg = SpacegroupAnalyzer(lowest_e_struct[0])
print("Space group of predicted crystallization product:", spg.get_space_group_symbol())

spg_counter = defaultdict(int)
for struct in pymatgen_relaxed_struct_list:
    sym_data = MoyoDataset(MoyoAdapter.from_py_obj(struct[0]))
    sp = (sym_data.number, SpaceGroupType(sym_data.number).arithmetic_symbol)
spg_counter[sp] += 1

print("All space groups encountered:", dict(spg_counter))
si_diamond = Structure(
    lattice=[
        [0.0, 2.732954, 2.732954],
        [2.732954, 0.0, 2.732954],
        [2.732954, 2.732954, 0.0],
    ],
    species=["Si", "Si"],
    coords=[[0.5, 0.5, 0.5], [0.75, 0.75, 0.75]],
    coords_are_cartesian=False,
)
struct_match = StructureMatcher().fit(lowest_e_struct[0], si_diamond)
print("Prediction matches diamond-cubic Si?", struct_match)

end_time = time.perf_counter()
print(f"Total time taken to run the workflow: {end_time - start_time:.2f} seconds")
