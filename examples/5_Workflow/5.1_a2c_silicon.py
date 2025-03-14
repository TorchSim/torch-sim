# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
#     "moyopy>=0.4.1",
#     "pymatgen>=2025.2.18",
# ]
# ///
"""Demo of the amorphous-to-crystalline (A2C) algorithm for a-Si, ported to torchsim from
jax-md https://github.com/jax-md/jax-md/blob/main/jax_md/a2c/a2c_workflow.py.
"""

import os
from collections import defaultdict

import torch
from mace.calculators.foundations_models import mace_mp
from moyopy import MoyoDataset, SpaceGroupType
from moyopy.interface import MoyoAdapter
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Element, Structure
from tqdm import tqdm

from torchsim.models.mace import UnbatchedMaceModel
from torchsim.neighbors import vesin_nl_ts
from torchsim.quantities import temperature
from torchsim.transforms import get_fractional_coordinates
from torchsim.unbatched_integrators import (
    NVTNoseHooverState,
    nvt_nose_hoover,
    nvt_nose_hoover_invariant,
)
from torchsim.unbatched_optimizers import (
    FIREState,
    UnitCellFIREState,
    fire,
    unit_cell_fire,
)
from torchsim.units import MetalUnits as Units
from torchsim.workflows import (
    get_subcells_to_crystallize,
    get_target_temperature,
    random_packed_structure,
    subcells_to_structures,
)


"""
# Example of how to use random_packed_structure_multi
from torchsim.workflows import random_packed_structure_multi

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


# Helper functions
def get_unit_cell_relaxed_structure(
    fractional_positions: torch.Tensor,
    cell: torch.Tensor,
    species: list[str],
    model: torch.nn.Module,
    max_iter: int = 200,
) -> tuple[UnitCellFIREState, dict]:
    """Relax both atomic positions and cell parameters using FIRE algorithm.

    This function performs geometry optimization of both atomic positions and unit cell
    parameters simultaneously. Uses the Fast Inertial Relaxation Engine (FIRE) algorithm
    to minimize forces on atoms and stresses on the cell.

    Args:
        fractional_positions: Fractional atomic coordinates with shape [n_atoms, 3]
        cell: Unit cell tensor with shape [3, 3] containing lattice vectors
        species: List of atomic species symbols
        model: Model to compute energies, forces, and stresses
        max_iter: Maximum number of FIRE iterations. Defaults to 200.

    Returns:
        tuple containing:
            - UnitCellFIREState: Final state containing relaxed positions, cell and more
            - dict: Logger with energy and stress trajectories
            - float: Final energy in eV
            - float: Final pressure in eV/Å³
    """
    # Get atomic masses from species
    atomic_numbers = [Element(s).Z for s in species]
    atomic_numbers = torch.tensor(atomic_numbers, device=device, dtype=torch.int)
    atomic_masses = [Element(s).atomic_mass for s in species]
    positions = torch.matmul(fractional_positions, cell)
    logger = {
        "energy": torch.zeros((max_iter, 1), device=device, dtype=dtype),
        "stress": torch.zeros((max_iter, 3, 3), device=device, dtype=dtype),
    }

    # Make sure to compute stress
    model.compute_stress = True

    StateDict = {
        "positions": positions,
        "masses": torch.tensor(atomic_masses, device=device, dtype=dtype),
        "cell": cell,
        "pbc": PERIODIC,
        "atomic_numbers": atomic_numbers,
    }
    results = model(positions=positions, cell=cell, atomic_numbers=atomic_numbers)
    init_energy = results["energy"].item()
    init_stress = results["stress"]
    init_pressure = (-torch.trace(init_stress) / 3.0).item()
    print(f"Initial energy: {init_energy} eV, Initial pressure: {init_pressure} eV/A^3")

    unit_cell_fire_init, unit_cell_fire_update = unit_cell_fire(
        model=model,
    )
    state = unit_cell_fire_init(StateDict)

    def step_fn(
        step: int, state: UnitCellFIREState, logger: dict
    ) -> tuple[UnitCellFIREState, dict]:
        logger["energy"][step] = state.energy
        logger["stress"][step] = state.stress
        state = unit_cell_fire_update(state)
        return state, logger

    for step in range(max_iter):
        state, logger = step_fn(step, state, logger)
        # energy, stress = logger["energy"][step].item(), logger["stress"][step]
        # pressure = -torch.trace(stress) / 3.0
        # print(f"Step {step}: Energy = {energy} eV: Pressure = {pressure} eV/A^3")

    # Get final results
    final_results = model(
        positions=state.positions, cell=state.cell, atomic_numbers=atomic_numbers
    )

    final_energy = final_results["energy"].item()
    final_stress = final_results["stress"]
    final_pressure = (-torch.trace(final_stress) / 3.0).item()
    print(f"Final energy: {final_energy} eV, Final pressure: {final_pressure} eV/A^3")
    return state, logger, final_energy, final_pressure


def get_relaxed_structure(
    fractional_positions: torch.Tensor,
    cell: torch.Tensor,
    species: list[str],
    model: torch.nn.Module,
    max_iter: int = 200,
) -> tuple[FIREState, dict]:
    """Relax atomic positions at fixed cell parameters using FIRE algorithm.

    Does geometry optimization of atomic positions while keeping the unit cell fixed.
    Uses the Fast Inertial Relaxation Engine (FIRE) algorithm to minimize forces on atoms.

    Args:
        fractional_positions: Fractional atomic coordinates with shape [n_atoms, 3]
        cell: Unit cell tensor with shape [3, 3] containing lattice vectors
        species: List of atomic species symbols
        model: Model to compute energies, forces, and stresses
        max_iter: Maximum number of FIRE iterations. Defaults to 200.

    Returns:
        tuple containing:
            - FIREState: Final state containing relaxed positions and other quantities
            - dict: Logger with energy trajectory
            - float: Final energy in eV
            - float: Final pressure in eV/Å³
    """
    # Get atomic masses from species
    atomic_numbers = [Element(s).Z for s in species]
    atomic_numbers = torch.tensor(atomic_numbers, device=device, dtype=torch.int)
    atomic_masses = [Element(s).atomic_mass for s in species]
    positions = torch.matmul(fractional_positions, cell)
    logger = {"energy": torch.zeros((max_iter, 1), device=device, dtype=dtype)}

    results = model(positions=positions, cell=cell, atomic_numbers=atomic_numbers)
    Initial_energy = results["energy"]
    print(f"Initial energy: {Initial_energy.item()} eV")

    state = UnitCellFIREState(
        positions=positions,
        cell=cell,
        atomic_numbers=atomic_numbers,
        masses=torch.tensor(atomic_masses, device=device, dtype=dtype),
    )
    state_init_fn, fire_update = fire(model=model)
    state = state_init_fn(state)

    def step_fn(idx: int, state: FIREState, logger: dict) -> tuple[FIREState, dict]:
        logger["energy"][idx] = state.energy
        state = fire_update(state)
        return state, logger

    for idx in range(max_iter):
        state, logger = step_fn(idx, state, logger)
        # print(f"Step {i}: Energy = {logger['energy'][i].item()} eV")

    # Get final results
    model.compute_stress = True
    final_results = model(
        positions=state.positions, cell=state.cell, atomic_numbers=atomic_numbers
    )

    final_energy = final_results["energy"].item()
    final_stress = final_results["stress"]
    final_pressure = (-torch.trace(final_stress) / 3.0).item()
    print(f"Final energy: {final_energy} eV, Final pressure: {final_pressure} eV/A^3")
    return state, logger, final_energy, final_pressure


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
    compute_stress=False,
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
max_optim_steps = (
    1 if os.getenv("CI") else 100
)  # Number of optimization steps for unit cell relaxation

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
    d_frac=0.2,
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

relaxed_structures = []
for struct in tqdm(candidate_structures):
    state, logger, final_energy, final_pressure = get_unit_cell_relaxed_structure(
        fractional_positions=struct[0],
        cell=struct[1],
        species=struct[2],
        model=model,
        max_iter=max_optim_steps,
    )

    pymatgen_struct = Structure(
        lattice=state.cell.detach().cpu().numpy(),
        species=struct[2],
        coords=state.positions.detach().cpu().numpy(),
        coords_are_cartesian=True,
    )
    relaxed_structures.append((pymatgen_struct, logger, final_energy, final_pressure))

lowest_e_struct = sorted(relaxed_structures, key=lambda x: x[-2] / x[0].num_sites)[0]
spg = SpacegroupAnalyzer(lowest_e_struct[0])
print("Space group of predicted crystallization product:", spg.get_space_group_symbol())

spg_counter = defaultdict(int)
for struct in relaxed_structures:
    sym_data = MoyoDataset(MoyoAdapter.from_py_obj(struct[0]))
    sp = (sym_data.number, SpaceGroupType(sym_data.number).arithmetic_symbol)
spg_counter[sp] += 1

print("All space groups encountered:", dict(spg_counter))
si_diamond = Structure.from_str(
    """Si
1.0
0.000000000000   2.732954000000   2.732954000000
2.732954000000   0.000000000000   2.732954000000
2.732954000000   2.732954000000   0.000000000000
Si
2
Direct
0.500000000000   0.500000000000   0.500000000000
0.750000000000   0.750000000000   0.750000000000""",
    fmt="poscar",
)
struct_match = StructureMatcher().fit(lowest_e_struct[0], si_diamond)
print("Prediction matches diamond-cubic Si?", struct_match)
