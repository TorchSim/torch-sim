# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
#     "moyopy>=0.4.1",
#     "pymatgen>=2025.2.18",
# ]
# ///
"""
Demo of the amorphous-to-crystalline (A2C) algorithm for a-Si, ported to torchsim from
jax-md https://github.com/jax-md/jax-md/blob/main/jax_md/a2c/a2c_workflow.py.
"""

from tqdm import tqdm
from collections import defaultdict
from pymatgen.core import Composition, Element, Structure
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher

import torch
from torchsim.models.mace import UnbatchedMaceModel
from torchsim.neighbors import vesin_nl_ts
from torchsim.units import MetalUnits as Units
from torchsim.workflows import get_target_temperature
from torchsim.unbatched_integrators import nvt_nose_hoover, nvt_nose_hoover_invariant
from torchsim.unbatched_optimizers import (
    unit_cell_fire,
    UnitCellFIREState,
    fire,
    FIREState,
)
from torchsim.quantities import temperature
from torchsim.workflows import random_packed_structure
from torchsim.workflows import get_subcells_to_crystallize, subcells_to_structures
from torchsim.transforms import get_fractional_coordinates
from mace.calculators.foundations_models import mace_mp

from moyopy import MoyoDataset, SpaceGroupType
from moyopy.interface import MoyoAdapter

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
    parameters simultaneously. It uses the Fast Inertial Relaxation Engine (FIRE) algorithm
    to minimize forces on atoms and stresses on the cell.

    Args:
        fractional_positions: Atomic positions in fractional coordinates with shape [n_atoms, 3]
        cell: Unit cell tensor with shape [3, 3] containing lattice vectors
        species: List of atomic species symbols
        model: Model to compute energies, forces, and stresses
        max_iter: Maximum number of FIRE iterations. Defaults to 200.

    Returns:
        tuple containing:
            - UnitCellFIREState: Final state containing relaxed positions, cell and other quantities
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

    results = model(positions=positions, cell=cell, atomic_numbers=atomic_numbers)
    Initial_energy = results["energy"]
    Initial_stress = results["stress"]
    Initial_pressure = -torch.trace(Initial_stress) / 3.0
    print(
        f"Initial energy: {Initial_energy.item()} eV, Initial pressure: {Initial_pressure.item()} eV/A^3"
    )

    state, unit_cell_fire_update = unit_cell_fire(
        model=model,
        positions=positions,
        masses=torch.tensor(atomic_masses, device=device, dtype=dtype),
        cell=cell,
        pbc=PERIODIC,
        atomic_numbers=atomic_numbers,
    )

    def step_fn(step, state, logger):
        logger["energy"][step] = state.energy
        logger["stress"][step] = state.stress
        state = unit_cell_fire_update(state)
        return state, logger

    for step in range(max_iter):
        state, logger = step_fn(step, state, logger)
        # print(f"Step {i}: Energy = {logger['energy'][i].item()} eV: Pressure = {-torch.trace(logger['stress'][i]) / 3.0} eV/A^3")

    # Get final results
    final_results = model(
        positions=state.positions, cell=state.cell, atomic_numbers=atomic_numbers
    )

    final_energy = final_results["energy"]
    final_stress = final_results["stress"]
    final_pressure = -torch.trace(final_stress) / 3.0
    print(
        f"Final energy: {final_energy.item()} eV, Final pressure: {final_pressure.item()} eV/A^3"
    )
    return state, logger, final_energy, final_pressure


def get_relaxed_structure(
    fractional_positions: torch.Tensor,
    cell: torch.Tensor,
    species: list[str],
    model: torch.nn.Module,
    max_iter: int = 200,
) -> tuple[FIREState, dict]:
    """Relax atomic positions at fixed cell parameters using FIRE algorithm.

    This function performs geometry optimization of atomic positions while keeping the unit
    cell fixed. It uses the Fast Inertial Relaxation Engine (FIRE) algorithm to minimize
    forces on atoms.

    Args:
        fractional_positions: Atomic positions in fractional coordinates with shape [n_atoms, 3]
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

    state, fire_update = fire(
        model=model,
        positions=positions,
        masses=torch.tensor(atomic_masses, device=device, dtype=dtype),
        cell=cell,
        pbc=PERIODIC,
        atomic_numbers=atomic_numbers,
    )

    def step_fn(idx, state, logger):
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

    final_energy = final_results["energy"]
    final_stress = final_results["stress"]
    final_pressure = -torch.trace(final_stress) / 3.0
    print(
        f"Final energy: {final_energy.item()} eV, Final pressure: {final_pressure.item()} eV/A^3"
    )
    return state, logger, final_energy, final_pressure


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model"
raw_model = mace_mp(model=mace_checkpoint_url, return_raw_model=True)
PERIODIC = True

# Define system and model
comp = Composition("Si64")
cell = torch.tensor(
    [[11.1, 0.0, 0.0], [0.0, 11.1, 0.0], [0.0, 0.0, 11.1]], dtype=dtype, device=device
)
atomic_numbers = [Element(el).Z for el in comp.get_el_amt_dict().keys()] * int(
    comp.num_atoms
)

atomic_numbers = torch.tensor(atomic_numbers, device=device, dtype=torch.int)
atomic_masses = [Element(el).atomic_mass for el in comp.get_el_amt_dict().keys()] * int(
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

equi_steps = 2500  # MD steps for melt equilibration
cool_steps = 2500  # MD steps for quenching equilibration
final_steps = 2500  # MD steps for amorphous phase equilibration
T_high = 2000  # Melt temperature
T_low = 300  # Quench to this temperature
dt = 0.002 * Units.time  # time step = 2fs
tau = 40 * dt  # oscillation period in Nose-Hoover thermostat
simulation_steps = equi_steps + cool_steps + final_steps
max_optim_steps = 100  # Number of optimization steps for unit cell relaxation

state, nvt_nose_hoover_update = nvt_nose_hoover(
    model=model,
    positions=structure.positions,
    masses=torch.tensor(atomic_masses, device=device, dtype=dtype),
    cell=cell,
    pbc=PERIODIC,
    kT=T_high * Units.temperature,
    dt=dt,
    seed=1,
    atomic_numbers=atomic_numbers,
)

logger = {
    "T": torch.zeros((simulation_steps, 1), device=device, dtype=dtype),
    "H": torch.zeros((simulation_steps, 1), device=device, dtype=dtype),
}


def step_fn(step, state, logger):
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
    print(
        f"Step {step}: Temperature: {logger['T'][step].item()} K: H: {logger['H'][step].item()} eV"
    )

print(
    f"Amorphous structure is ready: positions\n = {state.positions}\ncell\n = {state.cell}\nspecies = {species}"
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
for idx, struct in tqdm(enumerate(candidate_structures)):
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
