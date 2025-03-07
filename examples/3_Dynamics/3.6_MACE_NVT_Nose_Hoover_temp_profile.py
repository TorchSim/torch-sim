"""Example NVT Nose Hoover MD simulation of random alloy using MACE model with
temperature profile.
"""

# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
#     "plotly>=6",
#     "kaleido<=1.0.0rc0",
# ]
# ///

import os

import numpy as np
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp
from plotly.subplots import make_subplots

from torchsim.models.mace import UnbatchedMaceModel
from torchsim.neighbors import vesin_nl_ts
from torchsim.quantities import temperature
from torchsim.unbatched_integrators import nvt_nose_hoover, nvt_nose_hoover_invariant
from torchsim.units import MetalUnits as Units


def get_kT(
    step: int,
    n_steps_initial: int,
    n_steps_ramp_up: int,
    n_steps_melt: int,
    n_steps_ramp_down: int,
    n_steps_anneal: int,
    melt_temp: float,
    cool_temp: float,
    anneal_temp: float,
    device: torch.device,
) -> torch.Tensor:
    """Determine target kT based on current simulation step.
    Temperature profile:
    300K (initial) → ramp to 3_000K → hold at 3_000K → quench to 300K → hold at 300K.
    """
    if step < n_steps_initial:
        # Initial equilibration at cool temperature
        return torch.tensor(cool_temp, device=device)
    if step < (n_steps_initial + n_steps_ramp_up):
        # Linear ramp from cool_temp to melt_temp
        progress = (step - n_steps_initial) / n_steps_ramp_up
        current_kT = cool_temp + (melt_temp - cool_temp) * progress
        return torch.tensor(current_kT, device=device)
    if step < (n_steps_initial + n_steps_ramp_up + n_steps_melt):
        # Hold at melting temperature
        return torch.tensor(melt_temp, device=device)
    if step < (n_steps_initial + n_steps_ramp_up + n_steps_melt + n_steps_ramp_down):
        # Linear cooling from melt_temp to cool_temp
        progress = (
            step - (n_steps_initial + n_steps_ramp_up + n_steps_melt)
        ) / n_steps_ramp_down
        current_kT = melt_temp - (melt_temp - cool_temp) * progress
        return torch.tensor(current_kT, device=device)
    if step < (
        n_steps_initial
        + n_steps_ramp_up
        + n_steps_melt
        + n_steps_ramp_down
        + n_steps_anneal
    ):
        # Hold at annealing temperature
        return torch.tensor(anneal_temp, device=device)
    # Hold at annealing temperature
    return torch.tensor(anneal_temp, device=device)


# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Model configuration
# Option 1: Load from URL (uncomment to use)
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# Option 2: Load from local file
# MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
# loaded_model = torch.load(MODEL_PATH, map_location=device, weights_only=False)

PERIODIC = True

# Temperature profile settings
init_temp = 300
melting_temp = 1000
cooling_temp = 300
annealing_temp = 300

# Step counts for different phases
n_steps_initial = 20 if os.getenv("CI") else 200
n_steps_ramp_up = 20 if os.getenv("CI") else 200
n_steps_melt = 20 if os.getenv("CI") else 200
n_steps_ramp_down = 20 if os.getenv("CI") else 200
n_steps_anneal = 20 if os.getenv("CI") else 200

n_steps = (
    n_steps_initial + n_steps_ramp_up + n_steps_melt + n_steps_ramp_down + n_steps_anneal
)

# Create a random alloy system
# Define possible species and their probabilities
species = ["Cu", "Mn", "Fe"]
probabilities = [0.33, 0.33, 0.34]

# Create base FCC structure with Cu (using Cu lattice parameter)
fcc_lattice = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((2, 2, 2))

# Randomly assign species
random_species = np.random.default_rng(seed=0).choice(
    species, size=len(fcc_lattice), p=probabilities
)
fcc_lattice.set_chemical_symbols(random_species)

# Prepare input tensors
positions = torch.tensor(fcc_lattice.positions, device=device, dtype=dtype)
cell = torch.tensor(fcc_lattice.cell.array, device=device, dtype=dtype)
atomic_numbers = torch.tensor(
    fcc_lattice.get_atomic_numbers(), device=device, dtype=torch.int
)
masses = torch.tensor(fcc_lattice.get_masses(), device=device, dtype=dtype)

# Initialize the unbatched MACE model
model = UnbatchedMaceModel(
    model=loaded_model,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    periodic=PERIODIC,
    compute_force=True,
    compute_stress=False,
    dtype=dtype,
    enable_cueq=False,
)

# Run initial inference
results = model(positions=positions, cell=cell, atomic_numbers=atomic_numbers)

# Set up simulation parameters
dt = 0.002 * Units.time
kT = init_temp * Units.temperature

state = {
    "positions": positions,
    "masses": masses,
    "cell": cell,
    "pbc": PERIODIC,
    "atomic_numbers": atomic_numbers,
}

nvt_init, nvt_update = nvt_nose_hoover(model=model, kT=kT, dt=dt)
state = nvt_init(state, kT=kT, seed=1)

# Run simulation with temperature profile
actual_temps = np.zeros(n_steps)
expected_temps = np.zeros(n_steps)

for step in range(n_steps):
    # Get target temperature for current step
    current_kT = get_kT(  # noqa: N816
        step=step,
        n_steps_initial=n_steps_initial,
        n_steps_ramp_up=n_steps_ramp_up,
        n_steps_melt=n_steps_melt,
        n_steps_ramp_down=n_steps_ramp_down,
        n_steps_anneal=n_steps_anneal,
        melt_temp=melting_temp,
        cool_temp=cooling_temp,
        anneal_temp=annealing_temp,
        device=device,
    )

    # Calculate current temperature and save data
    temp = temperature(masses=state.masses, momenta=state.momenta) / Units.temperature
    actual_temps[step] = temp
    expected_temps[step] = current_kT

    # Calculate invariant and progress report
    invariant = nvt_nose_hoover_invariant(state, kT=current_kT * Units.temperature)
    print(f"{step=}: Temperature: {temp.item():.4f}: invariant: {invariant.item():.4f}")

    # Update simulation state
    state = nvt_update(state, kT=current_kT * Units.temperature)

# Visualize temperature profile
fig = make_subplots()
fig.add_scatter(
    x=np.arange(n_steps) * 0.002, y=actual_temps, name="Simulated Temperature"
)
fig.add_scatter(
    x=np.arange(n_steps) * 0.002, y=expected_temps, name="Desired Temperature"
)
fig.layout.xaxis.title = "time (ps)"
fig.layout.yaxis.title = "Temperature (K)"
fig.write_image("nvt_visualization_temperature.pdf")
