"""Example NVT Nose Hoover MD simulation of random alloy using MACE model with
temperature profile.
"""

# /// script
# dependencies = [
#     "mace-torch>=0.3.12",
#     "plotly>=6",
#     "kaleido",
# ]
# ///

import os

import numpy as np
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp
from plotly.subplots import make_subplots

import torch_sim as ts
from torch_sim.integrators.nvt import nvt_nose_hoover, nvt_nose_hoover_invariant
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.quantities import calc_kT
from torch_sim.units import MetalUnits as Units


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
loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# Option 2: Load from local file
# MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
# loaded_model = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# Temperature profile settings
init_temp = 300
melting_temp = 1000
cooling_temp = 300
annealing_temp = 300

# Step counts for different phases
SMOKE_TEST = os.getenv("CI") is not None
n_steps_initial = 20 if SMOKE_TEST else 200
n_steps_ramp_up = 20 if SMOKE_TEST else 200
n_steps_melt = 20 if SMOKE_TEST else 200
n_steps_ramp_down = 20 if SMOKE_TEST else 200
n_steps_anneal = 20 if SMOKE_TEST else 200

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

# Initialize the MACE model
model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=False,
    dtype=dtype,
    enable_cueq=False,
)
state = ts.io.atoms_to_state(fcc_lattice, device=device, dtype=dtype)

# Run initial inference
results = model(state)

# Set up simulation parameters
dt = 0.002 * Units.time
kT = torch.tensor(init_temp, device=device, dtype=dtype) * Units.temperature

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
    temp = (
        calc_kT(masses=state.masses, momenta=state.momenta, system_idx=state.system_idx)
        / Units.temperature
    )
    actual_temps[step] = temp
    expected_temps[step] = current_kT

    # Calculate invariant and progress report
    invariant = float(nvt_nose_hoover_invariant(state, kT=current_kT * Units.temperature))
    print(f"{step=}: Temperature: {temp.item():.4f}: {invariant=:.4f}")

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
