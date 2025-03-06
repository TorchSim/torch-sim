"""Example NVT Nose Hoover MD simulation of random alloy using MACE model with temperature profile."""

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
    num_steps_initial: int,
    num_steps_ramp_up: int,
    num_steps_melt: int,
    num_steps_ramp_down: int,
    num_steps_anneal: int,
    melt_temp: float,
    cool_temp: float,
    anneal_temp: float,
    device: torch.device,
) -> torch.Tensor:
    """Determine target kT based on current simulation step.
    Temperature profile:
    300K (initial) → ramp to 3_000K → hold at 3_000K → quench to 300K → hold at 300K.
    """
    if step < num_steps_initial:
        # Initial equilibration at cool temperature
        return torch.tensor(cool_temp, device=device)
    if step < (num_steps_initial + num_steps_ramp_up):
        # Linear ramp from cool_temp to melt_temp
        progress = (step - num_steps_initial) / num_steps_ramp_up
        current_kT = cool_temp + (melt_temp - cool_temp) * progress
        return torch.tensor(current_kT, device=device)
    if step < (num_steps_initial + num_steps_ramp_up + num_steps_melt):
        # Hold at melting temperature
        return torch.tensor(melt_temp, device=device)
    if step < (
        num_steps_initial + num_steps_ramp_up + num_steps_melt + num_steps_ramp_down
    ):
        # Linear cooling from melt_temp to cool_temp
        progress = (
            step - (num_steps_initial + num_steps_ramp_up + num_steps_melt)
        ) / num_steps_ramp_down
        current_kT = melt_temp - (melt_temp - cool_temp) * progress
        return torch.tensor(current_kT, device=device)
    if step < (
        num_steps_initial
        + num_steps_ramp_up
        + num_steps_melt
        + num_steps_ramp_down
        + num_steps_anneal
    ):
        # Hold at annealing temperature
        return torch.tensor(anneal_temp, device=device)
    # Hold at annealing temperature
    return torch.tensor(anneal_temp, device=device)


# Set device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
Initial_temperature = 300
Melting_temperature = 3_000
Cooling_temperature = 300
Annealing_temperature = 300

# Step counts for different phases
Num_steps_initial = 1_000
Num_steps_ramp_up = 1_000
Num_steps_melt = 1_000
Num_steps_ramp_down = 1_000
Num_steps_anneal = 1_000

Num_steps = (
    Num_steps_initial
    + Num_steps_ramp_up
    + Num_steps_melt
    + Num_steps_ramp_down
    + Num_steps_anneal
)

# Create a random alloy system
# Define possible species and their probabilities
species = ["Cu", "Mn", "Fe"]
probabilities = [0.33, 0.33, 0.34]

# Create base FCC structure with Cu (using Cu lattice parameter)
fcc_lattice = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((2, 2, 2))

# Randomly assign species
random_species = np.random.choice(species, size=len(fcc_lattice), p=probabilities)
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
kT = Initial_temperature * Units.temperature

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
Temperature = np.zeros(Num_steps)
Expected_temperature = np.zeros(Num_steps)

for step in range(Num_steps):
    # Get target temperature for current step
    current_kT = get_kT(
        step=step,
        num_steps_initial=Num_steps_initial,
        num_steps_ramp_up=Num_steps_ramp_up,
        num_steps_melt=Num_steps_melt,
        num_steps_ramp_down=Num_steps_ramp_down,
        num_steps_anneal=Num_steps_anneal,
        melt_temp=Melting_temperature,
        cool_temp=Cooling_temperature,
        anneal_temp=Annealing_temperature,
        device=device,
    )

    # Calculate current temperature and save data
    temp = temperature(masses=state.masses, momenta=state.momenta) / Units.temperature
    Temperature[step] = temp
    Expected_temperature[step] = current_kT

    # Calculate invariant and progress report
    invariant = nvt_nose_hoover_invariant(state, kT=current_kT * Units.temperature)
    print(f"{step=}: Temperature: {temp.item():.4f}: invariant: {invariant.item():.4f}")

    # Update simulation state
    state = nvt_update(state, kT=current_kT * Units.temperature)

# Visualize temperature profile
fig = make_subplots()
fig.add_scatter(
    x=np.arange(Num_steps) * 0.002, y=Temperature, name="Simulated Temperature"
)
fig.add_scatter(
    x=np.arange(Num_steps) * 0.002, y=Expected_temperature, name="Desired Temperature"
)
fig.update_layout(
    xaxis_title="time (ps)",
)
fig.update_yaxes(title_text="Temperature (K)")
fig.write_image("nvt_visualization_temperature.pdf")
