"""NPT simulation with MACE and Nose-Hoover thermostat."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
# ]
# ///

import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

from torchsim.models.mace import UnbatchedMaceModel
from torchsim.neighbors import vesin_nl_ts
from torchsim.quantities import kinetic_energy, temperature
from torchsim.unbatched_integrators import (
    npt_nose_hoover,
    npt_nose_hoover_invariant,
    nvt_nose_hoover,
    nvt_nose_hoover_invariant,
)
from torchsim.units import MetalUnits as Units


# Set device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Option 1: Load the raw model from the downloaded model
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# Option 2: Load from local file (comment out Option 1 to use this)
# MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
# loaded_model = torch.load(MODEL_PATH, map_location=device)

PERIODIC = True

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))

# Prepare input tensors
positions = torch.tensor(si_dc.positions, device=device, dtype=dtype)
cell = torch.tensor(si_dc.cell.array, device=device, dtype=dtype)
atomic_numbers = torch.tensor(si_dc.get_atomic_numbers(), device=device, dtype=torch.int)
masses = torch.tensor(si_dc.get_masses(), device=device, dtype=dtype)

# Print shapes for verification
print(f"Positions: {positions.shape}")
print(f"Cell: {cell.shape}")

# Initialize the unbatched MACE model
model = UnbatchedMaceModel(
    model=loaded_model,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    periodic=PERIODIC,
    compute_force=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

# Run initial inference
results = model(positions=positions, cell=cell, atomic_numbers=atomic_numbers)

N_steps_nvt = 500
N_steps_npt = 500
dt = 0.001 * Units.time  # Time step (1 ps)
kT = 300 * Units.temperature  # Initial temperature (300 K)
target_pressure = 0.0 * Units.pressure  # Target pressure (0 bar)

state = {
    "positions": positions,
    "masses": masses,
    "cell": cell,
    "pbc": PERIODIC,
    "atomic_numbers": atomic_numbers,
}

nvt_init, nvt_update = nvt_nose_hoover(model=model, kT=kT, dt=dt)
state = nvt_init(state=state, seed=1)

for step in range(N_steps_nvt):
    if step % 10 == 0:
        temp = temperature(masses=state.masses, momenta=state.momenta) / Units.temperature
        invariant = nvt_nose_hoover_invariant(state, kT=kT).item()
        print(f"{step=}: Temperature: {temp:.4f}: invariant: {invariant:.4f}, ")
    state = nvt_update(state, kT=kT)

npt_init, npt_update = npt_nose_hoover(
    model=model,
    kT=kT,
    dt=dt,
    external_pressure=target_pressure,
)
state = npt_init(state=state, seed=1)


def get_pressure(stress, kinetic_energy, volume, dim=3):
    """Compute the pressure from the stress tensor.

    The stress tensor is defined as 1/volume * dU/de_ij
    So the pressure is -1/volume * trace(dU/de_ij)
    """
    return 1 / (dim) * ((2 * kinetic_energy / volume) - torch.trace(stress))


for step in range(N_steps_npt):
    if step % 10 == 0:
        temp = temperature(masses=state.masses, momenta=state.momenta) / Units.temperature
        invariant = npt_nose_hoover_invariant(
            state, kT=kT, external_pressure=target_pressure
        ).item()
        print(
            f"{step=}: Temperature: {temp:.4f}: invariant: {invariant:.4f}, "
            f"pressure: {
                get_pressure(
                    model(
                        positions=state.positions,
                        cell=state.current_box,
                        atomic_numbers=state.atomic_numbers,
                    )['stress'],
                    kinetic_energy(masses=state.masses, momenta=state.momenta),
                    torch.det(state.current_box),
                ).item()
                / Units.pressure:.4f
            }, "
            f"box xx yy zz: {state.current_box[0, 0].item():.4f}, {state.current_box[1, 1].item():.4f}, {state.current_box[2, 2].item():.4f}"
        )
    state = npt_update(state, kT=kT, external_pressure=target_pressure)

print(
    f"Final temperature: {temperature(masses=state.masses, momenta=state.momenta) / Units.temperature:.4f}"
)
print(
    f"Final pressure: {get_pressure(model(positions=state.positions, cell=state.current_box, atomic_numbers=state.atomic_numbers)['stress'], kinetic_energy(masses=state.masses, momenta=state.momenta), torch.det(state.current_box)).item() / Units.pressure:.4f}"
)
