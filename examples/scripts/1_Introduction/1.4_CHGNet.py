"""CHGNet model example for TorchSim."""

# /// script
# dependencies = ["chgnet>=0.4.2", "mace-torch>=0.3.12"]
# ///

import os
import warnings

import torch
from ase import Atoms
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.models.chgnet import CHGNetModel
from torch_sim.models.mace import MaceModel, MaceUrls


# Silence warnings
warnings.filterwarnings("ignore")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

print("CHGNet Example for TorchSim")
print("=" * 40)

# Create CHGNet model
model = CHGNetModel(
    device=device,
    dtype=dtype,
    compute_forces=True,
    compute_stress=True,
)

# Create test systems
al_atoms = bulk("Al", "fcc", a=4.05, cubic=True)
c_atoms = bulk("C", "diamond", a=3.57, cubic=True)
mg_atoms = bulk("Mg", "hcp", a=3.21, c=5.21)
a_perovskite = 3.84
ca_tio3_atoms = Atoms(
    ["Ca", "Ti", "O", "O", "O"],
    positions=[
        [0, 0, 0],
        [a_perovskite / 2, a_perovskite / 2, a_perovskite / 2],
        [a_perovskite / 2, 0, 0],
        [0, a_perovskite / 2, 0],
        [0, 0, a_perovskite / 2],
    ],
    cell=[a_perovskite, a_perovskite, a_perovskite],
    pbc=True,
)

# Convert to TorchSim state
state = ts.io.atoms_to_state([al_atoms, c_atoms, mg_atoms], device, dtype)

# Load MACE model for comparison
raw_mace_mp = mace_mp(model=MaceUrls.mace_mp_small, return_raw_model=True)
mace_model = MaceModel(
    model=raw_mace_mp,
    device=device,
    dtype=dtype,
    compute_forces=True,
    compute_stress=True,
)
mace_available = True

# Single comprehensive results table
print(
    "\nCHGNet vs MACE Results "
    "(E: Total Energy, F: Maximum Force, S: Maximum Stress, M: Maximum Magnetic Moment)"
)
print("=" * 87)
print(
    f"{'System':<10} {'CHGNet E':<12} {'CHGNet F':<12} {'CHGNet S':<12} "
    f"{'CHGNet M':<12} {'MACE E':<12} {'MACE F':<12}"
)
print("-" * 87)

# Test equilibrium structures
for i, system_name in enumerate(["Al", "C", "Mg"]):
    single_state = ts.io.atoms_to_state([[al_atoms, c_atoms, mg_atoms][i]], device, dtype)

    # CHGNet results
    chgnet_result = model.forward(single_state)
    chgnet_energy = chgnet_result["energy"].item()
    chgnet_force = torch.norm(chgnet_result["forces"], dim=1).max().item()
    chgnet_stress = torch.norm(chgnet_result["stress"], dim=(1, 2)).max().item()
    chgnet_magmom = (
        torch.norm(chgnet_result.get("magnetic_moments", torch.zeros(1, 3)), dim=-1)
        .max()
        .item()
    )

    # MACE results
    mace_result = mace_model.forward(single_state)
    mace_energy = mace_result["energy"].item()
    mace_force = torch.norm(mace_result["forces"], dim=1).max().item()
    print(
        f"{system_name:<10} {chgnet_energy:<12.3f} {chgnet_force:<12.3f} "
        f"{chgnet_stress:<12.3f} {chgnet_magmom:<12.3f} {mace_energy:<12.3f} "
        f"{mace_force:<12.3f}"
    )

# Test optimization on displaced structures
for atoms, system_name in zip(
    [al_atoms, c_atoms, ca_tio3_atoms], ["Al", "C", "CaTiO3"], strict=False
):
    single_state = ts.io.atoms_to_state([atoms], device, dtype)
    displacement = torch.randn_like(single_state.positions) * 0.1
    displaced_state = single_state.clone()
    displaced_state.positions = single_state.positions + displacement

    # CHGNet optimization
    chgnet_optimized = ts.optimize(
        displaced_state, model, optimizer=ts.optimizers.Optimizer.fire, max_steps=100
    )
    chgnet_final = model.forward(chgnet_optimized)
    chgnet_final_energy = chgnet_final["energy"].item()
    chgnet_final_force = torch.norm(chgnet_final["forces"], dim=1).max().item()
    chgnet_final_stress = torch.norm(chgnet_final["stress"], dim=(1, 2)).max().item()
    chgnet_final_magmom = (
        torch.norm(chgnet_final.get("magnetic_moments", torch.zeros(1, 3)), dim=-1)
        .max()
        .item()
    )

    # MACE optimization
    mace_optimized = ts.optimize(
        displaced_state,
        mace_model,
        optimizer=ts.optimizers.Optimizer.fire,
        max_steps=100,
    )
    mace_final = mace_model.forward(mace_optimized)
    mace_final_energy = mace_final["energy"].item()
    mace_final_force = torch.norm(mace_final["forces"], dim=1).max().item()
    print(
        f"{system_name + '_opt':<10} {chgnet_final_energy:<12.3f} "
        f"{chgnet_final_force:<12.3f} {chgnet_final_stress:<12.3f} "
        f"{chgnet_final_magmom:<12.3f} {mace_final_energy:<12.3f} "
        f"{mace_final_force:<12.3f}"
    )

print("=" * 87)
print("CHGNet example completed successfully!")
