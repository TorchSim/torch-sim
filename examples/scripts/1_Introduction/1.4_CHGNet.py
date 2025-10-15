"""CHGNet model example for TorchSim."""

# /// script
# dependencies = ["chgnet>=0.4.2"]
# ///

import warnings
import os
import torch
import torch_sim as ts
from ase.build import bulk
from ase import Atoms
from torch_sim.models.chgnet import CHGNetModel

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
ca_tio3_atoms = Atoms(['Ca', 'Ti', 'O', 'O', 'O'],
                     positions=[[0, 0, 0], [a_perovskite/2, a_perovskite/2, a_perovskite/2], 
                               [a_perovskite/2, 0, 0], [0, a_perovskite/2, 0], [0, 0, a_perovskite/2]],
                     cell=[a_perovskite, a_perovskite, a_perovskite], pbc=True)

# Convert to TorchSim state
state = ts.io.atoms_to_state([al_atoms, c_atoms, mg_atoms], device, dtype)

# Load MACE model for comparison
try:
    from torch_sim.models.mace import MaceModel, MaceUrls
    from mace.calculators.foundations_models import mace_mp
    
    raw_mace_mp = mace_mp(model=MaceUrls.mace_mp_small, return_raw_model=True)
    mace_model = MaceModel(
        model=raw_mace_mp,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=True,
    )
    mace_available = True
except ImportError:
    mace_available = False
    print("MACE not available for comparison")

# In this table we compare CHGNet and MACE on the equilibrium structures
print("\nTABLE 1: Equilibrium Structures")
print("=" * 80)
print(f"{'System':<10} {'CHGNet E (eV)':<15} {'CHGNet F (eV/Å)':<15} {'MACE E (eV)':<15} {'MACE F (eV/Å)':<15}")
print("-" * 80)

for i, system_name in enumerate(["Al", "C", "Mg"]):
    # Get states
    single_state = ts.io.atoms_to_state([[al_atoms, c_atoms, mg_atoms][i]], device, dtype)
    
    # CHGNet results
    chgnet_result = model.forward(single_state)
    chgnet_energy = chgnet_result['energy'].item()
    chgnet_force = torch.norm(chgnet_result['forces'], dim=1).max().item()
    
    # MACE results
    if mace_available:
        mace_result = mace_model.forward(single_state)
        mace_energy = mace_result['energy'].item()
        mace_force = torch.norm(mace_result['forces'], dim=1).max().item()
        print(f"{system_name:<10} {chgnet_energy:<15.6f} {chgnet_force:<15.6f} {mace_energy:<15.6f} {mace_force:<15.6f}")
    else:
        print(f"{system_name:<10} {chgnet_energy:<15.6f} {chgnet_force:<15.6f} {'N/A':<15} {'N/A':<15}")

# In this table we compare CHGNet and MACE on the displaced and optimized structures
print("\nTABLE 2: Displaced and Optimized Structures")
print("=" * 100)
print(f"{'System':<10} {'CHGNet Init E':<15} {'CHGNet Fin E':<15} {'CHGNet Fin F':<15} {'MACE Init E':<15} {'MACE Fin E':<15} {'MACE Fin F':<15}")
print("-" * 120)

for i, (atoms, system_name) in enumerate(zip([al_atoms, c_atoms, ca_tio3_atoms], ["Al", "C", "CaTiO3"])):
    # Create displaced state
    single_state = ts.io.atoms_to_state([atoms], device, dtype)
    displacement = torch.randn_like(single_state.positions) * 0.1
    displaced_state = single_state.clone()
    displaced_state.positions = single_state.positions + displacement
    
    # CHGNet optimization
    chgnet_initial = model.forward(displaced_state)
    chgnet_initial_energy = chgnet_initial['energy'].item()
    
    chgnet_optimized = ts.optimize(
        displaced_state,
        model,
        optimizer=ts.optimizers.Optimizer.fire,
        max_steps=100,
    )
    
    chgnet_final = model.forward(chgnet_optimized)
    chgnet_final_energy = chgnet_final['energy'].item()
    chgnet_final_force = torch.norm(chgnet_final['forces'], dim=1).max().item()
    
    # MACE optimization
    if mace_available:
        mace_initial = mace_model.forward(displaced_state)
        mace_initial_energy = mace_initial['energy'].item()
        
        mace_optimized = ts.optimize(
            displaced_state,
            mace_model,
            optimizer=ts.optimizers.Optimizer.fire,
            max_steps=100,
        )
        
        mace_final = mace_model.forward(mace_optimized)
        mace_final_energy = mace_final['energy'].item()
        mace_final_force = torch.norm(mace_final['forces'], dim=1).max().item()
        
        print(f"{system_name:<10} {chgnet_initial_energy:<15.6f} {chgnet_final_energy:<15.6f} {chgnet_final_force:<15.6f} {mace_initial_energy:<15.6f} {mace_final_energy:<15.6f} {mace_final_force:<15.6f}")
    else:
        print(f"{system_name:<10} {chgnet_initial_energy:<15.6f} {chgnet_final_energy:<15.6f} {chgnet_final_force:<15.6f} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

print("\n" + "="*100)
print("CHGNet example completed successfully!")
print("="*100)

