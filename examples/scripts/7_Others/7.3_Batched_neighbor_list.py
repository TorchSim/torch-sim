"""Batched neighbor list."""

# /// script
# dependencies = ["ase>=3.26", "scipy>=1.15"]
# ///
import torch
from ase.build import bulk

import torch_sim as ts
from torch_sim import transforms
from torch_sim.neighbors import torch_nl_linked_cell, torch_nl_n2


atoms_list = [bulk("Si", "diamond", a=5.43), bulk("Ge", "diamond", a=5.65)]
state = ts.io.atoms_to_state(atoms_list, device=torch.device("cpu"), dtype=torch.float32)
pos, cell, pbc = state.positions, state.cell, state.pbc
system_idx, n_atoms = state.system_idx, state.n_atoms
cutoff = 4.0
self_interaction = False

# Fix: Ensure pbc has the correct shape [n_systems, 3]
pbc_tensor = torch.tensor([[pbc] * 3] * len(atoms_list), dtype=torch.bool)

mapping, mapping_system, shifts_idx = torch_nl_linked_cell(
    cutoff, pos, cell, pbc_tensor, system_idx, self_interaction
)
cell_shifts = transforms.compute_cell_shifts(cell, shifts_idx, mapping_system)
dds = transforms.compute_distances_with_cell_shifts(pos, mapping, cell_shifts)

print(mapping.shape)
print(mapping_system.shape)
print(shifts_idx.shape)
print(cell_shifts.shape)
print(dds.shape)

mapping_n2, mapping_system_n2, shifts_idx_n2 = torch_nl_n2(
    cutoff, pos, cell, pbc_tensor, system_idx, self_interaction
)
cell_shifts_n2 = transforms.compute_cell_shifts(cell, shifts_idx_n2, mapping_system_n2)
dds_n2 = transforms.compute_distances_with_cell_shifts(pos, mapping_n2, cell_shifts_n2)

print(mapping_n2.shape)
print(mapping_system_n2.shape)
print(shifts_idx_n2.shape)
print(cell_shifts_n2.shape)
print(dds_n2.shape)
