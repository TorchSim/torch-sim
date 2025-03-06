import torch
from ase.build import bulk

from torchsim.neighbors import torch_nl_linked_cell, torch_nl_n2
from torchsim.runners import atoms_to_state
from torchsim.transforms import compute_cell_shifts, compute_distances_with_cell_shifts


atoms_list = [bulk("Si", "diamond", a=5.43), bulk("Ge", "diamond", a=5.65)]
state = atoms_to_state(atoms_list, device="cpu", dtype=torch.float32)
pos, cell, pbc = state.positions, state.cell, state.pbc
batch, n_atoms = state.batch, state.n_atoms
cutoff = 4.0
self_interaction = False

mapping, mapping_batch, shifts_idx = torch_nl_linked_cell(
    cutoff, pos, cell, pbc, batch, self_interaction
)
cell_shifts = compute_cell_shifts(cell, shifts_idx, mapping_batch)
dds = compute_distances_with_cell_shifts(pos, mapping, cell_shifts)

print(mapping.shape)
print(mapping_batch.shape)
print(shifts_idx.shape)
print(cell_shifts.shape)
print(dds.shape)

mapping_n2, mapping_batch_n2, shifts_idx_n2 = torch_nl_n2(
    cutoff, pos, cell, pbc, batch, self_interaction
)
cell_shifts_n2 = compute_cell_shifts(cell, shifts_idx_n2, mapping_batch_n2)
dds_n2 = compute_distances_with_cell_shifts(pos, mapping_n2, cell_shifts_n2)

print(mapping_n2.shape)
print(mapping_batch_n2.shape)
print(shifts_idx_n2.shape)
print(cell_shifts_n2.shape)
print(dds_n2.shape)
