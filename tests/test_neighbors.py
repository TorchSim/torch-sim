import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule
from ase.neighborlist import neighbor_list

from torchsim.neighbors import (
    standard_nl,
    torch_nl_linked_cell,
    torch_nl_n2,
    vesin_nl,
    vesin_nl_ts,
    wrapping_nl,
)
from torchsim.transforms import compute_cell_shifts, compute_distances_with_cell_shifts


def ase_to_torch_batch(
    atoms_list: list[Atoms],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a list of ASE Atoms objects into tensors suitable for PyTorch.

    Args:
        atoms_list (list[Atoms]): A list of ASE Atoms objects
            representing atomic structures.
        device (torch.device, optional): The device to which
            the tensors will be moved. Defaults to "cpu".
        dtype (torch.dtype, optional): The data type of the tensors.
            Defaults to torch.float32.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing:
            - pos: Tensor of atomic positions.
            - cell: Tensor of unit cell vectors.
            - pbc: Tensor indicating periodic boundary conditions.
            - batch: Tensor indicating the batch index for each atom.
            - n_atoms: Tensor containing the number of atoms in each structure.
    """
    n_atoms = torch.tensor([len(atoms) for atoms in atoms_list], dtype=torch.long)
    pos = torch.cat([torch.from_numpy(atoms.get_positions()) for atoms in atoms_list])
    cell = torch.cat([torch.from_numpy(atoms.get_cell().array) for atoms in atoms_list])
    pbc = torch.cat([torch.from_numpy(atoms.get_pbc()) for atoms in atoms_list])

    stride = torch.cat((torch.tensor([0]), n_atoms.cumsum(0)))
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    for ii, (st, end) in enumerate(
        zip(stride[:-1], stride[1:], strict=True)  # noqa: RUF007
    ):
        batch[st:end] = ii
    n_atoms = torch.Tensor(n_atoms[1:]).to(dtype=torch.long)
    return (
        pos.to(dtype=dtype, device=device),
        cell.to(dtype=dtype, device=device),
        pbc.to(device=device),
        batch.to(device=device),
        n_atoms.to(device=device),
    )


DEVICE = torch.device("cpu")
DTYPE = torch.float64

# Adapted from torch_nl test
# https://github.com/felixmusil/torch_nl/blob/main/torch_nl/test_nl.py

# triclinic atomic structure
CaCrP2O7_mvc_11955_symmetrized = {
    "positions": [
        [3.68954016, 5.03568186, 4.64369552],
        [5.12301681, 2.13482791, 2.66220405],
        [1.99411973, 0.94691001, 1.25068234],
        [6.81843724, 6.22359976, 6.05521724],
        [2.63005662, 4.16863452, 0.86090529],
        [6.18250036, 3.00187525, 6.44499428],
        [2.11497733, 1.98032773, 4.53610884],
        [6.69757964, 5.19018203, 2.76979073],
        [1.39215545, 2.94386142, 5.60917746],
        [7.42040152, 4.22664834, 1.69672212],
        [2.43224207, 5.4571615, 6.70305327],
        [6.3803149, 1.71334827, 0.6028463],
        [1.11265639, 1.50166318, 3.48760997],
        [7.69990058, 5.66884659, 3.8182896],
        [3.56971588, 5.20836551, 1.43673437],
        [5.2428411, 1.96214426, 5.8691652],
        [3.12282634, 2.72812741, 1.05450432],
        [5.68973063, 4.44238236, 6.25139525],
        [3.24868468, 2.83997522, 3.99842386],
        [5.56387229, 4.33053455, 3.30747571],
        [2.60835346, 0.74421609, 5.3236629],
        [6.20420351, 6.42629368, 1.98223667],
    ],
    "cell": [
        [6.19330899, 0.0, 0.0],
        [2.4074486111396207, 6.149627748674982, 0.0],
        [0.2117993724186579, 1.0208820183960539, 7.305899571570074],
    ],
    "numbers": [*[20] * 2, *[24] * 2, *[15] * 4, *[8] * 14],
    "pbc": [True, True, True],
}


def periodic_structures():
    return [
        bulk("Si", "diamond", a=6, cubic=True),
        bulk("Si", "diamond", a=6),
        bulk("Cu", "fcc", a=3.6),
        bulk("Si", "bct", a=6, c=3),
        # test very skewed unit cell
        bulk("Bi", "rhombohedral", a=6, alpha=20),
        bulk("Bi", "rhombohedral", a=6, alpha=10),
        bulk("Bi", "rhombohedral", a=6, alpha=5),
        bulk("SiCu", "rocksalt", a=6),
        bulk("SiFCu", "fluorite", a=6),
        Atoms(**CaCrP2O7_mvc_11955_symmetrized),
    ]


def structure_set() -> list:
    return [
        molecule("CH3CH2NH2"),
        molecule("H2O"),
        molecule("methylenecyclopropane"),
        *periodic_structures(),
        molecule("OCHCHO"),
        molecule("C3H9C"),
    ]


@pytest.mark.parametrize("cutoff", [1, 3, 5, 7])
def test_wrapping_nl(*, cutoff: int) -> None:
    """Check that wrapping_nl gives the same NL as ASE by comparing
    the resulting sorted list of distances between neighbors.
    """
    structures = [bulk("Si", "diamond", a=6, cubic=True)]

    for structure in structures:
        # Convert to torch tensors
        pos = torch.tensor(structure.positions, device=DEVICE, dtype=DTYPE)
        cell = torch.tensor(structure.cell.array, device=DEVICE, dtype=DTYPE)

        pbc = structure.pbc.any()

        # Get the neighbor list from wrapping_nl
        mapping, shifts = wrapping_nl(
            positions=pos,
            cell=cell,
            pbc=pbc,
            cutoff=torch.tensor(cutoff, device=DEVICE, dtype=DTYPE),
        )

        # Calculate distances with cell shifts
        cell_shifts = torch.mm(shifts, cell)
        dds = compute_distances_with_cell_shifts(pos, mapping, cell_shifts)
        dds = np.sort(dds.numpy())

        # Get the neighbor list from ase
        idx_i, idx_j, shifts_ref, dist = neighbor_list(
            quantities="ijSd",
            a=structure,
            cutoff=cutoff,
            self_interaction=False,
            max_nbins=1e6,
        )

        # Convert to torch tensors
        idx_i = torch.tensor(idx_i, dtype=torch.long, device=torch.device("cpu"))
        idx_j = torch.tensor(idx_j, dtype=torch.long, device=torch.device("cpu"))

        # Create mapping and shifts
        mapping_ref = torch.stack((idx_i, idx_j), dim=0)
        shifts_ref = torch.tensor(
            shifts_ref, dtype=torch.float64, device=torch.device("cpu")
        )

        # Calculate distances with cell shifts
        cell_shifts_ref = torch.mm(shifts_ref, cell)
        dds_ref = compute_distances_with_cell_shifts(pos, mapping_ref, cell_shifts_ref)

        # Sort the distances
        dds_ref = np.sort(dds_ref.numpy())
        dist_ref = np.sort(dist)

        # Debugging information
        print(dds.shape, dds_ref.shape, cutoff, structure.get_chemical_formula())

        # Check that the distances are the same with ase and torchsim logic
        np.testing.assert_allclose(dds_ref, dist_ref)

        # Check that the distances are the same with both methods
        np.testing.assert_allclose(dds, dds_ref)

        # Check that the distances are the same with ase direct neighbor list
        np.testing.assert_allclose(dds, dist_ref)


@pytest.mark.parametrize("cutoff", [1, 3, 5, 7])
def test_standard_nl(*, cutoff: float) -> None:
    """Check that standard_nl gives the same NL as ASE by comparing
    the resulting sorted list of distances between neighbors.
    """
    structures = structure_set()

    for structure in structures:
        # Convert to torch tensors
        pos = torch.tensor(structure.positions, device=DEVICE, dtype=DTYPE)
        cell = torch.tensor(structure.cell.array, device=DEVICE, dtype=DTYPE)

        pbc = structure.pbc.any() if structure.pbc.any() else False

        # Get the neighbor list from standard_nl
        # Note: No self-interaction
        mapping, shifts = standard_nl(
            positions=pos,
            cell=cell,
            pbc=pbc,
            cutoff=torch.tensor(cutoff, dtype=DTYPE, device=DEVICE),
        )

        # Calculate distances with cell shifts
        cell_shifts = torch.mm(shifts, cell)
        dds = compute_distances_with_cell_shifts(pos, mapping, cell_shifts)
        dds = np.sort(dds.numpy())

        # Get the neighbor list from ase
        idx_i, idx_j, shifts_ref, dist = neighbor_list(
            quantities="ijSd",
            a=structure,
            cutoff=cutoff,
            self_interaction=False,
            max_nbins=1e6,
        )

        # Convert to torch tensors
        idx_i = torch.tensor(idx_i, dtype=torch.long, device=torch.device("cpu"))
        idx_j = torch.tensor(idx_j, dtype=torch.long, device=torch.device("cpu"))

        # Create mapping and shifts
        mapping_ref = torch.stack((idx_i, idx_j), dim=0)
        shifts_ref = torch.tensor(
            shifts_ref, dtype=torch.float64, device=torch.device("cpu")
        )

        # Calculate distances with cell shifts
        cell_shifts_ref = torch.mm(shifts_ref, cell)
        dds_ref = compute_distances_with_cell_shifts(pos, mapping_ref, cell_shifts_ref)

        # Sort the distances
        dds_ref = np.sort(dds_ref.numpy())
        dist_ref = np.sort(dist)

        # Check that the distances are the same with ase and torchsim logic
        np.testing.assert_allclose(dds_ref, dist_ref)

        # Check that the distances are the same with both methods
        np.testing.assert_allclose(dds, dds_ref)

        # Check that the distances are the same with ase direct neighbor list
        np.testing.assert_allclose(dds, dist_ref)


@pytest.mark.parametrize("cutoff", [1, 3, 5, 7])
def test_vesin_nl_ts(*, cutoff: float) -> None:
    """Check that vesin_nl gives the same NL as ASE by comparing
    the resulting sorted list of distances between neighbors.
    """
    structures = structure_set()

    for structure in structures:
        # Convert to torch tensors
        pos = torch.tensor(structure.positions, device=DEVICE, dtype=DTYPE)
        cell = torch.tensor(structure.cell.array, device=DEVICE, dtype=DTYPE)

        pbc = structure.pbc.any()

        # Get the neighbor list from vesin_nl_ts
        # Note: No self-interaction
        mapping, shifts = vesin_nl_ts(
            positions=pos,
            cell=cell,
            pbc=pbc,
            cutoff=torch.tensor(cutoff, dtype=DTYPE, device=DEVICE),
        )

        # Calculate distances with cell shifts
        cell_shifts = torch.mm(shifts, cell)
        dds = compute_distances_with_cell_shifts(pos, mapping, cell_shifts)
        dds = np.sort(dds.numpy())

        # Get the neighbor list from ase
        idx_i, idx_j, shifts_ref, dist = neighbor_list(
            quantities="ijSd",
            a=structure,
            cutoff=cutoff,
            self_interaction=False,
            max_nbins=1e6,
        )

        # Convert to torch tensors
        idx_i = torch.tensor(idx_i, dtype=torch.long, device=torch.device("cpu"))
        idx_j = torch.tensor(idx_j, dtype=torch.long, device=torch.device("cpu"))

        # Create mapping and shifts
        mapping_ref = torch.stack((idx_i, idx_j), dim=0)
        shifts_ref = torch.tensor(
            shifts_ref, dtype=torch.float64, device=torch.device("cpu")
        )

        # Calculate distances with cell shifts
        cell_shifts_ref = torch.mm(shifts_ref, cell)
        dds_ref = compute_distances_with_cell_shifts(pos, mapping_ref, cell_shifts_ref)

        # Sort the distances
        dds_ref = np.sort(dds_ref.numpy())
        dist_ref = np.sort(dist)

        # Check that the distances are the same with ase and torchsim logic
        np.testing.assert_allclose(dds_ref, dist_ref)

        # Check that the distances are the same with both methods
        np.testing.assert_allclose(dds, dds_ref)

        # Check that the distances are the same with ase direct neighbor list
        np.testing.assert_allclose(dds, dist_ref)


@pytest.mark.parametrize("cutoff", [1, 3, 5, 7])
def test_vesin_nl(*, cutoff: float) -> None:
    """Check that vesin_nl gives the same NL as ASE by comparing
    the resulting sorted list of distances between neighbors.
    """
    structures = structure_set()

    for structure in structures:
        # Convert to torch tensors
        pos = torch.tensor(structure.positions, device=DEVICE, dtype=DTYPE)
        cell = torch.tensor(structure.cell.array, device=DEVICE, dtype=DTYPE)

        pbc = structure.pbc.any()

        # Get the neighbor list from vesin_nl
        # Note: No self-interaction
        mapping, shifts = vesin_nl(
            positions=pos,
            cell=cell,
            pbc=pbc,
            cutoff=torch.tensor(cutoff, device=DEVICE, dtype=DTYPE),
        )

        # Calculate distances with cell shifts
        cell_shifts = torch.mm(shifts, cell)
        dds = compute_distances_with_cell_shifts(pos, mapping, cell_shifts)
        dds = np.sort(dds.numpy())

        # Get the neighbor list from ase
        idx_i, idx_j, shifts_ref, dist = neighbor_list(
            quantities="ijSd",
            a=structure,
            cutoff=cutoff,
            self_interaction=False,
            max_nbins=1e6,
        )

        # Convert to torch tensors
        idx_i = torch.tensor(idx_i, dtype=torch.long, device=torch.device("cpu"))
        idx_j = torch.tensor(idx_j, dtype=torch.long, device=torch.device("cpu"))

        # Create mapping and shifts
        mapping_ref = torch.stack((idx_i, idx_j), dim=0)
        shifts_ref = torch.tensor(
            shifts_ref, dtype=torch.float64, device=torch.device("cpu")
        )

        # Calculate distances with cell shifts
        cell_shifts_ref = torch.mm(shifts_ref, cell)
        dds_ref = compute_distances_with_cell_shifts(pos, mapping_ref, cell_shifts_ref)

        # Sort the distances
        dds_ref = np.sort(dds_ref.numpy())
        dist_ref = np.sort(dist)

        # Check that the distances are the same with ase and torchsim logic
        np.testing.assert_allclose(dds_ref, dist_ref)

        # Check that the distances are the same with both methods
        np.testing.assert_allclose(dds, dds_ref)

        # Check that the distances are the same with ase direct neighbor list
        np.testing.assert_allclose(dds, dist_ref)


@pytest.mark.parametrize("cutoff", [1, 3, 5, 7])
@pytest.mark.parametrize("self_interaction", [True, False])
def test_torch_nl_n2(*, cutoff: float, self_interaction: bool) -> None:
    """Check that torch_neighbor_list gives the same NL as ASE by comparing
    the resulting sorted list of distances between neighbors.
    """
    structures = structure_set()

    # Convert to torch batch (concatenate all tensors)
    pos, cell, pbc, batch, _ = ase_to_torch_batch(structures, device=DEVICE, dtype=DTYPE)

    # Get the neighbor list from torch_nl_n2
    mapping, mapping_batch, shifts_idx = torch_nl_n2(
        cutoff, pos, cell, pbc, batch, self_interaction
    )

    # Calculate distances with cell shifts (batch version)
    cell_shifts = compute_cell_shifts(cell, shifts_idx, mapping_batch)
    dds = compute_distances_with_cell_shifts(pos, mapping, cell_shifts)
    dds = np.sort(dds.numpy())

    # Get the neighbor list from ase
    dd_ref = []
    for structure in structures:
        idx_i, idx_j, idx_S, dist = neighbor_list(
            quantities="ijSd",
            a=structure,
            cutoff=cutoff,
            self_interaction=self_interaction,
            max_nbins=1e6,
        )
        dd_ref.extend(dist)
    dd_ref = np.sort(dd_ref)

    # Check that the distances are the same with ase and torchsim
    np.testing.assert_allclose(dd_ref, dds)


@pytest.mark.parametrize("cutoff", [1, 3, 5, 7])
@pytest.mark.parametrize("self_interaction", [True, False])
def test_torch_nl_linked_cell(*, cutoff: float, self_interaction: bool) -> None:
    """Check that torch_neighbor_list gives the same NL as ASE by comparing
    the resulting sorted list of distances between neighbors.
    """
    structures = structure_set()
    pos, cell, pbc, batch, _ = ase_to_torch_batch(structures, device=DEVICE, dtype=DTYPE)

    # Get the neighbor list from torch_nl_linked_cell
    mapping, mapping_batch, shifts_idx = torch_nl_linked_cell(
        cutoff, pos, cell, pbc, batch, self_interaction
    )

    # Calculate distances with cell shifts (batch version)
    cell_shifts = compute_cell_shifts(cell, shifts_idx, mapping_batch)
    dds = compute_distances_with_cell_shifts(pos, mapping, cell_shifts)
    dds = np.sort(dds.numpy())

    # Get the neighbor list from ase
    dd_ref = []
    for structure in structures:
        idx_i, idx_j, idx_S, dist = neighbor_list(
            quantities="ijSd",
            a=structure,
            cutoff=cutoff,
            self_interaction=self_interaction,
            max_nbins=1e6,
        )
        dd_ref.extend(dist)

    # Convert to torch tensors
    idx_S = torch.from_numpy(idx_S).to(torch.float64)

    missing_entries = []
    for idx_neigh in range(idx_i.shape[0]):
        mask = torch.logical_and(
            idx_i[idx_neigh] == mapping[0], idx_j[idx_neigh] == mapping[1]
        )

        if torch.any(torch.all(idx_S[idx_neigh] == shifts_idx[mask], dim=1)):
            pass
        else:
            missing_entries.append((idx_i[idx_neigh], idx_j[idx_neigh], idx_S[idx_neigh]))
            print(missing_entries[-1])
            print(
                compute_cell_shifts(
                    cell,
                    idx_S[idx_neigh].view((1, -1)),
                    torch.tensor([0], dtype=torch.long),
                )
            )

    dd_ref = np.sort(dd_ref)

    # Check that the distances are the same with ase and torchsim
    np.testing.assert_allclose(dd_ref, dds)
