import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule
from ase.neighborlist import neighbor_list

from torch_sim.neighbors import (
    primitive_neighbor_list,
    standard_nl,
    strict_nl,
    torch_nl_linked_cell,
    torch_nl_n2,
    vesin_nl,
    vesin_nl_ts,
)
from torch_sim.transforms import compute_cell_shifts, compute_distances_with_cell_shifts


@pytest.fixture
def dtype() -> torch.dtype:
    return torch.float64


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
    # NOTE we leave the cell in the row vector convention rather than converting
    # to the column vector convention because we want to test the row vector
    # convention in the neighbor list functions.
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


@pytest.fixture
def periodic_atoms_set():
    return [
        bulk("Si", "diamond", a=6, cubic=True),
        bulk("Si", "diamond", a=6),
        bulk("Cu", "fcc", a=3.6),
        bulk("Si", "bct", a=6, c=3),
        bulk("Ti", "hcp", a=2.94, c=4.64, orthorhombic=False),
        # test very skewed rhombohedral cells
        bulk("Bi", "rhombohedral", a=6, alpha=20),
        bulk(
            "Bi", "rhombohedral", a=6, alpha=10
        ),  # very skewed, by far the slowest test case
        bulk("SiCu", "rocksalt", a=6),
        bulk("SiFCu", "fluorite", a=6),
        Atoms(**CaCrP2O7_mvc_11955_symmetrized),
    ]


@pytest.fixture
def molecule_atoms_set() -> list:
    return [
        *map(molecule, ("CH3CH2NH2", "H2O", "methylenecyclopropane", "OCHCHO", "C3H9C")),
    ]


@pytest.mark.parametrize("cutoff", [1, 3, 5, 7])
@pytest.mark.parametrize("use_jit", [True, False])
@pytest.mark.parametrize("atoms_list", ["periodic_atoms_set", "molecule_atoms_set"])
def test_primitive_neighbor_list(
    *,
    cutoff: float,
    atoms_list: str,
    device: torch.device,
    dtype: torch.dtype,
    use_jit: bool,
    request: pytest.FixtureRequest,
) -> None:
    """Check that primitive_neighbor_list gives the same NL as ASE by comparing
    the resulting sorted list of distances between neighbors.

    Args:
        cutoff: Cutoff distance for neighbor search
        device: Torch device to use
        dtype: Torch dtype to use
        use_jit: Whether to use the jitted version or disable JIT
    """
    atoms_list = request.getfixturevalue(atoms_list)

    # Create a non-jitted version of the function if requested
    if use_jit:
        neighbor_list_fn = primitive_neighbor_list
    else:
        # Create wrapper that disables JIT
        import os

        old_jit_setting = os.environ.get("PYTORCH_JIT")
        os.environ["PYTORCH_JIT"] = "0"

        # Import the function again to get the non-jitted version
        from importlib import reload

        import torch_sim.neighbors

        reload(torch_sim.neighbors)
        neighbor_list_fn = torch_sim.neighbors.primitive_neighbor_list

        # Restore JIT setting after test
        if old_jit_setting is not None:
            os.environ["PYTORCH_JIT"] = old_jit_setting
        else:
            os.environ.pop("PYTORCH_JIT", None)

    for atoms in atoms_list:
        # Convert to torch tensors
        pos = torch.tensor(atoms.positions, device=device, dtype=dtype)
        row_vector_cell = torch.tensor(atoms.cell.array, device=device, dtype=dtype)

        pbc = atoms.pbc.any()

        # Get the neighbor list using the appropriate function (jitted or non-jitted)
        # Note: No self-interaction
        idx_i, idx_j, shifts_tensor = neighbor_list_fn(
            quantities="ijS",
            positions=pos,
            cell=row_vector_cell,
            pbc=(pbc, pbc, pbc),
            cutoff=torch.tensor(cutoff, dtype=dtype, device=device),
            device=device,
            dtype=dtype,
            self_interaction=False,
            use_scaled_positions=False,
            max_n_bins=int(1e6),
        )

        # Create mapping
        mapping = torch.stack((idx_i, idx_j), dim=0)

        # Convert shifts_tensor to the same dtype as cell before matrix multiplication
        shifts_tensor = shifts_tensor.to(dtype=dtype)

        # Calculate distances with cell shifts
        cell_shifts_prim = torch.mm(shifts_tensor, row_vector_cell)
        dds_prim = compute_distances_with_cell_shifts(pos, mapping, cell_shifts_prim)
        dds_prim = np.sort(dds_prim.numpy())

        # Get the neighbor list from ase
        idx_i_ref, idx_j_ref, shifts_ref, dist_ref = neighbor_list(
            quantities="ijSd",
            a=atoms,
            cutoff=cutoff,
            self_interaction=False,
            max_nbins=1e6,
        )

        # Convert to torch tensors
        idx_i_ref = torch.tensor(idx_i_ref, dtype=torch.long, device=device)
        idx_j_ref = torch.tensor(idx_j_ref, dtype=torch.long, device=device)

        # Create mapping and shifts
        mapping_ref = torch.stack((idx_i_ref, idx_j_ref), dim=0)
        shifts_ref = torch.tensor(shifts_ref, dtype=dtype, device=device)

        # Calculate distances with cell shifts
        cell_shifts_ref = torch.mm(shifts_ref, row_vector_cell)
        dds_ref = compute_distances_with_cell_shifts(pos, mapping_ref, cell_shifts_ref)

        # Sort the distances
        dds_ref = np.sort(dds_ref.numpy())
        dist_ref = np.sort(dist_ref)

        # Check that the distances are the same with ase and torchsim logic
        np.testing.assert_allclose(dds_ref, dist_ref)

        # Check that the primitive_neighbor_list distances match ASE's
        np.testing.assert_allclose(
            dds_prim, dist_ref, err_msg=f"Failed with use_jit={use_jit}"
        )


@pytest.mark.parametrize("cutoff", [1, 3, 5, 7])
@pytest.mark.parametrize("atoms_list", ["periodic_atoms_set", "molecule_atoms_set"])
@pytest.mark.parametrize("nl_implementation", [standard_nl, vesin_nl, vesin_nl_ts])
def test_neighbor_list_implementations(
    *,
    cutoff: float,
    atoms_list: str,
    nl_implementation: callable,
    device: torch.device,
    dtype: torch.dtype,
    request: pytest.FixtureRequest,
) -> None:
    """Check that different neighbor list implementations give the same results as ASE
    by comparing the resulting sorted list of distances between neighbors.
    """
    atoms_list = request.getfixturevalue(atoms_list)

    for atoms in atoms_list:
        # Convert to torch tensors
        pos = torch.tensor(atoms.positions, device=device, dtype=dtype)
        row_vector_cell = torch.tensor(atoms.cell.array, device=device, dtype=dtype)
        pbc = atoms.pbc.any()

        # Get the neighbor list from the implementation being tested
        mapping, shifts = nl_implementation(
            positions=pos,
            cell=row_vector_cell,
            pbc=pbc,
            cutoff=torch.tensor(cutoff, dtype=dtype, device=device),
        )

        # Calculate distances with cell shifts
        cell_shifts = torch.mm(shifts, row_vector_cell)
        dds = compute_distances_with_cell_shifts(pos, mapping, cell_shifts)
        dds = np.sort(dds.numpy())

        # Get the reference neighbor list from ASE
        idx_i, idx_j, shifts_ref, dist = neighbor_list(
            quantities="ijSd",
            a=atoms,
            cutoff=cutoff,
            self_interaction=False,
            max_nbins=1e6,
        )

        # Convert to torch tensors and calculate reference distances
        idx_i = torch.tensor(idx_i, dtype=torch.long, device=torch.device("cpu"))
        idx_j = torch.tensor(idx_j, dtype=torch.long, device=torch.device("cpu"))
        mapping_ref = torch.stack((idx_i, idx_j), dim=0)
        shifts_ref = torch.tensor(
            shifts_ref, dtype=torch.float64, device=torch.device("cpu")
        )
        cell_shifts_ref = torch.mm(shifts_ref, row_vector_cell)
        dds_ref = compute_distances_with_cell_shifts(pos, mapping_ref, cell_shifts_ref)
        dds_ref = np.sort(dds_ref.numpy())
        dist_ref = np.sort(dist)

        # Verify results
        np.testing.assert_allclose(dds_ref, dist_ref)
        np.testing.assert_allclose(dds, dds_ref)
        np.testing.assert_allclose(dds, dist_ref)


@pytest.mark.parametrize("cutoff", [1, 3, 5, 7])
@pytest.mark.parametrize("self_interaction", [True, False])
@pytest.mark.parametrize("nl_implementation", [torch_nl_n2, torch_nl_linked_cell])
def test_torch_nl_implementations(
    *,
    cutoff: float,
    self_interaction: bool,
    nl_implementation: callable,
    device: torch.device,
    dtype: torch.dtype,
    molecule_atoms_set: list[Atoms],
    periodic_atoms_set: list[Atoms],
) -> None:
    """Check that torch neighbor list implementations give the same results as ASE."""
    atoms_list = molecule_atoms_set + periodic_atoms_set

    # Convert to torch batch (concatenate all tensors)
    # NOTE we can't use atoms_to_state here because we want to test mixed
    # periodic and non-periodic systems
    pos, row_vector_cell, pbc, batch, _ = ase_to_torch_batch(
        atoms_list, device=device, dtype=dtype
    )

    # Get the neighbor list from the implementation being tested
    mapping, mapping_batch, shifts_idx = nl_implementation(
        cutoff, pos, row_vector_cell, pbc, batch, self_interaction
    )

    # Calculate distances
    cell_shifts = compute_cell_shifts(row_vector_cell, shifts_idx, mapping_batch)
    dds = compute_distances_with_cell_shifts(pos, mapping, cell_shifts)
    dds = np.sort(dds.numpy())

    # Get reference results from ASE
    dd_ref = []
    for atoms in atoms_list:
        _, _, _, dist = neighbor_list(
            quantities="ijSd",
            a=atoms,
            cutoff=cutoff,
            self_interaction=self_interaction,
            max_nbins=1e6,
        )
        dd_ref.extend(dist)
    dd_ref = np.sort(dd_ref)

    # Verify results
    np.testing.assert_allclose(dd_ref, dds)


def test_primitive_neighbor_list_edge_cases(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Test edge cases for primitive_neighbor_list."""
    # Test different PBC combinations
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype) * 2.0
    cutoff = torch.tensor(1.5, device=device, dtype=dtype)

    # Test all PBC combinations
    for pbc in [(True, False, False), (False, True, False), (False, False, True)]:
        idx_i, idx_j, shifts = primitive_neighbor_list(
            quantities="ijS",
            positions=pos,
            cell=cell,
            pbc=pbc,
            cutoff=cutoff,
            device=device,
            dtype=dtype,
        )
        assert len(idx_i) > 0  # Should find at least one neighbor

    # Test self-interaction
    idx_i, idx_j, shifts = primitive_neighbor_list(
        quantities="ijS",
        positions=pos,
        cell=cell,
        pbc=(True, True, True),
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        self_interaction=True,
    )
    # Should find self-interactions
    assert torch.any(idx_i == idx_j)


def test_standard_nl_edge_cases(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Test edge cases for standard_nl."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype) * 2.0
    cutoff = torch.tensor(1.5, device=device, dtype=dtype)

    # Test different PBC combinations
    for pbc in (True, False):
        mapping, shifts = standard_nl(
            positions=pos,
            cell=cell,
            pbc=pbc,
            cutoff=cutoff,
        )
        assert len(mapping[0]) > 0  # Should find neighbors

    # Test sort_id
    mapping, shifts = standard_nl(
        positions=pos,
        cell=cell,
        pbc=True,
        cutoff=cutoff,
        sort_id=True,
    )
    # Check if indices are sorted
    assert torch.all(mapping[0][1:] >= mapping[0][:-1])


def test_vesin_nl_edge_cases(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Test edge cases for vesin_nl and vesin_nl_ts."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype) * 2.0
    cutoff = torch.tensor(1.5, device=device, dtype=dtype)

    # Test both implementations
    for nl_fn in (vesin_nl, vesin_nl_ts):
        # Test different PBC combinations
        for pbc in (True, False):
            mapping, shifts = nl_fn(
                positions=pos,
                cell=cell,
                pbc=pbc,
                cutoff=cutoff,
            )
            assert len(mapping[0]) > 0  # Should find neighbors

        # Test sort_id
        mapping, shifts = nl_fn(
            positions=pos,
            cell=cell,
            pbc=True,
            cutoff=cutoff,
            sort_id=True,
        )
        # Check if indices are sorted
        assert torch.all(mapping[0][1:] >= mapping[0][:-1])

        # Test different precisions
        if nl_fn == vesin_nl:  # vesin_nl_ts doesn't support float32
            pos_f32 = pos.to(dtype=torch.float32)
            cell_f32 = cell.to(dtype=torch.float32)
            cutoff_f32 = cutoff.to(dtype=torch.float32)
            mapping, shifts = nl_fn(
                positions=pos_f32,
                cell=cell_f32,
                pbc=True,
                cutoff=cutoff_f32,
            )
            assert len(mapping[0]) > 0  # Should find neighbors


def test_strict_nl_edge_cases(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Test edge cases for strict_nl."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], device=device, dtype=dtype)
    # Create a cell tensor for each batch
    cell = torch.eye(3, device=device, dtype=torch.long).repeat(2, 1, 1) * 2

    # Test with no cell shifts
    mapping = torch.tensor([[0], [1]], device=device, dtype=torch.long)
    batch_mapping = torch.tensor([0], device=device, dtype=torch.long)
    shifts_idx = torch.zeros((1, 3), device=device, dtype=torch.long)

    new_mapping, new_batch, new_shifts = strict_nl(
        cutoff=1.5,
        positions=pos,
        cell=cell,
        mapping=mapping,
        batch_mapping=batch_mapping,
        shifts_idx=shifts_idx,
    )
    assert len(new_mapping[0]) > 0  # Should find neighbors

    # Test with different batch mappings
    mapping = torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.long)
    batch_mapping = torch.tensor([0, 1], device=device, dtype=torch.long)
    shifts_idx = torch.zeros((2, 3), device=device, dtype=torch.long)

    new_mapping, new_batch, new_shifts = strict_nl(
        cutoff=1.5,
        positions=pos,
        cell=cell,
        mapping=mapping,
        batch_mapping=batch_mapping,
        shifts_idx=shifts_idx,
    )
    assert len(new_mapping[0]) > 0  # Should find neighbors
