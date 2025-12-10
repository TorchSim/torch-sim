import time
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import psutil
import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule
from ase.neighborlist import neighbor_list

from tests.conftest import DEVICE, DTYPE
from torch_sim import neighbors, transforms


def ase_to_torch_batch(
    atoms_list: list[Atoms], device: torch.device, dtype: torch.dtype = torch.float32
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
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing:
            - pos: Tensor of atomic positions.
            - cell: Tensor of unit cell vectors.
            - pbc: Tensor indicating periodic boundary conditions.
            - system_idx: Tensor indicating the system index for each atom.
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
    system_idx = torch.zeros(pos.shape[0], dtype=torch.long)
    for ii, (st, end) in enumerate(
        zip(stride[:-1], stride[1:], strict=True)  # noqa: RUF007
    ):
        system_idx[st:end] = ii
    n_atoms = torch.Tensor(n_atoms[1:]).to(dtype=torch.long)
    return (
        pos.to(dtype=dtype, device=device),
        cell.to(dtype=dtype, device=device),
        pbc.to(device=device),
        system_idx.to(device=device),
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
    *, cutoff: float, atoms_list: str, use_jit: bool, request: pytest.FixtureRequest
) -> None:
    """Check that primitive_neighbor_list gives the same NL as ASE by comparing
    the resulting sorted list of distances between neighbors.

    Args:
        cutoff: Cutoff distance for neighbor search
        atoms_list: List of atoms to test
        use_jit: Whether to use the jitted version or disable JIT
    """
    atoms_list = request.getfixturevalue(atoms_list)

    # Create a non-jitted version of the function if requested
    if use_jit:
        neighbor_list_fn = neighbors.primitive_neighbor_list
    else:
        # Create wrapper that disables JIT
        import os

        old_jit_setting = os.environ.get("PYTORCH_JIT")
        os.environ["PYTORCH_JIT"] = "0"

        # Import the function again to get the non-jitted version
        from importlib import reload

        import torch_sim as ts

        reload(ts.neighbors)
        neighbor_list_fn = ts.neighbors.primitive_neighbor_list

        # Restore JIT setting after test
        if old_jit_setting is not None:
            os.environ["PYTORCH_JIT"] = old_jit_setting
        else:
            os.environ.pop("PYTORCH_JIT", None)

    for atoms in atoms_list:
        # Convert to torch tensors
        pos = torch.tensor(atoms.positions, device=DEVICE, dtype=DTYPE)
        row_vector_cell = torch.tensor(atoms.cell.array, device=DEVICE, dtype=DTYPE)

        pbc = torch.tensor(atoms.pbc, device=DEVICE, dtype=DTYPE)

        # Get the neighbor list using the appropriate function (jitted or non-jitted)
        # Note: No self-interaction
        idx_i, idx_j, shifts_tensor = neighbor_list_fn(
            quantities="ijS",
            positions=pos,
            cell=row_vector_cell,
            pbc=pbc,
            cutoff=torch.tensor(cutoff, dtype=DTYPE, device=DEVICE),
            device=DEVICE,
            dtype=DTYPE,
            self_interaction=False,
            use_scaled_positions=False,
            max_n_bins=int(1e6),
        )

        # Create mapping
        mapping = torch.stack((idx_i, idx_j), dim=0)

        # Convert shifts_tensor to the same dtype as cell before matrix multiplication
        shifts_tensor = shifts_tensor.to(dtype=DTYPE)

        # Calculate distances with cell shifts
        cell_shifts_prim = torch.mm(shifts_tensor, row_vector_cell)
        dds_prim = transforms.compute_distances_with_cell_shifts(
            pos, mapping, cell_shifts_prim
        )
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
        idx_i_ref = torch.tensor(idx_i_ref, dtype=torch.long, device=DEVICE)
        idx_j_ref = torch.tensor(idx_j_ref, dtype=torch.long, device=DEVICE)

        # Create mapping and shifts
        mapping_ref = torch.stack((idx_i_ref, idx_j_ref), dim=0)
        shifts_ref = torch.tensor(shifts_ref, dtype=DTYPE, device=DEVICE)

        # Calculate distances with cell shifts
        cell_shifts_ref = torch.mm(shifts_ref, row_vector_cell)
        dds_ref = transforms.compute_distances_with_cell_shifts(
            pos, mapping_ref, cell_shifts_ref
        )

        # Sort the distances
        dds_ref = np.sort(dds_ref.numpy())
        dist_ref = np.sort(dist_ref)

        # Check that the distances are the same with ase and TorchSim logic
        np.testing.assert_allclose(dds_ref, dist_ref)

        # Check that the primitive_neighbor_list distances match ASE's
        np.testing.assert_allclose(
            dds_prim, dist_ref, err_msg=f"Failed with use_jit={use_jit}"
        )


@pytest.mark.parametrize("cutoff", [1, 3, 5, 7])
@pytest.mark.parametrize("atoms_list", ["periodic_atoms_set", "molecule_atoms_set"])
@pytest.mark.parametrize(
    "nl_implementation",
    [neighbors.standard_nl]
    + ([neighbors.vesin_nl, neighbors.vesin_nl_ts] if neighbors.VESIN_AVAILABLE else []),
)
def test_neighbor_list_implementations(
    *,
    cutoff: float,
    atoms_list: str,
    nl_implementation: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    request: pytest.FixtureRequest,
) -> None:
    """Check that different neighbor list implementations give the same results as ASE
    by comparing the resulting sorted list of distances between neighbors.
    """
    atoms_list = request.getfixturevalue(atoms_list)

    for atoms in atoms_list:
        # Convert to torch tensors
        pos = torch.tensor(atoms.positions, device=DEVICE, dtype=DTYPE)
        row_vector_cell = torch.tensor(atoms.cell.array, device=DEVICE, dtype=DTYPE)
        pbc = torch.tensor(atoms.pbc, device=DEVICE, dtype=DTYPE)

        # Create system_idx for single system (all atoms belong to system 0)
        system_idx = torch.zeros(len(pos), dtype=torch.long, device=DEVICE)

        # Get the neighbor list from the implementation being tested
        mapping, _sys_map, shifts = nl_implementation(
            positions=pos,
            cell=row_vector_cell,
            pbc=pbc,
            cutoff=torch.tensor(cutoff, dtype=DTYPE, device=DEVICE),
            system_idx=system_idx,
        )

        # Calculate distances with cell shifts
        # (shifts are now shift indices, same as shifts for single system)
        cell_shifts = torch.mm(shifts, row_vector_cell)
        dds = transforms.compute_distances_with_cell_shifts(pos, mapping, cell_shifts)
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
        idx_i = torch.tensor(idx_i, dtype=torch.long, device=DEVICE)
        idx_j = torch.tensor(idx_j, dtype=torch.long, device=DEVICE)
        mapping_ref = torch.stack((idx_i, idx_j), dim=0)
        shifts_ref = torch.tensor(shifts_ref, dtype=torch.float64, device=DEVICE)
        cell_shifts_ref = torch.mm(shifts_ref, row_vector_cell)
        dds_ref = transforms.compute_distances_with_cell_shifts(
            pos, mapping_ref, cell_shifts_ref
        )
        dds_ref = np.sort(dds_ref.numpy())
        dist_ref = np.sort(dist)

        # Verify results
        np.testing.assert_allclose(dds_ref, dist_ref)
        np.testing.assert_allclose(dds, dds_ref)
        np.testing.assert_allclose(dds, dist_ref)


@pytest.mark.parametrize("cutoff", [1, 3, 5, 7])
@pytest.mark.parametrize("self_interaction", [True, False])
@pytest.mark.parametrize(
    "nl_implementation",
    [
        neighbors.torch_nl_n2,
        neighbors.torch_nl_linked_cell,
        neighbors.standard_nl,
    ]
    + ([neighbors.vesin_nl, neighbors.vesin_nl_ts] if neighbors.VESIN_AVAILABLE else []),
)
def test_torch_nl_implementations(
    *,
    cutoff: float,
    self_interaction: bool,
    nl_implementation: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    molecule_atoms_set: list[Atoms],
    periodic_atoms_set: list[Atoms],
) -> None:
    """Check that batched neighbor list implementations give the same results as ASE.

    This tests the native batched implementations (torch_nl_n2, torch_nl_linked_cell)
    and the unified implementations (standard_nl, vesin_nl) in batched mode.
    """
    atoms_list = molecule_atoms_set + periodic_atoms_set

    # Convert to torch batch (concatenate all tensors)
    # NOTE we can't use atoms_to_state here because we want to test mixed
    # periodic and non-periodic systems
    pos, row_vector_cell, pbc, batch, _ = ase_to_torch_batch(
        atoms_list, device=DEVICE, dtype=DTYPE
    )

    # Get the neighbor list from the implementation being tested
    mapping, mapping_system, shifts_idx = nl_implementation(
        cutoff=torch.tensor(cutoff, dtype=DTYPE, device=DEVICE),
        positions=pos,
        cell=row_vector_cell,
        pbc=pbc,
        system_idx=batch,
        self_interaction=self_interaction,
    )

    # Calculate distances
    cell_shifts = transforms.compute_cell_shifts(
        row_vector_cell, shifts_idx, mapping_system
    )
    dds = transforms.compute_distances_with_cell_shifts(pos, mapping, cell_shifts)
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


def test_primitive_neighbor_list_edge_cases() -> None:
    """Test edge cases for primitive_neighbor_list."""
    # Test different PBC combinations
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], device=DEVICE, dtype=DTYPE)
    cell = torch.eye(3, device=DEVICE, dtype=DTYPE) * 2.0
    cutoff = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    # Test all PBC combinations
    for pbc in [(True, False, False), (False, True, False), (False, False, True)]:
        idx_i, idx_j, _shifts = neighbors.primitive_neighbor_list(
            quantities="ijS",
            positions=pos,
            cell=cell,
            pbc=torch.tensor(pbc, device=DEVICE, dtype=DTYPE),
            cutoff=cutoff,
            device=DEVICE,
            dtype=DTYPE,
        )
        assert len(idx_i) > 0  # Should find at least one neighbor

    # Test self-interaction
    idx_i, idx_j, _shifts = neighbors.primitive_neighbor_list(
        quantities="ijS",
        positions=pos,
        cell=cell,
        pbc=torch.Tensor([True, True, True]),
        cutoff=cutoff,
        device=DEVICE,
        dtype=DTYPE,
        self_interaction=True,
    )
    # Should find self-interactions
    assert torch.any(idx_i == idx_j)


def test_standard_nl_edge_cases() -> None:
    """Test edge cases for standard_nl."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], device=DEVICE, dtype=DTYPE)
    cell = torch.eye(3, device=DEVICE, dtype=DTYPE) * 2.0
    cutoff = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    system_idx = torch.zeros(2, dtype=torch.long, device=DEVICE)

    # Test different PBC combinations
    for pbc in (True, False):
        mapping, _sys_map, _shifts = neighbors.standard_nl(
            positions=pos,
            cell=cell,
            pbc=torch.tensor([pbc] * 3, device=DEVICE, dtype=DTYPE),
            cutoff=cutoff,
            system_idx=system_idx,
        )
        assert len(mapping[0]) > 0  # Should find neighbors


@pytest.mark.skipif(not neighbors.VESIN_AVAILABLE, reason="Vesin not available")
def test_vesin_nl_edge_cases() -> None:
    """Test edge cases for vesin_nl and vesin_nl_ts."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], device=DEVICE, dtype=DTYPE)
    cell = torch.eye(3, device=DEVICE, dtype=DTYPE) * 2.0
    cutoff = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    system_idx = torch.zeros(2, dtype=torch.long, device=DEVICE)

    # Test both implementations
    for nl_fn in (neighbors.vesin_nl, neighbors.vesin_nl_ts):
        # Test different PBC combinations
        for pbc in (
            torch.Tensor([True, True, True]),
            torch.Tensor([False, False, False]),
        ):
            mapping, _sys_map, _shifts = nl_fn(
                positions=pos, cell=cell, pbc=pbc, cutoff=cutoff, system_idx=system_idx
            )
            assert len(mapping[0]) > 0  # Should find neighbors

        # Test different precisions
        if nl_fn == neighbors.vesin_nl:  # vesin_nl_ts doesn't support float32
            pos_f32 = pos.to(dtype=torch.float32)
            cell_f32 = cell.to(dtype=torch.float32)
            system_idx_f32 = torch.zeros(2, dtype=torch.long, device=DEVICE)
            mapping, _sys_map, _shifts = nl_fn(
                positions=pos_f32,
                cell=cell_f32,
                pbc=torch.Tensor([True, True, True]),
                cutoff=cutoff,
                system_idx=system_idx_f32,
            )
            assert len(mapping[0]) > 0  # Should find neighbors


def test_torchsim_nl_availability() -> None:
    """Test that VESIN_AVAILABLE flag is correctly set."""
    assert isinstance(neighbors.VESIN_AVAILABLE, bool)
    if neighbors.VESIN_AVAILABLE:
        assert neighbors.VesinNeighborList is not None
        assert neighbors.VesinNeighborListTorch is not None
    else:
        assert neighbors.VesinNeighborList is None
        assert neighbors.VesinNeighborListTorch is None


def test_torchsim_nl_consistency() -> None:
    """Test that torchsim_nl produces consistent results."""
    device = torch.device("cpu")
    dtype = torch.float32

    # Simple 4-atom test system
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    cell = torch.eye(3, device=device, dtype=dtype) * 3.0
    pbc = torch.tensor([False, False, False], device=device)
    cutoff = torch.tensor(1.5, device=device, dtype=dtype)
    system_idx = torch.zeros(4, dtype=torch.long, device=device)

    # Test torchsim_nl against standard_nl
    mapping_torchsim, sys_map_ts, shifts_torchsim = neighbors.torchsim_nl(
        positions, cell, pbc, cutoff, system_idx
    )
    mapping_standard, sys_map_std, shifts_standard = neighbors.standard_nl(
        positions, cell, pbc, cutoff, system_idx
    )

    # torchsim_nl should always give consistent shape with standard_nl
    assert mapping_torchsim.shape == mapping_standard.shape
    assert shifts_torchsim.shape == shifts_standard.shape
    assert sys_map_ts.shape == sys_map_std.shape

    # When vesin is unavailable, torchsim_nl should match standard_nl exactly
    if not neighbors.VESIN_AVAILABLE:
        torch.testing.assert_close(mapping_torchsim, mapping_standard)
        torch.testing.assert_close(shifts_torchsim, shifts_standard)
        torch.testing.assert_close(sys_map_ts, sys_map_std)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available for testing")
def test_torchsim_nl_gpu() -> None:
    """Test that torchsim_nl works on GPU (CUDA/ROCm)."""
    device = torch.device("cuda")
    dtype = torch.float32

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    cell = torch.eye(3, device=device, dtype=dtype) * 3.0
    pbc = torch.tensor([True, True, True], device=device)
    cutoff = torch.tensor(1.5, device=device, dtype=dtype)
    system_idx = torch.zeros(2, dtype=torch.long, device=device)

    # Should work on GPU regardless of vesin availability
    mapping, sys_map, shifts = neighbors.torchsim_nl(
        positions, cell, pbc, cutoff, system_idx
    )

    assert mapping.device.type == "cuda"
    assert shifts.device.type == "cuda"
    assert sys_map.device.type == "cuda"
    assert mapping.shape[0] == 2  # (2, num_neighbors)


def test_torchsim_nl_fallback_when_vesin_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that torchsim_nl falls back to standard_nl when vesin is unavailable.

    This test simulates the case where vesin is not installed by monkeypatching
    VESIN_AVAILABLE to False. This ensures the fallback logic is tested even in
    CI environments where vesin is actually installed.
    """
    device = torch.device("cpu")
    dtype = torch.float32

    # Simple 4-atom test system
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    cell = torch.eye(3, device=device, dtype=dtype) * 3.0
    pbc = torch.tensor([False, False, False], device=device)
    cutoff = torch.tensor(1.5, device=device, dtype=dtype)
    system_idx = torch.zeros(4, dtype=torch.long, device=device)

    # Monkeypatch VESIN_AVAILABLE to False to simulate vesin not being installed
    monkeypatch.setattr(neighbors, "VESIN_AVAILABLE", False)

    # Call torchsim_nl with mocked unavailable vesin
    mapping_torchsim, sys_map_ts, shifts_torchsim = neighbors.torchsim_nl(
        positions, cell, pbc, cutoff, system_idx
    )

    # Call standard_nl directly for comparison
    mapping_standard, sys_map_std, shifts_standard = neighbors.standard_nl(
        positions, cell, pbc, cutoff, system_idx
    )

    # When VESIN_AVAILABLE is False, torchsim_nl should use standard_nl
    # and produce identical results
    torch.testing.assert_close(mapping_torchsim, mapping_standard)
    torch.testing.assert_close(shifts_torchsim, shifts_standard)
    torch.testing.assert_close(sys_map_ts, sys_map_std)


def test_strict_nl_edge_cases() -> None:
    """Test edge cases for strict_nl."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], device=DEVICE, dtype=DTYPE)
    # Create a cell tensor for each batch
    cell = torch.eye(3, device=DEVICE, dtype=torch.long).repeat(2, 1, 1) * 2

    # Test with no cell shifts
    mapping = torch.tensor([[0], [1]], device=DEVICE, dtype=torch.long)
    system_mapping = torch.tensor([0], device=DEVICE, dtype=torch.long)
    shifts_idx = torch.zeros((1, 3), device=DEVICE, dtype=torch.long)

    new_mapping, _new_batch, _new_shifts = neighbors.strict_nl(
        cutoff=1.5,
        positions=pos,
        cell=cell,
        mapping=mapping,
        system_mapping=system_mapping,
        shifts_idx=shifts_idx,
    )
    assert len(new_mapping[0]) > 0  # Should find neighbors

    # Test with different batch mappings
    mapping = torch.tensor([[0, 1], [1, 0]], device=DEVICE, dtype=torch.long)
    system_mapping = torch.tensor([0, 1], device=DEVICE, dtype=torch.long)
    shifts_idx = torch.zeros((2, 3), device=DEVICE, dtype=torch.long)

    new_mapping, _new_batch, _new_shifts = neighbors.strict_nl(
        cutoff=1.5,
        positions=pos,
        cell=cell,
        mapping=mapping,
        system_mapping=system_mapping,
        shifts_idx=shifts_idx,
    )
    assert len(new_mapping[0]) > 0  # Should find neighbors


def test_neighbor_lists_time_and_memory() -> None:
    """Test performance and memory characteristics of neighbor list implementations."""
    # Create a smaller system to reduce memory usage
    n_atoms = 100
    pos = torch.rand(n_atoms, 3, device=DEVICE, dtype=DTYPE)
    cell = torch.eye(3, device=DEVICE, dtype=DTYPE) * 10.0
    cutoff = torch.tensor(2.0, device=DEVICE, dtype=DTYPE)

    # Test different implementations
    nl_implementations = [
        neighbors.standard_nl,
        neighbors.torch_nl_n2,
        neighbors.torch_nl_linked_cell,
    ]
    if neighbors.VESIN_AVAILABLE:
        nl_implementations.extend(
            [
                neighbors.vesin_nl_ts,
                cast("Callable[..., Any]", neighbors.vesin_nl),
            ]
        )

    for nl_fn in nl_implementations:
        # Get initial memory usage
        process = psutil.Process()
        initial_cpu_memory = process.memory_info().rss  # in bytes

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            initial_gpu_memory = torch.cuda.memory_allocated()

        # Time the execution
        start_time = time.perf_counter()

        # All neighbor list functions now use the unified API with system_idx
        system_idx = torch.zeros(n_atoms, dtype=torch.long, device=DEVICE)
        # Fix pbc tensor shape
        pbc = torch.tensor([[True, True, True]], device=DEVICE)
        _mapping, _mapping_system, _shifts_idx = nl_fn(
            positions=pos,
            cell=cell,
            pbc=pbc,
            cutoff=cutoff,
            system_idx=system_idx,
            self_interaction=False,
        )

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Get final memory usage
        final_cpu_memory = process.memory_info().rss
        cpu_memory_used = final_cpu_memory - initial_cpu_memory
        fn_name = str(nl_fn)

        # Warning: cuda case was never tested, to be tweaked later
        if DEVICE.type == "cuda":
            final_gpu_memory = torch.cuda.memory_allocated()
            gpu_memory_used = final_gpu_memory - initial_gpu_memory
            assert execution_time < 0.01, f"{fn_name} took too long: {execution_time}s"
            assert gpu_memory_used < 5e8, (
                f"{fn_name} used too much GPU memory: {gpu_memory_used / 1e6:.2f}MB"
            )
            torch.cuda.empty_cache()
        else:
            assert cpu_memory_used < 5e8, (
                f"{fn_name} used too much CPU memory: {cpu_memory_used / 1e6:.2f}MB"
            )
            if nl_fn == neighbors.standard_nl:
                # this function is just quite slow. So we have a higher tolerance.
                # I tried removing "@jit.script" and it was still slow.
                # (This nl function is just slow)
                assert execution_time < 3, f"{fn_name} took too long: {execution_time}s"
            else:
                assert execution_time < 0.8, f"{fn_name} took too long: {execution_time}s"
