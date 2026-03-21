import re
from collections.abc import Callable
from typing import Any

import numpy as np
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


def _make_triclinic_atoms() -> Atoms:
    """CaCrP2O7 (mvc-11955) triclinic structure.

    Adapted from https://github.com/felixmusil/torch_nl/blob/main/torch_nl/test_nl.py
    """
    return Atoms(
        positions=[
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
        cell=[
            [6.19330899, 0.0, 0.0],
            [2.4074486111396207, 6.149627748674982, 0.0],
            [0.2117993724186579, 1.0208820183960539, 7.305899571570074],
        ],
        numbers=[*[20] * 2, *[24] * 2, *[15] * 4, *[8] * 14],
        pbc=[True, True, True],
    )


def _make_very_skewed_atoms() -> Atoms:
    """Bi rhombohedral α=10° — extremely skewed, triggers nvalchemiops overflow."""
    atoms = bulk("Bi", "rhombohedral", a=6, alpha=10)
    atoms.info["very_skewed"] = True
    return atoms


@pytest.fixture
def periodic_atoms_unwrap_subset() -> list[Atoms]:
    """Fully periodic crystals used to test invariance under lattice translations."""
    return [
        bulk("Si", "diamond", a=6, cubic=True),
        bulk("Cu", "fcc", a=3.6),
        bulk("Si", "diamond", a=6),
        _make_triclinic_atoms(),
    ]


@pytest.fixture
def periodic_atoms_set():
    return [
        bulk("Si", "diamond", a=6, cubic=True),
        bulk("Si", "diamond", a=6),
        bulk("Cu", "fcc", a=3.6),
        bulk("Si", "bct", a=6, c=3),
        bulk("Ti", "hcp", a=2.94, c=4.64, orthorhombic=False),
        bulk("Bi", "rhombohedral", a=6, alpha=20),
        bulk("SiCu", "rocksalt", a=6),
        bulk("SiFCu", "fluorite", a=6),
        _make_triclinic_atoms(),
        _make_very_skewed_atoms(),
    ]


@pytest.fixture
def molecule_atoms_set() -> list:
    return [
        *map(molecule, ("CH3CH2NH2", "H2O", "methylenecyclopropane", "OCHCHO", "C3H9C")),
    ]


def _sorted_mic_distances(
    positions: torch.Tensor,
    row_vector_cell: torch.Tensor,
    mapping: torch.Tensor,
    mapping_system: torch.Tensor,
    shifts_idx: torch.Tensor,
) -> np.ndarray:
    cell_shifts = transforms.compute_cell_shifts(
        row_vector_cell, shifts_idx, mapping_system
    )
    d = transforms.compute_distances_with_cell_shifts(positions, mapping, cell_shifts)
    return np.sort(d.detach().cpu().numpy())


def _integer_lattice_shift_positions(
    positions: torch.Tensor,
    cell_batched: torch.Tensor,
    system_idx: torch.Tensor,
    integers: torch.Tensor,
) -> torch.Tensor:
    cell_per_atom = cell_batched[system_idx.to(cell_batched.device)]
    delta = (integers.to(cell_per_atom.dtype).unsqueeze(-1) * cell_per_atom).sum(dim=1)
    return positions + delta


def _all_nl_backends() -> list[Any]:
    """All NL backends as pytest.params with skipif marks for optional deps."""
    _skip_vesin = pytest.mark.skipif(
        not neighbors.VESIN_AVAILABLE, reason="Vesin is not installed"
    )
    _skip_vesin_ts = pytest.mark.skipif(
        not neighbors.VESIN_TORCH_AVAILABLE, reason="Vesin is not installed"
    )

    _skip_alchemiops = pytest.mark.skipif(
        not neighbors.ALCHEMIOPS_AVAILABLE, reason="nvalchemiops is not installed"
    )
    return [
        pytest.param(neighbors.torch_nl_n2, id="torch_nl_n2"),
        pytest.param(neighbors.torch_nl_linked_cell, id="torch_nl_linked_cell"),
        pytest.param(neighbors.vesin_nl, id="vesin_nl", marks=_skip_vesin),
        pytest.param(neighbors.vesin_nl_ts, id="vesin_nl_ts", marks=_skip_vesin_ts),
        pytest.param(
            neighbors.alchemiops_nl_n2,
            id="alchemiops_nl_n2",
            marks=_skip_alchemiops,
        ),
        pytest.param(
            neighbors.alchemiops_nl_cell_list,
            id="alchemiops_nl_cell_list",
            marks=_skip_alchemiops,
        ),
    ]


def _nl_backends_x_cutoffs(cutoffs: list[float] | None = None) -> list[Any]:
    """Cross-product of all NL backends x cutoffs, preserving skip marks."""
    if cutoffs is None:
        cutoffs = [1, 3, 5, 7]
    return [
        pytest.param(p.values[0], c, id=f"{p.values[0].__name__}-{c}", marks=p.marks)
        for p in _all_nl_backends()
        for c in cutoffs
    ]


@pytest.mark.parametrize("cutoff", [2.0, 5.0, 7.0])
@pytest.mark.parametrize("self_interaction", [True, False])
@pytest.mark.parametrize("shift_mode", ["uniform", "per_atom"])
@pytest.mark.parametrize("nl_implementation", _all_nl_backends())
def test_neighbor_list_invariant_under_lattice_image_shifts(
    *,
    cutoff: float,
    self_interaction: bool,
    shift_mode: str,
    nl_implementation: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    periodic_atoms_unwrap_subset: list[Atoms],
) -> None:
    """NL backends: same sorted MIC distances and pair count after lattice-image shifts.

    ``uniform``: one integer triplet per system applied to all its atoms (rigid image).
    ``per_atom``: independent integer triplet per atom (same structure mod PBC).

    Backends: ``torch_nl_n2``, ``torch_nl_linked_cell``, ``vesin_nl``, ``vesin_nl_ts``,
    ``alchemiops_nl_n2``, ``alchemiops_nl_cell_list`` (latter four skip if optional
    deps missing). See TorchSim/torch-sim#423, #437.
    """
    atoms_list = periodic_atoms_unwrap_subset
    pos_wrapped, cell_flat, pbc_flat, batch, _ = ase_to_torch_batch(
        atoms_list, device=DEVICE, dtype=DTYPE
    )
    n_sys = len(atoms_list)
    cell_b = cell_flat.view(n_sys, 3, 3)
    pbc_b = pbc_flat.view(n_sys, 3)
    pbc_on_atom = pbc_b[batch]
    if shift_mode == "uniform":
        triplets = torch.tensor(
            [[2, -1, 1], [-3, 0, 2], [1, 1, -2], [2, 2, -3]],
            dtype=torch.long,
            device=DEVICE,
        )
        per_system = triplets[torch.arange(n_sys, device=DEVICE) % triplets.shape[0]]
        ints = per_system[batch] * pbc_on_atom.long()
    elif shift_mode == "per_atom":
        n_atoms = pos_wrapped.shape[0]
        ar = torch.arange(n_atoms, device=DEVICE, dtype=torch.long)
        ints = torch.stack(
            [(ar % 3) - 1, (ar % 5) - 2, (ar % 4) - 2],
            dim=1,
        )
        ints = ints * pbc_on_atom.long()
    else:
        raise AssertionError(f"unknown shift_mode: {shift_mode}")
    pos_shifted = _integer_lattice_shift_positions(pos_wrapped, cell_b, batch, ints)
    assert not torch.allclose(pos_shifted, pos_wrapped, rtol=0.0, atol=1e-12), (
        "expected non-trivial lattice shifts along periodic axes"
    )
    c_tensor = torch.tensor(cutoff, dtype=DTYPE, device=DEVICE)
    map_w, sys_w, sh_w = nl_implementation(
        cutoff=c_tensor,
        positions=pos_wrapped,
        cell=cell_b,
        pbc=pbc_b,
        system_idx=batch,
        self_interaction=self_interaction,
    )
    map_s, sys_s, sh_s = nl_implementation(
        cutoff=c_tensor,
        positions=pos_shifted,
        cell=cell_b,
        pbc=pbc_b,
        system_idx=batch,
        self_interaction=self_interaction,
    )
    d_w = _sorted_mic_distances(pos_wrapped, cell_b, map_w, sys_w, sh_w)
    d_s = _sorted_mic_distances(pos_shifted, cell_b, map_s, sys_s, sh_s)
    np.testing.assert_allclose(d_w, d_s, rtol=1e-5, atol=1e-5)
    assert map_w.shape[1] == map_s.shape[1]
    assert torch.equal(batch[map_w[0]], batch[map_w[1]])
    assert torch.equal(batch[map_s[0]], batch[map_s[1]])


@pytest.mark.parametrize(
    ("nl_implementation", "cutoff"),
    _nl_backends_x_cutoffs(),
)
@pytest.mark.parametrize("self_interaction", [True, False])
def test_neighbor_list_implementations(
    *,
    nl_implementation: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cutoff: float,
    self_interaction: bool,
    molecule_atoms_set: list[Atoms],
    periodic_atoms_set: list[Atoms],
) -> None:
    """Check that neighbor list implementations give the same results as ASE.

    Tests all implementations in batched mode with mixed periodic and non-periodic
    systems, comparing sorted distances against ASE reference values.
    """
    atoms_list = molecule_atoms_set + periodic_atoms_set
    is_alchemiops = "alchemiops" in nl_implementation.__name__
    if is_alchemiops and cutoff >= 3:
        atoms_list = [a for a in atoms_list if not a.info.get("very_skewed")]

    # NOTE we can't use atoms_to_state here because we want to test mixed
    # periodic and non-periodic systems
    pos, row_vector_cell, pbc, batch, _ = ase_to_torch_batch(
        atoms_list, device=DEVICE, dtype=DTYPE
    )
    mapping, mapping_system, shifts_idx = nl_implementation(
        cutoff=torch.tensor(cutoff, dtype=DTYPE, device=DEVICE),
        positions=pos,
        cell=row_vector_cell,
        pbc=pbc,
        system_idx=batch,
        self_interaction=self_interaction,
    )

    cell_shifts = transforms.compute_cell_shifts(
        row_vector_cell, shifts_idx, mapping_system
    )
    dds = np.sort(
        transforms.compute_distances_with_cell_shifts(pos, mapping, cell_shifts).numpy()
    )

    # Build batched ASE reference with global atom indices
    ref_i, ref_j, ref_shifts, ref_sys = [], [], [], []
    offset = 0
    for sys_idx, atoms in enumerate(atoms_list):
        idx_i, idx_j, shifts_ref, _ = neighbor_list(
            quantities="ijSd",
            a=atoms,
            cutoff=cutoff,
            self_interaction=self_interaction,
            max_nbins=1e6,
        )
        ref_i.append(torch.tensor(idx_i, dtype=torch.long) + offset)
        ref_j.append(torch.tensor(idx_j, dtype=torch.long) + offset)
        ref_shifts.append(torch.tensor(shifts_ref, dtype=DTYPE))
        ref_sys.append(torch.full((len(idx_i),), sys_idx, dtype=torch.long))
        offset += len(atoms)

    mapping_ref = torch.stack([torch.cat(ref_i), torch.cat(ref_j)], dim=0).to(DEVICE)
    shifts_ref_t = torch.cat(ref_shifts).to(DEVICE)
    mapping_system_ref = torch.cat(ref_sys).to(DEVICE)

    cell_shifts_ref = transforms.compute_cell_shifts(
        row_vector_cell, shifts_ref_t, mapping_system_ref
    )
    dds_ref = np.sort(
        transforms.compute_distances_with_cell_shifts(
            pos, mapping_ref, cell_shifts_ref
        ).numpy()
    )

    # Compare distances and mapping counts
    np.testing.assert_allclose(dds_ref, dds)
    assert mapping.shape[1] == mapping_ref.shape[1], (
        f"Pair count mismatch: got {mapping.shape[1]}, expected {mapping_ref.shape[1]}"
    )
    # Ensure pair/system mapping stays consistent in batched mode.
    assert torch.equal(batch[mapping[0]], batch[mapping[1]])
    assert torch.equal(batch[mapping[0]], mapping_system)


@pytest.mark.parametrize("self_interaction", [True, False])
@pytest.mark.parametrize("pbc_val", [True, False])
@pytest.mark.parametrize("nl_implementation", _all_nl_backends())
def test_nl_pbc_edge_cases(
    *, pbc_val: bool, self_interaction: bool, nl_implementation: Callable[..., Any]
) -> None:
    """Test all NL implementations find neighbors for periodic and non-periodic
    systems with and without self-interaction.
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], device=DEVICE, dtype=DTYPE)
    cell = torch.eye(3, device=DEVICE, dtype=DTYPE) * 2.0
    cutoff = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    pbc = torch.tensor([pbc_val, pbc_val, pbc_val], device=DEVICE)
    system_idx = torch.zeros(2, dtype=torch.long, device=DEVICE)

    mapping, sys_map, _shifts = nl_implementation(
        positions=pos,
        cell=cell,
        pbc=pbc,
        cutoff=cutoff,
        system_idx=system_idx,
        self_interaction=self_interaction,
    )
    assert mapping.shape[1] > 0
    assert (sys_map == 0).all()


@pytest.mark.skipif(not neighbors.VESIN_AVAILABLE, reason="Vesin not available")
def test_vesin_nl_float32() -> None:
    """Test that vesin_nl (not vesin_nl_ts) accepts float32 inputs."""
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], device=DEVICE, dtype=torch.float32
    )
    cell = torch.eye(3, device=DEVICE, dtype=torch.float32) * 2.0
    cutoff = torch.tensor(1.5, device=DEVICE, dtype=torch.float32)
    pbc = torch.tensor([True, True, True], device=DEVICE)
    system_idx = torch.zeros(2, dtype=torch.long, device=DEVICE)

    mapping, _sys_map, _shifts = neighbors.vesin_nl(
        positions=pos, cell=cell, pbc=pbc, cutoff=cutoff, system_idx=system_idx
    )
    assert mapping.shape[1] > 0


def _minimal_neighbor_list_inputs(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create minimal valid tensor inputs for neighbor-list API smoke checks."""
    positions = torch.zeros((1, 3), dtype=torch.float32, device=device)
    cell = torch.eye(3, dtype=torch.float32, device=device)
    pbc = torch.tensor([False, False, False], dtype=torch.bool, device=device)
    cutoff = torch.tensor(1.0, dtype=torch.float32, device=device)
    system_idx = torch.zeros(1, dtype=torch.long, device=device)
    return positions, cell, pbc, cutoff, system_idx


def test_optional_neighbor_backends_expose_flags_and_entrypoints() -> None:
    """Public API: booleans and callables always present after import."""
    assert isinstance(neighbors.VESIN_AVAILABLE, bool)
    assert isinstance(neighbors.ALCHEMIOPS_AVAILABLE, bool)
    for name in (
        "vesin_nl",
        "vesin_nl_ts",
        "alchemiops_nl_n2",
        "alchemiops_nl_cell_list",
    ):
        assert callable(getattr(neighbors, name))


@pytest.mark.parametrize(
    ("fn_names", "message"),
    [
        (
            ("vesin_nl", "vesin_nl_ts"),
            "Vesin is not installed. Install it with: pip install vesin",
        ),
        (
            ("alchemiops_nl_n2", "alchemiops_nl_cell_list"),
            "nvalchemiops is not installed. Install it with: pip install nvalchemiops",
        ),
    ],
)
def test_neighbor_list_stub_import_errors_match_documentation(
    monkeypatch: pytest.MonkeyPatch,
    fn_names: tuple[str, ...],
    message: str,
) -> None:
    """Stubs must raise the same ImportError as optional-backend fallbacks."""

    def _stub(*args: object, **kwargs: object) -> None:  # noqa: ARG001
        raise ImportError(message)

    for fn_name in fn_names:
        monkeypatch.setattr(neighbors, fn_name, _stub)
    args = _minimal_neighbor_list_inputs(DEVICE)
    for fn_name in fn_names:
        with pytest.raises(ImportError, match=re.escape(message)):
            getattr(neighbors, fn_name)(*args)


def test_fallback_when_alchemiops_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that torch-sim works correctly without alchemiops (CI compatibility)."""
    # This test ensures CI works even if alchemiops fails to import
    # torchsim_nl should fall back to pure PyTorch implementations
    device = torch.device("cpu")
    dtype = torch.float32

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    cell = torch.eye(3, device=device, dtype=dtype) * 3.0
    pbc = torch.tensor([False, False, False], device=device)
    cutoff = torch.tensor(1.5, device=device, dtype=dtype)
    system_idx = torch.zeros(4, dtype=torch.long, device=device)

    # Use monkeypatch to temporarily disable alchemiops
    monkeypatch.setattr(neighbors, "ALCHEMIOPS_AVAILABLE", False)

    # torchsim_nl should always work (with fallback)
    mapping, sys_map, _shifts = neighbors.torchsim_nl(
        positions, cell, pbc, cutoff, system_idx
    )

    # Should find neighbors
    assert mapping.shape[0] == 2
    assert mapping.shape[1] > 0
    assert sys_map.shape[0] == mapping.shape[1]

    # default_batched_nl should always be available
    assert neighbors.default_batched_nl is not None
    mapping2, _sys_map2, _shifts2 = neighbors.default_batched_nl(
        positions, cell, pbc, cutoff, system_idx
    )
    assert mapping2.shape[1] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available for testing")
def test_torchsim_nl_gpu() -> None:
    """Test that torchsim_nl works on GPU (CUDA/ROCm)."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

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

    # Should work on GPU regardless of implementation availability
    mapping, sys_map, shifts = neighbors.torchsim_nl(
        positions, cell, pbc, cutoff, system_idx
    )

    assert mapping.device.type == "cuda"
    assert shifts.device.type == "cuda"
    assert sys_map.device.type == "cuda"
    assert mapping.shape[0] == 2  # (2, num_neighbors)

    # Cleanup
    torch.cuda.empty_cache()


def test_torchsim_nl_fallback_when_vesin_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that torchsim_nl falls back to torch_nl when alchemiops/vesin unavailable.

    This test simulates the case where alchemiops and vesin are not available by
    monkeypatching their availability flags to False. This ensures the fallback logic
    is tested even in environments where they are actually installed.
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

    # Monkeypatch both availability flags to False
    monkeypatch.setattr(neighbors, "VESIN_AVAILABLE", False)
    monkeypatch.setattr(neighbors, "ALCHEMIOPS_AVAILABLE", False)

    # Call torchsim_nl with mocked unavailable implementations
    mapping_torchsim, sys_map_ts, shifts_torchsim = neighbors.torchsim_nl(
        positions, cell, pbc, cutoff, system_idx
    )

    # Call torch_nl_linked_cell directly for comparison
    mapping_expected, sys_map_exp, shifts_expected = neighbors.torch_nl_linked_cell(
        positions, cell, pbc, cutoff, system_idx
    )

    # When both are unavailable, torchsim_nl should use torch_nl_linked_cell
    # and produce identical results
    torch.testing.assert_close(mapping_torchsim, mapping_expected)
    torch.testing.assert_close(shifts_torchsim, shifts_expected)
    torch.testing.assert_close(sys_map_ts, sys_map_exp)


def _no_neighbor_inputs() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Build a simple no-neighbor system."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],
        device=DEVICE,
        dtype=DTYPE,
    )
    cell = torch.eye(3, device=DEVICE, dtype=DTYPE) * 20.0
    pbc = torch.tensor([False, False, False], device=DEVICE)
    cutoff = torch.tensor(1.0, device=DEVICE, dtype=DTYPE)
    return positions, cell, pbc, cutoff


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
