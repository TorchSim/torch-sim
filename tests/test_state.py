import typing
from dataclasses import asdict

import pytest
import torch

import torch_sim as ts
from tests.conftest import DEVICE
from torch_sim.integrators import MDState
from torch_sim.state import (
    DeformGradMixin,
    SimState,
    _normalize_system_indices,
    _pop_states,
    _slice_state,
    coerce_prng,
    get_attrs_for_scope,
)


if typing.TYPE_CHECKING:
    from ase import Atoms
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure


def test_get_attrs_for_scope(si_sim_state: SimState) -> None:
    """Test getting attributes for a scope."""
    per_atom_attrs = dict(get_attrs_for_scope(si_sim_state, "per-atom"))
    assert set(per_atom_attrs) == {"positions", "masses", "atomic_numbers", "system_idx"}
    per_system_attrs = dict(get_attrs_for_scope(si_sim_state, "per-system"))
    assert set(per_system_attrs) == {"cell", "charge", "spin"}
    global_attrs = dict(get_attrs_for_scope(si_sim_state, "global"))
    assert set(global_attrs) == {"pbc", "_rng"}


def test_all_attributes_must_be_specified_in_scopes() -> None:
    """Test that an error is raised when we forget to specify the scope
    for an attribute in a child SimState class."""
    with pytest.raises(TypeError) as exc_info:

        class ChildState(SimState):
            attribute_specified_in_scopes: bool
            attribute_not_specified_in_scopes: bool

            _atom_attributes = (
                SimState._atom_attributes | {"attribute_specified_in_scopes"}  # noqa: SLF001
            )

    assert "attribute_not_specified_in_scopes" in str(exc_info.value)
    assert "attribute_specified_in_scopes" not in str(exc_info.value)


def test_no_duplicate_attributes_in_scopes() -> None:
    """Test that no attributes are specified in multiple scopes."""

    # Capture the exception information using "as exc_info"
    with pytest.raises(TypeError) as exc_info:

        class ChildState(SimState):
            duplicated_attribute: bool

            _system_attributes = SimState._system_attributes | {"duplicated_attribute"}  # noqa: SLF001
            _global_attributes = SimState._global_attributes | {"duplicated_attribute"}  # noqa: SLF001

    assert "are declared multiple times" in str(exc_info.value)
    assert "duplicated_attribute" in str(exc_info.value)


def test_slice_substate(si_double_sim_state: SimState, si_sim_state: SimState) -> None:
    """Test slicing a substate from the SimState."""
    for system_index in range(2):
        substate = _slice_state(si_double_sim_state, [system_index])
        assert isinstance(substate, SimState)
        assert substate.positions.shape == (8, 3)
        assert substate.masses.shape == (8,)
        assert substate.cell.shape == (1, 3, 3)
        assert torch.allclose(substate.positions, si_sim_state.positions)
        assert torch.allclose(substate.masses, si_sim_state.masses)
        assert torch.allclose(substate.cell, si_sim_state.cell)
        assert torch.allclose(substate.atomic_numbers, si_sim_state.atomic_numbers)
        assert torch.allclose(substate.system_idx, torch.zeros_like(substate.system_idx))


def test_slice_md_substate(si_double_sim_state: SimState) -> None:
    state = MDState(
        **asdict(si_double_sim_state),
        momenta=torch.randn_like(si_double_sim_state.positions),
        energy=torch.zeros((2,), device=si_double_sim_state.device),
        forces=torch.randn_like(si_double_sim_state.positions),
    )
    for system_index in range(2):
        substate = _slice_state(state, [system_index])
        assert isinstance(substate, MDState)
        assert substate.positions.shape == (8, 3)
        assert substate.masses.shape == (8,)
        assert substate.cell.shape == (1, 3, 3)
        assert substate.momenta.shape == (8, 3)
        assert substate.forces.shape == (8, 3)
        assert substate.energy.shape == (1,)


def test_slice_state_permuted_order(
    si_sim_state: SimState,
    ar_supercell_sim_state: SimState,
    fe_supercell_sim_state: SimState,
) -> None:
    """Slice with [2, 0, 1] should yield [Fe, Si, Ar]."""
    concatenated = ts.concatenate_states(
        [si_sim_state, ar_supercell_sim_state, fe_supercell_sim_state]
    )
    result = _slice_state(concatenated, [2, 0, 1])

    assert torch.allclose(result[0].positions, fe_supercell_sim_state.positions)
    assert torch.allclose(result[1].positions, si_sim_state.positions)
    assert torch.allclose(result[2].positions, ar_supercell_sim_state.positions)
    assert torch.allclose(result.cell[0], fe_supercell_sim_state.cell[0])
    assert torch.allclose(result.cell[1], si_sim_state.cell[0])
    assert torch.allclose(result.cell[2], ar_supercell_sim_state.cell[0])


def test_slice_state_reversed_subset(
    si_sim_state: SimState,
    ar_supercell_sim_state: SimState,
    fe_supercell_sim_state: SimState,
) -> None:
    """Slice with [2, 0] should match concatenate([Fe, Si])."""
    concatenated = ts.concatenate_states(
        [si_sim_state, ar_supercell_sim_state, fe_supercell_sim_state]
    )
    result = _slice_state(concatenated, [2, 0])
    expected = ts.concatenate_states([fe_supercell_sim_state, si_sim_state])

    assert torch.allclose(result.positions, expected.positions)
    assert torch.allclose(result.cell, expected.cell)
    assert torch.allclose(result.atomic_numbers, expected.atomic_numbers)
    assert torch.allclose(result.masses, expected.masses)
    assert result.n_systems == expected.n_systems


def test_concatenate_two_si_states(
    si_sim_state: SimState, si_double_sim_state: SimState
) -> None:
    """Test concatenating two identical silicon states."""
    # Concatenate two copies of the sim state
    concatenated = ts.concatenate_states([si_sim_state, si_sim_state])

    # Check that the result is the same as the double state
    assert isinstance(concatenated, SimState)
    assert concatenated.positions.shape == si_double_sim_state.positions.shape
    assert concatenated.masses.shape == si_double_sim_state.masses.shape
    assert concatenated.cell.shape == si_double_sim_state.cell.shape
    assert concatenated.atomic_numbers.shape == si_double_sim_state.atomic_numbers.shape
    assert concatenated.system_idx.shape == si_double_sim_state.system_idx.shape

    # Check system indices
    tensor_args = dict(dtype=torch.int64, device=si_sim_state.device)
    expected_system_indices = torch.cat(
        [
            torch.zeros(si_sim_state.n_atoms, **tensor_args),
            torch.ones(si_sim_state.n_atoms, **tensor_args),
        ]
    )
    assert torch.all(concatenated.system_idx == expected_system_indices)

    # Check that positions match (accounting for system indices)
    for sys_idx in range(2):
        mask_concat = concatenated.system_idx == sys_idx
        mask_double = si_double_sim_state.system_idx == sys_idx
        assert torch.allclose(
            concatenated.positions[mask_concat],
            si_double_sim_state.positions[mask_double],
        )


def test_concatenate_si_and_fe_states(
    si_sim_state: SimState, fe_supercell_sim_state: SimState
) -> None:
    """Test concatenating silicon and argon states."""
    # Concatenate silicon and argon states
    concatenated = ts.concatenate_states([si_sim_state, fe_supercell_sim_state])

    # Check basic properties
    assert isinstance(concatenated, SimState)
    assert (
        concatenated.positions.shape[0]
        == si_sim_state.positions.shape[0] + fe_supercell_sim_state.positions.shape[0]
    )
    assert (
        concatenated.masses.shape[0]
        == si_sim_state.masses.shape[0] + fe_supercell_sim_state.masses.shape[0]
    )
    assert concatenated.cell.shape[0] == 2  # One cell per system

    # Check system indices
    si_atoms = si_sim_state.n_atoms
    fe_atoms = fe_supercell_sim_state.n_atoms
    expected_system_indices = torch.cat(
        [
            torch.zeros(si_atoms, dtype=torch.int64, device=si_sim_state.device),
            torch.ones(fe_atoms, dtype=torch.int64, device=fe_supercell_sim_state.device),
        ]
    )
    assert torch.all(concatenated.system_idx == expected_system_indices)

    # check n_atoms_per_system
    assert torch.all(
        concatenated.n_atoms_per_system
        == torch.tensor(
            [si_sim_state.n_atoms, fe_supercell_sim_state.n_atoms],
            device=concatenated.device,
        )
    )

    # Check that positions match for each original state
    assert torch.allclose(concatenated.positions[:si_atoms], si_sim_state.positions)
    assert torch.allclose(
        concatenated.positions[si_atoms:], fe_supercell_sim_state.positions
    )

    # Check that atomic numbers are correct
    assert torch.all(concatenated.atomic_numbers[:si_atoms] == 14)  # Si
    assert torch.all(concatenated.atomic_numbers[si_atoms:] == 26)  # Fe


def test_concatenate_double_si_and_fe_states(
    si_double_sim_state: SimState, fe_supercell_sim_state: SimState
) -> None:
    """Test concatenating a double silicon state and an argon state."""
    # Concatenate double silicon and argon states
    concatenated = ts.concatenate_states([si_double_sim_state, fe_supercell_sim_state])

    # Check basic properties
    assert isinstance(concatenated, SimState)
    assert (
        concatenated.positions.shape[0]
        == si_double_sim_state.positions.shape[0]
        + fe_supercell_sim_state.positions.shape[0]
    )
    assert (
        concatenated.cell.shape[0] == 3
    )  # One cell for each original system (2 Si + 1 Ar)

    # Check system indices
    fe_atoms = fe_supercell_sim_state.n_atoms

    # The double Si state already has systems 0 and 1, so Ar should be system 2
    expected_system_indices = torch.cat(
        [
            si_double_sim_state.system_idx,
            torch.full(
                (fe_atoms,), 2, dtype=torch.int64, device=fe_supercell_sim_state.device
            ),
        ]
    )
    assert torch.all(concatenated.system_idx == expected_system_indices)
    assert torch.unique(concatenated.system_idx).shape[0] == 3

    # Check that we can slice back to the original states
    si_slice_0 = concatenated[0]
    si_slice_1 = concatenated[1]
    fe_slice = concatenated[2]

    # Check that the slices match the original states
    assert torch.allclose(si_slice_0.positions, si_double_sim_state[0].positions)
    assert torch.allclose(si_slice_1.positions, si_double_sim_state[1].positions)
    assert torch.allclose(fe_slice.positions, fe_supercell_sim_state.positions)


def test_split_state(si_double_sim_state: SimState) -> None:
    """Test splitting a state into a list of states."""
    states = si_double_sim_state.split()
    assert len(states) == si_double_sim_state.n_systems
    for state in states:
        assert isinstance(state, SimState)
        assert state.positions.shape == (8, 3)
        assert state.masses.shape == (8,)
        assert state.cell.shape == (1, 3, 3)
        assert state.atomic_numbers.shape == (8,)
        assert torch.allclose(state.system_idx, torch.zeros_like(state.system_idx))


def test_split_many_states(
    si_sim_state: SimState,
    ar_supercell_sim_state: SimState,
    fe_supercell_sim_state: SimState,
) -> None:
    """Test splitting a state into a list of states."""
    states = [si_sim_state, ar_supercell_sim_state, fe_supercell_sim_state]
    concatenated = ts.concatenate_states(states)
    split_states = concatenated.split()
    for state, sub_state in zip(states, split_states, strict=True):
        assert isinstance(sub_state, SimState)
        assert torch.allclose(sub_state.positions, state.positions)
        assert torch.allclose(sub_state.masses, state.masses)
        assert torch.allclose(sub_state.cell, state.cell)
        assert torch.allclose(sub_state.atomic_numbers, state.atomic_numbers)
        assert torch.allclose(sub_state.system_idx, state.system_idx)

    assert len(states) == 3


def test_pop_states(
    si_sim_state: SimState,
    ar_supercell_sim_state: SimState,
    fe_supercell_sim_state: SimState,
) -> None:
    """Test popping states from a state."""
    states = [si_sim_state, ar_supercell_sim_state, fe_supercell_sim_state]
    concatenated_states = ts.concatenate_states(states)
    kept_state, popped_states = _pop_states(
        concatenated_states, torch.tensor([0], device=concatenated_states.device)
    )

    assert isinstance(kept_state, SimState)
    assert isinstance(popped_states, list)
    assert len(popped_states) == 1
    assert isinstance(popped_states[0], SimState)
    assert popped_states[0].positions.shape == si_sim_state.positions.shape

    len_kept = ar_supercell_sim_state.n_atoms + fe_supercell_sim_state.n_atoms
    assert kept_state.positions.shape == (len_kept, 3)
    assert kept_state.masses.shape == (len_kept,)
    assert kept_state.cell.shape == (2, 3, 3)
    assert kept_state.atomic_numbers.shape == (len_kept,)
    assert kept_state.system_idx.shape == (len_kept,)


def test_initialize_state_from_structure(si_structure: "Structure") -> None:
    """Test conversion from pymatgen Structure to state tensors."""
    state = ts.initialize_state([si_structure], DEVICE, torch.float64)
    assert isinstance(state, SimState)
    assert state.positions.shape == si_structure.cart_coords.shape
    assert state.cell.shape[1:] == si_structure.lattice.matrix.shape


def test_initialize_state_from_state(ar_supercell_sim_state: SimState) -> None:
    """Test conversion from SimState to SimState."""
    state = ts.initialize_state(ar_supercell_sim_state, DEVICE, torch.float64)
    assert isinstance(state, SimState)
    assert state.positions.shape == ar_supercell_sim_state.positions.shape
    assert state.masses.shape == ar_supercell_sim_state.masses.shape
    assert state.cell.shape == ar_supercell_sim_state.cell.shape


def test_initialize_state_from_atoms(si_atoms: "Atoms") -> None:
    """Test conversion from ASE Atoms to SimState."""
    state = ts.initialize_state([si_atoms], DEVICE, torch.float64)
    assert isinstance(state, SimState)
    assert state.positions.shape == si_atoms.positions.shape
    assert state.masses.shape == si_atoms.get_masses().shape
    assert state.cell.shape[1:] == si_atoms.cell.array.T.shape


def test_initialize_state_from_phonopy_atoms(si_phonopy_atoms: "PhonopyAtoms") -> None:
    """Test conversion from PhonopyAtoms to SimState."""
    state = ts.initialize_state([si_phonopy_atoms], DEVICE, torch.float64)
    assert isinstance(state, SimState)
    assert state.positions.shape == si_phonopy_atoms.positions.shape
    assert state.masses.shape == si_phonopy_atoms.masses.shape
    assert state.cell.shape[1:] == si_phonopy_atoms.cell.shape


def test_state_pop_method(
    si_sim_state: SimState,
    ar_supercell_sim_state: SimState,
    fe_supercell_sim_state: SimState,
) -> None:
    """Test the pop method of SimState."""
    # Create a concatenated state
    states = [si_sim_state, ar_supercell_sim_state, fe_supercell_sim_state]
    concatenated = ts.concatenate_states(states)

    # Test popping a single batch
    popped_states = concatenated.pop(1)
    assert len(popped_states) == 1
    assert isinstance(popped_states[0], SimState)
    assert torch.allclose(popped_states[0].positions, ar_supercell_sim_state.positions)

    # Verify the original state was modified
    assert concatenated.n_systems == 2
    assert torch.unique(concatenated.system_idx).tolist() == [0, 1]

    # Test popping multiple batches
    multi_state = ts.concatenate_states(states)
    popped_multi = multi_state.pop([0, 2])
    assert len(popped_multi) == 2
    assert torch.allclose(popped_multi[0].positions, si_sim_state.positions)
    assert torch.allclose(popped_multi[1].positions, fe_supercell_sim_state.positions)

    # Verify the original multi-state was modified
    assert multi_state.n_systems == 1
    assert torch.unique(multi_state.system_idx).tolist() == [0]
    assert torch.allclose(multi_state.positions, ar_supercell_sim_state.positions)


def test_state_getitem(
    si_sim_state: SimState,
    ar_supercell_sim_state: SimState,
    fe_supercell_sim_state: SimState,
) -> None:
    """Test the __getitem__ method of SimState."""
    # Create a concatenated state
    states = [si_sim_state, ar_supercell_sim_state, fe_supercell_sim_state]
    concatenated = ts.concatenate_states(states)

    # Test integer indexing
    single_state = concatenated[1]
    assert isinstance(single_state, SimState)
    assert torch.allclose(single_state.positions, ar_supercell_sim_state.positions)
    assert single_state.n_systems == 1

    # Test list indexing
    multi_state = concatenated[[0, 2]]
    assert isinstance(multi_state, SimState)
    assert multi_state.n_systems == 2
    assert torch.allclose(multi_state[0].positions, si_sim_state.positions)
    assert torch.allclose(multi_state[1].positions, fe_supercell_sim_state.positions)

    # Test slice indexing
    slice_state = concatenated[1:3]
    assert isinstance(slice_state, SimState)
    assert slice_state.n_systems == 2
    assert torch.allclose(slice_state[0].positions, ar_supercell_sim_state.positions)
    assert torch.allclose(slice_state[1].positions, fe_supercell_sim_state.positions)

    # Test negative indexing
    neg_state = concatenated[-1]
    assert isinstance(neg_state, SimState)
    assert torch.allclose(neg_state.positions, fe_supercell_sim_state.positions)

    # Test step in slice
    step_state = concatenated[::2]
    assert isinstance(step_state, SimState)
    assert step_state.n_systems == 2
    assert torch.allclose(step_state[0].positions, si_sim_state.positions)
    assert torch.allclose(step_state[1].positions, fe_supercell_sim_state.positions)

    full_state = concatenated[:]
    assert torch.allclose(full_state.positions, concatenated.positions)
    # Verify original state is unchanged
    assert concatenated.n_systems == 3


def test_normalize_system_indices(si_double_sim_state: SimState) -> None:
    """Test the _normalize_system_indices utility method."""
    state = si_double_sim_state  # State with 2 batches
    n_systems, device = state.n_systems, state.device

    # Test integer indexing
    assert _normalize_system_indices(0, n_systems, device).tolist() == [0]
    assert _normalize_system_indices(1, n_systems, device).tolist() == [1]

    # Test negative integer indexing
    assert _normalize_system_indices(-1, n_systems, device).tolist() == [1]
    assert _normalize_system_indices(-2, n_systems, device).tolist() == [0]

    # Test list indexing
    assert _normalize_system_indices([0, 1], n_systems, device).tolist() == [0, 1]

    # Test list with negative indices
    assert _normalize_system_indices([0, -1], n_systems, device).tolist() == [0, 1]
    assert _normalize_system_indices([-2, -1], n_systems, device).tolist() == [0, 1]

    # Test slice indexing
    indices = _normalize_system_indices(slice(0, 2), n_systems, device)
    assert isinstance(indices, torch.Tensor)
    assert torch.all(indices == torch.tensor([0, 1], device=state.device))

    # Test slice with negative indices
    indices = _normalize_system_indices(slice(-2, None), n_systems, device)
    assert isinstance(indices, torch.Tensor)
    assert torch.all(indices == torch.tensor([0, 1], device=state.device))

    # Test slice with step
    indices = _normalize_system_indices(slice(0, 2, 2), n_systems, device)
    assert isinstance(indices, torch.Tensor)
    assert torch.all(indices == torch.tensor([0], device=state.device))

    # Test tensor indexing
    tensor_indices = torch.tensor([0, 1], device=state.device)
    indices = _normalize_system_indices(tensor_indices, n_systems, device)
    assert isinstance(indices, torch.Tensor)
    assert torch.all(indices == tensor_indices)

    # Test tensor with negative indices
    tensor_indices = torch.tensor([0, -1], device=state.device)
    indices = _normalize_system_indices(tensor_indices, n_systems, device)
    assert isinstance(indices, torch.Tensor)
    assert torch.all(indices == torch.tensor([0, 1], device=state.device))

    # Test error for unsupported type
    try:
        _normalize_system_indices((0, 1), n_systems, device)  # Tuple is not supported
        raise ValueError("Should have raised TypeError")
    except TypeError:
        pass


def test_row_vector_cell(si_sim_state: SimState) -> None:
    """Test the row_vector_cell property getter and setter."""
    # Test getter - should return transposed cell
    original_cell = si_sim_state.cell.clone()
    row_vector = si_sim_state.row_vector_cell
    assert torch.allclose(row_vector, original_cell.mT)

    # Test setter - should update cell with transposed value
    new_cell = torch.randn_like(original_cell)
    si_sim_state.row_vector_cell = new_cell.mT
    assert torch.allclose(si_sim_state.cell, new_cell)

    # Test consistency of getter after setting
    assert torch.allclose(si_sim_state.row_vector_cell, new_cell.mT)


def test_column_vector_cell(si_sim_state: SimState) -> None:
    """Test the column_vector_cell property getter and setter."""
    # Test getter - should return cell directly since it's already in column vector format
    original_cell = si_sim_state.cell.clone()
    column_vector = si_sim_state.column_vector_cell
    assert torch.allclose(column_vector, original_cell)

    # Test setter - should update cell directly
    new_cell = torch.randn_like(original_cell)
    si_sim_state.column_vector_cell = new_cell
    assert torch.allclose(si_sim_state.cell, new_cell)

    # Test consistency of getter after setting
    assert torch.allclose(si_sim_state.column_vector_cell, new_cell)


class DeformState(SimState, DeformGradMixin):
    """Test class that combines SimState with DeformGradMixin."""

    _system_attributes = (
        SimState._system_attributes  # noqa: SLF001
        | DeformGradMixin._system_attributes  # noqa: SLF001
    )

    def __init__(
        self, *args, velocities: torch.Tensor, reference_cell: torch.Tensor, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.velocities = velocities
        self.reference_cell = reference_cell


@pytest.fixture
def deform_grad_state() -> DeformState:
    """Create a test state with deformation gradient support."""

    positions = torch.randn(10, 3, device=DEVICE)
    masses = torch.ones(10, device=DEVICE)
    velocities = torch.randn(10, 3, device=DEVICE)
    reference_cell = torch.eye(3, device=DEVICE).unsqueeze(0)
    current_cell = 2 * reference_cell

    return DeformState(
        positions=positions,
        masses=masses,
        cell=current_cell,
        pbc=True,
        atomic_numbers=torch.ones(10, device=DEVICE, dtype=torch.long),
        velocities=velocities,
        reference_cell=reference_cell,
    )


def test_deform_grad_reference_cell(deform_grad_state: DeformState) -> None:
    """Test reference cell getter/setter in DeformGradMixin."""
    original_ref_cell = deform_grad_state.reference_cell.clone()

    # Test getter
    assert torch.allclose(
        deform_grad_state.reference_row_vector_cell, original_ref_cell.mT
    )

    # Test setter
    new_ref_cell = 3 * torch.eye(3, device=deform_grad_state.device).unsqueeze(0)
    deform_grad_state.reference_row_vector_cell = new_ref_cell.mT
    assert torch.allclose(deform_grad_state.reference_cell, new_ref_cell)


def test_deform_grad_uniform(deform_grad_state: DeformState) -> None:
    """Test deformation gradient calculation for uniform deformation."""
    # For 2x uniform expansion, deformation gradient should be 2x identity matrix
    deform_grad = deform_grad_state.deform_grad()
    expected = 2 * torch.eye(3, device=deform_grad_state.device).unsqueeze(0)
    assert torch.allclose(deform_grad, expected)


def test_deform_grad_non_uniform() -> None:
    """Test deformation gradient calculation for non-uniform deformation."""
    reference_cell = torch.eye(3, device=DEVICE).unsqueeze(0)
    current_cell = torch.tensor(
        [[[2.0, 0.1, 0.0], [0.1, 1.5, 0.0], [0.0, 0.0, 1.8]]], device=DEVICE
    )

    state = DeformState(
        positions=torch.randn(10, 3, device=DEVICE),
        masses=torch.ones(10, device=DEVICE),
        cell=current_cell,
        pbc=True,
        atomic_numbers=torch.ones(10, device=DEVICE, dtype=torch.long),
        velocities=torch.randn(10, 3, device=DEVICE),
        reference_cell=reference_cell,
    )

    deform_grad = state.deform_grad()
    # Verify that deformation gradient correctly transforms reference cell to current cell
    reconstructed_cell = torch.matmul(reference_cell, deform_grad.mT)
    assert torch.allclose(reconstructed_cell, current_cell)


def test_deform_grad_batched() -> None:
    """Test deformation gradient calculation with batched states."""
    batch_size, n_atoms = 3, 10

    reference_cell = torch.eye(3, device=DEVICE).unsqueeze(0).repeat(batch_size, 1, 1)
    current_cell = torch.stack(
        [
            2.0 * torch.eye(3, device=DEVICE),  # Uniform expansion
            torch.eye(3, device=DEVICE),  # No deformation
            0.5 * torch.eye(3, device=DEVICE),  # Uniform compression
        ]
    )

    state = DeformState(
        positions=torch.randn(n_atoms * batch_size, 3, device=DEVICE),
        masses=torch.ones(n_atoms * batch_size, device=DEVICE),
        cell=current_cell,
        pbc=True,
        atomic_numbers=torch.ones(n_atoms * batch_size, device=DEVICE, dtype=torch.long),
        velocities=torch.randn(n_atoms * batch_size, 3, device=DEVICE),
        reference_cell=reference_cell,
        system_idx=torch.repeat_interleave(
            torch.arange(batch_size, device=DEVICE), n_atoms
        ),
    )

    deform_grad = state.deform_grad()
    assert deform_grad.shape == (batch_size, 3, 3)

    expected_factors = torch.tensor([2.0, 1.0, 0.5], device=DEVICE)
    for batch_idx in range(batch_size):
        expected = expected_factors[batch_idx] * torch.eye(3, device=DEVICE)
        assert torch.allclose(deform_grad[batch_idx], expected)


def test_derived_classes_trigger_init_subclass() -> None:
    """Test that derived classes cannot have attributes that are "tensors | None"."""

    with pytest.raises(TypeError) as exc_info:

        class DerivedState(SimState):
            invalid_attr: torch.Tensor | None = None

    assert "is not allowed to be of type 'torch.Tensor | None' because torch.cat" in str(
        exc_info.value
    )


def test_state_to_device_no_side_effects(si_sim_state: SimState) -> None:
    """Test that SimState.to() doesn't modify the original state."""
    # Store original values
    original_positions = si_sim_state.positions.clone()
    original_dtype = si_sim_state.dtype
    original_device = si_sim_state.device

    # Convert to different dtype
    new_state = si_sim_state.to(dtype=torch.float64)

    # Verify original state is unchanged
    assert torch.allclose(si_sim_state.positions, original_positions), (
        "Original state was modified!"
    )
    assert si_sim_state.dtype == original_dtype, "Original state dtype was changed!"
    assert si_sim_state.device == original_device, "Original state device was changed!"
    assert si_sim_state is not new_state, "New state is not a different object!"
    assert new_state.dtype == torch.float64, "New state doesn't have correct dtype!"

    # Test device conversion
    if torch.cuda.is_available():
        new_state_gpu = si_sim_state.to(device=torch.device("cuda"))
        assert si_sim_state.device == original_device, (
            "Original state device was changed!"
        )
        assert new_state_gpu.device.type == "cuda", (
            "New state doesn't have correct device!"
        )
        assert si_sim_state is not new_state_gpu, "New state is not a different object!"


def test_state_set_cell(ti_sim_state: SimState) -> None:
    """Test the set_cell method of SimState."""
    new_cell = (
        torch.diag_embed(
            torch.tensor(
                [3.0, 4.0, 5.0], device=ti_sim_state.device, dtype=ti_sim_state.dtype
            )
        )
        @ ti_sim_state.cell
    )
    ase_atoms = ti_sim_state.to_atoms()[0]
    ti_sim_state.set_cell(new_cell, scale_atoms=True)
    ase_atoms.set_cell(new_cell[0].T.cpu().numpy(), scale_atoms=True)
    assert torch.allclose(
        ti_sim_state.positions.cpu(), torch.from_numpy(ase_atoms.positions)
    )

    M = torch.tensor(
        [[[1.0, 0.2, 0.0], [0.1, 1.5, 0.0], [0.0, 0.0, 2.0]]],
        device=DEVICE,
        dtype=ti_sim_state.dtype,
    )
    new_cell = M @ ti_sim_state.cell
    ase_atoms = ti_sim_state.to_atoms()[0]
    ti_sim_state.set_cell(new_cell, scale_atoms=True)
    ase_atoms.set_cell(new_cell[0].T.cpu().numpy(), scale_atoms=True)
    assert torch.allclose(
        ti_sim_state.positions.cpu(), torch.from_numpy(ase_atoms.positions)
    )


def test_wrap_positions_no_pbc(si_sim_state: SimState) -> None:
    """Test wrap_positions returns unwrapped positions when pbc=False."""
    state = si_sim_state.clone()
    state.pbc = torch.tensor([False, False, False])
    # Move some atoms outside the cell
    state.positions = state.positions + 100.0
    # With no pbc, wrap_positions should return positions unchanged
    assert torch.allclose(state.wrap_positions, state.positions)


def test_wrap_positions_with_pbc(si_sim_state: SimState) -> None:
    """Test wrap_positions wraps positions when pbc=True."""
    state = si_sim_state.clone()
    state.pbc = torch.tensor([True, True, True])
    original_positions = state.positions.clone()
    # Add one lattice vector to move atoms outside
    lattice_shift = state.row_vector_cell[0, 0]  # first lattice vector
    state.positions = state.positions + lattice_shift
    # Wrapped positions should be back to original (within tolerance)
    wrapped = state.wrap_positions
    assert torch.allclose(wrapped, original_positions, atol=1e-5)


def test_wrap_positions_mixed_pbc(si_sim_state: SimState) -> None:
    """Test wrap_positions with mixed pbc (True in some dimensions, False in others)."""
    state = si_sim_state.clone()
    state.pbc = torch.tensor([True, False, True])  # periodic in x and z, not y
    original_positions = state.positions.clone()
    # Shift by lattice vectors in all directions
    shift_x = state.row_vector_cell[0, 0]  # first lattice vector (x)
    shift_y = state.row_vector_cell[0, 1]  # second lattice vector (y)
    shift_z = state.row_vector_cell[0, 2]  # third lattice vector (z)
    state.positions = state.positions + shift_x + shift_y + shift_z
    wrapped = state.wrap_positions
    # x and z should be wrapped back, y should remain shifted
    expected = original_positions + shift_y
    assert torch.allclose(wrapped, expected, atol=1e-5)


def test_wrap_positions_batched(si_double_sim_state: SimState) -> None:
    """Test wrap_positions works with batched systems."""
    state = si_double_sim_state.clone()
    state.pbc = torch.tensor([True, True, True])
    original_positions = state.positions.clone()
    # Shift all positions by one lattice vector (using first system's cell)
    for sys_idx in range(state.n_systems):
        mask = state.system_idx == sys_idx
        lattice_shift = state.row_vector_cell[sys_idx, 0]
        state.positions[mask] = state.positions[mask] + lattice_shift
    wrapped = state.wrap_positions
    assert torch.allclose(wrapped, original_positions, atol=1e-5)


# ── rng property tests ──────────────────────────────────────────────────────


def test_rng_lazy_init(si_sim_state: SimState) -> None:
    """rng property creates a Generator on first access when _rng is None."""
    state = si_sim_state.clone()
    state.rng = None
    assert state._rng is None  # noqa: SLF001
    gen = state.rng
    assert isinstance(gen, torch.Generator)
    assert gen.device == state.device


def test_rng_int_seed_via_constructor() -> None:
    """Passing an int _rng to SimState coerces it to a Generator in __post_init__."""
    state = SimState(
        positions=torch.randn(2, 3),
        masses=torch.ones(2),
        cell=torch.eye(3).unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.ones(2, dtype=torch.int),
        _rng=42,
    )
    assert isinstance(state._rng, torch.Generator)  # noqa: SLF001
    assert isinstance(state.rng, torch.Generator)


def test_rng_int_seed_via_property(si_sim_state: SimState) -> None:
    """Setting rng to an int via the property coerces on next access."""
    state = si_sim_state.clone()
    state.rng = 123
    gen = state.rng
    assert isinstance(gen, torch.Generator)


def test_rng_generator_passthrough(si_sim_state: SimState) -> None:
    """Setting rng to a Generator stores it directly."""
    state = si_sim_state.clone()
    gen = torch.Generator(device=state.device)
    gen.manual_seed(7)
    state.rng = gen
    assert state.rng is gen


def test_rng_none_resets(si_sim_state: SimState) -> None:
    """Setting rng = None resets; next access lazily re-initialises."""
    state = si_sim_state.clone()
    state.rng = 42
    first_gen = state.rng
    state.rng = None
    second_gen = state.rng
    assert isinstance(second_gen, torch.Generator)
    assert second_gen is not first_gen


def test_rng_int_seed_reproducible() -> None:
    """Same int seed produces the same random sequence."""

    def _make_state(seed: int) -> SimState:
        return SimState(
            positions=torch.randn(4, 3),
            masses=torch.ones(4),
            cell=torch.eye(3).unsqueeze(0),
            pbc=True,
            atomic_numbers=torch.ones(4, dtype=torch.int),
            _rng=seed,
        )

    s1 = _make_state(99)
    s2 = _make_state(99)
    r1 = torch.randn(5, generator=s1.rng)
    r2 = torch.randn(5, generator=s2.rng)
    assert torch.equal(r1, r2)


def test_rng_clone_independent(si_sim_state: SimState) -> None:
    """Cloned state has an independent Generator with the same state."""
    state = si_sim_state.clone()
    state.rng = 42
    clone = state.clone()

    assert torch.equal(state.rng.get_state(), clone.rng.get_state())
    assert clone.rng is not state.rng

    # Drawing from one does not affect the other
    torch.randn(3, generator=state.rng)
    assert not torch.equal(state.rng.get_state(), clone.rng.get_state())


def test_rng_clone_none_preserved(si_sim_state: SimState) -> None:
    """Cloning a state with _rng=None keeps it None on the clone."""
    state = si_sim_state.clone()
    state.rng = None
    clone = state.clone()
    assert clone._rng is None  # noqa: SLF001


def test_rng_from_state_mdstate(si_sim_state: SimState) -> None:
    """MDState.from_state copies the rng from the source SimState."""
    state = si_sim_state.clone()
    state.rng = 42
    original_rng_state = state.rng.get_state().clone()

    md = MDState.from_state(
        state,
        momenta=torch.zeros_like(state.positions),
        energy=torch.zeros(state.n_systems, device=state.device),
        forces=torch.zeros_like(state.positions),
    )
    assert isinstance(md.rng, torch.Generator)
    assert torch.equal(md.rng.get_state(), original_rng_state)
    assert md.rng is not state.rng


def test_rng_mdstate_inherits_lazy_init() -> None:
    """MDState without explicit _rng still lazily initialises via the property."""
    md = MDState(
        positions=torch.randn(2, 3),
        masses=torch.ones(2),
        cell=torch.eye(3).unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.ones(2, dtype=torch.int),
        momenta=torch.zeros(2, 3),
        energy=torch.zeros(1),
        forces=torch.zeros(2, 3),
    )
    assert md._rng is None  # noqa: SLF001
    gen = md.rng
    assert isinstance(gen, torch.Generator)


def test_rng_concat_takes_first(si_sim_state: SimState) -> None:
    """concatenate_states takes rng from the first state."""
    s1 = si_sim_state.clone()
    s2 = si_sim_state.clone()
    s1.rng = 11
    s2.rng = 22
    combined = ts.concatenate_states([s1, s2])
    assert torch.equal(combined.rng.get_state(), s1.rng.get_state())


def test_rng_split_preserves(si_sim_state: SimState) -> None:
    """Splitting a batched state shares the same rng value to each piece."""
    batched = ts.concatenate_states([si_sim_state, si_sim_state])
    batched.rng = 77
    parts = batched.split()
    assert len(parts) == 2
    for part in parts:
        assert isinstance(part.rng, torch.Generator)


def test_coerce_prng_none() -> None:
    """None seed creates an unseeded Generator."""
    gen = coerce_prng(None, device=DEVICE)
    assert isinstance(gen, torch.Generator)


def test_coerce_prng_int_seed() -> None:
    """Int seed creates a deterministically-seeded Generator."""
    g1 = coerce_prng(42, device=DEVICE)
    g2 = coerce_prng(42, device=DEVICE)
    r1 = torch.randn(5, generator=g1)
    r2 = torch.randn(5, generator=g2)
    assert torch.equal(r1, r2)


def test_coerce_prng_different_seeds_diverge() -> None:
    """Different int seeds produce different random streams."""
    g1 = coerce_prng(1, device=DEVICE)
    g2 = coerce_prng(2, device=DEVICE)
    r1 = torch.randn(5, generator=g1)
    r2 = torch.randn(5, generator=g2)
    assert not torch.equal(r1, r2)


def test_coerce_prng_generator_passthrough() -> None:
    """Passing a Generator returns the exact same object."""
    gen = torch.Generator()
    gen.manual_seed(7)
    result = coerce_prng(gen, device=DEVICE)
    assert result is gen


def test_coerce_prng_default_no_arg() -> None:
    """Calling with no argument (default None) returns a Generator."""
    gen = coerce_prng(None, device=DEVICE)
    assert isinstance(gen, torch.Generator)
