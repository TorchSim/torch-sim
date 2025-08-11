import typing
from dataclasses import asdict

import pytest
import torch

import torch_sim as ts
from torch_sim.integrators import MDState
from torch_sim.state import (
    DeformGradMixin,
    SimState,
    _normalize_system_indices,
    _pop_states,
    _slice_state,
    concatenate_states,
    get_attrs_for_scope,
    initialize_state,
)


if typing.TYPE_CHECKING:
    from ase import Atoms
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure


def test_get_attrs_for_scope(si_sim_state: ts.SimState) -> None:
    """Test getting attributes for a scope."""
    per_atom_attrs = dict(get_attrs_for_scope(si_sim_state, "per-atom"))
    assert set(per_atom_attrs.keys()) == {
        "positions",
        "masses",
        "atomic_numbers",
        "system_idx",
    }
    per_system_attrs = dict(get_attrs_for_scope(si_sim_state, "per-system"))
    assert set(per_system_attrs.keys()) == {"cell"}
    global_attrs = dict(get_attrs_for_scope(si_sim_state, "global"))
    assert set(global_attrs.keys()) == {"pbc"}


def test_all_attributes_must_be_specified_in_scopes() -> None:
    """Test that an error is raised when we forget to specify the scope
    for an attribute in a child SimState class."""
    with pytest.raises(TypeError) as excinfo:

        class ChildState(SimState):
            attribute_specified_in_scopes: bool
            attribute_not_specified_in_scopes: bool

            _atom_attributes: typing.ClassVar[set[str]] = (
                SimState._atom_attributes | {"attribute_specified_in_scopes"}  # noqa: SLF001
            )

    assert "attribute_not_specified_in_scopes" in str(excinfo.value)
    assert "attribute_specified_in_scopes" not in str(excinfo.value)


def test_no_duplicate_attributes_in_scopes() -> None:
    """Test that no attributes are specified in multiple scopes."""

    # Capture the exception information using "as excinfo"
    with pytest.raises(TypeError) as excinfo:

        class ChildState(SimState):
            duplicated_attribute: bool

            _system_attributes: typing.ClassVar[set[str]] = (
                SimState._system_attributes | {"duplicated_attribute"}  # noqa: SLF001
            )
            _global_attributes: typing.ClassVar[set[str]] = (
                SimState._global_attributes | {"duplicated_attribute"}  # noqa: SLF001
            )

    assert "are declared multiple times" in str(excinfo.value)
    assert "duplicated_attribute" in str(excinfo.value)


def test_slice_substate(
    si_double_sim_state: ts.SimState, si_sim_state: ts.SimState
) -> None:
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


def test_slice_md_substate(si_double_sim_state: ts.SimState) -> None:
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


def test_concatenate_two_si_states(
    si_sim_state: ts.SimState, si_double_sim_state: ts.SimState
) -> None:
    """Test concatenating two identical silicon states."""
    # Concatenate two copies of the sim state
    concatenated = concatenate_states([si_sim_state, si_sim_state])

    # Check that the result is the same as the double state
    assert isinstance(concatenated, SimState)
    assert concatenated.positions.shape == si_double_sim_state.positions.shape
    assert concatenated.masses.shape == si_double_sim_state.masses.shape
    assert concatenated.cell.shape == si_double_sim_state.cell.shape
    assert concatenated.atomic_numbers.shape == si_double_sim_state.atomic_numbers.shape
    assert concatenated.system_idx.shape == si_double_sim_state.system_idx.shape

    # Check system indices
    expected_system_indices = torch.cat(
        [
            torch.zeros(
                si_sim_state.n_atoms, dtype=torch.int64, device=si_sim_state.device
            ),
            torch.ones(
                si_sim_state.n_atoms, dtype=torch.int64, device=si_sim_state.device
            ),
        ]
    )
    assert torch.all(concatenated.system_idx == expected_system_indices)

    # Check that positions match (accounting for system indices)
    for system_idx in range(2):
        mask_concat = concatenated.system_idx == system_idx
        mask_double = si_double_sim_state.system_idx == system_idx
        assert torch.allclose(
            concatenated.positions[mask_concat],
            si_double_sim_state.positions[mask_double],
        )


def test_concatenate_si_and_fe_states(
    si_sim_state: ts.SimState, fe_supercell_sim_state: ts.SimState
) -> None:
    """Test concatenating silicon and argon states."""
    # Concatenate silicon and argon states
    concatenated = concatenate_states([si_sim_state, fe_supercell_sim_state])

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
    si_double_sim_state: ts.SimState, fe_supercell_sim_state: ts.SimState
) -> None:
    """Test concatenating a double silicon state and an argon state."""
    # Concatenate double silicon and argon states
    concatenated = concatenate_states([si_double_sim_state, fe_supercell_sim_state])

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


def test_split_state(si_double_sim_state: ts.SimState) -> None:
    """Test splitting a state into a list of states."""
    states = si_double_sim_state.split()
    assert len(states) == si_double_sim_state.n_systems
    for state in states:
        assert isinstance(state, ts.SimState)
        assert state.positions.shape == (8, 3)
        assert state.masses.shape == (8,)
        assert state.cell.shape == (1, 3, 3)
        assert state.atomic_numbers.shape == (8,)
        assert torch.allclose(state.system_idx, torch.zeros_like(state.system_idx))


def test_split_many_states(
    si_sim_state: ts.SimState,
    ar_supercell_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
) -> None:
    """Test splitting a state into a list of states."""
    states = [si_sim_state, ar_supercell_sim_state, fe_supercell_sim_state]
    concatenated = concatenate_states(states)
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
    si_sim_state: ts.SimState,
    ar_supercell_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
) -> None:
    """Test popping states from a state."""
    states = [si_sim_state, ar_supercell_sim_state, fe_supercell_sim_state]
    concatenated_states = concatenate_states(states)
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


def test_initialize_state_from_structure(
    si_structure: "Structure", device: torch.device
) -> None:
    """Test conversion from pymatgen Structure to state tensors."""
    state = initialize_state([si_structure], device, torch.float64)
    assert isinstance(state, ts.SimState)
    assert state.positions.shape == si_structure.cart_coords.shape
    assert state.cell.shape[1:] == si_structure.lattice.matrix.shape


def test_initialize_state_from_state(
    ar_supercell_sim_state: ts.SimState, device: torch.device
) -> None:
    """Test conversion from SimState to SimState."""
    state = initialize_state(ar_supercell_sim_state, device, torch.float64)
    assert isinstance(state, ts.SimState)
    assert state.positions.shape == ar_supercell_sim_state.positions.shape
    assert state.masses.shape == ar_supercell_sim_state.masses.shape
    assert state.cell.shape == ar_supercell_sim_state.cell.shape


def test_initialize_state_from_atoms(si_atoms: "Atoms", device: torch.device) -> None:
    """Test conversion from ASE Atoms to SimState."""
    state = initialize_state([si_atoms], device, torch.float64)
    assert isinstance(state, ts.SimState)
    assert state.positions.shape == si_atoms.positions.shape
    assert state.masses.shape == si_atoms.get_masses().shape
    assert state.cell.shape[1:] == si_atoms.cell.array.T.shape


def test_initialize_state_from_phonopy_atoms(
    si_phonopy_atoms: "PhonopyAtoms", device: torch.device
) -> None:
    """Test conversion from PhonopyAtoms to SimState."""
    state = initialize_state([si_phonopy_atoms], device, torch.float64)
    assert isinstance(state, ts.SimState)
    assert state.positions.shape == si_phonopy_atoms.positions.shape
    assert state.masses.shape == si_phonopy_atoms.masses.shape
    assert state.cell.shape[1:] == si_phonopy_atoms.cell.shape


def test_state_pop_method(
    si_sim_state: ts.SimState,
    ar_supercell_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
) -> None:
    """Test the pop method of SimState."""
    # Create a concatenated state
    states = [si_sim_state, ar_supercell_sim_state, fe_supercell_sim_state]
    concatenated = concatenate_states(states)

    # Test popping a single batch
    popped_states = concatenated.pop(1)
    assert len(popped_states) == 1
    assert isinstance(popped_states[0], SimState)
    assert torch.allclose(popped_states[0].positions, ar_supercell_sim_state.positions)

    # Verify the original state was modified
    assert concatenated.n_systems == 2
    assert torch.unique(concatenated.system_idx).tolist() == [0, 1]

    # Test popping multiple batches
    multi_state = concatenate_states(states)
    popped_multi = multi_state.pop([0, 2])
    assert len(popped_multi) == 2
    assert torch.allclose(popped_multi[0].positions, si_sim_state.positions)
    assert torch.allclose(popped_multi[1].positions, fe_supercell_sim_state.positions)

    # Verify the original multi-state was modified
    assert multi_state.n_systems == 1
    assert torch.unique(multi_state.system_idx).tolist() == [0]
    assert torch.allclose(multi_state.positions, ar_supercell_sim_state.positions)


def test_state_getitem(
    si_sim_state: ts.SimState,
    ar_supercell_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
) -> None:
    """Test the __getitem__ method of SimState."""
    # Create a concatenated state
    states = [si_sim_state, ar_supercell_sim_state, fe_supercell_sim_state]
    concatenated = concatenate_states(states)

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


def test_normalize_system_indices(si_double_sim_state: ts.SimState) -> None:
    """Test the _normalize_system_indices utility method."""
    state = si_double_sim_state  # State with 2 batches
    n_systems = state.n_systems
    device = state.device

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


def test_row_vector_cell(si_sim_state: ts.SimState) -> None:
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


def test_column_vector_cell(si_sim_state: ts.SimState) -> None:
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

    _system_attributes: typing.ClassVar[set[str]] = (
        SimState._system_attributes  # noqa: SLF001
        | DeformGradMixin._system_attributes  # noqa: SLF001
    )

    def __init__(
        self,
        *args,
        velocities: torch.Tensor,
        reference_cell: torch.Tensor,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.velocities = velocities
        self.reference_cell = reference_cell


@pytest.fixture
def deform_grad_state(device: torch.device) -> DeformState:
    """Create a test state with deformation gradient support."""

    positions = torch.randn(10, 3, device=device)
    masses = torch.ones(10, device=device)
    velocities = torch.randn(10, 3, device=device)
    reference_cell = torch.eye(3, device=device).unsqueeze(0)
    current_cell = 2 * reference_cell

    return DeformState(
        positions=positions,
        masses=masses,
        cell=current_cell,
        pbc=True,
        atomic_numbers=torch.ones(10, device=device, dtype=torch.long),
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


def test_deform_grad_non_uniform(device: torch.device) -> None:
    """Test deformation gradient calculation for non-uniform deformation."""
    reference_cell = torch.eye(3, device=device).unsqueeze(0)
    current_cell = torch.tensor(
        [[[2.0, 0.1, 0.0], [0.1, 1.5, 0.0], [0.0, 0.0, 1.8]]], device=device
    )

    state = DeformState(
        positions=torch.randn(10, 3, device=device),
        masses=torch.ones(10, device=device),
        cell=current_cell,
        pbc=True,
        atomic_numbers=torch.ones(10, device=device, dtype=torch.long),
        velocities=torch.randn(10, 3, device=device),
        reference_cell=reference_cell,
    )

    deform_grad = state.deform_grad()
    # Verify that deformation gradient correctly transforms reference cell to current cell
    reconstructed_cell = torch.matmul(reference_cell, deform_grad.mT)
    assert torch.allclose(reconstructed_cell, current_cell)


def test_deform_grad_batched(device: torch.device) -> None:
    """Test deformation gradient calculation with batched states."""
    batch_size = 3
    n_atoms = 10

    reference_cell = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    current_cell = torch.stack(
        [
            2.0 * torch.eye(3, device=device),  # Uniform expansion
            torch.eye(3, device=device),  # No deformation
            0.5 * torch.eye(3, device=device),  # Uniform compression
        ]
    )

    state = DeformState(
        positions=torch.randn(n_atoms * batch_size, 3, device=device),
        masses=torch.ones(n_atoms * batch_size, device=device),
        cell=current_cell,
        pbc=True,
        atomic_numbers=torch.ones(n_atoms * batch_size, device=device, dtype=torch.long),
        velocities=torch.randn(n_atoms * batch_size, 3, device=device),
        reference_cell=reference_cell,
        system_idx=torch.repeat_interleave(
            torch.arange(batch_size, device=device), n_atoms
        ),
    )

    deform_grad = state.deform_grad()
    assert deform_grad.shape == (batch_size, 3, 3)

    expected_factors = torch.tensor([2.0, 1.0, 0.5], device=device)
    for i in range(batch_size):
        expected = expected_factors[i] * torch.eye(3, device=device)
        assert torch.allclose(deform_grad[i], expected)


def test_deprecated_batch_properties_equal_to_new_system_properties(
    device: torch.device,
) -> None:
    """Test that deprecated batch properties are equal to new system properties.

    This tests that the rename from batch to system is not breaking anything."""
    state = SimState(
        positions=torch.randn(10, 3, device=device),
        masses=torch.ones(10, device=device),
        cell=torch.eye(3, device=device).unsqueeze(0).repeat(2, 1, 1),
        pbc=True,
        atomic_numbers=torch.ones(10, device=device, dtype=torch.long),
        system_idx=torch.repeat_interleave(torch.arange(2, device=device), 5),
    )
    assert state.batch is state.system_idx
    assert state.n_batches == state.n_systems
    assert torch.allclose(state.n_atoms_per_batch, state.n_atoms_per_system)

    # now test that assigning the old .batch property behaves the same
    new_system_idx = torch.arange(4, device=device)
    state.batch = new_system_idx
    assert torch.allclose(state.system_idx, new_system_idx)
    assert torch.allclose(state.batch, new_system_idx)


def test_derived_classes_trigger_init_subclass() -> None:
    """Test that derived classes cannot have attributes that are "tensors | None"."""

    with pytest.raises(TypeError) as excinfo:

        class DerivedState(SimState):
            invalid_attr: torch.Tensor | None = None

    assert "is not allowed to be of type 'torch.Tensor | None' because torch.cat" in str(
        excinfo.value
    )
