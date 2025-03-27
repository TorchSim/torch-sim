import typing
from dataclasses import asdict

import torch

from torch_sim.integrators import MDState
from torch_sim.state import (
    SimState,
    _normalize_batch_indices,
    concatenate_states,
    infer_property_scope,
    initialize_state,
    _pop_states,
    _slice_state,
)


if typing.TYPE_CHECKING:
    from ase import Atoms
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure


def test_infer_sim_state_property_scope(si_sim_state: SimState) -> None:
    """Test inference of property scope."""
    scope = infer_property_scope(si_sim_state)
    assert set(scope["global"]) == {"pbc"}
    assert set(scope["per_atom"]) == {"positions", "masses", "atomic_numbers", "batch"}
    assert set(scope["per_batch"]) == {"cell"}


def test_infer_md_state_property_scope(si_sim_state: SimState) -> None:
    """Test inference of property scope."""
    state = MDState(
        **asdict(si_sim_state),
        momenta=torch.randn_like(si_sim_state.positions),
        forces=torch.randn_like(si_sim_state.positions),
        energy=torch.zeros((1,)),
    )
    scope = infer_property_scope(state)
    assert set(scope["global"]) == {"pbc"}
    assert set(scope["per_atom"]) == {
        "positions",
        "masses",
        "atomic_numbers",
        "batch",
        "forces",
        "momenta",
    }
    assert set(scope["per_batch"]) == {"cell", "energy"}


def test_slice_substate(si_double_sim_state: SimState, si_sim_state: SimState) -> None:
    """Test slicing a substate from the SimState."""
    for batch_index in range(2):
        substate = _slice_state(si_double_sim_state, [batch_index])
        assert isinstance(substate, SimState)
        assert substate.positions.shape == (8, 3)
        assert substate.masses.shape == (8,)
        assert substate.cell.shape == (1, 3, 3)
        assert torch.allclose(substate.positions, si_sim_state.positions)
        assert torch.allclose(substate.masses, si_sim_state.masses)
        assert torch.allclose(substate.cell, si_sim_state.cell)
        assert torch.allclose(substate.atomic_numbers, si_sim_state.atomic_numbers)
        assert torch.allclose(substate.batch, torch.zeros_like(substate.batch))


def test_slice_md_substate(si_double_sim_state: SimState) -> None:
    state = MDState(
        **asdict(si_double_sim_state),
        momenta=torch.randn_like(si_double_sim_state.positions),
        energy=torch.zeros((2,), device=si_double_sim_state.device),
        forces=torch.randn_like(si_double_sim_state.positions),
    )
    for batch_index in range(2):
        substate = _slice_state(state, [batch_index])
        assert isinstance(substate, MDState)
        assert substate.positions.shape == (8, 3)
        assert substate.masses.shape == (8,)
        assert substate.cell.shape == (1, 3, 3)
        assert substate.momenta.shape == (8, 3)
        assert substate.forces.shape == (8, 3)
        assert substate.energy.shape == (1,)


def test_concatenate_two_si_states(
    si_sim_state: SimState, si_double_sim_state: SimState
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
    assert concatenated.batch.shape == si_double_sim_state.batch.shape

    # Check batch indices
    expected_batch = torch.cat(
        [
            torch.zeros(
                si_sim_state.n_atoms, dtype=torch.int64, device=si_sim_state.device
            ),
            torch.ones(
                si_sim_state.n_atoms, dtype=torch.int64, device=si_sim_state.device
            ),
        ]
    )
    assert torch.all(concatenated.batch == expected_batch)

    # Check that positions match (accounting for batch indices)
    for batch_idx in range(2):
        mask_concat = concatenated.batch == batch_idx
        mask_double = si_double_sim_state.batch == batch_idx
        assert torch.allclose(
            concatenated.positions[mask_concat],
            si_double_sim_state.positions[mask_double],
        )


def test_concatenate_si_and_fe_states(
    si_sim_state: SimState, fe_fcc_sim_state: SimState
) -> None:
    """Test concatenating silicon and argon states."""
    # Concatenate silicon and argon states
    concatenated = concatenate_states([si_sim_state, fe_fcc_sim_state])

    # Check basic properties
    assert isinstance(concatenated, SimState)
    assert (
        concatenated.positions.shape[0]
        == si_sim_state.positions.shape[0] + fe_fcc_sim_state.positions.shape[0]
    )
    assert (
        concatenated.masses.shape[0]
        == si_sim_state.masses.shape[0] + fe_fcc_sim_state.masses.shape[0]
    )
    assert concatenated.cell.shape[0] == 2  # One cell per batch

    # Check batch indices
    si_atoms = si_sim_state.n_atoms
    fe_atoms = fe_fcc_sim_state.n_atoms
    expected_batch = torch.cat(
        [
            torch.zeros(si_atoms, dtype=torch.int64, device=si_sim_state.device),
            torch.ones(fe_atoms, dtype=torch.int64, device=fe_fcc_sim_state.device),
        ]
    )
    assert torch.all(concatenated.batch == expected_batch)

    # Check that positions match for each original state
    assert torch.allclose(concatenated.positions[:si_atoms], si_sim_state.positions)
    assert torch.allclose(concatenated.positions[si_atoms:], fe_fcc_sim_state.positions)

    # Check that atomic numbers are correct
    assert torch.all(concatenated.atomic_numbers[:si_atoms] == 14)  # Si
    assert torch.all(concatenated.atomic_numbers[si_atoms:] == 26)  # Fe


def test_concatenate_double_si_and_fe_states(
    si_double_sim_state: SimState, fe_fcc_sim_state: SimState
) -> None:
    """Test concatenating a double silicon state and an argon state."""
    # Concatenate double silicon and argon states
    concatenated = concatenate_states([si_double_sim_state, fe_fcc_sim_state])

    # Check basic properties
    assert isinstance(concatenated, SimState)
    assert (
        concatenated.positions.shape[0]
        == si_double_sim_state.positions.shape[0] + fe_fcc_sim_state.positions.shape[0]
    )
    assert (
        concatenated.cell.shape[0] == 3
    )  # One cell for each original batch (2 Si + 1 Ar)

    # Check batch indices
    fe_atoms = fe_fcc_sim_state.n_atoms

    # The double Si state already has batches 0 and 1, so Ar should be batch 2
    expected_batch = torch.cat(
        [
            si_double_sim_state.batch,
            torch.full((fe_atoms,), 2, dtype=torch.int64, device=fe_fcc_sim_state.device),
        ]
    )
    assert torch.all(concatenated.batch == expected_batch)
    assert torch.unique(concatenated.batch).shape[0] == 3

    # Check that we can slice back to the original states
    si_slice_0 = concatenated[0]
    si_slice_1 = concatenated[1]
    fe_slice = concatenated[2]

    # Check that the slices match the original states
    assert torch.allclose(si_slice_0.positions, si_double_sim_state[0].positions)
    assert torch.allclose(si_slice_1.positions, si_double_sim_state[1].positions)
    assert torch.allclose(fe_slice.positions, fe_fcc_sim_state.positions)


def test_split_state(si_double_sim_state: SimState) -> None:
    """Test splitting a state into a list of states."""
    states = si_double_sim_state.split()
    assert len(states) == si_double_sim_state.n_batches
    for state in states:
        assert isinstance(state, SimState)
        assert state.positions.shape == (8, 3)
        assert state.masses.shape == (8,)
        assert state.cell.shape == (1, 3, 3)
        assert state.atomic_numbers.shape == (8,)
        assert torch.allclose(state.batch, torch.zeros_like(state.batch))


def test_split_many_states(
    si_sim_state: SimState, ar_sim_state: SimState, fe_fcc_sim_state: SimState
) -> None:
    """Test splitting a state into a list of states."""
    states = [si_sim_state, ar_sim_state, fe_fcc_sim_state]
    concatenated = concatenate_states(states)
    split_states = concatenated.split()
    for state, sub_state in zip(states, split_states, strict=True):
        assert isinstance(sub_state, SimState)
        assert torch.allclose(sub_state.positions, state.positions)
        assert torch.allclose(sub_state.masses, state.masses)
        assert torch.allclose(sub_state.cell, state.cell)
        assert torch.allclose(sub_state.atomic_numbers, state.atomic_numbers)
        assert torch.allclose(sub_state.batch, state.batch)

    assert len(states) == 3


def test_pop_states(
    si_sim_state: SimState, ar_sim_state: SimState, fe_fcc_sim_state: SimState
) -> None:
    """Test popping states from a state."""
    states = [si_sim_state, ar_sim_state, fe_fcc_sim_state]
    concatenated_states = concatenate_states(states)
    kept_state, popped_states = _pop_states(
        concatenated_states, torch.tensor([0], device=concatenated_states.device)
    )

    assert isinstance(kept_state, SimState)
    assert isinstance(popped_states, list)
    assert len(popped_states) == 1
    assert isinstance(popped_states[0], SimState)
    assert popped_states[0].positions.shape == si_sim_state.positions.shape

    len_kept = ar_sim_state.n_atoms + fe_fcc_sim_state.n_atoms
    assert kept_state.positions.shape == (len_kept, 3)
    assert kept_state.masses.shape == (len_kept,)
    assert kept_state.cell.shape == (2, 3, 3)
    assert kept_state.atomic_numbers.shape == (len_kept,)
    assert kept_state.batch.shape == (len_kept,)


def test_initialize_state_from_structure(
    si_structure: "Structure", device: torch.device
) -> None:
    """Test conversion from pymatgen Structure to state tensors."""
    state = initialize_state([si_structure], device, torch.float64)
    assert isinstance(state, SimState)
    assert state.positions.shape == si_structure.cart_coords.shape
    assert state.cell.shape[1:] == si_structure.lattice.matrix.shape


def test_initialize_state_from_state(
    ar_sim_state: SimState, device: torch.device
) -> None:
    """Test conversion from SimState to SimState."""
    state = initialize_state(ar_sim_state, device, torch.float64)
    assert isinstance(state, SimState)
    assert state.positions.shape == ar_sim_state.positions.shape
    assert state.masses.shape == ar_sim_state.masses.shape
    assert state.cell.shape == ar_sim_state.cell.shape


def test_initialize_state_from_atoms(si_atoms: "Atoms", device: torch.device) -> None:
    """Test conversion from ASE Atoms to SimState."""
    state = initialize_state([si_atoms], device, torch.float64)
    assert isinstance(state, SimState)
    assert state.positions.shape == si_atoms.positions.shape
    assert state.masses.shape == si_atoms.get_masses().shape
    assert state.cell.shape[1:] == si_atoms.cell.array.T.shape


def test_initialize_state_from_phonopy_atoms(
    si_phonopy_atoms: "PhonopyAtoms", device: torch.device
) -> None:
    """Test conversion from PhonopyAtoms to SimState."""
    state = initialize_state([si_phonopy_atoms], device, torch.float64)
    assert isinstance(state, SimState)
    assert state.positions.shape == si_phonopy_atoms.positions.shape
    assert state.masses.shape == si_phonopy_atoms.masses.shape
    assert state.cell.shape[1:] == si_phonopy_atoms.cell.shape


def test_state_pop_method(
    si_sim_state: SimState, ar_sim_state: SimState, fe_fcc_sim_state: SimState
) -> None:
    """Test the pop method of SimState."""
    # Create a concatenated state
    states = [si_sim_state, ar_sim_state, fe_fcc_sim_state]
    concatenated = concatenate_states(states)

    # Test popping a single batch
    popped_states = concatenated.pop(1)
    assert len(popped_states) == 1
    assert isinstance(popped_states[0], SimState)
    assert torch.allclose(popped_states[0].positions, ar_sim_state.positions)

    # Verify the original state was modified
    assert concatenated.n_batches == 2
    assert torch.unique(concatenated.batch).tolist() == [0, 1]

    # Test popping multiple batches
    multi_state = concatenate_states(states)
    popped_multi = multi_state.pop([0, 2])
    assert len(popped_multi) == 2
    assert torch.allclose(popped_multi[0].positions, si_sim_state.positions)
    assert torch.allclose(popped_multi[1].positions, fe_fcc_sim_state.positions)

    # Verify the original multi-state was modified
    assert multi_state.n_batches == 1
    assert torch.unique(multi_state.batch).tolist() == [0]
    assert torch.allclose(multi_state.positions, ar_sim_state.positions)


def test_state_getitem(
    si_sim_state: SimState, ar_sim_state: SimState, fe_fcc_sim_state: SimState
) -> None:
    """Test the __getitem__ method of SimState."""
    # Create a concatenated state
    states = [si_sim_state, ar_sim_state, fe_fcc_sim_state]
    concatenated = concatenate_states(states)

    # Test integer indexing
    single_state = concatenated[1]
    assert isinstance(single_state, SimState)
    assert torch.allclose(single_state.positions, ar_sim_state.positions)
    assert single_state.n_batches == 1

    # Test list indexing
    multi_state = concatenated[[0, 2]]
    assert isinstance(multi_state, SimState)
    assert multi_state.n_batches == 2
    assert torch.allclose(multi_state[0].positions, si_sim_state.positions)
    assert torch.allclose(multi_state[1].positions, fe_fcc_sim_state.positions)

    # Test slice indexing
    slice_state = concatenated[1:3]
    assert isinstance(slice_state, SimState)
    assert slice_state.n_batches == 2
    assert torch.allclose(slice_state[0].positions, ar_sim_state.positions)
    assert torch.allclose(slice_state[1].positions, fe_fcc_sim_state.positions)

    # Test negative indexing
    neg_state = concatenated[-1]
    assert isinstance(neg_state, SimState)
    assert torch.allclose(neg_state.positions, fe_fcc_sim_state.positions)

    # Test step in slice
    step_state = concatenated[::2]
    assert isinstance(step_state, SimState)
    assert step_state.n_batches == 2
    assert torch.allclose(step_state[0].positions, si_sim_state.positions)
    assert torch.allclose(step_state[1].positions, fe_fcc_sim_state.positions)

    full_state = concatenated[:]
    assert torch.allclose(full_state.positions, concatenated.positions)
    # Verify original state is unchanged
    assert concatenated.n_batches == 3


def test_normalize_batch_indices(si_double_sim_state: SimState) -> None:
    """Test the _normalize_batch_indices utility method."""
    state = si_double_sim_state  # State with 2 batches
    n_batches = state.n_batches
    device = state.device

    # Test integer indexing
    assert _normalize_batch_indices(0, n_batches, device).tolist() == [0]
    assert _normalize_batch_indices(1, n_batches, device).tolist() == [1]

    # Test negative integer indexing
    assert _normalize_batch_indices(-1, n_batches, device).tolist() == [1]
    assert _normalize_batch_indices(-2, n_batches, device).tolist() == [0]

    # Test list indexing
    assert _normalize_batch_indices([0, 1], n_batches, device).tolist() == [0, 1]

    # Test list with negative indices
    assert _normalize_batch_indices([0, -1], n_batches, device).tolist() == [0, 1]
    assert _normalize_batch_indices([-2, -1], n_batches, device).tolist() == [0, 1]

    # Test slice indexing
    indices = _normalize_batch_indices(slice(0, 2), n_batches, device)
    assert isinstance(indices, torch.Tensor)
    assert torch.all(indices == torch.tensor([0, 1], device=state.device))

    # Test slice with negative indices
    indices = _normalize_batch_indices(slice(-2, None), n_batches, device)
    assert isinstance(indices, torch.Tensor)
    assert torch.all(indices == torch.tensor([0, 1], device=state.device))

    # Test slice with step
    indices = _normalize_batch_indices(slice(0, 2, 2), n_batches, device)
    assert isinstance(indices, torch.Tensor)
    assert torch.all(indices == torch.tensor([0], device=state.device))

    # Test tensor indexing
    tensor_indices = torch.tensor([0, 1], device=state.device)
    indices = _normalize_batch_indices(tensor_indices, n_batches, device)
    assert isinstance(indices, torch.Tensor)
    assert torch.all(indices == tensor_indices)

    # Test tensor with negative indices
    tensor_indices = torch.tensor([0, -1], device=state.device)
    indices = _normalize_batch_indices(tensor_indices, n_batches, device)
    assert isinstance(indices, torch.Tensor)
    assert torch.all(indices == torch.tensor([0, 1], device=state.device))

    # Test error for unsupported type
    try:
        _normalize_batch_indices((0, 1), n_batches, device)  # Tuple is not supported
        raise ValueError("Should have raised TypeError")
    except TypeError:
        pass
