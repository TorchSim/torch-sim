from dataclasses import asdict

import torch

from torch_sim.state import (
    BaseState,
    concatenate_states,
    infer_property_scope,
    pop_states,
    slice_substate,
    split_state,
)
from torch_sim.unbatched.unbatched_integrators import MDState


def test_infer_base_state_property_scope(si_base_state: BaseState) -> None:
    """Test inference of property scope."""
    scope = infer_property_scope(si_base_state)
    assert set(scope["global"]) == {"pbc"}
    assert set(scope["per_atom"]) == {"positions", "masses", "atomic_numbers", "batch"}
    assert set(scope["per_batch"]) == {"cell"}


def test_infer_md_state_property_scope(si_base_state: BaseState) -> None:
    """Test inference of property scope."""
    state = MDState(
        **asdict(si_base_state),
        momenta=torch.randn_like(si_base_state.positions),
        forces=torch.randn_like(si_base_state.positions),
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


def test_slice_substate(
    si_double_base_state: BaseState, si_base_state: BaseState
) -> None:
    """Test slicing a substate from the BaseState."""
    for batch_index in range(2):
        substate = slice_substate(si_double_base_state, batch_index)
        assert isinstance(substate, BaseState)
        assert substate.positions.shape == (8, 3)
        assert substate.masses.shape == (8,)
        assert substate.cell.shape == (1, 3, 3)
        assert torch.allclose(substate.positions, si_base_state.positions)
        assert torch.allclose(substate.masses, si_base_state.masses)
        assert torch.allclose(substate.cell, si_base_state.cell)
        assert torch.allclose(substate.atomic_numbers, si_base_state.atomic_numbers)
        assert torch.allclose(substate.batch, torch.zeros_like(substate.batch))


def test_slice_md_substate(si_double_base_state: BaseState) -> None:
    state = MDState(
        **asdict(si_double_base_state),
        momenta=torch.randn_like(si_double_base_state.positions),
        energy=torch.zeros((2,)),
        forces=torch.randn_like(si_double_base_state.positions),
    )
    for batch_index in range(2):
        substate = slice_substate(state, batch_index)
        assert isinstance(substate, MDState)
        assert substate.positions.shape == (8, 3)
        assert substate.masses.shape == (8,)
        assert substate.cell.shape == (1, 3, 3)
        assert substate.momenta.shape == (8, 3)
        assert substate.forces.shape == (8, 3)
        assert substate.energy.shape == (1,)


def test_concatenate_two_si_states(
    si_base_state: BaseState, si_double_base_state: BaseState
) -> None:
    """Test concatenating two identical silicon states."""
    # Concatenate two copies of the base state
    concatenated = concatenate_states([si_base_state, si_base_state])

    # Check that the result is the same as the double state
    assert isinstance(concatenated, BaseState)
    assert concatenated.positions.shape == si_double_base_state.positions.shape
    assert concatenated.masses.shape == si_double_base_state.masses.shape
    assert concatenated.cell.shape == si_double_base_state.cell.shape
    assert concatenated.atomic_numbers.shape == si_double_base_state.atomic_numbers.shape
    assert concatenated.batch.shape == si_double_base_state.batch.shape

    # Check batch indices
    expected_batch = torch.cat(
        [
            torch.zeros(
                si_base_state.n_atoms, dtype=torch.int64, device=si_base_state.device
            ),
            torch.ones(
                si_base_state.n_atoms, dtype=torch.int64, device=si_base_state.device
            ),
        ]
    )
    assert torch.all(concatenated.batch == expected_batch)

    # Check that positions match (accounting for batch indices)
    for batch_idx in range(2):
        mask_concat = concatenated.batch == batch_idx
        mask_double = si_double_base_state.batch == batch_idx
        assert torch.allclose(
            concatenated.positions[mask_concat],
            si_double_base_state.positions[mask_double],
        )


def test_concatenate_si_and_fe_states(
    si_base_state: BaseState, fe_fcc_state: BaseState
) -> None:
    """Test concatenating silicon and argon states."""
    # Concatenate silicon and argon states
    concatenated = concatenate_states([si_base_state, fe_fcc_state])

    # Check basic properties
    assert isinstance(concatenated, BaseState)
    assert (
        concatenated.positions.shape[0]
        == si_base_state.positions.shape[0] + fe_fcc_state.positions.shape[0]
    )
    assert (
        concatenated.masses.shape[0]
        == si_base_state.masses.shape[0] + fe_fcc_state.masses.shape[0]
    )
    assert concatenated.cell.shape[0] == 2  # One cell per batch

    # Check batch indices
    si_atoms = si_base_state.n_atoms
    fe_atoms = fe_fcc_state.n_atoms
    expected_batch = torch.cat(
        [
            torch.zeros(si_atoms, dtype=torch.int64, device=si_base_state.device),
            torch.ones(fe_atoms, dtype=torch.int64, device=fe_fcc_state.device),
        ]
    )
    assert torch.all(concatenated.batch == expected_batch)

    # Check that positions match for each original state
    assert torch.allclose(concatenated.positions[:si_atoms], si_base_state.positions)
    assert torch.allclose(concatenated.positions[si_atoms:], fe_fcc_state.positions)

    # Check that atomic numbers are correct
    assert torch.all(concatenated.atomic_numbers[:si_atoms] == 14)  # Si
    assert torch.all(concatenated.atomic_numbers[si_atoms:] == 26)  # Fe


def test_concatenate_double_si_and_fe_states(
    si_double_base_state: BaseState, fe_fcc_state: BaseState
) -> None:
    """Test concatenating a double silicon state and an argon state."""
    # Concatenate double silicon and argon states
    concatenated = concatenate_states([si_double_base_state, fe_fcc_state])

    # Check basic properties
    assert isinstance(concatenated, BaseState)
    assert (
        concatenated.positions.shape[0]
        == si_double_base_state.positions.shape[0] + fe_fcc_state.positions.shape[0]
    )
    assert (
        concatenated.cell.shape[0] == 3
    )  # One cell for each original batch (2 Si + 1 Ar)

    # Check batch indices
    fe_atoms = fe_fcc_state.n_atoms

    # The double Si state already has batches 0 and 1, so Ar should be batch 2
    expected_batch = torch.cat(
        [
            si_double_base_state.batch,
            torch.full((fe_atoms,), 2, dtype=torch.int64, device=fe_fcc_state.device),
        ]
    )
    assert torch.all(concatenated.batch == expected_batch)
    assert torch.unique(concatenated.batch).shape[0] == 3

    # Check that we can slice back to the original states
    si_slice_0 = slice_substate(concatenated, 0)
    si_slice_1 = slice_substate(concatenated, 1)
    fe_slice = slice_substate(concatenated, 2)

    # Check that the slices match the original states
    assert torch.allclose(
        si_slice_0.positions, slice_substate(si_double_base_state, 0).positions
    )
    assert torch.allclose(
        si_slice_1.positions, slice_substate(si_double_base_state, 1).positions
    )
    assert torch.allclose(fe_slice.positions, fe_fcc_state.positions)


def test_split_state(si_double_base_state: BaseState) -> None:
    """Test splitting a state into a list of states."""
    states = split_state(si_double_base_state)
    assert len(states) == si_double_base_state.n_batches
    for state in states:
        assert isinstance(state, BaseState)
        assert state.positions.shape == (8, 3)
        assert state.masses.shape == (8,)
        assert state.cell.shape == (1, 3, 3)
        assert state.atomic_numbers.shape == (8,)
        assert torch.allclose(state.batch, torch.zeros_like(state.batch))


def test_split_many_states(
    si_base_state: BaseState, ar_base_state: BaseState, fe_fcc_state: BaseState
) -> None:
    """Test splitting a state into a list of states."""
    states = [si_base_state, ar_base_state, fe_fcc_state]
    concatenated = concatenate_states(states)
    split_states = split_state(concatenated)
    for state, sub_state in zip(states, split_states, strict=True):
        assert isinstance(sub_state, BaseState)
        assert torch.allclose(sub_state.positions, state.positions)
        assert torch.allclose(sub_state.masses, state.masses)
        assert torch.allclose(sub_state.cell, state.cell)
        assert torch.allclose(sub_state.atomic_numbers, state.atomic_numbers)
        assert torch.allclose(sub_state.batch, state.batch)

    assert len(states) == 3


def test_pop_states(
    si_base_state: BaseState, ar_base_state: BaseState, fe_fcc_state: BaseState
) -> None:
    """Test popping states from a state."""
    states = [si_base_state, ar_base_state, fe_fcc_state]
    concatenated_states = concatenate_states(states)
    kept_state, popped_states = pop_states(concatenated_states, torch.tensor([0]))

    assert isinstance(kept_state, BaseState)
    assert isinstance(popped_states, list)
    assert len(popped_states) == 1
    assert isinstance(popped_states[0], BaseState)
    assert popped_states[0].positions.shape == si_base_state.positions.shape

    len_kept = ar_base_state.n_atoms + fe_fcc_state.n_atoms
    assert kept_state.positions.shape == (len_kept, 3)
    assert kept_state.masses.shape == (len_kept,)
    assert kept_state.cell.shape == (2, 3, 3)
    assert kept_state.atomic_numbers.shape == (len_kept,)
    assert kept_state.batch.shape == (len_kept,)
