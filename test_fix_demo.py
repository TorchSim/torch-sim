#!/usr/bin/env python3
"""
Demonstration script showing that the state_to_device side effect fix works correctly.

This script demonstrates that:
1. SimState.to() now creates a new state without modifying the original
2. concatenate_states no longer modifies input states
3. initialize_state no longer modifies input states
"""

import torch
import torch_sim as ts


def test_state_to_device_fix():
    """Test that SimState.to() doesn't modify the original state."""
    print("Testing SimState.to() side effect fix...")
    
    # Create a test state
    state = ts.SimState(
        positions=torch.randn(4, 3),
        masses=torch.ones(4),
        cell=torch.eye(3).unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.ones(4, dtype=torch.long)
    )
    
    # Store original values
    original_positions = state.positions.clone()
    original_dtype = state.dtype
    
    # Convert to different dtype
    new_state = state.to(dtype=torch.float64)
    
    # Verify original state is unchanged
    assert torch.allclose(state.positions, original_positions), "Original state was modified!"
    assert state.dtype == original_dtype, "Original state dtype was changed!"
    assert state is not new_state, "New state is not a different object!"
    assert new_state.dtype == torch.float64, "New state doesn't have correct dtype!"
    
    print("[OK] SimState.to() fix works correctly")


def test_concatenate_states_fix():
    """Test that concatenate_states doesn't modify input states."""
    print("Testing concatenate_states side effect fix...")
    
    # Create two test states
    state1 = ts.SimState(
        positions=torch.randn(4, 3),
        masses=torch.ones(4),
        cell=torch.eye(3).unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.ones(4, dtype=torch.long)
    )
    
    state2 = ts.SimState(
        positions=torch.randn(6, 3),
        masses=torch.ones(6),
        cell=torch.eye(3).unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.ones(6, dtype=torch.long)
    )
    
    # Store original values
    original_positions1 = state1.positions.clone()
    original_positions2 = state2.positions.clone()
    
    # Concatenate states
    concatenated = ts.concatenate_states([state1, state2])
    
    # Verify original states are unchanged
    assert torch.allclose(state1.positions, original_positions1), "State1 was modified!"
    assert torch.allclose(state2.positions, original_positions2), "State2 was modified!"
    assert concatenated.n_atoms == 10, "Concatenated state has wrong number of atoms!"
    
    print("[OK] concatenate_states fix works correctly")


def test_initialize_state_fix():
    """Test that initialize_state doesn't modify input state."""
    print("Testing initialize_state side effect fix...")
    
    # Create a test state
    original_state = ts.SimState(
        positions=torch.randn(4, 3),
        masses=torch.ones(4),
        cell=torch.eye(3).unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.ones(4, dtype=torch.long)
    )
    
    # Store original values
    original_positions = original_state.positions.clone()
    original_dtype = original_state.dtype
    
    # Initialize from existing state
    new_state = ts.initialize_state(original_state, torch.device('cpu'), torch.float64)
    
    # Verify original state is unchanged
    assert torch.allclose(original_state.positions, original_positions), "Original state was modified!"
    assert original_state.dtype == original_dtype, "Original state dtype was changed!"
    assert original_state is not new_state, "New state is not a different object!"
    assert new_state.dtype == torch.float64, "New state doesn't have correct dtype!"
    
    print("[OK] initialize_state fix works correctly")


if __name__ == "__main__":
    print("Demonstrating state_to_device side effect fix (Issue #293)")
    print("=" * 60)
    
    test_state_to_device_fix()
    test_concatenate_states_fix()
    test_initialize_state_fix()
    
    print("=" * 60)
    print("All tests passed! The fix successfully resolves the side effect issues.")
