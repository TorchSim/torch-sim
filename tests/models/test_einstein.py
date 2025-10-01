"""Tests for the Einstein model implementation."""

import pytest
import torch
from ase.build import bulk

import torch_sim as ts
from torch_sim.models.einstein import EinsteinModel


class TestEinsteinModel:
    """Test Einstein model implementation."""

    @pytest.fixture
    def simple_system(self):
        """Create a simple test system."""
        device = torch.device("cpu")
        dtype = torch.float64

        # Create a simple 2-atom system
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype, device=device
        )
        masses = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
        cell = torch.eye(3, dtype=dtype, device=device) * 10.0  # Large cell
        atomic_numbers = torch.tensor([1, 1], dtype=torch.int64, device=device)

        state = ts.SimState(
            positions=positions,
            masses=masses,
            cell=cell.unsqueeze(0),
            pbc=True,
            atomic_numbers=atomic_numbers,
        )

        return state, device, dtype

    @pytest.fixture
    def batched_system(self):
        """Create a batched test system."""
        device = torch.device("cpu")
        dtype = torch.float64

        # Create two different 2-atom systems
        si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
        fe_atoms = bulk("Fe", "bcc", a=2.87, cubic=True)

        state = ts.io.atoms_to_state([si_atoms, fe_atoms], device, dtype)
        return state, device, dtype

    def test_einstein_model_creation(
        self, simple_system: tuple[ts.SimState, torch.device, torch.dtype]
    ):
        """Test basic Einstein model creation."""
        state, device, dtype = simple_system

        # Create equilibrium positions and frequencies
        equilibrium_pos = state.positions
        frequencies = torch.ones(2, dtype=dtype, device=device) * 0.1  # [N]

        model = EinsteinModel(
            equilibrium_position=equilibrium_pos,
            frequencies=frequencies,
            masses=state.masses,
            device=device,
            dtype=dtype,
        )

        assert model.device == device
        assert model.dtype == dtype
        assert model.compute_forces is True

    def test_einstein_model_forward_single_system(
        self, simple_system: tuple[ts.SimState, torch.device, torch.dtype]
    ):
        """Test forward pass with single system."""
        state, device, dtype = simple_system

        equilibrium_pos = state.positions
        frequencies = torch.ones(2, dtype=dtype, device=device) * 0.1

        model = EinsteinModel(
            equilibrium_position=equilibrium_pos,
            frequencies=frequencies,
            masses=state.masses,
            device=device,
            dtype=dtype,
        )

        # Displace atoms slightly from equilibrium
        displaced_state = state.clone()
        displaced_state.positions += 0.1

        results = model(displaced_state)

        assert "energy" in results
        assert "forces" in results
        assert results["energy"].shape == (1,)
        assert results["forces"].shape == (2, 3)  # [N_atoms, 3]

        # Forces should point back toward equilibrium
        expected_force_direction = -(displaced_state.positions - equilibrium_pos)
        force_directions = results["forces"]
        # Check that forces point in the right direction (dot product > 0)
        for i in range(2):
            dot_product = torch.dot(force_directions[i], expected_force_direction[i])
            assert dot_product > 0

    def test_einstein_model_forward_batched_system(
        self, batched_system: tuple[ts.SimState, torch.device, torch.dtype]
    ):
        """Test forward pass with batched system."""
        state, device, dtype = batched_system

        # Create equilibrium positions for the batched system
        n_atoms = state.n_atoms
        equilibrium_pos = state.positions
        frequencies = torch.ones(n_atoms, dtype=dtype, device=device) * 0.05

        model = EinsteinModel(
            equilibrium_position=equilibrium_pos,
            frequencies=frequencies,
            masses=state.masses,
            device=device,
            dtype=dtype,
        )

        # Displace atoms slightly
        displaced_state = state.clone()
        displaced_state.positions += 0.05

        results = model(displaced_state)

        assert "energy" in results
        assert "forces" in results
        assert results["energy"].shape == (2,)  # [n_systems]
        assert results["forces"].shape == (n_atoms, 3)  # [total_atoms, 3]

    def test_einstein_model_from_frequencies(self):
        """Test creation from frequencies class method."""
        device = torch.device("cpu")
        dtype = torch.float64

        # Create ASE atoms
        atoms = bulk("Si", "diamond", a=5.43, cubic=True)
        state = ts.io.atoms_to_state([atoms], device, dtype)

        frequencies = torch.ones(len(atoms), dtype=dtype, device=device) * 0.05

        model = EinsteinModel.from_atom_and_frequencies(
            atom=state,
            frequencies=frequencies,
            reference_energy=1.0,
            device=device,
            dtype=dtype,
        )

        assert torch.allclose(model.reference_energy, torch.tensor(1.0, dtype=dtype))
        assert model.frequencies.shape[0] == len(atoms)

    def test_periodic_boundary_conditions(
        self, simple_system: tuple[ts.SimState, torch.device, torch.dtype]
    ):
        """Test that PBC are handled correctly."""
        state, device, dtype = simple_system

        # Create model with equilibrium at origin
        equilibrium_pos = torch.zeros((2, 3), dtype=dtype, device=device)
        frequencies = torch.ones(2, dtype=dtype, device=device) * 1

        model = EinsteinModel(
            equilibrium_position=equilibrium_pos,
            frequencies=frequencies,
            device=device,
            dtype=dtype,
        )

        # Place atoms near opposite faces of the cell
        test_state = state.clone()
        test_state.positions = torch.tensor(
            [
                [0.1, 0.0, 0.0],  # Near one face
                [9.9, 0.0, 0.0],  # Near opposite face
            ],
            dtype=dtype,
            device=device,
        )

        results = model(test_state)

        # Should handle PBC correctly - both atoms far from origin
        # but should compute minimum image distances
        assert torch.isfinite(results["energy"])
        assert torch.isfinite(results["forces"]).all()

        spring = frequencies**2  # since mass=1
        target_energies = 0.5 * spring * (torch.tensor([0.1, -0.1]) ** 2)
        target_forces = -spring[:, None] * torch.tensor(
            [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=dtype, device=device
        )
        assert torch.allclose(results["energy"], target_energies.sum(), atol=1e-6)
        assert torch.allclose(results["forces"], target_forces, atol=1e-6)

    def test_energy_force_consistency(
        self, simple_system: tuple[ts.SimState, torch.device, torch.dtype]
    ):
        """Test that forces are consistent with energy gradients."""
        state, device, dtype = simple_system

        equilibrium_pos = state.positions.clone()
        frequencies = torch.ones(2, dtype=dtype, device=device) * 0.1

        model = EinsteinModel(
            equilibrium_position=equilibrium_pos,
            frequencies=frequencies,
            masses=state.masses,
            device=device,
            dtype=dtype,
        )

        # Create a displaced state with gradients enabled
        test_positions = state.positions.clone() + 0.1
        test_positions.requires_grad_(requires_grad=True)

        test_state = state.clone()
        test_state.positions = test_positions

        results = model(test_state)
        energy = results["energy"]

        # Compute forces from gradients
        forces_from_grad = -torch.autograd.grad(
            energy, test_positions, create_graph=False
        )[0]
        forces_direct = results["forces"]

        # Forces should match (within numerical precision)
        torch.testing.assert_close(forces_direct, forces_from_grad, atol=1e-6, rtol=1e-6)

    def test_get_free_energy(
        self, simple_system: tuple[ts.SimState, torch.device, torch.dtype]
    ):
        """Test free energy calculation."""
        state, device, dtype = simple_system

        equilibrium_pos = state.positions
        frequencies = torch.ones(2, dtype=dtype, device=device) * 0.1  # THz

        model = EinsteinModel(
            equilibrium_position=equilibrium_pos,
            frequencies=frequencies,
            masses=state.masses,
            device=device,
            dtype=dtype,
        )

        temperature = 300.0  # K
        results = model.get_free_energy(temperature)

        # Check that result is a dictionary with free energy
        assert isinstance(results, dict)
        assert "free_energy" in results
        free_energy = results["free_energy"]
        assert isinstance(free_energy, torch.Tensor)
        assert free_energy.shape == (1,)  # Single system

        # Free energy should be finite
        assert torch.isfinite(free_energy).all()

        # At higher temperature, free energy should be lower (more negative)
        results_high = model.get_free_energy(600.0)
        free_energy_high = results_high["free_energy"]
        assert free_energy_high < free_energy

    def test_get_free_energy_batched(
        self, batched_system: tuple[ts.SimState, torch.device, torch.dtype]
    ):
        """Test free energy calculation for batched systems."""
        state, device, dtype = batched_system

        n_atoms = state.n_atoms
        equilibrium_pos = state.positions
        frequencies = torch.ones(n_atoms, dtype=dtype, device=device) * 0.05

        model = EinsteinModel(
            equilibrium_position=equilibrium_pos,
            frequencies=frequencies,
            masses=state.masses,
            system_idx=state.system_idx,
            device=device,
            dtype=dtype,
        )

        temperature = 300.0
        results = model.get_free_energy(temperature)

        # Check result format and shape
        assert isinstance(results, dict)
        assert "free_energy" in results
        free_energy = results["free_energy"]

        # Should have one free energy per system
        assert free_energy.shape == (2,)  # Two systems
        assert torch.isfinite(free_energy).all()

    def test_sample_method(
        self, simple_system: tuple[ts.SimState, torch.device, torch.dtype]
    ):
        """Test sampling from Einstein model."""
        state, device, dtype = simple_system

        equilibrium_pos = state.positions
        frequencies = torch.ones(2, dtype=dtype, device=device) * 0.1

        model = EinsteinModel(
            equilibrium_position=equilibrium_pos,
            frequencies=frequencies,
            masses=state.masses,
            device=device,
            dtype=dtype,
        )

        temperature = 300.0
        sampled_state = model.sample(state, temperature)

        # Check that sampled state has correct shape and type
        assert isinstance(sampled_state, ts.SimState)
        assert sampled_state.positions.shape == state.positions.shape
        assert sampled_state.positions.dtype == dtype
        assert sampled_state.positions.device == device

        # Sampled positions should have finite values
        assert torch.isfinite(sampled_state.positions).all()
