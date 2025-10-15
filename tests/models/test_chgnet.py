import traceback

import pytest
import torch
from ase.atoms import Atoms

import torch_sim as ts
from tests.conftest import DEVICE
from tests.models.conftest import (
    make_validate_model_outputs_test,
)


try:
    from chgnet.model.model import CHGNet
    from torch_sim.models.chgnet import CHGNetModel
except (ImportError, ValueError):
    pytest.skip(f"CHGNet not installed: {traceback.format_exc()}", allow_module_level=True)


DTYPE = torch.float32


@pytest.fixture
def ts_chgnet_model() -> CHGNetModel:
    """Create a TorchSim CHGNet model for testing."""
    return CHGNetModel(
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_chgnet_dtype_working(si_atoms: Atoms, dtype: torch.dtype) -> None:
    """Test that CHGNet works with different dtypes."""
    model = CHGNetModel(
        device=DEVICE,
        dtype=dtype,
        compute_forces=True,
    )

    state = ts.io.atoms_to_state([si_atoms], DEVICE, dtype)
    result = model.forward(state)
    
    # Check that results have correct dtype
    assert result["energy"].dtype == dtype
    assert result["forces"].dtype == dtype
    assert result["stress"].dtype == dtype


def test_chgnet_batched_calculations() -> None:
    """Test that CHGNet handles batched calculations correctly."""
    from ase.build import bulk
    
    # Create multiple systems
    si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    cu_atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    
    model = CHGNetModel(
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )
    
    # Test batched calculation
    state = ts.io.atoms_to_state([si_atoms, cu_atoms], DEVICE, DTYPE)
    result = model.forward(state)
    
    # Check output shapes
    assert result["energy"].shape == (2,)
    assert result["forces"].shape == (si_atoms.get_global_number_of_atoms() + cu_atoms.get_global_number_of_atoms(), 3)
    assert result["stress"].shape == (2, 3, 3)
    
    # Check that energies are different (different materials)
    assert not torch.allclose(result["energy"][0], result["energy"][1], atol=1e-3)


def test_chgnet_single_vs_batched_consistency() -> None:
    """Test that single and batched calculations give consistent results."""
    from ase.build import bulk
    
    si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    
    model = CHGNetModel(
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )
    
    # Single system calculation
    single_state = ts.io.atoms_to_state([si_atoms], DEVICE, DTYPE)
    single_result = model.forward(single_state)
    
    # Batched calculation (same system twice)
    batched_state = ts.io.atoms_to_state([si_atoms, si_atoms], DEVICE, DTYPE)
    batched_result = model.forward(batched_state)
    
    # Check consistency
    assert torch.allclose(single_result["energy"][0], batched_result["energy"][0], atol=1e-5)
    assert torch.allclose(single_result["energy"][0], batched_result["energy"][1], atol=1e-5)
    assert torch.allclose(single_result["forces"], batched_result["forces"][:single_state.n_atoms], atol=1e-5)
    assert torch.allclose(single_result["forces"], batched_result["forces"][single_state.n_atoms:], atol=1e-5)
    assert torch.allclose(single_result["stress"][0], batched_result["stress"][0], atol=1e-5)
    assert torch.allclose(single_result["stress"][0], batched_result["stress"][1], atol=1e-5)


def test_chgnet_missing_atomic_numbers() -> None:
    """Test that CHGNet raises appropriate error when atomic numbers are missing."""
    model = CHGNetModel(
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
    )
    
    # Create state without atomic numbers by using a state dict
    state_dict = {
        "positions": torch.randn(8, 3, device=DEVICE, dtype=DTYPE),
        "cell": torch.eye(3, device=DEVICE, dtype=DTYPE).unsqueeze(0),
        "pbc": True,
        "atomic_numbers": None,  # Missing atomic numbers
        "system_idx": torch.zeros(8, dtype=torch.long, device=DEVICE),
    }
    
    with pytest.raises(ValueError, match="Atomic numbers must be provided"):
        model.forward(state_dict)


def test_chgnet_custom_model() -> None:
    """Test that CHGNet can be initialized with a custom model."""
    # Load a custom model instance
    custom_model = CHGNet.load()
    
    model = CHGNetModel(
        model=custom_model,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
    )
    
    from ase.build import bulk
    si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    state = ts.io.atoms_to_state([si_atoms], DEVICE, DTYPE)
    
    result = model.forward(state)
    assert "energy" in result
    assert "forces" in result
    assert "stress" in result


def test_chgnet_compute_forces_false() -> None:
    """Test CHGNet with compute_forces=False."""
    model = CHGNetModel(
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=False,
        compute_stress=True,
    )
    
    from ase.build import bulk
    si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    state = ts.io.atoms_to_state([si_atoms], DEVICE, DTYPE)
    
    result = model.forward(state)
    assert "energy" in result
    assert "forces" not in result
    assert "stress" in result


def test_chgnet_compute_stress_false() -> None:
    """Test CHGNet with compute_stress=False."""
    model = CHGNetModel(
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=False,
    )
    
    from ase.build import bulk
    si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    state = ts.io.atoms_to_state([si_atoms], DEVICE, DTYPE)
    
    result = model.forward(state)
    assert "energy" in result
    assert "forces" in result
    assert "stress" not in result


def test_chgnet_compute_both_false() -> None:
    """Test CHGNet with both compute_forces=False and compute_stress=False."""
    model = CHGNetModel(
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=False,
        compute_stress=False,
    )
    
    from ase.build import bulk
    si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    state = ts.io.atoms_to_state([si_atoms], DEVICE, DTYPE)
    
    result = model.forward(state)
    assert "energy" in result
    assert "forces" not in result
    assert "stress" not in result


test_chgnet_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="ts_chgnet_model", dtype=DTYPE
)


def test_chgnet_import_error() -> None:
    """Test that CHGNetModel raises ImportError when CHGNet is not available."""
    from torch_sim.models.chgnet import CHGNetModel
    
    # Should not raise an error when CHGNet is available
    model = CHGNetModel(device=DEVICE, dtype=DTYPE)
    assert isinstance(model, CHGNetModel)
