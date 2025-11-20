"""Test vesin fallback functionality for ROCm/AMD GPU support."""

import pytest
import torch

from torch_sim import neighbors


def test_vesin_availability_flag() -> None:
    """Test that VESIN_AVAILABLE flag is correctly set."""
    assert isinstance(neighbors.VESIN_AVAILABLE, bool)
    if neighbors.VESIN_AVAILABLE:
        assert neighbors.VesinNeighborList is not None
        assert neighbors.VesinNeighborListTorch is not None
    else:
        assert neighbors.VesinNeighborList is None
        assert neighbors.VesinNeighborListTorch is None


def test_fallback_consistency() -> None:
    """Test that fallback implementation produces consistent results."""
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

    # All three functions should give same result
    mapping_standard, shifts_standard = neighbors.standard_nl(
        positions, cell, pbc, cutoff
    )
    mapping_vesin, shifts_vesin = neighbors.vesin_nl(positions, cell, pbc, cutoff)
    mapping_vesin_ts, shifts_vesin_ts = neighbors.vesin_nl_ts(
        positions, cell, pbc, cutoff
    )

    # When vesin is unavailable, vesin_nl and vesin_nl_ts should match standard_nl
    if not neighbors.VESIN_AVAILABLE:
        torch.testing.assert_close(mapping_vesin, mapping_standard)
        torch.testing.assert_close(shifts_vesin, shifts_standard)
        torch.testing.assert_close(mapping_vesin_ts, mapping_standard)
        torch.testing.assert_close(shifts_vesin_ts, shifts_standard)

    # All should have same shape regardless
    assert mapping_vesin.shape == mapping_standard.shape
    assert mapping_vesin_ts.shape == mapping_standard.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available for testing")
def test_fallback_gpu_compatibility() -> None:
    """Test that fallback works on GPU (CUDA/ROCm)."""
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

    # Should work on GPU regardless of vesin availability
    mapping, shifts = neighbors.vesin_nl(positions, cell, pbc, cutoff)

    assert mapping.device.type == "cuda"
    assert shifts.device.type == "cuda"
    assert mapping.shape[0] == 2  # (2, num_neighbors)
