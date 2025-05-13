import numpy as np
import torch
from ase.stress import full_3x3_to_voigt_6_stress as ase_full_3x3_to_voigt_6_stress
from ase.stress import voigt_6_to_full_3x3_stress as ase_voigt_6_to_full_3x3_stress

from torch_sim.elastic import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress


def test_voigt_to_full_basic():
    # Test with simple input
    voigt = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = voigt_6_to_full_3x3_stress(voigt)

    expected = torch.tensor([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])

    assert torch.allclose(result, expected)


def test_full_to_voigt_basic():
    # Test with simple input
    stress = torch.tensor([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])

    result = full_3x3_to_voigt_6_stress(stress)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert torch.allclose(result, expected)


def test_roundtrip_conversion():
    test_cases = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # simple case
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # zeros
        [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0],  # mixed signs
        [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],  # small values
        [1e3, 1e3, 1e3, 1e3, 1e3, 1e3],  # large values
    ]

    # Test that converting from Voigt to full and back gives original tensor
    for voigt in test_cases:
        voigt_original = torch.tensor(voigt)
        full = voigt_6_to_full_3x3_stress(voigt_original)
        voigt_result = full_3x3_to_voigt_6_stress(full)

        assert torch.allclose(voigt_original, voigt_result)


def test_batch_conversion():
    # Test with batched input
    batch_voigt = torch.rand(2, 6)  # 2 batch size
    result = voigt_6_to_full_3x3_stress(batch_voigt)

    assert result.shape == (2, 3, 3)

    # Test each batch independently
    for i in range(2):
        single_result = voigt_6_to_full_3x3_stress(batch_voigt[i])
        assert torch.allclose(result[i], single_result)


def test_symmetry():
    # Test that the resulting stress tensor is symmetric
    voigt = torch.rand(6)
    full = voigt_6_to_full_3x3_stress(voigt)

    assert torch.allclose(full, full.transpose(-2, -1))


def test_device_dtype_preservation():
    if torch.cuda.is_available():
        # Test device preservation
        voigt_cuda = torch.rand(6, device="cuda")
        result_cuda = voigt_6_to_full_3x3_stress(voigt_cuda)
        assert result_cuda.device.type == "cuda"

    # Test dtype preservation
    voigt_double = torch.rand(6, dtype=torch.float64)
    result_double = voigt_6_to_full_3x3_stress(voigt_double)
    assert result_double.dtype == torch.float64


def test_nonsymmetric_input():
    # Test that full_3x3_to_voigt_6_stress properly symmetrizes non-symmetric input
    nonsym_stress = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    result = full_3x3_to_voigt_6_stress(nonsym_stress)

    # The function should average corresponding off-diagonal elements
    expected = torch.tensor(
        [
            1.0,  # xx
            5.0,  # yy
            9.0,  # zz
            7.0,  # yz = (6 + 8)/2
            5.0,  # xz = (3 + 7)/2
            3.0,  # xy = (2 + 4)/2
        ]
    )

    assert torch.allclose(result, expected)


def test_against_ase_implementation():
    # Test various inputs against ASE implementation
    test_cases = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # simple case
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # zeros
        [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0],  # mixed signs
        [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],  # small values
        [1e3, 1e3, 1e3, 1e3, 1e3, 1e3],  # large values
    ]

    for voigt in test_cases:
        # Convert to appropriate types
        voigt_torch = torch.tensor(voigt, dtype=torch.float64)
        voigt_numpy = np.array(voigt, dtype=np.float64)

        # Get results from both implementations
        torch_result = voigt_6_to_full_3x3_stress(voigt_torch)
        ase_result = ase_voigt_6_to_full_3x3_stress(voigt_numpy)

        # Convert torch result to numpy for comparison
        torch_result_np = torch_result.numpy()

        # Compare results
        assert np.allclose(torch_result_np, ase_result), (
            f"Mismatch for input {voigt}:\nTorch:\n{torch_result_np}\nASE:\n{ase_result}"
        )


def test_against_ase_implementation_3x3_to_voigt():
    # Test various inputs against ASE implementation
    test_cases = [
        [[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]],  # simple case
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # zeros
        [[-1.0, 6.0, -5.0], [6.0, 2.0, -4.0], [-5.0, -4.0, -3.0]],  # mixed signs
        [[1e-3, 1e-3, 1e-3], [1e-3, 1e-3, 1e-3], [1e-3, 1e-3, 1e-3]],  # small values
        [[1e3, 1e3, 1e3], [1e3, 1e3, 1e3], [1e3, 1e3, 1e3]],  # large values
    ]

    for stress in test_cases:
        # Convert to appropriate types
        stress_torch = torch.tensor(stress, dtype=torch.float64)
        stress_numpy = np.array(stress, dtype=np.float64)

        # Get results from both implementations
        torch_result = full_3x3_to_voigt_6_stress(stress_torch)
        ase_result = ase_full_3x3_to_voigt_6_stress(stress_numpy)

        # Convert torch result to numpy for comparison
        torch_result_np = torch_result.numpy()

        # Compare results
        assert np.allclose(torch_result_np, ase_result), (
            f"Mismatch for input {stress}:\nTorch:\n{torch_result_np}\nASE:\n{ase_result}"
        )


def test_arbitrary_batch_dimensions():
    # Test both conversion functions with arbitrary batch dimensions

    # Test with multiple batch dimensions (2, 3, 6) -> (2, 3, 3, 3)
    batch_voigt = torch.rand(2, 3, 6)
    full_result = voigt_6_to_full_3x3_stress(batch_voigt)
    assert full_result.shape == (2, 3, 3, 3)

    # Test conversion back to Voigt
    voigt_result = full_3x3_to_voigt_6_stress(full_result)
    assert voigt_result.shape == (2, 3, 6)
    assert torch.allclose(batch_voigt, voigt_result)

    # Test with even more batch dimensions (4, 2, 3, 6) -> (4, 2, 3, 3, 3)
    complex_batch_voigt = torch.rand(4, 2, 3, 6)
    complex_full_result = voigt_6_to_full_3x3_stress(complex_batch_voigt)
    assert complex_full_result.shape == (4, 2, 3, 3, 3)

    # Test conversion back
    complex_voigt_result = full_3x3_to_voigt_6_stress(complex_full_result)
    assert complex_voigt_result.shape == (4, 2, 3, 6)
    assert torch.allclose(complex_batch_voigt, complex_voigt_result)

    # Verify that each batch element is processed correctly
    for i in range(4):
        for j in range(2):
            for k in range(3):
                single_voigt = complex_batch_voigt[i, j, k]
                single_full = voigt_6_to_full_3x3_stress(single_voigt)
                assert torch.allclose(single_full, complex_full_result[i, j, k])
