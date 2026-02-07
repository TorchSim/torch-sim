"""Tests for the math module. Adapted from https://github.com/abhijeetgangan/torch_matfunc"""


# ruff: noqa: SLF001

import numpy as np
import scipy
import torch
from numpy.testing import assert_allclose

import torch_sim.math as fm
from tests.conftest import DTYPE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestExpmFrechet:
    """Test suite for expm_frechet for 3x3 matrices."""

    def test_expm_frechet(self):
        """Test basic functionality of expm_frechet against scipy implementation."""
        A = np.array([[1, 2, 0], [5, 6, 0], [0, 0, 1]], dtype=np.float64)
        E = np.array([[3, 4, 0], [7, 8, 0], [0, 0, 0]], dtype=np.float64)
        expected_expm = scipy.linalg.expm(A)
        expected_frechet = scipy.linalg.expm_frechet(A, E)[1]

        A = torch.from_numpy(A).to(device=device)
        E = torch.from_numpy(E).to(device=device)
        observed_expm, observed_frechet = fm.expm_frechet(A, E)
        assert_allclose(expected_expm, observed_expm.cpu().numpy(), atol=1e-14)
        assert_allclose(expected_frechet, observed_frechet.cpu().numpy(), atol=1e-14)

    def test_small_norm_expm_frechet(self):
        """Test matrices with small norms."""
        A = np.array([[0.1, 0.2, 0], [0.5, 0.6, 0], [0, 0, 0.1]], dtype=np.float64)
        E = np.array([[0.3, 0.4, 0], [0.7, 0.8, 0], [0, 0, 0]], dtype=np.float64)
        expected_expm = scipy.linalg.expm(A)
        expected_frechet = scipy.linalg.expm_frechet(A, E)[1]

        A = torch.from_numpy(A).to(device=device, dtype=DTYPE)
        E = torch.from_numpy(E).to(device=device, dtype=DTYPE)
        observed_expm, observed_frechet = fm.expm_frechet(A, E)
        assert_allclose(expected_expm, observed_expm.cpu().numpy(), atol=1e-14)
        assert_allclose(expected_frechet, observed_frechet.cpu().numpy(), atol=1e-14)

    def test_fuzz(self):
        """Test with a variety of random 3x3 inputs to ensure robustness."""
        rng = np.random.default_rng(1726500908359153)
        ntests = 20
        for _ in range(ntests):
            target_norm_1 = rng.exponential()
            A_original = rng.standard_normal((3, 3))
            E_original = rng.standard_normal((3, 3))
            A_original_norm_1 = scipy.linalg.norm(A_original, 1)
            scale = target_norm_1 / A_original_norm_1
            A = scale * A_original
            E = scale * E_original
            expected_expm = scipy.linalg.expm(A)
            expected_frechet = scipy.linalg.expm_frechet(A, E)[1]
            A = torch.from_numpy(A).to(device=device, dtype=DTYPE)
            E = torch.from_numpy(E).to(device=device, dtype=DTYPE)
            observed_expm, observed_frechet = fm.expm_frechet(A, E)
            assert_allclose(expected_expm, observed_expm.cpu().numpy(), atol=5e-8)
            assert_allclose(expected_frechet, observed_frechet.cpu().numpy(), atol=1e-7)

    def test_large_norm_matrices(self):
        """Test expm_frechet with larger norm matrices requiring scaling."""
        A = np.array(
            [[1.5, 0.8, 0.3], [0.6, 1.2, 0.5], [0.4, 0.7, 1.8]], dtype=np.float64
        )
        E = np.array(
            [[0.1, 0.2, 0.1], [0.2, 0.15, 0.1], [0.1, 0.1, 0.2]], dtype=np.float64
        )
        expected_expm = scipy.linalg.expm(A)
        expected_frechet = scipy.linalg.expm_frechet(A, E)[1]

        A = torch.from_numpy(A).to(device=device, dtype=DTYPE)
        E = torch.from_numpy(E).to(device=device, dtype=DTYPE)
        observed_expm, observed_frechet = fm.expm_frechet(A, E)
        assert_allclose(expected_expm, observed_expm.cpu().numpy(), atol=1e-12)
        assert_allclose(expected_frechet, observed_frechet.cpu().numpy(), atol=1e-12)


class TestExpmFrechetTorch:
    """Test suite for expm_frechet using native torch tensors."""

    def test_expm_frechet(self):
        """Test basic functionality of expm_frechet against torch.linalg.matrix_exp."""
        A = torch.tensor([[1, 2, 0], [5, 6, 0], [0, 0, 1]], dtype=DTYPE, device=device)
        E = torch.tensor([[3, 4, 0], [7, 8, 0], [0, 0, 0]], dtype=DTYPE, device=device)
        expected_expm = torch.linalg.matrix_exp(A)

        observed_expm, _ = fm.expm_frechet(A, E)
        torch.testing.assert_close(expected_expm, observed_expm)

    def test_fuzz(self):
        """Test with a variety of random 3x3 inputs using torch tensors."""
        rng = np.random.default_rng(1726500908359153)
        ntests = 20
        for _ in range(ntests):
            target_norm_1 = rng.exponential()
            A_original = torch.tensor(rng.standard_normal((3, 3)), device=device)
            E_original = torch.tensor(rng.standard_normal((3, 3)), device=device)
            A_original_norm_1 = torch.linalg.norm(A_original, 1)
            scale = target_norm_1 / A_original_norm_1
            A = scale * A_original
            E = scale * E_original
            expected_expm = torch.linalg.matrix_exp(A)
            observed_expm, _ = fm.expm_frechet(A, E)
            torch.testing.assert_close(expected_expm, observed_expm, atol=5e-8, rtol=1e-5)


class TestLogM33:
    """Test suite for the 3x3 matrix logarithm implementation.

    This class contains tests that verify the correctness of the matrix logarithm
    implementation for 3x3 matrices against analytical solutions, scipy implementation,
    and various edge cases.
    """

    def test_logm_33_reference(self):
        """Test matrix logarithm implementation for 3x3 matrices
        against analytical solutions.

        Tests against scipy implementation as well.

        This test verifies the implementation against known analytical
        solutions from the paper:

        https://link.springer.com/article/10.1007/s10659-008-9169-x

        I test several cases:
        - Case 1b: All eigenvalues equal with q(T) = (T - λI)²
        - Case 1c: All eigenvalues equal with q(T) = (T - λI)³
        - Case 2b: Two distinct eigenvalues with q(T) = (T - μI)(T - λI)²
        - Identity matrix (should return zero matrix)
        - Diagonal matrix with distinct eigenvalues (Case 3)
        """
        # Set precision for comparisons
        rtol = 1e-5
        atol = 1e-8

        # Case 1b: All eigenvalues equal with q(T) = (T - λI)²
        # Example: T = [[e, 1, 0], [0, e, 0], [0, 0, e]]
        e_val = torch.exp(torch.tensor(1.0))  # e = exp(1)
        T_1b = torch.tensor(
            [[e_val, 1.0, 0.0], [0.0, e_val, 0.0], [0.0, 0.0, e_val]],
            dtype=DTYPE,
            device=device,
        )

        # Expected solution: log T = [[1, 1/e, 0], [0, 1, 0], [0, 0, 1]]
        expected_1b = torch.tensor(
            [[1.0, 1.0 / e_val, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=DTYPE,
            device=device,
        )

        # Compute using our implementation and compare
        result_1b = fm._matrix_log_33(T_1b)
        (
            torch.testing.assert_close(result_1b, expected_1b, rtol=rtol, atol=atol),
            f"Case 1b failed: \nExpected:\n{expected_1b}\nGot:\n{result_1b}",
        )

        # Compare with scipy
        scipy_result_1b = fm.matrix_log_scipy(T_1b)
        msg = (
            f"Case 1b differs from scipy: Expected:\n{scipy_result_1b}\nGot:\n{result_1b}"
        )
        torch.testing.assert_close(
            result_1b, scipy_result_1b, rtol=rtol, atol=atol, msg=msg
        )

        # Case 1c: All eigenvalues equal with q(T) = (T - λI)³
        # Example: T = [[e, 1, 1], [0, e, 1], [0, 0, e]]
        T_1c = torch.tensor(
            [[e_val, 1.0, 1.0], [0.0, e_val, 1.0], [0.0, 0.0, e_val]],
            dtype=DTYPE,
            device=device,
        )

        # Expected solution: log T = [[1, 1/e, (2e-1)/(2e²)], [0, 1, 1/e], [0, 0, 1]]
        expected_1c = torch.tensor(
            [
                [1.0, 1.0 / e_val, (2 * e_val - 1) / (2 * e_val * e_val)],
                [0.0, 1.0, 1.0 / e_val],
                [0.0, 0.0, 1.0],
            ],
            dtype=DTYPE,
            device=device,
        )

        # Compute using our implementation and compare
        result_1c = fm._matrix_log_33(T_1c)
        msg = f"Case 1c failed: \nExpected:\n{expected_1c}\nGot:\n{result_1c}"
        torch.testing.assert_close(result_1c, expected_1c, rtol=rtol, atol=atol, msg=msg)

        # Compare with scipy
        scipy_result_1c = fm.matrix_log_scipy(T_1c)
        msg = (
            f"Case 1c differs from scipy: Expected:\n{scipy_result_1c}\nGot:\n{result_1c}"
        )
        torch.testing.assert_close(
            result_1c, scipy_result_1c, rtol=rtol, atol=atol, msg=msg
        )

        # Case 2b: Two distinct eigenvalues with q(T) = (T - μI)(T - λI)²
        # Example: T = [[e, 1, 1], [0, e², 1], [0, 0, e²]]
        e_squared = e_val * e_val
        e_cubed = e_squared * e_val
        T_2b = torch.tensor(
            [[e_val, 1.0, 1.0], [0.0, e_squared, 1.0], [0.0, 0.0, e_squared]],
            dtype=DTYPE,
            device=device,
        )

        # Expected solution: log T = [[1, 1/(e(e-1)), (e³-e²-1)/(e³(e-1)²)],
        # [0, 2, 1/e²], [0, 0, 2]]
        expected_2b = torch.tensor(
            [
                [
                    1.0,
                    1.0 / (e_val * (e_val - 1.0)),
                    (e_cubed - e_squared - 1) / (e_cubed * (e_val - 1.0) * (e_val - 1.0)),
                ],
                [0.0, 2.0, 1.0 / e_squared],
                [0.0, 0.0, 2.0],
            ],
            dtype=DTYPE,
            device=device,
        )

        # Compute using our implementation and compare
        result_2b = fm._matrix_log_33(T_2b)
        msg = f"Case 2b failed: \nExpected:\n{expected_2b}\nGot:\n{result_2b}"
        torch.testing.assert_close(result_2b, expected_2b, rtol=rtol, atol=atol, msg=msg)

        # Compare with scipy
        scipy_result_2b = fm.matrix_log_scipy(T_2b)
        msg = (
            f"Case 2b differs from scipy: Expected:\n{scipy_result_2b}\nGot:\n{result_2b}"
        )
        torch.testing.assert_close(
            result_2b, scipy_result_2b, rtol=rtol, atol=atol, msg=msg
        )

        # Additional test: identity matrix (should return zero matrix)
        identity = torch.eye(3, dtype=DTYPE, device=device)
        log_identity = fm._matrix_log_33(identity)
        expected_log_identity = torch.zeros((3, 3), dtype=DTYPE, device=device)
        msg = f"log(I) failed: \nExpected:\n{expected_log_identity}\nGot:\n{log_identity}"
        torch.testing.assert_close(
            log_identity, expected_log_identity, rtol=rtol, atol=atol, msg=msg
        )

        # Additional test: diagonal matrix with distinct eigenvalues (Case 3)
        D = torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=DTYPE, device=device))
        log_D = fm._matrix_log_33(D)
        expected_log_D = torch.diag(
            torch.log(torch.tensor([2.0, 3.0, 4.0], dtype=DTYPE, device=device))
        )
        msg = f"log(diag) failed: \nExpected:\n{expected_log_D}\nGot:\n{log_D}"
        torch.testing.assert_close(log_D, expected_log_D, rtol=rtol, atol=atol, msg=msg)

    def test_random_float(self):
        """Test matrix logarithm on random 3x3 matrices.

        This test generates a random 3x3 matrix and compares the implementation
        against scipy's implementation to ensure consistency.
        """
        torch.manual_seed(1234)
        n = 3
        M = torch.randn(n, n, dtype=DTYPE, device=device)
        M_logm = fm.matrix_log_33(M)
        scipy_logm = scipy.linalg.logm(M.cpu().numpy())
        torch.testing.assert_close(
            M_logm, torch.tensor(scipy_logm, dtype=DTYPE, device=device)
        )

    def test_nearly_degenerate(self):
        """Test matrix logarithm on nearly degenerate matrices.

        This test verifies that the implementation handles matrices with
        nearly degenerate eigenvalues correctly by comparing against scipy's
        implementation.
        """
        eps = 1e-6
        M = torch.tensor(
            [[1.0, 1.0, eps], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=DTYPE,
            device=device,
        )
        M_logm = fm._matrix_log_33(M)
        scipy_logm = scipy.linalg.logm(M.cpu().numpy())
        torch.testing.assert_close(
            M_logm, torch.tensor(scipy_logm, dtype=DTYPE, device=device)
        )

    def test_batched_positive_definite(self):
        """Test batched matrix logarithm with positive definite matrices."""
        batch_size = 3
        rng = np.random.default_rng(42)
        L = rng.standard_normal((batch_size, 3, 3))
        M_np = np.array([L[i] @ L[i].T + 0.5 * np.eye(3) for i in range(batch_size)])
        M_torch = torch.tensor(M_np, dtype=torch.float64)

        log_torch = fm.matrix_log_33(M_torch)

        for i in range(batch_size):
            log_scipy = scipy.linalg.logm(M_np[i]).real
            assert_allclose(log_torch[i].numpy(), log_scipy, atol=1e-12)
            # Verify round-trip: exp(log(M)) = M
            M_recovered = torch.matrix_exp(log_torch[i])
            assert_allclose(M_recovered.numpy(), M_np[i], atol=1e-10)


class TestFrechetCellFilterIntegration:
    """Integration tests for the Frechet cell filter pipeline."""

    def test_frechet_derivatives_vs_scipy(self):
        """Test Frechet derivative computation matches scipy."""
        n_systems = 2
        torch.manual_seed(42)

        # Create small deformations
        deform_log = torch.randn(n_systems, 3, 3, dtype=torch.float64) * 0.01

        # Compute Frechet derivatives for all 9 directions
        idx_flat = torch.arange(9)
        i_idx, j_idx = idx_flat // 3, idx_flat % 3
        directions = torch.zeros((9, 3, 3), dtype=torch.float64)
        directions[idx_flat, i_idx, j_idx] = 1.0

        A_batch = deform_log.unsqueeze(1).expand(n_systems, 9, 3, 3).reshape(-1, 3, 3)
        E_batch = directions.unsqueeze(0).expand(n_systems, 9, 3, 3).reshape(-1, 3, 3)
        _, frechet_torch = fm.expm_frechet(A_batch, E_batch)
        frechet_torch = frechet_torch.reshape(n_systems, 9, 3, 3)

        # Compare with scipy
        for sys_idx in range(n_systems):
            for dir_idx in range(9):
                A_np = deform_log[sys_idx].numpy()
                E_np = directions[dir_idx].numpy()
                _, frechet_scipy = scipy.linalg.expm_frechet(A_np, E_np)
                assert_allclose(
                    frechet_torch[sys_idx, dir_idx].numpy(), frechet_scipy, atol=1e-12
                )
