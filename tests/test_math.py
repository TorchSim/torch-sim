"""Tests for the math module."""

# ruff: noqa: SLF001

import itertools

import numpy as np
import scipy
import torch
from numpy.testing import assert_allclose

from torch_sim.math import (
    _matrix_log_33,
    ell_table_61,
    expm,
    expm_frechet,
    matrix_log_33,
    matrix_log_scipy,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

"""Below tests are adapted from https://github.com/abhijeetgangan/torch_matfunc"""


class TestExpmFrechet:
    """Test suite for expm_frechet using numpy arrays converted to torch tensors."""

    def test_expm_frechet(self):
        """Test basic functionality of expm_frechet against scipy implementation."""
        M = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [0, 0, 1, 2],
                [0, 0, 5, 6],
            ],
            dtype=np.float64,
        )
        A = np.array(
            [
                [1, 2],
                [5, 6],
            ],
            dtype=np.float64,
        )
        E = np.array(
            [
                [3, 4],
                [7, 8],
            ],
            dtype=np.float64,
        )
        expected_expm = scipy.linalg.expm(A)
        expected_frechet = scipy.linalg.expm(M)[:2, 2:]

        A = torch.from_numpy(A).to(device=device)
        E = torch.from_numpy(E).to(device=device)
        for kwargs in ({}, {"method": "SPS"}, {"method": "blockEnlarge"}):
            # Convert it to numpy arrays before passing it to the function
            observed_expm, observed_frechet = expm_frechet(A, E, **kwargs)
            assert_allclose(expected_expm, observed_expm.cpu().numpy())
            assert_allclose(expected_frechet, observed_frechet.cpu().numpy())

    def test_small_norm_expm_frechet(self):
        """Test matrices with a range of norms for better coverage."""
        M_original = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [0, 0, 1, 2],
                [0, 0, 5, 6],
            ],
            dtype=np.float64,
        )
        A_original = np.array(
            [
                [1, 2],
                [5, 6],
            ],
            dtype=np.float64,
        )
        E_original = np.array(
            [
                [3, 4],
                [7, 8],
            ],
            dtype=np.float64,
        )
        A_original_norm_1 = scipy.linalg.norm(A_original, 1)
        selected_m_list = [1, 3, 5, 7, 9, 11, 13, 15]

        m_neighbor_pairs = itertools.pairwise(selected_m_list)
        for ma, mb in m_neighbor_pairs:
            ell_a = scipy.linalg._expm_frechet.ell_table_61[ma]
            ell_b = scipy.linalg._expm_frechet.ell_table_61[mb]
            target_norm_1 = 0.5 * (ell_a + ell_b)
            scale = target_norm_1 / A_original_norm_1
            M = scale * M_original
            A = scale * A_original
            E = scale * E_original
            expected_expm = scipy.linalg.expm(A)
            expected_frechet = scipy.linalg.expm(M)[:2, 2:]
            A = torch.from_numpy(A).to(device=device, dtype=dtype)
            E = torch.from_numpy(E).to(device=device, dtype=dtype)
            # Convert it to numpy arrays before passing it to the function
            observed_expm, observed_frechet = expm_frechet(A, E)
            assert_allclose(expected_expm, observed_expm.cpu().numpy())
            assert_allclose(expected_frechet, observed_frechet.cpu().numpy())

    def test_fuzz(self):
        """Test with a variety of random inputs to ensure robustness."""
        rng = np.random.default_rng(1726500908359153)
        # try a bunch of crazy inputs
        rfuncs = (
            np.random.uniform,
            np.random.normal,
            np.random.standard_cauchy,
            np.random.exponential,
        )
        ntests = 100
        for _ in range(ntests):
            rfunc = rfuncs[rng.choice(4)]
            target_norm_1 = rng.exponential()
            n = rng.integers(2, 16)
            A_original = rfunc(size=(n, n))
            E_original = rfunc(size=(n, n))
            A_original_norm_1 = scipy.linalg.norm(A_original, 1)
            scale = target_norm_1 / A_original_norm_1
            A = scale * A_original
            E = scale * E_original
            M = np.vstack([np.hstack([A, E]), np.hstack([np.zeros_like(A), A])])
            expected_expm = scipy.linalg.expm(A)
            expected_frechet = scipy.linalg.expm(M)[:n, n:]
            A = torch.from_numpy(A).to(device=device, dtype=dtype)
            E = torch.from_numpy(E).to(device=device, dtype=dtype)
            # Convert it to numpy arrays before passing it to the function
            observed_expm, observed_frechet = expm_frechet(A, E)
            assert_allclose(expected_expm, observed_expm.cpu().numpy(), atol=5e-8)
            assert_allclose(expected_frechet, observed_frechet.cpu().numpy(), atol=1e-7)

    def test_problematic_matrix(self):
        """Test a specific matrix that previously uncovered a bug."""
        A = np.array(
            [
                [1.50591997, 1.93537998],
                [0.41203263, 0.23443516],
            ],
            dtype=np.float64,
        )
        E = np.array(
            [
                [1.87864034, 2.07055038],
                [1.34102727, 0.67341123],
            ],
            dtype=np.float64,
        )
        A = torch.from_numpy(A).to(device=device, dtype=dtype)
        E = torch.from_numpy(E).to(device=device, dtype=dtype)
        # Convert it to numpy arrays before passing it to the function
        sps_expm, sps_frechet = expm_frechet(A, E, method="SPS")
        blockEnlarge_expm, blockEnlarge_frechet = expm_frechet(
            A, E, method="blockEnlarge"
        )
        assert_allclose(sps_expm.cpu().numpy(), blockEnlarge_expm.cpu().numpy())
        assert_allclose(sps_frechet.cpu().numpy(), blockEnlarge_frechet.cpu().numpy())

    def test_medium_matrix(self):
        """Test with a medium-sized matrix to compare performance between methods."""
        n = 1000
        rng = np.random.default_rng()
        A = rng.exponential(size=(n, n))
        E = rng.exponential(size=(n, n))

        A = torch.from_numpy(A).to(device=device, dtype=dtype)
        E = torch.from_numpy(E).to(device=device, dtype=dtype)
        # Convert it to numpy arrays before passing it to the function
        sps_expm, sps_frechet = expm_frechet(A, E, method="SPS")
        blockEnlarge_expm, blockEnlarge_frechet = expm_frechet(
            A, E, method="blockEnlarge"
        )
        assert_allclose(sps_expm.cpu().numpy(), blockEnlarge_expm.cpu().numpy())
        assert_allclose(sps_frechet.cpu().numpy(), blockEnlarge_frechet.cpu().numpy())


class TestExpmFrechetTorch:
    """Test suite for expm_frechet using native torch tensors."""

    def test_expm_frechet(self):
        """Test basic functionality of expm_frechet against torch.linalg.matrix_exp."""
        M = torch.tensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [0, 0, 1, 2],
                [0, 0, 5, 6],
            ],
            dtype=dtype,
            device=device,
        )
        A = torch.tensor(
            [
                [1, 2],
                [5, 6],
            ],
            dtype=dtype,
            device=device,
        )
        E = torch.tensor(
            [
                [3, 4],
                [7, 8],
            ],
            dtype=dtype,
            device=device,
        )
        expected_expm = torch.linalg.matrix_exp(A)
        expected_frechet = torch.linalg.matrix_exp(M)[:2, 2:]

        for kwargs in ({}, {"method": "SPS"}, {"method": "blockEnlarge"}):
            observed_expm, observed_frechet = expm_frechet(A, E, **kwargs)
            torch.testing.assert_close(expected_expm, observed_expm)
            torch.testing.assert_close(expected_frechet, observed_frechet)

    def test_small_norm_expm_frechet(self):
        """Test matrices with a range of norms for better coverage using torch tensors."""
        M_original = torch.tensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [0, 0, 1, 2],
                [0, 0, 5, 6],
            ],
            dtype=dtype,
            device=device,
        )
        A_original = torch.tensor(
            [
                [1, 2],
                [5, 6],
            ],
            dtype=dtype,
            device=device,
        )
        E_original = torch.tensor(
            [
                [3, 4],
                [7, 8],
            ],
            dtype=dtype,
            device=device,
        )
        A_original_norm_1 = torch.linalg.norm(A_original, 1)
        selected_m_list = [1, 3, 5, 7, 9, 11, 13, 15]
        m_neighbor_pairs = itertools.pairwise(selected_m_list)
        for ma, mb in m_neighbor_pairs:
            ell_a = ell_table_61[ma]
            ell_b = ell_table_61[mb]
            target_norm_1 = 0.5 * (ell_a + ell_b)
            scale = target_norm_1 / A_original_norm_1
            M = scale * M_original
            A = scale * A_original
            E = scale * E_original
            expected_expm = torch.linalg.matrix_exp(A)
            expected_frechet = torch.linalg.matrix_exp(M)[:2, 2:]
            observed_expm, observed_frechet = expm_frechet(A, E)
            torch.testing.assert_close(expected_expm, observed_expm)
            torch.testing.assert_close(expected_frechet, observed_frechet)

    def test_fuzz(self):
        """Test with a variety of random inputs using torch tensors."""
        rng = np.random.default_rng(1726500908359153)
        # try a bunch of crazy inputs
        # Convert random functions to tensor-generating functions
        tensor_rfuncs = (
            lambda size, device="cpu": torch.tensor(
                rng.uniform(size=size), device=device
            ),
            lambda size, device="cpu": torch.tensor(rng.normal(size=size), device=device),
            lambda size, device="cpu": torch.tensor(
                rng.standard_cauchy(size=size), device=device
            ),
            lambda size, device="cpu": torch.tensor(
                rng.exponential(size=size), device=device
            ),
        )
        ntests = 100
        for _ in range(ntests):
            rfunc = tensor_rfuncs[torch.tensor(rng.choice(4))]
            target_norm_1 = torch.tensor(rng.exponential())
            n = torch.tensor(rng.integers(2, 16))
            A_original = rfunc(size=(n, n))
            E_original = rfunc(size=(n, n))
            A_original_norm_1 = torch.linalg.norm(A_original, 1)
            scale = target_norm_1 / A_original_norm_1
            A = scale * A_original
            E = scale * E_original
            M = torch.vstack(
                [torch.hstack([A, E]), torch.hstack([torch.zeros_like(A), A])]
            )
            expected_expm = torch.linalg.matrix_exp(A)
            expected_frechet = torch.linalg.matrix_exp(M)[:n, n:]
            observed_expm, observed_frechet = expm_frechet(A, E)
            torch.testing.assert_close(expected_expm, observed_expm, atol=5e-8, rtol=1e-5)
            torch.testing.assert_close(
                expected_frechet, observed_frechet, atol=1e-7, rtol=1e-5
            )

    def test_problematic_matrix(self):
        """Test a specific matrix that previously uncovered a bug using torch tensors."""
        A = torch.tensor(
            [
                [1.50591997, 1.93537998],
                [0.41203263, 0.23443516],
            ],
            dtype=dtype,
            device=device,
        )
        E = torch.tensor(
            [
                [1.87864034, 2.07055038],
                [1.34102727, 0.67341123],
            ],
            dtype=dtype,
            device=device,
        )
        sps_expm, sps_frechet = expm_frechet(A, E, method="SPS")
        blockEnlarge_expm, blockEnlarge_frechet = expm_frechet(
            A, E, method="blockEnlarge"
        )
        torch.testing.assert_close(sps_expm, blockEnlarge_expm)
        torch.testing.assert_close(sps_frechet, blockEnlarge_frechet)

    def test_medium_matrix(self):
        """Test with a medium-sized matrix to compare performance
        between methods using torch tensors.
        """
        n = 1000
        rng = np.random.default_rng()
        A = torch.tensor(rng.exponential(size=(n, n)))
        E = torch.tensor(rng.exponential(size=(n, n)))

        sps_expm, sps_frechet = expm_frechet(A, E, method="SPS")
        blockEnlarge_expm, blockEnlarge_frechet = expm_frechet(
            A, E, method="blockEnlarge"
        )
        torch.testing.assert_close(sps_expm, blockEnlarge_expm)
        torch.testing.assert_close(sps_frechet, blockEnlarge_frechet)


class TestExpmFrechetTorchGrad:
    """Test suite for gradient computation with expm and its Frechet derivative."""

    def test_expm_frechet(self):
        """Test gradient computation for matrix exponential and its Frechet derivative."""
        M = torch.tensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [0, 0, 1, 2],
                [0, 0, 5, 6],
            ],
            dtype=dtype,
            device=device,
        )
        A = torch.tensor(
            [
                [1, 2],
                [5, 6],
            ],
            dtype=dtype,
            device=device,
        )
        E = torch.tensor(
            [
                [3, 4],
                [7, 8],
            ],
            dtype=dtype,
            device=device,
        )
        expected_expm = torch.linalg.matrix_exp(A)
        expected_frechet = torch.linalg.matrix_exp(M)[:2, 2:]
        # expm will use the SPS method as default
        observed_expm = expm.apply(A)
        torch.testing.assert_close(expected_expm, observed_expm)
        # Compute the Frechet derivative in the direction of grad_output
        A.requires_grad = True
        observed_expm = expm.apply(A)
        (observed_frechet,) = torch.autograd.grad(observed_expm, A, E, retain_graph=True)
        torch.testing.assert_close(expected_frechet, observed_frechet)


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
            dtype=dtype,
            device=device,
        )

        # Expected solution: log T = [[1, 1/e, 0], [0, 1, 0], [0, 0, 1]]
        expected_1b = torch.tensor(
            [[1.0, 1.0 / e_val, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
            device=device,
        )

        # Compute using our implementation and compare
        result_1b = _matrix_log_33(T_1b)
        (
            torch.testing.assert_close(result_1b, expected_1b, rtol=rtol, atol=atol),
            f"Case 1b failed: \nExpected:\n{expected_1b}\nGot:\n{result_1b}",
        )

        # Compare with scipy
        scipy_result_1b = matrix_log_scipy(T_1b)
        msg = (
            f"Case 1b differs from scipy: Expected:\n{scipy_result_1b}\nGot:\n{result_1b}"
        )
        torch.testing.assert_close(
            result_1b,
            scipy_result_1b,
            rtol=rtol,
            atol=atol,
            msg=msg,
        )

        # Case 1c: All eigenvalues equal with q(T) = (T - λI)³
        # Example: T = [[e, 1, 1], [0, e, 1], [0, 0, e]]
        T_1c = torch.tensor(
            [[e_val, 1.0, 1.0], [0.0, e_val, 1.0], [0.0, 0.0, e_val]],
            dtype=dtype,
            device=device,
        )

        # Expected solution: log T = [[1, 1/e, (2e-1)/(2e²)], [0, 1, 1/e], [0, 0, 1]]
        expected_1c = torch.tensor(
            [
                [1.0, 1.0 / e_val, (2 * e_val - 1) / (2 * e_val * e_val)],
                [0.0, 1.0, 1.0 / e_val],
                [0.0, 0.0, 1.0],
            ],
            dtype=dtype,
            device=device,
        )

        # Compute using our implementation and compare
        result_1c = _matrix_log_33(T_1c)
        msg = f"Case 1c failed: \nExpected:\n{expected_1c}\nGot:\n{result_1c}"
        torch.testing.assert_close(result_1c, expected_1c, rtol=rtol, atol=atol, msg=msg)

        # Compare with scipy
        scipy_result_1c = matrix_log_scipy(T_1c)
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
            dtype=dtype,
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
            dtype=dtype,
            device=device,
        )

        # Compute using our implementation and compare
        result_2b = _matrix_log_33(T_2b)
        msg = f"Case 2b failed: \nExpected:\n{expected_2b}\nGot:\n{result_2b}"
        torch.testing.assert_close(result_2b, expected_2b, rtol=rtol, atol=atol, msg=msg)

        # Compare with scipy
        scipy_result_2b = matrix_log_scipy(T_2b)
        msg = (
            f"Case 2b differs from scipy: Expected:\n{scipy_result_2b}\nGot:\n{result_2b}"
        )
        torch.testing.assert_close(
            result_2b, scipy_result_2b, rtol=rtol, atol=atol, msg=msg
        )

        # Additional test: identity matrix (should return zero matrix)
        identity = torch.eye(3, dtype=dtype, device=device)
        log_identity = _matrix_log_33(identity)
        expected_log_identity = torch.zeros((3, 3), dtype=dtype, device=device)
        msg = f"log(I) failed: \nExpected:\n{expected_log_identity}\nGot:\n{log_identity}"
        torch.testing.assert_close(
            log_identity, expected_log_identity, rtol=rtol, atol=atol, msg=msg
        )

        # Additional test: diagonal matrix with distinct eigenvalues (Case 3)
        D = torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=dtype, device=device))
        log_D = _matrix_log_33(D)
        expected_log_D = torch.diag(
            torch.log(torch.tensor([2.0, 3.0, 4.0], dtype=dtype, device=device))
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
        M = torch.randn(n, n, dtype=dtype, device=device)
        M_logm = matrix_log_33(M)
        scipy_logm = scipy.linalg.logm(M.cpu().numpy())
        torch.testing.assert_close(
            M_logm, torch.tensor(scipy_logm, dtype=dtype, device=device)
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
            dtype=dtype,
            device=device,
        )
        M_logm = matrix_log_33(M)
        scipy_logm = scipy.linalg.logm(M.cpu().numpy())
        torch.testing.assert_close(
            M_logm, torch.tensor(scipy_logm, dtype=dtype, device=device)
        )
