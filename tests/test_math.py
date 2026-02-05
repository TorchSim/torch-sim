"""Tests for the math module."""

# ruff: noqa: SLF001

import numpy as np
import pytest
import scipy
import scipy.linalg
import torch

import torch_sim.math as fm
from tests.conftest import DTYPE
from torch_sim.math import expm_frechet, matrix_log_33
from torch_sim.optimizers.cell_filters import deform_grad


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tolerances for numerical comparisons
ATOL_STRICT = 1e-14  # For operations that should be exact
ATOL_NORMAL = 1e-12  # For normal numerical operations
ATOL_RELAXED = 1e-10  # For operations with some numerical accumulation


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


class TestExpmFrechet:
    """Tests for expm_frechet against scipy.linalg.expm_frechet."""

    def test_small_matrices_batched(self):
        """Test expm_frechet with small 3x3 matrices in a batch."""
        A_np = np.array(
            [
                [[0.1, 0.2, 0.05], [0.15, 0.1, 0.1], [0.05, 0.1, 0.15]],
                [[0.15, 0.1, 0.08], [0.12, 0.18, 0.05], [0.08, 0.05, 0.12]],
            ],
            dtype=np.float64,
        )
        E_np = np.array(
            [
                [[0.01, 0.02, 0.03], [0.02, 0.01, 0.02], [0.03, 0.02, 0.01]],
                [[0.02, 0.01, 0.02], [0.01, 0.03, 0.01], [0.02, 0.01, 0.02]],
            ],
            dtype=np.float64,
        )

        A_torch = torch.tensor(A_np, dtype=torch.float64)
        E_torch = torch.tensor(E_np, dtype=torch.float64)

        # torch_sim batched implementation
        expm_torch, frechet_torch = expm_frechet(A_torch, E_torch)

        # scipy reference (unbatched, for comparison)
        for i in range(A_np.shape[0]):
            expm_scipy, frechet_scipy = scipy.linalg.expm_frechet(A_np[i], E_np[i])
            np.testing.assert_allclose(
                expm_torch[i].numpy(), expm_scipy, atol=ATOL_STRICT, rtol=0
            )
            np.testing.assert_allclose(
                frechet_torch[i].numpy(), frechet_scipy, atol=ATOL_STRICT, rtol=0
            )

    def test_large_norm_matrices_batched(self):
        """Test expm_frechet with larger norm matrices requiring scaling."""
        A_np = np.array(
            [
                [[1.5, 0.8, 0.3], [0.6, 1.2, 0.5], [0.4, 0.7, 1.8]],
                [[1.2, 0.6, 0.4], [0.5, 1.4, 0.6], [0.3, 0.5, 1.6]],
            ],
            dtype=np.float64,
        )
        E_np = np.array(
            [
                [[0.1, 0.2, 0.1], [0.2, 0.15, 0.1], [0.1, 0.1, 0.2]],
                [[0.15, 0.1, 0.15], [0.1, 0.2, 0.1], [0.15, 0.1, 0.15]],
            ],
            dtype=np.float64,
        )

        A_torch = torch.tensor(A_np, dtype=torch.float64)
        E_torch = torch.tensor(E_np, dtype=torch.float64)

        expm_torch, frechet_torch = expm_frechet(A_torch, E_torch)

        for i in range(A_np.shape[0]):
            expm_scipy, frechet_scipy = scipy.linalg.expm_frechet(A_np[i], E_np[i])
            np.testing.assert_allclose(
                expm_torch[i].numpy(), expm_scipy, atol=ATOL_NORMAL, rtol=0
            )
            np.testing.assert_allclose(
                frechet_torch[i].numpy(), frechet_scipy, atol=ATOL_NORMAL, rtol=0
            )

    @pytest.mark.parametrize("seed", range(5))
    def test_random_matrices_batched(self, seed: int) -> None:
        """Test expm_frechet with random batched matrices."""
        batch_size = 3
        rng = np.random.default_rng(seed)
        A_np = rng.standard_normal((batch_size, 3, 3)) * 0.5
        E_np = rng.standard_normal((batch_size, 3, 3)) * 0.2

        expm_torch, frechet_torch = expm_frechet(
            torch.tensor(A_np, dtype=torch.float64),
            torch.tensor(E_np, dtype=torch.float64),
        )

        for i in range(batch_size):
            expm_scipy, frechet_scipy = scipy.linalg.expm_frechet(A_np[i], E_np[i])
            np.testing.assert_allclose(
                expm_torch[i].numpy(), expm_scipy, atol=ATOL_STRICT, rtol=0
            )
            np.testing.assert_allclose(
                frechet_torch[i].numpy(), frechet_scipy, atol=ATOL_STRICT, rtol=0
            )


class TestMatrixLog33Batched:
    """Additional tests for matrix_log_33 with batched inputs and deformation cases."""

    def test_distinct_eigenvalues_batched(self):
        """Test log of matrices with 3 distinct eigenvalues (case3)."""
        batch_size = 3
        eigenvalues_list = [
            np.array([1.5, 2.0, 3.0]),
            np.array([1.2, 1.8, 2.5]),
            np.array([1.1, 1.6, 2.2]),
        ]
        Q_base = np.array(
            [[0.6, 0.7, 0.3], [0.7, -0.5, 0.5], [0.4, 0.5, -0.8]], dtype=np.float64
        )
        Q_np, _ = np.linalg.qr(Q_base)

        T_batch_np = np.array([Q_np @ np.diag(eig) @ Q_np.T for eig in eigenvalues_list])
        T_batch_torch = torch.tensor(T_batch_np, dtype=torch.float64)

        log_torch = matrix_log_33(T_batch_torch)

        for i in range(batch_size):
            log_scipy = scipy.linalg.logm(T_batch_np[i])
            np.testing.assert_allclose(
                log_torch[i].numpy(), log_scipy.real, atol=ATOL_STRICT, rtol=0
            )

            # Verify round-trip: exp(log(T)) = T
            T_recovered = torch.matrix_exp(log_torch[i])
            np.testing.assert_allclose(
                T_recovered.numpy(), T_batch_np[i], atol=ATOL_NORMAL, rtol=0
            )

    def test_batched_matrices(self):
        """Test batched matrix logarithm with random positive definite matrices."""
        batch_size = 5
        rng = np.random.default_rng(42)
        L_batch = rng.standard_normal((batch_size, 3, 3))
        T_batch_np = np.array(
            [L_batch[i] @ L_batch[i].T + 0.5 * np.eye(3) for i in range(batch_size)]
        )
        T_batch_torch = torch.tensor(T_batch_np, dtype=torch.float64)

        # scipy unbatched (for comparison)
        log_batch_scipy = np.array(
            [scipy.linalg.logm(T_batch_np[i]).real for i in range(batch_size)]
        )

        # torch_sim batched
        log_batch_torch = matrix_log_33(T_batch_torch)

        np.testing.assert_allclose(
            log_batch_torch.numpy(), log_batch_scipy, atol=ATOL_NORMAL, rtol=0
        )

    @pytest.mark.parametrize("eps", [1e-2, 1e-4, 1e-6, 1e-8])
    def test_near_identity_batched(self, eps: float) -> None:
        """Test log of near-identity matrices with various perturbation sizes."""
        batch_size = 3
        perturbations = np.array(
            [
                [[0.1, 0.2, 0.1], [0.2, 0.15, 0.05], [0.1, 0.05, 0.2]],
                [[0.15, 0.1, 0.08], [0.1, 0.2, 0.1], [0.08, 0.1, 0.15]],
                [[0.08, 0.15, 0.12], [0.15, 0.1, 0.08], [0.12, 0.08, 0.18]],
            ],
            dtype=np.float64,
        )
        M_np = np.eye(3) + eps * perturbations
        M_torch = torch.tensor(M_np, dtype=torch.float64)

        log_torch = matrix_log_33(M_torch)

        for i in range(batch_size):
            log_scipy = scipy.linalg.logm(M_np[i]).real
            np.testing.assert_allclose(
                log_torch[i].numpy(), log_scipy, atol=ATOL_NORMAL, rtol=0
            )


class TestDeformGrad:
    """Tests for deformation gradient computation."""

    @pytest.mark.parametrize("stretch_factor", [0.9, 1.0, 1.05, 1.1, 1.5])
    def test_uniaxial_stretch_batched(self, stretch_factor: float) -> None:
        """Test deformation gradient for uniaxial stretch."""
        batch_size = 2
        ref_cell = torch.tensor(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=torch.float64
        )
        ref_cells = ref_cell.unsqueeze(0).expand(batch_size, -1, -1).clone()
        current_cells = ref_cells.clone()
        current_cells[:, 0, 0] *= stretch_factor

        F = deform_grad(ref_cells.transpose(-1, -2), current_cells)

        expected = np.eye(3)
        expected[0, 0] = stretch_factor
        for i in range(batch_size):
            np.testing.assert_allclose(F[i].numpy(), expected, atol=ATOL_STRICT, rtol=0)

    @pytest.mark.parametrize("shear", [0.01, 0.05, 0.1])
    def test_shear_deformation_batched(self, shear: float) -> None:
        """Test deformation gradient for shear deformation (batched)."""
        batch_size = 2
        ref_cell = torch.tensor(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=torch.float64
        )
        ref_cells = ref_cell.unsqueeze(0).expand(batch_size, -1, -1).clone()
        current_cells = ref_cells.clone()
        current_cells[:, 0, 1] = shear * ref_cells[:, 1, 1]

        F = deform_grad(ref_cells.transpose(-1, -2), current_cells)

        expected = np.eye(3)
        expected[1, 0] = shear  # Note: transpose in deform_grad
        for i in range(batch_size):
            np.testing.assert_allclose(F[i].numpy(), expected, atol=ATOL_STRICT, rtol=0)


class TestFrechetCellFilterIntegration:
    """Integration tests for the full Frechet cell filter pipeline."""

    def test_frechet_cell_step_vs_scipy(self):
        """Test Frechet cell step computation matches scipy."""
        n_systems = 2
        cell_factor = torch.tensor([10.0, 12.0], dtype=torch.float64).view(
            n_systems, 1, 1
        )
        cell_lr = torch.tensor([0.01, 0.01], dtype=torch.float64)

        # Initialize cell positions (in log space)
        cell_positions = torch.zeros((n_systems, 3, 3), dtype=torch.float64)

        # Fixed random forces for reproducibility
        torch.manual_seed(42)
        cell_forces = torch.randn(n_systems, 3, 3, dtype=torch.float64) * 0.1

        # Perform step update
        cell_wise_lr = cell_lr.view(n_systems, 1, 1)
        cell_step = cell_wise_lr * cell_forces
        cell_positions_new = cell_positions + cell_step

        # Convert from log space to deformation gradient
        deform_grad_log_new = cell_positions_new / cell_factor
        deform_grad_new_torch = torch.matrix_exp(deform_grad_log_new)

        # Compare with scipy
        deform_grad_new_scipy = np.array(
            [scipy.linalg.expm(deform_grad_log_new[i].numpy()) for i in range(n_systems)]
        )

        np.testing.assert_allclose(
            deform_grad_new_torch.numpy(), deform_grad_new_scipy, atol=ATOL_STRICT, rtol=0
        )

    def test_frechet_force_computation_vs_scipy(self):
        """Test full Frechet force computation matches scipy-based computation."""
        n_systems = 2

        # Setup deformation gradients
        torch.manual_seed(42)
        cell_factor = torch.tensor([10.0, 12.0], dtype=torch.float64).view(
            n_systems, 1, 1
        )

        # Create small deformations
        deform_log = torch.randn(n_systems, 3, 3, dtype=torch.float64) * 0.001
        cur_deform_grad = torch.matrix_exp(deform_log)

        torch.manual_seed(123)
        virial = torch.randn(n_systems, 3, 3, dtype=torch.float64) * 0.5

        # UCF cell gradient
        ucf_cell_grad = torch.bmm(
            virial, torch.linalg.inv(cur_deform_grad.transpose(-1, -2))
        )

        # torch_sim log computation
        deform_grad_log_torch = matrix_log_33(cur_deform_grad)

        # scipy log computation
        deform_grad_log_scipy = np.array(
            [scipy.linalg.logm(cur_deform_grad[i].numpy()).real for i in range(n_systems)]
        )

        np.testing.assert_allclose(
            deform_grad_log_torch.numpy(), deform_grad_log_scipy, atol=ATOL_NORMAL, rtol=0
        )

        # Compute Frechet derivatives
        device, dtype = virial.device, virial.dtype
        idx_flat = torch.arange(9, device=device)
        i_idx, j_idx = idx_flat // 3, idx_flat % 3
        directions = torch.zeros((9, 3, 3), device=device, dtype=dtype)
        directions[idx_flat, i_idx, j_idx] = 1.0

        # torch_sim batched Frechet
        A_batch = (
            deform_grad_log_torch.unsqueeze(1)
            .expand(n_systems, 9, 3, 3)
            .reshape(-1, 3, 3)
        )
        E_batch = directions.unsqueeze(0).expand(n_systems, 9, 3, 3).reshape(-1, 3, 3)
        _, expm_derivs_batch_torch = expm_frechet(A_batch, E_batch)
        expm_derivs_torch = expm_derivs_batch_torch.reshape(n_systems, 9, 3, 3)

        # scipy Frechet (for comparison)
        expm_derivs_scipy = np.zeros((n_systems, 9, 3, 3))
        for sys_idx in range(n_systems):
            for dir_idx in range(9):
                A_sys = deform_grad_log_scipy[sys_idx]
                E_dir = directions[dir_idx].numpy()
                _, expm_derivs_scipy[sys_idx, dir_idx] = scipy.linalg.expm_frechet(
                    A_sys, E_dir
                )

        np.testing.assert_allclose(
            expm_derivs_torch.numpy(), expm_derivs_scipy, atol=ATOL_NORMAL, rtol=0
        )

        # Final cell forces
        forces_flat_torch = (expm_derivs_torch * ucf_cell_grad.unsqueeze(1)).sum(
            dim=(2, 3)
        )
        cell_forces_torch = forces_flat_torch.reshape(n_systems, 3, 3) / cell_factor

        forces_flat_scipy = (
            expm_derivs_scipy * ucf_cell_grad.numpy()[:, np.newaxis, :, :]
        ).sum(axis=(2, 3))
        cell_forces_scipy = (
            forces_flat_scipy.reshape(n_systems, 3, 3) / cell_factor.numpy()
        )

        np.testing.assert_allclose(
            cell_forces_torch.numpy(), cell_forces_scipy, atol=ATOL_NORMAL, rtol=0
        )


class TestRoundTripConsistency:
    """Tests for round-trip consistency exp(log(M)) = M."""

    @pytest.mark.parametrize("seed", range(5))
    def test_positive_definite_roundtrip_batched(self, seed: int) -> None:
        """Test that exp(log(M)) = M for positive definite matrices."""
        batch_size = 3
        rng = np.random.default_rng(seed)
        L = rng.standard_normal((batch_size, 3, 3))
        M_np = np.array([L[i] @ L[i].T + 0.5 * np.eye(3) for i in range(batch_size)])
        M_torch = torch.tensor(M_np, dtype=torch.float64)

        log_torch = matrix_log_33(M_torch)
        M_recovered = torch.matrix_exp(log_torch)

        np.testing.assert_allclose(M_recovered.numpy(), M_np, atol=ATOL_RELAXED, rtol=0)
