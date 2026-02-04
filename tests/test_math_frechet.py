"""Numerical behavior tests for Frechet cell filter and matrix log/exp functions.

This file focuses on cell-filter-specific numerical validation that complements
the core math tests in tests/test_math.py. Core functionality tests (identity,
random matrices, method comparisons) are in tests/test_math.py.

Tests included here:
1. expm_frechet: 3x3 matrix-specific tests with scipy comparison
   - Small and large norm 3x3 matrices
   - Batched (B, 3, 3) inputs

2. matrix_log_33: Deformation-specific cases
   - Small deformations near identity
   - Distinct eigenvalue cases
   - Batched inputs
   - Near-identity perturbation tests

3. deform_grad: Deformation gradient computation
   - Identity, stretch, and shear deformations
   - Batched deformation gradients

4. Frechet cell filter integration
   - Full pipeline: log space update -> exp -> cell update
   - End-to-end scipy comparison

5. Round-trip consistency: exp(log(M)) = M
"""

import numpy as np
import pytest
import scipy.linalg
import torch

from torch_sim.math import expm_frechet, matrix_log_33
from torch_sim.optimizers.cell_filters import deform_grad


# Tolerances for numerical comparisons
ATOL_STRICT = 1e-14  # For operations that should be exact
ATOL_NORMAL = 1e-12  # For normal numerical operations
ATOL_RELAXED = 1e-10  # For operations with some numerical accumulation


class TestExpmFrechet:
    """Tests for expm_frechet against scipy.linalg.expm_frechet."""

    def test_small_matrix(self):
        """Test expm_frechet with a small 3x3 matrix."""
        A_np = np.array(
            [[0.1, 0.2, 0.05], [0.15, 0.1, 0.1], [0.05, 0.1, 0.15]], dtype=np.float64
        )
        E_np = np.array(
            [[0.01, 0.02, 0.03], [0.02, 0.01, 0.02], [0.03, 0.02, 0.01]], dtype=np.float64
        )

        A_torch = torch.tensor(A_np, dtype=torch.float64)
        E_torch = torch.tensor(E_np, dtype=torch.float64)

        # scipy reference
        expm_scipy, frechet_scipy = scipy.linalg.expm_frechet(A_np, E_np)

        # torch_sim implementation
        expm_torch, frechet_torch = expm_frechet(A_torch, E_torch, method="SPS")

        np.testing.assert_allclose(
            expm_torch.numpy(), expm_scipy, atol=ATOL_STRICT, rtol=0
        )
        np.testing.assert_allclose(
            frechet_torch.numpy(), frechet_scipy, atol=ATOL_STRICT, rtol=0
        )

    def test_large_norm_matrix(self):
        """Test expm_frechet with a larger norm matrix requiring scaling."""
        A_np = np.array(
            [[1.5, 0.8, 0.3], [0.6, 1.2, 0.5], [0.4, 0.7, 1.8]], dtype=np.float64
        )
        E_np = np.array(
            [[0.1, 0.2, 0.1], [0.2, 0.15, 0.1], [0.1, 0.1, 0.2]], dtype=np.float64
        )

        A_torch = torch.tensor(A_np, dtype=torch.float64)
        E_torch = torch.tensor(E_np, dtype=torch.float64)

        expm_scipy, frechet_scipy = scipy.linalg.expm_frechet(A_np, E_np)
        expm_torch, frechet_torch = expm_frechet(A_torch, E_torch, method="SPS")

        np.testing.assert_allclose(
            expm_torch.numpy(), expm_scipy, atol=ATOL_NORMAL, rtol=0
        )
        np.testing.assert_allclose(
            frechet_torch.numpy(), frechet_scipy, atol=ATOL_NORMAL, rtol=0
        )

    @pytest.mark.parametrize("seed", range(10))
    def test_random_matrices(self, seed):
        """Test expm_frechet with random matrices."""
        np.random.seed(seed)
        A_np = np.random.randn(3, 3) * 0.5
        E_np = np.random.randn(3, 3) * 0.2

        expm_scipy, frechet_scipy = scipy.linalg.expm_frechet(A_np, E_np)
        expm_torch, frechet_torch = expm_frechet(
            torch.tensor(A_np, dtype=torch.float64),
            torch.tensor(E_np, dtype=torch.float64),
            method="SPS",
        )

        np.testing.assert_allclose(
            expm_torch.numpy(), expm_scipy, atol=ATOL_STRICT, rtol=0
        )
        np.testing.assert_allclose(
            frechet_torch.numpy(), frechet_scipy, atol=ATOL_STRICT, rtol=0
        )

    def test_batched_matrices(self):
        """Test expm_frechet with batched 3x3 matrices."""
        batch_size = 4
        np.random.seed(42)
        A_batch_np = np.random.randn(batch_size, 3, 3) * 0.3
        E_batch_np = np.random.randn(batch_size, 3, 3) * 0.1

        A_batch_torch = torch.tensor(A_batch_np, dtype=torch.float64)
        E_batch_torch = torch.tensor(E_batch_np, dtype=torch.float64)

        # torch_sim batched
        expm_batch_torch, frechet_batch_torch = expm_frechet(
            A_batch_torch, E_batch_torch
        )

        # scipy unbatched (for comparison)
        expm_batch_scipy = np.zeros_like(A_batch_np)
        frechet_batch_scipy = np.zeros_like(A_batch_np)
        for i in range(batch_size):
            expm_batch_scipy[i], frechet_batch_scipy[i] = scipy.linalg.expm_frechet(
                A_batch_np[i], E_batch_np[i]
            )

        np.testing.assert_allclose(
            expm_batch_torch.numpy(), expm_batch_scipy, atol=ATOL_STRICT, rtol=0
        )
        np.testing.assert_allclose(
            frechet_batch_torch.numpy(), frechet_batch_scipy, atol=ATOL_STRICT, rtol=0
        )

class TestMatrixLog33:
    """Tests for matrix_log_33 against scipy.linalg.logm.

    Note: Basic tests (identity, random) are in tests/test_math.py.
    These tests focus on deformation-specific cases.
    """

    def test_small_deformation(self):
        """Test log of matrix close to identity."""
        deform_np = np.eye(3, dtype=np.float64) + 0.05 * np.array(
            [[0.1, 0.2, 0.1], [0.2, 0.15, 0.05], [0.1, 0.05, 0.2]], dtype=np.float64
        )
        deform_torch = torch.tensor(deform_np, dtype=torch.float64)

        log_scipy = scipy.linalg.logm(deform_np)
        log_torch = matrix_log_33(deform_torch)

        np.testing.assert_allclose(
            log_torch.numpy(), log_scipy.real, atol=ATOL_STRICT, rtol=0
        )

    def test_distinct_eigenvalues(self):
        """Test log of matrix with 3 distinct eigenvalues (case3)."""
        eigenvalues = np.array([1.5, 2.0, 3.0])
        Q_np = np.array(
            [[0.6, 0.7, 0.3], [0.7, -0.5, 0.5], [0.4, 0.5, -0.8]], dtype=np.float64
        )
        Q_np, _ = np.linalg.qr(Q_np)
        T_np = Q_np @ np.diag(eigenvalues) @ Q_np.T
        T_torch = torch.tensor(T_np, dtype=torch.float64)

        log_scipy = scipy.linalg.logm(T_np)
        log_torch = matrix_log_33(T_torch)

        np.testing.assert_allclose(
            log_torch.numpy(), log_scipy.real, atol=ATOL_STRICT, rtol=0
        )

        # Verify round-trip: exp(log(T)) = T
        T_recovered = torch.matrix_exp(log_torch)
        np.testing.assert_allclose(
            T_recovered.numpy(), T_np, atol=ATOL_NORMAL, rtol=0
        )

    def test_batched_matrices(self):
        """Test batched matrix logarithm."""
        batch_size = 5
        np.random.seed(42)
        L_batch = np.random.randn(batch_size, 3, 3)
        T_batch_np = np.array(
            [L_batch[i] @ L_batch[i].T + 0.5 * np.eye(3) for i in range(batch_size)]
        )
        T_batch_torch = torch.tensor(T_batch_np, dtype=torch.float64)

        # scipy unbatched
        log_batch_scipy = np.array(
            [scipy.linalg.logm(T_batch_np[i]).real for i in range(batch_size)]
        )

        # torch_sim batched
        log_batch_torch = matrix_log_33(T_batch_torch)

        np.testing.assert_allclose(
            log_batch_torch.numpy(), log_batch_scipy, atol=ATOL_NORMAL, rtol=0
        )

    @pytest.mark.parametrize("eps", [1e-2, 1e-4, 1e-6, 1e-8])
    def test_near_identity(self, eps):
        """Test log of near-identity matrices with various perturbation sizes."""
        M_np = np.eye(3) + eps * np.array(
            [[0.1, 0.2, 0.1], [0.2, 0.15, 0.05], [0.1, 0.05, 0.2]]
        )
        M_torch = torch.tensor(M_np, dtype=torch.float64)

        log_scipy = scipy.linalg.logm(M_np).real
        log_torch = matrix_log_33(M_torch)

        np.testing.assert_allclose(
            log_torch.numpy(), log_scipy, atol=ATOL_NORMAL, rtol=0
        )


class TestDeformGrad:
    """Tests for deformation gradient computation."""

    def test_identity_deformation(self):
        """Test deformation gradient for identity deformation."""
        ref_cell = torch.tensor(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=torch.float64
        )
        current_cell = ref_cell.clone()

        F = deform_grad(ref_cell.T, current_cell)

        np.testing.assert_allclose(
            F.numpy(), np.eye(3), atol=ATOL_STRICT, rtol=0
        )

    @pytest.mark.parametrize("stretch_factor", [0.9, 1.0, 1.05, 1.1, 1.5])
    def test_uniaxial_stretch(self, stretch_factor):
        """Test deformation gradient for uniaxial stretch."""
        ref_cell = torch.tensor(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=torch.float64
        )
        current_cell = ref_cell.clone()
        current_cell[0, 0] *= stretch_factor

        F = deform_grad(ref_cell.T, current_cell)

        expected = np.eye(3)
        expected[0, 0] = stretch_factor
        np.testing.assert_allclose(F.numpy(), expected, atol=ATOL_STRICT, rtol=0)

    @pytest.mark.parametrize("shear", [0.01, 0.05, 0.1])
    def test_shear_deformation(self, shear):
        """Test deformation gradient for shear deformation."""
        ref_cell = torch.tensor(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=torch.float64
        )
        current_cell = ref_cell.clone()
        current_cell[0, 1] = shear * ref_cell[1, 1]

        F = deform_grad(ref_cell.T, current_cell)

        expected = np.eye(3)
        expected[1, 0] = shear  # Note: transpose in deform_grad
        np.testing.assert_allclose(F.numpy(), expected, atol=ATOL_STRICT, rtol=0)

    def test_batched_deformation(self):
        """Test batched deformation gradient computation."""
        batch_size = 3
        ref_cell = torch.tensor(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=torch.float64
        )
        ref_cells = ref_cell.unsqueeze(0).expand(batch_size, -1, -1).clone()
        current_cells = ref_cells.clone()

        stretches = torch.tensor([1.0, 1.05, 1.1], dtype=torch.float64)
        for i in range(batch_size):
            current_cells[i, 0, 0] *= stretches[i]

        F_batch = deform_grad(ref_cells.transpose(-1, -2), current_cells)

        for i in range(batch_size):
            expected = np.eye(3)
            expected[0, 0] = stretches[i].item()
            np.testing.assert_allclose(
                F_batch[i].numpy(), expected, atol=ATOL_STRICT, rtol=0
            )


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
        cell_forces_scipy = forces_flat_scipy.reshape(n_systems, 3, 3) / cell_factor.numpy()

        np.testing.assert_allclose(
            cell_forces_torch.numpy(), cell_forces_scipy, atol=ATOL_NORMAL, rtol=0
        )


class TestRoundTripConsistency:
    """Tests for round-trip consistency exp(log(M)) = M."""

    @pytest.mark.parametrize("seed", range(20))
    def test_positive_definite_roundtrip(self, seed):
        """Test that exp(log(M)) = M for positive definite matrices."""
        np.random.seed(seed)
        L = np.random.randn(3, 3)
        M_np = L @ L.T + 0.5 * np.eye(3)
        M_torch = torch.tensor(M_np, dtype=torch.float64)

        log_torch = matrix_log_33(M_torch)
        M_recovered = torch.matrix_exp(log_torch)

        np.testing.assert_allclose(
            M_recovered.numpy(), M_np, atol=ATOL_RELAXED, rtol=0
        )

    def test_batched_roundtrip(self):
        """Test round-trip for batched matrices."""
        batch_size = 10
        np.random.seed(42)

        M_batch = []
        for _ in range(batch_size):
            L = np.random.randn(3, 3)
            M_batch.append(L @ L.T + 0.5 * np.eye(3))
        M_batch_np = np.array(M_batch)
        M_batch_torch = torch.tensor(M_batch_np, dtype=torch.float64)

        log_batch = matrix_log_33(M_batch_torch)
        M_recovered = torch.matrix_exp(log_batch)

        np.testing.assert_allclose(
            M_recovered.numpy(), M_batch_np, atol=ATOL_RELAXED, rtol=0
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
