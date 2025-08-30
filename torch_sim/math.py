"""Mathematical operations and utilities."""

# ruff: noqa: FBT001, FBT002, RUF002, RUF003, RET503

from typing import Any, Final

import torch
from torch.autograd import Function


@torch.jit.script
def torch_divmod(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute division and modulo operations for tensors.

    Args:
        a: Dividend tensor
        b: Divisor tensor

    Returns:
        tuple containing:
            - Quotient tensor
            - Remainder tensor
    """
    d = torch.div(a, b, rounding_mode="floor")
    m = a % b
    return d, m


"""Below code is taken from https://github.com/abhijeetgangan/torch_matfunc"""

def expm_frechet(
    A: torch.Tensor,
    E: torch.Tensor,
    method: str | None = None,
    check_finite: bool = True,
) -> torch.Tensor:
    """Frechet derivative of the matrix exponential of A in the direction E.

    Args:
        A: (N, N) array_like. Matrix of which to take the matrix exponential.
        E: (N, N) array_like. Matrix direction in which to take the Frechet derivative.
        method: str, optional. Choice of algorithm. Should be one of
            - `SPS` (default)
            - `blockEnlarge`
        check_finite: bool, optional. Whether to check that the input matrix contains
            only finite numbers. Disabling may give a performance gain, but may result
            in problems (crashes, non-termination) if the inputs do contain
            infinities or NaNs.

    Returns: ndarray. Frechet derivative of the matrix exponential of A in the direction E.
    """
    return expm_frechet_with_expm_A(A, E, method, check_finite)[1]


def expm_frechet_with_expm_A(
    A: torch.Tensor,
    E: torch.Tensor,
    method: str | None = None,
    check_finite: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frechet derivative of the matrix exponential of A in the direction E.

    Args:
        A: (N, N) array_like. Matrix of which to take the matrix exponential.
        E: (N, N) array_like. Matrix direction in which to take the Frechet derivative.
        method: str, optional. Choice of algorithm. Should be one of
            - `SPS` (default)
            - `blockEnlarge`
        check_finite: bool, optional. Whether to check that the input matrix contains
            only finite numbers. Disabling may give a performance gain, but may result
            in problems (crashes, non-termination) if the inputs do contain
            infinities or NaNs.

    Returns:
        expm_A: ndarray. Matrix exponential of A.
        expm_frechet_AE: ndarray. Frechet derivative of the matrix exponential of A
            in the direction E.
    """
    if check_finite:
        if not torch.isfinite(A).all():
            raise ValueError("Matrix A contains non-finite values")
        if not torch.isfinite(E).all():
            raise ValueError("Matrix E contains non-finite values")

    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected A to be a square matrix")
    if E.dim() != 2 or E.shape[0] != E.shape[1]:
        raise ValueError("expected E to be a square matrix")
    if A.shape != E.shape:
        raise ValueError("expected A and E to be the same shape")

    if method is None:
        method = "SPS"

    if method == "SPS":
        expm_A, expm_frechet_AE = expm_frechet_algo_64(A, E)
    elif method == "blockEnlarge":
        expm_A, expm_frechet_AE = expm_frechet_block_enlarge(A, E)
    else:
        raise ValueError(f"Unknown implementation {method}")

    return expm_A, expm_frechet_AE


def expm_frechet_block_enlarge(
    A: torch.Tensor, E: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper function for testing and profiling.

    Args:
        A: Input matrix
        E: Direction matrix

    Returns:
        expm_A: Matrix exponential of A
        expm_frechet_AE: torch.Tensor
            Frechet derivative of the matrix exponential of A in the direction E
    """
    n = A.shape[0]
    # Create block matrix M = [[A, E], [0, A]]
    M = torch.zeros((2 * n, 2 * n), dtype=A.dtype, device=A.device)
    M[:n, :n] = A
    M[:n, n:] = E
    M[n:, n:] = A

    # Use matrix exponential
    expm_M = matrix_exp(M)
    return expm_M[:n, :n], expm_M[:n, n:]


# Maximal values ell_m of ||2**-s A|| such that the backward error bound
# does not exceed 2**-53.
ell_table_61: Final = (
    0,
    # 1
    2.11e-8,
    3.56e-4,
    1.08e-2,
    6.49e-2,
    2.00e-1,
    4.37e-1,
    7.83e-1,
    1.23e0,
    1.78e0,
    2.42e0,
    # 11
    3.13e0,
    3.90e0,
    4.74e0,
    5.63e0,
    6.56e0,
    7.52e0,
    8.53e0,
    9.56e0,
    1.06e1,
    1.17e1,
)


def _diff_pade3(
    A: torch.Tensor, E: torch.Tensor, ident: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Padé approximation of order 3 for matrix exponential and
    its Frechet derivative.

    Args:
        A: Input matrix
        E: Direction matrix
        ident: Identity matrix of same shape as A

    Returns:
        U, V, Lu, Lv: Components needed for computing the matrix exponential and
        its Frechet derivative
    """
    b = (120.0, 60.0, 12.0, 1.0)
    A2 = torch.matmul(A, A)
    M2 = torch.matmul(A, E) + torch.matmul(E, A)
    U = torch.matmul(A, b[3] * A2 + b[1] * ident)
    V = b[2] * A2 + b[0] * ident
    Lu = torch.matmul(A, b[3] * M2) + torch.matmul(E, b[3] * A2 + b[1] * ident)
    Lv = b[2] * M2
    return U, V, Lu, Lv


def _diff_pade5(
    A: torch.Tensor, E: torch.Tensor, ident: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Padé approximation of order 5 for matrix exponential and
    its Frechet derivative.

    Args:
        A: Input matrix
        E: Direction matrix
        ident: Identity matrix of same shape as A

    Returns:
        U, V, Lu, Lv: Components needed for computing the matrix exponential and
        its Frechet derivative
    """
    b = (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0)
    A2 = torch.matmul(A, A)
    M2 = torch.matmul(A, E) + torch.matmul(E, A)
    A4 = torch.matmul(A2, A2)
    M4 = torch.matmul(A2, M2) + torch.matmul(M2, A2)
    U = torch.matmul(A, b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[4] * A4 + b[2] * A2 + b[0] * ident
    Lu = torch.matmul(A, b[5] * M4 + b[3] * M2) + torch.matmul(
        E, b[5] * A4 + b[3] * A2 + b[1] * ident
    )
    Lv = b[4] * M4 + b[2] * M2
    return U, V, Lu, Lv


def _diff_pade7(
    A: torch.Tensor, E: torch.Tensor, ident: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Padé approximation of order 7 for matrix exponential and
    its Frechet derivative.

    Args:
        A: Input matrix
        E: Direction matrix
        ident: Identity matrix of same shape as A

    Returns:
        U, V, Lu, Lv: Components needed for computing the matrix exponential and
        its Frechet derivative
    """
    b = (17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0)
    A2 = torch.matmul(A, A)
    M2 = torch.matmul(A, E) + torch.matmul(E, A)
    A4 = torch.matmul(A2, A2)
    M4 = torch.matmul(A2, M2) + torch.matmul(M2, A2)
    A6 = torch.matmul(A2, A4)
    M6 = torch.matmul(A4, M2) + torch.matmul(M4, A2)
    U = torch.matmul(A, b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    Lu = torch.matmul(A, b[7] * M6 + b[5] * M4 + b[3] * M2) + torch.matmul(
        E, b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident
    )
    Lv = b[6] * M6 + b[4] * M4 + b[2] * M2
    return U, V, Lu, Lv


def _diff_pade9(
    A: torch.Tensor, E: torch.Tensor, ident: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Padé approximation of order 9 for matrix exponential and
    its Frechet derivative.

    Args:
        A: Input matrix
        E: Direction matrix
        ident: Identity matrix of same shape as A

    Returns:
        U, V, Lu, Lv: Components needed for computing the matrix exponential and
        its Frechet derivative
    """
    b = (
        17643225600.0,
        8821612800.0,
        2075673600.0,
        302702400.0,
        30270240.0,
        2162160.0,
        110880.0,
        3960.0,
        90.0,
        1.0,
    )
    A2 = torch.matmul(A, A)
    M2 = torch.matmul(A, E) + torch.matmul(E, A)
    A4 = torch.matmul(A2, A2)
    M4 = torch.matmul(A2, M2) + torch.matmul(M2, A2)
    A6 = torch.matmul(A2, A4)
    M6 = torch.matmul(A4, M2) + torch.matmul(M4, A2)
    A8 = torch.matmul(A4, A4)
    M8 = torch.matmul(A4, M4) + torch.matmul(M4, A4)
    U = torch.matmul(A, b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[8] * A8 + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    Lu = torch.matmul(A, b[9] * M8 + b[7] * M6 + b[5] * M4 + b[3] * M2) + torch.matmul(
        E, b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident
    )
    Lv = b[8] * M8 + b[6] * M6 + b[4] * M4 + b[2] * M2
    return U, V, Lu, Lv


def expm_frechet_algo_64(
    A: torch.Tensor, E: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute matrix exponential and its Frechet derivative using Algorithm 6.4.

    This implementation follows Al-Mohy and Higham's Algorithm 6.4 from
    "Computing the Frechet Derivative of the Matrix Exponential, with an
    application to Condition Number Estimation".

    Args:
        A: Input matrix
        E: Direction matrix

    Returns:
        R: Matrix exponential of A
        L: Frechet derivative of the matrix exponential in the direction E
    """
    n = A.shape[0]
    s = None
    ident = torch.eye(n, dtype=A.dtype, device=A.device)
    A_norm_1 = torch.norm(A, p=1)
    m_pade_pairs = (
        (3, _diff_pade3),
        (5, _diff_pade5),
        (7, _diff_pade7),
        (9, _diff_pade9),
    )

    for m, pade in m_pade_pairs:
        if A_norm_1 <= ell_table_61[m]:
            U, V, Lu, Lv = pade(A, E, ident)
            s = 0
            break

    if s is None:
        # scaling
        s = max(0, int(torch.ceil(torch.log2(A_norm_1 / ell_table_61[13]))))
        A = A * 2.0**-s
        E = E * 2.0**-s
        # pade order 13
        A2 = torch.matmul(A, A)
        M2 = torch.matmul(A, E) + torch.matmul(E, A)
        A4 = torch.matmul(A2, A2)
        M4 = torch.matmul(A2, M2) + torch.matmul(M2, A2)
        A6 = torch.matmul(A2, A4)
        M6 = torch.matmul(A4, M2) + torch.matmul(M4, A2)
        b = (
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        )
        W1 = b[13] * A6 + b[11] * A4 + b[9] * A2
        W2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident
        Z1 = b[12] * A6 + b[10] * A4 + b[8] * A2
        Z2 = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
        W = torch.matmul(A6, W1) + W2
        U = torch.matmul(A, W)
        V = torch.matmul(A6, Z1) + Z2
        Lw1 = b[13] * M6 + b[11] * M4 + b[9] * M2
        Lw2 = b[7] * M6 + b[5] * M4 + b[3] * M2
        Lz1 = b[12] * M6 + b[10] * M4 + b[8] * M2
        Lz2 = b[6] * M6 + b[4] * M4 + b[2] * M2
        Lw = torch.matmul(A6, Lw1) + torch.matmul(M6, W1) + Lw2
        Lu = torch.matmul(A, Lw) + torch.matmul(E, W)
        Lv = torch.matmul(A6, Lz1) + torch.matmul(M6, Z1) + Lz2

    # Solve the system (-U + V)X = (U + V) for R
    R = torch.linalg.solve(-U + V, U + V)

    # Solve the system (-U + V)X = (Lu + Lv + (Lu - Lv)R) for L
    L = torch.linalg.solve(-U + V, Lu + Lv + torch.matmul(Lu - Lv, R))

    # squaring
    for _ in range(s):
        L = torch.matmul(R, L) + torch.matmul(L, R)
        R = torch.matmul(R, R)

    return R, L


def matrix_exp(A: torch.Tensor) -> torch.Tensor:
    """Compute the matrix exponential of A using PyTorch's matrix_exp.

    Args:
        A: Input matrix

    Returns:
        torch.Tensor: Matrix exponential of A
    """
    return torch.matrix_exp(A)


def vec(M: torch.Tensor) -> torch.Tensor:
    """Stack columns of M to construct a single vector.

    This is somewhat standard notation in linear algebra.

    Args:
        M: Input matrix

    Returns:
        torch.Tensor: Output vector
    """
    return M.t().reshape(-1)


def expm_frechet_kronform(
    A: torch.Tensor, method: str | None = None, check_finite: bool = True
) -> torch.Tensor:
    """Construct the Kronecker form of the Frechet derivative of expm.

    Args:
        A: Square matrix tensor with shape (N, N)
        method: Optional extra keyword to be passed to expm_frechet
        check_finite: Whether to check that the input matrix contains only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns:
        torch.Tensor: Kronecker form of the Frechet derivative of the matrix exponential
            with shape (N*N, N*N)
    """
    if check_finite and not torch.isfinite(A).all():
        raise ValueError("Matrix A contains non-finite values")

    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected a square matrix")

    n = A.shape[0]
    ident = torch.eye(n, dtype=A.dtype, device=A.device)
    cols = []

    for i in range(n):
        for j in range(n):
            E = torch.outer(ident[i], ident[j])
            F = expm_frechet(A, E, method=method, check_finite=False)
            cols.append(vec(F))

    return torch.stack(cols, dim=1)


def expm_cond(A: torch.Tensor, check_finite: bool = True) -> torch.Tensor:
    """Relative condition number of the matrix exponential in the Frobenius norm.

    Args:
        A: Square input matrix with shape (N, N)
        check_finite: Whether to check that the input matrix contains only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns:
        kappa: The relative condition number of the matrix exponential
            in the Frobenius norm
    """
    if check_finite and not torch.isfinite(A).all():
        raise ValueError("Matrix A contains non-finite values")

    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected a square matrix")

    X = matrix_exp(A)
    K = expm_frechet_kronform(A, check_finite=False)

    # The following norm choices are deliberate.
    # norms of A and X are Frobenius norms, and norm of K is the induced 2-norm.
    norm_p = "fro"  # codespell:ignore
    A_norm = torch.norm(A, p=norm_p)
    X_norm = torch.norm(X, p=norm_p)
    K_norm = torch.linalg.matrix_norm(K, ord=2)

    return (K_norm * A_norm) / X_norm  # kappa


class expm(Function):  # noqa: N801
    """Compute the matrix exponential of a matrix or batch of matrices."""

    @staticmethod
    def forward(ctx: Any, A: torch.Tensor) -> torch.Tensor:
        """Compute the matrix exponential of A.

        Args:
            ctx: ctx
            A: Input matrix or batch of matrices

        Returns:
            Matrix exponential of A
        """
        # Save A for backward pass
        ctx.save_for_backward(A)
        # Use the matrix_exp function we already have
        return matrix_exp(A)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """Compute the gradient of matrix exponential.

        Args:
            ctx: ctx
            grad_output: Gradient with respect to the output

        Returns:
            Gradient with respect to the input
        """
        # Retrieve saved tensor
        (A,) = ctx.saved_tensors

        # Compute the Frechet derivative in the direction of grad_output
        return expm_frechet(
            A, grad_output, method="SPS", check_finite=False
        )


def _is_valid_matrix(T: torch.Tensor, n: int = 3) -> bool:
    """Check if T is a valid nxn matrix.

    Args:
        T: The matrix to check
        n: The expected dimension of the matrix, default=3

    Returns:
        bool: True if T is a valid nxn tensor, False otherwise
    """
    return isinstance(T, torch.Tensor) and T.shape == (n, n)


def _determine_eigenvalue_case(  # noqa: C901
    T: torch.Tensor, eigenvalues: torch.Tensor, num_tol: float = 1e-16
) -> str:
    """Determine the eigenvalue structure case of matrix T.

    Args:
        T: The 3x3 matrix to analyze
        eigenvalues: The eigenvalues of T
        num_tol: Numerical tolerance for comparing eigenvalues, default=1e-16

    Returns:
        The case identifier ("case1a", "case1b", etc.)

    Raises:
        ValueError: If the eigenvalue structure cannot be determined
    """
    # Get unique values and their counts directly with one call
    unique_vals, counts = torch.unique(eigenvalues, return_counts=True)

    # Use np.isclose to group eigenvalues that are numerically close
    # We can create a mask for each unique value to see if other values are close to it
    if len(unique_vals) > 1:
        # Check if some "unique" values should actually be considered the same
        i = 0
        while i < len(unique_vals):
            # Find all values close to the current one
            close_mask = torch.isclose(unique_vals, unique_vals[i], rtol=0, atol=num_tol)
            close_count = torch.sum(close_mask)

            if close_count > 1:  # If there are other close values
                # Merge them (keep the first one, remove the others)
                counts[i] = torch.sum(counts[close_mask])
                unique_vals = unique_vals[
                    ~(close_mask & torch.arange(len(close_mask)) != i)
                ]
                counts = counts[~(close_mask & torch.arange(len(counts)) != i)]
            else:
                i += 1

    # Now determine the case based on the number of unique eigenvalues
    if len(unique_vals) == 1:
        # Case 1: All eigenvalues are equal (λ, λ, λ)
        lambda_val = unique_vals[0]
        Identity = torch.eye(3, dtype=lambda_val.dtype, device=lambda_val.device)
        T_minus_lambdaI = T - lambda_val * Identity

        rank1 = torch.linalg.matrix_rank(T_minus_lambdaI)
        if rank1 == 0:
            return "case1a"  # q(T) = (T - λI)

        rank2 = torch.linalg.matrix_rank(T_minus_lambdaI @ T_minus_lambdaI)
        if rank2 == 0:
            return "case1b"  # q(T) = (T - λI)²

        return "case1c"  # q(T) = (T - λI)³

    if len(unique_vals) == 2:
        # Case 2: Two distinct eigenvalues
        # The counts array already tells us which eigenvalue is repeated
        if counts.max() != 2 or counts.min() != 1:
            raise ValueError("Unexpected eigenvalue pattern for Case 2")

        mu = unique_vals[torch.argmin(counts)]  # The non-repeated eigenvalue
        lambda_val = unique_vals[torch.argmax(counts)]  # The repeated eigenvalue

        Identity = torch.eye(3, dtype=lambda_val.dtype, device=lambda_val.device)
        T_minus_muI = T - mu * Identity
        T_minus_lambdaI = T - lambda_val * Identity

        # Check if (T - μI)(T - λI) annihilates T
        if torch.allclose(
            T_minus_muI @ T_minus_lambdaI @ T,
            torch.zeros((3, 3), dtype=lambda_val.dtype, device=lambda_val.device),
        ):
            return "case2a"  # q(T) = (T - λI)(T - μI)
        return "case2b"  # q(T) = (T - μI)(T - λI)²

    if len(unique_vals) == 3:
        # Case 3: Three distinct eigenvalues (λ, μ, ν)
        return "case3"  # q(T) = (T - λI)(T - μI)(T - νI)

    raise ValueError("Could not determine eigenvalue structure")


def _matrix_log_case1a(T: torch.Tensor, lambda_val: torch.Tensor) -> torch.Tensor:
    """Compute log(T) when q(T) = (T - λI).

    This is the case where T is a scalar multiple of the identity matrix.

    Args:
        T: The matrix whose logarithm is to be computed
        lambda_val: The eigenvalue of T (a complex or real tensor)

    Returns:
        The logarithm of T, which is log(λ)·I
    """
    n = T.shape[0]
    Identity = torch.eye(n, dtype=lambda_val.dtype, device=lambda_val.device)
    return torch.log(lambda_val) * Identity


def _matrix_log_case1b(
    T: torch.Tensor, lambda_val: torch.Tensor, num_tol: float = 1e-16
) -> torch.Tensor:
    """Compute log(T) when q(T) = (T - λI)².

    This is the case where T has a Jordan block of size 2.

    Args:
        T: The matrix whose logarithm is to be computed
        lambda_val: The eigenvalue of T (a complex or real tensor)
        num_tol: Numerical tolerance for stability checks, default=1e-16

    Returns:
        The logarithm of T
    """
    n = T.shape[0]
    Identity = torch.eye(n, dtype=lambda_val.dtype, device=lambda_val.device)
    T_minus_lambdaI = T - lambda_val * Identity

    # For numerical stability, scale appropriately
    if abs(lambda_val) > 1:
        scaled_T_minus_lambdaI = T_minus_lambdaI / lambda_val
        return torch.log(lambda_val) * Identity + scaled_T_minus_lambdaI
    # Alternative computation for small lambda
    return torch.log(lambda_val) * Identity + T_minus_lambdaI / max(lambda_val, num_tol) # type: ignore[call-overload]


def _matrix_log_case1c(
    T: torch.Tensor, lambda_val: torch.Tensor, num_tol: float = 1e-16
) -> torch.Tensor:
    """Compute log(T) when q(T) = (T - λI)³.

    This is the case where T has a Jordan block of size 3.

    Args:
        T: The matrix whose logarithm is to be computed
        lambda_val: The eigenvalue of T (a complex or real tensor)
        num_tol: Numerical tolerance for stability checks, default=1e-16

    Returns:
        The logarithm of T
    """
    n = T.shape[0]
    Identity = torch.eye(n, dtype=lambda_val.dtype, device=lambda_val.device)
    T_minus_lambdaI = T - lambda_val * Identity

    # Compute (T - λI)² with better numerical stability
    T_minus_lambdaI_squared = T_minus_lambdaI @ T_minus_lambdaI

    # For numerical stability
    lambda_squared = lambda_val * lambda_val

    term1 = torch.log(lambda_val) * Identity
    term2 = T_minus_lambdaI / max(lambda_val, num_tol) # type: ignore[call-overload]
    term3 = T_minus_lambdaI_squared / max(2 * lambda_squared, num_tol) # type: ignore[call-overload]

    return term1 + term2 - term3


def _matrix_log_case2a(
    T: torch.Tensor, lambda_val: torch.Tensor, mu: torch.Tensor, num_tol: float = 1e-16
) -> torch.Tensor:
    """Compute log(T) when q(T) = (T - λI)(T - μI) with λ≠μ.

    This is the case with two distinct eigenvalues.
    Formula: log T = log μ((T - λI)/(μ - λ)) + log λ((T - μI)/(λ - μ))

    Args:
        T: The matrix whose logarithm is to be computed
        lambda_val: The repeated eigenvalue of T (a complex or real tensor)
        mu: The non-repeated eigenvalue of T (a complex or real tensor)
        num_tol: Numerical tolerance for stability checks, default=1e-16

    Returns:
        The logarithm of T

    Raises:
        ValueError: If λ and μ are too close for numerical stability
    """
    n = T.shape[0]
    Identity = torch.eye(n, dtype=lambda_val.dtype, device=lambda_val.device)
    lambda_minus_mu = lambda_val - mu

    # Check for numerical stability
    if torch.abs(lambda_minus_mu) < num_tol:
        raise ValueError("λ and μ are too close, computation may be unstable")

    T_minus_lambdaI = T - lambda_val * Identity
    T_minus_muI = T - mu * Identity

    # Compute each term separately for better numerical stability
    term1 = torch.log(mu) * (T_minus_lambdaI / (mu - lambda_val))
    term2 = torch.log(lambda_val) * (T_minus_muI / (lambda_val - mu))

    return term1 + term2


def _matrix_log_case2b(
    T: torch.Tensor, lambda_val: torch.Tensor, mu: complex, num_tol: float = 1e-16
) -> torch.Tensor:
    """Compute log(T) when q(T) = (T - μI)(T - λI)² with λ≠μ.

    This is the case with one eigenvalue of multiplicity 2 and one distinct eigenvalue.
    Formula: log T = log μ((T - λI)²/(λ - μ)²) -
             log λ((T - μI)(T - (2λ - μ)I)/(λ - μ)²) +
             ((T - λI)(T - μI)/(λ(λ - μ)))

    Args:
        T: The matrix whose logarithm is to be computed
        lambda_val: The repeated eigenvalue of T (a complex or real tensor)
        mu: The non-repeated eigenvalue of T
        num_tol: Numerical tolerance for stability checks, default=1e-16

    Returns:
        The logarithm of T

    Raises:
        ValueError: If λ and μ are too close for numerical stability or
        if λ is too close to zero
    """
    n = T.shape[0]
    Identity = torch.eye(n, dtype=lambda_val.dtype, device=lambda_val.device)
    lambda_minus_mu = lambda_val - mu
    lambda_minus_mu_squared = lambda_minus_mu * lambda_minus_mu

    # Check for numerical stability
    if torch.abs(lambda_minus_mu) < num_tol:
        raise ValueError("λ and μ are too close, computation may be unstable")

    if torch.abs(lambda_val) < num_tol:
        raise ValueError("λ is too close to zero, computation may be unstable")

    T_minus_lambdaI = T - lambda_val * Identity
    T_minus_muI = T - mu * Identity
    T_minus_lambdaI_squared = T_minus_lambdaI @ T_minus_lambdaI

    # The term (T - (2λ - μ)I)
    T_minus_2lambda_plus_muI = T - (2 * lambda_val - mu) * Identity

    # Compute each term separately for better numerical stability
    term1 = torch.log(mu) * (T_minus_lambdaI_squared / lambda_minus_mu_squared)
    term2 = -torch.log(lambda_val) * (
        (T_minus_muI @ T_minus_2lambda_plus_muI) / lambda_minus_mu_squared
    )
    term3 = (T_minus_lambdaI @ T_minus_muI) / (lambda_val * lambda_minus_mu)

    return term1 + term2 + term3


def _matrix_log_case3(
    T: torch.Tensor, lambda_val: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor, num_tol: float = 1e-16
) -> torch.Tensor:
    """Compute log(T) when q(T) = (T - λI)(T - μI)(T - νI) with λ≠μ≠ν≠λ.
    This is the case with three distinct eigenvalues.

    Formula: log T = log λ((T - μI)(T - νI)/((λ - μ)(λ - ν)))
                    + log μ((T - λI)(T - νI)/((μ - λ)(μ - ν)))
                    + log ν((T - λI)(T - μI)/((ν - λ)(ν - μ)))

    Args:
        T: The matrix whose logarithm is to be computed
        lambda_val: First eigenvalue of T
        mu: Second eigenvalue of T
        nu: Third eigenvalue of T
        num_tol: Numerical tolerance for stability checks, default=1e-6

    Returns:
        The logarithm of T

    Raises:
        ValueError: If any pair of eigenvalues are too close for numerical stability
    """
    n = T.shape[0]
    Identity = torch.eye(n, dtype=lambda_val.dtype, device=lambda_val.device)

    # Check if eigenvalues are distinct enough for numerical stability
    if (
        min(torch.abs(lambda_val - mu), torch.abs(lambda_val - nu), torch.abs(mu - nu))
        < num_tol
    ):
        raise ValueError("Eigenvalues are too close, computation may be unstable")

    T_minus_lambdaI = T - lambda_val * Identity
    T_minus_muI = T - mu * Identity
    T_minus_nuI = T - nu * Identity

    # Compute the terms for λ
    lambda_term_numerator = T_minus_muI @ T_minus_nuI
    lambda_term_denominator = (lambda_val - mu) * (lambda_val - nu)
    lambda_term = torch.log(lambda_val) * (
        lambda_term_numerator / lambda_term_denominator
    )

    # Compute the terms for μ
    mu_term_numerator = T_minus_lambdaI @ T_minus_nuI
    mu_term_denominator = (mu - lambda_val) * (mu - nu)
    mu_term = torch.log(mu) * (mu_term_numerator / mu_term_denominator)

    # Compute the terms for ν
    nu_term_numerator = T_minus_lambdaI @ T_minus_muI
    nu_term_denominator = (nu - lambda_val) * (nu - mu)
    nu_term = torch.log(nu) * (nu_term_numerator / nu_term_denominator)

    return lambda_term + mu_term + nu_term


def _matrix_log_33(  # noqa: C901
    T: torch.Tensor, case: str = "auto", dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Compute the logarithm of 3x3 matrix T based on its eigenvalue structure.
    The logarithm of this matrix is known exactly as given the in the references.

    Args:
        T: The matrix whose logarithm is to be computed
        case: One of "auto", "case1a", "case1b", "case1c", "case2a", "case2b", "case3"
            - "auto": Automatically determine the structure
            - "case1a": All eigenvalues are equal, q(T) = (T - λI)
            - "case1b": All eigenvalues are equal, q(T) = (T - λI)²
            - "case1c": All eigenvalues are equal, q(T) = (T - λI)³
            - "case2a": Two distinct eigenvalues, q(T) = (T - λI)(T - μI)
            - "case2b": Two distinct eigenvalues, q(T) = (T - μI)(T - λI)²
            - "case3": Three distinct eigenvalues, q(T) = (T - λI)(T - μI)(T - νI)
        dtype: The data type to use for numerical tolerance, default=torch.float64

    Returns:
        The logarithm of T

    References:
        - https://link.springer.com/article/10.1007/s10659-008-9169-x
    """
    num_tol = 1e-16 if dtype == torch.float64 else 1e-8

    if not _is_valid_matrix(T):
        raise ValueError("Input must be a 3x3 matrix")

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(T)
    # Convert eigenvalues to real if they're complex but with tiny imaginary parts
    eigenvalues = (
        torch.real(eigenvalues)
        if torch.allclose(
            torch.imag(eigenvalues),
            torch.zeros_like(torch.imag(eigenvalues)),
            atol=num_tol,
        )
        else eigenvalues
    )

    # If automatic detection, determine the structure
    if case == "auto":
        case = _determine_eigenvalue_case(T, eigenvalues, num_tol)

    # Case 1: All eigenvalues are equal (λ, λ, λ)
    if case in ["case1a", "case1b", "case1c"]:
        lambda_val = eigenvalues[0]

        # Check for numerical stability
        if torch.abs(lambda_val) < num_tol:
            raise ValueError("Eigenvalue too close to zero, computation may be unstable")

        if case == "case1a":
            return _matrix_log_case1a(T, lambda_val)
        if case == "case1b":
            return _matrix_log_case1b(T, lambda_val, num_tol)
        if case == "case1c":
            return _matrix_log_case1c(T, lambda_val, num_tol)

    # Case 2: Two distinct eigenvalues (μ, λ, λ)
    elif case in ["case2a", "case2b"]:
        # Find the unique eigenvalue (μ) and the repeated eigenvalue (λ)
        unique_vals, counts = torch.unique(
            torch.round(eigenvalues, decimals=10), return_counts=True
        )
        if len(unique_vals) != 2 or counts.max() != 2:
            raise ValueError(
                "Case 2 requires exactly two distinct eigenvalues with one repeated"
            )

        mu = unique_vals[torch.argmin(counts)]  # The non-repeated eigenvalue
        lambda_val = unique_vals[torch.argmax(counts)]  # The repeated eigenvalue

        if case == "case2a":
            return _matrix_log_case2a(T, lambda_val, mu, num_tol)
        if case == "case2b":
            return _matrix_log_case2b(T, lambda_val, mu, num_tol)

    # Case 3: Three distinct eigenvalues (λ, μ, ν)
    elif case == "case3":
        if len(torch.unique(torch.round(eigenvalues, decimals=10))) != 3:
            raise ValueError("Case 3 requires three distinct eigenvalues")

        lambda_val, mu, nu = torch.sort(eigenvalues).values  # Sort for consistency
        return _matrix_log_case3(T, lambda_val, mu, nu, num_tol)

    raise ValueError(f"Unknown eigenvalue {case=}")


def matrix_log_scipy(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the matrix logarithm of a square matrix using scipy.linalg.logm.

    This function handles tensors on CPU or GPU and preserves gradients.

    Args:
        matrix: A square matrix tensor

    Returns:
        torch.Tensor: The matrix logarithm of the input matrix
    """
    import scipy.linalg

    # Save original device and dtype
    device = matrix.device
    dtype = matrix.dtype
    requires_grad = matrix.requires_grad

    # Detach and move to CPU for scipy
    matrix_cpu = matrix.detach().cpu().numpy()

    # Compute the logarithm using scipy
    result_np = scipy.linalg.logm(matrix_cpu)

    # Convert back to tensor and move to original device
    result = torch.tensor(result_np, dtype=dtype, device=device)

    # If input requires gradient, make the output require gradient too
    if requires_grad:
        result = result.requires_grad_()

    return result


def matrix_log_33(
    matrix: torch.Tensor,
    sim_dtype: torch.dtype = torch.float64,
    fallback_warning: bool = False,
) -> torch.Tensor:
    """Compute the matrix logarithm of a square 3x3 matrix.

    Args:
        matrix: A square 3x3 matrix tensor
        sim_dtype: Simulation dtype, default=torch.float64
        fallback_warning: Whether to print a warning when falling back to scipy,
            default=False

    Returns:
        The matrix logarithm of the input matrix

    This function attempts to use the exact formula for 3x3 matrices first,
    and falls back to scipy implementation if that fails.
    """
    # Convert to double precision for stability
    matrix = matrix.to(torch.float64)
    try:
        return _matrix_log_33(matrix).to(sim_dtype)
    except (ValueError, RuntimeError) as exc:
        msg = (
            f"Error computing matrix logarithm with _matrix_log_33 {exc} \n"
            "Falling back to scipy"
        )
        if fallback_warning:
            print(msg)  # noqa: T201
        # Fall back to scipy implementation
        return matrix_log_scipy(matrix).to(sim_dtype)


def batched_vdot(
    x: torch.Tensor, y: torch.Tensor, batch_indices: torch.Tensor
) -> torch.Tensor:
    """Computes batched vdot (sum of element-wise product) for groups of vectors.

    Args:
        x: Tensor of shape [N_total_entities, D] (e.g., forces, velocities).
        y: Tensor of shape [N_total_entities, D].
        batch_indices: Tensor of shape [N_total_entities] indicating batch membership.

    Returns:
        Tensor: shape [n_systems] where each element is the sum(x_i * y_i)
    for entities belonging to that batch,
        summed over all components D and all entities in the batch.
    """
    if (
        x.ndim != 2
        or y.ndim != 2
        or batch_indices.ndim != 1
        or x.shape != y.shape
        or x.shape[0] != batch_indices.shape[0]
    ):
        raise ValueError(f"Invalid input shapes: {x.shape=}, {batch_indices.shape=}")

    if batch_indices.min() < 0:
        raise ValueError("batch_indices must be non-negative")

    output = torch.zeros(batch_indices.max() + 1, dtype=x.dtype, device=x.device)
    output.scatter_add_(dim=0, index=batch_indices, src=(x * y).sum(dim=1))

    return output
