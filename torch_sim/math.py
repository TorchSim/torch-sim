"""Mathematical operations and utilities. Adapted from https://github.com/abhijeetgangan/torch_matfunc."""

# ruff: noqa: FBT001, FBT002, RUF002

from typing import Final

import torch


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


def expm_frechet(  # noqa: C901
    A: torch.Tensor,
    E: torch.Tensor,
    method: str | None = None,
    check_finite: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frechet derivative of the matrix exponential of A in the direction E.

    Optimized for batched 3x3 matrices. Also handles single 3x3 matrices by
    auto-adding a batch dimension.

    Method notes:
        - ``SPS`` uses scaling-Pade-squaring for the matrix exponential and its
          Frechet derivative.
        - ``blockEnlarge`` uses the block matrix identity
          exp([[A, E], [0, A]]) = [[exp(A), L_exp(A, E)], [0, exp(A)]].

    References:
        - Awad H. Al-Mohy and Nicholas J. Higham (2009), "Computing the Frechet
          Derivative of the Matrix Exponential, with an Application to Condition
          Number Estimation", SIAM J. Matrix Anal. Appl. 30(4):1639-1657.
          https://doi.org/10.1137/080716426
        - Nicholas J. Higham (2008), "Functions of Matrices: Theory and
          Computation", SIAM. (See the Frechet derivative block-matrix identity.)

    Args:
        A: (B, 3, 3) or (3, 3) tensor. Matrix of which to take the matrix exponential.
        E: (B, 3, 3) or (3, 3) tensor. Matrix direction in which to take the Frechet
            derivative. Must have same shape as A.
        method: str, optional. Choice of algorithm. Should be one of
            - `SPS` (default)
            - `blockEnlarge`
        check_finite: bool, optional. Whether to check that the input matrix contains
            only finite numbers. Disabling may give a performance gain, but may result
            in problems (crashes, non-termination) if the inputs do contain
            infinities or NaNs.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            expm_A: Matrix exponential of A.
            expm_frechet_AE: Frechet derivative of the matrix exponential of A
                in the direction E.
    """
    if check_finite:
        if not torch.isfinite(A).all():
            raise ValueError("Matrix A contains non-finite values")
        if not torch.isfinite(E).all():
            raise ValueError("Matrix E contains non-finite values")

    # Convert inputs to torch tensors if they aren't already
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float64)
    if not isinstance(E, torch.Tensor):
        E = torch.tensor(E, dtype=torch.float64)

    if A.shape != E.shape:
        raise ValueError("expected A and E to be the same shape")

    if method is None:
        method = "SPS"

    if method == "blockEnlarge":
        if A.dim() != 3 or A.shape[1] != A.shape[2]:
            raise ValueError("expected A to be (B, N, N)")
        return expm_frechet_block_enlarge(A, E)

    if method == "SPS":
        return expm_frechet_sps(A, E)
    raise ValueError(f"Unknown {method=}")


def matrix_exp(A: torch.Tensor) -> torch.Tensor:
    """Compute the matrix exponential of A using PyTorch's matrix_exp.

    Args:
        A: Input matrix

    Returns:
        torch.Tensor: Matrix exponential of A
    """
    return torch.matrix_exp(A)


def expm_frechet_sps(
    A: torch.Tensor, E: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """SPS helper for Frechet derivative of exp(A) on 3x3 matrices.

    SPS = scaling-Pade-squaring. This implementation follows the approach in:
    Awad H. Al-Mohy and Nicholas J. Higham (2009), SIAM J. Matrix Anal.
    Appl. 30(4):1639-1657. https://doi.org/10.1137/080716426
    """
    # Handle unbatched 3x3 input by adding batch dimension
    unbatched = A.dim() == 2
    if unbatched:
        if A.shape != (3, 3):
            raise ValueError("expected A to be (3, 3) or (B, 3, 3)")
        A = A.unsqueeze(0)
        E = E.unsqueeze(0)

    if A.dim() != 3 or A.shape[1:] != (3, 3):
        raise ValueError("expected A, E to be (B, 3, 3) with same shape")

    batch_size = A.shape[0]
    device, dtype = A.device, A.dtype
    ident = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, 3, 3)

    A_norm_1 = torch.norm(A, p=1, dim=(-2, -1))
    scale_val = torch.log2(
        torch.clamp(A_norm_1.max() / ell_table_61[13], min=1.0, max=2.0**64)
    )
    s = max(0, min(int(torch.ceil(scale_val).item()), 64))
    A = A * 2.0**-s
    E = E * 2.0**-s

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

    R = torch.linalg.solve(-U + V, U + V)
    L = torch.linalg.solve(-U + V, Lu + Lv + torch.matmul(Lu - Lv, R))

    for _ in range(s):
        L = torch.matmul(R, L) + torch.matmul(L, R)
        R = torch.matmul(R, R)

    if unbatched:
        return R.squeeze(0), L.squeeze(0)
    return R, L


def expm_frechet_block_enlarge(
    A: torch.Tensor, E: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-enlarge helper for Frechet derivative via matrix exponential.

    Builds M = [[A, E], [0, A]], computes exp(M), and extracts:
    - exp(A) from the top-left block
    - L_exp(A, E) from the top-right block

    Reference:
        Nicholas J. Higham (2008), "Functions of Matrices: Theory and
        Computation", SIAM. (Frechet derivative block-matrix identity.)

    Args:
        A: (B, N, N) Batch of input matrices.
        E: (B, N, N) Batch of direction matrices. Must have same shape as A.

    Returns:
        expm_A: Matrix exponential of A
        expm_frechet_AE: torch.Tensor
            Frechet derivative of the matrix exponential of A in the direction E
    """
    batch, n, _ = A.shape
    # Create block matrix M = [[A, E], [0, A]] of shape (B, 2N, 2N)
    M = torch.zeros((batch, 2 * n, 2 * n), dtype=A.dtype, device=A.device)
    M[:, :n, :n] = A
    M[:, :n, n:] = E
    M[:, n:, n:] = A

    # Use matrix exponential (supports batched input)
    expm_M = matrix_exp(M)
    return expm_M[:, :n, :n], expm_M[:, :n, n:]


# Maximal values ell_m of ||2**-s A|| such that the backward error bound
# does not exceed 2**-53.
ell_table_61: Final = (
    None,
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


def _identity_for_t(
    T: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Return identity (3, 3) or (n, 3, 3) matching T's batch shape."""
    if T.dim() == 3:
        n = T.shape[0]
        return torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(n, -1, -1)
    return torch.eye(3, dtype=dtype, device=device)


def _matrix_log_case1a(T: torch.Tensor, lambda_val: torch.Tensor) -> torch.Tensor:
    """Compute log(T) when q(T) = (T - λI).

    This is the case where T is a scalar multiple of the identity matrix.
    T may be (3, 3) or (n, 3, 3); lambda_val scalar or (n, 1, 1).

    Args:
        T: The matrix whose logarithm is to be computed
        lambda_val: The eigenvalue of T as a tensor

    Returns:
        The logarithm of T, which is log(λ)·I
    """
    dtype, device = lambda_val.dtype, lambda_val.device
    identity = _identity_for_t(T, dtype, device)
    return torch.log(lambda_val) * identity


def _matrix_log_case1b(
    T: torch.Tensor, lambda_val: torch.Tensor, num_tol: float = 1e-16
) -> torch.Tensor:
    """Compute log(T) when q(T) = (T - λI)².

    This is the case where T has a Jordan block of size 2.
    T may be (3, 3) or (n, 3, 3); lambda_val scalar or (n, 1, 1).

    Args:
        T: The matrix whose logarithm is to be computed
        lambda_val: The eigenvalue of T
        num_tol: Numerical tolerance for stability checks, default=1e-16

    Returns:
        The logarithm of T
    """
    dtype, device = lambda_val.dtype, lambda_val.device
    identity = _identity_for_t(T, dtype, device)
    T_minus_lambdaI = T - lambda_val * identity
    denom = torch.clamp(lambda_val.abs(), min=num_tol)
    scale = torch.where(lambda_val.abs() > 1, lambda_val, denom)
    return torch.log(lambda_val) * identity + T_minus_lambdaI / scale


def _ensure_batched(
    T: torch.Tensor, *eigenvalues: torch.Tensor
) -> tuple[bool, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Ensure T and eigenvalues are in batched form for matrix log computation.

    Args:
        T: Matrix of shape (3, 3) or (n, 3, 3)
        *eigenvalues: Scalar or (n, 1, 1) shaped eigenvalue tensors

    Returns:
        Tuple of (unbatched, T, eigenvalues) where unbatched is True if input was
        unbatched, T has shape (n, 3, 3), and eigenvalues have shape (n, 1, 1)
    """
    unbatched = T.dim() == 2
    if unbatched:
        T = T.unsqueeze(0)
        eigenvalues = tuple(ev.view(1, 1, 1) for ev in eigenvalues)
    return unbatched, T, eigenvalues


def _matrix_log_case1c(
    T: torch.Tensor, lambda_val: torch.Tensor, num_tol: float = 1e-16
) -> torch.Tensor:
    """Compute log(T) when q(T) = (T - λI)³.

    This is the case where T has a Jordan block of size 3.
    T may be (3, 3) or (n, 3, 3); lambda_val scalar or (n, 1, 1).

    Args:
        T: The matrix whose logarithm is to be computed
        lambda_val: The eigenvalue of T
        num_tol: Numerical tolerance for stability checks, default=1e-16

    Returns:
        The logarithm of T
    """
    unbatched, T, (lambda_val,) = _ensure_batched(T, lambda_val)
    dtype, device = lambda_val.dtype, lambda_val.device
    identity = _identity_for_t(T, dtype, device)
    T_minus_lambdaI = T - lambda_val * identity
    T_minus_lambdaI_squared = torch.bmm(T_minus_lambdaI, T_minus_lambdaI)
    lambda_squared = lambda_val * lambda_val
    term1 = torch.log(lambda_val) * identity
    term2 = T_minus_lambdaI / torch.clamp(lambda_val.abs(), min=num_tol)
    term3 = T_minus_lambdaI_squared / torch.clamp(2 * lambda_squared, min=num_tol)
    result = term1 + term2 - term3
    return result.squeeze(0) if unbatched else result


def _matrix_log_case2a(
    T: torch.Tensor, lambda_val: torch.Tensor, mu: torch.Tensor, num_tol: float = 1e-16
) -> torch.Tensor:
    """Compute log(T) when q(T) = (T - λI)(T - μI) with λ≠μ.

    This is the case with two distinct eigenvalues.
    T may be (3, 3) or (n, 3, 3); lambda_val, mu scalar or (n, 1, 1).

    Formula: log T = log μ((T - λI)/(μ - λ)) + log λ((T - μI)/(λ - μ))

    Args:
        T: The matrix whose logarithm is to be computed
        lambda_val: The repeated eigenvalue of T
        mu: The non-repeated eigenvalue of T
        num_tol: Numerical tolerance for stability checks, default=1e-16

    Returns:
        The logarithm of T

    Raises:
        ValueError: If λ and μ are too close
    """
    unbatched, T, (lambda_val, mu) = _ensure_batched(T, lambda_val, mu)
    dtype, device = lambda_val.dtype, lambda_val.device
    identity = _identity_for_t(T, dtype, device)
    if (torch.abs(lambda_val - mu) < num_tol).any():
        raise ValueError("λ and μ are too close, computation may be unstable")
    T_minus_lambdaI = T - lambda_val * identity
    T_minus_muI = T - mu * identity
    term1 = torch.log(mu) * (T_minus_lambdaI / (mu - lambda_val))
    term2 = torch.log(lambda_val) * (T_minus_muI / (lambda_val - mu))
    result = term1 + term2
    return result.squeeze(0) if unbatched else result


def _matrix_log_case2b(
    T: torch.Tensor, lambda_val: torch.Tensor, mu: torch.Tensor, num_tol: float = 1e-16
) -> torch.Tensor:
    """Compute log(T) when q(T) = (T - μI)(T - λI)² with λ≠μ.

    This is the case with one eigenvalue of multiplicity 2 and one distinct.
    T may be (3, 3) or (n, 3, 3); lambda_val, mu scalar or (n, 1, 1).

    Formula: log T = log μ((T - λI)²/(λ - μ)²) -
             log λ((T - μI)(T - (2λ - μ)I)/(λ - μ)²) +
             ((T - λI)(T - μI)/(λ(λ - μ)))

    Args:
        T: The matrix whose logarithm is to be computed
        lambda_val: The repeated eigenvalue of T
        mu: The non-repeated eigenvalue of T
        num_tol: Numerical tolerance for stability checks, default=1e-16

    Returns:
        The logarithm of T

    Raises:
        ValueError: If λ and μ are too close or λ≈0
    """
    unbatched, T, (lambda_val, mu) = _ensure_batched(T, lambda_val, mu)
    dtype, device = lambda_val.dtype, lambda_val.device
    identity = _identity_for_t(T, dtype, device)
    lambda_minus_mu = lambda_val - mu
    if (torch.abs(lambda_minus_mu) < num_tol).any():
        raise ValueError("λ and μ are too close, computation may be unstable")
    if (torch.abs(lambda_val) < num_tol).any():
        raise ValueError("λ is too close to zero, computation may be unstable")
    lambda_minus_mu_squared = lambda_minus_mu * lambda_minus_mu
    T_minus_lambdaI = T - lambda_val * identity
    T_minus_muI = T - mu * identity
    T_minus_lambdaI_squared = torch.bmm(T_minus_lambdaI, T_minus_lambdaI)
    T_minus_2lambda_plus_muI = T - (2 * lambda_val - mu) * identity
    term2_mat = torch.bmm(T_minus_muI, T_minus_2lambda_plus_muI)
    term1 = torch.log(mu) * (T_minus_lambdaI_squared / lambda_minus_mu_squared)
    term2 = -torch.log(lambda_val) * (term2_mat / lambda_minus_mu_squared)
    term3_mat = torch.bmm(T_minus_lambdaI, T_minus_muI)
    term3 = term3_mat / (lambda_val * lambda_minus_mu)
    result = term1 + term2 + term3
    return result.squeeze(0) if unbatched else result


def _matrix_log_case3(
    T: torch.Tensor,
    lambda_val: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    num_tol: float = 1e-16,
) -> torch.Tensor:
    """Compute log(T) when q(T) = (T - λI)(T - μI)(T - νI) with λ≠μ≠ν≠λ.

    This is the case with three distinct eigenvalues.
    T may be (3, 3) or (n, 3, 3); lambda_val, mu, nu scalar or (n, 1, 1).

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
        ValueError: If eigenvalues are too close
    """
    unbatched, T, (lambda_val, mu, nu) = _ensure_batched(T, lambda_val, mu, nu)
    dtype, device = lambda_val.dtype, lambda_val.device
    identity = _identity_for_t(T, dtype, device)
    min_diff = torch.minimum(
        torch.minimum(
            torch.abs(lambda_val - mu),
            torch.abs(lambda_val - nu),
        ),
        torch.abs(mu - nu),
    )
    if (min_diff < num_tol).any():
        raise ValueError("Eigenvalues are too close, computation may be unstable")
    T_minus_lambdaI = T - lambda_val * identity
    T_minus_muI = T - mu * identity
    T_minus_nuI = T - nu * identity
    lambda_term_num = torch.bmm(T_minus_muI, T_minus_nuI)
    lambda_term = torch.log(lambda_val) * (
        lambda_term_num / ((lambda_val - mu) * (lambda_val - nu))
    )
    mu_term_num = torch.bmm(T_minus_lambdaI, T_minus_nuI)
    mu_term = torch.log(mu) * (mu_term_num / ((mu - lambda_val) * (mu - nu)))
    nu_term_num = torch.bmm(T_minus_lambdaI, T_minus_muI)
    nu_term = torch.log(nu) * (nu_term_num / ((nu - lambda_val) * (nu - mu)))
    result = lambda_term + mu_term + nu_term
    return result.squeeze(0) if unbatched else result


def _determine_matrix_log_cases(
    T: torch.Tensor,
    sorted_eig: torch.Tensor,
    diff: torch.Tensor,
    n_unique: torch.Tensor,
    valid: torch.Tensor,
    num_tol: float,
) -> torch.Tensor:
    """Determine which matrix log case applies to each system.

    Args:
        T: Input matrices of shape (n_systems, 3, 3)
        sorted_eig: Sorted eigenvalues of shape (n_systems, 3)
        diff: Differences between consecutive eigenvalues (n_systems, 2)
        n_unique: Number of unique eigenvalues per system (n_systems,)
        valid: Boolean mask of valid systems (n_systems,)
        num_tol: Numerical tolerance

    Returns:
        Case indices: 0=case1a, 1=case1b, 2=case1c, 3=case2a, 4=case2b, 5=case3,
        -1=fallback
    """
    n_systems = T.shape[0]
    device, dtype_out = T.device, T.dtype
    case_indices = torch.full((n_systems,), -1, dtype=torch.long, device=device)

    if not valid.any():
        return case_indices

    eye3 = torch.eye(3, dtype=dtype_out, device=device).unsqueeze(0)

    # Case 1: all eigenvalues equal
    m1 = valid & (n_unique == 1)
    if m1.any():
        lam = sorted_eig[:, 0:1].unsqueeze(-1)
        T_lam = T - lam * eye3
        rank1 = torch.linalg.matrix_rank(T_lam)
        rank2 = torch.linalg.matrix_rank(torch.bmm(T_lam, T_lam))
        case_indices.masked_fill_(m1 & (rank1 == 0), 0)
        case_indices.masked_fill_(m1 & (rank1 > 0) & (rank2 == 0), 1)
        case_indices.masked_fill_(m1 & (rank1 > 0) & (rank2 > 0), 2)

    # Case 2: two distinct eigenvalues
    m2 = valid & (n_unique == 2)
    if m2.any():
        lam_rep = torch.where(
            diff[:, 0:1] <= num_tol, sorted_eig[:, 0:1], sorted_eig[:, 2:3]
        ).unsqueeze(-1)
        mu_val = torch.where(
            diff[:, 0:1] <= num_tol, sorted_eig[:, 2:3], sorted_eig[:, 0:1]
        ).unsqueeze(-1)
        M = torch.bmm(T - mu_val * eye3, torch.bmm(T - lam_rep * eye3, T))
        case2a = m2 & (torch.linalg.norm(M, dim=(-2, -1)) < num_tol)
        case_indices.masked_fill_(case2a, 3)
        case_indices.masked_fill_(m2 & ~case2a, 4)

    # Case 3: three distinct eigenvalues
    case_indices.masked_fill_(valid & (n_unique == 3), 5)

    return case_indices


def _process_matrix_log_case(
    case_int: int,
    idx_t: torch.Tensor,
    T_sub: torch.Tensor,
    sorted_sub: torch.Tensor,
    dtype_out: torch.dtype,
    device: torch.device,
    num_tol: float,
) -> torch.Tensor:
    """Process a single matrix log case for the given indices.

    Args:
        case_int: Case identifier (-1 to 5)
        idx_t: Indices of systems belonging to this case
        T_sub: Subset of matrices for this case
        sorted_sub: Sorted eigenvalues for this case
        dtype_out: Output dtype
        device: Device for computation
        num_tol: Numerical tolerance

    Returns:
        Computed matrix logarithms for the subset
    """
    if case_int == -1:  # Fallback to scipy for complex eigenvalues
        n_sub = idx_t.numel()
        result = torch.zeros_like(T_sub)
        for i in range(n_sub):
            result[i] = matrix_log_scipy(T_sub[i].cpu()).to(device)
    elif case_int <= 2:  # Cases 1a, 1b, 1c
        lam = sorted_sub[:, 0:1].unsqueeze(-1).to(dtype_out)
        case1_funcs = {
            0: lambda: _matrix_log_case1a(T_sub, lam),
            1: lambda: _matrix_log_case1b(T_sub, lam, num_tol),
            2: lambda: _matrix_log_case1c(T_sub, lam, num_tol),
        }
        result = case1_funcs[case_int]()
    elif case_int <= 4:  # Cases 2a, 2b
        d = sorted_sub[:, 1:2] - sorted_sub[:, 0:1]
        lam_rep = (
            torch.where(d <= num_tol, sorted_sub[:, 0:1], sorted_sub[:, 2:3])
            .unsqueeze(-1)
            .to(dtype_out)
        )
        mu_val = (
            torch.where(d <= num_tol, sorted_sub[:, 2:3], sorted_sub[:, 0:1])
            .unsqueeze(-1)
            .to(dtype_out)
        )
        case2_func = _matrix_log_case2a if case_int == 3 else _matrix_log_case2b
        result = case2_func(T_sub, lam_rep, mu_val, num_tol)
    else:  # Case 3: three distinct eigenvalues
        lam = sorted_sub[:, 0:1].unsqueeze(-1).to(dtype_out)
        mu_val = sorted_sub[:, 1:2].unsqueeze(-1).to(dtype_out)
        nu_val = sorted_sub[:, 2:3].unsqueeze(-1).to(dtype_out)
        result = _matrix_log_case3(T_sub, lam, mu_val, nu_val, num_tol)
    return result


def _matrix_log_33(T: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Compute the logarithm of 3x3 matrix T based on its eigenvalue structure.

    The logarithm of this matrix is known exactly as given the in the references.
    Supports both single matrix (3, 3) and batched input (n_systems, 3, 3).

    Args:
        T: The matrix whose logarithm is to be computed, shape (3, 3) or (n_systems, 3, 3)
        dtype: The data type to use for numerical tolerance, default=torch.float64

    Returns:
        The logarithm of T, same shape as input

    References:
        - https://link.springer.com/article/10.1007/s10659-008-9169-x
    """
    num_tol = 1e-16 if dtype == torch.float64 else 1e-8

    # Handle unbatched input by adding batch dimension
    unbatched = T.dim() == 2
    if unbatched:
        if T.shape != (3, 3):
            raise ValueError("Input must be a 3x3 matrix")
        T = T.unsqueeze(0)
    elif T.shape[1:] != (3, 3):
        raise ValueError("Batched input must have shape (n_systems, 3, 3)")

    device, dtype_out = T.device, T.dtype
    eigenvalues = torch.linalg.eigvals(T)

    # Check for complex eigenvalues - require scipy fallback
    imag_magnitude = torch.abs(torch.imag(eigenvalues))
    has_complex_eig = (imag_magnitude > num_tol).any(dim=1)
    eigenvalues_real = torch.real(eigenvalues)

    # Sort eigenvalues once for all systems
    sorted_eig, _ = torch.sort(eigenvalues_real, dim=1)
    diff = sorted_eig[:, 1:] - sorted_eig[:, :-1]
    n_unique = 1 + (diff > num_tol).sum(dim=1)
    valid = ~has_complex_eig & torch.isfinite(eigenvalues_real).all(dim=1)

    # Determine case for each system
    case_indices = _determine_matrix_log_cases(
        T, sorted_eig, diff, n_unique, valid, num_tol
    )

    # Process each case
    out = torch.zeros_like(T)
    for case_int in range(-1, 6):
        mask = case_indices == case_int
        if not mask.any():
            continue
        idx_t = mask.nonzero(as_tuple=True)[0]
        out[idx_t] = _process_matrix_log_case(
            case_int, idx_t, T[idx_t], sorted_eig[idx_t], dtype_out, device, num_tol
        )

    return out.squeeze(0) if unbatched else out


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
    device, dtype, requires_grad = matrix.device, matrix.dtype, matrix.requires_grad

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

    Also supports batched input of shape (n_systems, 3, 3).

    Args:
        matrix: A square 3x3 matrix tensor, or batch of shape (n_systems, 3, 3)
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
        if matrix.dim() == 3:
            out = torch.zeros_like(matrix, dtype=sim_dtype)
            for i in range(matrix.shape[0]):
                out[i] = matrix_log_scipy(matrix[i].cpu()).to(matrix.device).to(sim_dtype)
            return out
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

    output = torch.zeros(int(batch_indices.max()) + 1, dtype=x.dtype, device=x.device)
    output.scatter_add_(dim=0, index=batch_indices, src=(x * y).sum(dim=1))

    return output
