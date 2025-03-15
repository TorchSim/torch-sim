"""Utility functions for the torchsim package."""

from collections.abc import Callable, Iterable
from functools import wraps

import numpy as np
import torch


def safe_mask(
    mask: torch.Tensor,
    fn: torch.jit.ScriptFunction,
    operand: torch.Tensor,
    placeholder: float = 0.0,
) -> torch.Tensor:
    """Safely applies a function to masked values in a tensor.

    This function applies the given function only to elements where the mask is True,
    avoiding potential numerical issues with masked-out values. Masked-out positions
    are filled with the placeholder value.

    Args:
        mask: Boolean tensor indicating which elements to process (True) or mask (False)
        fn: TorchScript function to apply to the masked elements
        operand: Input tensor to apply the function to
        placeholder: Value to use for masked-out positions (default: 0.0)

    Returns:
        torch.Tensor: Result tensor where fn is applied to masked elements and
            placeholder value is used for masked-out elements

    Example:
        >>> x = torch.tensor([1.0, 2.0, -1.0])
        >>> mask = torch.tensor([True, True, False])
        >>> safe_mask(mask, torch.log, x)
        tensor([0.0000, 0.6931, 0.0000])
    """
    masked = torch.where(mask, operand, torch.zeros_like(operand))
    return torch.where(mask, fn(masked), torch.full_like(operand, placeholder))


def high_precision_sum(
    x: torch.Tensor,
    dim: int | Iterable[int] | None = None,
    *,
    keepdim: bool = False,
) -> torch.Tensor:
    """Sums tensor elements over specified dimensions at 64-bit precision.

    This function casts the input tensor to a higher precision type (64-bit),
    performs the summation, and then casts back to the original dtype. This helps
    prevent numerical instability issues that can occur when summing many numbers,
    especially with floating point values.

    Args:
        x: Input tensor to sum
        dim: Dimension(s) along which to sum. If None, sum over all dimensions
        keepdim: If True, retains reduced dimensions with length 1

    Returns:
        torch.Tensor: Sum of elements cast back to original dtype

    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        >>> high_precision_sum(x)
        tensor(6., dtype=torch.float32)
    """
    if torch.is_complex(x):
        high_precision_dtype = torch.complex128
    elif torch.is_floating_point(x):
        high_precision_dtype = torch.float64
    else:  # integer types
        high_precision_dtype = torch.int64

    # Cast to high precision, sum, and cast back to original dtype
    return torch.sum(x.to(high_precision_dtype), dim=dim, keepdim=keepdim).to(x.dtype)


def diagonal_mask(x: torch.Tensor) -> torch.Tensor:
    """Sets the diagonal elements of a matrix to zero.

    This function creates a mask that zeros out the diagonal elements of a square matrix
    or batch of square matrices. It handles 2D (single matrix) and 3D (batch of matrices)
    tensors.

    Args:
        x: Input tensor of shape (N, N) or (N, N, C) where N is the matrix dimension
           and C is an optional channel dimension

    Returns:
        torch.Tensor: Input tensor with diagonal elements set to zero

    Raises:
        ValueError: If input tensor is not square or has rank other than 2 or 3

    Example:
        >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> diagonal_mask(x)
        tensor([[0., 2.],
                [3., 0.]])
    """
    if x.shape[0] != x.shape[1]:
        msg = f"Diagonal mask requires square matrices. Found {x.shape[0]}x{x.shape[1]}"
        raise ValueError(msg)

    if len(x.shape) > 3:
        msg = (
            f"Diagonal mask only supports rank-2 or rank-3 tensors. "
            f"Found rank {len(x.shape)}"
        )
        raise ValueError(msg)

    # Replace NaN values with 0
    x = torch.nan_to_num(x, nan=0.0)

    # Create mask that's 1 everywhere except diagonal
    mask = 1.0 - torch.eye(x.shape[0], dtype=x.dtype, device=x.device)

    # For 3D tensors, reshape mask to broadcast across channels
    if len(x.shape) == 3:
        mask = mask.unsqueeze(-1)

    return mask * x


def compute_angles_3d(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate all possible angles between atoms given their 3D coordinates.

    Args:
        positions: torch.Tensor of shape (n_atoms, 3) containing XYZ coordinates

    Returns:
        angles: torch.Tensor containing angles in radians
        angle_indices: torch.Tensor containing atom indices for each angle
            (i-j-k where j is center atom)
    """
    n_atoms = positions.shape[0]
    if n_atoms < 3:
        return torch.tensor([], device=positions.device), torch.tensor(
            [], device=positions.device, dtype=torch.long
        )

    # Create indices for all possible combinations
    i, j, k = torch.meshgrid(
        torch.arange(n_atoms),
        torch.arange(n_atoms),
        torch.arange(n_atoms),
        indexing="ij",
    )

    # Flatten and stack the indices
    indices = torch.stack([i.flatten(), j.flatten(), k.flatten()], dim=1)

    # Filter out invalid combinations:
    # 1. Remove cases where any two indices are the same
    # 2. Keep only cases where i < k to avoid duplicates
    # 3. Ensure j is the center atom
    mask = (
        (indices[:, 0] != indices[:, 1])
        & (indices[:, 1] != indices[:, 2])
        & (indices[:, 0] != indices[:, 2])
        & (indices[:, 0] < indices[:, 2])
    )

    angle_indices = indices[mask]

    if len(angle_indices) == 0:
        return torch.tensor([], device=positions.device), angle_indices

    # Get coordinates for each atom in the angles
    coords_i = positions[angle_indices[:, 0]]
    coords_j = positions[angle_indices[:, 1]]  # center atoms
    coords_k = positions[angle_indices[:, 2]]

    # Calculate vectors from center atom to the other two atoms
    v1 = coords_i - coords_j
    v2 = coords_k - coords_j

    # Normalize vectors
    v1_norm = torch.nn.functional.normalize(v1, dim=1)
    v2_norm = torch.nn.functional.normalize(v2, dim=1)

    # Calculate dot product
    dot_product = torch.sum(v1_norm * v2_norm, dim=1)

    # Clip values to avoid numerical errors
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Calculate angles in radians
    angles = torch.acos(dot_product)

    return angles, angle_indices


def compute_angles_3d_nl(
    positions: torch.Tensor,
    neighbor_list: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate angles between atoms given their 3D coordinates and a neighbor list.

    Args:
        positions: torch.Tensor of shape (n_atoms, 3) containing XYZ coordinates
        neighbor_list: torch.Tensor of shape (n_atoms, max_neighbors) containing indices
            of neighboring atoms for each atom. Padded with -1 for atoms with fewer
            neighbors.

    Returns:
        angles: torch.Tensor containing angles in radians
        angle_indices: torch.Tensor containing atom indices for each angle (i-j-k where
            j is center atom)
    """
    n_atoms, max_neighbors = neighbor_list.shape
    if n_atoms < 3:
        return torch.tensor([], device=positions.device), torch.tensor(
            [], device=positions.device, dtype=torch.long
        )

    device = positions.device

    # Create valid neighbor mask
    valid_mask = neighbor_list >= 0  # [n_atoms, max_neighbors]

    # Create center atom indices that will be repeated
    center_atoms = torch.arange(n_atoms, device=device)

    # Create all possible neighbor pairs for each center atom
    # First expand center_atoms to match neighbor pairs dimension
    center_expanded = (
        center_atoms.unsqueeze(1)
        .unsqueeze(2)
        .expand(n_atoms, max_neighbors, max_neighbors)
    )

    # Create neighbor pairs using meshgrid
    n1, n2 = torch.meshgrid(
        torch.arange(max_neighbors, device=device),
        torch.arange(max_neighbors, device=device),
        indexing="ij",
    )

    # Get neighbor indices for pairs
    neighbor1 = neighbor_list[:, n1]  # [n_atoms, max_neighbors, max_neighbors]
    neighbor2 = neighbor_list[:, n2]  # [n_atoms, max_neighbors, max_neighbors]

    # Create validity mask for pairs
    valid_pairs = (
        valid_mask[:, n1]  # first neighbor is valid
        & valid_mask[:, n2]  # second neighbor is valid
        & (neighbor1 < neighbor2)  # avoid duplicates and self-pairs
    )

    # Flatten and filter valid triplets
    center_flat = center_expanded[valid_pairs]
    n1_flat = neighbor1[valid_pairs]
    n2_flat = neighbor2[valid_pairs]

    if len(center_flat) == 0:
        return torch.tensor([], device=device), torch.tensor(
            [], device=device, dtype=torch.long
        )

    # Calculate vectors from center to neighbors
    v1 = positions[n1_flat] - positions[center_flat]
    v2 = positions[n2_flat] - positions[center_flat]

    # Normalize vectors
    v1_norm = torch.nn.functional.normalize(v1, dim=1)
    v2_norm = torch.nn.functional.normalize(v2, dim=1)

    # Calculate dot product and angles
    dot_product = torch.sum(v1_norm * v2_norm, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    angles = torch.acos(dot_product)

    # Create angle indices tensor
    angle_indices = torch.stack([n1_flat, center_flat, n2_flat], dim=1)

    return angles, angle_indices


@torch.jit.script
def matrix_log(matrix: torch.Tensor, order: int = 10) -> torch.Tensor:
    """Compute the matrix logarithm of a square matrix using power series expansion.

    Args:
        matrix: A square matrix of shape (..., n, n)
        order: Number of terms to use in the power series expansion
    Returns:
        The matrix logarithm of the input matrix using power series expansion:
        log(I + X) = X - X^2/2 + X^3/3 - X^4/4 + ...
    """
    identity = torch.eye(matrix.shape[-1], device=matrix.device).expand_as(matrix)
    X = matrix - identity
    result = torch.zeros_like(matrix)
    X_power = X

    # Use first 10 terms of power series
    for n in range(1, order + 1):
        result = result + ((-1) ** (n + 1)) * X_power / n
        X_power = torch.matmul(X_power, X)

    return result


def calculate_canonical_atomic_masses() -> torch.Tensor:
    """Get canonical atomic masses for elements.

    Returns a tensor where index corresponds to atomic number and value is atomic mass.
    If both pymatgen and ASE are available, returns average of both.
    Falls back to individual sources if only one is available.

    Returns:
        torch.Tensor: Atomic masses indexed by atomic number, shape [n_elements]
    """
    pmg_ase_avg_masses = {
        "H": 1.0079699754714966,
        "He": 4.0026021003723145,
        "Li": 6.940499782562256,
        "Be": 9.012182235717773,
        "B": 10.810500144958496,
        "C": 12.010849952697754,
        "N": 14.006850242614746,
        "O": 15.999199867248535,
        "F": 18.998403549194336,
        "Ne": 20.17970085144043,
        "Na": 22.989768981933594,
        "Mg": 24.30500030517578,
        "Al": 26.981538772583008,
        "Si": 28.085250854492188,
        "P": 30.97376251220703,
        "S": 32.0625,
        "Cl": 35.451499938964844,
        "Ar": 39.948001861572266,
        "K": 39.09830093383789,
        "Ca": 40.077999114990234,
        "Sc": 44.955909729003906,
        "Ti": 47.867000579833984,
        "V": 50.94150161743164,
        "Cr": 51.99610137939453,
        "Mn": 54.938045501708984,
        "Fe": 55.845001220703125,
        "Co": 58.93319320678711,
        "Ni": 58.69340133666992,
        "Cu": 63.54600143432617,
        "Zn": 65.39450073242188,
        "Ga": 69.7229995727539,
        "Ge": 72.63500213623047,
        "As": 74.92160034179688,
        "Se": 78.96549987792969,
        "Br": 79.90399932861328,
        "Kr": 83.7979965209961,
        "Rb": 85.4677963256836,
        "Sr": 87.62000274658203,
        "Y": 88.90584564208984,
        "Zr": 91.2239990234375,
        "Nb": 92.9063720703125,
        "Mo": 95.94499969482422,
        "Tc": 97.95360565185547,
        "Ru": 101.06999969482422,
        "Rh": 102.90550231933594,
        "Pd": 106.41999816894531,
        "Ag": 107.86820220947266,
        "Cd": 112.4124984741211,
        "In": 114.81800079345703,
        "Sn": 118.70999908447266,
        "Sb": 121.76000213623047,
        "Te": 127.5999984741211,
        "I": 126.90447235107422,
        "Xe": 131.29299926757812,
        "Cs": 132.90545654296875,
        "Ba": 137.32699584960938,
        "La": 138.9054718017578,
        "Ce": 140.11599731445312,
        "Pr": 140.90765380859375,
        "Nd": 144.24200439453125,
        "Pm": 144.9563751220703,
        "Sm": 150.36000061035156,
        "Eu": 151.96400451660156,
        "Gd": 157.25,
        "Tb": 158.92535400390625,
        "Dy": 162.5,
        "Ho": 164.93032836914062,
        "Er": 167.25900268554688,
        "Tm": 168.93421936035156,
        "Yb": 173.0469970703125,
        "Lu": 174.96690368652344,
        "Hf": 178.49000549316406,
        "Ta": 180.9478759765625,
        "W": 183.83999633789062,
        "Re": 186.20700073242188,
        "Os": 190.22999572753906,
        "Ir": 192.2169952392578,
        "Pt": 195.08399963378906,
        "Au": 196.96656799316406,
        "Hg": 200.59100341796875,
        "Tl": 204.38165283203125,
        "Pb": 207.1999969482422,
        "Bi": 208.9803924560547,
        "Po": 209.4912109375,
        "At": 209.9935760498047,
        "Rn": 221.0087890625,
        "Fr": 223.00987243652344,
        "Ra": 226.01271057128906,
        "Ac": 227.0138702392578,
        "Th": 232.0378875732422,
        "Pa": 231.03587341308594,
        "U": 238.02891540527344,
        "Np": 237.02407836914062,
        "Pu": 244.0321044921875,
        "Am": 243.0306854248047,
        "Cm": 247.03517150878906,
        "Bk": 247.03515625,
        "Cf": 251.039794921875,
        "Es": 252.04150390625,
        "Fm": 257.04754638671875,
        "Md": 258.0492248535156,
        "No": 259.0505065917969,
        "Lr": 262.05499267578125,
        "Rf": 267.0610046386719,
        "Db": 268.06298828125,
        "Sg": 270.0669860839844,
        "Bh": 270.0664978027344,
        "Hs": 269.56689453125,
        "Mt": 278.0780029296875,
        "Ds": 281.0824890136719,
    }

    pmg_masses: list[float] | None = None
    ase_masses: list[float] | None = None

    # Try to get pymatgen masses
    try:
        from pymatgen.core import Element

        pmg_masses = [element.atomic_mass for element in Element]
        pmg_masses.pop(Element.D.Z)
        pmg_masses.pop(Element.T.Z)
        pmg_masses = np.array(pmg_masses)[:110]
    except ImportError:
        pass

    # Try to get ASE masses
    try:
        from ase.data import atomic_masses

        # ASE array starts with a dummy element
        ase_masses = atomic_masses[1:111]
    except ImportError:
        pass

    if pmg_masses is not None and ase_masses is not None:
        masses = (pmg_masses + ase_masses) / 2
    elif pmg_masses is not None:
        masses = pmg_masses
    elif ase_masses is not None:
        masses = ase_masses
    else:
        masses = np.array(list(pmg_ase_avg_masses.values()))

    return torch.tensor(masses, dtype=torch.float32)


def infer_atomic_numbers(
    masses: torch.Tensor, canonical_masses: torch.Tensor | None = None
) -> torch.Tensor:
    """Infer atomic numbers from atomic masses by matching to canonical masses.

    This function takes a tensor of atomic masses and matches each mass to the closest
    canonical atomic mass to determine the corresponding atomic number. The canonical
    masses can be provided or will be calculated if not specified.

    Args:
        masses: Tensor of atomic masses to match
        canonical_masses: Optional tensor of canonical atomic masses to match against.
            If None, will be calculated using calculate_canonical_atomic_masses()

    Returns:
        Tensor of inferred atomic numbers corresponding to the input masses
    """
    # could break out this function call and create
    # a global CANONICAL_MASSES variable in this file
    if canonical_masses is None:
        canonical_masses = calculate_canonical_atomic_masses()

    canonical_masses = canonical_masses.to(masses.device)

    # Compute pairwise absolute differences between input masses and canonical masses
    # Shape: [n_input_masses, n_canonical_masses]
    diffs = torch.abs(masses[:, None] - canonical_masses[None, :])

    # Find indices (atomic numbers - 1) of minimum differences
    # Add 1 to convert from 0-based index to atomic number
    return torch.argmin(diffs, dim=1) + 1


def multiplicative_isotropic_cutoff(
    fn: Callable[..., torch.Tensor],
    r_onset: torch.Tensor,
    r_cutoff: torch.Tensor,
) -> Callable[..., torch.Tensor]:
    """Creates a smoothly truncated version of an isotropic function.

    Takes an isotropic function f(r) and constructs a new function f'(r) that smoothly
    transitions to zero between r_onset and r_cutoff. The resulting function is C¹
    continuous (continuous in both value and first derivative).

    The truncation is achieved by multiplying the original function by a smooth
    switching function S(r) where:
    - S(r) = 1 for r < r_onset
    - S(r) = 0 for r > r_cutoff
    - S(r) smoothly transitions between 1 and 0 for r_onset < r < r_cutoff

    The switching function follows the form used in HOOMD-blue:
    S(r) = (rc² - r²)² * (rc² + 2r² - 3ro²) / (rc² - ro²)³
    where rc = r_cutoff and ro = r_onset

    Args:
        fn: Function to be truncated. Should take a tensor of distances [n, m]
            as first argument, plus optional additional arguments.
        r_onset: Distance at which the function begins to be modified.
        r_cutoff: Distance at which the function becomes zero.

    Returns:
        A new function with the same signature as fn that smoothly goes to zero
        between r_onset and r_cutoff.

    References:
        HOOMD-blue documentation:
        https://hoomd-blue.readthedocs.io/en/stable/module-md-pair.html#hoomd.md.pair.pair
    """
    r_c = r_cutoff**2
    r_o = r_onset**2

    def smooth_fn(dr: torch.Tensor) -> torch.Tensor:
        """Compute the smooth switching function."""
        r = dr**2

        # Compute switching function for intermediate region
        numerator = (r_c - r) ** 2 * (r_c + 2 * r - 3 * r_o)
        denominator = (r_c - r_o) ** 3
        intermediate = torch.where(
            dr < r_cutoff, numerator / denominator, torch.zeros_like(dr)
        )

        # Return 1 for r < r_onset, switching function for r_onset < r < r_cutoff
        return torch.where(dr < r_onset, torch.ones_like(dr), intermediate)

    @wraps(fn)
    def cutoff_fn(dr: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply the switching function to the original function."""
        return smooth_fn(dr) * fn(dr, *args, **kwargs)

    return cutoff_fn


def stillinger_weber_pair(
    dr: torch.Tensor,
    sigma: torch.Tensor,
    B: torch.Tensor,
    cutoff: torch.Tensor,
) -> torch.Tensor:
    """Calculate pairwise Stillinger-Weber two-body interaction energies.

    Implements the radial (two-body) term of the Stillinger-Weber potential,
    originally developed for modeling silicon. The potential combines an inverse
    power repulsion with an exponential cutoff.

    The functional form is:
    V(r) = (B*(r/sigma)^(-p) - 1) * exp(1/(r/sigma - a))  for r < cutoff
    V(r) = 0   for r ≥ cutoff

    where:
    - p = 4 (fixed power term)
    - a = cutoff/sigma (reduced cutoff distance)

    Args:
        dr: Pairwise distances between particles. Shape: [n, m].
        sigma: Length scale parameter. Either a scalar float or tensor
            of shape [n, m] for particle-specific scales.
        B: Energy scale parameter. Either a scalar float or tensor
            of shape [n, m] for pair-specific interaction strengths.
        cutoff: Distance beyond which potential is zero. Either a scalar float
            or tensor of shape [n, m].

    Returns:
        Pairwise Stillinger-Weber interaction energies. Shape: [n, m].
        Each element [i,j] represents the interaction energy between particles i and j.
    """
    # Calculate reduced cutoff parameter
    a = cutoff / sigma
    p = 4.0

    # Calculate the power law term
    term1 = B * (dr / sigma).pow(-p) - 1.0

    # Create mask for valid distances
    mask = (dr > 0) & (dr < cutoff)

    # Calculate exponential term (only where mask is True)
    def fn(dr: torch.Tensor) -> torch.Tensor:
        return term1 * torch.exp(1.0 / (dr / sigma - a))

    # Apply mask and handle edge cases
    return safe_mask(mask, fn, dr)


def calculate_momenta(
    positions: torch.Tensor,
    masses: torch.Tensor,
    kT: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    seed: int | None = None,
) -> torch.Tensor:
    """Calculate momenta from positions and masses."""
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    # Generate random momenta from normal distribution
    momenta = torch.randn(
        positions.shape, device=device, dtype=dtype, generator=generator
    ) * torch.sqrt(masses * kT).unsqueeze(-1)

    # Center the momentum if more than one particle
    if positions.shape[0] > 1:
        momenta = momenta - torch.mean(momenta, dim=0, keepdim=True)

    return momenta
