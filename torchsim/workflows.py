"""Workflows for generating random packed structures."""

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from pymatgen.core.composition import Composition

from torchsim.models.soft_sphere import (
    UnbatchedSoftSphereModel,
    UnbatchedSoftSphereMultiModel,
)
from torchsim.transforms import get_pair_displacements
from torchsim.unbatched_optimizers import FIREState, fire


def min_distance(
    positions: torch.Tensor,
    cell: torch.Tensor,
    distance_tolerance: float = 0.0001,
) -> torch.Tensor:
    """Calculate the minimum distance between any pair of atoms in a periodic structure.

    This function computes all pairwise distances between atoms in a periodic system and
    returns the smallest non-zero distance. Self-interactions (an atom's distance to
    itself) are excluded using the distance_tolerance parameter.

    Args:
        positions: Atomic positions tensor of shape [n_atoms, 3], where each row contains
            the (x,y,z) coordinates of an atom.
        cell: Unit cell tensor of shape [3, 3] containing the three lattice vectors that
            define the periodic boundary conditions.
        distance_tolerance: Minimum distance threshold used to exclude self-interactions.
            Distances smaller than this value are masked out. Defaults to 0.0001 Å.

    Returns:
        torch.Tensor: The minimum distance between any two different atoms in the
            structure, considering periodic boundary conditions.

    Note:
        The function uses periodic boundary conditions by default. This means atoms near
        the cell boundaries are properly connected to atoms on the opposite side of the
        cell.
    """
    # Calculate all pairwise distances between atoms, considering periodic boundaries
    # Returns both displacement vectors and scalar distances, but we only need distances
    _, distances = get_pair_displacements(
        positions=positions,
        cell=cell,
        pbc=True,  # Use periodic boundary conditions
    )

    # Create a mask for distances below tolerance to exclude self-interactions
    # These very small distances occur when an atom is compared with itself
    mask = distances < distance_tolerance

    # Replace masked distances with infinity so they won't be selected as the minimum
    distances = distances.masked_fill(mask, torch.inf)

    # Return the smallest non-masked distance
    return distances.min()


def get_diameter(composition: Composition) -> float:
    """Calculate the minimum atomic diameter for a given composition.

    This function estimates the minimum atomic diameter by considering different atomic
    radii. For multi-element compositions, it finds the minimum possible separation
    between any pair of atoms by summing their ionic radii. For single elements, it uses
    element-specific radii: metallic radius for metals, and atomic/ionic radius for
    non-metals.

    The diameter represents a reasonable estimate for the closest approach between atoms,
    which is useful for initializing atomic positions or setting cutoff distances in
    simulations.

    Args:
        composition (Composition): A pymatgen Composition object representing the chemical
            composition. Can contain any number of elements.

    Returns:
        float: The estimated minimum atomic diameter in Angstroms. For multi-element
            systems, this is the smallest possible separation between any pair of atoms.
            For single elements, it is twice the appropriate atomic radius.

    Examples:
        >>> from pymatgen.core.composition import Composition
        >>> # Multi-element example: Fe2O3
        >>> comp = Composition("Fe2O3")
        >>> diameter = get_diameter(comp)  # Returns minimum Fe-O or O-O separation
        >>> print(f"{diameter:.2f}")

        >>> # Single-element example: Cu metal
        >>> comp = Composition("Cu")
        >>> diameter = get_diameter(comp)  # Returns 2 * metallic radius
        >>> print(f"{diameter:.2f}")

    Notes:
        - For multi-element systems, uses ionic radii to handle both metals and non-metals
        - For single metals, uses metallic radius which better represents bonding
        - For single non-metals, prefers atomic radius but falls back to ionic if needed
        - All radii are obtained from pymatgen's element database
    """
    # Handle multi-element compositions
    if len(composition.elements) > 1:
        # Get all possible pairs of ionic radii and find minimum sum
        diameter = np.array(
            list(
                itertools.combinations(
                    [e.average_ionic_radius for e in composition.elements], 2
                )
            )
        )
        diameter = float(min(diameter.sum(axis=1)))
    # Handle single-element compositions
    else:
        elem = composition.elements[0]
        if elem.is_metal:
            # Use metallic radius for metals (better for metallic bonding)
            diameter = float(elem.metallic_radius) * 2
        else:
            # For non-metals, prefer atomic radius but fall back to ionic
            diameter = (
                float(elem.atomic_radius) * 2
                if elem.atomic_radius
                else float(elem.average_ionic_radius) * 2
            )
    return diameter


def get_diameter_matrix(
    composition: Composition,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Calculate the matrix of atomic diameters for a given composition.

    This function constructs a symmetric matrix containing the
    minimum atomic diameters between all pairs of elements in the
    composition. The diameters are calculated based on:
    - For metals: metallic radii
    - For non-metals: atomic radii if available, otherwise ionic radii
    - For metal-metal pairs: sum of metallic radii
    - For other pairs: sum of atomic/ionic radii

    Args:
        composition (Composition): A pymatgen Composition object representing the chemical
            composition. The composition can contain any number of elements.
        device (torch.device | None, optional): PyTorch device to place the output tensor
            on. If None, uses CPU. Defaults to None.
        dtype (torch.dtype, optional): PyTorch data type for the output tensor.
            Defaults to torch.float32.

    Returns:
        torch.Tensor: A symmetric matrix of shape [n_elements, n_elements] containing the
            minimum atomic diameters in Angstroms between each pair of elements. The value
            at [i,j] represents the minimum separation between elements i and j.

    Examples:
        >>> from pymatgen.core.composition import Composition
        >>> comp = Composition("Fe2O3")
        >>> diameters = get_diameter_matrix(comp)
        >>> print(diameters)  # Shows Fe-Fe, Fe-O, and O-O separations
    """
    # Extract unique elements and create empty diameter matrix
    elements = composition.elements
    n_elements = len(elements)
    diameter_matrix = torch.zeros((n_elements, n_elements), dtype=dtype, device=device)

    for i, elem1 in enumerate(elements):
        for j, elem2 in enumerate(elements):
            # Handle same-element pairs (diagonal elements)
            if i == j:
                if elem1.is_metal:
                    # For metals, use 2x metallic radius
                    diameter = float(elem1.metallic_radius) * 2
                else:
                    # For non-metals, prefer atomic radius, fallback to ionic
                    diameter = (
                        float(elem1.atomic_radius) * 2
                        if elem1.atomic_radius
                        else float(elem1.average_ionic_radius) * 2
                    )
            # Handle different-element pairs (off-diagonal elements)
            elif elem1.is_metal and elem2.is_metal:
                # For metal-metal pairs, sum metallic radii
                diameter = float(elem1.metallic_radius + elem2.metallic_radius)
            else:
                # For other pairs, sum atomic (preferred) or ionic radii
                radius1 = float(
                    elem1.atomic_radius
                    if elem1.atomic_radius
                    else elem1.average_ionic_radius
                )
                radius2 = float(
                    elem2.atomic_radius
                    if elem2.atomic_radius
                    else elem2.average_ionic_radius
                )
                diameter = radius1 + radius2

            # Fill both symmetric positions in the matrix
            diameter_matrix[i, j] = diameter
            diameter_matrix[j, i] = diameter

    return diameter_matrix


def random_packed_structure(
    composition: Composition,
    cell: torch.Tensor,
    *,
    seed: int = 42,
    diameter: float | None = None,
    auto_diameter: bool = False,
    max_iter: int = 100,
    distance_tolerance: float = 0.0001,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    log: Any | None = None,
) -> FIREState:
    """Generates a random packed atomic structure and minimizes atomic overlaps.

    This function creates a random atomic structure within a given cell and optionally
    minimizes atomic overlaps using a soft-sphere potential and FIRE optimization.

    Args:
        composition: A pymatgen Composition object specifying the atomic composition
            (e.g. Fe80B20). The numbers indicate actual atom counts.
        cell: A 3x3 tensor defining the triclinic simulation box in Angstroms.
            Does not need to be cubic.
        seed: Random seed for reproducible structure generation. If None, uses
            random initialization.
        diameter: The minimum allowed interatomic distance. Atoms closer than this
            distance are considered overlapping. Used for soft-sphere potential.
        auto_diameter: If True, automatically calculates appropriate diameter based
            on atomic/ionic radii from pymatgen.
        max_iter: Maximum number of FIRE optimization steps to minimize overlaps.
            Stops early if minimum distance criterion is met.
        distance_tolerance: Threshold below which atoms are considered at the same
            position when computing minimum distances.
        device: PyTorch device for calculations (CPU/GPU).
        dtype: PyTorch data type for numerical precision.
        log: List to store positions at each iteration.

    Returns:
        FIREState: The optimized structure state containing positions, forces,
            energies and other optimization parameters.

    Notes:
        - If both diameter and auto_diameter are None, no overlap minimization
          is performed.
        - The overlap minimization uses a soft-sphere potential that creates
          repulsive forces between overlapping atoms.
        - The FIRE algorithm is used to minimize the potential energy and reduce
          overlaps.
    """
    # Extract number of atoms for each element from composition
    element_counts = [int(i) for i in composition.as_dict().values()]

    # Set up reproducible random number generator
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    # Generate initial random positions in fractional coordinates
    N_atoms = sum(element_counts)
    positions = torch.rand((N_atoms, 3), device=device, dtype=dtype, generator=generator)

    # Calculate appropriate diameter if auto_diameter is enabled
    if auto_diameter:
        diameter = get_diameter(composition)
        print(f"Using random pack diameter of {diameter}")

    # Perform overlap minimization if diameter is specified
    if diameter is not None:
        print("Reduce atom overlap using the soft_sphere potential")
        # Convert fractional to cartesian coordinates
        positions_cart = torch.matmul(positions, cell)

        # Initialize soft sphere potential calculator
        model = UnbatchedSoftSphereModel(
            sigma=diameter,
            device=device,
            dtype=dtype,
            compute_force=True,
            use_neighbor_list=True,
        )

        # Dummy atomic numbers
        atomic_numbers = torch.ones_like(positions_cart, device=device, dtype=torch.int)

        # Set up FIRE optimizer with unit masses
        state = FIREState(
            positions=positions_cart,
            masses=torch.ones(N_atoms, device=device, dtype=dtype),
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=True,
        )
        state_init_fn, fire_update = fire(model=model)
        state = state_init_fn(state)
        print(f"Initial energy: {state.energy.item():.4f}")
        # Run FIRE optimization until convergence or max iterations
        for _ in range(max_iter):
            # Check if minimum distance criterion is met (95% of target diameter)
            if min_distance(state.positions, cell, distance_tolerance) > diameter * 0.95:
                break

            if log is not None:
                log.append(state.positions.cpu().numpy())

            state = fire_update(state)

        print(f"Final energy: {state.energy.item():.4f}")

    if log is not None:
        return state, log
    return state


def random_packed_structure_multi(
    composition: Composition,
    cell: torch.Tensor,
    *,
    seed: int = 42,
    diameter_matrix: torch.Tensor | None = None,
    auto_diameter: bool = False,
    max_iter: int = 100,
    distance_tolerance: float = 0.0001,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> FIREState:
    """Generates a random packed atomic structure with multiple species
    and minimizes overlaps.

    This function creates a random atomic structure with multiple atomic species within a
    given cell and optionally minimizes atomic overlaps using a species-specific
    soft-sphere potential and FIRE optimization. The interatomic distances can be
    controlled separately for each pair of species.

    Args:
        composition: A pymatgen Composition object specifying the atomic composition
            (e.g. Fe80B20). The numbers indicate actual atom counts.
        cell: A 3x3 tensor defining the triclinic simulation box in Angstroms.
            Does not need to be cubic.
        seed: Random seed for reproducible structure generation. If None, uses
            random initialization. Defaults to 42.
        diameter_matrix: A symmetric matrix of shape [n_species, n_species] specifying
            the minimum allowed distance between each pair of species. If None and
            auto_diameter is False, no overlap minimization is performed.
        auto_diameter: If True, automatically calculates appropriate diameters based
            on atomic/ionic radii from pymatgen. Defaults to False.
        max_iter: Maximum number of FIRE optimization steps to minimize overlaps.
            Stops early if minimum distance criterion is met. Defaults to 30.
        distance_tolerance: Threshold below which atoms are considered at the same
            position when computing minimum distances. Defaults to 0.0001.
        device: PyTorch device for calculations (CPU/GPU).
        dtype: PyTorch data type for numerical precision. Defaults to torch.float32.

    Returns:
        FIREState: The optimized structure state containing positions, forces,
            energies and other optimization parameters.

    Notes:
        - The overlap minimization uses a species-specific soft-sphere potential
          that creates repulsive forces between overlapping atoms.
        - The FIRE algorithm is used to minimize the potential energy and reduce overlaps.
        - The minimum distance criterion for convergence is 95% of the smallest diameter
          in the diameter matrix.
        - For each species pair (i,j), the diameter_matrix[i,j] specifies the minimum
          allowed distance between atoms of species i and j.
    """
    # Extract element information from composition into a robust dictionary format
    element_dict = composition.as_dict()
    element_symbols = list(element_dict.keys())  # Get unique elements
    element_counts = [
        int(element_dict[el]) for el in element_symbols
    ]  # Get counts directly

    # Create species indices tensor mapping each atom to its species type
    # e.g. for Fe80B20: [0,0,...,0,1,1,...,1] where 0=Fe, 1=B
    species_idx = torch.tensor(
        [i for i, count in enumerate(element_counts) for _ in range(count)], device=device
    )

    # Calculate total atoms and number of unique species
    N_atoms = sum(element_counts)
    print(f"Creating structure with {N_atoms} atoms: {element_dict}")

    # Set up random number generator with optional seed for reproducibility
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    # Generate initial random positions in fractional coordinates [0,1]
    positions = torch.rand((N_atoms, 3), device=device, dtype=dtype, generator=generator)

    # If auto_diameter enabled, calculate species-specific diameter matrix
    if auto_diameter:
        diameter_matrix = get_diameter_matrix(composition, device=device, dtype=dtype)
        print(f"Using random pack diameter matrix:\n{diameter_matrix.cpu().numpy()}")

    # Perform overlap minimization if diameter matrix is specified
    if diameter_matrix is not None:
        print("Reduce atom overlap using the soft_sphere potential")
        # Convert fractional to cartesian coordinates
        positions_cart = torch.matmul(positions, cell)

        # Initialize multi-species soft sphere potential calculator
        model = UnbatchedSoftSphereMultiModel(
            species=species_idx,
            sigma_matrix=diameter_matrix,
            device=device,
            dtype=dtype,
            compute_force=True,
            use_neighbor_list=True,
        )

        # Dummy atomic numbers
        atomic_numbers = torch.ones_like(positions_cart, device=device, dtype=torch.int)

        # Set up FIRE optimizer with unit masses for all atoms
        state_init_fn, fire_update = fire(model=model)
        state = state_init_fn(
            positions=positions_cart,
            masses=torch.ones(N_atoms, device=device, dtype=dtype),
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=True,
        )
        print(f"Initial energy: {state.energy.item():.4f}")
        # Run FIRE optimization until convergence or max iterations
        for _ in range(max_iter):
            # Check if minimum distance criterion is met (95% of smallest target diameter)
            if (
                min_distance(state.positions, cell, distance_tolerance)
                > diameter_matrix.min() * 0.95
            ):
                break
            state = fire_update(state)
        print(f"Final energy: {state.energy.item():.4f}")

    return state


def valid_subcell(
    positions: torch.Tensor,
    cell: torch.Tensor,
    initial_energy: float,
    final_energy: float,
    e_tol: float = 0.001,
    fe_lower_limit: float = -5.0,
    fe_upper_limit: float = 0.0,
    fusion_distance: float = 1.5,
    distance_tolerance: float = 0.0001,
) -> bool:
    """Validate a relaxed subcell structure against physical and numerical criteria.

    This function checks if a relaxed subcell structure meets several validation criteria:
    1. Formation energy is physically reasonable (not too negative)
    2. Energy decreased during relaxation (optimization worked properly)
    3. Final energy is low enough (good convergence)
    4. Atoms are not too close (no atomic fusion)

    Args:
        positions: Atomic positions tensor of shape [n_atoms, 3], where each row contains
            the (x,y,z) coordinates of an atom.
        cell: Unit cell tensor of shape [3, 3] containing the three lattice vectors that
            define the periodic boundary conditions.
        initial_energy: Total energy of the structure before relaxation, in eV.
        final_energy: Total energy of the structure after relaxation, in eV.
        e_tol: Energy tolerance for comparing initial and final energies, in eV.
            Used to check if optimization reduced the energy. Defaults to 0.001 eV.
        fe_lower_limit: Lower limit for formation energy, in eV/atom. Values below this
            are considered unphysical. Defaults to -5.0 eV/atom.
        fe_upper_limit: Upper limit for formation energy, in eV/atom. Values above this
            indicate poor convergence. Defaults to 0.0 eV/atom.
        fusion_distance: Minimum allowed distance between any pair of atoms, in Å.
            Distances below this indicate atomic fusion. Defaults to 1.5 Å.
        distance_tolerance: Distance below which atoms are considered to be at the same
            position when computing minimum distances. Defaults to 0.0001 Å.

    Returns:
        bool: True if the structure passes all validation checks, False otherwise.

    Notes:
        - The function uses periodic boundary conditions when checking atomic distances
        - Formation energies are typically negative but not extremely negative
        - The optimization should reduce energy unless already at a minimum
        - Atomic fusion (distances < ~1.5 Å) indicates an unphysical structure
    """
    # Check if formation energy is unphysically negative
    if final_energy < fe_lower_limit:
        return False

    # Check if optimization properly reduced the energy
    # A small tolerance accounts for numerical noise
    if not (final_energy >= initial_energy + e_tol):
        return False

    # Check if final energy is low enough to indicate good convergence
    if not (final_energy <= fe_upper_limit + e_tol):
        return False

    # Check minimum interatomic distances to detect atomic fusion
    # Uses periodic boundary conditions via min_distance function
    min_dist = min_distance(positions, cell, distance_tolerance)
    if min_dist < fusion_distance:
        print("Bad structure! Fusion found.")
        return False

    # Structure passed all validation checks
    return True


def get_subcells_to_crystallize(
    fractional_positions: torch.Tensor,
    species: list[str],
    d_frac: float = 0.05,
    n_min: int = 1,
    n_max: int = 48,
    restrict_to_compositions: Sequence[str] | None = None,
    max_coeff: int | None = None,
    elements: Sequence[str] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Extract subcell structures from a larger structure for crystallization.

    This function slices a large structure (e.g. amorphous) into smaller subcells that
    can be independently relaxed to find stable crystal structures. It uses a grid-based
    approach to systematically sample different regions of the structure.

    Args:
        fractional_positions: Atomic positions tensor of shape [n_atoms, 3],
            where each row contains the (x,y,z) coordinates of an atom.
        species: List of chemical element symbols corresponding to each atom position.
        d_frac: Grid spacing in fractional coordinates used to define subcell boundaries.
            Smaller values create more overlap between subcells. Defaults to 0.05.
        n_min: Minimum number of atoms required in a subcell for it to be considered
            valid. Defaults to 1.
        n_max: Maximum number of atoms allowed in a subcell. Larger subcells take longer
            to relax. Defaults to 48.
        restrict_to_compositions: Optional list of chemical formulas (e.g. ["AB", "AB2"])
            to restrict which subcell compositions are extracted. Defaults to None.
        max_coeff: Optional maximum stoichiometric coefficient. If provided, only formulas
            up to this coefficient are considered (e.g. max_coeff=2 allows AB2 but not
            AB3). Defaults to None.
        elements: List of elements to consider when generating stoichiometries. Required
            if max_coef is provided. Defaults to None.

    Returns:
        List of tuples, where each tuple contains:
        - indices: Tensor of atom indices included in the subcell
        - lower: Tensor of lower bounds for subcell in fractional coords [3]
        - upper: Tensor of upper bounds for subcell in fractional coords [3]

    Notes:
        - The function uses fractional coordinates internally for consistent slicing
        - Subcells can overlap, providing redundancy in the search
        - Composition restrictions help focus on chemically relevant structures
        - The max_coef parameter is useful for limiting complexity of compositions
    """
    # Get device and dtype from input tensors for consistency
    device = fractional_positions.device
    dtype = fractional_positions.dtype

    # Convert species list to numpy array for easier composition handling
    species_array = np.array(species)

    # Generate allowed stoichiometries if max_coef is specified
    if max_coeff:
        if elements is None:
            raise ValueError("elements must be provided when max_coef is specified")
        # Generate all possible stoichiometry combinations up to max_coef
        stoichs = list(itertools.product(range(max_coeff + 1), repeat=len(elements)))
        stoichs.pop(0)  # Remove the empty composition (0,0,...)
        # Convert stoichiometries to composition formulas
        comps = []
        for stoich in stoichs:
            comp = dict(zip(elements, stoich, strict=False))
            comps.append(Composition.from_dict(comp).reduced_formula)
        restrict_to_compositions = set(comps)

    # Ensure compositions are in reduced formula form if provided
    if restrict_to_compositions:
        restrict_to_compositions = [
            Composition(comp).reduced_formula for comp in restrict_to_compositions
        ]

    # Create orthorhombic grid for systematic subcell generation
    bins = int(1 / d_frac)
    grid = torch.linspace(0, 1, bins + 1, device=device, dtype=dtype)
    # Generate lower and upper bounds for all possible subcells
    l_bound = (
        torch.stack(torch.meshgrid(grid[:-1], grid[:-1], grid[:-1], indexing="ij"))
        .reshape(3, -1)
        .T
    )
    u_bound = (
        torch.stack(torch.meshgrid(grid[1:], grid[1:], grid[1:], indexing="ij"))
        .reshape(3, -1)
        .T
    )

    candidates = []
    # Iterate through all possible subcell boundary combinations
    for lb, ub in itertools.product(l_bound, u_bound):
        if torch.all(ub > lb):  # Ensure valid subcell dimensions
            # Find atoms within the subcell bounds
            mask = torch.logical_and(
                torch.all(ub >= fractional_positions, dim=1),
                torch.all(lb <= fractional_positions, dim=1),
            )
            ids = torch.nonzero(mask).flatten()

            # Check if number of atoms meets size constraints
            if n_min <= len(ids) <= n_max:
                # Apply composition restrictions if specified
                if restrict_to_compositions:
                    subcell_comp = Composition(
                        "".join(species_array[ids.cpu().numpy()])
                    ).reduced_formula
                    if subcell_comp not in restrict_to_compositions:
                        continue
                candidates.append((ids, lb, ub))

    return candidates


def subcells_to_structures(
    candidates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    fractional_positions: torch.Tensor,
    cell: torch.Tensor,
    species: list[str],
) -> list[tuple[torch.Tensor, torch.Tensor, list[str]]]:
    """Convert subcell candidates to structure tuples.

    Args:
        candidates: List of (ids, lower_bound, upper_bound)
        tuples from get_subcells_to_crystallize
        fractional_positions: Fractional coordinates of atoms
        cell: Unit cell tensor
        species: List of atomic species symbols

    Returns:
        List of (fractional_positions, cell, species) tuples for each valid subcell
    """
    list_subcells = []
    for ids, l, h in candidates:  # noqa: E741
        # Get positions of atoms in this subcell
        pos = fractional_positions[ids]

        # Shift positions to start from origin
        new_frac_pos = pos - l

        # Scale positions to [0,1] range
        new_frac_pos = new_frac_pos / (h - l)

        # Calculate new cell parameters
        new_cell = cell * (h - l).unsqueeze(0)

        # Convert tensor indices to list/numpy array before indexing species list
        species_indices = ids.cpu().numpy()  # Convert to numpy array
        subcell_species = [
            species[int(i)] for i in species_indices
        ]  # Get species for these atoms

        list_subcells.append((new_frac_pos, new_cell, subcell_species))

    return list_subcells


def get_target_temperature(
    step: int, equi_steps: int, cool_steps: int, T_high: float, T_low: float
) -> float:
    """Calculate temperature at each step of a melt-quench-equilibrate profile.

    Args:
        step: Current simulation step
        equi_steps: Number of equilibration steps at high temperature
        cool_steps: Number of cooling steps
        T_high: Initial high temperature in Kelvin
        T_low: Final low temperature in Kelvin

    Returns:
        float: Temperature
    """
    if step < equi_steps:
        return T_high
    if step < cool_steps + equi_steps:
        # Linear cooling
        cooling_fraction = 1.0 - (step - equi_steps) / cool_steps
        return T_low + (T_high - T_low) * cooling_fraction
    return T_low
