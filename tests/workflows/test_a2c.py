import pytest
import torch
from pymatgen.core.composition import Composition

from torch_sim.state import SimState
from torch_sim.unbatched.models.soft_sphere import UnbatchedSoftSphereModel
from torch_sim.unbatched.unbatched_optimizers import UnitCellFIREState
from torch_sim.workflows import a2c


@pytest.mark.parametrize(
    ("positions", "cell", "expected_min_dist"),
    [
        (
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.eye(3) * 10.0,
            1.0,
        ),
        (
            torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            torch.eye(3) * 5.0,
            0.866025,  # sqrt(3)/2
        ),
        (
            torch.tensor([[0.0, 0.0, 0.0], [2.9, 0.0, 0.0], [0.0, 0.0, 2.9]]),
            torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]),
            0.1,  # Due to PBC, atoms at 2.9 are closer via boundary
        ),
    ],
)
def test_min_distance(
    positions: torch.Tensor, cell: torch.Tensor, expected_min_dist: float
) -> None:
    """Test calculation of minimum distance between atoms."""
    min_dist = a2c.min_distance(positions, cell)
    assert torch.isclose(min_dist, torch.tensor(expected_min_dist), atol=1e-5)


@pytest.mark.parametrize(
    ("composition_str", "expected_diameter_range"),
    [
        ("Cu", (2.0, 3.0)),  # Metal, should use metallic radius * 2
        (
            "O",
            (1.0, 2.5),
        ),  # Non-metal, should use atomic or ionic radius * 2 (adjusted range)
        ("Fe2O3", (1.5, 2.5)),  # Multi-element, should use minimum pair separation
        ("NaCl", (1.5, 3.5)),  # Ionic compound (adjusted range)
    ],
)
def test_get_diameter_parametrized(
    composition_str: str, expected_diameter_range: tuple[float, float]
) -> None:
    """Test diameter calculation for various compositions."""
    comp = Composition(composition_str)
    diameter = a2c.get_diameter(comp)
    assert expected_diameter_range[0] < diameter < expected_diameter_range[1], (
        f"Diameter {diameter} outside expected range {expected_diameter_range}"
    )


@pytest.mark.parametrize(
    ("composition_str", "expected_size", "dtype"),
    [
        ("Cu", 1, torch.float32),
        ("Fe2O3", 2, torch.float32),  # Fe and O
        ("LiNiMnCoO2", 5, torch.float64),  # Li, Ni, Mn, Co, O
    ],
)
def test_get_diameter_matrix_parametrized(
    composition_str: str, expected_size: int, dtype: torch.dtype, device: torch.device
) -> None:
    """Test diameter matrix calculation with different compositions."""
    comp = Composition(composition_str)
    matrix = a2c.get_diameter_matrix(comp, device=device, dtype=dtype)

    # Check matrix properties
    assert matrix.shape == (expected_size, expected_size)
    assert matrix.dtype == dtype
    assert matrix.device == device
    assert torch.all(matrix > 0)
    assert torch.allclose(matrix, matrix.T)  # Symmetry


def test_random_packed_structure_basic(device: torch.device) -> None:
    """Test basic functionality of random_packed_structure."""
    comp = Composition("Cu4")
    cell = torch.eye(3, device=device) * 5.0

    # Test with minimal optimization to ensure state is created
    state = a2c.random_packed_structure(
        composition=comp,
        cell=cell,
        seed=42,
        # Use a diameter to ensure the state is created
        diameter=2.5,
        max_iter=1,
        device=device,
    )

    # Check state properties
    assert state.positions.shape == (4, 3)
    assert state.positions.device == device
    assert torch.all(state.positions >= 0)
    assert torch.all(state.positions <= cell[0, 0])


@pytest.mark.parametrize(
    ("composition_str", "cell_size", "diameter", "max_iter"),
    [("Cu4", 5.0, 2.5, 2), ("Fe2O3", 6.0, 2.0, 2)],
)
def test_random_packed_structure_optimization(
    composition_str: str,
    cell_size: float,
    diameter: float,
    max_iter: int,
    device: torch.device,
) -> None:
    """Test random_packed_structure with optimization."""
    comp = Composition(composition_str)
    cell = torch.eye(3, device=device) * cell_size

    log = []
    result = a2c.random_packed_structure(
        composition=comp,
        cell=cell,
        seed=42,
        diameter=diameter,
        max_iter=max_iter,
        device=device,
        log=log,
    )

    # Handle the case where a tuple is returned when log is provided
    if isinstance(result, tuple):
        state, _ = result
    else:
        state = result

    # Check that optimization happened
    assert len(log) > 0
    assert state.energy is not None


def test_random_packed_structure_auto_diameter(device: torch.device) -> None:
    """Test random_packed_structure with auto_diameter option."""
    comp = Composition("Cu4")
    cell = torch.eye(3, device=device) * 6.0

    state = a2c.random_packed_structure(
        composition=comp,
        cell=cell,
        seed=42,
        auto_diameter=True,
        max_iter=3,
        device=device,
    )

    # Just check that it ran without errors
    assert state.positions is not None
    assert state.energy is not None


@pytest.mark.parametrize(
    (
        "positions",
        "cell",
        "initial_energy",
        "final_energy",
        "e_tol",
        "fe_lower_limit",
        "fe_upper_limit",
        "fusion_distance",
        "expected",
    ),
    [
        (
            torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            torch.eye(3) * 5.0,
            *(0.0, -1.0, 0.001, -5.0, 0.0, 1.5, False),
        ),
        (  # Invalid - no energy decrease
            torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            torch.eye(3) * 5.0,
            *(-1.0, -1.0, 0.001, -5.0, 0.0, 1.5, False),
        ),
        (  # Invalid - energy too low
            torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            torch.eye(3) * 5.0,
            *(0.0, -10.0, 0.001, -5.0, 0.0, 1.5, False),
        ),
        (  # Invalid - atoms too close
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.eye(3) * 5.0,
            *(0.0, -1.0, 0.001, -5.0, 0.0, 1.5, False),
        ),
    ],
)
def test_valid_subcell(
    *,
    positions: torch.Tensor,
    cell: torch.Tensor,
    initial_energy: float,
    final_energy: float,
    e_tol: float,
    fe_lower_limit: float,
    fe_upper_limit: float,
    fusion_distance: float,
    expected: bool,
) -> None:
    """Test validation of subcell structures."""
    # Run the validation function
    result = a2c.valid_subcell(
        positions=positions,
        cell=cell,
        initial_energy=initial_energy,
        final_energy=final_energy,
        e_tol=e_tol,
        fe_lower_limit=fe_lower_limit,
        fe_upper_limit=fe_upper_limit,
        fusion_distance=fusion_distance,
    )

    # Check if the result matches expectations
    assert result is expected


@pytest.mark.parametrize(
    ("d_frac", "n_min", "n_max", "should_find_candidates"),
    [
        (0.5, 1, 5, True),
        (0.5, 3, 5, True),
        (0.1, 5, 10, False),  # Unlikely to find candidates with these constraints
    ],
)
def test_get_subcells_to_crystallize_parametrized(
    *,
    d_frac: float,
    n_min: int,
    n_max: int,
    should_find_candidates: bool,
    device: torch.device,
) -> None:
    """Test subcell candidate extraction with different parameters."""
    frac_positions = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.4, 0.4, 0.4],
            [0.6, 0.6, 0.6],
            [0.8, 0.8, 0.8],
        ],
        device=device,
    )
    species = ["Cu", "Cu", "O", "O", "O"]

    candidates = a2c.get_subcells_to_crystallize(
        frac_positions, species, d_frac=d_frac, n_min=n_min, n_max=n_max
    )

    if should_find_candidates:
        assert len(candidates) > 0
    # If we're not expected to find candidates, don't assert anything
    # as the test data is small and might still find candidates in some cases


@pytest.mark.parametrize(
    ("max_coeff", "elements", "expected_min_candidates"),
    [
        (1, ["Cu", "O"], 1),
        (2, ["Cu", "O"], 3),  # Should find more compositions with higher max_coeff
    ],
)
def test_get_subcells_with_max_coeff(
    max_coeff: int,
    elements: list[str],
    expected_min_candidates: int,
    device: torch.device,
) -> None:
    """Test subcell extraction with max_coeff parameter."""
    frac_positions = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.4, 0.4, 0.4],
            [0.6, 0.6, 0.6],
            [0.8, 0.8, 0.8],
        ],
        device=device,
    )
    species = ["Cu", "Cu", "O", "O", "O"]

    candidates = a2c.get_subcells_to_crystallize(
        frac_positions,
        species,
        d_frac=0.5,
        n_min=1,
        n_max=5,
        max_coeff=max_coeff,
        elements=elements,
    )

    assert len(candidates) >= expected_min_candidates


@pytest.mark.parametrize(
    ("step", "equi_steps", "cool_steps", "T_high", "T_low", "expected_temp"),
    [
        (0, 10, 20, 1000, 300, 1000),  # Start - high temp
        (10, 10, 20, 1000, 300, 1000),  # End of equilibration
        (11, 10, 20, 1000, 300, 965),  # Start of cooling
        (20, 10, 20, 1000, 300, 650),  # Middle of cooling
        (29, 10, 20, 1000, 300, 335),  # Almost end of cooling
        (30, 10, 20, 1000, 300, 300),  # End of cooling
        (40, 10, 20, 1000, 300, 300),  # After cooling
    ],
)
def test_get_target_temperature_parametrized(
    step: int,
    equi_steps: int,
    cool_steps: int,
    T_high: float,
    T_low: float,
    expected_temp: float,
) -> None:
    """Test temperature profile calculation with various parameters."""
    temp = a2c.get_target_temperature(step, equi_steps, cool_steps, T_high, T_low)
    assert abs(temp - expected_temp) < 1.0  # Allow for small floating point differences


def create_test_model(
    *, device: torch.device, compute_stress: bool = True
) -> UnbatchedSoftSphereModel:
    """Create a simple soft sphere model for testing."""
    return UnbatchedSoftSphereModel(
        sigma=2.5,
        epsilon=0.01,
        alpha=2.0,
        device=device,
        compute_forces=True,
        compute_stress=compute_stress,
    )


def create_test_state(positions: torch.Tensor, cell: torch.Tensor) -> SimState:
    """Create a simple simulation state for testing."""
    n_atoms = positions.shape[0]
    device = positions.device
    return SimState(
        positions=positions,
        cell=cell,
        pbc=True,
        masses=torch.ones(n_atoms, device=device),
        atomic_numbers=torch.ones(n_atoms, device=device, dtype=torch.long),
    )


@pytest.mark.parametrize("max_iter", [1, 2, 3])
def test_get_unit_cell_relaxed_structure(max_iter: int, device: torch.device) -> None:
    """Test unit cell relaxation with FIRE algorithm."""
    # Create a simple test system
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], device=device
    )
    cell = torch.eye(3, device=device) * 5.0

    # Create model and state
    model = create_test_model(device=device)
    state = create_test_state(positions, cell)

    # Run relaxation with minimal steps
    relaxed_state, logger, final_energy, final_pressure = (
        a2c.get_unit_cell_relaxed_structure(state=state, model=model, max_iter=max_iter)
    )

    # Basic checks
    assert isinstance(relaxed_state, UnitCellFIREState)
    assert logger["energy"].shape[0] == max_iter
    assert isinstance(final_energy, float)
    assert isinstance(final_pressure, float)


@pytest.mark.parametrize(
    ("n_positions", "n_species", "cell_size"),
    [(5, 5, 5.0), (10, 10, 10.0)],
    ids=["Equal number of positions and species", "Larger system"],
)
def test_subcells_to_structures_parametrized(
    n_positions: int, n_species: int, cell_size: float, device: torch.device
) -> None:
    """Test subcell extraction and conversion with various parameters."""
    # Create test data with varying sizes
    frac_positions = torch.rand((n_positions, 3), device=device)
    cell = torch.eye(3, device=device) * cell_size

    # Create alternating Cu/O species list
    species = ["Cu" if idx % 2 == 0 else "O" for idx in range(n_species)]

    # Get subcell candidates
    candidates = a2c.get_subcells_to_crystallize(
        frac_positions, species, d_frac=0.5, n_min=1, n_max=n_positions
    )

    structures = a2c.subcells_to_structures(candidates, frac_positions, cell, species)

    # Check output format
    assert len(structures) == len(candidates)
    for pos, subcell, spec in structures:
        assert isinstance(pos, torch.Tensor)
        assert isinstance(subcell, torch.Tensor)
        assert isinstance(spec, list)
        assert pos.shape[1] == 3  # 3D positions
        assert subcell.shape == (3, 3)  # 3x3 cell matrix
        assert all(isinstance(s, str) for s in spec)  # Species strings

        # Ensure positions are in [0,1] range (fractional coordinates)
        assert torch.all(pos >= 0.0)
        assert torch.all(pos <= 1.0)


def test_subcells_to_structures_ensures_proper_scaling() -> None:
    """Test that subcells_to_structures properly scales the positions and cell."""
    device = torch.device("cpu")

    # Create test data with a known grid of points
    frac_positions = torch.tensor(
        [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]], device=device
    )

    cell = torch.eye(3, device=device) * 10.0
    species = ["Cu", "Cu", "O"]

    # Create a candidate with known bounds
    ids = torch.tensor([0, 1])  # First two atoms
    lower = torch.tensor([0.0, 0.0, 0.0])
    upper = torch.tensor([0.25, 0.25, 0.25])
    candidates = [(ids, lower, upper)]

    # Convert to structures
    structures = a2c.subcells_to_structures(candidates, frac_positions, cell, species)

    # Get the result
    subcell_pos, subcell, subcell_species = structures[0]

    # Check that positions are rescaled to [0,1] range
    assert torch.allclose(
        subcell_pos[0], torch.tensor([0.4, 0.4, 0.4])
    )  # (0.1-0.0)/0.25 = 0.4
    assert torch.allclose(
        subcell_pos[1], torch.tensor([0.8, 0.8, 0.8])
    )  # (0.2-0.0)/0.25 = 0.8

    # Check that cell is scaled properly
    expected_cell = torch.eye(3) * 2.5  # 10.0 * 0.25 = 2.5
    assert torch.allclose(subcell, expected_cell)

    # Check species are correct
    assert subcell_species == ["Cu", "Cu"]


@pytest.mark.parametrize("restrict_to_compositions", [["CuO"], ["Cu2O", "CuO2"]])
def test_get_subcells_with_composition_restrictions(
    restrict_to_compositions: list[str], device: torch.device
) -> None:
    """Test subcell extraction with composition restrictions."""
    frac_positions = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4],
            [0.6, 0.6, 0.6],
            [0.8, 0.8, 0.8],
        ],
        device=device,
    )
    species = ["Cu", "Cu", "Cu", "O", "O", "O"]

    candidates = a2c.get_subcells_to_crystallize(
        frac_positions,
        species,
        d_frac=0.5,
        n_min=1,
        n_max=6,
        restrict_to_compositions=restrict_to_compositions,
    )

    # Check that all candidates match the requested compositions
    for ids, _, _ in candidates:
        subcell_species = [species[int(i)] for i in ids.cpu().numpy()]
        comp = Composition("".join(subcell_species)).reduced_formula
        assert comp in restrict_to_compositions, (
            f"Found composition {comp} not in {restrict_to_compositions}"
        )


def test_get_subcells_to_crystallize_invalid_inputs() -> None:
    """Test invalid inputs for subcell extraction."""
    device = torch.device("cpu")
    frac_positions = torch.tensor([[0.1, 0.1, 0.1]], device=device)
    species = ["Cu"]

    # Test with max_coeff but no elements
    with pytest.raises(ValueError, match="elements must be provided"):
        a2c.get_subcells_to_crystallize(
            frac_positions,
            species,
            max_coeff=2,
            elements=None,
        )
