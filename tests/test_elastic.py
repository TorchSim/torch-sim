import pytest
import torch

import torch_sim as ts
from tests.conftest import DEVICE
from torch_sim.elastic import (
    calculate_elastic_moduli,
    calculate_elastic_tensor,
    get_bravais_type,
    get_cart_deformed_cell,
    get_elementary_deformations,
    get_strain,
)
from torch_sim.typing import BravaisType
from torch_sim.units import UnitConversion


try:
    from mace.calculators.foundations_models import mace_mp

    from torch_sim.models.mace import MaceModel
except ImportError:
    pytest.skip("MACE not installed", allow_module_level=True)


def test_get_strain_zero_deformation(cu_sim_state: ts.SimState) -> None:
    """Test that zero deformation produces zero strain."""
    # Test with same state as reference and deformed - should give zero strain
    strain = get_strain(cu_sim_state, cu_sim_state)

    expected_strain = torch.zeros(6, device=cu_sim_state.device, dtype=cu_sim_state.dtype)
    torch.testing.assert_close(strain, expected_strain, atol=1e-12, rtol=1e-12)


def test_get_strain_pure_normal_strain(cu_sim_state: ts.SimState) -> None:
    """Test pure normal strain calculations (uniaxial extension/compression)."""
    device = cu_sim_state.device
    dtype = cu_sim_state.dtype

    # Test pure xx strain (axis 0)
    strain_magnitude = 0.05
    deformed_state = get_cart_deformed_cell(cu_sim_state, axis=0, size=strain_magnitude)
    calculated_strain = get_strain(deformed_state, cu_sim_state)

    # Expected: only εxx should be non-zero and equal to strain_magnitude
    # For pure normal strain, the symmetric tensor should give εxx = strain_magnitude
    expected_strain = torch.zeros(6, device=device, dtype=dtype)
    expected_strain[0] = strain_magnitude  # εxx

    torch.testing.assert_close(calculated_strain, expected_strain, atol=1e-12, rtol=1e-12)

    # Test pure yy strain (axis 1)
    deformed_state = get_cart_deformed_cell(cu_sim_state, axis=1, size=strain_magnitude)
    calculated_strain = get_strain(deformed_state, cu_sim_state)

    expected_strain = torch.zeros(6, device=device, dtype=dtype)
    expected_strain[1] = strain_magnitude  # εyy

    torch.testing.assert_close(calculated_strain, expected_strain, atol=1e-12, rtol=1e-12)

    # Test pure zz strain (axis 2)
    deformed_state = get_cart_deformed_cell(cu_sim_state, axis=2, size=strain_magnitude)
    calculated_strain = get_strain(deformed_state, cu_sim_state)

    expected_strain = torch.zeros(6, device=device, dtype=dtype)
    expected_strain[2] = strain_magnitude  # εzz

    torch.testing.assert_close(calculated_strain, expected_strain, atol=1e-12, rtol=1e-12)


def test_get_strain_pure_shear_strain(cu_sim_state: ts.SimState) -> None:
    """Test pure shear strain calculations and verify symmetric strain tensor."""
    device = cu_sim_state.device
    dtype = cu_sim_state.dtype

    # Test yz shear strain (axis 3)
    shear_magnitude = 0.08
    deformed_state = get_cart_deformed_cell(cu_sim_state, axis=3, size=shear_magnitude)
    calculated_strain = get_strain(deformed_state, cu_sim_state)

    # For shear deformation, the displacement gradient u will have:
    # u[1, 2] = shear_magnitude, but the symmetric strain is (u + u^T)/2
    # So εyz = (u[1,2] + u[2,1])/2 = (shear_magnitude + 0)/2 = shear_magnitude/2
    # This demonstrates the key symmetric strain tensor calculation at line 815
    expected_strain = torch.zeros(6, device=device, dtype=dtype)
    expected_strain[3] = shear_magnitude / 2  # εyz = symmetric shear strain

    torch.testing.assert_close(calculated_strain, expected_strain, atol=1e-12, rtol=1e-12)

    # Test xz shear strain (axis 4)
    deformed_state = get_cart_deformed_cell(cu_sim_state, axis=4, size=shear_magnitude)
    calculated_strain = get_strain(deformed_state, cu_sim_state)

    expected_strain = torch.zeros(6, device=device, dtype=dtype)
    expected_strain[4] = shear_magnitude / 2  # εxz = symmetric shear strain

    torch.testing.assert_close(calculated_strain, expected_strain, atol=1e-12, rtol=1e-12)

    # Test xy shear strain (axis 5)
    deformed_state = get_cart_deformed_cell(cu_sim_state, axis=5, size=shear_magnitude)
    calculated_strain = get_strain(deformed_state, cu_sim_state)

    expected_strain = torch.zeros(6, device=device, dtype=dtype)
    expected_strain[5] = shear_magnitude / 2  # εxy = symmetric shear strain

    torch.testing.assert_close(calculated_strain, expected_strain, atol=1e-12, rtol=1e-12)


def test_get_strain_hydrostatic_strain(cu_sim_state: ts.SimState) -> None:
    """Test hydrostatic strain (equal expansion/compression in all directions)."""
    device = cu_sim_state.device
    dtype = cu_sim_state.dtype

    # Create hydrostatic deformation by scaling all cell vectors equally
    hydro_strain = 0.03
    original_cell = cu_sim_state.row_vector_cell.squeeze()

    # Scale the cell uniformly (hydrostatic deformation)
    hydro_deformation = torch.eye(3, device=device, dtype=dtype) * (1 + hydro_strain)
    deformed_cell = torch.matmul(original_cell, hydro_deformation)

    # Create deformed state manually
    deformed_positions = cu_sim_state.positions * (1 + hydro_strain)
    deformed_state = ts.SimState(
        positions=deformed_positions,
        cell=deformed_cell.mT.unsqueeze(0),
        masses=cu_sim_state.masses,
        pbc=cu_sim_state.pbc,
        atomic_numbers=cu_sim_state.atomic_numbers,
    )

    calculated_strain = get_strain(deformed_state, cu_sim_state)

    # For hydrostatic strain, εxx = εyy = εzz = hydro_strain, all shear components = 0
    expected_strain = torch.zeros(6, device=device, dtype=dtype)
    expected_strain[0] = hydro_strain  # εxx
    expected_strain[1] = hydro_strain  # εyy
    expected_strain[2] = hydro_strain  # εzz
    # εyz, εxz, εxy should remain zero

    torch.testing.assert_close(calculated_strain, expected_strain, atol=1e-12, rtol=1e-12)


def test_get_strain_symmetry_property(cu_sim_state: ts.SimState) -> None:
    """Test that the strain tensor calculation properly enforces symmetry (u + u^T)/2."""
    device = cu_sim_state.device
    dtype = cu_sim_state.dtype

    # Create a deformation that would produce an asymmetric displacement gradient
    # We'll manually create a deformed cell that would result in u[0,1] != u[1,0]
    # but the symmetric strain tensor should symmetrize this

    original_cell = cu_sim_state.row_vector_cell.squeeze()

    # Create an asymmetric deformation matrix
    asymmetric_deformation = torch.tensor(
        [
            [1.02, 0.03, 0.0],  # This creates both normal and shear components
            [0.0, 1.01, 0.0],  # Different from symmetric case
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )

    deformed_cell = torch.matmul(original_cell, asymmetric_deformation)

    # Convert positions to fractional, then back with new cell
    frac_coords = torch.matmul(cu_sim_state.positions, torch.linalg.inv(original_cell))
    deformed_positions = torch.matmul(frac_coords, deformed_cell)

    deformed_state = ts.SimState(
        positions=deformed_positions,
        cell=deformed_cell.mT.unsqueeze(0),
        masses=cu_sim_state.masses,
        pbc=cu_sim_state.pbc,
        atomic_numbers=cu_sim_state.atomic_numbers,
    )

    calculated_strain = get_strain(deformed_state, cu_sim_state)

    # Manually calculate what the symmetric strain should be
    cell_diff = deformed_cell - original_cell
    u = torch.matmul(torch.linalg.inv(original_cell), cell_diff)
    symmetric_strain_tensor = (u + u.mT) / 2

    expected_strain = torch.tensor(
        [
            symmetric_strain_tensor[0, 0],  # εxx
            symmetric_strain_tensor[1, 1],  # εyy
            symmetric_strain_tensor[2, 2],  # εzz
            symmetric_strain_tensor[2, 1],  # εyz
            symmetric_strain_tensor[2, 0],  # εxz
            symmetric_strain_tensor[1, 0],  # εxy
        ],
        device=device,
        dtype=dtype,
    )

    torch.testing.assert_close(calculated_strain, expected_strain, atol=1e-12, rtol=1e-12)

    # Verify that the shear components are properly symmetrized
    # εxy should equal the average of the off-diagonal terms
    expected_xy_strain = (u[1, 0] + u[0, 1]) / 2
    assert torch.allclose(calculated_strain[5], expected_xy_strain, atol=1e-12)


def test_get_elementary_deformations_strain_consistency(
    cu_sim_state: ts.SimState,
) -> None:
    """Test that deformations generated by get_elementary_deformations produce expected
    strains."""
    max_strain_normal = 0.02
    max_strain_shear = 0.05
    n_deform = 3

    deformed_states = get_elementary_deformations(
        cu_sim_state,
        n_deform=n_deform,
        max_strain_normal=max_strain_normal,
        max_strain_shear=max_strain_shear,
        bravais_type=BravaisType.triclinic,  # Test all axes
    )

    # Should generate deformations for all 6 axes (triclinic)
    # Each axis generates n_deform-1 strains when n_deform is odd (excluding zero),
    # or n_deform strains when n_deform is even (zero not included in linspace)
    strains_per_axis = n_deform - 1 if n_deform % 2 == 1 else n_deform
    expected_n_states = 6 * strains_per_axis
    assert len(deformed_states) == expected_n_states

    # Check that each deformed state produces a strain with expected dominant component
    axis_to_strain_idx = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}  # axis -> Voigt index

    for def_idx, deformed_state in enumerate(deformed_states):
        strain = get_strain(deformed_state, cu_sim_state)

        # Determine which axis this deformation corresponds to
        axis = def_idx // strains_per_axis  # Integer division to get axis index
        strain_idx = axis_to_strain_idx[axis]

        # The strain component corresponding to this axis should be the largest
        max_strain_component = torch.max(torch.abs(strain))
        assert torch.isclose(
            torch.abs(strain[strain_idx]), max_strain_component, rtol=1e-10, atol=1e-12
        )

        # Verify strain magnitude is within expected bounds
        if axis < 3:  # Normal strain
            assert torch.abs(strain[strain_idx]) <= max_strain_normal + 1e-12
        else:  # Shear strain (factor of 2 due to symmetric strain tensor)
            assert torch.abs(strain[strain_idx]) <= max_strain_shear / 2 + 1e-12


@pytest.fixture
def mace_model() -> MaceModel:
    """Create a MACE model fixture for testing."""
    mace_model = mace_mp(model="medium", default_dtype="float64", return_raw_model=True)

    return MaceModel(
        model=mace_model,
        device=DEVICE,
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=True,
    )


@pytest.mark.parametrize(
    ("sim_state_name", "expected_bravais_type", "atol"),
    [
        ("cu_sim_state", BravaisType.cubic, 2e-1),
        ("mg_sim_state", BravaisType.hexagonal, 5e-1),
        ("sb_sim_state", BravaisType.trigonal, 5e-1),
        ("tio2_sim_state", BravaisType.tetragonal, 5e-1),
        ("ga_sim_state", BravaisType.orthorhombic, 5e-1),
        ("niti_sim_state", BravaisType.monoclinic, 5e-1),
    ],
)
def test_elastic_tensor_symmetries(
    sim_state_name: str,
    mace_model: MaceModel,
    expected_bravais_type: BravaisType,
    atol: float,
    request: pytest.FixtureRequest,
) -> None:
    """Test elastic tensor calculations for different crystal systems.

    Args:
        sim_state_name: Name of the fixture containing the simulation state
        mace_model: MACE model fixture
        expected_bravais_type: Expected Bravais lattice type
        atol: Absolute tolerance for comparing elastic tensors
        request: Pytest fixture request object
    """
    # Get fixtures
    model = mace_model
    state = request.getfixturevalue(sim_state_name)

    # Verify the Bravais type of the unrelaxed structure
    actual_bravais_type = get_bravais_type(state)
    assert actual_bravais_type == expected_bravais_type, (
        f"Unrelaxed structure has incorrect Bravais type. "
        f"Expected {expected_bravais_type}, got {actual_bravais_type}"
    )

    # Relax positions and cell
    state = ts.fire_init(
        model=model,
        state=state,
        scalar_pressure=0.0,
        cell_filter=ts.CellFilter.frechet,
    )
    fmax = 1e-5

    for _ in range(300):
        pressure = (
            -torch.trace(state.stress.squeeze()) / 3 * UnitConversion.eV_per_Ang3_to_GPa
        )
        current_fmax = torch.max(torch.abs(state.forces.squeeze()))
        if current_fmax < fmax and abs(pressure) < 1e-2:
            break
        state = ts.fire_step(model=model, state=state)

    # Verify the Bravais type of the relaxed structure
    actual_bravais_type = get_bravais_type(state)
    assert actual_bravais_type == expected_bravais_type, (
        f"Relaxed structure has incorrect Bravais type. "
        f"Expected {expected_bravais_type}, got {actual_bravais_type}"
    )

    # Calculate elastic tensors
    C_symmetric = (
        calculate_elastic_tensor(model, state=state, bravais_type=expected_bravais_type)
        * UnitConversion.eV_per_Ang3_to_GPa
    )
    C_triclinic = (
        calculate_elastic_tensor(model, state=state, bravais_type=BravaisType.triclinic)
        * UnitConversion.eV_per_Ang3_to_GPa
    )

    # Check if the elastic tensors are equal
    assert torch.allclose(C_symmetric, C_triclinic, atol=atol), (
        f"Elastic tensor mismatch for {expected_bravais_type} structure:\n"
        f"Difference matrix:\n{C_symmetric - C_triclinic}"
    )


def test_copper_elastic_properties(
    mace_model: MaceModel, cu_sim_state: ts.SimState
) -> None:
    """Test calculation of elastic properties for copper."""

    # Relax positions and cell
    state = ts.fire_init(
        model=mace_model,
        state=cu_sim_state,
        scalar_pressure=0.0,
        cell_filter=ts.CellFilter.frechet,
    )
    fmax = 1e-5
    for _ in range(300):
        pressure = (
            -torch.trace(state.stress.squeeze()) / 3 * UnitConversion.eV_per_Ang3_to_GPa
        )
        current_fmax = torch.max(torch.abs(state.forces.squeeze()))
        if current_fmax < fmax and abs(pressure) < 1e-2:
            break
        state = ts.fire_step(model=mace_model, state=state)

    # Calculate elastic tensor
    bravais_type = get_bravais_type(state)
    elastic_tensor = calculate_elastic_tensor(
        mace_model, state=state, bravais_type=bravais_type
    )

    # Convert to GPa
    elastic_tensor = elastic_tensor * UnitConversion.eV_per_Ang3_to_GPa

    # Calculate elastic moduli
    bulk_modulus, shear_modulus, _, _ = calculate_elastic_moduli(elastic_tensor)

    device, dtype = state.device, state.dtype

    # Expected values
    expected_elastic_tensor = torch.tensor(
        [
            [171.2151, 130.5025, 130.5025, 0, 0, 0],
            [130.5025, 171.2151, 130.5025, 0, 0, 0],
            [130.5025, 130.5025, 171.2151, 0, 0, 0],
            [0, 0, 0, 70.8029, 0, 0],
            [0, 0, 0, 0, 70.8029, 0],
            [0, 0, 0, 0, 0, 70.8029],
        ],
        device=device,
        dtype=dtype,
    )

    expected_bulk_modulus = 144.12
    expected_shear_modulus = 43.11

    # Assert with tolerance
    assert torch.allclose(elastic_tensor, expected_elastic_tensor, rtol=1e-2)
    assert abs(bulk_modulus - expected_bulk_modulus) < 1e-2 * expected_bulk_modulus
    assert abs(shear_modulus - expected_shear_modulus) < 1e-2 * expected_shear_modulus
