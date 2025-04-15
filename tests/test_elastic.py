import pytest
import torch

from torch_sim.elastic import (
    BravaisType,
    calculate_elastic_moduli,
    calculate_elastic_tensor,
    get_bravais_type,
)
from torch_sim.optimizers import frechet_cell_fire
from torch_sim.state import SimState
from torch_sim.units import UnitConversion


try:
    from mace.calculators.foundations_models import mace_mp

    from torch_sim.models.mace import MaceModel
except ImportError:
    pytest.skip("MACE not installed", allow_module_level=True)


@pytest.fixture
def mace_model(device: torch.device) -> MaceModel:
    mace_model = mace_mp(model="medium", default_dtype="float64", return_raw_model=True)

    return MaceModel(
        model=mace_model,
        device=device,
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=True,
    )


@pytest.mark.parametrize(
    ("sim_state_name", "expected_bravais_type", "atol"),
    [
        ("cu_sim_state", BravaisType.CUBIC, 2e-1),
        ("mg_sim_state", BravaisType.HEXAGONAL, 5e-1),
        ("sb_sim_state", BravaisType.TRIGONAL, 5e-1),
        ("tio2_sim_state", BravaisType.TETRAGONAL, 5e-1),
        ("ga_sim_state", BravaisType.ORTHORHOMBIC, 5e-1),
        ("niti_sim_state", BravaisType.MONOCLINIC, 5e-1),
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
        model_fixture_name: Name of the model fixture to use
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
    fire_init, fire_update = frechet_cell_fire(model=model, scalar_pressure=0.0)
    state = fire_init(state=state)
    fmax = 1e-5

    for _ in range(300):
        pressure = (
            -torch.trace(state.stress.squeeze()) / 3 * UnitConversion.eV_per_Ang3_to_GPa
        )
        current_fmax = torch.max(torch.abs(state.forces.squeeze()))
        if current_fmax < fmax and abs(pressure) < 1e-2:
            break
        state = fire_update(state=state)

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
        calculate_elastic_tensor(model, state=state, bravais_type=BravaisType.TRICLINIC)
        * UnitConversion.eV_per_Ang3_to_GPa
    )

    # Check if the elastic tensors are equal
    assert torch.allclose(C_symmetric, C_triclinic, atol=atol), (
        f"Elastic tensor mismatch for {expected_bravais_type} structure:\n"
        f"Difference matrix:\n{C_symmetric - C_triclinic}"
    )


def test_copper_elastic_properties(mace_model: MaceModel, cu_sim_state: SimState):
    """Test calculation of elastic properties for copper."""

    # Relax positions and cell
    fire_init, fire_update = frechet_cell_fire(model=mace_model, scalar_pressure=0.0)
    state = fire_init(state=cu_sim_state)
    fmax = 1e-5
    for _ in range(300):
        pressure = (
            -torch.trace(state.stress.squeeze()) / 3 * UnitConversion.eV_per_Ang3_to_GPa
        )
        current_fmax = torch.max(torch.abs(state.forces.squeeze()))
        if current_fmax < fmax and abs(pressure) < 1e-2:
            break
        state = fire_update(state=state)

    # Calculate elastic tensor
    bravais_type = get_bravais_type(state)
    elastic_tensor = calculate_elastic_tensor(
        mace_model, state=state, bravais_type=bravais_type
    )

    # Convert to GPa
    elastic_tensor = elastic_tensor * UnitConversion.eV_per_Ang3_to_GPa

    # Calculate elastic moduli
    bulk_modulus, shear_modulus, _, _ = calculate_elastic_moduli(elastic_tensor)

    device = state.device
    dtype = state.dtype

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
