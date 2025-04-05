from typing import Any

import pytest
import torch
from ase.build import bulk
from ase.spacegroup import crystal

from torch_sim.elastic import (
    BravaisType,
    calculate_elastic_moduli,
    calculate_elastic_tensor,
    get_bravais_type,
)
from torch_sim.io import atoms_to_state
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.state import SimState
from torch_sim.unbatched.unbatched_optimizers import frechet_cell_fire
from torch_sim.units import UnitConversion


try:
    from mace.calculators.foundations_models import mace_mp

    from torch_sim.unbatched.models.mace import UnbatchedMaceModel
except ImportError:
    pytest.skip("MACE not installed", allow_module_level=True)


@pytest.fixture
def mg_atoms() -> Any:
    """Create crystalline magnesium using ASE."""
    return bulk("Mg", "hcp", a=3.17, c=5.14)


@pytest.fixture
def sb_atoms() -> Any:
    """Create crystalline antimony using ASE."""
    return bulk("Sb", "rhombohedral", a=4.58, alpha=60)


@pytest.fixture
def tio2_atoms() -> Any:
    """Create crystalline TiO2 using ASE."""
    a, c = 4.60, 2.96
    symbols = ["Ti", "O", "O"]
    basis = [
        (0.5, 0.5, 0),  # Ti
        (0.695679, 0.695679, 0.5),  # O
    ]
    return crystal(
        symbols,
        basis=basis,
        spacegroup=136,  # P4_2/mnm
        cellpar=[a, a, c, 90, 90, 90],
    )


@pytest.fixture
def ga_atoms() -> Any:
    """Create crystalline Ga using ASE."""
    a, b, c = 4.43, 7.60, 4.56
    symbols = ["Ga"]
    basis = [
        (0, 0.344304, 0.415401),  # Ga
    ]
    return crystal(
        symbols,
        basis=basis,
        spacegroup=64,  # Cmce
        cellpar=[a, b, c, 90, 90, 90],
    )


@pytest.fixture
def niti_atoms() -> Any:
    """Create crystalline NiTi using ASE."""
    a, b, c = 2.89, 3.97, 4.83
    alpha, beta, gamma = 90.00, 105.23, 90.00
    symbols = ["Ni", "Ti"]
    basis = [
        (0.369548, 0.25, 0.217074),  # Ni
        (0.076622, 0.25, 0.671102),  # Ti
    ]
    return crystal(
        symbols,
        basis=basis,
        spacegroup=11,
        cellpar=[a, b, c, alpha, beta, gamma],
    )


@pytest.fixture
def sb_sim_state(sb_atoms: Any, device: torch.device) -> Any:
    """Create a basic state from sb_atoms."""
    return atoms_to_state(sb_atoms, device, torch.float64)


@pytest.fixture
def cu_sim_state(cu_atoms: Any, device: torch.device) -> Any:
    """Create a basic state from cu_atoms."""
    return atoms_to_state(cu_atoms, device, torch.float64)


@pytest.fixture
def mg_sim_state(mg_atoms: Any, device: torch.device) -> Any:
    """Create a basic state from mg_atoms."""
    return atoms_to_state(mg_atoms, device, torch.float64)


@pytest.fixture
def tio2_sim_state(tio2_atoms: Any, device: torch.device) -> Any:
    """Create a basic state from tio2_atoms."""
    return atoms_to_state(tio2_atoms, device, torch.float64)


@pytest.fixture
def ga_sim_state(ga_atoms: Any, device: torch.device) -> Any:
    """Create a basic state from ga_atoms."""
    return atoms_to_state(ga_atoms, device, torch.float64)


@pytest.fixture
def niti_sim_state(niti_atoms: Any, device: torch.device) -> Any:
    """Create a basic state from niti_atoms."""
    return atoms_to_state(niti_atoms, device, torch.float64)


@pytest.fixture
def torchsim_mace_model(device: torch.device) -> UnbatchedMaceModel:
    mace_model = mace_mp(model="medium", default_dtype="float64", return_raw_model=True)

    return UnbatchedMaceModel(
        model=mace_model,
        neighbor_list_fn=vesin_nl_ts,
        device=device,
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=True,
    )


@pytest.mark.parametrize(
    ("sim_state_name", "model_fixture_name", "expected_bravais_type", "atol"),
    [
        ("cu_sim_state", "torchsim_mace_model", BravaisType.CUBIC, 2e-1),
        ("mg_sim_state", "torchsim_mace_model", BravaisType.HEXAGONAL, 5e-1),
        ("sb_sim_state", "torchsim_mace_model", BravaisType.TRIGONAL, 5e-1),
        ("tio2_sim_state", "torchsim_mace_model", BravaisType.TETRAGONAL, 5e-1),
        ("ga_sim_state", "torchsim_mace_model", BravaisType.ORTHORHOMBIC, 5e-1),
        ("niti_sim_state", "torchsim_mace_model", BravaisType.MONOCLINIC, 5e-1),
    ],
)
def test_elastic_tensor_symmetries(
    sim_state_name: str,
    model_fixture_name: str,
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
    model = request.getfixturevalue(model_fixture_name)
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


@pytest.mark.flaky(reruns=3)
def test_copper_elastic_properties(
    torchsim_mace_model: UnbatchedMaceModel, cu_sim_state: SimState
):
    """Test calculation of elastic properties for copper."""

    # Relax positions and cell
    fire_init, fire_update = frechet_cell_fire(
        model=torchsim_mace_model, scalar_pressure=0.0
    )
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
        torchsim_mace_model, state=state, bravais_type=bravais_type
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
            [171.2151, 130.5025, 130.5025, 0.0000, 0.0000, 0.0000],
            [130.5025, 171.2151, 130.5025, 0.0000, 0.0000, 0.0000],
            [130.5025, 130.5025, 171.2151, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 70.8029, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 70.8029, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 70.8029],
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
