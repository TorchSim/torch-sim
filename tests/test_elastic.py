from typing import Any

import pytest
import torch
from ase.build import bulk
from ase.spacegroup import crystal

from torch_sim.elastic import (
    BravaisType,
    calculate_elastic_moduli,
    calculate_elastic_tensor,
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


def get_bravais_type(  # noqa: PLR0911
    state: SimState, length_tol: float = 1e-3, angle_tol: float = 0.1
) -> BravaisType:
    """Check and return the crystal system of a structure.

    This function determines the crystal system by analyzing the lattice
    parameters and angles without using spglib.

    Args:
        state: SimState object representing the crystal structure
        length_tol: Tolerance for floating-point comparisons of lattice lengths
        angle_tol: Tolerance for floating-point comparisons of lattice angles in degrees

    Returns:
        BravaisType: Bravais type
    """
    # Get cell parameters
    cell = state.cell.squeeze()
    a, b, c = torch.linalg.norm(cell, axis=1)

    # Get cell angles in degrees
    alpha = torch.rad2deg(torch.arccos(torch.dot(cell[1], cell[2]) / (b * c)))
    beta = torch.rad2deg(torch.arccos(torch.dot(cell[0], cell[2]) / (a * c)))
    gamma = torch.rad2deg(torch.arccos(torch.dot(cell[0], cell[1]) / (a * b)))

    # Cubic: a = b = c, alpha = beta = gamma = 90°
    if (
        abs(a - b) < length_tol
        and abs(b - c) < length_tol
        and abs(alpha - 90) < angle_tol
        and abs(beta - 90) < angle_tol
        and abs(gamma - 90) < angle_tol
    ):
        return BravaisType.CUBIC

    # Hexagonal: a = b ≠ c, alpha = beta = 90°, gamma = 120°
    if (
        abs(a - b) < length_tol
        and abs(alpha - 90) < angle_tol
        and abs(beta - 90) < angle_tol
        and abs(gamma - 120) < angle_tol
    ):
        return BravaisType.HEXAGONAL

    # Tetragonal: a = b ≠ c, alpha = beta = gamma = 90°
    if (
        abs(a - b) < length_tol
        and abs(a - c) > length_tol
        and abs(alpha - 90) < angle_tol
        and abs(beta - 90) < angle_tol
        and abs(gamma - 90) < angle_tol
    ):
        return BravaisType.TETRAGONAL

    # Orthorhombic: a ≠ b ≠ c, alpha = beta = gamma = 90°
    if (
        abs(alpha - 90) < angle_tol
        and abs(beta - 90) < angle_tol
        and abs(gamma - 90) < angle_tol
        and abs(a - b) > length_tol
        and (abs(b - c) > length_tol or abs(a - c) > length_tol)
    ):
        return BravaisType.ORTHORHOMBIC

    # Monoclinic: a ≠ b ≠ c, alpha = gamma = 90°, beta ≠ 90°
    if (
        abs(alpha - 90) < angle_tol
        and abs(gamma - 90) < angle_tol
        and abs(beta - 90) > angle_tol
    ):
        return BravaisType.MONOCLINIC

    # Trigonal/Rhombohedral: a = b = c, alpha = beta = gamma ≠ 90°
    if (
        abs(a - b) < length_tol
        and abs(b - c) < length_tol
        and abs(alpha - beta) < angle_tol
        and abs(beta - gamma) < angle_tol
        and abs(alpha - 90) > angle_tol
    ):
        return BravaisType.TRIGONAL

    # Triclinic: a ≠ b ≠ c, alpha ≠ beta ≠ gamma ≠ 90°
    return BravaisType.TRICLINIC


def test_cubic(torchsim_mace_model: UnbatchedMaceModel, cu_sim_state: SimState):
    """Test the elastic tensor of a cubic structure of Cu"""

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

    # Verify the space group is cubic for the relaxed structure
    assert get_bravais_type(state) == BravaisType.CUBIC, (
        f"Structure is not cubic and has bravais type {get_bravais_type(state)}"
    )

    # Calculate elastic tensor
    C_cubic = (
        calculate_elastic_tensor(
            torchsim_mace_model, state=state, bravais_type=BravaisType.CUBIC
        )
        * UnitConversion.eV_per_Ang3_to_GPa
    )
    C_triclinic = (
        calculate_elastic_tensor(
            torchsim_mace_model, state=state, bravais_type=BravaisType.TRICLINIC
        )
        * UnitConversion.eV_per_Ang3_to_GPa
    )

    # Check if the elastic tensors are equal
    assert torch.allclose(C_cubic, C_triclinic, atol=2e-1), f"{C_cubic - C_triclinic}"


def test_hexagonal(torchsim_mace_model: UnbatchedMaceModel, mg_sim_state: SimState):
    """Test the elastic tensor of a hexagonal structure of Mg"""

    # Relax positions and cell
    fire_init, fire_update = frechet_cell_fire(
        model=torchsim_mace_model, scalar_pressure=0.0
    )
    state = fire_init(state=mg_sim_state)
    fmax = 1e-5
    for _ in range(300):
        pressure = (
            -torch.trace(state.stress.squeeze()) / 3 * UnitConversion.eV_per_Ang3_to_GPa
        )
        current_fmax = torch.max(torch.abs(state.forces.squeeze()))
        if current_fmax < fmax and abs(pressure) < 1e-2:
            break
        state = fire_update(state=state)

    # Verify the space group is hexagonal for the relaxed structure
    assert get_bravais_type(state) == BravaisType.HEXAGONAL, (
        f"Structure is not hexagonal and has bravais type {get_bravais_type(state)}"
    )

    # Calculate elastic tensor
    C_hexagonal = (
        calculate_elastic_tensor(
            torchsim_mace_model, state=state, bravais_type=BravaisType.HEXAGONAL
        )
        * UnitConversion.eV_per_Ang3_to_GPa
    )
    C_triclinic = (
        calculate_elastic_tensor(
            torchsim_mace_model, state=state, bravais_type=BravaisType.TRICLINIC
        )
        * UnitConversion.eV_per_Ang3_to_GPa
    )

    # Check if the elastic tensors are equal
    assert torch.allclose(C_hexagonal, C_triclinic, atol=5e-1), (
        f"{C_hexagonal - C_triclinic}"
    )


def test_trigonal(torchsim_mace_model: UnbatchedMaceModel, sb_sim_state: SimState):
    """Test the elastic tensor of a trigonal structure of Sb"""

    # Relax positions and cell
    fire_init, fire_update = frechet_cell_fire(
        model=torchsim_mace_model, scalar_pressure=0.0
    )
    state = fire_init(state=sb_sim_state)
    fmax = 1e-5
    for _ in range(300):
        pressure = (
            -torch.trace(state.stress.squeeze()) / 3 * UnitConversion.eV_per_Ang3_to_GPa
        )
        current_fmax = torch.max(torch.abs(state.forces.squeeze()))
        if current_fmax < fmax and abs(pressure) < 1e-2:
            break
        state = fire_update(state=state)

    # Verify the space group is trigonal for the relaxed structure
    assert get_bravais_type(state) == BravaisType.TRIGONAL, (
        f"Structure is not trigonal and has bravais type {get_bravais_type(state)}"
    )

    # Calculate elastic tensor
    C_trigonal = (
        calculate_elastic_tensor(
            torchsim_mace_model, state=state, bravais_type=BravaisType.TRIGONAL
        )
        * UnitConversion.eV_per_Ang3_to_GPa
    )
    C_triclinic = (
        calculate_elastic_tensor(
            torchsim_mace_model, state=state, bravais_type=BravaisType.TRICLINIC
        )
        * UnitConversion.eV_per_Ang3_to_GPa
    )

    # Check if the elastic tensors are equal
    assert torch.allclose(C_trigonal, C_triclinic, atol=5e-1), (
        f"{C_trigonal - C_triclinic}"
    )


def test_tetragonal(torchsim_mace_model: UnbatchedMaceModel, tio2_sim_state: SimState):
    """Test the elastic tensor of a tetragonal structure of BaTiO3"""

    # Relax positions and cell
    fire_init, fire_update = frechet_cell_fire(
        model=torchsim_mace_model, scalar_pressure=0.0
    )
    state = fire_init(state=tio2_sim_state)
    fmax = 1e-5
    for _ in range(300):
        pressure = (
            -torch.trace(state.stress.squeeze()) / 3 * UnitConversion.eV_per_Ang3_to_GPa
        )
        current_fmax = torch.max(torch.abs(state.forces.squeeze()))
        if current_fmax < fmax and abs(pressure) < 1e-2:
            break
        state = fire_update(state=state)

    # Verify the space group is tetragonal for the relaxed structure
    assert get_bravais_type(state) == BravaisType.TETRAGONAL, (
        f"Structure is not tetragonal and has bravais type {get_bravais_type(state)}"
    )

    # Calculate elastic tensor
    C_tetragonal = (
        calculate_elastic_tensor(
            torchsim_mace_model, state=state, bravais_type=BravaisType.TETRAGONAL
        )
        * UnitConversion.eV_per_Ang3_to_GPa
    )
    C_triclinic = (
        calculate_elastic_tensor(
            torchsim_mace_model, state=state, bravais_type=BravaisType.TRICLINIC
        )
        * UnitConversion.eV_per_Ang3_to_GPa
    )

    # Check if the elastic tensors are equal
    assert torch.allclose(C_tetragonal, C_triclinic, atol=5e-1), (
        f"{C_tetragonal - C_triclinic}"
    )


def test_orthorhombic(torchsim_mace_model: UnbatchedMaceModel, ga_sim_state: SimState):
    """Test the elastic tensor of a orthorhombic structure of Ga"""

    # Relax positions and cell
    fire_init, fire_update = frechet_cell_fire(
        model=torchsim_mace_model, scalar_pressure=0.0
    )
    state = fire_init(state=ga_sim_state)
    fmax = 1e-5
    for _ in range(300):
        pressure = (
            -torch.trace(state.stress.squeeze()) / 3 * UnitConversion.eV_per_Ang3_to_GPa
        )
        current_fmax = torch.max(torch.abs(state.forces.squeeze()))
        if current_fmax < fmax and abs(pressure) < 1e-2:
            break
        state = fire_update(state=state)

    # Verify the space group is orthorhombic for the relaxed structure
    assert get_bravais_type(state) == BravaisType.ORTHORHOMBIC, (
        f"Structure is not orthorhombic and has bravais type {get_bravais_type(state)}"
    )

    # Calculate elastic tensor
    C_orthorhombic = (
        calculate_elastic_tensor(
            torchsim_mace_model, state=state, bravais_type=BravaisType.ORTHORHOMBIC
        )
        * UnitConversion.eV_per_Ang3_to_GPa
    )
    C_triclinic = (
        calculate_elastic_tensor(
            torchsim_mace_model, state=state, bravais_type=BravaisType.TRICLINIC
        )
        * UnitConversion.eV_per_Ang3_to_GPa
    )

    # Check if the elastic tensors are equal
    assert torch.allclose(C_orthorhombic, C_triclinic, atol=5e-1), (
        f"{C_orthorhombic - C_triclinic}"
    )


def test_monoclinic(torchsim_mace_model: UnbatchedMaceModel, niti_sim_state: SimState):
    """Test the elastic tensor of a monoclinic structure of β-Ga2O3"""

    # Relax positions and cell
    fire_init, fire_update = frechet_cell_fire(
        model=torchsim_mace_model, scalar_pressure=0.0
    )
    state = fire_init(state=niti_sim_state)
    fmax = 1e-5
    for _ in range(300):
        pressure = (
            -torch.trace(state.stress.squeeze()) / 3 * UnitConversion.eV_per_Ang3_to_GPa
        )
        current_fmax = torch.max(torch.abs(state.forces.squeeze()))
        if current_fmax < fmax and abs(pressure) < 1e-2:
            break
        state = fire_update(state=state)

    # Verify the space group is monoclinic for the relaxed structure
    assert get_bravais_type(state) == BravaisType.MONOCLINIC, (
        f"Structure is not monoclinic and has bravais type {get_bravais_type(state)}"
    )

    # Calculate elastic tensor
    C_monoclinic = (
        calculate_elastic_tensor(
            torchsim_mace_model, state=state, bravais_type=BravaisType.MONOCLINIC
        )
        * UnitConversion.eV_per_Ang3_to_GPa
    )
    C_triclinic = (
        calculate_elastic_tensor(
            torchsim_mace_model, state=state, bravais_type=BravaisType.TRICLINIC
        )
        * UnitConversion.eV_per_Ang3_to_GPa
    )

    # Check if the elastic tensors are equal
    assert torch.allclose(C_monoclinic, C_triclinic, atol=5e-1), (
        f"{C_monoclinic - C_triclinic}"
    )


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
