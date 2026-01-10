import typing
from typing import Final

import pytest
import torch

import torch_sim as ts
from tests.conftest import DEVICE
from torch_sim.elastic import full_3x3_to_voigt_6_stress


if typing.TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

    from torch_sim.models.interface import ModelInterface


consistency_test_simstate_fixtures: Final[tuple[str, ...]] = (
    "cu_sim_state",
    "mg_sim_state",
    "sb_sim_state",
    "tio2_sim_state",
    "ga_sim_state",
    "niti_sim_state",
    "ti_sim_state",
    "si_sim_state",
    "rattled_si_sim_state",
    "sio2_sim_state",
    "rattled_sio2_sim_state",
    "ar_supercell_sim_state",
    "fe_supercell_sim_state",
    "casio3_sim_state",
    "benzene_sim_state",
)


def make_model_calculator_consistency_test(
    test_name: str,
    model_fixture_name: str,
    calculator_fixture_name: str,
    sim_state_names: tuple[str, ...],
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.float64,
    energy_rtol: float = 1e-5,
    energy_atol: float = 1e-5,
    force_rtol: float = 1e-5,
    force_atol: float = 1e-5,
    stress_rtol: float = 1e-5,
    stress_atol: float = 1e-5,
):
    """Factory function to create model-calculator consistency tests.

    Args:
        test_name: Name of the test (used in the function name and messages)
        model_fixture_name: Name of the model fixture
        calculator_fixture_name: Name of the calculator fixture
        sim_state_names: sim_state fixture names to test
        energy_rtol: Relative tolerance for energy comparisons
        energy_atol: Absolute tolerance for energy comparisons
        force_rtol: Relative tolerance for force comparisons
        force_atol: Absolute tolerance for force comparisons
        stress_rtol: Relative tolerance for stress comparisons
        stress_atol: Absolute tolerance for stress comparisons
    """

    @pytest.mark.parametrize("sim_state_name", sim_state_names)
    def test_model_calculator_consistency(
        sim_state_name: str, request: pytest.FixtureRequest
    ) -> None:
        """Test consistency between model and calculator implementations."""
        # Get the model and calculator fixtures dynamically
        model: ModelInterface = request.getfixturevalue(model_fixture_name)
        calculator: Calculator = request.getfixturevalue(calculator_fixture_name)

        # Get the sim_state fixture dynamically using the name
        sim_state: ts.SimState = request.getfixturevalue(sim_state_name).to(device, dtype)

        # Set up ASE calculator
        atoms = ts.io.state_to_atoms(sim_state)[0]
        atoms.calc = calculator

        # Get model results
        model_results = model(sim_state)

        # Get calculator results
        calc_forces = torch.tensor(
            atoms.get_forces(),
            device=device,
            dtype=model_results["forces"].dtype,
        )

        # Test consistency with specified tolerances
        torch.testing.assert_close(
            model_results["energy"].item(),
            atoms.get_potential_energy(),
            rtol=energy_rtol,
            atol=energy_atol,
        )
        torch.testing.assert_close(
            model_results["forces"],
            calc_forces,
            rtol=force_rtol,
            atol=force_atol,
        )

        if "stress" in model_results and all(sim_state.pbc):
            calc_stress = torch.tensor(
                atoms.get_stress(),
                device=device,
                dtype=model_results["stress"].dtype,
            ).unsqueeze(0)

            torch.testing.assert_close(
                full_3x3_to_voigt_6_stress(model_results["stress"]),
                calc_stress,
                rtol=stress_rtol,
                atol=stress_atol,
                equal_nan=True,
            )

    # Rename the function to include the test name
    test_model_calculator_consistency.__name__ = f"test_{test_name}_consistency"
    return test_model_calculator_consistency


def make_validate_model_outputs_test(
    model_fixture_name: str,
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.float64,
):
    """Factory function to create model output validation tests.

    Args:
        model_fixture_name: Name of the model fixture to validate
        device: Device to run validation on
        dtype: Data type to use for validation
    """
    from torch_sim.models.interface import validate_model_outputs

    def test_model_output_validation(request: pytest.FixtureRequest) -> None:
        """Test that a model implementation follows the ModelInterface contract."""
        model: ModelInterface = request.getfixturevalue(model_fixture_name)
        validate_model_outputs(model, device, dtype)

    test_model_output_validation.__name__ = f"test_{model_fixture_name}_output_validation"
    return test_model_output_validation
