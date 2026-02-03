"""Pytest fixtures and test factories for model testing."""

import typing

import pytest
import torch

from tests.conftest import DEVICE, DTYPE
from torch_sim.testing import SIMSTATE_GENERATORS, assert_model_calculator_consistency


if typing.TYPE_CHECKING:
    from torch_sim.models.interface import ModelInterface


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
        test_name: Name of the test (used in the function name)
        model_fixture_name: Name of the model fixture
        calculator_fixture_name: Name of the calculator fixture
        sim_state_names: sim_state fixture names to test
        device: Device to run tests on
        dtype: Data type to use for tests
        energy_rtol: Relative tolerance for energy comparisons
        energy_atol: Absolute tolerance for energy comparisons
        force_rtol: Relative tolerance for force comparisons
        force_atol: Absolute tolerance for force comparisons
        stress_rtol: Relative tolerance for stress comparisons
        stress_atol: Absolute tolerance for stress comparisons

    Returns:
        A pytest test function that can be assigned to a module-level variable
    """

    @pytest.mark.parametrize("sim_state_name", sim_state_names)
    def _model_calculator_consistency_test(
        sim_state_name: str, request: pytest.FixtureRequest
    ) -> None:
        """Test consistency between model and calculator implementations."""
        model = request.getfixturevalue(model_fixture_name)
        calculator = request.getfixturevalue(calculator_fixture_name)

        # Generate sim_state from the generator
        generator = SIMSTATE_GENERATORS[sim_state_name]
        sim_state = generator(device, dtype)

        assert_model_calculator_consistency(
            model=model,
            calculator=calculator,
            sim_state=sim_state,
            energy_rtol=energy_rtol,
            energy_atol=energy_atol,
            force_rtol=force_rtol,
            force_atol=force_atol,
            stress_rtol=stress_rtol,
            stress_atol=stress_atol,
        )

    _model_calculator_consistency_test.__name__ = f"test_{test_name}_consistency"
    return _model_calculator_consistency_test


def make_validate_model_outputs_test(
    model_fixture_name: str,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
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
