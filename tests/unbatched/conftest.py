import typing

import pytest
import torch

import torch_sim as ts
from torch_sim.elastic import full_3x3_to_voigt_6_stress


if typing.TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

    from torch_sim.models.interface import ModelInterface
    from torch_sim.state import SimState


consistency_test_simstate_fixtures = (
    "cu_sim_state",
    "mg_sim_state",
    "sb_sim_state",
    "tio2_sim_state",
    "ga_sim_state",
    "niti_sim_state",
    "ti_sim_state",
    "si_sim_state",
    "sio2_sim_state",
    "ar_supercell_sim_state",
    "fe_supercell_sim_state",
    "benzene_sim_state",
)


def make_unbatched_model_calculator_consistency_test(
    test_name: str,
    model_fixture_name: str,
    calculator_fixture_name: str,
    sim_state_names: tuple[str, ...],
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    """Factory function to create unbatched model-calculator consistency tests.

    Args:
        test_name: Name of the test (used in the function name and messages)
        model_fixture_name: Name of the model fixture
        calculator_fixture_name: Name of the calculator fixture
        sim_state_names: sim_state fixture names to test
        rtol: Relative tolerance for numerical comparisons
        atol: Absolute tolerance for numerical comparisons
    """

    @pytest.mark.parametrize("sim_state_name", sim_state_names)
    def test_unbatched_model_calculator_consistency(
        sim_state_name: str,
        request: pytest.FixtureRequest,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Test consistency between unbatched model and calculator implementations."""
        # Get the model and calculator fixtures dynamically
        model: ModelInterface = request.getfixturevalue(model_fixture_name)
        calculator: Calculator = request.getfixturevalue(calculator_fixture_name)

        # Get the sim_state fixture dynamically using the name
        sim_state: SimState = (
            request.getfixturevalue(sim_state_name).to(device, dtype).split()[0]
        )

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
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(
            model_results["forces"],
            calc_forces,
            rtol=rtol,
            atol=atol,
        )

        if "stress" in model_results:
            calc_stress = torch.tensor(
                atoms.get_stress(),
                device=device,
                dtype=model_results["stress"].dtype,
            )

            torch.testing.assert_close(
                full_3x3_to_voigt_6_stress(model_results["stress"]),
                calc_stress,
                rtol=rtol,
                atol=atol,
            )

    # Rename the function to include the test name
    test_unbatched_model_calculator_consistency.__name__ = (
        f"test_unbatched_{test_name}_consistency"
    )
    return test_unbatched_model_calculator_consistency
