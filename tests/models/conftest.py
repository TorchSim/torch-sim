import typing
from typing import Final

import pytest
import torch

import torch_sim as ts
from torch_sim.elastic import full_3x3_to_voigt_6_stress


if typing.TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

    from torch_sim.models.interface import ModelInterface
    from torch_sim.state import SimState


consistency_test_simstate_fixtures: Final[tuple[str, ...]] = (
    "cu_sim_state",
    "mg_sim_state",
    "sb_sim_state",
    "tio2_sim_state",
    "ga_sim_state",
    "niti_sim_state",
    "ti_sim_state",
    "si_sim_state",
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
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    """Factory function to create model-calculator consistency tests.

    Args:
        test_name: Name of the test (used in the function name and messages)
        model_fixture_name: Name of the model fixture
        calculator_fixture_name: Name of the calculator fixture
        sim_state_names: sim_state fixture names to test
        rtol: Relative tolerance for numerical comparisons
        atol: Absolute tolerance for numerical comparisons
    """

    @pytest.mark.parametrize("sim_state_name", sim_state_names)
    def test_model_calculator_consistency(
        sim_state_name: str,
        request: pytest.FixtureRequest,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Test consistency between model and calculator implementations."""
        # Get the model and calculator fixtures dynamically
        model: ModelInterface = request.getfixturevalue(model_fixture_name)
        calculator: Calculator = request.getfixturevalue(calculator_fixture_name)

        # Get the sim_state fixture dynamically using the name
        sim_state: SimState = request.getfixturevalue(sim_state_name).to(device, dtype)

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
            ).unsqueeze(0)

            torch.testing.assert_close(
                full_3x3_to_voigt_6_stress(model_results["stress"]),
                calc_stress,
                rtol=rtol,
                atol=atol,
            )

    # Rename the function to include the test name
    test_model_calculator_consistency.__name__ = f"test_{test_name}_consistency"
    return test_model_calculator_consistency


def make_validate_model_outputs_test(
    model_fixture_name: str,
):
    """Factory function to create model output validation tests.

    Args:
        test_name: Name of the test (used in the function name and messages)
        model_fixture_name: Name of the model fixture to validate
    """

    def test_model_output_validation(
        request: pytest.FixtureRequest,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Test that a model implementation follows the ModelInterface contract."""
        # Get the model fixture dynamically
        model: ModelInterface = request.getfixturevalue(model_fixture_name)

        from ase.build import bulk

        assert model.dtype is not None
        assert model.device is not None
        assert model.compute_stress is not None
        assert model.compute_forces is not None

        try:
            if not model.compute_stress:
                model.compute_stress = True
            stress_computed = True
        except NotImplementedError:
            stress_computed = False

        try:
            if not model.compute_forces:
                model.compute_forces = True
            force_computed = True
        except NotImplementedError:
            force_computed = False

        si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
        fe_atoms = bulk("Fe", "fcc", a=5.26, cubic=True).repeat([3, 1, 1])

        sim_state = ts.io.atoms_to_state([si_atoms, fe_atoms], device, dtype)

        og_positions = sim_state.positions.clone()
        og_cell = sim_state.cell.clone()
        og_batch = sim_state.batch.clone()
        og_atomic_numbers = sim_state.atomic_numbers.clone()

        model_output = model.forward(sim_state)

        # assert model did not mutate the input
        assert torch.allclose(og_positions, sim_state.positions)
        assert torch.allclose(og_cell, sim_state.cell)
        assert torch.allclose(og_batch, sim_state.batch)
        assert torch.allclose(og_atomic_numbers, sim_state.atomic_numbers)

        # assert model output has the correct keys
        assert "energy" in model_output
        assert "forces" in model_output if force_computed else True
        assert "stress" in model_output if stress_computed else True

        # assert model output shapes are correct
        assert model_output["energy"].shape == (2,)
        assert model_output["forces"].shape == (20, 3) if force_computed else True
        assert model_output["stress"].shape == (2, 3, 3) if stress_computed else True

        si_state = ts.io.atoms_to_state([si_atoms], device, dtype)
        fe_state = ts.io.atoms_to_state([fe_atoms], device, dtype)

        si_model_output = model.forward(si_state)
        assert torch.allclose(
            si_model_output["energy"], model_output["energy"][0], atol=10e-3
        )
        assert torch.allclose(
            si_model_output["forces"],
            model_output["forces"][: si_state.n_atoms],
            atol=10e-3,
        )
        # assert torch.allclose(
        #     si_model_output["stress"],
        #     model_output["stress"][0],
        #     atol=10e-3,
        # )

        fe_model_output = model.forward(fe_state)
        si_model_output = model.forward(si_state)

        assert torch.allclose(
            fe_model_output["energy"], model_output["energy"][1], atol=10e-2
        )
        assert torch.allclose(
            fe_model_output["forces"],
            model_output["forces"][si_state.n_atoms :],
            atol=10e-2,
        )
        # assert torch.allclose(
        #     fe_model_output["stress"],
        #     model_output["stress"][1],
        #     atol=10e-3,
        # )

    # Rename the function to include the test name
    test_model_output_validation.__name__ = f"test_{model_fixture_name}_output_validation"
    return test_model_output_validation
