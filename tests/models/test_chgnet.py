import traceback
from typing import Any, ClassVar

import pytest
import torch
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes

from tests.conftest import DEVICE
from tests.models.conftest import (
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)


try:
    from chgnet.model.model import CHGNet

    from torch_sim.models.chgnet import CHGNetModel
except (ImportError, ValueError):
    pytest.skip(
        f"CHGNet not installed: {traceback.format_exc()}", allow_module_level=True
    )


class CHGNetCalculator(Calculator):
    """ASE Calculator wrapper for CHGNet."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(self, model: CHGNet | None = None, **kwargs) -> None:
        Calculator.__init__(self, **kwargs)
        self.model = model or CHGNet.load()

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: Any = all_changes,
    ):
        if properties is None:
            properties = ["energy"]
        Calculator.calculate(self, atoms, properties, system_changes)

        # Convert ASE atoms to pymatgen Structure
        from pymatgen.io.ase import AseAtomsAdaptor

        structure = AseAtomsAdaptor.get_structure(atoms)

        # Get CHGNet predictions
        result = self.model.predict_structure(structure)

        # Convert to ASE format
        self.results = {}
        if "energy" in properties:
            # CHGNet returns energy per atom, convert to total energy
            self.results["energy"] = result["e"] * len(structure)

        if "forces" in properties:
            self.results["forces"] = result["f"]

        if "stress" in properties:
            self.results["stress"] = result["s"]


DTYPE = torch.float32


@pytest.fixture
def ts_chgnet_model() -> CHGNetModel:
    """Create a TorchSim CHGNet model for testing."""
    return CHGNetModel(
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )


@pytest.fixture
def ase_chgnet_calculator(ts_chgnet_model: CHGNetModel) -> CHGNetCalculator:
    """Create an ASE CHGNet calculator for testing."""
    # Use the same model instance to ensure consistency
    return CHGNetCalculator(model=ts_chgnet_model.model)


def test_chgnet_missing_atomic_numbers() -> None:
    """Test that CHGNet raises appropriate error when atomic numbers are missing."""
    model = CHGNetModel(
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )

    # Create state without atomic numbers by using a state dict
    state_dict = {
        "positions": torch.randn(8, 3, device=DEVICE, dtype=DTYPE),
        "cell": torch.eye(3, device=DEVICE, dtype=DTYPE).unsqueeze(0),
        "pbc": True,
        "atomic_numbers": None,  # Missing atomic numbers
        "system_idx": torch.zeros(8, dtype=torch.long, device=DEVICE),
    }

    with pytest.raises(ValueError, match="Atomic numbers must be provided"):
        model.forward(state_dict)


test_chgnet_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="ts_chgnet_model", dtype=DTYPE
)

test_chgnet_consistency = make_model_calculator_consistency_test(
    test_name="chgnet",
    model_fixture_name="ts_chgnet_model",
    calculator_fixture_name="ase_chgnet_calculator",
    sim_state_names=("si_sim_state", "cu_sim_state", "mg_sim_state", "ti_sim_state"),
    dtype=DTYPE,
    energy_rtol=1e-4,
    energy_atol=1e-4,
    force_rtol=1e-4,
    force_atol=1e-4,
    stress_rtol=1e-3,
    stress_atol=1e-3,
)
