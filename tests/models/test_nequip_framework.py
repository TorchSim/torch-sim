import traceback
from pathlib import Path

import pytest

from tests.conftest import DEVICE, DTYPE
from tests.models.conftest import (
    consistency_test_simstate_fixtures,
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)


try:
    from nequip.ase import NequIPCalculator
    from nequip.scripts.compile import main

    from torch_sim.models.nequip_framework import (
        NequIPFrameworkModel,
        from_compiled_model,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        f"nequip not installed: {traceback.format_exc()}", allow_module_level=True
    )


@pytest.fixture(scope="session")
def compiled_nequip_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile NequIP OAM-L model from nequip.net."""
    tmp_path = tmp_path_factory.mktemp("nequip_compiled")
    output_model_name = "mir-group__NequIP-OAM-L__0.1.nequip.pt2"
    output_path = Path(tmp_path) / output_model_name

    main(
        args=[
            "nequip.net:mir-group/NequIP-OAM-L:0.1",
            str(output_path),
            "--mode",
            "aotinductor",
            "--device",
            DEVICE.type,
            "--target",
            "ase",
        ]
    )

    return output_path


@pytest.fixture(scope="session")
def nequip_model(compiled_nequip_model_path: Path) -> NequIPFrameworkModel:
    """Create an NequIPModel wrapper for the pretrained model."""
    compiled_model, (r_max, type_names) = from_compiled_model(
        compiled_nequip_model_path, device=DEVICE
    )
    return NequIPFrameworkModel(
        model=compiled_model,
        r_max=r_max,
        type_names=type_names,
        device=DEVICE,
    )


@pytest.fixture(scope="session")
def nequip_calculator(compiled_nequip_model_path: Path) -> NequIPCalculator:
    """Create an NequIPCalculator for the pretrained model."""
    return NequIPCalculator.from_compiled_model(
        str(compiled_nequip_model_path), device=DEVICE
    )


def test_nequip_initialization(compiled_nequip_model_path: Path) -> None:
    """Test that the NequIP model initializes correctly."""
    compiled_model, (r_max, type_names) = from_compiled_model(
        compiled_nequip_model_path, device=DEVICE
    )
    model = NequIPFrameworkModel(
        model=compiled_model,
        r_max=r_max,
        type_names=type_names,
        device=DEVICE,
    )
    assert model._device == DEVICE  # noqa: SLF001


test_nequip_consistency = make_model_calculator_consistency_test(
    test_name="nequip",
    model_fixture_name="nequip_model",
    calculator_fixture_name="nequip_calculator",
    sim_state_names=consistency_test_simstate_fixtures,
    energy_atol=5e-5,
    dtype=DTYPE,
    device=DEVICE,
)

test_nequip_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="nequip_model",
    dtype=DTYPE,
    device=DEVICE,
)
