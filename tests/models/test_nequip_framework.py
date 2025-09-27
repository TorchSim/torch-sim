import urllib.request
from enum import StrEnum
from pathlib import Path

import pytest

from tests.conftest import DEVICE
from tests.models.conftest import make_model_calculator_consistency_test


try:
    from nequip.ase import NequIPCalculator

    from torch_sim.models.nequip_framework import (
        NequIPFrameworkModel,
        from_compiled_model,
    )
except ImportError:
    pytest.skip("nequip not installed", allow_module_level=True)


class NequIPUrls(StrEnum):
    """Checkpoint download URLs for NequIP models."""

    Si = "https://github.com/abhijeetgangan/pt_model_checkpoints/raw/refs/heads/main/nequip/Si.nequip.pth"


@pytest.fixture(scope="session")
def model_path_nequip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("nequip_checkpoints")
    model_name = "Si.nequip.pth"
    model_path = Path(tmp_path) / model_name

    if not model_path.is_file():
        urllib.request.urlretrieve(NequIPUrls.Si, model_path)  # noqa: S310

    return model_path


@pytest.fixture
def nequip_model(model_path_nequip: Path) -> NequIPFrameworkModel:
    """Create an NequIPModel wrapper for the pretrained model."""
    compiled_model, (r_max, type_names) = from_compiled_model(
        model_path_nequip, device=DEVICE
    )
    return NequIPFrameworkModel(
        model=compiled_model,
        r_max=r_max,
        type_names=type_names,
        device=DEVICE,
    )


@pytest.fixture
def nequip_calculator(model_path_nequip: Path) -> NequIPCalculator:
    """Create an NequIPCalculator for the pretrained model."""
    return NequIPCalculator.from_compiled_model(str(model_path_nequip), device=DEVICE)


def test_nequip_initialization(model_path_nequip: Path) -> None:
    """Test that the NequIP model initializes correctly."""
    compiled_model, (r_max, type_names) = from_compiled_model(
        model_path_nequip, device=DEVICE
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
    sim_state_names=("si_sim_state", "rattled_si_sim_state"),
)

# TODO (AG): Test multi element models
