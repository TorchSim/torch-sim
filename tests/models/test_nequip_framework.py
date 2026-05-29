from __future__ import annotations

import traceback
import urllib.request
from pathlib import Path

import pytest

from tests.conftest import DEVICE, DTYPE
from tests.models.conftest import (
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)
from torch_sim.testing import SIMSTATE_BULK_GENERATORS, ModelTolerance


try:
    from nequip.ase import NequIPCalculator
    from nequip.scripts.compile import main

    from torch_sim.models.nequip_framework import NequIPFrameworkModel

    _IMPORT_ERROR: str | None = None
except (ImportError, ModuleNotFoundError):
    _IMPORT_ERROR = traceback.format_exc()

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None, reason=f"nequip not installed: {_IMPORT_ERROR}"
)


# Cache directory for compiled models (under tests/ for easy cleanup)
NEQUIP_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "nequip_compiled_models"

# Zenodo URL for NequIP-OAM-S model (more reliable than nequip.net for CI)
NEQUIP_OAM_S_ZENODO_URL = (
    "https://zenodo.org/records/18775904/files/NequIP-OAM-S-0.1.nequip.zip?download=1"
)
NEQUIP_OAM_S_ZIP_NAME = "NequIP-OAM-S-0.1.nequip.zip"


def _get_nequip_model_zip() -> Path:
    """Download NequIP-OAM-S model from Zenodo if not already cached."""
    NEQUIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = NEQUIP_CACHE_DIR / NEQUIP_OAM_S_ZIP_NAME

    if not zip_path.exists():
        urllib.request.urlretrieve(NEQUIP_OAM_S_ZENODO_URL, zip_path)  # noqa: S310

    return zip_path


@pytest.fixture(scope="session")
def compiled_ase_nequip_model_path() -> Path:
    """Compile NequIP OAM-S model from Zenodo for ASE (with persistent caching)."""
    NEQUIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    output_model_name = f"mir-group__NequIP-OAM-S__0.1__{DEVICE.type}_ase.nequip.pt2"
    output_path = NEQUIP_CACHE_DIR / output_model_name

    # Only compile if not already cached
    if not output_path.exists():
        model_zip_path = _get_nequip_model_zip()
        main(
            args=[
                str(model_zip_path),
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
def compiled_batch_nequip_model_path() -> Path:
    """Compile NequIP OAM-S model from Zenodo for batch (with persistent caching)."""
    NEQUIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    output_model_name = f"mir-group__NequIP-OAM-S__0.1__{DEVICE.type}_batch.nequip.pt2"
    output_path = NEQUIP_CACHE_DIR / output_model_name

    # Only compile if not already cached
    if not output_path.exists():
        model_zip_path = _get_nequip_model_zip()
        main(
            args=[
                str(model_zip_path),
                str(output_path),
                "--mode",
                "aotinductor",
                "--device",
                DEVICE.type,
                "--target",
                "batch",
            ]
        )

    return output_path


@pytest.fixture(scope="session")
def nequip_model(compiled_batch_nequip_model_path: Path) -> NequIPFrameworkModel:
    """Create an NequIPModel wrapper for the pretrained model."""
    return NequIPFrameworkModel.from_compiled_model(
        compiled_batch_nequip_model_path,
        device=DEVICE,
        chemical_species_to_atom_type_map=True,  # Use identity mapping without warning
    )


@pytest.fixture(scope="session")
def nequip_calculator(compiled_ase_nequip_model_path: Path) -> NequIPCalculator:
    """Create an NequIPCalculator for the pretrained model."""
    return NequIPCalculator.from_compiled_model(
        str(compiled_ase_nequip_model_path), device=DEVICE
    )


# NOTE: skip molecule sim states as stress in NequIP gave inf.
test_nequip_consistency = make_model_calculator_consistency_test(
    test_name="nequip",
    model_fixture_name="nequip_model",
    calculator_fixture_name="nequip_calculator",
    sim_state_names=tuple(SIMSTATE_BULK_GENERATORS.keys()),
    energy_atol=ModelTolerance.LOOSE,
    dtype=DTYPE,
    device=DEVICE,
)

test_nequip_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="nequip_model",
    dtype=DTYPE,
    device=DEVICE,
)
