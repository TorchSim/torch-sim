import pytest
import torch
from ase.atoms import Atoms

import torch_sim as ts
from tests.conftest import MaceUrls
from tests.unbatched.conftest import make_unbatched_model_calculator_consistency_test


try:
    from mace.calculators import MACECalculator
    from mace.calculators.foundations_models import mace_mp, mace_off

    from torch_sim.unbatched.models.mace import UnbatchedMaceModel
except ImportError:
    pytest.skip("MACE not installed", allow_module_level=True)


mace_model = mace_mp(model=MaceUrls.mace_small, return_raw_model=True)
mace_off_model = mace_off(model=MaceUrls.mace_off_small, return_raw_model=True)


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture to provide the default dtype for testing."""
    return torch.float32


@pytest.fixture
def ase_mace_calculator() -> MACECalculator:
    return mace_mp(
        model=MaceUrls.mace_small,
        device="cpu",
        default_dtype="float32",
        dispersion=False,
    )


@pytest.fixture
def torchsim_unbatched_mace_model(
    device: torch.device, dtype: torch.dtype
) -> UnbatchedMaceModel:
    return UnbatchedMaceModel(
        model=mace_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=True,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unbatched_mace_dtype_working(
    si_atoms: Atoms, dtype: torch.dtype, device: torch.device
) -> None:
    model = UnbatchedMaceModel(
        model=mace_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
    )

    state = ts.io.atoms_to_state(si_atoms, device, dtype)

    model.forward(state)


@pytest.fixture
def ase_mace_off_calculator() -> MACECalculator:
    return mace_off(
        model="small",
        device="cpu",
        default_dtype="float32",
        dispersion=False,
    )


@pytest.fixture
def torchsim_unbatched_mace_off_model(
    device: torch.device, dtype: torch.dtype
) -> UnbatchedMaceModel:
    return UnbatchedMaceModel(
        model=mace_off_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
    )


test_unbatched_mace_off_consistency = make_unbatched_model_calculator_consistency_test(
    test_name="mace_off",
    model_fixture_name="torchsim_unbatched_mace_off_model",
    calculator_fixture_name="ase_mace_off_calculator",
    sim_state_names=[
        "benzene_sim_state",
    ],
)


test_unbatched_mace_consistency = make_unbatched_model_calculator_consistency_test(
    test_name="mace",
    model_fixture_name="torchsim_unbatched_mace_model",
    calculator_fixture_name="ase_mace_calculator",
    sim_state_names=[
        "cu_sim_state",
        "mg_sim_state",
        "sb_sim_state",
        "tio2_sim_state",
        "ga_sim_state",
        "niti_sim_state",
        "ti_sim_state",
        "si_sim_state",
        "sio2_sim_state",
    ],
)
