import pytest
import torch
from ase.build import bulk, molecule

from tests.models.conftest import (
    consistency_test_simstate_fixtures,
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)
from torch_sim.io import atoms_to_state
from torch_sim.models.graphpes import GraphPESWrapper


try:
    from graph_pes.atomic_graph import AtomicGraph, to_batch
    from graph_pes.interfaces import mace_mp
    from graph_pes.models import LennardJones, SchNet, TensorNet, ZEmbeddingNequIP
except ImportError:
    pytest.skip("graph-pes not installed", allow_module_level=True)


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture to provide the default dtype for testing."""
    return torch.float32


def test_graphpes_isolated(device: torch.device):
    # test that the raw model and torch_sim wrapper give the same results
    # for an isolated, unbatched structure

    water_atoms = molecule("H2O")
    water_atoms.center(vacuum=10.0)

    gp_model = SchNet(cutoff=5.5)
    gp_graph = AtomicGraph.from_ase(water_atoms, cutoff=5.5)
    gp_energy = gp_model.predict_energy(gp_graph)

    ts_model = GraphPESWrapper(
        gp_model,
        device=device,
        dtype=torch.float32,
        compute_forces=True,
        compute_stress=False,
    )
    ts_output = ts_model(atoms_to_state([water_atoms], device, torch.float32))
    assert set(ts_output.keys()) == {"energy", "forces"}
    assert ts_output["energy"].shape == (1,)

    assert gp_energy.item() == pytest.approx(ts_output["energy"].item(), abs=1e-5)


def test_graphpes_periodic(device: torch.device):
    # test that the raw model and torch_sim wrapper give the same results
    # for a periodic, unbatched structure

    bulk_atoms = bulk("Al", "hcp", a=4.05)
    assert bulk_atoms.pbc.all()

    gp_model = TensorNet(cutoff=5.5)
    gp_graph = AtomicGraph.from_ase(bulk_atoms, cutoff=5.5)
    gp_forces = gp_model.predict_forces(gp_graph)

    ts_model = GraphPESWrapper(
        gp_model,
        device=device,
        dtype=torch.float32,
        compute_forces=True,
        compute_stress=True,
    )
    ts_output = ts_model(atoms_to_state([bulk_atoms], device, torch.float32))
    assert set(ts_output.keys()) == {"energy", "forces", "stress"}
    assert ts_output["energy"].shape == (1,)
    assert ts_output["forces"].shape == (len(bulk_atoms), 3)
    assert ts_output["stress"].shape == (1, 3, 3)

    torch.testing.assert_close(ts_output["forces"].to("cpu"), gp_forces)


def test_batching(device: torch.device):
    # test that the raw model and torch_sim wrapper give the same results
    # when batching is done via torch_sim's atoms_to_state function

    water = molecule("H2O")
    methane = molecule("CH4")
    systems = [water, methane]
    for s in systems:
        s.center(vacuum=10.0)

    gp_model = SchNet(cutoff=5.5)
    gp_graphs = [AtomicGraph.from_ase(s, cutoff=5.5) for s in systems]

    gp_energies = gp_model.predict_energy(to_batch(gp_graphs))

    ts_model = GraphPESWrapper(
        gp_model,
        device=device,
        dtype=torch.float32,
        compute_forces=True,
        compute_stress=True,
    )
    ts_output = ts_model(atoms_to_state(systems, device, torch.float32))

    assert set(ts_output.keys()) == {"energy", "forces", "stress"}
    assert ts_output["energy"].shape == (2,)
    assert ts_output["forces"].shape == (sum(len(s) for s in systems), 3)
    assert ts_output["stress"].shape == (2, 3, 3)

    assert gp_energies[0].item() == pytest.approx(ts_output["energy"][0].item(), abs=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_graphpes_dtype(device: torch.device, dtype: torch.dtype):
    water = molecule("H2O")

    model = SchNet()

    ts_wrapper = GraphPESWrapper(model, device=device, dtype=dtype, compute_stress=False)
    ts_output = ts_wrapper(atoms_to_state([water], device, dtype))
    assert ts_output["energy"].dtype == dtype
    assert ts_output["forces"].dtype == dtype


_nequip_model = ZEmbeddingNequIP()


@pytest.fixture
def ts_nequip_model(device: torch.device, dtype: torch.dtype):
    return GraphPESWrapper(
        _nequip_model,
        device=device,
        dtype=dtype,
        compute_stress=False,
    )


@pytest.fixture
def ase_nequip_calculator(device: torch.device, dtype: torch.dtype):
    return _nequip_model.to(device, dtype).ase_calculator(skin=0.0)


test_graphpes_nequip_consistency = make_model_calculator_consistency_test(
    test_name="graphpes-nequip",
    model_fixture_name="ts_nequip_model",
    calculator_fixture_name="ase_nequip_calculator",
    sim_state_names=consistency_test_simstate_fixtures,
)

test_graphpes_nequip_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="ts_nequip_model",
)


@pytest.fixture
def ts_mace_model(device: torch.device, dtype: torch.dtype):
    return GraphPESWrapper(
        mace_mp("medium-mpa-0"),
        device=device,
        dtype=dtype,
        compute_stress=False,
    )


@pytest.fixture
def ase_mace_calculator(device: torch.device, dtype: torch.dtype):
    return mace_mp("medium-mpa-0").to(device, dtype).ase_calculator(skin=0.0)


test_graphpes_mace_consistency = make_model_calculator_consistency_test(
    test_name="graphpes-mace",
    model_fixture_name="ts_mace_model",
    calculator_fixture_name="ase_mace_calculator",
    sim_state_names=consistency_test_simstate_fixtures,
    # graph-pes passes data directly to the underlying mace-torch model
    # from test_mace.py, it seems that these mace-torch models can be
    # surprisingly variable in the CI (these tests pass locally on
    # MacBooks with no need for high tolerances)
    # While investigating, I found that mace-torch model predictions are
    # mildly sensitive to the order of items in the neighbourlist - this
    # could be the cause of the discrepancies between the ASE calculator
    # and the TorchSim wrapper, both here and in test_mace.py
    rtol=6e-4,
    atol=1e-5,
)

test_graphpes_mace_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="ts_mace_model",
)


_lj_model = LennardJones(sigma=0.5)


@pytest.fixture
def ts_lj_model(device: torch.device, dtype: torch.dtype):
    return GraphPESWrapper(
        _lj_model,
        device=device,
        dtype=dtype,
        compute_stress=False,
    )


@pytest.fixture
def ase_lj_calculator(device: torch.device, dtype: torch.dtype):
    return _lj_model.to(device, dtype).ase_calculator(skin=0.0)


test_graphpes_lj_consistency = make_model_calculator_consistency_test(
    test_name="graphpes-lj",
    model_fixture_name="ts_lj_model",
    calculator_fixture_name="ase_lj_calculator",
    sim_state_names=consistency_test_simstate_fixtures,
)
