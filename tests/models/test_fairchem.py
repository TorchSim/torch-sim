import pytest
import torch

from tests.models.conftest import make_validate_model_outputs_test


try:
    from collections.abc import Callable

    from ase.build import bulk, fcc100, molecule
    from huggingface_hub.utils._auth import get_token

    import torch_sim as ts
    from torch_sim.models.fairchem import FairChemModel

except ImportError:
    pytest.skip("FairChem not installed", allow_module_level=True)


@pytest.fixture
def eqv2_uma_model_pbc(device: torch.device) -> FairChemModel:
    """UMA model for periodic boundary condition systems."""
    cpu = device.type == "cpu"
    return FairChemModel(model=None, model_name="uma-s-1", task_name="omat", cpu=cpu)


# Removed calculator consistency tests since we're using predictor interface only


@pytest.mark.skipif(
    get_token() is None, reason="Requires HuggingFace authentication for UMA model access"
)
@pytest.mark.parametrize("task_name", ["omat", "omol", "oc20"])
def test_task_initialization(task_name: str) -> None:
    """Test that different UMA task names work correctly."""
    model = FairChemModel(model=None, model_name="uma-s-1", task_name=task_name, cpu=True)
    assert model.task_name.value == task_name
    assert hasattr(model, "predictor")


@pytest.mark.skipif(
    get_token() is None, reason="Requires HuggingFace authentication for UMA model access"
)
@pytest.mark.parametrize(
    ("task_name", "systems_func"),
    [
        (
            "omat",
            lambda: [
                bulk("Si", "diamond", a=5.43),
                bulk("Al", "fcc", a=4.05),
                bulk("Fe", "bcc", a=2.87),
                bulk("Cu", "fcc", a=3.61),
            ],
        ),
        (
            "omol",
            lambda: [molecule("H2O"), molecule("CO2"), molecule("CH4"), molecule("NH3")],
        ),
    ],
)
def test_homogeneous_batching(
    task_name: str, systems_func: Callable, device: torch.device, dtype: torch.dtype
) -> None:
    """Test batching multiple systems with the same task."""
    systems = systems_func()

    # Add molecular properties for molecules
    if task_name == "omol":
        for mol in systems:
            mol.info.update({"charge": 0, "spin": 1})

    model = FairChemModel(
        model=None, model_name="uma-s-1", task_name=task_name, cpu=device.type == "cpu"
    )
    state = ts.io.atoms_to_state(systems, device=device, dtype=dtype)
    results = model(state)

    # Check batch dimensions
    assert results["energy"].shape == (4,)
    assert results["forces"].shape[0] == sum(len(s) for s in systems)
    assert results["forces"].shape[1] == 3

    # Check that different systems have different energies
    energies = results["energy"]
    unique_energies = torch.unique(energies, dim=0)
    assert len(unique_energies) > 1, "Different systems should have different energies"


@pytest.mark.skipif(
    get_token() is None, reason="Requires HuggingFace authentication for UMA model access"
)
def test_heterogeneous_tasks(device: torch.device, dtype: torch.dtype) -> None:
    """Test different task types work with appropriate systems."""
    # Test molecule, material, and catalysis systems separately
    test_cases = [
        ("omol", [molecule("H2O")]),
        ("omat", [bulk("Pt", cubic=True)]),
        ("oc20", [fcc100("Cu", (2, 2, 3), vacuum=8, periodic=True)]),
    ]

    for task_name, systems in test_cases:
        if task_name == "omol":
            systems[0].info.update({"charge": 0, "spin": 1})

        model = FairChemModel(
            model=None,
            model_name="uma-s-1",
            task_name=task_name,
            cpu=device.type == "cpu",
        )
        state = ts.io.atoms_to_state(systems, device=device, dtype=dtype)
        results = model(state)

        assert "energy" in results
        assert "forces" in results
        assert results["energy"].shape[0] == 1
        assert results["forces"].dim() == 2
        assert results["forces"].shape[1] == 3


@pytest.mark.skipif(
    get_token() is None, reason="Requires HuggingFace authentication for UMA model access"
)
@pytest.mark.parametrize(
    ("systems_func", "expected_count"),
    [
        (lambda: [bulk("Si", "diamond", a=5.43)], 1),  # Single system
        (
            lambda: [
                bulk("H", "bcc", a=2.0),
                bulk("Li", "bcc", a=3.0),
                bulk("Si", "diamond", a=5.43),
                bulk("Al", "fcc", a=4.05).repeat((2, 1, 1)),
            ],
            4,
        ),  # Mixed sizes
        (
            lambda: [
                bulk(element, "fcc", a=4.0)
                for element in ["Al", "Cu", "Ni", "Pd", "Pt"] * 3
            ],
            15,
        ),  # Large batch
    ],
)
def test_batch_size_variations(
    systems_func: Callable, expected_count: int, device: torch.device, dtype: torch.dtype
) -> None:
    """Test batching with different numbers and sizes of systems."""
    systems = systems_func()

    model = FairChemModel(
        model=None, model_name="uma-s-1", task_name="omat", cpu=device.type == "cpu"
    )
    state = ts.io.atoms_to_state(systems, device=device, dtype=dtype)
    results = model(state)

    assert results["energy"].shape == (expected_count,)
    assert results["forces"].shape[0] == sum(len(s) for s in systems)
    assert results["forces"].shape[1] == 3
    assert torch.isfinite(results["energy"]).all()
    assert torch.isfinite(results["forces"]).all()


@pytest.mark.skipif(
    get_token() is None, reason="Requires HuggingFace authentication for UMA model access"
)
@pytest.mark.parametrize("compute_stress", [True, False])
def test_stress_computation(
    *, compute_stress: bool, device: torch.device, dtype: torch.dtype
) -> None:
    """Test stress tensor computation."""
    systems = [bulk("Si", "diamond", a=5.43), bulk("Al", "fcc", a=4.05)]

    model = FairChemModel(
        model=None,
        model_name="uma-s-1",
        task_name="omat",
        cpu=device.type == "cpu",
        compute_stress=compute_stress,
    )
    state = ts.io.atoms_to_state(systems, device=device, dtype=dtype)
    results = model(state)

    if compute_stress:
        assert "stress" in results
        assert results["stress"].shape == (2, 3, 3)
        assert torch.isfinite(results["stress"]).all()
    else:
        assert "stress" not in results


@pytest.mark.skipif(
    get_token() is None, reason="Requires HuggingFace authentication for UMA model access"
)
def test_device_consistency(dtype: torch.dtype) -> None:
    """Test device consistency between model and data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = device.type == "cpu"

    model = FairChemModel(model=None, model_name="uma-s-1", task_name="omat", cpu=cpu)
    system = bulk("Si", "diamond", a=5.43)
    state = ts.io.atoms_to_state([system], device=device, dtype=dtype)

    results = model(state)
    assert results["energy"].device == device
    assert results["forces"].device == device


@pytest.mark.skipif(
    get_token() is None, reason="Requires HuggingFace authentication for UMA model access"
)
def test_empty_batch_error() -> None:
    """Test that empty batches raise appropriate errors."""
    model = FairChemModel(model=None, model_name="uma-s-1", task_name="omat", cpu=True)
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        model(ts.io.atoms_to_state([], device="cpu", dtype=torch.float32))


test_fairchem_uma_model_outputs = pytest.mark.skipif(
    get_token() is None,
    reason="Requires HuggingFace authentication for UMA model access",
)(make_validate_model_outputs_test(model_fixture_name="eqv2_uma_model_pbc"))
