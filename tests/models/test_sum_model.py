"""Tests for the SumModel composite model."""

import pytest
import torch

import torch_sim as ts
from tests.conftest import DEVICE, DTYPE
from tests.models.conftest import make_validate_model_outputs_test
from torch_sim.models.interface import SumModel
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.models.morse import MorseModel


@pytest.fixture
def lj_model_a() -> LennardJonesModel:
    return LennardJonesModel(
        sigma=3.405,
        epsilon=0.0104,
        cutoff=2.5 * 3.405,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )


@pytest.fixture
def morse_model() -> MorseModel:
    return MorseModel(
        sigma=2.55,
        epsilon=0.436,
        alpha=1.359,
        cutoff=6.0,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )


@pytest.fixture
def sum_model(lj_model_a: LennardJonesModel, morse_model: MorseModel) -> SumModel:
    return SumModel(lj_model_a, morse_model)


test_sum_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="sum_model", device=DEVICE, dtype=DTYPE
)


def test_sum_model_requires_two_models(lj_model_a: LennardJonesModel) -> None:
    with pytest.raises(ValueError, match="at least two"):
        SumModel(lj_model_a)


def test_sum_model_device_mismatch() -> None:
    m1 = LennardJonesModel(sigma=1.0, epsilon=1.0, cutoff=2.5, device=torch.device("cpu"))
    m2 = LennardJonesModel(sigma=1.0, epsilon=1.0, cutoff=2.5, device=torch.device("cpu"))
    object.__setattr__(m2, "_device", torch.device("meta"))
    with pytest.raises(ValueError, match="Device mismatch"):
        SumModel(m1, m2)


def test_sum_model_dtype_mismatch() -> None:
    m1 = LennardJonesModel(sigma=1.0, epsilon=1.0, cutoff=2.5, dtype=torch.float64)
    m2 = LennardJonesModel(sigma=1.0, epsilon=1.0, cutoff=2.5, dtype=torch.float32)
    with pytest.raises(ValueError, match="Dtype mismatch"):
        SumModel(m1, m2)


def test_sum_model_additivity(
    lj_model_a: LennardJonesModel,
    morse_model: MorseModel,
    sum_model: SumModel,
    si_sim_state: ts.SimState,
) -> None:
    lj_out = lj_model_a(si_sim_state)
    morse_out = morse_model(si_sim_state)
    sum_out = sum_model(si_sim_state)
    torch.testing.assert_close(sum_out["energy"], lj_out["energy"] + morse_out["energy"])
    torch.testing.assert_close(sum_out["forces"], lj_out["forces"] + morse_out["forces"])
    torch.testing.assert_close(sum_out["stress"], lj_out["stress"] + morse_out["stress"])


def test_sum_model_setters(
    lj_model_a: LennardJonesModel, morse_model: MorseModel
) -> None:
    sm = SumModel(lj_model_a, morse_model)
    assert sm.compute_stress is True
    sm.compute_stress = False
    assert sm.compute_stress is False
    sm.compute_forces = False
    assert sm.compute_forces is False


def test_sum_model_retain_graph(
    lj_model_a: LennardJonesModel, morse_model: MorseModel
) -> None:
    sm = SumModel(lj_model_a, morse_model)
    assert sm.retain_graph is False
    sm.retain_graph = True
    assert lj_model_a.retain_graph is True
    assert morse_model.retain_graph is True
    assert sm.retain_graph is True
