"""Tests for the SumModel composite model."""

import pytest
import torch

import torch_sim as ts
from tests.conftest import DEVICE, DTYPE
from torch_sim.models.interface import SumModel, validate_model_outputs
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
def lj_model_b() -> LennardJonesModel:
    return LennardJonesModel(
        sigma=2.0,
        epsilon=0.005,
        cutoff=5.0,
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


def test_sum_model_properties(sum_model: SumModel) -> None:
    assert sum_model.device == DEVICE
    assert sum_model.dtype == DTYPE
    assert sum_model.compute_stress is True
    assert sum_model.compute_forces is True


def test_sum_model_energy_summation(
    lj_model_a: LennardJonesModel,
    morse_model: MorseModel,
    sum_model: SumModel,
    si_sim_state: ts.SimState,
) -> None:
    lj_out = lj_model_a(si_sim_state)
    morse_out = morse_model(si_sim_state)
    sum_out = sum_model(si_sim_state)
    expected_energy = lj_out["energy"] + morse_out["energy"]
    torch.testing.assert_close(sum_out["energy"], expected_energy)


def test_sum_model_forces_summation(
    lj_model_a: LennardJonesModel,
    morse_model: MorseModel,
    sum_model: SumModel,
    si_sim_state: ts.SimState,
) -> None:
    lj_out = lj_model_a(si_sim_state)
    morse_out = morse_model(si_sim_state)
    sum_out = sum_model(si_sim_state)
    expected_forces = lj_out["forces"] + morse_out["forces"]
    torch.testing.assert_close(sum_out["forces"], expected_forces)


def test_sum_model_stress_summation(
    lj_model_a: LennardJonesModel,
    morse_model: MorseModel,
    sum_model: SumModel,
    si_sim_state: ts.SimState,
) -> None:
    lj_out = lj_model_a(si_sim_state)
    morse_out = morse_model(si_sim_state)
    sum_out = sum_model(si_sim_state)
    expected_stress = lj_out["stress"] + morse_out["stress"]
    torch.testing.assert_close(sum_out["stress"], expected_stress)


def test_sum_model_batched(
    lj_model_a: LennardJonesModel,
    morse_model: MorseModel,
    sum_model: SumModel,
    si_double_sim_state: ts.SimState,
) -> None:
    lj_out = lj_model_a(si_double_sim_state)
    morse_out = morse_model(si_double_sim_state)
    sum_out = sum_model(si_double_sim_state)
    torch.testing.assert_close(sum_out["energy"], lj_out["energy"] + morse_out["energy"])
    torch.testing.assert_close(sum_out["forces"], lj_out["forces"] + morse_out["forces"])
    torch.testing.assert_close(sum_out["stress"], lj_out["stress"] + morse_out["stress"])


def test_sum_model_three_models(
    lj_model_a: LennardJonesModel,
    lj_model_b: LennardJonesModel,
    morse_model: MorseModel,
    si_sim_state: ts.SimState,
) -> None:
    triple = SumModel(lj_model_a, lj_model_b, morse_model)
    a_out = lj_model_a(si_sim_state)
    b_out = lj_model_b(si_sim_state)
    c_out = morse_model(si_sim_state)
    sum_out = triple(si_sim_state)
    torch.testing.assert_close(
        sum_out["energy"], a_out["energy"] + b_out["energy"] + c_out["energy"]
    )


def test_sum_model_compute_stress_setter(
    lj_model_a: LennardJonesModel, morse_model: MorseModel
) -> None:
    sm = SumModel(lj_model_a, morse_model)
    assert sm.compute_stress is True
    sm.compute_stress = False
    assert sm.compute_stress is False


def test_sum_model_compute_forces_setter(
    lj_model_a: LennardJonesModel, morse_model: MorseModel
) -> None:
    sm = SumModel(lj_model_a, morse_model)
    sm.compute_forces = False
    assert sm.compute_forces is False


def test_sum_model_memory_scales_with(
    lj_model_a: LennardJonesModel, morse_model: MorseModel
) -> None:
    sm = SumModel(lj_model_a, morse_model)
    assert sm.memory_scales_with == "n_atoms_x_density"


def test_sum_model_force_conservation(
    sum_model: SumModel, si_double_sim_state: ts.SimState
) -> None:
    results = sum_model(si_double_sim_state)
    for sys_idx in range(si_double_sim_state.n_systems):
        mask = si_double_sim_state.system_idx == sys_idx
        assert torch.allclose(
            results["forces"][mask].sum(dim=0),
            torch.zeros(3, dtype=DTYPE),
            atol=1e-10,
        )


def test_sum_model_validate_outputs(sum_model: SumModel) -> None:
    validate_model_outputs(sum_model, DEVICE, DTYPE, check_detached=True)


def test_sum_model_retain_graph(
    lj_model_a: LennardJonesModel, morse_model: MorseModel
) -> None:
    sm = SumModel(lj_model_a, morse_model)
    assert sm.retain_graph is False
    sm.retain_graph = True
    assert lj_model_a.retain_graph is True
    assert morse_model.retain_graph is True
    assert sm.retain_graph is True
