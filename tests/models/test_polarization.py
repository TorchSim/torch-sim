"""Tests for the polarization electric-field correction model."""

import pytest
import torch

import torch_sim as ts
from tests.conftest import DEVICE, DTYPE
from torch_sim.models.interface import ModelInterface, SerialSumModel
from torch_sim.models.polarization import UniformPolarizationModel


class DummyPolarResponseModel(ModelInterface):
    def __init__(
        self,
        *,
        include_born_effective_charges: bool = True,
        include_polarizability: bool = True,
        include_total_polarization: bool = True,
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        self.include_born_effective_charges = include_born_effective_charges
        self.include_polarizability = include_polarizability
        self.include_total_polarization = include_total_polarization
        self._device = device
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = True

    def forward(self, state: ts.SimState, **kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        energy = torch.arange(
            1, state.n_systems + 1, device=state.device, dtype=state.dtype
        )
        forces = (
            torch.arange(state.n_atoms * 3, device=state.device, dtype=state.dtype)
            .reshape(state.n_atoms, 3)
            .div(10.0)
        )
        stress = (
            torch.arange(state.n_systems * 9, device=state.device, dtype=state.dtype)
            .reshape(state.n_systems, 3, 3)
            .div(100.0)
        )
        polarization = (
            torch.arange(state.n_systems * 3, device=state.device, dtype=state.dtype)
            .reshape(state.n_systems, 3)
            .add(0.5)
        )
        output: dict[str, torch.Tensor] = {
            "energy": energy,
            "forces": forces,
            "stress": stress,
        }
        if self.include_total_polarization:
            output["total_polarization"] = polarization
        if self.include_polarizability:
            diag = torch.tensor([1.0, 2.0, 3.0], device=state.device, dtype=state.dtype)
            output["polarizability"] = torch.diag_embed(diag.repeat(state.n_systems, 1))
        if self.include_born_effective_charges:
            born_effective_charges = torch.zeros(
                state.n_atoms, 3, 3, device=state.device, dtype=state.dtype
            )
            born_effective_charges[:, 0, 0] = 1.0
            born_effective_charges[:, 1, 1] = 2.0
            born_effective_charges[:, 2, 2] = 3.0
            output["born_effective_charges"] = born_effective_charges
        return output


def test_polarization_model_requires_external_e_field(
    si_double_sim_state: ts.SimState,
) -> None:
    base_model = DummyPolarResponseModel()
    combined_model = SerialSumModel(
        base_model,
        UniformPolarizationModel(device=DEVICE, dtype=DTYPE),
    )

    with pytest.raises(ValueError, match="external_E_field"):
        combined_model(si_double_sim_state)


def test_polarization_model_applies_linear_response_corrections(
    si_double_sim_state: ts.SimState,
) -> None:
    base_model = DummyPolarResponseModel()
    combined_model = SerialSumModel(
        base_model,
        UniformPolarizationModel(device=DEVICE, dtype=DTYPE),
    )
    field = torch.tensor(
        [[0.2, -0.1, 0.05], [-0.3, 0.4, 0.1]],
        device=DEVICE,
        dtype=DTYPE,
    )
    state = ts.SimState.from_state(si_double_sim_state, external_E_field=field)

    base_output = base_model(state)
    combined_output = combined_model(state)
    expected_polarization = base_output["total_polarization"] + torch.einsum(
        "sij,sj->si", base_output["polarizability"], field
    )
    expected_energy = base_output["energy"] - torch.einsum(
        "si,si->s", field, base_output["total_polarization"]
    )
    expected_energy = expected_energy - 0.5 * torch.einsum(
        "si,sij,sj->s", field, base_output["polarizability"], field
    )
    expected_forces = base_output["forces"] + torch.einsum(
        "imn,im->in",
        base_output["born_effective_charges"],
        field[state.system_idx],
    )

    torch.testing.assert_close(combined_output["energy"], expected_energy)
    torch.testing.assert_close(combined_output["forces"], expected_forces)
    torch.testing.assert_close(
        combined_output["total_polarization"], expected_polarization
    )
    torch.testing.assert_close(combined_output["stress"], base_output["stress"])


def test_polarization_model_returns_additive_total_polarization_delta(
    si_double_sim_state: ts.SimState,
) -> None:
    base_model = DummyPolarResponseModel()
    combined_model = SerialSumModel(
        base_model,
        UniformPolarizationModel(device=DEVICE, dtype=DTYPE),
    )
    field = torch.tensor([[0.1, 0.0, 0.0], [0.0, -0.2, 0.3]], device=DEVICE, dtype=DTYPE)
    state = ts.SimState.from_state(si_double_sim_state, external_E_field=field)

    base_output = base_model(state)
    combined_output = combined_model(state)
    expected_total_polarization = base_output["total_polarization"] + torch.einsum(
        "sij,sj->si", base_output["polarizability"], field
    )

    torch.testing.assert_close(
        combined_output["total_polarization"], expected_total_polarization
    )
    serialized_state = state.clone()
    serialized_state.store_model_extras(base_output)
    correction_output = UniformPolarizationModel(device=DEVICE, dtype=DTYPE)(
        serialized_state
    )
    torch.testing.assert_close(
        correction_output["total_polarization"],
        expected_total_polarization,
    )


def test_polarization_model_requires_born_effective_charges_for_force_correction(
    si_double_sim_state: ts.SimState,
) -> None:
    base_model = DummyPolarResponseModel(include_born_effective_charges=False)
    combined_model = SerialSumModel(
        base_model,
        UniformPolarizationModel(device=DEVICE, dtype=DTYPE),
    )
    state = ts.SimState.from_state(
        si_double_sim_state,
        external_E_field=torch.ones(
            si_double_sim_state.n_systems, 3, device=DEVICE, dtype=DTYPE
        ),
    )

    with pytest.raises(ValueError, match="born_effective_charges"):
        combined_model(state)


def test_polarization_model_requires_total_polarization(
    si_double_sim_state: ts.SimState,
) -> None:
    base_model = DummyPolarResponseModel(include_total_polarization=False)
    combined_model = SerialSumModel(
        base_model,
        UniformPolarizationModel(device=DEVICE, dtype=DTYPE),
    )
    state = ts.SimState.from_state(
        si_double_sim_state,
        external_E_field=torch.ones(
            si_double_sim_state.n_systems, 3, device=DEVICE, dtype=DTYPE
        ),
    )

    with pytest.raises(ValueError, match="total_polarization"):
        combined_model(state)


def test_polarization_model_rejects_non_uniform_field_shape(
    si_double_sim_state: ts.SimState,
) -> None:
    state = ts.SimState.from_state(
        si_double_sim_state,
        external_E_field=torch.zeros(
            si_double_sim_state.n_systems, 3, device=DEVICE, dtype=DTYPE
        ),
    )
    state._system_extras["external_E_field"] = torch.zeros(  # noqa: SLF001
        state.n_atoms, 3, device=DEVICE, dtype=DTYPE
    )
    model = UniformPolarizationModel(device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="shape \\(n_systems, 3\\)"):
        model(state)
