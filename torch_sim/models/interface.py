"""Interface for all TorchSim models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import torch


class ModelInterface(ABC):
    """Interface for all TorchSim models."""

    @abstractmethod
    def __init__(
        self,
        model: str | Path | torch.nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        atomic_numbers: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> Self:
        """Abstract base class for all models that provides implementation guidelines.
        All models should inherit from this class.

        Within the __init__ method, the model must set self._device and self._dtype.

        If the model must be loaded from a file, it should accept a model argument.
        That argument can be a path to a model checkpoint or a torch.nn.Module.
        Implemented models should support both cases and raise an error if the wrong
        type of argument is provided.

        If the model requires atomic information, it should accept atomic_numbers and
        batch as arguments.

        The model must also set self._compute_stress. Model.compute_stress must be
        True for some integrators. It can be set to False if the model does not
        compute stresses, in which case those integrators will fail with a warning.
        """

    @property
    def device(self) -> torch.device:
        """The device of the model."""
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        raise NotImplementedError(
            "No device setter has been defined for this model"
            " so the device cannot be changed after initialization."
        )

    @property
    def dtype(self) -> torch.dtype:
        """The data type of the model."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype) -> None:
        raise NotImplementedError(
            "No dtype setter has been defined for this model"
            " so the dtype cannot be changed after initialization."
        )

    @property
    def compute_stress(self) -> bool:
        """Whether the model computes stresses."""
        return self._compute_stress

    @compute_stress.setter
    def compute_stress(self, compute_stress: bool) -> None:
        raise NotImplementedError(
            "No compute_stress setter has been defined for this model"
            " so compute_stress cannot be set after initialization."
        )

    @property
    def compute_force(self) -> bool:
        """Whether the model computes forces."""
        return self._compute_force

    @compute_force.setter
    def compute_force(self, compute_force: bool) -> None:
        raise NotImplementedError(
            "No compute_force setter has been defined for this model"
            " so compute_force cannot be set after initialization."
        )

    @abstractmethod
    def forward(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor | None,
        batch: torch.Tensor | None,
        atomic_numbers: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Abstract method that must be implemented by all models.

        All models must accept positions, cell, and batch as inputs.

        Positions must have shape (n_atoms, 3), where n_atoms is the total
        number of atoms summed over all batches. Batch must have shape
        (n_atoms,) and should be consecutive integers.

        If the model requires atomic information and that information
        can be updated on-the-fly, the model should accept atomic_numbers and
        batch as inputs.

        The output must be a dictionary with at least "energy" and "forces".
        The "stress" must also be computed if self.compute_stress is True.

        The shape of all per-atom properties must be

        # TODO: specify torch.Tensor types and shapes
        """


def validate_model_outputs(
    model: ModelInterface,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Validate the outputs of a model."""
    from ase.build import bulk

    from torch_sim.runners import atoms_to_state

    assert model.dtype is not None
    assert model.device is not None
    assert model.compute_stress is not None
    assert model.compute_force is not None

    try:
        if not model.compute_stress:
            model.compute_stress = True
        stress_computed = True
    except NotImplementedError:
        stress_computed = False

    try:
        if not model.compute_force:
            model.compute_force = True
        force_computed = True
    except NotImplementedError:
        force_computed = False

    si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    fe_atoms = bulk("Fe", "fcc", a=5.26, cubic=True).repeat([3, 1, 1])

    base_state = atoms_to_state([si_atoms, fe_atoms], device, dtype)

    og_positions = base_state.positions.clone()
    og_cell = base_state.cell.clone()
    og_batch = base_state.batch.clone()
    og_atomic_numbers = base_state.atomic_numbers.clone()

    model_output = model.forward(
        positions=base_state.positions,
        cell=base_state.cell,
        batch=base_state.batch,
        atomic_numbers=base_state.atomic_numbers,
    )

    # assert model did not mutate the input
    assert torch.allclose(og_positions, base_state.positions)
    assert torch.allclose(og_cell, base_state.cell)
    assert torch.allclose(og_batch, base_state.batch)
    assert torch.allclose(og_atomic_numbers, base_state.atomic_numbers)

    # assert model output has the correct keys
    assert "energy" in model_output
    assert "forces" in model_output if force_computed else True
    assert "stress" in model_output if stress_computed else True

    # assert model output shapes are correct
    assert model_output["energy"].shape == (2,)
    assert model_output["forces"].shape == (20, 3) if force_computed else True
    assert model_output["stress"].shape == (2, 3, 3) if stress_computed else True

    si_state = atoms_to_state([si_atoms], device, dtype)
    ar_state = atoms_to_state([fe_atoms], device, dtype)

    si_model_output = model.forward(
        positions=si_state.positions,
        cell=si_state.cell,
        batch=si_state.batch,
        atomic_numbers=si_state.atomic_numbers,
    )
    assert torch.allclose(
        si_model_output["energy"], model_output["energy"][0], atol=10e-3
    )
    assert torch.allclose(
        si_model_output["forces"],
        model_output["forces"][: si_state.n_atoms],
        atol=10e-3,
    )
    # assert torch.allclose(
    #     si_model_output["stress"],
    #     model_output["stress"][0],
    #     atol=10e-3,
    # )

    ar_model_output = model.forward(
        positions=ar_state.positions,
        cell=ar_state.cell,
        batch=ar_state.batch,
        atomic_numbers=ar_state.atomic_numbers,
    )
    si_model_output = model.forward(
        positions=si_state.positions,
        cell=si_state.cell,
        batch=si_state.batch,
        atomic_numbers=si_state.atomic_numbers,
    )

    # print("ar single batch energy", ar_model_output["energy"])
    # print("si single batch energy", si_model_output["energy"])

    # print("si multi batch energy", model_output["energy"][0])
    # print("ar multi batch energy", model_output["energy"][1])

    assert torch.allclose(
        ar_model_output["energy"], model_output["energy"][1], atol=10e-2
    )
    assert torch.allclose(
        ar_model_output["forces"],
        model_output["forces"][si_state.n_atoms :],
        atol=10e-2,
    )
    # assert torch.allclose(
    #     arr_model_output["stress"],
    #     model_output["stress"][1],
    #     atol=10e-3,
    # )
