"""Model Interface: Core interfaces for all simulation models in torchsim.

This module defines the abstract base class that all torchsim models must implement.
It establishes a common API for interacting with different force and energy models,
ensuring consistent behavior regardless of the underlying implementation. The module
also provides validation utilities to verify model conformance to the interface.

Examples:
    ```python
    # Creating a custom model that implements the interface
    class MyModel(ModelInterface):
        def __init__(self, device=None, dtype=torch.float64):
            self._device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._dtype = dtype
            self._compute_stress = True
            self._compute_force = True

        def forward(self, positions, cell, batch, atomic_numbers=None, **kwargs):
            # Implementation that returns energy, forces, and stress
            return {"energy": energy, "forces": forces, "stress": stress}
    ```

Notes:
    Models must explicitly declare support for stress computation through the
    compute_stress property, as some integrators require stress calculations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Self

import torch


class ModelInterface(ABC):
    """Abstract base class for all simulation models in torchsim.

    This interface provides a common structure for all energy and force models,
    ensuring they implement the required methods and properties. It defines how
    models should process atomic positions and system information to compute energies,
    forces, and stresses.

    Attributes:
        device (torch.device): Device where the model runs computations.
        dtype (torch.dtype): Data type used for tensor calculations.
        compute_stress (bool): Whether the model calculates stress tensors.
        compute_force (bool): Whether the model calculates atomic forces.

    Examples:
        ```python
        # Using a model that implements ModelInterface
        model = LennardJonesModel(device=torch.device("cuda"))

        # Forward pass with a simulation state
        output = model(sim_state)

        # Access computed properties
        energy = output["energy"]  # Shape: [n_batches]
        forces = output["forces"]  # Shape: [n_atoms, 3]
        stress = output["stress"]  # Shape: [n_batches, 3, 3]
        ```
    """

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
        """Initialize a model implementation.

        Implementations must set device, dtype and compute capability flags
        to indicate what operations the model supports. Models may optionally
        load parameters from a file or existing module.

        Args:
            model (str | Path | torch.nn.Module | None): Model specification, which
                can be:
                - Path to a model checkpoint
                - Pre-configured torch.nn.Module
                - None for default initialization
                Defaults to None.
            device (torch.device | None): Device where the model will run. If None,
                a default device will be selected. Defaults to None.
            dtype (torch.dtype): Data type for model calculations. Defaults to
                torch.float64.
            atomic_numbers (torch.Tensor | None): Atomic numbers for atom type information
                with shape [n_atoms]. Defaults to None.
            batch (torch.Tensor | None): Batch indices for atoms with shape [n_atoms].
                Defaults to None.
            **kwargs: Additional model-specific parameters.

        Notes:
            All implementing classes must set self._device, self._dtype,
            self._compute_stress and self._compute_force in their __init__ method.
        """

    @property
    def device(self) -> torch.device:
        """The device of the model.

        Returns:
            The device of the model
        """
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        raise NotImplementedError(
            "No device setter has been defined for this model"
            " so the device cannot be changed after initialization."
        )

    @property
    def dtype(self) -> torch.dtype:
        """The data type of the model.

        Returns:
            The data type of the model
        """
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype) -> None:
        raise NotImplementedError(
            "No dtype setter has been defined for this model"
            " so the dtype cannot be changed after initialization."
        )

    @property
    def compute_stress(self) -> bool:
        """Whether the model computes stresses.

        Returns:
            Whether the model computes stresses
        """
        return self._compute_stress

    @compute_stress.setter
    def compute_stress(self, compute_stress: bool) -> None:
        raise NotImplementedError(
            "No compute_stress setter has been defined for this model"
            " so compute_stress cannot be set after initialization."
        )

    @property
    def compute_force(self) -> bool:
        """Whether the model computes forces.

        Returns:
            Whether the model computes forces
        """
        return self._compute_force

    @compute_force.setter
    def compute_force(self, compute_force: bool) -> None:
        raise NotImplementedError(
            "No compute_force setter has been defined for this model"
            " so compute_force cannot be set after initialization."
        )

    @property
    def memory_scales_with(self) -> Literal["n_atoms", "n_atoms_x_density"]:
        """The metric that the model scales with. 

        Models with radial neighbor cutoffs scale with "n_atoms_x_density",
        while models with a fixed number of neighbors scale with "n_atoms".
        Default is "n_atoms_x_density" because most models are radial cutoff based.

        Returns:
            The metric that the model scales with
        """
        return getattr(self, "_memory_scales_with", "n_atoms_x_density")

    @abstractmethod
    def forward(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor | None,
        batch: torch.Tensor | None,
        atomic_numbers: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Calculate energies, forces, and stresses for a atomistic system.

        This is the main computational method that all model implementations must provide.
        It takes atomic positions and system information as input and returns a dictionary
        containing computed physical properties.

        Args:
            positions (torch.Tensor): Atomic positions with shape [n_atoms, 3],
                where n_atoms is the total number of atoms across all batches.
            cell (torch.Tensor | None): Unit cell vectors with shape [n_batches, 3, 3]
                or None for non-periodic systems.
            batch (torch.Tensor | None): Batch indices for each atom with shape
                [n_atoms], containing consecutive integers identifying which batch
                each atom belongs to.
            atomic_numbers (torch.Tensor | None): Atomic numbers with shape [n_atoms]
                specifying the element type of each atom. Defaults to None.
            **kwargs: Additional model-specific parameters.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing computed properties:
                - "energy": Potential energy with shape [n_batches]
                - "forces": Atomic forces with shape [n_atoms, 3]
                - "stress": Stress tensor with shape [n_batches, 3, 3] (if
                    compute_stress=True)
                - May include additional model-specific outputs

        Examples:
            ```python
            # Compute energies and forces with a model
            output = model.forward(
                positions=state.positions,
                cell=state.cell,
                batch=state.batch,
                atomic_numbers=state.atomic_numbers,
            )

            energy = output["energy"]
            forces = output["forces"]
            stress = output.get("stress", None)
            ```

        Notes:
            Implementations must not modify the input tensors. The output forces should
            follow the convention that negative gradient of energy corresponds to forces.
        """


def validate_model_outputs(
    model: ModelInterface,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Validate the outputs of a model implementation against the interface requirements.

    Runs a series of tests to ensure a model implementation correctly follows the
    ModelInterface contract. The tests include creating sample systems, running
    forward passes, and verifying output shapes and consistency.

    Args:
        model (ModelInterface): Model implementation to validate.
        device (torch.device): Device to run the validation tests on.
        dtype (torch.dtype): Data type to use for validation tensors.

    Raises:
        AssertionError: If the model doesn't conform to the required interface,
            including issues with output shapes, types, or behavior consistency.

    Examples:
        ```python
        # Create a new model implementation
        model = MyCustomModel(device=torch.device("cuda"))

        # Validate that it correctly implements the interface
        validate_model_outputs(model, device=torch.device("cuda"), dtype=torch.float64)
        ```

    Notes:
        This validator creates small test systems (silicon and iron) for validation.
        It tests both single and multi-batch processing capabilities.
    """
    from ase.build import bulk

    from torch_sim.io import atoms_to_state

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

    sim_state = atoms_to_state([si_atoms, fe_atoms], device, dtype)

    og_positions = sim_state.positions.clone()
    og_cell = sim_state.cell.clone()
    og_batch = sim_state.batch.clone()
    og_atomic_numbers = sim_state.atomic_numbers.clone()

    model_output = model.forward(sim_state)

    # assert model did not mutate the input
    assert torch.allclose(og_positions, sim_state.positions)
    assert torch.allclose(og_cell, sim_state.cell)
    assert torch.allclose(og_batch, sim_state.batch)
    assert torch.allclose(og_atomic_numbers, sim_state.atomic_numbers)

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

    si_model_output = model.forward(si_state)
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

    ar_model_output = model.forward(ar_state)
    si_model_output = model.forward(si_state)

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
