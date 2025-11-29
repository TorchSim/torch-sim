"""Core interfaces for all models in TorchSim.

This module defines the abstract base class that all TorchSim models must implement.
It establishes a common API for interacting with different force and energy models,
ensuring consistent behavior regardless of the underlying implementation. The module
also provides validation utilities to verify model conformance to the interface.

Example::

    # Creating a custom model that implements the interface
    class MyModel(ModelInterface):
        def __init__(self, device=None, dtype=torch.float64):
            self._device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._dtype = dtype
            self._compute_stress = True
            self._compute_forces = True

        def forward(self, positions, cell, batch, atomic_numbers=None, **kwargs):
            # Implementation that returns energy, forces, and stress
            return {"energy": energy, "forces": forces, "stress": stress}

Notes:
    Models must explicitly declare support for stress computation through the
    compute_stress property, as some integrators require stress calculations.
"""

from abc import ABC, abstractmethod
from typing import TypedDict
from typing_extensions import deprecated

import torch

import torch_sim as ts
from torch_sim.state import SimState
from torch_sim.typing import MemoryScaling, StateDict


class ModelInterfaceOutput(TypedDict):
    """The expected output of a model forward pass implementation."""

    atom_attributes: dict[str, torch.Tensor]
    system_attributes: dict[str, torch.Tensor]
    global_attributes: dict[str, torch.Tensor]

    # deprecated attributes. People who've written their own model interfaces should move
    # away from this and write their results to atom_attributes, system_attributes, and
    # global_attributes.
    energy: torch.Tensor | None
    forces: torch.Tensor | None
    stress: torch.Tensor | None


class ModelInterface(torch.nn.Module, ABC):
    """Abstract base class for all simulation models in TorchSim.

    This interface provides a common structure for all energy and force models,
    ensuring they implement the required methods and properties. It defines how
    models should process atomic positions and system information to compute
    system-wide attributes like energies/stresses, or atom-wise attributes like forces.

    Attributes:
        device (torch.device): Device where the model runs computations.
        dtype (torch.dtype): Data type used for tensor calculations.
        compute_stress (bool): Whether the model calculates stress tensors.
        compute_forces (bool): Whether the model calculates atomic forces.
        memory_scales_with (MemoryScaling): The metric
            that the model scales with. "n_atoms" uses only atom count and is suitable
            for models that have a fixed number of neighbors. "n_atoms_x_density" uses
            atom count multiplied by number density and is better for models with
            radial cutoffs. Defaults to "n_atoms_x_density".

    Examples:
        ```py
        # Using a model that implements ModelInterface
        model = LennardJonesModel(device=torch.device("cuda"))

        # Forward pass with a simulation state
        output = model(sim_state)

        # Access computed properties
        energy = output["energy"]  # Shape: [n_systems]
        forces = output["forces"]  # Shape: [n_atoms, 3]
        stress = output["stress"]  # Shape: [n_systems, 3, 3]
        ```
    """

    _device: torch.device
    _dtype: torch.dtype
    _compute_stress: bool
    _compute_forces: bool

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
    def compute_forces(self) -> bool:
        """Whether the model computes forces."""
        return self._compute_forces

    @compute_forces.setter
    def compute_forces(self, compute_forces: bool) -> None:
        raise NotImplementedError(
            "No compute_forces setter has been defined for this model"
            " so compute_forces cannot be set after initialization."
        )

    @property
    def memory_scales_with(self) -> MemoryScaling:
        """The metric that the model scales with.

        Models with radial neighbor cutoffs scale with "n_atoms_x_density",
        while models with a fixed number of neighbors scale with "n_atoms".
        Default is "n_atoms_x_density" because most models are radial cutoff based.
        """
        return getattr(self, "_memory_scales_with", "n_atoms_x_density")

    @abstractmethod
    def forward(self, state: SimState | StateDict, **kwargs) -> ModelInterfaceOutput:
        """Calculate energies, forces, and stresses for a atomistic system.

        This is the main computational method that all model implementations must provide.
        It takes atomic positions and system information as input and returns a dictionary
        containing computed physical properties.

        Args:
            state (SimState | StateDict): Simulation state or state dictionary. The state
                dictionary is dependent on the model but typically must contain the
                following keys:
                - "positions": Atomic positions with shape [n_atoms, 3]
                - "cell": Unit cell vectors with shape [n_systems, 3, 3]
                - "system_idx": System indices for each atom with shape [n_atoms]
                - "atomic_numbers": Atomic numbers with shape [n_atoms] (optional)
            **kwargs: Additional model-specific parameters.

        Returns:
            ModelInterfaceOutput: Computed properties:
                - "atom_attributes": Dictionary of atom-wise attributes
                - "system_attributes": Dictionary of system-wide attributes
                - "global_attributes": Dictionary of global attributes

        Examples:
            ```py
            # Compute energies and forces with a model
            output = model.forward(state)

            energy = output["system_attributes"]["energy"]
            forces = output["atom_attributes"]["forces"]
            stress = output["system_attributes"].get("stress")
            ```
        """


# TODO: we should put this logic inside __init_subclass__ of Modelinterface to
# automatically validate the model outputs when the model is subclassed.
def validate_model_outputs(  # noqa: C901, PLR0915
    model: ModelInterface,
    device: torch.device,
    dtype: torch.dtype,
    expected_output_atom_attributes: set[str],
    expected_output_system_attributes: set[str],
    expected_output_global_attributes: set[str],
) -> None:
    """Validate the outputs of a model implementation against the interface requirements.

    Runs a series of tests to ensure a model implementation correctly follows the
    ModelInterface contract. The tests include creating sample systems, running
    forward passes, and verifying output shapes and consistency.

    Args:
        model (ModelInterface): Model implementation to validate.
        device (torch.device): Device to run the validation tests on.
        dtype (torch.dtype): Data type to use for validation tensors.
        expected_output_attributes (set[str]): The attributes that the model is expected
            to return.
    Raises:
        AssertionError: If the model doesn't conform to the required interface,
            including issues with output shapes, types, or behavior consistency.

    Example::

        # Create a new model implementation
        model = MyCustomModel(device=torch.device("cuda"))

        # Validate that it correctly implements the interface
        validate_model_outputs(model, device=torch.device("cuda"), dtype=torch.float64)

    Notes:
        This validator creates small test systems (silicon and iron) for validation.
        It tests both single and multi-batch processing capabilities.
    """
    from ase.build import bulk

    for attr in ("dtype", "device", "compute_stress", "compute_forces"):
        if not hasattr(model, attr):
            raise ValueError(f"model.{attr} is not set")

    try:
        if not model.compute_stress:
            model.compute_stress = True  # type: ignore[unresolved-attribute]
        stress_computed = True
    except NotImplementedError:
        stress_computed = False

    try:
        if not model.compute_forces:
            model.compute_forces = True  # type: ignore[unresolved-attribute]
        force_computed = True
    except NotImplementedError:
        force_computed = False

    si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    fe_atoms = bulk("Fe", "fcc", a=5.26, cubic=True).repeat([3, 1, 1])

    sim_state = ts.io.atoms_to_state([si_atoms, fe_atoms], device, dtype)

    og_positions = sim_state.positions.clone()
    og_cell = sim_state.cell.clone()
    og_system_idx = sim_state.system_idx.clone()
    og_atomic_nums = sim_state.atomic_numbers.clone()

    model_output = model.forward(sim_state)

    # assert model did not mutate the input
    if not torch.allclose(og_positions, sim_state.positions):
        raise ValueError(f"{og_positions=} != {sim_state.positions=}")
    if not torch.allclose(og_cell, sim_state.cell):
        raise ValueError(f"{og_cell=} != {sim_state.cell=}")
    if not torch.allclose(og_system_idx, sim_state.system_idx):
        raise ValueError(f"{og_system_idx=} != {sim_state.system_idx=}")
    if not torch.allclose(og_atomic_nums, sim_state.atomic_numbers):
        raise ValueError(f"{og_atomic_nums=} != {sim_state.atomic_numbers=}")

    # assert model output has the correct keys
    for attr in expected_output_atom_attributes:
        if attr not in model_output["atom_attributes"]:
            raise ValueError(f"{attr} not in model output")
    for attr in expected_output_system_attributes:
        if attr not in model_output["system_attributes"]:
            raise ValueError(f"{attr} not in model output")
    for attr in expected_output_global_attributes:
        if attr not in model_output["global_attributes"]:
            raise ValueError(f"{attr} not in model output")

    si_state = ts.io.atoms_to_state([si_atoms], device, dtype)
    fe_state = ts.io.atoms_to_state([fe_atoms], device, dtype)

    si_model_output = model.forward(si_state)
    fe_model_output = model.forward(fe_state)

    for attr in expected_output_atom_attributes:
        if attr in model_output["atom_attributes"]:
            si_attr = si_model_output["atom_attributes"][attr]
            batched_attr = model_output["atom_attributes"][attr]
            expected_attr = batched_attr[: si_state.n_atoms]
            if not torch.allclose(si_attr, expected_attr, atol=10e-3):
                raise ValueError(f"{attr}: {si_attr=} != {expected_attr=}")

            fe_attr = fe_model_output["atom_attributes"][attr]
            expected_fe_attr = batched_attr[si_state.n_atoms :]
            if not torch.allclose(fe_attr, expected_fe_attr, atol=10e-2):
                raise ValueError(f"{attr}: {fe_attr=} != {expected_fe_attr=}")

    for attr in expected_output_system_attributes:
        if attr in model_output["system_attributes"]:
            si_attr = si_model_output["system_attributes"][attr]
            batched_attr = model_output["system_attributes"][attr]
            expected_attr = batched_attr[0]
            if not torch.allclose(si_attr, expected_attr, atol=10e-3):
                raise ValueError(f"{attr}: {si_attr=} != {expected_attr=}")

            fe_attr = fe_model_output["system_attributes"][attr]
            expected_fe_attr = batched_attr[1]
            if not torch.allclose(fe_attr, expected_fe_attr, atol=10e-2):
                raise ValueError(f"{attr}: {fe_attr=} != {expected_fe_attr=}")

    for attr in expected_output_global_attributes:
        if attr in model_output["global_attributes"]:
            si_attr = si_model_output["global_attributes"][attr]
            fe_attr = fe_model_output["global_attributes"][attr]
            batched_attr = model_output["global_attributes"][attr]
            if not torch.allclose(si_attr, batched_attr, atol=10e-3):
                raise ValueError(f"{attr}: {si_attr=} != {batched_attr=}")
            if not torch.allclose(fe_attr, batched_attr, atol=10e-2):
                raise ValueError(f"{attr}: {fe_attr=} != {batched_attr=}")
