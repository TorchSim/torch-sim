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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

import torch_sim as ts


if TYPE_CHECKING:
    from torch_sim.state import SimState
    from torch_sim.typing import MemoryScaling


VALIDATE_ATOL = 1e-4

_MEMORY_SCALING_PRIORITY: dict[MemoryScaling, int] = {
    "n_atoms": 0,
    "n_atoms_x_density": 1,
    "n_edges": 2,
}


class ModelInterface(torch.nn.Module, ABC):
    """Abstract base class for all simulation models in TorchSim.

    This interface provides a common structure for all energy and force models,
    ensuring they implement the required methods and properties. It defines how
    models should process atomic positions and system information to compute energies,
    forces, and stresses.

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
    def forward(self, state: SimState, **kwargs) -> dict[str, torch.Tensor]:
        """Calculate energies, forces, and stresses for a atomistic system.

        This is the main computational method that all model implementations must provide.
        It takes atomic positions and system information as input and returns a dictionary
        containing computed physical properties.

        Args:
            state (SimState): Simulation state containing:
                - positions: Atomic positions with shape [n_atoms, 3]
                - cell: Unit cell vectors with shape [n_systems, 3, 3]
                - system_idx: System indices for each atom with shape [n_atoms]
                - atomic_numbers: Atomic numbers with shape [n_atoms] (optional)
            **kwargs: Additional model-specific parameters.

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                - "energy": Potential energy with shape [n_systems]
                - "forces": Atomic forces with shape [n_atoms, 3]
                - "stress": Stress tensor with shape [n_systems, 3, 3] (if
                    compute_stress=True)
                - May include additional model-specific outputs

        Examples:
            ```py
            # Compute energies and forces with a model
            output = model.forward(state)

            energy = output["energy"]
            forces = output["forces"]
            stress = output.get("stress", None)
            ```
        """


class SumModel(ModelInterface):
    """Additive composition of multiple :class:`ModelInterface` models.

    Calls each child model's :meth:`forward` and sums the output tensors
    key-by-key, so energies, forces, and stresses are combined additively.
    This is the standard way to layer a dispersion correction (e.g. DFT-D3),
    an Ewald electrostatic term, or a local pair potential on top of a primary
    machine-learning potential.

    Args:
        models: Two or more :class:`ModelInterface` instances that share the
            same ``device`` and ``dtype``.

    Raises:
        ValueError: If fewer than two models are given or if ``device``/``dtype``
            do not match across all models.

    Examples:
        ```py
        sum_model = SumModel(mace_model, d3_model)
        output = sum_model(sim_state)
        ```
    """

    def __init__(self, *models: ModelInterface) -> None:
        """Initialize the sum model.

        Args:
            models: Two or more :class:`ModelInterface` instances. All must
                share the same ``device`` and ``dtype``.
        """
        super().__init__()
        if len(models) < 2:
            raise ValueError("SumModel requires at least two child models")
        first = models[0]
        for i, m in enumerate(models[1:], start=1):
            if m.device != first.device:
                raise ValueError(
                    f"Device mismatch: model 0 has {first.device}, "
                    f"model {i} has {m.device}"
                )
            if m.dtype != first.dtype:
                raise ValueError(
                    f"Dtype mismatch: model 0 has {first.dtype}, model {i} has {m.dtype}"
                )
        self.models = torch.nn.ModuleList(models)
        self._device = first.device
        self._dtype = first.dtype
        self._compute_stress = all(m.compute_stress for m in models)
        self._compute_forces = all(m.compute_forces for m in models)

    @ModelInterface.compute_stress.setter
    def compute_stress(self, value: bool) -> None:  # noqa: FBT001
        """Propagate ``compute_stress`` to all child models that support it."""
        for m in self.models:
            try:
                m.compute_stress = value
            except NotImplementedError:
                if value:
                    raise
        self._compute_stress = value

    @ModelInterface.compute_forces.setter
    def compute_forces(self, value: bool) -> None:  # noqa: FBT001
        """Propagate ``compute_forces`` to all child models that support it."""
        for m in self.models:
            try:
                m.compute_forces = value
            except NotImplementedError:
                if value:
                    raise
        self._compute_forces = value

    @property
    def retain_graph(self) -> bool:
        """Whether any child model retains the computation graph."""
        return any(getattr(m, "retain_graph", False) for m in self.models)

    @retain_graph.setter
    def retain_graph(self, value: bool) -> None:
        for m in self.models:
            if hasattr(m, "retain_graph"):
                m.retain_graph = value

    @property
    def memory_scales_with(self) -> MemoryScaling:
        """Most conservative memory-scaling among all child models."""
        best: MemoryScaling = "n_atoms"
        for m in self.models:
            scaling = m.memory_scales_with
            if _MEMORY_SCALING_PRIORITY[scaling] > _MEMORY_SCALING_PRIORITY[best]:
                best = scaling
        return best

    def forward(self, state: SimState, **kwargs) -> dict[str, torch.Tensor]:
        """Sum the outputs of all child models.

        Each child model is called with the same ``state`` and ``**kwargs``.
        Output tensors that appear in multiple children are summed element-wise;
        keys unique to a single child are passed through unchanged.

        Args:
            state: Simulation state (see :class:`ModelInterface`).
            **kwargs: Forwarded to every child model.

        Returns:
            Combined output dictionary with summed tensors.
        """
        combined: dict[str, torch.Tensor] = {}
        for model in self.models:
            output = model(state, **kwargs)
            for key, tensor in output.items():
                if key in combined:
                    combined[key] = combined[key] + tensor
                else:
                    combined[key] = tensor
        return combined


def _check_output_detached(
    output: dict[str, torch.Tensor], model: ModelInterface
) -> None:
    """Check that output tensors match the model's graph retention setting.

    When ``retain_graph`` is absent or ``False``, all tensors must be detached.
    When ``retain_graph`` is ``True``, all tensors must have ``requires_grad``.

    Args:
        output: Model output dictionary mapping keys to tensors.
        model: The model that produced the output.

    Raises:
        ValueError: If tensors are not detached when ``retain_graph`` is
            ``False``, or lack gradients when ``retain_graph`` is ``True``.
    """
    retain_graph = getattr(model, "retain_graph", False)
    for key, tensor in output.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if retain_graph and not tensor.requires_grad:
            raise ValueError(
                f"Output tensor '{key}' does not have gradients but model.retain_graph "
                "is True. Ensure the tensor is part of the computation graph."
            )
        if not retain_graph and tensor.requires_grad:
            raise ValueError(
                f"Output tensor '{key}' is not detached from the computation graph. "
                "Call .detach() on the tensor before returning it, or set "
                "model.retain_graph = True if graph retention is intentional."
            )


def validate_model_outputs(  # noqa: C901, PLR0915
    model: ModelInterface,
    device: torch.device,
    dtype: torch.dtype,
    *,
    check_detached: bool = False,
) -> None:
    """Validate the outputs of a model implementation against the interface requirements.

    Runs a series of tests to ensure a model implementation correctly follows the
    ModelInterface contract. The tests include creating sample systems, running
    forward passes, and verifying output shapes and consistency.

    Args:
        model (ModelInterface): Model implementation to validate.
        device (torch.device): Device to run the validation tests on.
        dtype (torch.dtype): Data type to use for validation tensors.
        check_detached (bool): If ``True``, assert that all output tensors are
            detached from the autograd graph, unless the model has a
            ``retain_graph`` attribute set to ``True``. Defaults to ``False`` so
            that external callers are not immediately broken.

    Raises:
        AssertionError: If the model doesn't conform to the required interface,
            including issues with output shapes, types, or behavior consistency.

    Example::

        # Create a new model implementation
        model = MyCustomModel(device=torch.device("cuda"))

        # Validate that it correctly implements the interface
        validate_model_outputs(model, device=torch.device("cuda"), dtype=torch.float64)

    Notes:
        This validator creates small test systems (diamond silicon, HCP magnesium,
        and primitive BCC iron) for validation. It tests both single and
        multi-batch processing capabilities.
    """
    from ase.build import bulk

    for attr in ("dtype", "device", "compute_stress", "compute_forces"):
        if not hasattr(model, attr):
            raise ValueError(f"model.{attr} is not set")

    try:
        if not model.compute_stress:
            model.compute_stress = True
        stress_computed = True
    except NotImplementedError:
        stress_computed = False

    try:
        if not model.compute_forces:
            model.compute_forces = True
        force_computed = True
    except NotImplementedError:
        force_computed = False

    si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    mg_atoms = bulk("Mg", "hcp", a=3.21, c=5.21).repeat([3, 2, 1])
    fe_atoms = bulk("Fe", "bcc", a=2.87)
    sim_state = ts.io.atoms_to_state([si_atoms, mg_atoms, fe_atoms], device, dtype)

    og_positions = sim_state.positions.clone()
    og_cell = sim_state.cell.clone()
    system_idx = sim_state.system_idx
    og_system_idx = system_idx.clone()
    og_atomic_nums = sim_state.atomic_numbers.clone()

    if check_detached and hasattr(model, "retain_graph"):
        model.__dict__["retain_graph"] = True
        _check_output_detached(model.forward(sim_state), model)
        model.__dict__["retain_graph"] = False

    model_output = model.forward(sim_state)

    if check_detached:
        _check_output_detached(model_output, model)

    # assert model did not mutate the input
    if not torch.allclose(og_positions, sim_state.positions):
        raise ValueError(f"{og_positions=} != {sim_state.positions=}")
    if not torch.allclose(og_cell, sim_state.cell):
        raise ValueError(f"{og_cell=} != {sim_state.cell=}")
    if not torch.allclose(og_system_idx, system_idx):
        raise ValueError(f"{og_system_idx=} != {sim_state.system_idx=}")
    if not torch.allclose(og_atomic_nums, sim_state.atomic_numbers):
        raise ValueError(f"{og_atomic_nums=} != {sim_state.atomic_numbers=}")

    # assert model output has the correct keys
    if "energy" not in model_output:
        raise ValueError("energy not in model output")
    if force_computed and "forces" not in model_output:
        raise ValueError("forces not in model output")
    if stress_computed and "stress" not in model_output:
        raise ValueError("stress not in model output")

    # assert model output shapes are correct
    if model_output["energy"].shape != (3,):
        raise ValueError(f"{model_output['energy'].shape=} != (3,)")
    if force_computed and model_output["forces"].shape != (21, 3):
        raise ValueError(f"{model_output['forces'].shape=} != (21, 3)")
    if stress_computed and model_output["stress"].shape != (3, 3, 3):
        raise ValueError(f"{model_output['stress'].shape=} != (3, 3, 3)")

    # Test single Si system output shapes (8 atoms)
    si_state = ts.io.atoms_to_state([si_atoms], device, dtype)

    si_model_output = model.forward(si_state)
    if not torch.allclose(
        si_model_output["energy"], model_output["energy"][0], atol=VALIDATE_ATOL
    ):
        raise ValueError(f"{si_model_output['energy']=} != {model_output['energy'][0]=}")
    if not torch.allclose(
        forces := si_model_output["forces"],
        expected_forces := model_output["forces"][: si_state.n_atoms],
        atol=VALIDATE_ATOL,
    ):
        raise ValueError(f"{forces=} != {expected_forces=}")

    if si_model_output["energy"].shape != (1,):
        raise ValueError(f"{si_model_output['energy'].shape=} != (1,)")
    if force_computed and si_model_output["forces"].shape != (8, 3):
        raise ValueError(f"{si_model_output['forces'].shape=} != (8, 3)")
    if stress_computed and si_model_output["stress"].shape != (1, 3, 3):
        raise ValueError(f"{si_model_output['stress'].shape=} != (1, 3, 3)")

    # Test single Mg system output shapes (12 atoms)
    mg_state = ts.io.atoms_to_state([mg_atoms], device, dtype)
    mg_model_output = model.forward(mg_state)
    if not torch.allclose(
        mg_model_output["energy"], model_output["energy"][1], atol=VALIDATE_ATOL
    ):
        raise ValueError(f"{mg_model_output['energy']=} != {model_output['energy'][1]=}")
    mg_n = mg_state.n_atoms
    mg_slice = slice(si_state.n_atoms, si_state.n_atoms + mg_n)
    if not torch.allclose(
        forces := mg_model_output["forces"],
        expected_forces := model_output["forces"][mg_slice],
        atol=VALIDATE_ATOL,
    ):
        raise ValueError(f"{forces=} != {expected_forces=}")

    if mg_model_output["energy"].shape != (1,):
        raise ValueError(f"{mg_model_output['energy'].shape=} != (1,)")
    if force_computed and mg_model_output["forces"].shape != (12, 3):
        raise ValueError(f"{mg_model_output['forces'].shape=} != (12, 3)")
    if stress_computed and mg_model_output["stress"].shape != (1, 3, 3):
        raise ValueError(f"{mg_model_output['stress'].shape=} != (1, 3, 3)")

    # Test single Fe system output shapes (1 atom)
    # This catches that models do not squeeze away singleton dimensions.
    fe_state = ts.io.atoms_to_state([fe_atoms], device, dtype)
    fe_model_output = model.forward(fe_state)
    if not torch.allclose(
        fe_model_output["energy"], model_output["energy"][2], atol=VALIDATE_ATOL
    ):
        raise ValueError(f"{fe_model_output['energy']=} != {model_output['energy'][2]=}")
    if not torch.allclose(
        forces := fe_model_output["forces"],
        expected_forces := model_output["forces"][si_state.n_atoms + mg_n :],
        atol=VALIDATE_ATOL,
    ):
        raise ValueError(f"{forces=} != {expected_forces=}")

    if fe_model_output["energy"].shape != (1,):
        raise ValueError(f"{fe_model_output['energy'].shape=} != (1,)")
    if force_computed and fe_model_output["forces"].shape != (1, 3):
        raise ValueError(f"{fe_model_output['forces'].shape=} != (1, 3)")
    if stress_computed and fe_model_output["stress"].shape != (1, 3, 3):
        raise ValueError(f"{fe_model_output['stress'].shape=} != (1, 3, 3)")

    # Translating one atom by a full lattice vector should not change outputs.
    # This catches models that fail to apply periodic boundary conditions.
    shifted_state = si_state.clone()
    lattice_vec = shifted_state.cell[0, :, 0]  # column convention
    shifted_state.positions[0] = shifted_state.positions[0] + 3 * lattice_vec
    shifted_output = model.forward(shifted_state)
    if not torch.allclose(
        shifted_output["energy"], si_model_output["energy"], atol=VALIDATE_ATOL
    ):
        raise ValueError(
            "Energy changed after translating an atom by a lattice "
            f"vector: {shifted_output['energy']=} != "
            f"{si_model_output['energy']=}"
        )
    if force_computed and not torch.allclose(
        shifted_output["forces"], si_model_output["forces"], atol=VALIDATE_ATOL
    ):
        raise ValueError(
            "Forces changed after translating an atom by a lattice "
            "vector: max diff = "
            f"{(shifted_output['forces'] - si_model_output['forces']).abs().max()}"
        )
    if stress_computed and not torch.allclose(
        shifted_output["stress"], si_model_output["stress"], atol=VALIDATE_ATOL
    ):
        raise ValueError(
            "Stress changed after translating an atom by a lattice "
            "vector: max diff = "
            f"{(shifted_output['stress'] - si_model_output['stress']).abs().max()}"
        )
