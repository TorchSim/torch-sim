"""Electric-field corrections for polarization-aware models."""

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState
from torch_sim.typing import AtomExtras, SystemExtras


class UniformPolarizationModel(ModelInterface):
    """Calculates the energy and force contributions from the application
    of a constant electric field to a polarizable system.

    This model is intended to run after an upstream model inside
    :class:`~torch_sim.models.interface.SerialSumModel`.

    Required state extras:

    * ``external_E_field``
    * ``total_polarization``
    * ``polarizability``
    * ``born_effective_charges`` when ``compute_forces`` is enabled
    """

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        *,
        compute_forces: bool = True,
        compute_stress: bool = True,
        retain_graph: bool = False,
    ) -> None:
        """Initialize a uniform-field polarization correction model."""
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._retain_graph = retain_graph
        self._memory_scales_with = "n_atoms"

    @ModelInterface.compute_stress.setter
    def compute_stress(self, value: bool) -> None:  # noqa: FBT001
        """Set whether the model returns an additive stress tensor."""
        self._compute_stress = value

    @ModelInterface.compute_forces.setter
    def compute_forces(self, value: bool) -> None:  # noqa: FBT001
        """Set whether the model returns additive force corrections."""
        self._compute_forces = value

    @property
    def retain_graph(self) -> bool:
        """Whether outputs should remain attached to the autograd graph."""
        return self._retain_graph

    @retain_graph.setter
    def retain_graph(self, value: bool) -> None:
        """Set whether outputs should remain attached to the autograd graph."""
        self._retain_graph = value

    def _finalize_output(
        self, output: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Detach outputs unless graph retention is enabled."""
        if self.retain_graph:
            return output
        return {
            key: val.detach() if isinstance(val, torch.Tensor) else val
            for key, val in output.items()
        }

    def _apply_nonzero_field(
        self,
        state: SimState,
        output: dict[str, torch.Tensor],
        field: torch.Tensor,
    ) -> None:
        """Apply constant-field linear-response corrections.

        Computes the additive updates
        - delta_energy = -E·P0 - E·alpha·E
        - total_polarization = P0 + alpha·E
        - delta_forces = Z*·E
        """
        required_keys = [
            SystemExtras.TOTAL_POLARIZATION.value,
            SystemExtras.POLARIZABILITY.value,
        ]
        if self.compute_forces:
            required_keys.append(AtomExtras.BORN_EFFECTIVE_CHARGES.value)

        missing_keys = [key for key in required_keys if not state.has_extras(key)]
        if missing_keys:
            missing = ", ".join(f"'{key}'" for key in missing_keys)
            raise ValueError(
                f"UniformPolarizationModel requires {missing} on the state "
                "when external_E_field is non-zero"
            )

        dipole_coupling = torch.einsum("si,si->s", field, state.total_polarization)
        polarization_response = torch.einsum(
            "si,sij,sj->s", field, state.polarizability, field
        )
        output["energy"] = -dipole_coupling - polarization_response
        output[SystemExtras.TOTAL_POLARIZATION.value] = (
            torch.einsum(
                "sij,sj->si",
                state.polarizability,
                field,
            )
            + state.total_polarization
        )
        if self.compute_forces:
            output["forces"] = torch.einsum(
                "imn,im->in",
                state.born_effective_charges,
                field[state.system_idx],
            )

    def forward(self, state: SimState, **kwargs) -> dict[str, torch.Tensor]:
        """Return additive uniform-field corrections for a polarization model."""
        del kwargs
        output: dict[str, torch.Tensor] = {
            "energy": torch.zeros(state.n_systems, device=state.device, dtype=state.dtype)
        }
        if self.compute_forces:
            output["forces"] = torch.zeros_like(state.positions)
        if self.compute_stress:
            # V1 intentionally applies no field-induced stress correction.
            output["stress"] = torch.zeros(
                state.n_systems, 3, 3, device=state.device, dtype=state.dtype
            )

        if not state.has_extras(SystemExtras.EXTERNAL_E_FIELD.value):
            raise ValueError(
                "UniformPolarizationModel requires 'external_E_field' on the state"
            )

        field = getattr(state, SystemExtras.EXTERNAL_E_FIELD.value)
        if field.shape != (state.n_systems, 3):
            raise ValueError(
                "UniformPolarizationModel requires external_E_field to have shape "
                "(n_systems, 3)"
            )
        if not torch.any(field != 0):
            return self._finalize_output(output)

        self._apply_nonzero_field(state, output, field)
        return self._finalize_output(output)


__all__ = ["UniformPolarizationModel"]
