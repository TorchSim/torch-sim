"""Electrostatics models: DSF, Ewald, and PME.

Wraps the ``nvalchemiops`` Warp-accelerated electrostatics implementations as
:class:`~torch_sim.models.interface.ModelInterface` subclasses, with full PBC,
stress (virial), and batched system support.  Per-atom partial charges are read
from ``state.partial_charges`` (a SimState atom extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from nvalchemiops.torch.interactions.electrostatics import (
    dsf_coulomb,
    ewald_summation,
    particle_mesh_ewald,
)

from torch_sim._duecredit import dcite
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import torchsim_nl
from torch_sim.units import UnitConversion


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch_sim.state import SimState


def _zero_result(
    state: SimState,
    dtype: torch.dtype,
    compute_forces: bool,  # noqa: FBT001
    compute_stress: bool,  # noqa: FBT001
) -> dict[str, torch.Tensor]:
    """Return zero energy / forces / stress for non-periodic states."""
    dev = state.positions.device
    results: dict[str, torch.Tensor] = {
        "energy": torch.zeros(state.n_systems, dtype=dtype, device=dev),
    }
    if compute_forces:
        results["forces"] = torch.zeros(state.n_atoms, 3, dtype=dtype, device=dev)
    if compute_stress:
        results["stress"] = torch.zeros(state.n_systems, 3, 3, dtype=dtype, device=dev)
    return results


def _build_csr(
    state: SimState,
    cutoff: float,
    neighbor_list_fn: Callable,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a CSR neighbor list and integer unit-shift tensor."""
    edge_index, _mapping, unit_shifts = neighbor_list_fn(
        state.positions,
        state.row_vector_cell,
        state.pbc,
        cutoff,
        state.system_idx,
    )
    n_atoms = state.positions.shape[0]
    dev = state.positions.device
    neighbor_ptr = torch.zeros(n_atoms + 1, dtype=torch.int32, device=dev)
    neighbor_ptr[1:] = (
        torch.bincount(edge_index[0], minlength=n_atoms).cumsum(0).to(torch.int32)
    )
    return (
        edge_index.to(torch.int32),
        neighbor_ptr,
        unit_shifts.to(torch.int32),
    )


class DSFCoulombModel(ModelInterface):
    """Damped Shifted Force electrostatics as a :class:`ModelInterface`.

    Uses the ``nvalchemiops`` DSF kernel for O(N) electrostatic energy,
    forces, and (optionally) stress.  All user-facing quantities are in
    metal units (Angstrom / eV); the Coulomb constant ``ke`` is baked in.

    Per-atom partial charges are read from ``state.partial_charges``.

    Args:
        cutoff: Real-space cutoff in Angstrom.
        alpha: DSF damping parameter. 0.0 gives shifted-force bare Coulomb.
        device: Compute device. Defaults to CUDA if available, else CPU.
        dtype: Floating-point dtype. Defaults to ``torch.float64``.
        compute_forces: Whether to return forces. Defaults to True.
        compute_stress: Whether to return stress. Defaults to True.
        neighbor_list_fn: Neighbor-list constructor. Defaults to ``torchsim_nl``.
    """

    @dcite("10.1063/1.2206581", description="Fennell & Gezelter DSF method")
    def __init__(
        self,
        cutoff: float = 10.0,
        *,
        alpha: float = 0.2,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        compute_forces: bool = True,
        compute_stress: bool = True,
        neighbor_list_fn: Callable = torchsim_nl,
    ) -> None:
        """Initialize the DSF Coulomb model."""
        super().__init__()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._memory_scales_with = "n_atoms_x_density"
        self.neighbor_list_fn = neighbor_list_fn
        self.cutoff = cutoff
        self.alpha = alpha

    def forward(self, state: SimState, **_kwargs: object) -> dict[str, torch.Tensor]:
        """Compute DSF electrostatic energy, forces, and stress.

        Args:
            state: Simulation state with ``partial_charges`` set as an
                atom extra (shape ``[n_atoms]``).
            **_kwargs: Unused; accepted for interface compatibility.

        Returns:
            dict with ``"energy"`` [n_systems], ``"forces"`` [n_atoms, 3],
            and (if ``compute_stress``) ``"stress"`` [n_systems, 3, 3].
        """
        if not state.has_extras("partial_charges"):
            raise ValueError("Partial charges are required for DSF Coulomb summation.")

        charges = state.partial_charges
        edge_index, neighbor_ptr, unit_shifts = _build_csr(
            state, self.cutoff, self.neighbor_list_fn
        )
        cell = state.row_vector_cell.contiguous()
        dsf_args: dict = dict(
            positions=state.positions,
            charges=charges,
            cutoff=self.cutoff,
            alpha=self.alpha,
            neighbor_list=edge_index,
            neighbor_ptr=neighbor_ptr,
            unit_shifts=unit_shifts,
            cell=cell,
            batch_idx=state.system_idx.to(torch.int32),
            compute_forces=self._compute_forces,
            compute_virial=self._compute_stress,
            num_systems=state.n_systems,
        )
        out = dsf_coulomb(**dsf_args)
        if not isinstance(out, tuple):
            out = (out,)
        energy = (out[0] * UnitConversion.e2_per_Ang_to_eV).to(self._dtype).detach()
        results: dict[str, torch.Tensor] = {"energy": energy}
        if self._compute_forces:
            forces = out[1] * UnitConversion.e2_per_Ang_to_eV  # type: ignore[index]
            results["forces"] = forces.to(self._dtype).detach()
        if self._compute_stress:
            volumes = state.volume.unsqueeze(-1).unsqueeze(-1)
            stress = (out[-1] * UnitConversion.e2_per_Ang_to_eV) / volumes
            results["stress"] = stress.to(self._dtype).detach()
        return results


class EwaldModel(ModelInterface):
    """Classical Ewald summation as a :class:`ModelInterface`.

    Uses the ``nvalchemiops`` Ewald kernel for exact periodic electrostatics.
    Returns per-atom energies that are aggregated to per-system.  All
    user-facing quantities are in metal units (Angstrom / eV).

    Per-atom partial charges are read from ``state.partial_charges``.

    Requires periodic boundary conditions.

    Args:
        cutoff: Real-space cutoff in Angstrom.
        accuracy: Target accuracy for auto-estimated Ewald parameters.
        device: Compute device. Defaults to CUDA if available, else CPU.
        dtype: Floating-point dtype. Defaults to ``torch.float64``.
        compute_forces: Whether to return forces. Defaults to True.
        compute_stress: Whether to return stress. Defaults to True.
        neighbor_list_fn: Neighbor-list constructor. Defaults to ``torchsim_nl``.
    """

    @dcite("10.1002/andp.19213690304", description="Ewald summation method")
    def __init__(
        self,
        cutoff: float = 10.0,
        *,
        accuracy: float = 1e-6,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        compute_forces: bool = True,
        compute_stress: bool = True,
        neighbor_list_fn: Callable = torchsim_nl,
    ) -> None:
        """Initialize the Ewald model."""
        super().__init__()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._memory_scales_with = "n_atoms_x_density"
        self.neighbor_list_fn = neighbor_list_fn
        self.cutoff = cutoff
        self.accuracy = accuracy

    def forward(self, state: SimState, **_kwargs: object) -> dict[str, torch.Tensor]:
        """Compute Ewald electrostatic energy, forces, and stress.

        Args:
            state: Simulation state with ``partial_charges`` set as an
                atom extra (shape ``[n_atoms]``).  Returns zeros for
                non-periodic states.
            **_kwargs: Unused; accepted for interface compatibility.

        Returns:
            dict with ``"energy"`` [n_systems], ``"forces"`` [n_atoms, 3],
            and (if ``compute_stress``) ``"stress"`` [n_systems, 3, 3].
        """
        if not state.has_extras("partial_charges"):
            raise ValueError("Partial charges are required for Ewald summation.")

        if not state.pbc.any():
            return _zero_result(
                state, self._dtype, self._compute_forces, self._compute_stress
            )
        charges = state.partial_charges
        edge_index, neighbor_ptr, unit_shifts = _build_csr(
            state, self.cutoff, self.neighbor_list_fn
        )
        cell = state.row_vector_cell.contiguous()
        out = ewald_summation(
            positions=state.positions,
            charges=charges,
            cell=cell,
            neighbor_list=edge_index,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=unit_shifts,
            batch_idx=state.system_idx.to(torch.int32),
            compute_forces=self._compute_forces,
            compute_virial=self._compute_stress,
            accuracy=self.accuracy,
        )
        if not isinstance(out, tuple):
            out = (out,)
        per_atom_energy = out[0] * UnitConversion.e2_per_Ang_to_eV
        dev = state.positions.device
        energy = torch.zeros(state.n_systems, dtype=torch.float64, device=dev)
        energy.scatter_add_(0, state.system_idx.long(), per_atom_energy)
        results: dict[str, torch.Tensor] = {
            "energy": energy.to(self._dtype).detach(),
        }
        if self._compute_forces:
            forces = out[1] * UnitConversion.e2_per_Ang_to_eV  # type: ignore[index]
            results["forces"] = forces.to(self._dtype).detach()
        if self._compute_stress:
            volumes = state.volume.unsqueeze(-1).unsqueeze(-1)
            stress = (out[-1] * UnitConversion.e2_per_Ang_to_eV) / volumes
            results["stress"] = stress.to(self._dtype).detach()
        return results


class PMEModel(ModelInterface):
    """Particle Mesh Ewald electrostatics as a :class:`ModelInterface`.

    Uses the ``nvalchemiops`` PME kernel for O(N log N) periodic
    electrostatics.  Returns per-atom energies that are aggregated to
    per-system.  All user-facing quantities are in metal units
    (Angstrom / eV).

    Per-atom partial charges are read from ``state.partial_charges``.

    Requires periodic boundary conditions.

    Args:
        cutoff: Real-space cutoff in Angstrom.
        accuracy: Target accuracy for auto-estimated parameters.
        mesh_spacing: Optional mesh spacing (Angstrom) for automatic mesh sizing.
        mesh_dimensions: Explicit FFT mesh dimensions ``(nx, ny, nz)``.
        spline_order: B-spline interpolation order. Defaults to 4.
        device: Compute device. Defaults to CUDA if available, else CPU.
        dtype: Floating-point dtype. Defaults to ``torch.float64``.
        compute_forces: Whether to return forces. Defaults to True.
        compute_stress: Whether to return stress. Defaults to True.
        neighbor_list_fn: Neighbor-list constructor. Defaults to ``torchsim_nl``.
    """

    @dcite("10.1063/1.464397", description="Darden et al. PME method")
    def __init__(
        self,
        cutoff: float = 10.0,
        *,
        accuracy: float = 1e-6,
        mesh_spacing: float | None = None,
        mesh_dimensions: tuple[int, int, int] | None = None,
        spline_order: int = 4,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        compute_forces: bool = True,
        compute_stress: bool = True,
        neighbor_list_fn: Callable = torchsim_nl,
    ) -> None:
        """Initialize the PME model."""
        super().__init__()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._memory_scales_with = "n_atoms_x_density"
        self.neighbor_list_fn = neighbor_list_fn
        self.cutoff = cutoff
        self.accuracy = accuracy
        self.mesh_spacing = mesh_spacing
        self.mesh_dimensions = mesh_dimensions
        self.spline_order = spline_order

    def forward(self, state: SimState, **_kwargs: object) -> dict[str, torch.Tensor]:
        """Compute PME electrostatic energy, forces, and stress.

        Args:
            state: Simulation state with ``partial_charges`` set as an
                atom extra (shape ``[n_atoms]``).  Returns zeros for
                non-periodic states.
            **_kwargs: Unused; accepted for interface compatibility.

        Returns:
            dict with ``"energy"`` [n_systems], ``"forces"`` [n_atoms, 3],
            and (if ``compute_stress``) ``"stress"`` [n_systems, 3, 3].
        """
        if not state.has_extras("partial_charges"):
            raise ValueError("Partial charges are required for PME summation.")

        if not state.pbc.any():
            return _zero_result(
                state, self._dtype, self._compute_forces, self._compute_stress
            )
        charges = state.partial_charges
        edge_index, neighbor_ptr, unit_shifts = _build_csr(
            state, self.cutoff, self.neighbor_list_fn
        )
        cell = state.row_vector_cell.contiguous()
        batch_idx = state.system_idx.to(torch.int32) if state.n_systems > 1 else None
        pme_kwargs: dict = dict(
            positions=state.positions,
            charges=charges,
            cell=cell,
            neighbor_list=edge_index,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=unit_shifts,
            batch_idx=batch_idx,
            compute_forces=self._compute_forces,
            compute_virial=self._compute_stress,
            accuracy=self.accuracy,
            spline_order=self.spline_order,
        )
        if self.mesh_spacing is not None:
            pme_kwargs["mesh_spacing"] = self.mesh_spacing
        if self.mesh_dimensions is not None:
            pme_kwargs["mesh_dimensions"] = self.mesh_dimensions
        out = particle_mesh_ewald(**pme_kwargs)
        if not isinstance(out, tuple):
            out = (out,)
        per_atom_energy = out[0] * UnitConversion.e2_per_Ang_to_eV
        dev = state.positions.device
        energy = torch.zeros(state.n_systems, dtype=torch.float64, device=dev)
        energy.scatter_add_(0, state.system_idx.long(), per_atom_energy)
        results: dict[str, torch.Tensor] = {
            "energy": energy.to(self._dtype).detach(),
        }
        if self._compute_forces:
            forces = out[1] * UnitConversion.e2_per_Ang_to_eV  # type: ignore[index]
            results["forces"] = forces.to(self._dtype).detach()
        if self._compute_stress:
            volumes = state.volume.unsqueeze(-1).unsqueeze(-1)
            stress = (out[-1] * UnitConversion.e2_per_Ang_to_eV) / volumes
            results["stress"] = stress.to(self._dtype).detach()
        return results
