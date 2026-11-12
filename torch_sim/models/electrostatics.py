"""Electrostatics models: DSF, Ewald, and PME.

Wraps the ``nvalchemiops`` Warp-accelerated electrostatics implementations as
:class:`~torch_sim.models.interface.ModelInterface` subclasses, with full PBC,
stress (virial), and batched system support.  Per-atom partial charges are read
from ``state.partial_charges`` (a SimState atom extra).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from nvalchemiops.torch.interactions.electrostatics import (
    coulomb_energy_forces,
    dsf_coulomb,
    ewald_summation,
    particle_mesh_ewald,
)

from torch_sim import transforms
from torch_sim._duecredit import dcite
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import torchsim_nl
from torch_sim.units import UnitConversion


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch_sim.state import SimState


def _direct_coulomb_result(
    state: SimState,
    dtype: torch.dtype,
    compute_forces: bool,  # noqa: FBT001
    compute_stress: bool,  # noqa: FBT001
) -> dict[str, torch.Tensor]:
    """Return the bare Coulomb sum and zero stress for non-periodic states."""
    charges = state.partial_charges
    dev = state.positions.device
    # Every ordered pair of distinct atoms within one system
    n_atoms_per_system = state.n_atoms_per_system
    system_size = n_atoms_per_system[state.system_idx]
    system_start = (n_atoms_per_system.cumsum(0) - n_atoms_per_system)[state.system_idx]
    i = torch.repeat_interleave(torch.arange(state.n_atoms, device=dev), system_size)
    block_start = system_size.cumsum(0) - system_size
    j = system_start[i] + torch.arange(i.shape[0], device=dev) - block_start[i]
    mapping = torch.stack([i, j])[:, i != j]
    edge_index, neighbor_ptr, unit_shifts = transforms.build_csr_neighbor_list(
        mapping,
        state.system_idx[mapping[0]],
        mapping.new_zeros(mapping.shape[1], 3),
        state.n_atoms,
    )
    per_atom_energy, forces = coulomb_energy_forces(
        positions=state.positions,
        charges=charges,
        cell=state.row_vector_cell.contiguous(),
        cutoff=math.inf,
        neighbor_list=edge_index,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=unit_shifts,
        batch_idx=state.system_idx.to(torch.int32),
    )
    energy = torch.zeros(state.n_systems, dtype=per_atom_energy.dtype, device=dev)
    energy.scatter_add_(0, state.system_idx.long(), per_atom_energy)
    energy = energy * UnitConversion.e2_per_Ang_to_eV
    results: dict[str, torch.Tensor] = {"energy": energy.to(dtype).detach()}
    if compute_forces:
        forces = forces * UnitConversion.e2_per_Ang_to_eV
        results["forces"] = forces.to(dtype).detach()
    if compute_stress:
        results["stress"] = torch.zeros(state.n_systems, 3, 3, dtype=dtype, device=dev)
    return results


def _check_nonperiodic_cell_vectors(state: SimState) -> None:
    """Raise if a partially periodic state has a zero cell vector on a non-periodic axis.

    Ewald and PME treat the non-periodic axis as a periodic one padded with vacuum,
    so it needs a cell vector whose length sets that vacuum.
    """
    zero_vectors = (state.row_vector_cell == 0).all(dim=-1) & ~state.pbc
    if zero_vectors.any():
        raise ValueError(
            "Ewald and PME cannot compute valid values for a partially periodic "
            "state with a zero cell vector on a non-periodic axis."
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
        edge_index, _mapping, unit_shifts = self.neighbor_list_fn(
            state.positions,
            state.row_vector_cell,
            state.pbc,
            self.cutoff,
            state.system_idx,
        )
        edge_index, neighbor_ptr, unit_shifts = transforms.build_csr_neighbor_list(
            edge_index,
            _mapping,
            unit_shifts,
            state.positions.shape[0],
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
            # nvalchemiops returns the negative strain-gradient virial; TorchSim stress
            # follows dE / dstrain / volume, matching ASE and other TorchSim models.
            stress = -(out[-1] * UnitConversion.e2_per_Ang_to_eV) / volumes
            stress = torch.where(volumes > 0, stress, torch.zeros_like(stress))
            results["stress"] = stress.to(self._dtype).detach()
        return results


class EwaldModel(ModelInterface):
    """Classical Ewald summation as a :class:`ModelInterface`.

    Uses the ``nvalchemiops`` Ewald kernel for exact periodic electrostatics.
    Returns per-atom energies that are aggregated to per-system.  All
    user-facing quantities are in metal units (Angstrom / eV).

    Per-atom partial charges are read from ``state.partial_charges``.

    Non-periodic states get the bare Coulomb sum over all atom pairs and zero stress.

    Systems with one non-periodic axis are treated as slabs and get the slab
    correction. Partially periodic states need a non-zero cell vector with vacuum
    on every non-periodic axis.

    Args:
        cutoff: Real-space cutoff in Angstrom.
        accuracy: Target accuracy, which sets the Ewald splitting parameter and the
            reciprocal-space cutoff together with ``cutoff``.
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
                atom extra (shape ``[n_atoms]``).
            **_kwargs: Unused; accepted for interface compatibility.

        Returns:
            dict with ``"energy"`` [n_systems], ``"forces"`` [n_atoms, 3],
            and (if ``compute_stress``) ``"stress"`` [n_systems, 3, 3].
        """
        if not state.has_extras("partial_charges"):
            raise ValueError("Partial charges are required for Ewald summation.")

        if not state.pbc.any():
            return _direct_coulomb_result(
                state, self._dtype, self._compute_forces, self._compute_stress
            )
        if not state.pbc.all():
            _check_nonperiodic_cell_vectors(state)
        charges = state.partial_charges
        edge_index, _mapping, unit_shifts = self.neighbor_list_fn(
            state.positions,
            state.row_vector_cell,
            state.pbc,
            self.cutoff,
            state.system_idx,
        )
        edge_index, neighbor_ptr, unit_shifts = transforms.build_csr_neighbor_list(
            edge_index,
            _mapping,
            unit_shifts,
            state.positions.shape[0],
        )
        cell = state.row_vector_cell.contiguous()
        # The real-space cutoff is fixed, so alpha and k_cutoff follow from accuracy
        log_accuracy = math.log(self.accuracy)
        out = ewald_summation(
            positions=state.positions,
            charges=charges,
            cell=cell,
            alpha=math.sqrt(-log_accuracy) / self.cutoff,
            k_cutoff=-2.0 * log_accuracy / self.cutoff,
            neighbor_list=edge_index,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=unit_shifts,
            batch_idx=state.system_idx.to(torch.int32),
            compute_forces=self._compute_forces,
            compute_virial=self._compute_stress,
            accuracy=self.accuracy,
            pbc=state.pbc.expand(state.n_systems, 3),
            slab_correction=not state.pbc.all(),
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
            forces = out[1] * UnitConversion.e2_per_Ang_to_eV
            results["forces"] = forces.to(self._dtype).detach()
        if self._compute_stress:
            volumes = state.volume.unsqueeze(-1).unsqueeze(-1)
            # nvalchemiops returns the negative strain-gradient virial; TorchSim stress
            # follows dE / dstrain / volume, matching ASE and other TorchSim models.
            stress = -(out[-1] * UnitConversion.e2_per_Ang_to_eV) / volumes
            # A batch shares one k-space grid, which can give a small cell a 0/0 virial
            if state.n_systems > 1:
                non_finite = ~torch.isfinite(stress).all(dim=(1, 2))
                for sys_idx in non_finite.nonzero().flatten().tolist():
                    stress[sys_idx] = self(state[sys_idx])["stress"][0]
            results["stress"] = stress.to(self._dtype).detach()
        return results


class PMEModel(ModelInterface):
    """Particle Mesh Ewald electrostatics as a :class:`ModelInterface`.

    Uses the ``nvalchemiops`` PME kernel for O(N log N) periodic
    electrostatics.  Returns per-atom energies that are aggregated to
    per-system.  All user-facing quantities are in metal units
    (Angstrom / eV).

    Per-atom partial charges are read from ``state.partial_charges``.

    Non-periodic states get the bare Coulomb sum over all atom pairs and zero stress.

    Systems with one non-periodic axis are treated as slabs and get the slab
    correction. Partially periodic states need a non-zero cell vector with vacuum
    on every non-periodic axis.

    Args:
        cutoff: Real-space cutoff in Angstrom.
        accuracy: Target accuracy, which sets the Ewald splitting parameter and the
            default mesh together with ``cutoff``.
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
                atom extra (shape ``[n_atoms]``).
            **_kwargs: Unused; accepted for interface compatibility.

        Returns:
            dict with ``"energy"`` [n_systems], ``"forces"`` [n_atoms, 3],
            and (if ``compute_stress``) ``"stress"`` [n_systems, 3, 3].
        """
        if not state.has_extras("partial_charges"):
            raise ValueError("Partial charges are required for PME summation.")

        if not state.pbc.any():
            return _direct_coulomb_result(
                state, self._dtype, self._compute_forces, self._compute_stress
            )
        if not state.pbc.all():
            _check_nonperiodic_cell_vectors(state)
        charges = state.partial_charges
        edge_index, _mapping, unit_shifts = self.neighbor_list_fn(
            state.positions,
            state.row_vector_cell,
            state.pbc,
            self.cutoff,
            state.system_idx,
        )
        edge_index, neighbor_ptr, unit_shifts = transforms.build_csr_neighbor_list(
            edge_index,
            _mapping,
            unit_shifts,
            state.positions.shape[0],
        )
        cell = state.row_vector_cell.contiguous()
        batch_idx = state.system_idx.to(torch.int32) if state.n_systems > 1 else None
        pme_kwargs: dict = dict(
            positions=state.positions,
            charges=charges,
            cell=cell,
            # The real-space cutoff is fixed, so alpha follows from accuracy
            alpha=math.sqrt(-math.log(self.accuracy)) / self.cutoff,
            neighbor_list=edge_index,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=unit_shifts,
            batch_idx=batch_idx,
            compute_forces=self._compute_forces,
            compute_virial=self._compute_stress,
            accuracy=self.accuracy,
            spline_order=self.spline_order,
            pbc=state.pbc.expand(state.n_systems, 3),
            slab_correction=not state.pbc.all(),
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
            forces = out[1] * UnitConversion.e2_per_Ang_to_eV
            results["forces"] = forces.to(self._dtype).detach()
        if self._compute_stress:
            volumes = state.volume.unsqueeze(-1).unsqueeze(-1)
            # nvalchemiops returns the negative strain-gradient virial; TorchSim stress
            # follows dE / dstrain / volume, matching ASE and other TorchSim models.
            stress = -(out[-1] * UnitConversion.e2_per_Ang_to_eV) / volumes
            results["stress"] = stress.to(self._dtype).detach()
        return results
