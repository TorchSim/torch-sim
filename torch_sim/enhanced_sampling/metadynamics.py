"""Bias potentials for enhanced sampling.

This module provides two composable bias potentials -- a static spherical
confining wall and a history-dependent RMSD bias -- that add extra energy and
forces on top of a base potential. Each bias is a
:class:`~torch_sim.models.interface.ModelInterface`, so it layers onto any MLIP
(or classical potential) through
:class:`~torch_sim.models.interface.SumModel`::

    bias = RMSDCV(k_push=0.02, alpha_width=1.2)
    biased_model = SumModel(mace_model, bias)
    final_state = ts.integrate(
        system=state,
        model=biased_model,
        integrator=ts.Integrator.nvt_langevin,
        n_steps=1000,
        timestep=0.002,
        temperature=300,
    )

Both potentials follow the metadynamics scheme of Grimme
(10.1021/acs.jctc.9b00143, as implemented in xtb) and keep that paper's input
units (Hartree, Bohr) for their energy/width parameters; conversion to
TorchSim's eV/Angstrom internal units happens inside the classes.

Notes:
    These biases act on Cartesian coordinates and do not apply periodic
    boundary conditions, matching their intended use on non-periodic
    (molecular/cluster) systems. They contribute no stress.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torch_sim.enhanced_sampling.history import History
from torch_sim.models.interface import ModelInterface
from torch_sim.units import UnitConversion


if TYPE_CHECKING:
    from torch_sim.state import SimState


EPS = 1e-12


def _segment_sum(
    src: torch.Tensor, system_idx: torch.Tensor, n_systems: int, dim: int = 0
) -> torch.Tensor:
    """Sum *src* over atoms belonging to the same system.

    Args:
        src: Per-atom values whose dimension *dim* has size n_atoms.
        system_idx: System index of each atom with shape [n_atoms].
        n_systems: Number of systems in the batch.
        dim: The atom dimension of *src* indexed by *system_idx*. Defaults to 0.

    Returns:
        Per-system sums where the atom dimension is replaced by n_systems.
    """
    out_shape = list(src.shape)
    out_shape[dim] = n_systems
    out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    return out.index_add(dim, system_idx, src)


class LogfermiWall(ModelInterface):
    """Spherical confining potential with a smooth log-Fermi boundary.

    Holds the atoms of each system near a center point through the per-atom
    penalty ``k_wall * softplus(beta * (r - radius))``, where ``r`` is the
    atom's distance from the center. Well inside the radius the penalty is
    negligible; once an atom crosses the radius the softplus turns linear, so
    the inward force saturates at ``k_wall * beta`` rather than diverging --
    a soft, leak-proof boundary for keeping fragments from drifting apart. This
    is the log-Fermi restraint from Grimme's metadynamics-style biasing
    (10.1021/acs.jctc.9b00143); the default parameters follow that reference.

    Forces are computed analytically, so the model is safe to call under
    ``torch.no_grad()``.

    Args:
        radius: Wall radius in Angstrom. Defaults to 10.0.
        k_wall: Wall strength in Hartree (Grimme's units, converted to eV
            internally). Defaults to 0.019.
        beta: Wall steepness in 1/Bohr (converted to 1/Angstrom internally).
            Defaults to 10.0.
        center: Wall center with shape [3] (shared by all systems) or
            [n_systems, 3]. Defaults to the origin.
        energy_label: Non-canonical output key under which this wall's
            per-system energy is also reported, letting it survive SumModel
            and land on the SimState as an extra (e.g. ``state.PE_FermiWall``).
            Defaults to "PE_FermiWall"; give distinct labels if composing
            several walls.
        device: Device for computations. Defaults to CPU.
        dtype: Floating-point dtype. Defaults to torch.float64.

    Example::

        wall = LogfermiWall(radius=8.0, device=model.device, dtype=model.dtype)
        confined_model = SumModel(model, wall)
    """

    def __init__(
        self,
        radius: float = 10.0,
        k_wall: float = 0.019,
        beta: float = 10.0,
        center: torch.Tensor | None = None,
        energy_label: str = "PE_FermiWall",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initialize the log-Fermi wall."""
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = False
        self._memory_scales_with = "n_atoms"

        self.energy_label = str(energy_label)
        self.radius = float(radius)
        self.k_wall = float(k_wall) * UnitConversion.Hartree_to_eV
        self.beta = float(beta) * UnitConversion.Ang_to_Bohr  # 1/Bohr -> 1/Ang
        center = None if center is None else center.to(self._device, self._dtype)
        self.center: torch.Tensor | None
        self.register_buffer("center", center)

    def forward(self, state: SimState, **_kwargs) -> dict[str, torch.Tensor]:
        """Compute wall energies and forces.

        Args:
            state: Simulation state with positions [n_atoms, 3] and system_idx.
            **_kwargs: Unused, accepted for interface compatibility.

        Returns:
            Dict with "energy" [n_systems], "forces" [n_atoms, 3], and the
            same wall energy under ``self.energy_label`` (a non-canonical key
            so it survives SumModel and is stored on the SimState as a
            per-system extra).
        """
        positions = state.positions
        system_idx = state.system_idx

        if self.center is None:
            dvec = positions
        elif self.center.ndim == 1:
            dvec = positions - self.center
        else:
            dvec = positions - self.center[system_idx]

        r = torch.norm(dvec, dim=-1) + EPS  # (n_atoms,)
        x = self.beta * (r - self.radius)
        v_atom = self.k_wall * torch.nn.functional.softplus(x)  # log1p(e^x), no overflow
        energy = _segment_sum(v_atom, system_idx, state.n_systems)

        # F = -dV/dr * r_hat with dV/dr = k_wall * beta * sigmoid(x)
        dv_dr = self.k_wall * self.beta * torch.sigmoid(x)
        forces = -dv_dr.unsqueeze(-1) * dvec / r.unsqueeze(-1)

        return {"energy": energy, "forces": forces, self.energy_label: energy}


class RMSDCV(ModelInterface):
    """History-dependent Gaussian bias on an RMSD collective variable.

    Discourages the system from revisiting earlier geometries by summing a
    Gaussian repulsion over a rolling set of stored reference structures,
    ``E = k_push * sum_i exp(-alpha * d2_i)`` per system, where ``d2_i`` is the
    mean-square displacement from reference *i* after a least-squares (Kabsch)
    rotation that removes rigid-body orientation, averaged over the biased atoms
    and Cartesian components. As references accumulate along the trajectory the
    bias fills the basins already visited and steers the dynamics toward new
    configurations. The functional form and default parameters follow Grimme's
    RMSD-based metadynamics (10.1021/acs.jctc.9b00143).

    The references are held in a :class:`~torch_sim.enhanced_sampling.history.History`
    buffer, which owns the deposition cadence and capacity limit. The buffer is
    seeded on the first call (which returns zero bias), and a new reference is
    deposited every ``update_interval`` calls. TorchSim integrators evaluate the
    model once per MD step (plus once at initialization), so calls correspond to
    MD steps. Use :meth:`push_reference` and :meth:`reset` for manual control of
    the buffer.

    Forces are obtained by autograd through the alignment, so backprop through
    the SVD requires non-degenerate singular values (generic for molecular
    geometries). Because the buffer is shaped to the batch seen first, the
    model must be re-:meth:`reset` before reuse with a different batch.

    Args:
        k_push: Bias strength in Hartree (converted to eV internally).
            Defaults to 1.0.
        alpha_width: Gaussian width in 1/Bohr^2 (converted to 1/Angstrom^2
            internally). Defaults to 1.0.
        n_refs: Maximum number of stored references; oldest are dropped.
            Defaults to 10.
        update_interval: Deposit a new reference every this many calls.
            Defaults to 1.
        atom_mask: Boolean mask with shape [n_atoms] over the concatenated
            atoms selecting which atoms participate in the CV (True =
            biased). Excluded atoms feel no bias force. Defaults to all atoms.
        energy_label: Non-canonical output key under which this bias's
            per-system energy is also reported, letting it survive SumModel
            and land on the SimState as an extra (e.g. ``state.PE_RMSDCV``).
            Defaults to "PE_RMSDCV"; give distinct labels if composing
            several biases.
        device: Device for computations. Defaults to CPU.
        dtype: Floating-point dtype. Defaults to torch.float64.

    Example::

        bias = RMSDCV(k_push=0.02, alpha_width=1.2, n_refs=20, update_interval=50)
        metad_model = SumModel(mace_model, bias)
    """

    def __init__(
        self,
        k_push: float = 1.0,
        alpha_width: float = 1.0,
        n_refs: int = 10,
        update_interval: int = 1,
        atom_mask: torch.Tensor | None = None,
        energy_label: str = "PE_RMSDCV",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initialize the RMSD collective-variable bias."""
        super().__init__()
        if n_refs < 1:
            raise ValueError(f"{n_refs=} must be >= 1")
        if update_interval < 1:
            raise ValueError(f"{update_interval=} must be >= 1")
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = False
        self._memory_scales_with = "n_atoms"

        self.energy_label = str(energy_label)
        self.k_push = float(k_push) * UnitConversion.Hartree_to_eV
        self.alpha = float(alpha_width) * UnitConversion.Ang_to_Bohr**2
        self.n_refs = int(n_refs)
        self.update_interval = int(update_interval)
        atom_mask = None if atom_mask is None else atom_mask.to(self._device, torch.bool)
        self.atom_mask: torch.Tensor | None
        self.register_buffer("atom_mask", atom_mask)

        # rolling buffer of centered references, each entry (n_biased_atoms, 3)
        self.history = History(
            capacity=self.n_refs,
            stride=self.update_interval,
            device=self._device,
            dtype=self._dtype,
        )

    @property
    def ref_buf(self) -> torch.Tensor | None:
        """Stored references with shape (n_stored, n_biased_atoms, 3), or None."""
        return self.history.stack()

    def reset(self) -> None:
        """Clear all stored references and the internal deposition counter."""
        self.history.reset()

    @torch.no_grad()
    def push_reference(self, state: SimState) -> None:
        """Deposit the current configuration as a new reference.

        Args:
            state: Simulation state whose (masked) positions are stored.
        """
        self.history.push(self._centered(state))

    def _centered(self, state: SimState) -> torch.Tensor:
        """Return the biased atoms' positions with each system's COM removed."""
        positions, system_idx = self._masked(state)
        counts = torch.bincount(system_idx, minlength=state.n_systems)
        com = _segment_sum(positions, system_idx, state.n_systems)
        com = com / counts.unsqueeze(-1)
        return positions - com[system_idx]

    def _masked(self, state: SimState) -> tuple[torch.Tensor, torch.Tensor]:
        """Return positions and system indices of biased atoms only."""
        if self.atom_mask is None:
            return state.positions, state.system_idx
        return state.positions[self.atom_mask], state.system_idx[self.atom_mask]

    @staticmethod
    def _kabsch(
        rc: torch.Tensor, qc: torch.Tensor, system_idx: torch.Tensor, n_systems: int
    ) -> torch.Tensor:
        """Batched Kabsch rotations aligning current coords to many references.

        Args:
            rc: Centered current coordinates with shape [n_biased, 3].
            qc: Centered reference coordinates with shape [n_refs, n_biased, 3].
            system_idx: System index of each biased atom with shape [n_biased].
            n_systems: Number of systems in the batch.

        Returns:
            Rotation matrices with shape [n_refs, n_systems, 3, 3].
        """
        outer = rc.unsqueeze(0).unsqueeze(-1) * qc.unsqueeze(-2)  # (X, n, 3, 3)
        cov = _segment_sum(outer, system_idx, n_systems, dim=1)  # (X, M, 3, 3)
        u_mat, _, vh_mat = torch.linalg.svd(cov)
        v_mat = vh_mat.transpose(-2, -1)
        ut_mat = u_mat.transpose(-2, -1)
        det = torch.linalg.det(v_mat @ ut_mat)  # (X, n_systems)
        corr = torch.eye(3, device=cov.device, dtype=cov.dtype).expand_as(cov).clone()
        corr[..., 2, 2] = torch.where(det < 0, -1.0, 1.0)
        return v_mat @ corr @ ut_mat

    def forward(self, state: SimState, **_kwargs) -> dict[str, torch.Tensor]:
        """Compute bias energies and forces, depositing references as needed.

        Args:
            state: Simulation state with positions [n_atoms, 3] and system_idx.
            **_kwargs: Unused, accepted for interface compatibility.

        Returns:
            Dict with "energy" [n_systems], "forces" [n_atoms, 3], and the
            same bias energy under ``self.energy_label`` (a non-canonical key
            so it survives SumModel and is stored on the SimState as a
            per-system extra).
        """
        n_systems = state.n_systems

        if self.history.is_empty:
            self.push_reference(state)
            zero_energy = torch.zeros(
                n_systems, device=state.positions.device, dtype=state.positions.dtype
            )
            return {
                "energy": zero_energy,
                "forces": torch.zeros_like(state.positions),
                self.energy_label: zero_energy,
            }

        masked_pos, system_idx = self._masked(state)
        counts = torch.bincount(system_idx, minlength=n_systems)

        with torch.enable_grad():
            pos = masked_pos.detach().requires_grad_(requires_grad=True)  # (n_biased, 3)
            qc = self.history.stack().to(pos)  # (X, n_biased, 3), already centered

            com = _segment_sum(pos, system_idx, n_systems) / counts.unsqueeze(-1)
            rc = pos - com[system_idx]

            rot = self._kabsch(rc, qc, system_idx, n_systems)  # (X, M, 3, 3)
            rc_rot = torch.einsum("xncd,nd->xnc", rot[:, system_idx], rc)
            diff = rc_rot - qc
            # squared deviation averaged over atoms AND coords (matches the xtb input
            # convention; alpha absorbs the factor 3 vs the conventional RMSD^2)
            sq = diff.pow(2).sum(dim=-1)  # (X, n_biased)
            rmsd2 = _segment_sum(sq, system_idx, n_systems, dim=1) / (3 * counts)

            energy = self.k_push * torch.exp(-self.alpha * rmsd2).sum(dim=0)  # (M,)
            grad = torch.autograd.grad(energy.sum(), pos)[0]

        forces = torch.zeros_like(state.positions)
        if self.atom_mask is None:
            forces = -grad
        else:
            forces[self.atom_mask] = -grad

        self.history.maybe_push(self._centered(state))

        detached_energy = energy.detach()
        return {
            "energy": detached_energy,
            "forces": forces.detach(),
            self.energy_label: detached_energy,
        }
