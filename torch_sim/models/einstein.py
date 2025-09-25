"""Einstein model where each atom is treated as an independent 3D harmonic oscillator.

Contrary to other models, the model energies depend on an absolute reference position,
so the model can only be used on systems that the model was initialized with.
As a analytical model, it can provide its Helmholtz free energy and can also generate
samples from the Boltzmann distribution at a given temperature.
"""

import torch

import torch_sim as ts
from torch_sim import SimState, units
from torch_sim.models.interface import ModelInterface


class EinsteinModel(ModelInterface):
    """Einstein model where each atom is treated as an independent 3D harmonic oscillator.
    Each atom has its own frequency.

    For this model:
    E  = sum_i 0.5 * k_i * (x_i - x0_i)^2
    F  = -k_i * (x_i - x0_i)
    k_i = m_i * omega_i^2

    For best results, frequencies should be in the range of typical phonon frequencies.
    They can be set for each atom type individually following energy balance from
    a NVT simulation. From equipartition theorem:
    <E_pot> = 3/2 k_B T
    => omega = sqrt(3 k_B T / m <x^2>)
    """

    def __init__(
        self,
        equilibrium_position: torch.Tensor,  # shape [N, 3]
        frequencies: torch.Tensor,  # shape [N]
        system_idx: torch.Tensor | None = None,  # shape [N] or None
        masses: torch.Tensor | None = None,  # shape [N] or None
        reference_energy: float = 0.0,  # reference energy value
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        compute_forces: bool = True,
        compute_stress: bool = False,
    ) -> None:
        """Initialize the Einstein model.

        Args:
            equilibrium_position: Tensor of shape [N, 3] with equilibrium positions.
            frequencies: Tensor of shape [N] with frequencies for each atom
                (same frequency in all 3 directions).
            system_idx: Optional tensor of shape [N] with system indices for each atom.
                If None, all atoms are assumed to belong to the same system.
            masses: Optional tensor of shape [N] with masses for each atom.
                If None, all masses are set to 1.
            reference_energy: Reference energy value to add to the computed energy.
            device: Device to use for the model (default: CPU).
            dtype: Data type for the model (default: torch.float32).
            compute_forces: Whether to compute forces in the model.
            compute_stress: Whether to compute stress in the model.

        """
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress

        equilibrium_position = torch.as_tensor(
            equilibrium_position, device=self._device, dtype=self._dtype
        )
        frequencies = torch.as_tensor(
            frequencies, device=self._device, dtype=self._dtype
        )  # [N, 3]

        if frequencies.shape[0] != equilibrium_position.shape[0]:
            raise ValueError("frequencies shape must match equilibrium_position shape")
        if frequencies.min() < 0:
            raise ValueError("frequencies must be non-negative")
        if frequencies.ndim == 0:
            frequencies = frequencies.unsqueeze(0)
        if frequencies.ndim != 1:
            raise ValueError("frequencies must be a 1D tensor")

        if masses is None:
            masses = torch.ones(
                equilibrium_position.shape[0], dtype=self._dtype, device=self._device
            )
        else:
            masses = masses.to(self._device, self._dtype)

        if system_idx is not None:
            system_idx = system_idx.to(self._device)
        else:
            system_idx = torch.zeros(
                equilibrium_position.shape[0], dtype=torch.long, device=self._device
            )

        self.register_buffer("system_idx", system_idx.to(self._device))
        self.register_buffer("masses", masses)  # [N]
        self.register_buffer("x0", equilibrium_position)  # [N, 3]
        self.register_buffer("frequencies", frequencies)  # [N]
        self.register_buffer(
            "reference_energy",
            torch.tensor(reference_energy, dtype=self._dtype, device=self._device),
        )

    @classmethod
    def from_atom_and_frequencies(
        cls,
        atom: SimState,
        frequencies: torch.Tensor | float,
        *,
        reference_energy: float = 0.0,
        compute_forces: bool = True,
        compute_stress: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "EinsteinModel":
        """Create an EinsteinModel from an ASE Atoms object and frequencies.

        Args:
            atom: ASE Atoms object containing the reference structure.
            frequencies: Tensor of shape [N] with frequencies for each atom
                (same frequency in all 3 directions) or a scalar.
            reference_energy: Reference energy value.
            compute_forces: Whether to compute forces in the model.
            compute_stress: Whether to compute stress in the model.
            device: Device to use for the model (default: CPU).
            dtype: Data type for the model (default: torch.float32).

        Returns:
            EinsteinModel: An instance of the EinsteinModel.
        """
        # Get equilibrium positions from the atoms object
        equilibrium_position = atom.positions.clone().to(dtype=dtype, device=device)

        frequencies = torch.as_tensor(frequencies, dtype=dtype, device=device)
        if frequencies.ndim == 0:
            frequencies = frequencies.repeat(atom.positions.shape[0])
        if frequencies.shape[0] != atom.positions.shape[0]:
            raise ValueError(
                "frequencies must be a scalar or a tensor of shape [N] "
                "where N is the number of atoms"
            )

        # Create and return an instance of EinsteinModel
        return cls(
            equilibrium_position=equilibrium_position,
            frequencies=frequencies,
            masses=atom.masses,
            system_idx=atom.system_idx,
            reference_energy=reference_energy,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
            device=device,
            dtype=dtype,
        )

    def forward(self, state: ts.SimState) -> dict[str, torch.Tensor]:
        """Calculate energies and forces for the Einstein model.

        Args:
            state: SimState or StateDict containing positions, cell, etc.

        Returns:
            Dictionary containing energy, forces
        """
        pos = state.positions.to(self._dtype)  # [N, 3]
        cell = state.cell.to(self._dtype)

        if cell.ndim == 2:
            cell = cell.unsqueeze(0)  # [1, 3, 3]

        # Get model parameters
        x0 = torch.as_tensor(self.x0, dtype=self._dtype, device=self._device)
        frequencies = torch.as_tensor(
            self.frequencies, dtype=self._dtype, device=self._device
        )
        masses = torch.as_tensor(self.masses, dtype=self._dtype, device=self._device)

        # Calculate displacements using periodic boundary conditions
        if cell.shape[0] == 1:
            disp = ts.transforms.minimum_image_displacement(
                dr=pos - x0, cell=cell[0], pbc=state.pbc
            )
        else:
            disp = ts.transforms.minimum_image_displacement_batched(
                pos - x0, cell, system_idx=state.system_idx, pbc=state.pbc
            )

        # Spring constants: k = m * omega^2
        spring_constants = masses * (frequencies**2)  # [N]

        # Energy: E = 0.5 * k * x^2
        energies_per_mode = 0.5 * spring_constants * ((disp**2).sum(dim=1))  # [N]
        total_energy = torch.zeros(
            state.n_systems, dtype=self._dtype, device=self._device
        )
        total_energy.scatter_add_(0, state.system_idx, energies_per_mode)
        total_energy += self.reference_energy

        # Forces: F = -k * x
        forces = -spring_constants.unsqueeze(-1) * disp  # [N, 3]

        results = {
            "energy": total_energy,
            "forces": forces,
        }
        # Stress is not implemented for this model
        if self._compute_stress:
            results["stress"] = torch.zeros(
                (state.n_systems, 3, 3), dtype=self._dtype, device=self._device
            )

        return results

    def get_free_energy(self, temperature: float) -> dict[str, torch.Tensor]:
        """Compute free energy at a given temperature using Einstein model.

        Args:
            temperature: Temperature in Kelvin.

        Returns:
            Dictionary containing heat capacity, entropy, and free energy.
        """
        # Boltzmann constant in eV/K
        kB = units.BaseConstant.k_B / units.UnitConversion.eV_to_J
        T = temperature
        # Reduced Planck constant in eV*s
        hbar = units.BaseConstant.h_planck / (2 * units.pi * units.UnitConversion.eV_to_J)

        frequencies_tensor = (
            torch.as_tensor(self.frequencies).clone()
            * torch.as_tensor(
                units.UnitConversion.eV_to_J / units.BaseConstant.amu
            ).sqrt()
            / units.UnitConversion.Ang_to_met
        )  # Convert to rad/s
        free_energy_per_atom = (
            -3 * kB * T * torch.log(kB * T / (hbar * frequencies_tensor))
        )

        n_systems = self.system_idx.max().item() + 1
        free_energy_per_system = torch.zeros(
            n_systems, dtype=self._dtype, device=self._device
        )
        free_energy_per_system.scatter_add_(0, self.system_idx, free_energy_per_atom)

        return {"free_energy": free_energy_per_system}

    def sample(self, state: SimState, temperature: float) -> SimState:
        """Generate samples from the Einstein model at a given temperature.

        Args:
            state: Initial simulation state to sample from.
            temperature: Temperature in Kelvin.

        Returns:
            SimState containing sampled positions and velocities.

        The Boltzmann distribution for a harmonic oscillator leads to Gaussian
        distributions
        for both positions and velocities.
        """
        N = self.x0.shape[0]
        kB = units.BaseConstant.k_B / units.UnitConversion.eV_to_J
        beta = 1.0 / (kB * temperature)  # Inverse temperature in 1/eV

        # Sample positions from a normal distribution around equilibrium positions
        stddev = torch.sqrt(1.0 / (self.masses * (self.frequencies**2) * beta)).unsqueeze(
            -1
        )
        sampled_positions = self.x0 + torch.randn(N, 3, device=self._device) * stddev
        state = state.clone()
        state.positions = sampled_positions
        return state
