"""Testing utilities for torch-sim models.

This module provides reusable testing functions and SimState generators that can be
used to validate model implementations. These are designed to work both within
torch-sim's test suite and in external repositories that implement ModelInterface
models.

Example usage in another repo::

    import pytest
    import torch
    from torch_sim.testing import (
        assert_model_calculator_consistency,
        SIMSTATE_GENERATORS,
        CONSISTENCY_SIMSTATES,
    )

    DEVICE = torch.device("cpu")
    DTYPE = torch.float64


    @pytest.mark.parametrize("sim_state_name", CONSISTENCY_SIMSTATES)
    def test_my_model_consistency(sim_state_name, my_model, my_calculator):
        sim_state = SIMSTATE_GENERATORS[sim_state_name](DEVICE, DTYPE)
        assert_model_calculator_consistency(my_model, my_calculator, sim_state)
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Final

import torch

import torch_sim as ts
from torch_sim.elastic import full_3x3_to_voigt_6_stress


if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

    from torch_sim.models.interface import ModelInterface


def make_cu_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create crystalline copper (FCC)."""
    from ase.build import bulk

    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    return ts.io.atoms_to_state(atoms, device, dtype)


def make_mg_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create crystalline magnesium (HCP)."""
    from ase.build import bulk

    atoms = bulk("Mg", "hcp", a=3.17, c=5.14)
    return ts.io.atoms_to_state(atoms, device, dtype)


def make_sb_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create crystalline antimony (rhombohedral)."""
    from ase.build import bulk

    atoms = bulk("Sb", "rhombohedral", a=4.58, alpha=60)
    return ts.io.atoms_to_state(atoms, device, dtype)


def make_ti_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create crystalline titanium (HCP)."""
    from ase.build import bulk

    atoms = bulk("Ti", "hcp", a=2.94, c=4.64)
    return ts.io.atoms_to_state(atoms, device, dtype)


def make_tio2_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create crystalline TiO2 (rutile)."""
    from ase.spacegroup import crystal

    a, c = 4.60, 2.96
    basis = [("Ti", 0.5, 0.5, 0), ("O", 0.695679, 0.695679, 0.5)]
    atoms = crystal(
        symbols=[b[0] for b in basis],
        basis=[b[1:] for b in basis],
        spacegroup=136,
        cellpar=[a, a, c, 90, 90, 90],
    )
    return ts.io.atoms_to_state(atoms, device, dtype)


def make_ga_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create crystalline Ga (Cmce)."""
    from ase.spacegroup import crystal

    a, b, c = 4.43, 7.60, 4.56
    basis = [("Ga", 0, 0.344304, 0.415401)]
    atoms = crystal(
        symbols=[ba[0] for ba in basis],
        basis=[ba[1:] for ba in basis],
        spacegroup=64,
        cellpar=[a, b, c, 90, 90, 90],
    )
    return ts.io.atoms_to_state(atoms, device, dtype)


def make_niti_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create crystalline NiTi (monoclinic)."""
    from ase.spacegroup import crystal

    a, b, c = 2.89, 3.97, 4.83
    alpha, beta, gamma = 90.00, 105.23, 90.00
    basis = [
        ("Ni", 0.369548, 0.25, 0.217074),
        ("Ti", 0.076622, 0.25, 0.671102),
    ]
    atoms = crystal(
        symbols=[ba[0] for ba in basis],
        basis=[ba[1:] for ba in basis],
        spacegroup=11,
        cellpar=[a, b, c, alpha, beta, gamma],
    )
    return ts.io.atoms_to_state(atoms, device, dtype)


def make_si_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create crystalline silicon (diamond)."""
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    return ts.io.atoms_to_state(atoms, device, dtype)


def make_sio2_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create alpha-quartz SiO2."""
    from ase.spacegroup import crystal

    atoms = crystal(
        symbols=["O", "Si"],
        basis=[[0.413, 0.2711, 0.2172], [0.4673, 0, 0.3333]],
        spacegroup=152,
        cellpar=[4.9019, 4.9019, 5.3988, 90, 90, 120],
    )
    return ts.io.atoms_to_state(atoms, device, dtype)


def _rattle_sim_state(sim_state: ts.SimState, seed: int = 3) -> ts.SimState:
    """Apply Weibull-distributed random displacements to positions."""
    sim_state = sim_state.clone()
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        weibull = torch.distributions.weibull.Weibull(scale=0.1, concentration=1)
        rnd = torch.randn_like(sim_state.positions)
        rnd = rnd / torch.norm(rnd, dim=-1, keepdim=True)
        shifts = weibull.sample(rnd.shape).to(device=sim_state.positions.device) * rnd
        sim_state.positions = sim_state.positions + shifts
    finally:
        torch.random.set_rng_state(rng_state)
    return sim_state


def make_rattled_si_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create rattled silicon."""
    return _rattle_sim_state(make_si_sim_state(device, dtype))


def make_rattled_sio2_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create rattled alpha-quartz SiO2."""
    return _rattle_sim_state(make_sio2_sim_state(device, dtype))


def make_ar_supercell_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create FCC Argon 2x2x2 supercell."""
    from ase.build import bulk

    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat([2, 2, 2])
    return ts.io.atoms_to_state(atoms, device, dtype)


def make_fe_supercell_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create FCC iron 4x4x4 supercell."""
    from ase.build import bulk

    atoms = bulk("Fe", "fcc", a=5.26, cubic=True).repeat([4, 4, 4])
    return ts.io.atoms_to_state(atoms, device, dtype)


def make_casio3_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create CaSiO3 (wollastonite)."""
    from ase.spacegroup import crystal

    a, b, c = 7.9258, 7.3202, 7.0653
    alpha, beta, gamma = 90.055, 95.217, 103.426
    basis = [
        ("Ca", 0.19831, 0.42266, 0.76060),
        ("Ca", 0.20241, 0.92919, 0.76401),
        ("Ca", 0.50333, 0.75040, 0.52691),
        ("Si", 0.1851, 0.3875, 0.2684),
        ("Si", 0.1849, 0.9542, 0.2691),
        ("Si", 0.3973, 0.7236, 0.0561),
        ("O", 0.3034, 0.4616, 0.4628),
        ("O", 0.3014, 0.9385, 0.4641),
        ("O", 0.5705, 0.7688, 0.1988),
        ("O", 0.9832, 0.3739, 0.2655),
        ("O", 0.9819, 0.8677, 0.2648),
        ("O", 0.4018, 0.7266, 0.8296),
        ("O", 0.2183, 0.1785, 0.2254),
        ("O", 0.2713, 0.8704, 0.0938),
        ("O", 0.2735, 0.5126, 0.0931),
    ]
    atoms = crystal(
        symbols=[ba[0] for ba in basis],
        basis=[ba[1:] for ba in basis],
        spacegroup=2,
        cellpar=[a, b, c, alpha, beta, gamma],
    )
    return ts.io.atoms_to_state(atoms, device, dtype)


def make_benzene_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create benzene molecule (non-periodic)."""
    from ase.build import molecule

    atoms = molecule("C6H6")
    return ts.io.atoms_to_state(atoms, device, dtype)


def make_osn2_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create rhombohedral OsN2."""
    from pymatgen.core import Lattice, Structure

    a = 3.211996
    lattice = Lattice.from_parameters(a, a, a, 60, 60, 60)
    species = ["Os", "N"]
    frac_coords = [[0.75, 0.7501, -0.25], [0, 0, 0]]
    structure = Structure(lattice, species, frac_coords, coords_are_cartesian=False)
    return ts.initialize_state(structure, dtype=dtype, device=device)


def make_distorted_fcc_al_conventional_sim_state(
    device: torch.device | None = None, dtype: torch.dtype | None = None
) -> ts.SimState:
    """Create a slightly distorted FCC Al conventional cell (4 atoms)."""
    import numpy as np
    from ase.build import bulk

    atoms_fcc = bulk("Al", crystalstructure="fcc", a=4.05, cubic=True)

    strain_matrix = np.array([[1.0, 0.05, -0.03], [0.04, 1.0, 0.06], [-0.02, 0.03, 1.0]])
    original_cell = atoms_fcc.get_cell()
    new_cell = original_cell @ strain_matrix.T
    atoms_fcc.set_cell(new_cell, scale_atoms=True)

    positions = atoms_fcc.get_positions()
    np_rng = np.random.default_rng(seed=42)
    positions += np_rng.normal(scale=0.01, size=positions.shape)
    atoms_fcc.positions = positions

    return ts.io.atoms_to_state(atoms_fcc, device=device, dtype=dtype)


# Generator type alias
SimStateGenerator = Callable[[torch.device, torch.dtype], ts.SimState]

# Dict mapping names to generator functions
SIMSTATE_BULK_GENERATORS: Final[dict[str, SimStateGenerator]] = {
    "cu_sim_state": make_cu_sim_state,
    "mg_sim_state": make_mg_sim_state,
    "sb_sim_state": make_sb_sim_state,
    "tio2_sim_state": make_tio2_sim_state,
    "ga_sim_state": make_ga_sim_state,
    "niti_sim_state": make_niti_sim_state,
    "ti_sim_state": make_ti_sim_state,
    "si_sim_state": make_si_sim_state,
    "rattled_si_sim_state": make_rattled_si_sim_state,
    "sio2_sim_state": make_sio2_sim_state,
    "rattled_sio2_sim_state": make_rattled_sio2_sim_state,
    "ar_supercell_sim_state": make_ar_supercell_sim_state,
    "fe_supercell_sim_state": make_fe_supercell_sim_state,
    "casio3_sim_state": make_casio3_sim_state,
    "osn2_sim_state": make_osn2_sim_state,
    "distorted_fcc_al_conventional_sim_state": (
        make_distorted_fcc_al_conventional_sim_state
    ),
}

SIMSTATE_MOLECULE_GENERATORS: Final[
    dict[str, Callable[[torch.device, torch.dtype], ts.SimState]]
] = {
    "benzene_sim_state": make_benzene_sim_state,
}


SIMSTATE_GENERATORS: Final[dict[str, SimStateGenerator]] = {
    **SIMSTATE_BULK_GENERATORS,
    **SIMSTATE_MOLECULE_GENERATORS,
}

# Tuple of names for backward compat / parametrize usage
CONSISTENCY_SIMSTATES: Final[tuple[str, ...]] = tuple(SIMSTATE_GENERATORS.keys())


def assert_model_calculator_consistency(
    model: "ModelInterface",
    calculator: "Calculator",
    sim_state: ts.SimState,
    energy_rtol: float = 1e-5,
    energy_atol: float = 1e-5,
    force_rtol: float = 1e-5,
    force_atol: float = 1e-5,
    stress_rtol: float = 1e-5,
    stress_atol: float = 1e-5,
) -> None:
    """Assert consistency between model and calculator implementations.

    This function validates that a ModelInterface implementation produces
    the same results as an ASE Calculator implementation for a given
    simulation state. It compares energies, forces, and optionally stresses.

    Args:
        model: ModelInterface instance to test
        calculator: ASE Calculator instance to compare against
        sim_state: Simulation state to test with
        energy_rtol: Relative tolerance for energy comparisons
        energy_atol: Absolute tolerance for energy comparisons
        force_rtol: Relative tolerance for force comparisons
        force_atol: Absolute tolerance for force comparisons
        stress_rtol: Relative tolerance for stress comparisons
        stress_atol: Absolute tolerance for stress comparisons

    Raises:
        AssertionError: If model and calculator results don't match within tolerances
    """
    atoms = ts.io.state_to_atoms(sim_state)[0]
    atoms.calc = calculator

    model_results = model(sim_state)

    calc_forces = torch.tensor(
        atoms.get_forces(),
        device=sim_state.positions.device,
        dtype=model_results["forces"].dtype,
    )

    torch.testing.assert_close(
        model_results["energy"].item(),
        atoms.get_potential_energy(),
        rtol=energy_rtol,
        atol=energy_atol,
    )
    torch.testing.assert_close(
        model_results["forces"],
        calc_forces,
        rtol=force_rtol,
        atol=force_atol,
    )

    if "stress" in model_results:
        calc_stress = torch.tensor(
            atoms.get_stress(),
            device=sim_state.positions.device,
            dtype=model_results["stress"].dtype,
        ).unsqueeze(0)

        torch.testing.assert_close(
            full_3x3_to_voigt_6_stress(model_results["stress"]),
            calc_stress,
            rtol=stress_rtol,
            atol=stress_atol,
            equal_nan=True,
        )
