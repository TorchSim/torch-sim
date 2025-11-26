from typing import get_args

import pytest
import torch

import torch_sim as ts
from tests.conftest import DTYPE
from torch_sim.constraints import (
    Constraint,
    FixAtoms,
    FixCom,
    merge_constraints,
    validate_constraints,
)
from torch_sim.models.interface import ModelInterface
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.optimizers import FireFlavor
from torch_sim.transforms import get_centers_of_mass
from torch_sim.units import MetalUnits


def test_fix_com(ar_supercell_sim_state: ts.SimState, lj_model: LennardJonesModel):
    """Test adjustment of positions and momenta with FixCom constraint."""
    ar_supercell_sim_state.constraints = [FixCom([0])]
    initial_positions = ar_supercell_sim_state.positions.clone()
    ar_supercell_sim_state.set_positions(initial_positions + 0.5)
    assert torch.allclose(ar_supercell_sim_state.positions, initial_positions, atol=1e-8)

    ar_supercell_md_state = ts.nve_init(
        state=ar_supercell_sim_state,
        model=lj_model,
        kT=torch.tensor(10.0, dtype=DTYPE),
        seed=42,
    )
    ar_supercell_md_state.set_momenta(
        torch.randn_like(ar_supercell_md_state.momenta) * 0.1
    )
    assert torch.allclose(
        ar_supercell_md_state.momenta.mean(dim=0),
        torch.zeros(3, dtype=DTYPE),
        atol=1e-8,
    )


def test_fix_atoms(ar_supercell_sim_state: ts.SimState, lj_model: LennardJonesModel):
    """Test adjustment of positions and momenta with FixAtoms constraint."""
    indices_to_fix = torch.tensor([0, 5, 10], dtype=torch.long)
    ar_supercell_sim_state.constraints = [FixAtoms(atom_idx=indices_to_fix)]
    initial_positions = ar_supercell_sim_state.positions.clone()
    # displacement = torch.randn_like(ar_supercell_sim_state.positions) * 0.5
    displacement = 0.5
    ar_supercell_sim_state.set_positions(initial_positions + displacement)
    assert torch.allclose(
        ar_supercell_sim_state.positions[indices_to_fix],
        initial_positions[indices_to_fix],
        atol=1e-8,
    )
    # Check that other positions have changed
    unfixed_indices = torch.tensor(
        [i for i in range(ar_supercell_sim_state.n_atoms) if i not in indices_to_fix],
        dtype=torch.long,
    )
    assert not torch.allclose(
        ar_supercell_sim_state.positions[unfixed_indices],
        initial_positions[unfixed_indices],
        atol=1e-8,
    )

    ar_supercell_md_state = ts.nve_init(
        state=ar_supercell_sim_state,
        model=lj_model,
        kT=torch.tensor(10.0, dtype=DTYPE),
        seed=42,
    )
    ar_supercell_md_state.set_momenta(
        torch.randn_like(ar_supercell_md_state.momenta) * 0.1
    )
    assert torch.allclose(
        ar_supercell_md_state.momenta[indices_to_fix],
        torch.zeros_like(ar_supercell_md_state.momenta[indices_to_fix]),
        atol=1e-8,
    )


def test_fix_com_nvt_langevin(cu_sim_state: ts.SimState, lj_model: LennardJonesModel):
    """Test FixCom constraint in NVT Langevin dynamics."""
    n_steps = 1000
    dt = torch.tensor(0.001, dtype=DTYPE)
    kT = torch.tensor(300, dtype=DTYPE) * MetalUnits.temperature

    dofs_before = cu_sim_state.get_number_of_degrees_of_freedom()
    cu_sim_state.constraints = [FixCom([0])]
    assert torch.allclose(
        cu_sim_state.get_number_of_degrees_of_freedom(), dofs_before - 3
    )

    state = ts.nvt_langevin_init(state=cu_sim_state, model=lj_model, kT=kT, seed=42)
    positions = []
    system_masses = torch.zeros((state.n_systems, 1), dtype=DTYPE).scatter_add_(
        0,
        state.system_idx.unsqueeze(-1).expand(-1, 1),
        state.masses.unsqueeze(-1),
    )
    temperatures = []
    for _step in range(n_steps):
        state = ts.nvt_langevin_step(model=lj_model, state=state, dt=dt, kT=kT)
        positions.append(state.positions.clone())
        temp = ts.calc_kT(
            masses=state.masses,
            momenta=state.momenta,
            system_idx=state.system_idx,
            dof_per_system=state.get_number_of_degrees_of_freedom(),
        )
        temperatures.append(temp / MetalUnits.temperature)
    temperatures = torch.stack(temperatures)

    traj_positions = torch.stack(positions)

    coms = torch.zeros((n_steps, state.n_systems, 3), dtype=DTYPE).scatter_add_(
        1,
        state.system_idx[None, :, None].expand(n_steps, -1, 3),
        state.masses.unsqueeze(-1) * traj_positions,
    )
    coms /= system_masses
    coms_drift = coms - coms[0]
    assert torch.allclose(coms_drift, torch.zeros_like(coms_drift), atol=1e-6)
    assert (torch.mean(temperatures[len(temperatures) // 2 :]) - 300) / 300 < 0.30


def test_fix_atoms_nvt_langevin(cu_sim_state: ts.SimState, lj_model: LennardJonesModel):
    """Test FixAtoms constraint in NVT Langevin dynamics."""
    n_steps = 1000
    dt = torch.tensor(0.001, dtype=DTYPE)
    kT = torch.tensor(300, dtype=DTYPE) * MetalUnits.temperature

    dofs_before = cu_sim_state.get_number_of_degrees_of_freedom()
    cu_sim_state.constraints = [FixAtoms(atom_idx=torch.tensor([0, 1], dtype=torch.long))]
    assert torch.allclose(
        cu_sim_state.get_number_of_degrees_of_freedom(), dofs_before - torch.tensor([6])
    )
    state = ts.nvt_langevin_init(state=cu_sim_state, model=lj_model, kT=kT, seed=42)
    positions = []
    temperatures = []
    for _step in range(n_steps):
        state = ts.nvt_langevin_step(model=lj_model, state=state, dt=dt, kT=kT)
        positions.append(state.positions.clone())
        temp = ts.calc_kT(
            masses=state.masses,
            momenta=state.momenta,
            system_idx=state.system_idx,
            dof_per_system=state.get_number_of_degrees_of_freedom(),
        )
        temperatures.append(temp / MetalUnits.temperature)
    temperatures = torch.stack(temperatures)
    traj_positions = torch.stack(positions)

    diff_positions = traj_positions - traj_positions[0]
    assert torch.max(diff_positions[:, :2]) < 1e-8
    assert torch.max(diff_positions[:, 2:]) > 1e-3
    assert (torch.mean(temperatures[len(temperatures) // 2 :]) - 300) / 300 < 0.30


def test_state_manipulation_with_constraints(ar_double_sim_state: ts.SimState):
    """Test that constraints are properly propagated during state manipulation."""
    # Set up constraints on the original state
    ar_double_sim_state.constraints = [
        FixAtoms(atom_idx=torch.tensor([0, 1])),  # Only applied to first system
        FixCom([0, 1]),
    ]

    # Extract individual systems from the double system state
    first_system = ar_double_sim_state[0]  # FixAtoms + FixCom
    second_system = ar_double_sim_state[1]  # FixCom only
    concatenated_state = ts.concatenate_states(
        [first_system, first_system, second_system]
    )

    # Verify constraint propagation to subsystems
    assert len(first_system.constraints) == 2
    assert len(second_system.constraints) == 1
    assert len(concatenated_state.constraints) == 2

    # Verify FixAtoms constraint indices are correctly mapped
    assert torch.all(first_system.constraints[0].atom_idx == torch.tensor([0, 1]))
    assert torch.all(
        concatenated_state.constraints[0].atom_idx == torch.tensor([0, 1, 32, 33])
    )

    # Verify FixCom constraint system masks
    assert torch.all(
        concatenated_state.constraints[1].system_idx == torch.tensor([0, 1, 2])
    )

    # Test constraint propagation after splitting concatenated state
    split_systems = concatenated_state.split()
    assert len(split_systems[0].constraints) == 2
    assert torch.all(split_systems[0].constraints[0].atom_idx == torch.tensor([0, 1]))
    assert torch.all(split_systems[1].constraints[0].atom_idx == torch.tensor([0, 1]))
    assert len(split_systems[2].constraints) == 1

    # Test constraint manipulation with different configurations
    ar_double_sim_state.constraints = []
    ar_double_sim_state.constraints = [FixCom([0, 1])]
    isolated_system = ar_double_sim_state[0]
    assert torch.all(
        isolated_system.constraints[0].system_idx == torch.tensor([0], dtype=torch.long)
    )

    # Test concatenation with mixed constraint states
    isolated_system.constraints = []
    mixed_concatenated_state = ts.concatenate_states(
        [isolated_system, ar_double_sim_state, isolated_system]
    )
    assert torch.all(
        mixed_concatenated_state.constraints[0].system_idx == torch.tensor([1, 2])
    )


def test_fix_com_gradient_descent_optimization(
    ar_supercell_sim_state: ts.SimState, lj_model: ModelInterface
) -> None:
    """Test FixCom constraint in Gradient Descent optimization."""
    # Add some random displacement to positions
    perturbed_positions = (
        ar_supercell_sim_state.positions
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    ar_supercell_sim_state.positions = perturbed_positions
    initial_state = ar_supercell_sim_state
    ar_supercell_sim_state.constraints = [FixCom([0])]

    initial_coms = get_centers_of_mass(
        positions=initial_state.positions,
        masses=initial_state.masses,
        system_idx=initial_state.system_idx,
        n_systems=initial_state.n_systems,
    )

    # Initialize Gradient Descent optimizer
    state = ts.gradient_descent_init(
        state=ar_supercell_sim_state, model=lj_model, lr=0.01
    )

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    while abs(energies[-2] - energies[-1]) > 1e-6:
        state = ts.gradient_descent_step(state=state, model=lj_model, pos_lr=0.01)
        energies.append(state.energy.item())

    final_coms = get_centers_of_mass(
        positions=state.positions,
        masses=state.masses,
        system_idx=state.system_idx,
        n_systems=initial_state.n_systems,
    )

    assert torch.allclose(final_coms, initial_coms, atol=1e-4)
    assert not torch.allclose(state.positions, initial_state.positions)


def test_fix_atoms_gradient_descent_optimization(
    ar_supercell_sim_state: ts.SimState, lj_model: ModelInterface
) -> None:
    """Test FixAtoms constraint in Gradient Descent optimization."""
    # Add some random displacement to positions
    perturbed_positions = (
        ar_supercell_sim_state.positions
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    ar_supercell_sim_state.positions = perturbed_positions
    initial_state = ar_supercell_sim_state
    initial_state.constraints = [FixAtoms(atom_idx=[0])]
    initial_position = initial_state.positions[0].clone()

    # Initialize Gradient Descent optimizer
    state = ts.gradient_descent_init(
        state=ar_supercell_sim_state, model=lj_model, lr=0.01
    )

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    while abs(energies[-2] - energies[-1]) > 1e-6:
        state = ts.gradient_descent_step(state=state, model=lj_model, pos_lr=0.01)
        energies.append(state.energy.item())

    final_position = state.positions[0]

    assert torch.allclose(final_position, initial_position, atol=1e-5)
    assert not torch.allclose(state.positions, initial_state.positions)


@pytest.mark.parametrize("fire_flavor", get_args(FireFlavor))
def test_test_atoms_fire_optimization(
    ar_supercell_sim_state: ts.SimState,
    lj_model: ModelInterface,
    fire_flavor: FireFlavor,
) -> None:
    """Test FixAtoms constraint in FIRE optimization."""
    # Add some random displacement to positions
    # Create a fresh copy for each test run to avoid interference

    current_positions = (
        ar_supercell_sim_state.positions.clone()
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    current_sim_state = ts.SimState(
        positions=current_positions,
        masses=ar_supercell_sim_state.masses.clone(),
        cell=ar_supercell_sim_state.cell.clone(),
        pbc=ar_supercell_sim_state.pbc,
        atomic_numbers=ar_supercell_sim_state.atomic_numbers.clone(),
        system_idx=ar_supercell_sim_state.system_idx.clone(),
    )
    indices = torch.tensor([0, 2], dtype=torch.long)
    current_sim_state.constraints = [FixAtoms(atom_idx=indices)]

    # Initialize FIRE optimizer
    state = ts.fire_init(
        current_sim_state, lj_model, fire_flavor=fire_flavor, dt_start=0.1
    )
    initial_position = state.positions[indices].clone()

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    max_steps = 1000  # Add max step to prevent infinite loop
    steps_taken = 0
    while abs(energies[-2] - energies[-1]) > 1e-6 and steps_taken < max_steps:
        state = ts.fire_step(state=state, model=lj_model, dt_max=0.3)
        energies.append(state.energy.item())
        steps_taken += 1

    final_position = state.positions[indices]

    assert torch.allclose(final_position, initial_position, atol=1e-5)


@pytest.mark.parametrize("fire_flavor", get_args(FireFlavor))
def test_fix_com_fire_optimization(
    ar_supercell_sim_state: ts.SimState,
    lj_model: ModelInterface,
    fire_flavor: FireFlavor,
) -> None:
    """Test FixCom constraint in FIRE optimization."""
    # Add some random displacement to positions
    # Create a fresh copy for each test run to avoid interference

    current_positions = (
        ar_supercell_sim_state.positions.clone()
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    current_sim_state = ts.SimState(
        positions=current_positions,
        masses=ar_supercell_sim_state.masses.clone(),
        cell=ar_supercell_sim_state.cell.clone(),
        pbc=ar_supercell_sim_state.pbc,
        atomic_numbers=ar_supercell_sim_state.atomic_numbers.clone(),
        system_idx=ar_supercell_sim_state.system_idx.clone(),
    )
    current_sim_state.constraints = [FixCom([0])]

    # Initialize FIRE optimizer
    state = ts.fire_init(
        current_sim_state, lj_model, fire_flavor=fire_flavor, dt_start=0.1
    )
    initial_com = get_centers_of_mass(
        positions=state.positions,
        masses=state.masses,
        system_idx=state.system_idx,
        n_systems=state.n_systems,
    )

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    max_steps = 1000  # Add max step to prevent infinite loop
    steps_taken = 0
    while abs(energies[-2] - energies[-1]) > 1e-6 and steps_taken < max_steps:
        state = ts.fire_step(state=state, model=lj_model, dt_max=0.3)
        energies.append(state.energy.item())
        steps_taken += 1

    final_com = get_centers_of_mass(
        positions=state.positions,
        masses=state.masses,
        system_idx=state.system_idx,
        n_systems=state.n_systems,
    )

    assert torch.allclose(final_com, initial_com, atol=1e-4)


def test_fix_atoms_validation() -> None:
    """Test FixAtoms construction and validation."""
    # Boolean mask conversion
    mask = torch.zeros(10, dtype=torch.bool)
    mask[:3] = True
    assert torch.all(FixAtoms(atom_mask=mask).atom_idx == torch.tensor([0, 1, 2]))

    # Invalid indices
    with pytest.raises(ValueError, match="Indices must be integers"):
        FixAtoms(atom_idx=torch.tensor([0.5, 1.5]))
    with pytest.raises(ValueError, match="Duplicate"):
        FixAtoms(atom_idx=torch.tensor([0, 1, 1]))
    with pytest.raises(ValueError, match="wrong number of dimensions"):
        FixAtoms(atom_idx=torch.tensor([[0, 1]]))


def test_constraint_validation_warnings(ar_double_sim_state: ts.SimState) -> None:
    """Test validation warnings for constraint conflicts."""
    with pytest.warns(UserWarning, match="Multiple constraints.*same atoms"):
        validate_constraints(
            [FixAtoms(atom_idx=[0, 1, 2]), FixAtoms(atom_idx=[2, 3, 4])],
            ar_double_sim_state,
        )
    with pytest.warns(UserWarning, match="FixCom together with other constraints"):
        validate_constraints(
            [FixCom([0]), FixAtoms(atom_idx=[0, 1])], ar_double_sim_state
        )


def test_constraint_validation_errors(
    cu_sim_state: ts.SimState,
    ar_supercell_sim_state: ts.SimState,
) -> None:
    """Test validation errors for invalid constraints."""
    # Out of bounds
    with pytest.raises(ValueError, match=r"has indices up to.*only has.*atoms"):
        cu_sim_state.constraints = [FixAtoms(atom_idx=[0, 1, 100])]

    # Validation in __post_init__
    with pytest.raises(ValueError, match="Duplicate"):
        ts.SimState(
            positions=ar_supercell_sim_state.positions.clone(),
            masses=ar_supercell_sim_state.masses,
            cell=ar_supercell_sim_state.cell,
            pbc=ar_supercell_sim_state.pbc,
            atomic_numbers=ar_supercell_sim_state.atomic_numbers,
            system_idx=ar_supercell_sim_state.system_idx,
            _constraints=[FixAtoms(atom_idx=[0, 0, 1])],
        )


@pytest.mark.parametrize(
    ("integrator", "constraint", "n_steps"),
    [
        ("nve", FixAtoms(atom_idx=[0, 1]), 100),
        ("nvt_nose_hoover", FixCom([0]), 200),
        ("npt_langevin", FixAtoms(atom_idx=[0, 3]), 200),
        ("npt_nose_hoover", FixCom([0]), 200),
    ],
)
def test_integrators_with_constraints(
    cu_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    integrator: str,
    constraint: Constraint,
    n_steps: int,
) -> None:
    """Test all integrators respect constraints."""
    cu_sim_state.constraints = [constraint]
    kT = torch.tensor(300.0, dtype=DTYPE) * MetalUnits.temperature
    dt = torch.tensor(0.001, dtype=DTYPE)

    # Store initial state
    if isinstance(constraint, FixAtoms):
        initial = cu_sim_state.positions[constraint.atom_idx].clone()
    else:
        initial = get_centers_of_mass(
            cu_sim_state.positions,
            cu_sim_state.masses,
            cu_sim_state.system_idx,
            cu_sim_state.n_systems,
        )

    # Run integration
    if integrator == "nve":
        state = ts.nve_init(cu_sim_state, lj_model, kT=kT, seed=42)
        for _ in range(n_steps):
            state = ts.nve_step(state, lj_model, dt=dt)
    elif integrator == "nvt_nose_hoover":
        state = ts.nvt_nose_hoover_init(cu_sim_state, lj_model, kT=kT, dt=dt)
        for _ in range(n_steps):
            state = ts.nvt_nose_hoover_step(state, lj_model, dt=dt, kT=kT)
    elif integrator == "npt_langevin":
        state = ts.npt_langevin_init(cu_sim_state, lj_model, kT=kT, seed=42, dt=dt)
        for _ in range(n_steps):
            state = ts.npt_langevin_step(
                state,
                lj_model,
                dt=dt,
                kT=kT,
                external_pressure=torch.tensor(0.0, dtype=DTYPE),
            )
    else:  # npt_nose_hoover
        state = ts.npt_nose_hoover_init(cu_sim_state, lj_model, kT=kT, dt=dt)
        for _ in range(n_steps):
            state = ts.npt_nose_hoover_step(
                state,
                lj_model,
                dt=torch.tensor(0.001, dtype=DTYPE),
                kT=kT,
                external_pressure=torch.tensor(0.0, dtype=DTYPE),
            )

    # Verify constraint held
    if isinstance(constraint, FixAtoms):
        assert torch.allclose(state.positions[constraint.atom_idx], initial, atol=1e-6)
    else:
        final = get_centers_of_mass(
            state.positions, state.masses, state.system_idx, state.n_systems
        )
        assert torch.allclose(final, initial, atol=1e-5)


def test_multiple_constraints_and_dof(
    cu_sim_state: ts.SimState, lj_model: LennardJonesModel
) -> None:
    """Test multiple constraints together with correct DOF calculation."""
    # Test DOF calculation
    n = cu_sim_state.n_atoms
    assert torch.all(cu_sim_state.get_number_of_degrees_of_freedom() == 3 * n)
    cu_sim_state.constraints = [FixAtoms(atom_idx=[0])]
    assert torch.all(cu_sim_state.get_number_of_degrees_of_freedom() == 3 * n - 3)
    cu_sim_state.constraints = [FixCom([0]), FixAtoms(atom_idx=[0])]
    assert torch.all(cu_sim_state.get_number_of_degrees_of_freedom() == 3 * n - 6)

    # Verify both constraints hold during dynamics
    initial_pos = cu_sim_state.positions[0].clone()
    initial_com = get_centers_of_mass(
        cu_sim_state.positions,
        cu_sim_state.masses,
        cu_sim_state.system_idx,
        cu_sim_state.n_systems,
    )
    state = ts.nvt_langevin_init(
        cu_sim_state,
        lj_model,
        kT=torch.tensor(300.0, dtype=DTYPE) * MetalUnits.temperature,
        seed=42,
    )
    for _ in range(200):
        state = ts.nvt_langevin_step(
            state,
            lj_model,
            dt=torch.tensor(0.001, dtype=DTYPE),
            kT=torch.tensor(300.0, dtype=DTYPE) * MetalUnits.temperature,
        )
    assert torch.allclose(state.positions[0], initial_pos, atol=1e-6)
    final_com = get_centers_of_mass(
        state.positions, state.masses, state.system_idx, state.n_systems
    )
    assert torch.allclose(final_com, initial_com, atol=1e-5)


@pytest.mark.parametrize(
    ("cell_filter", "fire_flavor"),
    [
        (ts.CellFilter.unit, "ase_fire"),
        (ts.CellFilter.frechet, "ase_fire"),
        (ts.CellFilter.frechet, "vv_fire"),
    ],
)
def test_cell_optimization_with_constraints(
    ar_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    cell_filter: str,
    fire_flavor: FireFlavor,
) -> None:
    """Test cell filters work with constraints."""
    ar_supercell_sim_state.positions += (
        torch.randn_like(ar_supercell_sim_state.positions) * 0.05
    )
    ar_supercell_sim_state.constraints = [FixAtoms(atom_idx=[0, 1])]
    state = ts.fire_init(
        ar_supercell_sim_state,
        lj_model,
        cell_filter=cell_filter,
        fire_flavor=fire_flavor,
    )
    for _ in range(50):
        state = ts.fire_step(state, lj_model, dt_max=0.1)
        if state.forces.abs().max() < 0.05:
            break
    assert len(state.constraints) > 0


def test_batched_constraints(ar_double_sim_state: ts.SimState) -> None:
    """Test system-specific constraints in batched states."""
    s1, s2 = ar_double_sim_state.split()
    s1.constraints = [FixAtoms(atom_idx=[0, 1])]
    s2.constraints = [FixCom([0])]
    combined = ts.concatenate_states([s1, s2])
    assert len(combined.constraints) == 2
    assert isinstance(combined.constraints[0], FixAtoms)
    assert torch.all(combined.constraints[0].atom_idx == torch.tensor([0, 1]))
    assert isinstance(combined.constraints[1], FixCom)
    assert torch.all(combined.constraints[1].system_idx == torch.tensor([1]))


def test_constraints_with_non_pbc(lj_model: LennardJonesModel) -> None:
    """Test constraints work with non-periodic boundaries."""
    state = ts.SimState(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            dtype=DTYPE,
        ),
        masses=torch.ones(4, dtype=DTYPE) * 39.948,
        cell=torch.eye(3, dtype=DTYPE).unsqueeze(0) * 10.0,
        pbc=False,
        atomic_numbers=torch.full((4,), 18, dtype=torch.long),
        system_idx=torch.zeros(4, dtype=torch.long),
    )
    state.constraints = [FixCom([0])]
    initial = get_centers_of_mass(
        state.positions, state.masses, state.system_idx, state.n_systems
    )
    md_state = ts.nve_init(state, lj_model, kT=torch.tensor(100.0, dtype=DTYPE), seed=42)
    for _ in range(100):
        md_state = ts.nve_step(md_state, lj_model, dt=torch.tensor(0.001, dtype=DTYPE))
    final = get_centers_of_mass(
        md_state.positions, md_state.masses, md_state.system_idx, md_state.n_systems
    )
    assert torch.allclose(final, initial, atol=1e-5)


def test_high_level_api_with_constraints(
    cu_sim_state: ts.SimState,
    ar_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test high-level integrate() and optimize() APIs with constraints."""
    # Test integrate()
    cu_sim_state.constraints = [FixCom([0])]
    initial_com = get_centers_of_mass(
        cu_sim_state.positions,
        cu_sim_state.masses,
        cu_sim_state.system_idx,
        cu_sim_state.n_systems,
    )
    final = ts.integrate(
        cu_sim_state,
        lj_model,
        integrator=ts.Integrator.nvt_langevin,
        n_steps=50,
        temperature=300.0,
        timestep=0.001,
    )
    final_com = get_centers_of_mass(
        final.positions, final.masses, final.system_idx, final.n_systems
    )
    assert torch.allclose(final_com, initial_com, atol=1e-5)

    # Test optimize()
    ar_supercell_sim_state.positions += (
        torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )
    ar_supercell_sim_state.constraints = [FixAtoms(atom_idx=[0, 1, 2])]
    initial_pos = ar_supercell_sim_state.positions[[0, 1, 2]].clone()
    final = ts.optimize(
        ar_supercell_sim_state, lj_model, optimizer=ts.Optimizer.fire, max_steps=500
    )
    assert torch.allclose(final.positions[[0, 1, 2]], initial_pos, atol=1e-5)


def test_temperature_with_constrained_dof(
    cu_sim_state: ts.SimState, lj_model: LennardJonesModel
) -> None:
    """Test temperature calculation uses constrained DOF."""
    target = 300.0
    cu_sim_state.constraints = [FixAtoms(atom_idx=[0, 1, 2])]
    state = ts.nvt_langevin_init(
        cu_sim_state,
        lj_model,
        kT=torch.tensor(target, dtype=DTYPE) * MetalUnits.temperature,
        seed=42,
    )
    temps = []
    for _ in range(4000):
        state = ts.nvt_langevin_step(
            state,
            lj_model,
            dt=torch.tensor(0.001, dtype=DTYPE),
            kT=torch.tensor(target, dtype=DTYPE) * MetalUnits.temperature,
        )
        temp = state.calc_kT()
        temps.append(temp / MetalUnits.temperature)
    avg = torch.mean(torch.stack(temps)[500:])
    assert abs(avg - target) / target < 0.30


def test_system_constraint_update_and_select() -> None:
    """Test select_constraint and select_sub_constraint for SystemConstraint."""
    # Create a FixCom constraint for systems 0, 1, 2
    constraint = FixCom([0, 1, 2])

    # Test select_constraint with system_mask
    # Keep systems 0 and 2 (drop system 1)
    atom_mask = torch.ones(10, dtype=torch.bool)
    system_mask = torch.tensor([True, False, True], dtype=torch.bool)
    updated_constraint = constraint.select_constraint(atom_mask, system_mask)

    # System indices should be renumbered: [0, 2] -> [0, 1]
    assert torch.all(updated_constraint.system_idx == torch.tensor([0, 1]))

    # Test select_sub_constraint
    # Select system 1 from the original constraint
    constraint = FixCom([0, 1, 2])
    atom_idx = torch.arange(5, 10)  # Atoms for a specific system
    sys_idx = 1
    sub_constraint = constraint.select_sub_constraint(atom_idx, sys_idx)

    # Should return a constraint with system_idx = [0] (renumbered from 1)
    assert sub_constraint is not None
    assert torch.all(sub_constraint.system_idx == torch.tensor([0]))

    # Test when system is not in constraint
    constraint = FixCom([0, 2])
    sub_constraint = constraint.select_sub_constraint(atom_idx, sys_idx=1)
    assert sub_constraint is None


def test_atom_indexed_constraint_update_and_select() -> None:
    """Test select_constraint and select_sub_constraint for AtomConstraint."""
    # Create a FixAtoms constraint for atoms 0, 1, 5, 8
    constraint = FixAtoms(atom_idx=[0, 1, 5, 8])

    # Test select_constraint with atom_mask
    # Keep atoms 0, 1, 2, 3, 5, 6, 7, 8 (drop atoms 4)
    atom_mask = torch.tensor(
        [True, True, True, True, False, True, True, True, True], dtype=torch.bool
    )
    system_mask = torch.ones(2, dtype=torch.bool)
    updated_constraint = constraint.select_constraint(atom_mask, system_mask)

    # Atom indices should be renumbered:
    # Original: [0, 1, 5, 8]
    # After dropping atom 4: [0, 1, 4, 7] (indices shift down by 1 after index 4)
    assert torch.all(updated_constraint.atom_idx == torch.tensor([0, 1, 4, 7]))

    # Test select_sub_constraint
    # Select atoms that belong to a specific system
    constraint = FixAtoms(atom_idx=[0, 1, 5, 8])
    atom_idx = torch.tensor([0, 1, 2, 3, 4])  # Atoms for first system
    sys_idx = 0
    sub_constraint = constraint.select_sub_constraint(atom_idx, sys_idx)

    # Should return a constraint with only atoms 0, 1 (within atom_idx range)
    # Renumbered to start from 0
    assert sub_constraint is not None
    assert torch.all(sub_constraint.atom_idx == torch.tensor([0, 1]))

    # Test with different atom range
    constraint = FixAtoms(atom_idx=[0, 1, 5, 8])
    atom_idx = torch.tensor([5, 6, 7, 8, 9])  # Atoms for second system
    sys_idx = 1
    sub_constraint = constraint.select_sub_constraint(atom_idx, sys_idx)

    # Should return a constraint with atoms 5, 8 renumbered to [0, 3]
    assert sub_constraint is not None
    assert torch.all(sub_constraint.atom_idx == torch.tensor([0, 3]))

    # Test when no atoms in range
    constraint = FixAtoms(atom_idx=[0, 1])
    atom_idx = torch.tensor([5, 6, 7, 8])
    sub_constraint = constraint.select_sub_constraint(atom_idx, sys_idx=1)
    assert sub_constraint is None


def test_merge_constraints(ar_double_sim_state: ts.SimState) -> None:
    """Test merge_constraints combines constraints from multiple systems."""
    # Split the double system state
    s1, s2 = ar_double_sim_state.split()
    n_atoms_s1 = s1.n_atoms
    n_atoms_s2 = s2.n_atoms

    # Create constraints for each system
    # System 1: Fix atoms 0, 1 and fix COM for system 0
    s1_constraints = [
        FixAtoms(atom_idx=[0, 1]),
        FixCom([0]),
    ]

    # System 2: Fix atoms 2, 3 and fix COM for system 0
    s2_constraints = [
        FixAtoms(atom_idx=[2, 3]),
        FixCom([0]),
    ]

    # Merge constraints
    constraint_lists = [s1_constraints, s2_constraints]
    num_atoms_per_state = torch.tensor([n_atoms_s1, n_atoms_s2])
    merged_constraints = merge_constraints(constraint_lists, num_atoms_per_state)

    # Should have 2 constraints: one FixAtoms and one FixCom
    assert len(merged_constraints) == 2

    # Find FixAtoms and FixCom in merged list
    fix_atoms = None
    fix_com = None
    for constraint in merged_constraints:
        if isinstance(constraint, FixAtoms):
            fix_atoms = constraint
        elif isinstance(constraint, FixCom):
            fix_com = constraint

    assert fix_atoms is not None
    assert fix_com is not None

    # FixAtoms should have indices [0, 1] from s1 and [2+n_atoms_s1, 3+n_atoms_s1] from s2
    expected_atom_indices = torch.tensor([0, 1, 2 + n_atoms_s1, 3 + n_atoms_s1])
    assert torch.all(fix_atoms.atom_idx == expected_atom_indices)

    # FixCom should have system_idx [0, 1] (one for each original system)
    expected_system_indices = torch.tensor([0, 1])
    assert torch.all(fix_com.system_idx == expected_system_indices)

    # Test with three systems
    s3 = s1.clone()
    s3_constraints = [FixAtoms(atom_idx=[0])]
    constraint_lists = [s1_constraints, s2_constraints, s3_constraints]
    num_atoms_per_state = torch.tensor([n_atoms_s1, n_atoms_s2, s3.n_atoms])
    merged_constraints = merge_constraints(constraint_lists, num_atoms_per_state)

    # Find FixAtoms
    fix_atoms = None
    for constraint in merged_constraints:
        if isinstance(constraint, FixAtoms):
            fix_atoms = constraint
            break

    assert fix_atoms is not None
    # Should include atoms from all three systems with proper offsets
    expected_atom_indices = torch.tensor(
        [0, 1, 2 + n_atoms_s1, 3 + n_atoms_s1, 0 + n_atoms_s1 + n_atoms_s2]
    )
    assert torch.all(fix_atoms.atom_idx == expected_atom_indices)
