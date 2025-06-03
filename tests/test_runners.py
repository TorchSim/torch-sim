from pathlib import Path

import numpy as np
import pytest
import torch

import torch_sim as ts
from torch_sim.autobatching import BinningAutoBatcher, InFlightAutoBatcher
from torch_sim.integrators import nve, nvt_langevin
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.optimizers import unit_cell_fire
from torch_sim.quantities import calc_kinetic_energy
from torch_sim.trajectory import TorchSimTrajectory, TrajectoryReporter


def test_integrate_nve(
    ar_supercell_sim_state: ts.SimState, lj_model: LennardJonesModel, tmp_path: Path
) -> None:
    """Test NVE integration with LJ potential."""
    traj_file = tmp_path / "nve.h5md"
    reporter = TrajectoryReporter(
        filenames=traj_file,
        state_frequency=1,
        prop_calculators={
            1: {"ke": lambda state: calc_kinetic_energy(state.momenta, state.masses)}
        },
    )

    final_state = ts.integrate(
        system=ar_supercell_sim_state,
        model=lj_model,
        integrator=nve,
        n_steps=10,
        temperature=100.0,  # K
        timestep=0.001,  # ps
        trajectory_reporter=reporter,
    )

    assert isinstance(final_state, ts.SimState)
    assert traj_file.is_file()

    # Check energy conservation
    with TorchSimTrajectory(traj_file) as traj:
        energies = traj.get_array("ke")
        std_energy = np.std(energies)
        assert std_energy / np.mean(energies) < 0.1  # 10% tolerance


def test_integrate_single_nvt(
    ar_supercell_sim_state: ts.SimState, lj_model: LennardJonesModel, tmp_path: Path
) -> None:
    """Test NVT integration with LJ potential."""
    traj_file = tmp_path / "nvt.h5md"
    reporter = TrajectoryReporter(
        filenames=traj_file,
        state_frequency=1,
        prop_calculators={
            1: {"ke": lambda state: calc_kinetic_energy(state.momenta, state.masses)}
        },
    )

    final_state = ts.integrate(
        system=ar_supercell_sim_state,
        model=lj_model,
        integrator=nvt_langevin,
        n_steps=10,
        temperature=100.0,  # K
        timestep=0.001,  # ps
        trajectory_reporter=reporter,
        gamma=0.1,  # ps^-1
    )

    assert isinstance(final_state, ts.SimState)
    assert traj_file.is_file()

    # Check energy fluctuations
    with TorchSimTrajectory(traj_file) as traj:
        energies = traj.get_array("ke")
        std_energy = np.std(energies)
        assert std_energy / np.mean(energies) < 0.2  # 20% tolerance for NVT


def test_integrate_double_nvt(
    ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel
) -> None:
    """Test NVT integration with LJ potential."""
    final_state = ts.integrate(
        system=ar_double_sim_state,
        model=lj_model,
        integrator=nvt_langevin,
        n_steps=10,
        temperature=100.0,  # K
        timestep=0.001,  # ps
    )

    assert isinstance(final_state, ts.SimState)
    assert final_state.n_atoms == 64
    assert not torch.isnan(final_state.energy).any()


def test_integrate_double_nvt_with_reporter(
    ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel, tmp_path: Path
) -> None:
    """Test NVT integration with LJ potential."""
    trajectory_files = [tmp_path / "nvt_0.h5md", tmp_path / "nvt_1.h5md"]
    reporter = TrajectoryReporter(
        filenames=trajectory_files,
        state_frequency=1,
        prop_calculators={
            1: {"ke": lambda state: calc_kinetic_energy(state.momenta, state.masses)}
        },
    )

    final_state = ts.integrate(
        system=ar_double_sim_state,
        model=lj_model,
        integrator=nvt_langevin,
        n_steps=10,
        temperature=100.0,  # K
        timestep=0.001,  # ps
        trajectory_reporter=reporter,
        gamma=0.1,  # ps^-1
    )

    assert isinstance(final_state, ts.SimState)
    assert final_state.n_atoms == 64
    assert all(traj_file.is_file() for traj_file in trajectory_files)

    # Check energy fluctuations
    for traj_file in trajectory_files:
        with TorchSimTrajectory(traj_file) as traj:
            energies = traj.get_array("ke")
        std_energy = np.std(energies)
        assert (std_energy / np.mean(energies) < 0.2).all()  # 20% tolerance for NVT
    assert not torch.isnan(final_state.energy).any()


def test_integrate_many_nvt(
    ar_supercell_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    tmp_path: Path,
) -> None:
    """Test NVT integration with LJ potential."""
    triple_state = ts.initialize_state(
        [ar_supercell_sim_state, ar_supercell_sim_state, fe_supercell_sim_state],
        lj_model.device,
        lj_model.dtype,
    )
    trajectory_files = [
        tmp_path / f"nvt_{batch}.h5md" for batch in range(triple_state.n_batches)
    ]
    reporter = TrajectoryReporter(
        filenames=trajectory_files,
        state_frequency=1,
        prop_calculators={
            1: {"ke": lambda state: calc_kinetic_energy(state.momenta, state.masses)}
        },
    )

    final_state = ts.integrate(
        system=triple_state,
        model=lj_model,
        integrator=nve,
        n_steps=10,
        temperature=300.0,  # K
        timestep=0.001,  # ps
        trajectory_reporter=reporter,
    )

    assert isinstance(final_state, ts.SimState)
    assert all(traj_file.is_file() for traj_file in trajectory_files)
    assert not torch.isnan(final_state.energy).any()
    assert not torch.isnan(final_state.positions).any()
    assert not torch.isnan(final_state.momenta).any()

    assert torch.allclose(final_state.energy[0], final_state.energy[1], atol=1e-2)
    assert not torch.allclose(final_state.energy[0], final_state.energy[2], atol=1e-2)


def test_integrate_with_autobatcher(
    ar_supercell_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test integration with autobatcher."""
    states = [ar_supercell_sim_state, fe_supercell_sim_state, ar_supercell_sim_state]
    triple_state = ts.initialize_state(
        states,
        lj_model.device,
        lj_model.dtype,
    )
    autobatcher = BinningAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260,
    )
    final_states = ts.integrate(
        system=triple_state,
        model=lj_model,
        integrator=nve,
        n_steps=10,
        temperature=300.0,
        timestep=0.001,
        autobatcher=autobatcher,
    )

    assert isinstance(final_states, ts.SimState)
    for init_state, final_state in zip(states, final_states.split(), strict=True):
        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)


def test_integrate_with_autobatcher_and_reporting(
    ar_supercell_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    tmp_path: Path,
) -> None:
    """Test integration with autobatcher."""
    states = [ar_supercell_sim_state, fe_supercell_sim_state, ar_supercell_sim_state]
    triple_state = ts.initialize_state(
        states,
        lj_model.device,
        lj_model.dtype,
    )
    autobatcher = BinningAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260,
    )
    trajectory_files = [
        tmp_path / f"nvt_{batch}.h5md" for batch in range(triple_state.n_batches)
    ]
    reporter = TrajectoryReporter(
        filenames=trajectory_files,
        state_frequency=1,
        prop_calculators={1: {"pe": lambda state: state.energy}},
    )
    final_states = ts.integrate(
        system=triple_state,
        model=lj_model,
        integrator=nve,
        n_steps=10,
        temperature=300.0,
        timestep=0.001,
        trajectory_reporter=reporter,
        autobatcher=autobatcher,
    )

    assert all(traj_file.is_file() for traj_file in trajectory_files)

    assert isinstance(final_states, ts.SimState)
    for init_state, final_state in zip(states, final_states.split(), strict=True):
        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)

    for init_state, traj_file in zip(states, trajectory_files, strict=False):
        with TorchSimTrajectory(traj_file) as traj:
            final_state = traj.get_state(
                -1, device=init_state.device, dtype=init_state.dtype
            )
            energies = traj.get_array("pe")
            energy_steps = traj.get_steps("pe")
            assert len(energies) == 10
            assert len(energy_steps) == 10

        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)


def test_optimize_fire(
    ar_supercell_sim_state: ts.SimState, lj_model: LennardJonesModel, tmp_path: Path
) -> None:
    """Test FIRE optimization with LJ potential."""
    trajectory_files = [tmp_path / "opt.h5md"]
    reporter = TrajectoryReporter(
        filenames=[tmp_path / "opt.h5md"],
        prop_calculators={1: {"energy": lambda state: state.energy}},
    )
    ar_supercell_sim_state.positions += (
        torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    original_state = ar_supercell_sim_state.clone()

    final_state = ts.optimize(
        system=ar_supercell_sim_state,
        model=lj_model,
        optimizer=unit_cell_fire,
        convergence_fn=ts.generate_force_convergence_fn(force_tol=1e-1),
        trajectory_reporter=reporter,
    )

    with TorchSimTrajectory(trajectory_files[0]) as traj:
        energies = traj.get_array("energy")

    # Check force convergence
    assert torch.all(final_state.forces < 3e-1)
    assert energies.shape[0] >= 10
    assert energies[0] > energies[-1]
    assert not torch.allclose(original_state.positions, final_state.positions)


def test_default_converged_fn(
    ar_supercell_sim_state: ts.SimState, lj_model: LennardJonesModel, tmp_path: Path
) -> None:
    """Test default converged function."""
    ar_supercell_sim_state.positions += (
        torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    traj_file = tmp_path / "opt.h5md"
    reporter = TrajectoryReporter(
        filenames=traj_file, prop_calculators={1: {"energy": lambda state: state.energy}}
    )

    original_state = ar_supercell_sim_state.clone()

    final_state = ts.optimize(
        system=ar_supercell_sim_state,
        model=lj_model,
        optimizer=unit_cell_fire,
        trajectory_reporter=reporter,
    )

    with TorchSimTrajectory(traj_file) as traj:
        energies = traj.get_array("energy")

    # Check that overall energy decreases (first to last)
    assert energies[0] > energies[-1]
    assert not torch.allclose(original_state.positions, final_state.positions)


def test_batched_optimize_fire(
    ar_double_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    tmp_path: Path,
) -> None:
    """Test batched FIRE optimization with LJ potential."""
    trajectory_files = [
        tmp_path / f"nvt_{idx}.h5md" for idx in range(ar_double_sim_state.n_batches)
    ]
    reporter = TrajectoryReporter(
        filenames=trajectory_files,
        state_frequency=1,
        prop_calculators={
            1: {"ke": lambda state: calc_kinetic_energy(state.momenta, state.masses)}
        },
    )

    final_state = ts.optimize(
        system=ar_double_sim_state,
        model=lj_model,
        optimizer=unit_cell_fire,
        convergence_fn=ts.generate_force_convergence_fn(force_tol=1e-1),
        trajectory_reporter=reporter,
    )

    assert torch.all(final_state.forces < 1e-4)


def test_optimize_with_autobatcher(
    ar_supercell_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test optimize with autobatcher."""
    states = [ar_supercell_sim_state, fe_supercell_sim_state, ar_supercell_sim_state]
    triple_state = ts.initialize_state(
        states,
        lj_model.device,
        lj_model.dtype,
    )
    autobatcher = InFlightAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260,
    )
    final_states = ts.optimize(
        system=triple_state,
        model=lj_model,
        optimizer=unit_cell_fire,
        convergence_fn=ts.generate_force_convergence_fn(force_tol=1e-1),
        autobatcher=autobatcher,
    )

    assert isinstance(final_states, ts.SimState)
    for init_state, final_state in zip(states, final_states.split(), strict=True):
        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)


def test_optimize_with_autobatcher_and_reporting(
    ar_supercell_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    tmp_path: Path,
) -> None:
    """Test optimize with autobatcher and reporting."""
    states = [ar_supercell_sim_state, fe_supercell_sim_state, ar_supercell_sim_state]
    triple_state = ts.initialize_state(
        states,
        lj_model.device,
        lj_model.dtype,
    )
    triple_state.positions += torch.randn_like(triple_state.positions) * 0.1

    autobatcher = InFlightAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260,
    )

    trajectory_files = [
        tmp_path / f"opt_{batch}.h5md" for batch in range(triple_state.n_batches)
    ]
    reporter = TrajectoryReporter(
        filenames=trajectory_files,
        state_frequency=1,
        prop_calculators={1: {"pe": lambda state: state.energy}},
    )

    final_states = ts.optimize(
        system=triple_state,
        model=lj_model,
        optimizer=unit_cell_fire,
        convergence_fn=ts.generate_force_convergence_fn(force_tol=1e-1),
        trajectory_reporter=reporter,
        autobatcher=autobatcher,
    )

    assert all(traj_file.is_file() for traj_file in trajectory_files)

    assert isinstance(final_states, ts.SimState)
    for init_state, final_state in zip(states, final_states.split(), strict=True):
        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)
        assert torch.all(final_state.forces < 1e-1)

    for init_state, traj_file in zip(states, trajectory_files, strict=False):
        with TorchSimTrajectory(traj_file) as traj:
            traj_state = traj.get_state(
                -1, device=init_state.device, dtype=init_state.dtype
            )
            energies = traj.get_array("pe")
            energy_steps = traj.get_steps("pe")
            assert len(energies) > 0
            assert len(energy_steps) > 0
            # Check that energy decreases during optimization
            assert energies[0] > energies[-1]

        assert torch.all(traj_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(traj_state.positions != init_state.positions)


def test_integrate_with_default_autobatcher(
    ar_supercell_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test integration with autobatcher."""

    def mock_estimate(*args, **kwargs) -> float:  # noqa: ARG001
        return 10_000.0

    monkeypatch.setattr(
        "torch_sim.autobatching.estimate_max_memory_scaler", mock_estimate
    )

    states = [ar_supercell_sim_state, fe_supercell_sim_state, ar_supercell_sim_state]
    triple_state = ts.initialize_state(
        states,
        lj_model.device,
        lj_model.dtype,
    )

    final_states = ts.integrate(
        system=triple_state,
        model=lj_model,
        integrator=nve,
        n_steps=10,
        temperature=300.0,
        timestep=0.001,
        autobatcher=True,
    )

    assert isinstance(final_states, ts.SimState)
    for init_state, final_state in zip(states, final_states.split(), strict=True):
        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)


def test_optimize_with_default_autobatcher(
    ar_supercell_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test optimize with autobatcher."""

    def mock_estimate(*args, **kwargs) -> float:  # noqa: ARG001
        return 200

    monkeypatch.setattr("torch_sim.autobatching.determine_max_batch_size", mock_estimate)

    states = [ar_supercell_sim_state, fe_supercell_sim_state, ar_supercell_sim_state]
    triple_state = ts.initialize_state(
        states,
        lj_model.device,
        lj_model.dtype,
    )

    final_states = ts.optimize(
        system=triple_state,
        model=lj_model,
        optimizer=unit_cell_fire,
        convergence_fn=ts.generate_force_convergence_fn(force_tol=1e-1),
        autobatcher=True,
    )

    assert isinstance(final_states, ts.SimState)
    for init_state, final_state in zip(states, final_states.split(), strict=True):
        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)


def test_static_single(
    ar_supercell_sim_state: ts.SimState, lj_model: LennardJonesModel, tmp_path: Path
) -> None:
    """Test static calculation with LJ potential."""
    traj_file = tmp_path / "static.h5md"
    reporter = TrajectoryReporter(
        filenames=traj_file,
        state_frequency=1,
        prop_calculators={1: {"potential_energy": lambda state: state.energy}},
        state_kwargs={"save_forces": True},  # Enable force saving
    )

    props = ts.static(
        system=ar_supercell_sim_state,
        model=lj_model,
        trajectory_reporter=reporter,
    )

    assert isinstance(props, list)
    assert len(props) == 1  # Single system = single props dict
    assert "potential_energy" in props[0]
    assert traj_file.is_file()

    # Check that energy was computed and saved correctly
    with TorchSimTrajectory(traj_file) as traj:
        saved_energy = traj.get_array("potential_energy")
        assert len(saved_energy) == 1  # Static calc = single frame
        np.testing.assert_allclose(saved_energy[0], props[0]["potential_energy"].numpy())

        # Verify state_kwargs were applied correctly
        assert traj.get_array("atomic_numbers").shape == (
            1,
            ar_supercell_sim_state.n_atoms,
        )
        assert traj.get_array("masses").shape == (1, ar_supercell_sim_state.n_atoms)
        if lj_model.compute_forces:
            assert "forces" in traj.array_registry


def test_static_double(
    ar_double_sim_state: ts.SimState, lj_model: LennardJonesModel, tmp_path: Path
) -> None:
    """Test static calculation with multiple systems."""
    trajectory_files = [tmp_path / "static_0.h5md", tmp_path / "static_1.h5md"]
    reporter = TrajectoryReporter(
        filenames=trajectory_files,
        state_frequency=1,
        prop_calculators={1: {"potential_energy": lambda state: state.energy}},
    )

    props = ts.static(
        system=ar_double_sim_state,
        model=lj_model,
        trajectory_reporter=reporter,
    )

    assert isinstance(props, list)
    assert len(props) == 2  # Two systems = two prop dicts
    assert all("potential_energy" in p for p in props)
    assert all(f.is_file() for f in trajectory_files)

    # Check energies were saved correctly
    for idx, traj_file in enumerate(trajectory_files):
        with TorchSimTrajectory(traj_file) as traj:
            saved_energy = traj.get_array("potential_energy")
            assert len(saved_energy) == 1
            np.testing.assert_allclose(
                saved_energy[0], props[idx]["potential_energy"].numpy()
            )


def test_static_with_autobatcher(
    ar_supercell_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test static calculation with autobatcher."""
    states = [ar_supercell_sim_state, fe_supercell_sim_state, ar_supercell_sim_state]
    triple_state = ts.initialize_state(
        states,
        lj_model.device,
        lj_model.dtype,
    )
    autobatcher = BinningAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260,
    )

    props = ts.static(
        system=triple_state,
        model=lj_model,
        autobatcher=autobatcher,
    )

    assert isinstance(props, list)
    assert len(props) == 3  # Three systems = three prop dicts

    # Check that identical systems have identical energies
    assert torch.allclose(props[0]["potential_energy"], props[2]["potential_energy"])
    # Check that different systems have different energies
    assert not torch.allclose(props[0]["potential_energy"], props[1]["potential_energy"])


def test_static_with_autobatcher_and_reporting(
    lj_model: LennardJonesModel,  # Changed type from Any, removed unused fixtures
    tmp_path: Path,
) -> None:
    """Test static calculation with autobatcher, trajectory reporting, and robust
    reordering."""
    from ase.build import bulk

    # 1. Create diverse SimState objects for robust binning test
    # Atom counts: Ar(4), Fe(8), Cu(8), Ar(4, different lattice)
    s0_atoms = bulk("Ar", "fcc", a=5.2, cubic=True)
    s1_atoms = bulk("Fe", "bcc", a=2.8, cubic=True).repeat((2, 2, 1))
    s2_atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 1, 1))
    s3_atoms = bulk("Ar", "fcc", a=5.3, cubic=True)  # Different params from s0_atoms

    initial_sim_states: list[ts.SimState] = []
    for idx, atoms_obj in enumerate((s0_atoms, s1_atoms, s2_atoms, s3_atoms)):
        sim_state_batched = ts.initialize_state(
            atoms_obj, device=lj_model.device, dtype=lj_model.dtype
        )
        sim_state = sim_state_batched.split()[0]
        torch.manual_seed(idx)  # Ensure different perturbations for each state
        sim_state.positions += torch.randn_like(sim_state.positions) * 0.05
        initial_sim_states.append(sim_state)

    batched_initial_state = ts.initialize_state(
        initial_sim_states, lj_model.device, lj_model.dtype
    )
    split_initial_states = batched_initial_state.split()

    # 2. Pre-calculate expected potential energies
    expected_energies: list[float] = []
    for s_init in split_initial_states:
        energy = lj_model(s_init)["energy"]
        expected_energies.append(energy)

    uniq_energies = set(expected_energies)
    assert len(uniq_energies) == len(expected_energies), (
        f"Need unique energies for robust ordering test. Got: {expected_energies}"
    )

    # 3. Configure BinningAutoBatcher to force multiple batches
    # Atom counts: 4, 8, 8, 4. LennardJonesModel memory_scales_with="n_atoms" by default.
    # max_memory_scaler=10 should force batches like [4,4], [8], [8] or similar.
    autobatcher = BinningAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=10,
        return_indices=True,
    )

    # 4. Call ts.static with trajectory reporting
    trajectory_files = [
        tmp_path / f"static_merged_reorder_{idx}.h5md"
        for idx in range(len(split_initial_states))
    ]
    reporter = TrajectoryReporter(
        filenames=trajectory_files,
        state_frequency=1,
        prop_calculators={1: {"potential_energy": lambda state: state.energy}},
    )

    returned_props = ts.static(
        system=batched_initial_state,
        model=lj_model,
        autobatcher=autobatcher,
        trajectory_reporter=reporter,
    )

    # 5. Assertions
    assert len(returned_props) == len(expected_energies), (
        f"Expected {len(expected_energies)} prop dicts, got {len(returned_props)}"
    )
    assert all(traj_file.is_file() for traj_file in trajectory_files), (
        "Not all trajectory files were created."
    )

    for idx in range(len(expected_energies)):
        # Check returned properties list order
        actual_energy = returned_props[idx]["potential_energy"]
        err_msg = f"Energy mismatch in returned props for original state {idx}"
        np.testing.assert_allclose(
            actual_energy, expected_energies[idx], rtol=1e-5, err_msg=err_msg
        )

        # Check trajectory file content and order
        with TorchSimTrajectory(trajectory_files[idx]) as traj:
            saved_energies = traj.get_array("potential_energy")
            assert len(saved_energies) == 1
            saved_energy_traj = saved_energies[0]

            file_name = trajectory_files[idx].name
            err_msg = (
                f"Trajectory energy mismatch for original state={idx} in {file_name=}"
            )
            np.testing.assert_allclose(
                saved_energy_traj, expected_energies[idx], rtol=1e-5, err_msg=err_msg
            )

            original_state_for_traj = split_initial_states[idx]
            saved_atomic_numbers = traj.get_array("atomic_numbers")[0]
            np.testing.assert_equal(
                saved_atomic_numbers,
                original_state_for_traj.atomic_numbers[0],
                err_msg=f"Atomic numbers mismatch for state {idx} in {file_name=}",
            )


def test_static_no_filenames(
    ar_supercell_sim_state: ts.SimState, lj_model: LennardJonesModel
) -> None:
    """Test static calculation with no trajectory filenames."""
    reporter = TrajectoryReporter(
        filenames=None,
        state_frequency=1,
        prop_calculators={1: {"potential_energy": lambda state: state.energy}},
    )

    props = ts.static(
        system=ar_supercell_sim_state, model=lj_model, trajectory_reporter=reporter
    )

    assert isinstance(props, list)
    assert len(props) == 1
    assert "potential_energy" in props[0]
    assert isinstance(props[0]["potential_energy"], torch.Tensor)


def test_readme_example(lj_model: LennardJonesModel, tmp_path: Path) -> None:
    # this tests the example from the readme, update as needed

    from ase.build import bulk

    import torch_sim as ts

    cu_atoms = bulk("Cu", "fcc", a=3.58, cubic=True).repeat((2, 2, 2))
    many_cu_atoms = [cu_atoms] * 5
    trajectory_files = [tmp_path / f"Cu_traj_{i}" for i in range(len(many_cu_atoms))]

    # run them all simultaneously with batching
    final_state = ts.integrate(
        system=many_cu_atoms,
        model=lj_model,
        n_steps=50,
        timestep=0.002,
        temperature=1000,
        integrator=ts.nvt_langevin,
        trajectory_reporter=dict(filenames=trajectory_files, state_frequency=10),
    )
    final_atoms_list = final_state.to_atoms()  # noqa: F841

    # extract the final energy from the trajectory file
    final_energies = []
    for filename in trajectory_files:
        with ts.TorchSimTrajectory(filename) as traj:
            final_energies.append(traj.get_array("potential_energy")[-1])

    print(final_energies)

    # relax all of the high temperature states
    relaxed_state = ts.optimize(
        system=final_state,
        model=lj_model,
        optimizer=ts.frechet_cell_fire,
        # autobatcher=True,
    )

    print(relaxed_state.energy)
