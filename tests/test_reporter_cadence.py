from contextlib import ExitStack

import pytest
import torch

import torch_sim as ts
from torch_sim.properties.correlations import VelocityAutoCorrelation


@pytest.fixture(params=[1, 2])
def reported_md(request, ar_atoms, lj_model, tmp_path):
    n_systems = request.param
    state = ts.initialize_state([ar_atoms] * n_systems, lj_model.device, lj_model.dtype)
    state.rng = 5
    state = ts.nve_init(state, lj_model, kT=0.1)
    memory_calc = VelocityAutoCorrelation(
        window_size=4, device=state.device, use_running_average=False
    )
    file_calcs = [
        VelocityAutoCorrelation(
            window_size=4, device=state.device, use_running_average=False
        )
        for _ in range(n_systems)
    ]
    paths = [tmp_path / f"system-{index}.h5" for index in range(n_systems)]
    memory_counts = []
    off_cadence = []
    with ExitStack() as stack:
        memory = stack.enter_context(
            ts.TrajectoryReporter(
                None, state_frequency=0, prop_calculators={3: {"windows": memory_calc}}
            )
        )
        files = [
            stack.enter_context(
                ts.TrajectoryReporter(
                    path, state_frequency=0, prop_calculators={3: {"windows": calc}}
                )
            )
            for path, calc in zip(paths, file_calcs, strict=True)
        ]
        for step in range(10):
            if step:
                state = ts.nve_step(state, lj_model, dt=0.01)
            props = memory.report(state, step)
            for reporter, substate in zip(files, state.split(), strict=True):
                reporter.report(substate, step)
            if step % 3 == 0:
                memory_counts.append([entry["windows"].item() for entry in props])
            else:
                off_cadence.extend(props)

        memory_vacf = memory_calc.vacf.clone()
        expected_vacf = torch.stack([calc.vacf for calc in file_calcs])
        pending_samples = memory_calc.corr_calc.buffers["velocity"].count
        state = ts.nve_step(state, lj_model, dt=0.01)
        forced = memory.report(state, 10, force=True)
        for reporter, substate in zip(files, state.split(), strict=True):
            reporter.report(substate, 10, force=True)

    persisted = []
    for path in paths:
        with ts.TorchSimTrajectory(path) as trajectory:
            persisted.append(
                (
                    trajectory.get_steps("windows").tolist(),
                    trajectory.get_array("windows").reshape(-1).tolist(),
                )
            )
    return dict(
        n_systems=n_systems,
        counts=memory_counts,
        off_cadence=off_cadence,
        vacf=memory_vacf,
        expected_vacf=expected_vacf,
        pending_samples=pending_samples,
        forced=forced,
        samples_after_force=memory_calc.corr_calc.buffers["velocity"].count,
        persisted=persisted,
    )


def test_memory_reporter_vacf_sampling(reported_md):
    n_systems = reported_md["n_systems"]
    if reported_md["counts"] != [[count] * n_systems for count in (0, 0, 0, 1)]:
        pytest.fail("The memory reporter advanced VACF between scheduled samples")
    if reported_md["pending_samples"] != 0 or any(reported_md["off_cadence"]):
        pytest.fail("Off-cadence calls advanced the calculator or returned properties")
    if not torch.allclose(
        reported_md["vacf"].reshape(n_systems, 4),
        reported_md["expected_vacf"],
        atol=1e-12,
        rtol=1e-12,
    ):
        pytest.fail("Memory reporting changed the VACF sampling interval")


def test_forced_memory_reporter(reported_md):
    if any(set(entry) != {"windows"} for entry in reported_md["forced"]):
        pytest.fail("Forced memory reporting omitted off-cadence properties")
    if any(entry["windows"].item() != 1 for entry in reported_md["forced"]):
        pytest.fail("A forced report included unscheduled correlation samples")
    if reported_md["samples_after_force"] != 1:
        pytest.fail("A forced report did not sample the current batched velocities once")
    expected = ([0, 3, 6, 9, 10], [0, 0, 0, 1, 1])
    if any(values != expected for values in reported_md["persisted"]):
        pytest.fail("The persisted reference disagrees with the requested sampling steps")


def test_memory_reporter_per_system_steps(ar_atoms, lj_model):
    state = ts.initialize_state([ar_atoms, ar_atoms], lj_model.device, lj_model.dtype)
    state.rng = 5
    state = ts.nve_init(state, lj_model, kT=0.1)
    active = VelocityAutoCorrelation(window_size=8, device=state.device)
    disabled = VelocityAutoCorrelation(window_size=8, device=state.device)
    with ts.TrajectoryReporter(
        None,
        prop_calculators={3: {"windows": active}, 0: {"disabled": disabled}},
    ) as reporter:
        skipped = reporter.report(state, [1, 2])
        if any(skipped) or active.corr_calc.buffers["velocity"].count != 0:
            pytest.fail("A batch with no scheduled systems advanced its calculator")
        partial = reporter.report(state, [3, 4])
        if set(partial[0]) != {"windows"} or partial[1]:
            pytest.fail("Per-system steps did not select the scheduled result")
        if active.corr_calc.buffers["velocity"].count != 1:
            pytest.fail("A partially scheduled batch did not invoke its calculator once")
        skipped = reporter.report(state, [4, 5])
        if any(skipped) or active.corr_calc.buffers["velocity"].count != 1:
            pytest.fail("A later off-cadence batch advanced its calculator")
        forced = reporter.report(state, [4, 5], force=True)
        if any(set(entry) != {"windows"} for entry in forced):
            pytest.fail("Forced reporting did not return properties for every system")
        if active.corr_calc.buffers["velocity"].count != 2:
            pytest.fail("Forced reporting did not invoke the batched calculator once")
        if disabled.corr_calc.buffers["velocity"].count != 0:
            pytest.fail("A zero-frequency calculator was invoked")
