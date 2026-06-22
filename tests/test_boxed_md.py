import pytest
import torch
from ase.build import molecule

import torch_sim as ts
from torch_sim.enhanced_sampling.boxed_md import BoxedMD, run_boxed_md, velocity_inversion
from torch_sim.integrators.nvt import nvt_langevin_init
from torch_sim.models.interface import ModelInterface


DEVICE = torch.device("cpu")
DTYPE = torch.float64


class HarmonicModel(ModelInterface):
    """Per-atom harmonic well E = 1/2 k sum (x - x0)^2.

    A bound, stationary potential: the potential energy fluctuates around its
    minimum and keeps producing fresh maxima, so BXDE can ratchet the system
    outward and place a sequence of rising boundaries (unlike an unbound LJ
    cluster, which simply relaxes and never beats its initial energy).
    """

    def __init__(
        self,
        centers: torch.Tensor,
        k: float = 1.0,
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        self._device = device
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = False
        self.k = float(k)
        self.register_buffer("centers", centers.to(device, dtype))

    def forward(self, state: ts.SimState, **_kwargs) -> dict[str, torch.Tensor]:
        disp = state.positions - self.centers
        e_atom = 0.5 * self.k * disp.pow(2).sum(dim=-1)
        energy = torch.zeros(
            state.n_systems, device=self._device, dtype=self._dtype
        ).index_add(0, state.system_idx, e_atom)
        forces = -self.k * disp
        return {"energy": energy, "forces": forces}


@pytest.fixture
def ethanol_state() -> ts.SimState:
    return ts.io.atoms_to_state([molecule("CH3CH2OH")], DEVICE, DTYPE)


@pytest.fixture
def harmonic_model(ethanol_state: ts.SimState) -> HarmonicModel:
    return HarmonicModel(ethanol_state.positions.clone(), k=2.0)


class TestVelocityInversion:
    def _random_inputs(self, n_atoms: int = 6) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(0)
        momenta = torch.randn(n_atoms, 3, dtype=DTYPE)
        forces = torch.randn(n_atoms, 3, dtype=DTYPE)
        masses = torch.rand(n_atoms, dtype=DTYPE) + 0.5
        return momenta, forces, masses

    def test_reverses_gradient_projection(self) -> None:
        # grad(phi) = grad(E) = -F, so a valid reflection must flip grad(phi).v,
        # equivalently F . v' == -(F . v).
        momenta, forces, masses = self._random_inputs()
        velocities = momenta / masses.unsqueeze(-1)
        new_momenta = velocity_inversion(momenta, forces, masses)
        new_velocities = new_momenta / masses.unsqueeze(-1)

        f_dot_v = (forces * velocities).sum()
        f_dot_v_new = (forces * new_velocities).sum()
        torch.testing.assert_close(f_dot_v_new, -f_dot_v)

    def test_conserves_kinetic_energy(self) -> None:
        # The mass-metric reflection is elastic: KE = 1/2 sum p^2 / m is preserved.
        momenta, forces, masses = self._random_inputs()
        new_momenta = velocity_inversion(momenta, forces, masses)
        ke = (momenta.pow(2) / masses.unsqueeze(-1)).sum()
        ke_new = (new_momenta.pow(2) / masses.unsqueeze(-1)).sum()
        torch.testing.assert_close(ke_new, ke)

    def test_idempotent_pair(self) -> None:
        # Applying the inversion twice (with the same forces) returns the original.
        momenta, forces, masses = self._random_inputs()
        once = velocity_inversion(momenta, forces, masses)
        twice = velocity_inversion(once, forces, masses)
        torch.testing.assert_close(twice, momenta)


class TestRunBoxedMD:
    def test_rejects_multiple_systems(self, harmonic_model: HarmonicModel) -> None:
        two = ts.io.atoms_to_state(
            [molecule("CH3CH2OH"), molecule("H2O")], DEVICE, DTYPE
        )
        with pytest.raises(ValueError, match="single system"):
            run_boxed_md(
                two,
                harmonic_model,
                n_steps=10,
                i_samp=2,
                timestep=0.001,
                temperature=300,
            )

    def test_runs_and_returns_finite_state(
        self, harmonic_model: HarmonicModel, ethanol_state: ts.SimState
    ) -> None:
        final_state, floors = run_boxed_md(
            ethanol_state,
            harmonic_model,
            n_steps=200,
            i_samp=5,
            timestep=0.001,
            temperature=300,
            seed=1,
        )
        assert torch.isfinite(final_state.positions).all()
        assert torch.isfinite(final_state.energy).all()
        assert floors.ndim == 1

    def test_floors_monotonically_increase(
        self, harmonic_model: HarmonicModel, ethanol_state: ts.SimState
    ) -> None:
        _, floors = run_boxed_md(
            ethanol_state,
            harmonic_model,
            n_steps=500,
            i_samp=3,
            timestep=0.001,
            temperature=500,
            seed=2,
        )
        assert floors.numel() >= 2
        # each new box raises (never lowers) the accessible-energy floor
        assert (floors[1:] >= floors[:-1]).all()


class TestBoxedMDController:
    def _init(
        self, model: HarmonicModel, state: ts.SimState
    ) -> tuple[ts.SimState, float, torch.Tensor]:
        state.rng = 0
        kT = 300 * ts.units.UnitSystem.metal.temperature
        dt = 0.001 * ts.units.UnitSystem.metal.time
        return nvt_langevin_init(state, model, kT=kT), kT, dt

    def test_step_limit_then_resume(
        self, harmonic_model: HarmonicModel, ethanol_state: ts.SimState
    ) -> None:
        md_state, kT, dt = self._init(harmonic_model, ethanol_state)
        controller = BoxedMD(harmonic_model, i_samp=10_000, dt=dt, kT=kT)

        # i_samp is huge, so no boundary can be placed within a small budget
        state, used, status = controller.run_epoch(md_state, max_steps=5)
        assert used == 5
        assert status == BoxedMD.STEP_LIMIT
        assert controller.total_steps == 5
        assert controller.i == 5  # all accepted (no floor yet), window advanced

        # resuming continues the same window rather than restarting it
        controller.run_epoch(state, max_steps=3)
        assert controller.total_steps == 8
        assert controller.i == 8

    def test_new_boundary_status_and_record(
        self, harmonic_model: HarmonicModel, ethanol_state: ts.SimState
    ) -> None:
        md_state, kT, dt = self._init(harmonic_model, ethanol_state)
        controller = BoxedMD(harmonic_model, i_samp=3, dt=dt, kT=kT)

        _state, _used, status = controller.run_epoch(md_state, max_steps=500)
        assert status == BoxedMD.NEW_BOUNDARY
        assert controller.v_bxde is not None
        assert len(controller.floors) == 1
        assert controller.i == 0  # window counter reset for the next box

    def test_reset(
        self, harmonic_model: HarmonicModel, ethanol_state: ts.SimState
    ) -> None:
        md_state, kT, dt = self._init(harmonic_model, ethanol_state)
        controller = BoxedMD(harmonic_model, i_samp=3, dt=dt, kT=kT)
        controller.run_epoch(md_state, max_steps=500)
        controller.reset()
        assert controller.v_bxde is None
        assert controller.total_steps == 0
        assert controller.floors.is_empty
