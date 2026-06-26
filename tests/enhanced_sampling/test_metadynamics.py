import pytest
import torch
from ase.build import molecule

import torch_sim as ts
from torch_sim.enhanced_sampling.metadynamics import EPS, RMSDCV, LogfermiWall
from torch_sim.models.interface import SumModel
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.units import UnitConversion


DEVICE = torch.device("cpu")
DTYPE = torch.float64


@pytest.fixture
def ethanol_state() -> ts.SimState:
    """Single low-symmetry molecule (avoids degenerate Kabsch SVD)."""
    return ts.io.atoms_to_state([molecule("CH3CH2OH")], DEVICE, DTYPE)


@pytest.fixture
def ragged_state() -> ts.SimState:
    """Two molecules with different atom counts in one batch."""
    return ts.io.atoms_to_state([molecule("CH3CH2OH"), molecule("H2O")], DEVICE, DTYPE)


class TestLogfermiWall:
    def test_output_shapes(self, ragged_state: ts.SimState) -> None:
        wall = LogfermiWall(radius=10.0, device=DEVICE, dtype=DTYPE)
        output = wall(ragged_state)
        assert output["energy"].shape == (2,)
        assert output["forces"].shape == (ragged_state.n_atoms, 3)

    def test_energy_negligible_deep_inside(self, ethanol_state: ts.SimState) -> None:
        wall = LogfermiWall(radius=50.0, device=DEVICE, dtype=DTYPE)
        output = wall(ethanol_state)
        assert output["energy"].abs().max() < 1e-10
        assert output["forces"].abs().max() < 1e-10

    def test_restoring_force_outside_wall(self, ethanol_state: ts.SimState) -> None:
        state = ethanol_state
        state.positions = state.positions + torch.tensor([20.0, 0.0, 0.0])
        wall = LogfermiWall(radius=5.0, device=DEVICE, dtype=DTYPE)
        output = wall(state)
        assert output["energy"].min() > 0
        # all atoms sit at x ~ 20 > radius, so forces must point back toward origin
        assert (output["forces"][:, 0] < 0).all()

    def test_forces_match_autograd(self, ragged_state: ts.SimState) -> None:
        state = ragged_state
        wall = LogfermiWall(radius=2.0, beta=3.0, device=DEVICE, dtype=DTYPE)
        analytic = wall(state)["forces"]

        positions = state.positions.detach().clone().requires_grad_(requires_grad=True)
        dvec = positions
        r = torch.norm(dvec, dim=-1) + EPS
        v_atom = wall.k_wall * torch.nn.functional.softplus(wall.beta * (r - wall.radius))
        expected = -torch.autograd.grad(v_atom.sum(), positions)[0]
        torch.testing.assert_close(analytic, expected, atol=1e-10, rtol=1e-8)

    def test_per_system_center(self, ragged_state: ts.SimState) -> None:
        # centering the wall on each molecule's first atom changes nothing
        # qualitatively, but exercises the (n_systems, 3) center branch
        first_atom_idx = torch.tensor([0, ragged_state.system_idx.tolist().count(0)])
        center = ragged_state.positions[first_atom_idx]
        wall = LogfermiWall(radius=5.0, center=center, device=DEVICE, dtype=DTYPE)
        output = wall(ragged_state)
        assert torch.isfinite(output["energy"]).all()
        assert torch.isfinite(output["forces"]).all()


class TestRMSDCV:
    def test_first_call_seeds_and_returns_zero(self, ragged_state: ts.SimState) -> None:
        bias = RMSDCV(device=DEVICE, dtype=DTYPE)
        output = bias(ragged_state)
        assert output["energy"].abs().max() == 0
        assert output["forces"].abs().max() == 0
        assert bias.ref_buf is not None
        assert bias.ref_buf.shape == (1, ragged_state.n_atoms, 3)

    def test_unmoved_state_feels_full_bias(self, ragged_state: ts.SimState) -> None:
        k_push = 0.02
        bias = RMSDCV(k_push=k_push, update_interval=1000, device=DEVICE, dtype=DTYPE)
        bias(ragged_state)  # seed
        output = bias(ragged_state)
        expected = k_push * UnitConversion.Hartree_to_eV
        torch.testing.assert_close(
            output["energy"],
            torch.full((2,), expected, device=DEVICE, dtype=DTYPE),
        )
        # rmsd^2 = 0 is a stationary point of the bias, so forces vanish
        assert output["forces"].abs().max() < 1e-8

    def test_rotation_translation_invariance(self, ethanol_state: ts.SimState) -> None:
        bias = RMSDCV(k_push=0.02, update_interval=1000, device=DEVICE, dtype=DTYPE)
        bias(ethanol_state)  # seed
        reference_energy = bias(ethanol_state)["energy"]

        angle = torch.tensor(0.3, dtype=DTYPE)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rot = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=DTYPE
        )
        moved = ethanol_state.clone()
        moved.positions = moved.positions @ rot.T + torch.tensor([1.0, -2.0, 3.0])
        output = bias(moved)
        torch.testing.assert_close(output["energy"], reference_energy)

    def test_forces_match_finite_difference(self, ethanol_state: ts.SimState) -> None:
        bias = RMSDCV(
            k_push=0.02, alpha_width=1.0, update_interval=1000, device=DEVICE, dtype=DTYPE
        )
        bias(ethanol_state)  # seed
        perturbed = ethanol_state.clone()
        torch.manual_seed(42)
        perturbed.positions = perturbed.positions + 0.1 * torch.randn_like(
            perturbed.positions
        )
        forces = bias(perturbed)["forces"]

        delta = 1e-6
        for atom, coord in [(0, 0), (3, 1), (7, 2)]:
            plus = perturbed.clone()
            plus.positions[atom, coord] += delta
            minus = perturbed.clone()
            minus.positions[atom, coord] -= delta
            e_plus = bias(plus)["energy"].sum()
            e_minus = bias(minus)["energy"].sum()
            numerical = -(e_plus - e_minus) / (2 * delta)
            torch.testing.assert_close(
                forces[atom, coord], numerical, atol=1e-6, rtol=1e-4
            )

    def test_buffer_capped_at_n_refs(self, ethanol_state: ts.SimState) -> None:
        bias = RMSDCV(n_refs=3, update_interval=1, device=DEVICE, dtype=DTYPE)
        for _ in range(6):
            bias(ethanol_state)
        assert bias.ref_buf is not None
        assert bias.ref_buf.shape[0] == 3

    def test_update_interval_deposition(self, ethanol_state: ts.SimState) -> None:
        bias = RMSDCV(n_refs=100, update_interval=3, device=DEVICE, dtype=DTYPE)
        for _ in range(7):  # seed + deposits at calls 3 and 6
            bias(ethanol_state)
        assert bias.ref_buf is not None
        assert bias.ref_buf.shape[0] == 3

    def test_atom_mask_zeroes_fixed_atom_forces(self, ethanol_state: ts.SimState) -> None:
        mask = torch.ones(ethanol_state.n_atoms, dtype=torch.bool)
        mask[:3] = False
        bias = RMSDCV(atom_mask=mask, update_interval=1000, device=DEVICE, dtype=DTYPE)
        bias(ethanol_state)  # seed
        perturbed = ethanol_state.clone()
        torch.manual_seed(7)
        perturbed.positions = perturbed.positions + 0.2 * torch.randn_like(
            perturbed.positions
        )
        output = bias(perturbed)
        assert output["forces"][:3].abs().max() == 0
        assert output["forces"][3:].abs().max() > 0

    def test_reset(self, ethanol_state: ts.SimState) -> None:
        bias = RMSDCV(device=DEVICE, dtype=DTYPE)
        bias(ethanol_state)
        bias.reset()
        assert bias.ref_buf is None
        output = bias(ethanol_state)  # re-seeds without error
        assert output["energy"].abs().max() == 0


class TestIntegrationWithSumModel:
    @pytest.fixture
    def lj_model(self) -> LennardJonesModel:
        return LennardJonesModel(
            sigma=2.0,
            epsilon=0.01,
            device=DEVICE,
            dtype=DTYPE,
            compute_forces=True,
            cutoff=5.0,
        )

    def test_nvt_with_wall_and_rmsd_bias(
        self, lj_model: LennardJonesModel, ethanol_state: ts.SimState
    ) -> None:
        wall = LogfermiWall(radius=6.0, device=DEVICE, dtype=DTYPE)
        bias = RMSDCV(k_push=0.005, update_interval=2, device=DEVICE, dtype=DTYPE)
        model = SumModel(lj_model, wall, bias)

        final_state = ts.integrate(
            system=ethanol_state,
            model=model,
            integrator=ts.Integrator.nvt_langevin,
            n_steps=10,
            timestep=0.001,
            temperature=300,
        )
        assert torch.isfinite(final_state.positions).all()
        assert torch.isfinite(final_state.energy).all()
        assert bias.ref_buf is not None
        assert bias.ref_buf.shape[0] > 1
