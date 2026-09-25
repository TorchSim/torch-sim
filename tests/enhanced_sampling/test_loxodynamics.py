from pathlib import Path

import pytest
import torch
from ase.build import molecule

import torch_sim as ts
from torch_sim.enhanced_sampling.loxodynamics import (
    LoxodynamicsWall,
    PairDistanceDescriptor,
    run_loxodynamics,
)
from torch_sim.enhanced_sampling.skewencoder import (
    Skewencoder,
    SkewencoderConfig,
    fit_descriptor_normalizer,
)
from torch_sim.models.interface import ModelInterface


DEVICE = torch.device("cpu")
DTYPE = torch.float64


class HarmonicModel(ModelInterface):
    """Per-atom harmonic well E = 1/2 k sum (x - x0)^2 (single bound system)."""

    def __init__(
        self, centers: torch.Tensor, k: float = 1.0, dtype: torch.dtype = DTYPE
    ) -> None:
        super().__init__()
        self._device = DEVICE
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = False
        self.k = float(k)
        self.register_buffer("centers", centers.to(DEVICE, dtype))

    def forward(self, state: ts.SimState, **_kwargs) -> dict[str, torch.Tensor]:
        disp = state.positions - self.centers
        e_atom = 0.5 * self.k * disp.pow(2).sum(dim=-1)
        energy = torch.zeros(state.n_systems, device=DEVICE, dtype=self._dtype).index_add(
            0, state.system_idx, e_atom
        )
        return {"energy": energy, "forces": -self.k * disp}


@pytest.fixture
def water_state() -> ts.SimState:
    return ts.io.atoms_to_state([molecule("H2O")], DEVICE, DTYPE)


def _all_pairs_3() -> torch.Tensor:
    return torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.long)


class TestPairDistanceDescriptor:
    def test_correct_distances(self) -> None:
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=DTYPE
        )
        desc = PairDistanceDescriptor(_all_pairs_3())
        assert desc.n_descriptors == 3
        d = desc(positions)
        torch.testing.assert_close(d, torch.tensor([3.0, 4.0, 5.0], dtype=DTYPE))

    def test_gradients_flow(self) -> None:
        positions = torch.randn(3, 3, dtype=DTYPE, requires_grad=True)
        desc = PairDistanceDescriptor(_all_pairs_3())
        desc(positions).sum().backward()
        assert positions.grad is not None
        assert torch.isfinite(positions.grad).all()

    def test_bad_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="n_pairs, 2"):
            PairDistanceDescriptor(torch.tensor([0, 1, 2], dtype=torch.long))


class TestLoxodynamicsWall:
    def _components(self):
        """Deterministic (descriptor, encoder, fitted normalizer) for the wall."""
        torch.manual_seed(0)
        desc = PairDistanceDescriptor(_all_pairs_3())
        enc = Skewencoder(SkewencoderConfig(input_dim=3, hidden_dims=(8, 4))).to(DTYPE)
        sample = torch.randn(20, 3, dtype=DTYPE).abs() + 1.0
        return desc, enc, fit_descriptor_normalizer(sample)

    def _wall(self, *, offset: float) -> tuple[LoxodynamicsWall, int]:
        desc, enc, norm = self._components()
        wall = LoxodynamicsWall(
            desc,
            enc,
            norm,
            mu=0.0,
            sigma=1.0,  # standardized wall divides by sigma; use a unit scale
            skewness=1.0,  # positive -> sign +1, lower wall
            kappa=1.0,
            offset=offset,
            device=DEVICE,
            dtype=DTYPE,
        )
        return wall, 3

    def test_shapes(self, water_state: ts.SimState) -> None:
        wall, _ = self._wall(offset=1.0)
        out = wall(water_state)
        assert out["energy"].shape == (1,)
        assert out["forces"].shape == (water_state.n_atoms, 3)

    def test_zero_beyond_wall(self, water_state: ts.SimState) -> None:
        # a very negative offset puts the CV beyond the wall -> no penalty
        wall, _ = self._wall(offset=-1000.0)
        out = wall(water_state)
        assert out["energy"].item() == 0.0
        assert out["forces"].abs().max() == 0.0

    def test_positive_when_violating(self, water_state: ts.SimState) -> None:
        # a large positive offset forces a violation -> positive penalty
        wall, _ = self._wall(offset=1000.0)
        out = wall(water_state)
        assert out["energy"].item() > 0.0

    def test_rejects_multiple_systems(self) -> None:
        two = ts.io.atoms_to_state([molecule("H2O"), molecule("H2O")], DEVICE, DTYPE)
        wall, _ = self._wall(offset=1.0)
        with pytest.raises(ValueError, match="single system"):
            wall(two)

    def test_emits_latent_cv(self, water_state: ts.SimState) -> None:
        # the wall reports the raw (unsigned) latent CV under the "loxo_cv" key
        desc, enc, norm = self._components()
        wall = LoxodynamicsWall(
            desc,
            enc,
            norm,
            mu=0.0,
            sigma=1.0,
            skewness=1.0,
            kappa=1.0,
            offset=1.0,
            device=DEVICE,
            dtype=DTYPE,
        )
        out = wall(water_state)
        assert "loxo_cv" in out
        assert out["loxo_cv"].shape == (1,)
        with torch.no_grad():
            latent = enc.encode(
                norm.transform(desc(water_state.positions)).unsqueeze(0)
            ).reshape(1)
        torch.testing.assert_close(out["loxo_cv"], latent)

    def test_energy_scale_invariant_at_reference(self, water_state: ts.SimState) -> None:
        # The standardized wall acts on (s - mu)/sigma, so at the reference point
        # (mu == the current latent) the violation is exactly `offset` and the
        # energy is kappa*offset**2 regardless of sigma. A raw-latent wall would
        # instead give ~kappa*(sigma + offset)**2, which diverges as sigma grows.
        desc, enc, norm = self._components()
        with torch.no_grad():
            s0 = (
                enc.encode(norm.transform(desc(water_state.positions)).unsqueeze(0))
                .reshape(())
                .item()
            )
        energies = [
            LoxodynamicsWall(
                desc,
                enc,
                norm,
                mu=s0,
                sigma=sigma,
                skewness=1.0,
                kappa=1.0,
                offset=2.0,
                device=DEVICE,
                dtype=DTYPE,
            )(water_state)["energy"].item()
            for sigma in (0.1, 1.0, 100.0)
        ]
        for energy in energies:
            assert energy == pytest.approx(4.0, abs=1e-9)  # kappa * offset**2


class TestRunLoxodynamics:
    def _setup(
        self, water_state: ts.SimState
    ) -> tuple[HarmonicModel, PairDistanceDescriptor, SkewencoderConfig]:
        model = HarmonicModel(water_state.positions.clone(), k=2.0)
        desc = PairDistanceDescriptor(_all_pairs_3())
        cfg = SkewencoderConfig(
            input_dim=3,
            hidden_dims=(8, 4),
            max_epochs=5,
            batch_size=16,
            early_stopping_patience=3,
        )
        return model, desc, cfg

    def test_runs_to_budget(self, water_state: ts.SimState) -> None:
        model, desc, cfg = self._setup(water_state)
        result = run_loxodynamics(
            water_state,
            model,
            descriptor=desc,
            max_steps=20,
            segment_steps=5,
            initial_unbiased_steps=5,
            timestep=0.0005,
            temperature=300.0,
            sample_stride=1,
            min_local_samples=3,
            seed=0,
            skewencoder_config=cfg,
        )
        assert result.total_steps == 20
        assert len(result.training_reports) >= 1
        assert result.global_descriptors.shape[1] == 3
        assert result.global_descriptors.shape[0] > 0
        assert len(result.wall_stats) == len(result.training_reports)

    def test_rejects_multiple_systems(self) -> None:
        two = ts.io.atoms_to_state([molecule("H2O"), molecule("H2O")], DEVICE, DTYPE)
        model = HarmonicModel(two.positions.clone(), k=2.0)
        desc = PairDistanceDescriptor(_all_pairs_3())
        with pytest.raises(ValueError, match="single system"):
            run_loxodynamics(
                two,
                model,
                descriptor=desc,
                max_steps=10,
                segment_steps=5,
                timestep=0.0005,
                temperature=300.0,
                min_local_samples=3,
            )

    def test_checkpoint_dir_saves_loadable_models(
        self, water_state: ts.SimState, tmp_path: Path
    ) -> None:
        model, desc, cfg = self._setup(water_state)
        result = run_loxodynamics(
            water_state,
            model,
            descriptor=desc,
            max_steps=20,
            segment_steps=5,
            initial_unbiased_steps=5,
            timestep=0.0005,
            temperature=300.0,
            sample_stride=1,
            min_local_samples=3,
            seed=0,
            skewencoder_config=cfg,
            checkpoint_dir=tmp_path,
        )
        # one checkpoint per retrain/wall
        files = sorted(tmp_path.glob("skewencoder_iter*.pt"))
        assert len(files) == len(result.wall_stats) >= 1
        ckpt = torch.load(files[0], weights_only=False)
        assert {
            "skewencoder_state_dict",
            "skewencoder_config",
            "normalizer",
            "wall_stats",
            "training_report",
        } <= set(ckpt)
        # the saved weights reload into a fresh Skewencoder built from the config
        enc = Skewencoder(SkewencoderConfig(**ckpt["skewencoder_config"]))
        enc.load_state_dict(ckpt["skewencoder_state_dict"])


class TestExecutorDtype:
    def test_float32_model_end_to_end(self) -> None:
        # With no explicit dtype the executor adopts the model's dtype, so the
        # latent wall composes with a float32 model under SumModel and the whole
        # run stays in float32.
        f32_state = ts.io.atoms_to_state([molecule("H2O")], DEVICE, torch.float32)
        model = HarmonicModel(f32_state.positions.clone(), k=2.0, dtype=torch.float32)
        cfg = SkewencoderConfig(
            input_dim=3,
            hidden_dims=(8, 4),
            max_epochs=5,
            batch_size=16,
            early_stopping_patience=3,
        )
        result = run_loxodynamics(
            f32_state,
            model,
            descriptor=PairDistanceDescriptor(_all_pairs_3()),
            max_steps=20,
            segment_steps=5,
            initial_unbiased_steps=5,
            timestep=0.0005,
            temperature=300.0,
            sample_stride=1,
            min_local_samples=3,
            seed=0,
            skewencoder_config=cfg,
        )
        assert result.total_steps == 20
        assert result.global_descriptors.dtype == torch.float32
        assert result.final_state.positions.dtype == torch.float32
        assert len(result.training_reports) >= 1
