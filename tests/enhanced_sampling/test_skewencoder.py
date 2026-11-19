import pytest
import torch

from torch_sim.enhanced_sampling.skewencoder import (
    DescriptorNormalizer,
    Skewencoder,
    SkewencoderConfig,
    SkewencoderTrainer,
    fit_descriptor_normalizer,
    skewencoder_loss,
    skewness_1d,
)


DTYPE = torch.float64


def _small_config(input_dim: int = 4) -> SkewencoderConfig:
    return SkewencoderConfig(
        input_dim=input_dim,
        hidden_dims=(8, 4),
        max_epochs=5,
        batch_size=16,
        early_stopping_patience=3,
    )


class TestSkewness:
    def test_symmetric_near_zero(self) -> None:
        values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=DTYPE)
        assert skewness_1d(values).abs() < 1e-9

    def test_right_tail_positive(self) -> None:
        values = torch.tensor([1.0, 1.0, 1.0, 1.0, 10.0], dtype=DTYPE)
        assert skewness_1d(values) > 0

    def test_left_tail_negative(self) -> None:
        values = torch.tensor([-1.0, -1.0, -1.0, -1.0, -10.0], dtype=DTYPE)
        assert skewness_1d(values) < 0

    def test_differentiable(self) -> None:
        values = torch.randn(50, dtype=DTYPE, requires_grad=True)
        skewness_1d(values).backward()
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()


class TestSkewencoder:
    def test_output_shapes(self) -> None:
        model = Skewencoder(_small_config(input_dim=5)).to(DTYPE)
        x = torch.randn(8, 5, dtype=DTYPE)
        recon, latent = model(x)
        assert recon.shape == (8, 5)
        assert latent.shape == (8, 1)
        assert model.encode(x).shape == (8, 1)

    def test_latent_dim_must_be_one(self) -> None:
        with pytest.raises(ValueError, match="latent_dim"):
            SkewencoderConfig(input_dim=4, latent_dim=2)


class TestSkewencoderLoss:
    def test_finite_and_diagnostics(self) -> None:
        cfg = _small_config(input_dim=4)
        model = Skewencoder(cfg).to(DTYPE)
        global_x = torch.randn(32, 4, dtype=DTYPE)
        local_x = torch.randn(12, 4, dtype=DTYPE)
        loss, diag = skewencoder_loss(
            model, global_x, local_x, alpha=cfg.alpha, beta=cfg.beta, eps=cfg.eps
        )
        assert torch.isfinite(loss)
        assert set(diag) == {
            "loss_total",
            "loss_reconstruction",
            "loss_skew",
            "loss_l2",
            "local_skewness",
        }


class TestNormalizer:
    def test_roundtrip(self) -> None:
        x = torch.randn(100, 4, dtype=DTYPE) * 3 + 5
        norm = fit_descriptor_normalizer(x)
        assert isinstance(norm, DescriptorNormalizer)
        z = norm.transform(x)
        torch.testing.assert_close(
            z.mean(dim=0), torch.zeros(4, dtype=DTYPE), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(norm.inverse_transform(z), x, atol=1e-9, rtol=0)


class TestTrainer:
    def test_trains_and_encodes(self) -> None:
        torch.manual_seed(0)
        cfg = _small_config(input_dim=4)
        model = Skewencoder(cfg).to(DTYPE)
        trainer = SkewencoderTrainer(cfg)
        global_x = torch.randn(64, 4, dtype=DTYPE)
        local_x = torch.randn(20, 4, dtype=DTYPE)

        normalizer, report = trainer.train(model, global_x, local_x)
        assert isinstance(normalizer, DescriptorNormalizer)
        assert 1 <= report.n_epochs <= cfg.max_epochs
        assert report.final_loss == report.final_loss  # not NaN

        latent = model.encode(normalizer.transform(local_x))
        assert latent.shape == (20, 1)
