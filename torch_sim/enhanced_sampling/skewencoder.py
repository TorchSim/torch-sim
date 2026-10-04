"""Skewencoder: a 1-D-bottleneck autoencoder with a skewness auxiliary loss.

The Skewencoder learns a one-dimensional latent collective variable from
structural descriptors. Its training objective combines an autoencoder
reconstruction loss (on the *global* descriptor history) with an auxiliary loss
(on the *local*, most-recent segment) that rewards a large-magnitude skewness of
the latent distribution -- the signal Loxodynamics uses to pick a biasing
direction.

This is a lightweight, pure-PyTorch adaptation of the reference ``skewencoder``
package (MIT licensed) by the original authors; only the model structure, the
``log(1 + exp(-skew^2))`` skewness loss, and the shifted-softplus activation are
reused. None of the reference package's Lightning / mlcolvar / PLUMED machinery
is required here.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import nn


_ACTIVATIONS = ("shifted_softplus", "tanh", "relu", "silu")


@dataclass
class SkewencoderConfig:
    """Configuration for a :class:`Skewencoder` and its trainer.

    Attributes:
        input_dim: Number of input descriptors.
        hidden_dims: Encoder hidden layer widths; the decoder mirrors them.
            The default ``(90, 40, 20, 5)`` follows the reference architecture.
        latent_dim: Bottleneck width. Must be 1 in this version.
        activation: One of ``"shifted_softplus"``, ``"tanh"``, ``"relu"``,
            ``"silu"``.
        alpha: Coefficient of the skewness auxiliary loss.
        beta: Coefficient of the L2 weight regularization.
        learning_rate: Adam learning rate.
        batch_size: Minibatch size for the global reconstruction term.
        max_epochs: Maximum training epochs per call.
        early_stopping_patience: Epochs without improvement before stopping.
        min_delta: Minimum total-loss improvement counted as progress.
        eps: Numerical-stability floor used in skewness/normalization.
        verbose: If True, print per-epoch training diagnostics to stdout.
        verbose_stride: Print every this many epochs when ``verbose`` (the first
            and last epoch are always printed).
    """

    input_dim: int
    hidden_dims: tuple[int, ...] = (90, 40, 20, 5)
    latent_dim: int = 1
    activation: str = "shifted_softplus"
    alpha: float = 0.1
    beta: float = 1.0e-5
    learning_rate: float = 1.0e-3
    batch_size: int = 128
    max_epochs: int = 200
    early_stopping_patience: int = 10
    min_delta: float = 1.0e-6
    eps: float = 1.0e-12
    verbose: bool = False
    verbose_stride: int = 1

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.latent_dim != 1:
            raise ValueError(f"{self.latent_dim=} must be 1 in this version")
        if self.input_dim < 1:
            raise ValueError(f"{self.input_dim=} must be >= 1")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"{self.activation=} must be one of {_ACTIVATIONS}")


class ShiftedSoftplus(nn.Module):
    """Shifted softplus activation ``softplus(x) - log(2)``.

    Matches the reference implementation: softplus shifted so that the output is
    zero at the origin, giving a smooth, everywhere-differentiable nonlinearity.
    """

    def __init__(self) -> None:
        """Initialize the activation."""
        super().__init__()
        self._softplus = nn.Softplus()
        self._shift = math.log(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shifted softplus."""
        return self._softplus(x) - self._shift


def _make_activation(name: str) -> nn.Module:
    """Return a fresh activation module for the given name."""
    if name == "shifted_softplus":
        return ShiftedSoftplus()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation {name!r}")


def _build_mlp(dims: list[int], activation: str) -> nn.Sequential:
    """Build an MLP over ``dims`` with the activation between hidden layers.

    No activation is applied after the final linear layer, so the encoder
    bottleneck and the decoder output stay linear.
    """
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(_make_activation(activation))
    return nn.Sequential(*layers)


class Skewencoder(nn.Module):
    """Autoencoder with a one-dimensional latent bottleneck.

    The encoder maps ``input_dim`` descriptors through ``hidden_dims`` down to a
    single latent value; the decoder mirrors that path back to ``input_dim``.

    Args:
        config: The :class:`SkewencoderConfig`.
    """

    def __init__(self, config: SkewencoderConfig) -> None:
        """Build the encoder/decoder from the config."""
        super().__init__()
        self.config = config
        enc_dims = [config.input_dim, *config.hidden_dims, config.latent_dim]
        self.encoder = _build_mlp(enc_dims, config.activation)
        self.decoder = _build_mlp(list(reversed(enc_dims)), config.activation)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode descriptors ``[n_samples, input_dim]`` to latent ``[n_samples, 1]``."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent ``[n_samples, 1]`` back to ``[n_samples, input_dim]``."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(reconstruction, latent)`` for inputs ``[n_samples, input_dim]``."""
        z = self.encode(x)
        return self.decode(z), z


def skewness_1d(values: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """Differentiable sample skewness of a 1-D set of values.

    Args:
        values: Any tensor; it is flattened to 1-D.
        eps: Numerical-stability floor for the variance.

    Returns:
        A scalar tensor with the (biased) sample skewness
        ``mean(c^3) / (mean(c^2) + eps)^{1.5}`` where ``c = values - mean``.
    """
    flat = values.reshape(-1)
    centered = flat - flat.mean()
    m2 = centered.pow(2).mean()
    m3 = centered.pow(3).mean()
    return m3 / (m2 + eps).pow(1.5)


def skewness_loss(latent_local: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """Skewness auxiliary loss ``log(1 + exp(-skew^2)) = softplus(-skew^2)``.

    Minimizing this drives the latent skewness toward large magnitude (either
    sign), so the latent distribution develops a pronounced tail.

    Args:
        latent_local: Latent values of the local segment, any shape.
        eps: Numerical-stability floor passed to :func:`skewness_1d`.

    Returns:
        A scalar loss tensor.
    """
    gamma = skewness_1d(latent_local, eps=eps)
    return torch.nn.functional.softplus(-gamma.pow(2))


def _l2_weights(model: Skewencoder) -> torch.Tensor:
    """Sum of squared trainable parameters (weights and biases, not buffers)."""
    total = None
    for param in model.parameters():
        if not param.requires_grad:
            continue
        term = param.pow(2).sum()
        total = term if total is None else total + term
    if total is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return total


def skewencoder_loss(
    model: Skewencoder,
    global_x: torch.Tensor,
    local_x: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    eps: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the multitask Skewencoder loss.

    ``L = L_AE(global_x) + alpha * L_skew(local_x) + beta * L2_weights``

    Reconstruction is taken on the global dataset and the skewness term on the
    local dataset.

    Args:
        model: The Skewencoder.
        global_x: Normalized global descriptors ``[n_global, input_dim]``.
        local_x: Normalized local descriptors ``[n_local, input_dim]``.
        alpha: Skewness loss coefficient.
        beta: L2 regularization coefficient.
        eps: Numerical-stability floor for skewness.

    Returns:
        A tuple ``(loss_total, diagnostics)`` where ``diagnostics`` holds
        detached scalars: ``loss_total``, ``loss_reconstruction``,
        ``loss_skew``, ``loss_l2``, ``local_skewness``.
    """
    recon, _ = model(global_x)
    loss_ae = torch.nn.functional.mse_loss(recon, global_x)

    z_local = model.encode(local_x)
    loss_skew = skewness_loss(z_local, eps=eps)
    l2 = _l2_weights(model)

    loss_total = loss_ae + alpha * loss_skew + beta * l2
    diagnostics = {
        "loss_total": loss_total.detach(),
        "loss_reconstruction": loss_ae.detach(),
        "loss_skew": loss_skew.detach(),
        "loss_l2": (beta * l2).detach(),
        "local_skewness": skewness_1d(z_local, eps=eps).detach(),
    }
    return loss_total, diagnostics


@dataclass
class DescriptorNormalizer:
    """Affine descriptor normalizer ``(x - mean) / std``.

    Attributes:
        mean: Per-descriptor mean ``[input_dim]``.
        std: Per-descriptor standard deviation ``[input_dim]`` (clamped by eps).
        eps: Numerical-stability floor for the standard deviation.
    """

    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1.0e-12

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` to zero mean / unit variance per descriptor."""
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`transform`."""
        return x * self.std + self.mean

    def to(self, device: torch.device, dtype: torch.dtype) -> DescriptorNormalizer:
        """Return a copy with the statistics moved to ``device``/``dtype``."""
        return DescriptorNormalizer(
            mean=self.mean.to(device, dtype),
            std=self.std.to(device, dtype),
            eps=self.eps,
        )


def fit_descriptor_normalizer(
    x: torch.Tensor, eps: float = 1.0e-12
) -> DescriptorNormalizer:
    """Fit a :class:`DescriptorNormalizer` from descriptor samples.

    Args:
        x: Descriptor samples ``[n_samples, input_dim]``.
        eps: Floor applied to the standard deviation.

    Returns:
        A normalizer whose ``std`` is clamped to at least ``eps``.
    """
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(eps)
    return DescriptorNormalizer(mean=mean, std=std, eps=eps)


@dataclass
class SkewencoderTrainingReport:
    """Summary of a single :meth:`SkewencoderTrainer.train` call."""

    n_epochs: int
    final_loss: float
    final_reconstruction_loss: float
    final_skew_loss: float
    final_l2_loss: float
    final_local_skewness: float
    stopped_early: bool
    train_time_s: float = 0.0
    """Wall-clock time spent in :meth:`SkewencoderTrainer.train`, in seconds."""


class SkewencoderTrainer:
    """Adam trainer for the multitask Skewencoder loss with early stopping.

    Args:
        config: The :class:`SkewencoderConfig` (hyperparameters and stopping).
    """

    def __init__(self, config: SkewencoderConfig) -> None:
        """Store the training configuration."""
        self.config = config

    def train(  # noqa: C901
        self,
        model: Skewencoder,
        global_descriptors: torch.Tensor,
        local_descriptors: torch.Tensor,
        *,
        normalizer: DescriptorNormalizer | None = None,
    ) -> tuple[DescriptorNormalizer, SkewencoderTrainingReport]:
        """Train ``model`` in place; warm-start by reusing the same instance.

        Args:
            model: Skewencoder to train (modified in place).
            global_descriptors: Raw global descriptors ``[n_global, input_dim]``
                used for the reconstruction loss.
            local_descriptors: Raw local descriptors ``[n_local, input_dim]``
                used for the skewness loss.
            normalizer: Optional fixed normalizer; if ``None`` one is fit from
                ``global_descriptors``.

        Returns:
            A tuple ``(normalizer, report)``.
        """
        cfg = self.config
        t_start = time.perf_counter()
        param = next(model.parameters())
        device, dtype = param.device, param.dtype

        global_x = global_descriptors.to(device, dtype)
        local_x = local_descriptors.to(device, dtype)
        if normalizer is None:
            normalizer = fit_descriptor_normalizer(global_x, eps=cfg.eps)
        normalizer = normalizer.to(device, dtype)

        gx = normalizer.transform(global_x)
        lx = normalizer.transform(local_x)

        # Re-enable grad in case a previous wall put the shared encoder in a
        # frozen/eval state; warm-start retraining needs trainable parameters.
        for param in model.parameters():
            param.requires_grad_(requires_grad=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        n_global = gx.shape[0]
        batch_size = min(cfg.batch_size, n_global)

        best_loss = math.inf
        patience = 0
        stopped_early = False
        n_epochs = 0
        if cfg.verbose:
            print(  # noqa: T201
                f"    [skewencoder] train: n_global={n_global} n_local={lx.shape[0]} "
                f"max_epochs={cfg.max_epochs} batch={batch_size}",
                flush=True,
            )
        model.train()
        for epoch in range(1, cfg.max_epochs + 1):
            n_epochs = epoch
            perm = torch.randperm(n_global, device=device)
            for start in range(0, n_global, batch_size):
                idx = perm[start : start + batch_size]
                loss, _ = skewencoder_loss(
                    model, gx[idx], lx, alpha=cfg.alpha, beta=cfg.beta, eps=cfg.eps
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                epoch_loss, epoch_diag = skewencoder_loss(
                    model, gx, lx, alpha=cfg.alpha, beta=cfg.beta, eps=cfg.eps
                )
            current = float(epoch_loss)
            if best_loss - current > cfg.min_delta:
                best_loss = current
                patience = 0
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    stopped_early = True

            if cfg.verbose and (
                epoch == 1 or stopped_early or epoch % cfg.verbose_stride == 0
            ):
                print(  # noqa: T201
                    f"      epoch {epoch:>4d}/{cfg.max_epochs}  "
                    f"loss={current:.4e}  "
                    f"recon={float(epoch_diag['loss_reconstruction']):.4e}  "
                    f"skew={float(epoch_diag['loss_skew']):.4e}  "
                    f"gamma={float(epoch_diag['local_skewness']):+.4f}  "
                    f"patience={patience}",
                    flush=True,
                )
            if stopped_early:
                break

        if cfg.verbose:
            print(  # noqa: T201
                f"    [skewencoder] done: {n_epochs} epochs, "
                f"stopped_early={stopped_early}, best_loss={best_loss:.4e}",
                flush=True,
            )
        model.eval()
        with torch.no_grad():
            _, diag = skewencoder_loss(
                model, gx, lx, alpha=cfg.alpha, beta=cfg.beta, eps=cfg.eps
            )
        report = SkewencoderTrainingReport(
            n_epochs=n_epochs,
            final_loss=float(diag["loss_total"]),
            final_reconstruction_loss=float(diag["loss_reconstruction"]),
            final_skew_loss=float(diag["loss_skew"]),
            final_l2_loss=float(diag["loss_l2"]),
            final_local_skewness=float(diag["local_skewness"]),
            stopped_early=stopped_early,
            train_time_s=time.perf_counter() - t_start,
        )
        return normalizer, report
