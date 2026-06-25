"""Loxodynamics: skewness-guided latent-space enhanced sampling.

Loxodynamics explores reactions without a predefined product or hand-built
reaction coordinate. It learns a one-dimensional latent collective variable from
structural descriptors with a
:class:`~torch_sim.enhanced_sampling.skewencoder.Skewencoder`,
reads the *skewness* of the local latent distribution to decide which way the
basin has a low-barrier exit, and erects a half-harmonic wall in latent
space that pushes the system that way. After each biased MD segment it appends
the new samples to a global buffer, warm-start retrains the same Skewencoder,
and rebuilds the wall, iterating until the total step budget is spent.

This first version is an iterative biasing/exploration executor only: no
product-state detection, restart protocol, anti-backtracking, BXD boundaries,
PMF reconstruction, or multi-walker swarms. It requires a single system
(``state.n_systems == 1``). See ``instruction.md`` for the full scope.

Idea & default parameters are from 10.1038/s41467-026-69586-8.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from torch_sim.enhanced_sampling.skewencoder import (
    DescriptorNormalizer,
    Skewencoder,
    SkewencoderConfig,
    SkewencoderTrainer,
    SkewencoderTrainingReport,
    skewness_1d,
)
from torch_sim.integrators.md import MDState
from torch_sim.integrators.nvt import nvt_langevin_init, nvt_langevin_step
from torch_sim.models.interface import ModelInterface, SumModel
from torch_sim.units import UnitSystem


if TYPE_CHECKING:
    from torch_sim.state import SimState
    from torch_sim.trajectory import TrajectoryReporter


EPS = 1e-12


class PairDistanceDescriptor(torch.nn.Module):
    """Differentiable interatomic pair-distance descriptor (no PBC in v0).

    Args:
        pairs: Long tensor of atom index pairs with shape ``[n_pairs, 2]``.
        eps: Numerical-stability floor added under the square root.
    """

    pairs: torch.Tensor

    def __init__(self, pairs: torch.Tensor, *, eps: float = 1.0e-12) -> None:
        """Validate and store the pair list."""
        super().__init__()
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(
                f"pairs must have shape [n_pairs, 2], got {tuple(pairs.shape)}"
            )
        if pairs.shape[0] < 1:
            raise ValueError("pairs must contain at least one pair")
        self.eps = float(eps)
        self.register_buffer("pairs", pairs.long())

    @property
    def n_descriptors(self) -> int:
        """Number of pair distances produced."""
        return int(self.pairs.shape[0])

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute pair distances ``[n_pairs]`` from positions ``[n_atoms, 3]``."""
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"positions must have shape [n_atoms, 3], got {tuple(positions.shape)}"
            )
        i_idx, j_idx = self.pairs[:, 0], self.pairs[:, 1]
        diff = positions[i_idx] - positions[j_idx]
        return torch.sqrt(diff.pow(2).sum(dim=-1) + self.eps)


@dataclass
class LoxodynamicsWallStats:
    """Latent-space statistics and wall placement for one iteration."""

    iteration: int
    mu: float
    sigma: float
    skewness: float
    sign: float
    boundary: float
    n_local_samples: int
    n_global_samples: int


class LoxodynamicsWall(ModelInterface):
    """Half-harmonic wall in the *standardized* 1-D latent CV.

    The wall acts on the standardized latent ``z = (s - mu) / sigma``, where
    ``s = encode(normalize(descriptor))`` and ``mu``/``sigma`` are the local
    latent mean/std at build time. It confines ``z`` on the side opposite the
    skew tail, pushing the system toward the basin's low-barrier exit:

        scaled_z  = sign * (s - mu) / sigma
        violation = relu(offset - scaled_z)
        E_wall    = kappa * violation**2

    .. note::
        This **standardized** wall is a deliberate departure from the original
        Loxodynamics method (10.1038/s41467-026-69586-8), which places the
        half-harmonic wall directly on the **raw** latent ``s`` (``offset`` in
        raw latent units). We found the raw-latent wall numerically unstable:
        the latent ``s`` has an arbitrary, unanchored scale -- the encoder's
        output magnitude drifts or jumps across warm-start retrains -- so a raw
        wall ``relu(mu + sigma + offset - sign*s)`` injects an energy
        ``~kappa*(sigma + offset)**2`` that diverges as ``sigma`` grows large,
        producing a force spike that detonates the integrator (observed e.g. on
        rigid molecules whose autoencoder escapes a degenerate solution into a
        large-magnitude latent). Standardizing by ``sigma`` removes this: the
        initial violation is ``offset`` and the initial energy is
        ``kappa*offset**2`` regardless of ``sigma``, and the force scales as
        ``(1/sigma) * ds/dx`` so the encoder's scale cancels. Consequently
        ``offset`` here is in units of ``sigma`` (standard deviations past the
        mean), not raw latent units as in the paper.

    Args:
        descriptor: Pair-distance descriptor.
        skewencoder: Trained Skewencoder (used in eval mode, parameters frozen).
        normalizer: Descriptor normalizer fit on the global buffer.
        mu: Local latent mean.
        sigma: Local latent standard deviation.
        skewness: Local latent skewness (sets the wall orientation).
        kappa: Harmonic wall force constant.
        offset: Wall margin in units of the local latent std ``sigma`` (i.e. how
            many ``sigma`` past the mean to push the CV before the wall relaxes).
        min_abs_skew: Below this magnitude the skew sign is treated as positive.
        min_sigma: Floor applied to ``sigma`` to keep the standardization well
            defined when the latent is nearly degenerate.
        energy_label: Non-canonical output key for this wall's energy.
        cv_label: Non-canonical output key for the (raw) latent collective variable.
        device: Computation device. Defaults to CPU.
        dtype: Floating-point dtype. Defaults to ``torch.float64``.
    """

    def __init__(
        self,
        descriptor: PairDistanceDescriptor,
        skewencoder: Skewencoder,
        normalizer: DescriptorNormalizer,
        *,
        mu: torch.Tensor | float,
        sigma: torch.Tensor | float,
        skewness: torch.Tensor | float,
        kappa: torch.Tensor | float,
        offset: torch.Tensor | float,
        min_abs_skew: float = 1.0e-8,
        min_sigma: float = 1.0e-6,
        energy_label: str = "PE_LoxodynamicsWall",
        cv_label: str = "loxo_cv",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Build the wall from trained components and latent statistics."""
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = False
        self._memory_scales_with = "n_atoms"
        self.energy_label = str(energy_label)
        self.cv_label = str(cv_label)

        self.descriptor = descriptor.to(self._device)
        # Eval mode only; do not freeze parameters -- the encoder instance is
        # shared with the executor and warm-start retrained between segments.
        # Forces use autograd.grad(energy, [positions]), which differentiates
        # w.r.t. positions only, so parameter requires_grad is irrelevant here.
        self.skewencoder = skewencoder.to(self._device, self._dtype).eval()
        self.normalizer = normalizer.to(self._device, self._dtype)

        mu_t = torch.as_tensor(mu, device=self._device, dtype=self._dtype)
        sigma_t = torch.as_tensor(
            sigma, device=self._device, dtype=self._dtype
        ).clamp_min(min_sigma)
        skew_t = torch.as_tensor(skewness, device=self._device, dtype=self._dtype)
        offset_t = torch.as_tensor(offset, device=self._device, dtype=self._dtype)
        sign = torch.where(
            skew_t.abs() < min_abs_skew,
            torch.ones((), device=self._device, dtype=self._dtype),
            torch.sign(skew_t),
        )
        # Raw-latent value where the wall relaxes (z = offset), for reporting.
        boundary = mu_t + sign * offset_t * sigma_t
        self.register_buffer("sign", sign)
        self.register_buffer("mu", mu_t)
        self.register_buffer("sigma", sigma_t)
        self.register_buffer("offset", offset_t)
        self.register_buffer("boundary", boundary)
        self.register_buffer(
            "kappa", torch.as_tensor(kappa, device=self._device, dtype=self._dtype)
        )

    def forward(self, state: SimState, **_kwargs) -> dict[str, torch.Tensor]:
        """Compute wall energy ``[1]`` and forces ``[n_atoms, 3]`` from positions."""
        if state.n_systems != 1:
            raise ValueError(
                f"LoxodynamicsWall supports a single system, got {state.n_systems}"
            )
        with torch.enable_grad():
            pos = state.positions.detach().requires_grad_(requires_grad=True)
            desc = self.descriptor(pos)
            norm_desc = self.normalizer.transform(desc)
            latent = self.skewencoder.encode(norm_desc.unsqueeze(0)).reshape(())
            scaled_z = self.sign * (latent - self.mu) / self.sigma
            violation = torch.relu(self.offset - scaled_z)
            energy_scalar = self.kappa * violation.pow(2)
            grad = torch.autograd.grad(energy_scalar, pos)[0]

        forces = -grad
        energy = energy_scalar.detach().reshape(1)
        return {
            "energy": energy,
            "forces": forces.detach(),
            self.energy_label: energy,
            # The latent CV (raw, un-signed) so the trajectory can record the
            # collective variable the wall acts on, step by step.
            self.cv_label: latent.detach().reshape(1),
        }


@dataclass
class LoxodynamicsResult:
    """Outcome of a Loxodynamics run."""

    final_state: MDState
    wall_stats: list[LoxodynamicsWallStats]
    training_reports: list[SkewencoderTrainingReport]
    global_descriptors: torch.Tensor
    total_steps: int
    segment_times: list[float] = field(default_factory=list)
    """Wall-clock seconds per MD segment; index 0 is the initial unbiased
    segment, index ``i + 1`` is the biased segment that follows iteration ``i``."""


class LoxodynamicsExecutor:
    """Trajectory-level Loxodynamics controller (single system).

    Runs an initial unbiased segment, then repeatedly trains the Skewencoder,
    builds a latent wall, and runs a biased segment, until ``max_steps``
    attempted MD steps have been taken.

    Args:
        model: Base energy/force model.
        descriptor: Pair-distance descriptor.
        max_steps: Total attempted-step budget (the only stopping criterion).
        segment_steps: Steps per biased segment.
        timestep: Timestep in ``unit_system`` time units.
        temperature: Temperature in Kelvin.
        initial_unbiased_steps: Steps in the first unbiased segment. Defaults to
            ``segment_steps``.
        sample_stride: Collect a descriptor sample every this many steps.
        kappa: Wall force constant.
        wall_offset: Wall margin in units of the local latent std ``sigma`` --
            how many ``sigma`` past the mean to push the standardized CV before
            the wall relaxes. Defaults to 1.0 (one standard deviation).
        global_buffer_capacity: Max global descriptor samples retained (newest).
        min_local_samples: Minimum local samples required before training.
        gamma: Optional Langevin friction.
        seed: Optional RNG seed.
        unit_system: Unit system for conversions. Defaults to metal units.
        skewencoder_config: Optional config; one is built from the descriptor if
            omitted.
        trajectory_reporter: Optional reporter recording every attempted step.
        checkpoint_dir: If set, write a Skewencoder checkpoint after each retrain
            to ``<checkpoint_dir>/skewencoder_iter<N>.pt`` (model weights, config,
            normalizer, latent stats, and training report). The directory is
            created if needed.
        device: Computation device. Defaults to ``model.device``.
        dtype: Working dtype for the Skewencoder, wall, and descriptor buffers.
            Defaults to ``model.dtype`` so the wall composes with the base model
            under :class:`~torch_sim.models.interface.SumModel`.
        verbose: If True, print a progress hint before each segment/retrain and a
            timing summary at the end.
        **init_kwargs: Extra arguments forwarded to ``nvt_langevin_init``.
    """

    def __init__(
        self,
        model: ModelInterface,
        descriptor: PairDistanceDescriptor,
        *,
        max_steps: int,
        segment_steps: int,
        timestep: float,
        temperature: float,
        initial_unbiased_steps: int | None = None,
        sample_stride: int = 1,
        kappa: float = 1.0,
        wall_offset: float = 1.0,
        global_buffer_capacity: int = 50000,
        min_local_samples: int = 10,
        gamma: torch.Tensor | float | None = None,
        seed: int | None = None,
        unit_system: UnitSystem = UnitSystem.metal,
        skewencoder_config: SkewencoderConfig | None = None,
        trajectory_reporter: TrajectoryReporter | None = None,
        checkpoint_dir: str | Path | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        verbose: bool = False,
        **init_kwargs: Any,
    ) -> None:
        """Validate arguments and store configuration."""
        if max_steps < 1:
            raise ValueError(f"{max_steps=} must be >= 1")
        if segment_steps < 1:
            raise ValueError(f"{segment_steps=} must be >= 1")
        if sample_stride < 1:
            raise ValueError(f"{sample_stride=} must be >= 1")
        if min_local_samples < 1:
            raise ValueError(f"{min_local_samples=} must be >= 1")

        if skewencoder_config is None:
            skewencoder_config = SkewencoderConfig(input_dim=descriptor.n_descriptors)
        if descriptor.n_descriptors != skewencoder_config.input_dim:
            raise ValueError(
                f"descriptor.n_descriptors ({descriptor.n_descriptors}) != "
                f"skewencoder_config.input_dim ({skewencoder_config.input_dim})"
            )

        self.model = model
        self.descriptor = descriptor
        self.config = skewencoder_config
        self.max_steps = int(max_steps)
        self.segment_steps = int(segment_steps)
        self.timestep = timestep
        self.temperature = temperature
        self.initial_unbiased_steps = (
            int(initial_unbiased_steps)
            if initial_unbiased_steps is not None
            else int(segment_steps)
        )
        self.sample_stride = int(sample_stride)
        self.kappa = kappa
        self.wall_offset = wall_offset
        self.global_buffer_capacity = int(global_buffer_capacity)
        self.min_local_samples = int(min_local_samples)
        self.gamma = gamma
        self.seed = seed
        self.unit_system = unit_system
        self.trajectory_reporter = trajectory_reporter
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.verbose = bool(verbose)
        self._device = device or model.device
        self._dtype = dtype if dtype is not None else model.dtype
        self.init_kwargs = init_kwargs

    def _run_segment(
        self,
        state: MDState,
        model: ModelInterface,
        n_steps: int,
        step_offset: int,
        dt: torch.Tensor,
        kT: float,
    ) -> tuple[MDState, list[torch.Tensor], int]:
        """Run ``n_steps`` Langevin steps, collecting descriptors and reporting."""
        local: list[torch.Tensor] = []
        for i in range(n_steps):
            state = nvt_langevin_step(state, model, dt=dt, kT=kT, gamma=self.gamma)
            if (i + 1) % self.sample_stride == 0:
                local.append(self.descriptor(state.positions).detach())
            if self.trajectory_reporter is not None:
                self.trajectory_reporter.report(state, step_offset + i + 1, self.model)
        return state, local, n_steps

    def _extend_global(
        self, global_buf: list[torch.Tensor], new: list[torch.Tensor]
    ) -> None:
        """Append new descriptors and keep only the newest ``capacity`` samples."""
        global_buf.extend(d.to(self._device, self._dtype) for d in new)
        if len(global_buf) > self.global_buffer_capacity:
            del global_buf[: len(global_buf) - self.global_buffer_capacity]

    def _save_checkpoint(
        self,
        iteration: int,
        skewencoder: Skewencoder,
        normalizer: DescriptorNormalizer,
        stats: LoxodynamicsWallStats,
        report: SkewencoderTrainingReport,
    ) -> Path:
        """Snapshot the (warm-started) Skewencoder and its context to disk."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"skewencoder_iter{iteration}.pt"
        config = asdict(self.config) if is_dataclass(self.config) else self.config
        torch.save(
            {
                "iteration": iteration,
                "skewencoder_state_dict": {
                    k: v.detach().cpu() for k, v in skewencoder.state_dict().items()
                },
                "skewencoder_config": config,
                "normalizer": {
                    "mean": normalizer.mean.detach().cpu(),
                    "std": normalizer.std.detach().cpu(),
                    "eps": normalizer.eps,
                },
                "wall_stats": asdict(stats),
                "training_report": asdict(report),
            },
            path,
        )
        return path

    def run(self, state: SimState | MDState) -> LoxodynamicsResult:  # noqa: C901, PLR0915
        """Execute the Loxodynamics loop and return the result."""
        if state.n_systems != 1:
            raise ValueError(
                f"Loxodynamics supports a single system, got {state.n_systems}"
            )

        kT = float(self.temperature) * self.unit_system.temperature
        dt = torch.as_tensor(
            self.timestep * self.unit_system.time,
            device=self._device,
            dtype=self._dtype,
        )

        if self.seed is not None:
            state.rng = self.seed
        if not isinstance(state, MDState):
            state = nvt_langevin_init(state, self.model, kT=kT, **self.init_kwargs)
        if self.trajectory_reporter is not None:
            self.trajectory_reporter.report(state, 0, self.model)

        skewencoder = Skewencoder(self.config).to(self._device, self._dtype)
        trainer = SkewencoderTrainer(self.config)

        global_buf: list[torch.Tensor] = []
        wall_stats: list[LoxodynamicsWallStats] = []
        training_reports: list[SkewencoderTrainingReport] = []
        segment_times: list[float] = []

        t_run = time.perf_counter()
        total_steps = 0
        # initial unbiased segment
        n0 = min(self.initial_unbiased_steps, self.max_steps)
        if self.verbose:
            print(  # noqa: T201
                f"[loxodynamics] Now start unbiased sampling: {n0} steps at "
                f"{self.temperature:g} K, dt={self.timestep:g}",
                flush=True,
            )
        t_seg = time.perf_counter()
        state, local, used = self._run_segment(state, self.model, n0, total_steps, dt, kT)
        segment_times.append(time.perf_counter() - t_seg)
        total_steps += used
        self._extend_global(global_buf, local)
        if self.verbose:
            print(  # noqa: T201
                f"[loxodynamics] unbiased segment done: {used} steps in "
                f"{segment_times[-1]:.1f} s",
                flush=True,
            )

        iteration = 0
        while total_steps < self.max_steps:
            if len(local) < self.min_local_samples:
                raise ValueError(
                    f"collected only {len(local)} local samples (< "
                    f"{self.min_local_samples}); increase segment length or "
                    "decrease sample_stride / min_local_samples"
                )

            if self.verbose:
                print(  # noqa: T201
                    f"[loxodynamics] iter {iteration}: retraining Skewencoder on "
                    f"{len(global_buf)} global / {len(local)} local samples ...",
                    flush=True,
                )
            global_t = torch.stack(global_buf).to(self._device, self._dtype)
            local_t = torch.stack(local).to(self._device, self._dtype)
            _normalizer, report = trainer.train(skewencoder, global_t, local_t)
            training_reports.append(report)

            with torch.no_grad():
                latent = skewencoder.encode(_normalizer.transform(local_t)).reshape(-1)
                mu = latent.mean()
                sigma = latent.std(unbiased=False)
                skew = skewness_1d(latent, eps=self.config.eps)

            wall = LoxodynamicsWall(
                self.descriptor,
                skewencoder,
                _normalizer,
                mu=mu,
                sigma=sigma,
                skewness=skew,
                kappa=self.kappa,
                offset=self.wall_offset,
                device=self._device,
                dtype=self._dtype,
            )
            wall_stats.append(
                LoxodynamicsWallStats(
                    iteration=iteration,
                    mu=float(mu),
                    sigma=float(sigma),
                    skewness=float(skew),
                    sign=float(wall.sign),
                    boundary=float(wall.boundary),
                    n_local_samples=len(local),
                    n_global_samples=len(global_buf),
                )
            )
            if self.checkpoint_dir is not None:
                ckpt_path = self._save_checkpoint(
                    iteration, skewencoder, _normalizer, wall_stats[-1], report
                )
                if self.verbose:
                    print(  # noqa: T201
                        f"[loxodynamics] saved model checkpoint: {ckpt_path}", flush=True
                    )
            iteration += 1

            remaining = self.max_steps - total_steps
            if remaining <= 0:
                break
            seg = min(self.segment_steps, remaining)
            if self.verbose:
                w = wall_stats[-1]
                print(  # noqa: T201
                    f"[loxodynamics] Now start loxodynamics with wall settings: "
                    f"iter={w.iteration}, sign={w.sign:+.0f}, mu={w.mu:.4g}, "
                    f"sigma={w.sigma:.4g}, skewness={w.skewness:+.4g}, "
                    f"boundary={w.boundary:.4g}, kappa={float(self.kappa):g}, "
                    f"offset={float(self.wall_offset):g}; {seg} steps",
                    flush=True,
                )
            biased_model = SumModel(self.model, wall)
            t_seg = time.perf_counter()
            state, local, used = self._run_segment(
                state, biased_model, seg, total_steps, dt, kT
            )
            segment_times.append(time.perf_counter() - t_seg)
            total_steps += used
            self._extend_global(global_buf, local)
            if self.verbose:
                print(  # noqa: T201
                    f"[loxodynamics] biased segment (iter {wall_stats[-1].iteration}) "
                    f"done: {used} steps in {segment_times[-1]:.1f} s",
                    flush=True,
                )

        if self.verbose:
            print(  # noqa: T201
                f"[loxodynamics] simulation finished: {total_steps} steps, "
                f"{len(wall_stats)} walls, in {time.perf_counter() - t_run:.1f} s "
                "(time cost)",
                flush=True,
            )

        if global_buf:
            global_descriptors = torch.stack(global_buf)
        else:
            global_descriptors = torch.empty(
                0,
                self.descriptor.n_descriptors,
                device=self._device,
                dtype=self._dtype,
            )
        return LoxodynamicsResult(
            final_state=state,
            wall_stats=wall_stats,
            training_reports=training_reports,
            global_descriptors=global_descriptors,
            total_steps=total_steps,
            segment_times=segment_times,
        )


def run_loxodynamics(
    state: SimState,
    model: ModelInterface,
    *,
    descriptor: PairDistanceDescriptor,
    max_steps: int,
    segment_steps: int,
    timestep: float,
    temperature: float,
    initial_unbiased_steps: int | None = None,
    sample_stride: int = 1,
    kappa: float = 1.0,
    wall_offset: float = 1.0,
    global_buffer_capacity: int = 50000,
    min_local_samples: int = 10,
    gamma: torch.Tensor | float | None = None,
    seed: int | None = None,
    unit_system: UnitSystem = UnitSystem.metal,
    skewencoder_config: SkewencoderConfig | None = None,
    trajectory_reporter: TrajectoryReporter | None = None,
    checkpoint_dir: str | Path | None = None,
    verbose: bool = False,
    **init_kwargs: Any,
) -> LoxodynamicsResult:
    """Convenience wrapper: build a :class:`LoxodynamicsExecutor` and run it.

    See :class:`LoxodynamicsExecutor` for argument semantics.
    """
    executor = LoxodynamicsExecutor(
        model,
        descriptor,
        max_steps=max_steps,
        segment_steps=segment_steps,
        timestep=timestep,
        temperature=temperature,
        initial_unbiased_steps=initial_unbiased_steps,
        sample_stride=sample_stride,
        kappa=kappa,
        wall_offset=wall_offset,
        global_buffer_capacity=global_buffer_capacity,
        min_local_samples=min_local_samples,
        gamma=gamma,
        seed=seed,
        unit_system=unit_system,
        skewencoder_config=skewencoder_config,
        trajectory_reporter=trajectory_reporter,
        checkpoint_dir=checkpoint_dir,
        verbose=verbose,
        **init_kwargs,
    )
    return executor.run(state)
