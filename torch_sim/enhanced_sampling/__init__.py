"""Enhanced-sampling building blocks for TorchSim.

This subpackage collects enhanced-sampling methods and the reusable machinery
behind them. :class:`~torch_sim.enhanced_sampling.history.History` provides the
deposition/capacity bookkeeping shared by history-dependent biases. The
:mod:`~torch_sim.enhanced_sampling.metadynamics` module implements bias
potentials (:class:`LogfermiWall`, :class:`RMSDCV`) that compose with any MLIP
through :class:`~torch_sim.models.interface.SumModel`. The
:mod:`~torch_sim.enhanced_sampling.boxed_md` module implements boxed MD in energy
space (BXDE), and :mod:`~torch_sim.enhanced_sampling.loxodynamics` implements the
skewness-guided latent-space Loxodynamics method.
"""

from torch_sim.enhanced_sampling.boxed_md import BoxedMD, run_boxed_md, velocity_inversion
from torch_sim.enhanced_sampling.history import History
from torch_sim.enhanced_sampling.loxodynamics import (
    LoxodynamicsExecutor,
    LoxodynamicsResult,
    LoxodynamicsWall,
    LoxodynamicsWallStats,
    PairDistanceDescriptor,
    run_loxodynamics,
)
from torch_sim.enhanced_sampling.metadynamics import RMSDCV, LogfermiWall
from torch_sim.enhanced_sampling.skewencoder import (
    DescriptorNormalizer,
    Skewencoder,
    SkewencoderConfig,
    SkewencoderTrainer,
    SkewencoderTrainingReport,
)


__all__ = [
    "RMSDCV",
    "BoxedMD",
    "DescriptorNormalizer",
    "History",
    "LogfermiWall",
    "LoxodynamicsExecutor",
    "LoxodynamicsResult",
    "LoxodynamicsWall",
    "LoxodynamicsWallStats",
    "PairDistanceDescriptor",
    "Skewencoder",
    "SkewencoderConfig",
    "SkewencoderTrainer",
    "SkewencoderTrainingReport",
    "run_boxed_md",
    "run_loxodynamics",
    "velocity_inversion",
]
