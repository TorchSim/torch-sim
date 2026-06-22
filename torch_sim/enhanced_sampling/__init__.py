"""Enhanced-sampling building blocks for TorchSim.

This subpackage collects enhanced-sampling methods and the reusable machinery
behind them. :class:`~torch_sim.enhanced_sampling.history.History` provides the
deposition/capacity bookkeeping shared by history-dependent biases. The
:mod:`~torch_sim.enhanced_sampling.metadynamics` module implements bias
potentials (:class:`LogfermiWall`, :class:`RMSDCV`) that compose with any MLIP
through :class:`~torch_sim.models.interface.SumModel`.
"""

from torch_sim.enhanced_sampling.boxed_md import BoxedMD, run_boxed_md, velocity_inversion
from torch_sim.enhanced_sampling.history import History
from torch_sim.enhanced_sampling.metadynamics import RMSDCV, LogfermiWall


__all__ = [
    "RMSDCV",
    "BoxedMD",
    "History",
    "LogfermiWall",
    "run_boxed_md",
    "velocity_inversion",
]
