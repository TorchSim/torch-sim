"""Trajectory history buffer for history-dependent enhanced-sampling methods.

History-dependent biases accumulate a record of the trajectory and read it
back to build a bias energy: metadynamics deposits reference structures or
collective-variable values and later sums repulsive kernels over them, while
boundary methods track a quantity (an energy, a distance) over time. The
*what* and the energy mapping differ between methods, but the bookkeeping --
a deposition cadence, a capacity limit that drops the oldest entries, and
restart-safe storage -- is identical. :class:`History` owns exactly that
bookkeeping so each method only decides what to deposit and how to use it.
"""

from __future__ import annotations

import torch


class History(torch.nn.Module):
    """Rolling, capacity-bounded buffer of per-step quantities.

    Deposited values are stacked along a new leading axis, so every value
    passed to :meth:`push` must share the same trailing shape. The first
    ``capacity`` deposits grow the buffer; later deposits drop the oldest
    entry. Stored values are detached and held in a registered buffer, so the
    history moves with :meth:`~torch.nn.Module.to` and is captured by
    ``state_dict`` for checkpoint/restart.

    A bias typically calls :meth:`maybe_push` once per forward pass and lets
    the configured ``stride`` decide when a deposit actually happens; use
    :meth:`push` for unconditional, manually controlled deposits (e.g. seeding
    the buffer on the first step).

    Args:
        capacity: Maximum number of stored entries; the oldest are dropped
            once it is exceeded. Must be >= 1.
        stride: Deposit on every ``stride``-th :meth:`maybe_push` call. Must
            be >= 1. Defaults to 1 (deposit on every call).
        device: Device for the stored buffer. Defaults to CPU.
        dtype: Floating-point dtype of the stored buffer. Defaults to
            ``torch.float64``.

    Example::

        history = History(capacity=10, stride=5)
        history.push(value)            # unconditional seed
        deposited = history.maybe_push(value)  # True every 5th call
        record = history.stack()       # (n_stored, *value.shape)
    """

    data: torch.Tensor | None

    def __init__(
        self,
        capacity: int,
        stride: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initialize an empty history buffer."""
        super().__init__()
        if capacity < 1:
            raise ValueError(f"{capacity=} must be >= 1")
        if stride < 1:
            raise ValueError(f"{stride=} must be >= 1")
        self.capacity = int(capacity)
        self.stride = int(stride)
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        # registered (rather than a plain attribute) so it moves with .to() and
        # is saved/restored via state_dict; starts None until the first deposit.
        self.register_buffer("data", None)
        self._n_calls = 0

    @property
    def is_empty(self) -> bool:
        """Whether no value has been deposited yet."""
        return self.data is None

    def __len__(self) -> int:
        """Number of currently stored entries."""
        return 0 if self.data is None else self.data.shape[0]

    @torch.no_grad()
    def push(self, value: torch.Tensor) -> None:
        """Deposit *value* as the newest entry, dropping the oldest past capacity.

        Args:
            value: Tensor to store. Detached and cast to the buffer's device
                and dtype. Its shape must match earlier deposits.

        Raises:
            ValueError: If *value*'s shape differs from existing entries. A
                history shaped to one batch cannot ingest a different one;
                call :meth:`reset` before reusing it with another system.
        """
        entry = value.detach().to(self._device, self._dtype).unsqueeze(0)
        if self.data is None:
            self.data = entry
            return
        if entry.shape[1:] != self.data.shape[1:]:
            raise ValueError(
                f"value shape {tuple(value.shape)} does not match stored entry "
                f"shape {tuple(self.data.shape[1:])}; call reset() before reusing "
                "this history with a different system."
            )
        self.data = torch.cat([self.data, entry], dim=0)[-self.capacity :]

    def maybe_push(self, value: torch.Tensor) -> bool:
        """Advance the call counter and deposit *value* on ``stride`` boundaries.

        Args:
            value: Tensor to deposit if this call lands on a stride boundary.

        Returns:
            ``True`` if a deposit happened, ``False`` otherwise.
        """
        self._n_calls += 1
        if self._n_calls % self.stride == 0:
            self.push(value)
            return True
        return False

    def stack(self) -> torch.Tensor | None:
        """Return the stored record with shape ``(n_stored, *value.shape)``.

        Returns:
            The stacked entries, or ``None`` if the buffer is empty.
        """
        return self.data

    def reset(self) -> None:
        """Clear all stored entries and the deposition counter."""
        self.data = None
        self._n_calls = 0
