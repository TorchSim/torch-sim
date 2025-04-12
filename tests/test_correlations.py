"""Tests for the correlations module.

Test time series correlation functionality provided by
the correlations module. It includes tests for `CircularBuffer`
and `CorrelationCalculator`, using idealized signals
with known correlation properties.
"""

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch

from torch_sim.properties.correlations import (
    CircularBuffer,
    CorrelationCalculator,
    VelocityAutoCorrelation,
)


class MockState:
    """Mock state class for testing correlation calculations.

    Provides a minimal implementation of SimState interface with only
    the components needed for correlation calculations.
    """

    def __init__(self, velocities: torch.Tensor, device: torch.device) -> None:
        """Initialize mock state with provided data."""
        self.velocities = velocities
        self.device = device
        # Required for TrajectoryReporter
        self.n_batches = 1
        self.batch = torch.zeros(velocities.shape[0], device=device, dtype=torch.int64)

    def split(self) -> list["MockState"]:
        """Split state into batches."""
        # Just return self since 1 batch
        return [self]


@pytest.fixture
def buffer(device: torch.device) -> CircularBuffer:
    """Fixture for CircularBuffer instance."""
    return CircularBuffer(size=10, device=device)


@pytest.fixture
def mock_state_factory(device: torch.device) -> Callable[[torch.Tensor], MockState]:
    """Factory fixture for creating mock state objects."""

    def create_mock_state(velocities: torch.Tensor) -> MockState:
        """Create mock state with given data tensor."""
        return MockState(velocities, device)

    return create_mock_state


@pytest.fixture
def corr_calc(device: torch.device) -> CorrelationCalculator:
    """Fixture for creating a CorrelationCalculator instance."""
    window_size = 5

    def velocity_getter(state: Any) -> torch.Tensor:
        return state.velocities

    properties = {"velocity": velocity_getter}

    return CorrelationCalculator(
        window_size=window_size,
        properties=properties,
        device=device,
        normalize=True,
    )


class TestCircularBuffer:
    """Test suite for CircularBuffer functionality."""

    def test_circular_buffer_operations(self, device: torch.device) -> None:
        """Test core buffer operations including append, retrieval,
        and wraparound.

        Tests initialization, data append, retrieval and circular wrapping.
        """
        buffer = CircularBuffer(size=3, device=device)

        # Test initialization state
        assert buffer.size == 3
        assert buffer.head == 0
        assert buffer.count == 0
        assert buffer.buffer is None
        assert not buffer.is_full

        # Test append and retrieval
        buffer.append(torch.tensor([1.0], device=device))
        buffer.append(torch.tensor([2.0], device=device))

        assert buffer.count == 2
        assert buffer.head == 2

        result = buffer.get_array()
        expected = torch.tensor([[1.0], [2.0]], device=device)
        assert torch.allclose(result, expected)

        # Test wraparound behavior
        buffer.append(torch.tensor([3.0], device=device))
        assert buffer.is_full

        buffer.append(torch.tensor([4.0], device=device))
        assert buffer.count == 3
        assert buffer.head == 1

        result = buffer.get_array()
        expected = torch.tensor([[2.0], [3.0], [4.0]], device=device)
        assert torch.allclose(result, expected)


class TestCorrelationCalculator:
    """Test suite for CorrelationCalculator.

    Tests focus on validating the calculator's ability to compute accurate
    autocorrelation functions for idealized signals with known properties.
    """

    def test_initialization(self, corr_calc: CorrelationCalculator) -> None:
        """Test that calculator is initialized with correct properties."""
        assert corr_calc.window_size == 5
        assert "velocity" in corr_calc.properties
        assert corr_calc.normalize is True

    def test_update_frequency(
        self, corr_calc: CorrelationCalculator, mock_state_factory: Callable
    ) -> None:
        """Test update frequency functionality.

        Verify calculator processes updates when called.
        """
        corr_calc = CorrelationCalculator(
            window_size=3,
            properties={"velocity": lambda s: s.velocities},
            device=corr_calc.device,
        )

        v1 = torch.ones((2, 3), device=corr_calc.device)
        state1 = mock_state_factory(v1)

        # First update
        corr_calc.update(state1)
        assert corr_calc.buffers["velocity"].count == 1

        # Second update
        corr_calc.update(state1)
        assert corr_calc.buffers["velocity"].count == 2

    def test_constant_signal(
        self, device: torch.device, mock_state_factory: Callable
    ) -> None:
        """Test correlation of constant signals.

        Mean-centered constant signals should have zero autocorrelation.
        """
        win_size = 4
        corr_calc = CorrelationCalculator(
            window_size=win_size,
            properties={"velocity": lambda s: s.velocities},
            device=device,
            normalize=False,
        )

        # Constant signal
        const_vel = torch.ones((2, 3), device=device)

        # Identical states
        for _ in range(win_size):
            state = mock_state_factory(const_vel)
            corr_calc.update(state)

        # ACF should be zeros here
        acf = corr_calc.get_auto_correlations()["velocity"]
        assert torch.allclose(acf, torch.zeros_like(acf), atol=1e-5)

    def test_white_noise(
        self, device: torch.device, mock_state_factory: Callable
    ) -> None:
        """Test autocorrelation of white noise.

        White noise should have a delta function as its autocorrelation.
        """
        win_size = 10
        corr_calc = CorrelationCalculator(
            window_size=win_size,
            properties={"velocity": lambda s: s.velocities},
            device=device,
            normalize=True,
        )

        torch.manual_seed(42)

        # White noise
        for _ in range(win_size):
            noise = torch.randn(4, 3, device=device)
            state = mock_state_factory(noise)
            corr_calc.update(state)

        # ACF and average over atoms/dimensions
        acf = corr_calc.get_auto_correlations()["velocity"]
        acf_mean = torch.mean(acf, dim=(1, 2))

        # Delta function
        assert torch.isclose(acf_mean[0], torch.tensor(1.0, device=device))
        assert torch.all(torch.abs(acf_mean[1:]) < 0.3)

    def test_sinusoidal(self, device: torch.device, mock_state_factory: Callable) -> None:
        """Test autocorrelation of sinusoidal signals.

        Sine waves should have a cosine-like acf.
        """
        win_size = 32
        period = 8
        corr_calc = CorrelationCalculator(
            window_size=win_size,
            properties={"velocity": lambda s: s.velocities},
            device=device,
            normalize=True,
        )

        t = torch.arange(win_size, dtype=torch.float32, device=device)
        freq = 2 * math.pi / period

        # Sine
        for i in range(win_size):
            phase = freq * t[i]
            signal_val = torch.sin(phase)

            # Expand to shape [2, 3]
            data = signal_val.expand(2, 3)
            state = mock_state_factory(data)
            corr_calc.update(state)

        acf = corr_calc.get_auto_correlations()["velocity"]
        acf_mean = torch.mean(acf, dim=(1, 2))

        assert torch.isclose(acf_mean[0], torch.tensor(1.0, device=device))

        half_period = period // 2
        assert acf_mean[half_period] < 0

        assert acf_mean[period] > 0.5

    def test_reset(
        self, corr_calc: CorrelationCalculator, mock_state_factory: Callable
    ) -> None:
        """Test reset functionality."""

        vel = torch.ones((2, 3), device=corr_calc.device)
        state = mock_state_factory(vel)

        for _ in range(3):
            corr_calc.update(state)

        corr_calc.reset()

        # Buffer empty?
        assert corr_calc.buffers["velocity"].count == 0
        assert corr_calc.correlations == {}

    def test_normalization(
        self, device: torch.device, mock_state_factory: Callable
    ) -> None:
        """Test normalization of correlation functions.

        Validates that normalized correlations have first lag = 1.0.
        """
        corr_calc_norm = CorrelationCalculator(
            window_size=5,
            properties={"velocity": lambda s: s.velocities},
            device=device,
            normalize=True,
        )

        corr_calc_no_norm = CorrelationCalculator(
            window_size=5,
            properties={"velocity": lambda s: s.velocities},
            device=device,
            normalize=False,
        )

        torch.manual_seed(42)

        for _ in range(5):
            vel = torch.randn((2, 3), device=device)

            # Reuse data
            state = mock_state_factory(vel)
            corr_calc_norm.update(state)
            corr_calc_no_norm.update(state)

        corr_norm = corr_calc_norm.get_auto_correlations()["velocity"]
        corr_no_norm = corr_calc_no_norm.get_auto_correlations()["velocity"]

        norm_first = torch.mean(corr_norm[0])
        assert torch.isclose(norm_first, torch.tensor(1.0, device=device))

        no_norm_first = torch.mean(corr_no_norm[0])
        assert not torch.allclose(no_norm_first, torch.ones_like(no_norm_first))

        for a in range(corr_norm.shape[1]):
            for d in range(corr_norm.shape[2]):
                scale_factor = corr_no_norm[0, a, d].item()
                if abs(scale_factor) > 1e-5:
                    expected = corr_no_norm[:, a, d] / scale_factor
                    assert torch.allclose(corr_norm[:, a, d], expected, atol=1e-5)

    def test_cross_correlation_basics(
        self, device: torch.device, mock_state_factory: Callable
    ) -> None:
        """Test basic cross-correlation."""
        win_size = 10
        corr_calc = CorrelationCalculator(
            window_size=win_size,
            properties={
                "signal_a": lambda s: s.velocities[:1],
                "signal_b": lambda s: s.velocities[1:],
            },
            device=device,
            normalize=True,
        )

        # Generate data where sinal_a and signal_b are different but related
        torch.manual_seed(42)

        # Initialize prev_signal_a
        prev_signal_a = torch.randn(1, 3, device=device)

        for i in range(win_size):
            signal_a = torch.randn(1, 3, device=device)
            if i > 0:
                signal_b = prev_signal_a * 0.7 + torch.randn(1, 3, device=device) * 0.3
            else:
                signal_b = torch.randn(1, 3, device=device)

            prev_signal_a = signal_a.clone()

            velocities = torch.cat([signal_a, signal_b], dim=0)
            state = mock_state_factory(velocities)
            corr_calc.update(state)

        cross_corrs = corr_calc.get_cross_correlations()
        assert ("signal_a", "signal_b") in cross_corrs

        cross_corr = cross_corrs[("signal_a", "signal_b")]
        assert len(cross_corr) == win_size

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_migration(self, mock_state_factory: Callable) -> None:
        """Test migration between CPU and GPU devices.

        Validate that the calculator can be moved between devices.
        """
        cpu_device = torch.device("cpu")
        corr_calc = CorrelationCalculator(
            window_size=3,
            properties={"velocity": lambda s: s.velocities},
            device=cpu_device,
        )

        vel = torch.ones((2, 3), device=cpu_device)
        state = mock_state_factory(vel)

        for _ in range(3):
            corr_calc.update(state)

        cuda_device = torch.device("cuda")
        corr_calc = corr_calc.to(cuda_device)

        assert corr_calc.device == cuda_device
        assert corr_calc.buffers["velocity"].device == cuda_device
        if corr_calc.buffers["velocity"].buffer is not None:
            assert corr_calc.buffers["velocity"].buffer.device == cuda_device


def test_velocity_autocorrelation(
    device: torch.device, mock_state_factory: Callable
) -> None:
    """Test VACF calculation with cosine pattern velocities.

    Test checks:
    1. Normalized VACF(0) = 1.0
    2. Expected periodicity
    3. Exhibits sign changes at specific locations
    """
    window_size = 32
    period = 8

    vacf_calc = VelocityAutoCorrelation(
        window_size=window_size,
        device=device,
        use_running_average=False,
        normalize=True,
    )

    # Cosine velocity pattern
    t = torch.arange(window_size, dtype=torch.float32, device=device)
    freq = 2 * math.pi / period

    velocities = []
    for i in range(window_size):
        # cos(ωt) pattern
        val = torch.cos(freq * t[i])
        vel = torch.tensor([[val, val, val]], device=device)
        velocities.append(vel)

    for vel in velocities:
        state = mock_state_factory(vel)
        vacf_calc(state)

    vacf = vacf_calc.vacf
    assert vacf is not None

    # 1. First lag is 1.0
    assert torch.isclose(vacf[0], torch.tensor(1.0, device=device))

    # 2. Check periodicity expect
    # positive peaks at t=0, t=8, t=16, ...
    assert vacf[period] >= 0.5
    assert vacf[2 * period] >= 0.5

    # 3. Check negative regions
    assert vacf[period // 2] <= -0.5
    assert vacf[3 * period // 2] <= -0.5

    # 4. Check zero-crossings expect
    # around t=period/4, t=3*period/4, ...
    assert abs(vacf[period // 4]) < 0.2
    assert abs(vacf[3 * period // 4]) < 0.2

    # 5. Verify the general shape
    # min/max at [-1.0, 1.0]
    assert torch.max(vacf) <= 1.0 + 1e-2
    assert torch.min(vacf) >= -1.0 - 1e-2


def test_velocity_autocorrelation_with_trajectory_reporter(
    device: torch.device, mock_state_factory: Callable
) -> None:
    """Test VelocityAutoCorrelation integration with TrajectoryReporter.

    This test verifies that:
    1. ``VelocityAutoCorrelation`` as a property calculator
    2. ``TrajectoryReporter`` calls correctly
    """
    from torch_sim.properties.correlations import VelocityAutoCorrelation
    from torch_sim.trajectory import TrajectoryReporter

    window_size = 20
    vacf_calc = VelocityAutoCorrelation(
        window_size=window_size,
        device=device,
        use_running_average=True,
    )

    reporter = TrajectoryReporter(
        None,  # Don't write file
        state_frequency=100,
        prop_calculators={5: {"vacf": vacf_calc}},
    )

    torch.manual_seed(42)
    n_steps = 25
    for step in range(n_steps):
        # Mock state
        velocities = torch.randn(4, 3, device=device)
        state = mock_state_factory(velocities)

        props = reporter.report(state, step)

        # Check if VACF was calculated
        if step % 5 == 0:
            assert "vacf" in props[0]
            assert isinstance(props[0]["vacf"], torch.Tensor)
            assert props[0]["vacf"].shape == (1,)

    reporter.close()
