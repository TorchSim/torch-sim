"""Tests for quantities module functions."""

from collections.abc import Callable

import pytest
import torch
from numpy.testing import assert_allclose
from torch import Tensor

from tests.conftest import DEVICE, DTYPE
from torch_sim.quantities import (
    calc_heat_flux,
    calc_kinetic_energy,
    calc_kT,
    calc_temperature,
    cell_volume,
)
from torch_sim.units import MetalUnits


class TestHeatFlux:
    """Test suite for heat flux calculations."""

    @pytest.fixture
    def mock_simple_system(self) -> dict[str, torch.Tensor]:
        """Simple system with known values."""
        return {
            "velocities": torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 3.0],
                ],
                device=DEVICE,
            ),
            "energies": torch.tensor([1.0, 2.0, 3.0], device=DEVICE),
            "stress": torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
                ],
                device=DEVICE,
            ),
            "masses": torch.ones(3, device=DEVICE),
        }

    def test_unbatched_total_flux(
        self, mock_simple_system: dict[str, torch.Tensor]
    ) -> None:
        """Test total heat flux calculation for unbatched case."""
        flux = calc_heat_flux(
            momenta=None,
            masses=mock_simple_system["masses"],
            velocities=mock_simple_system["velocities"],
            energies=mock_simple_system["energies"],
            stresses=mock_simple_system["stress"],
            is_virial_only=False,
        )

        # Heat flux parts should cancel out
        expected = torch.zeros(3, device=flux.device)
        assert_allclose(flux.detach().cpu().numpy(), expected.detach().cpu().numpy())

    def test_unbatched_virial_only(
        self, mock_simple_system: dict[str, torch.Tensor]
    ) -> None:
        """Test virial-only heat flux calculation for unbatched case."""
        virial = calc_heat_flux(
            momenta=None,
            masses=mock_simple_system["masses"],
            velocities=mock_simple_system["velocities"],
            energies=mock_simple_system["energies"],
            stresses=mock_simple_system["stress"],
            is_virial_only=True,
        )

        expected = -torch.tensor([1.0, 4.0, 9.0], device=virial.device)
        assert_allclose(virial.detach().cpu().numpy(), expected.detach().cpu().numpy())

    def test_batched_calculation(self) -> None:
        """Test heat flux calculation with batched data."""
        velocities = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            device=DEVICE,
        )
        energies = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        stress = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            ],
            device=DEVICE,
        )
        batch = torch.tensor([0, 0, 1], device=DEVICE)

        flux = calc_heat_flux(
            momenta=None,
            masses=torch.ones(3, device=DEVICE),
            velocities=velocities,
            energies=energies,
            stresses=stress,
            batch=batch,
        )

        # Each batch should cancel heat flux parts
        expected = torch.zeros((2, 3), device=DEVICE)
        assert_allclose(flux.detach().cpu().numpy(), expected.detach().cpu().numpy())

    def test_centroid_stress(self) -> None:
        """Test heat flux with centroid stress formulation."""
        velocities = torch.tensor([[1.0, 1.0, 1.0]], device=DEVICE)
        energies = torch.tensor([1.0], device=DEVICE)

        # Symmetric cross-terms
        stress = torch.tensor(
            [[1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], device=DEVICE
        )

        flux = calc_heat_flux(
            momenta=None,
            masses=torch.ones(1, device=DEVICE),
            velocities=velocities,
            energies=energies,
            stresses=stress,
            is_centroid_stress=True,
        )

        # Heatflux should be [-1,-1,-1]
        expected = torch.full((3,), -1.0, device=DEVICE)
        assert_allclose(flux.detach().cpu().numpy(), expected.detach().cpu().numpy())

    def test_momenta_input(self) -> None:
        """Test heat flux calculation using momenta instead."""
        momenta = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)
        masses = torch.tensor([2.0], device=DEVICE)
        energies = torch.tensor([1.0], device=DEVICE)
        stress = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=DEVICE)

        flux = calc_heat_flux(
            momenta=momenta,
            masses=masses,
            velocities=None,
            energies=energies,
            stresses=stress,
        )

        # Heat flux terms should cancel out
        expected = torch.zeros(3, device=DEVICE)
        assert_allclose(flux.detach().cpu().numpy(), expected.detach().cpu().numpy())


@pytest.fixture
def single_system_data() -> dict[str, Tensor]:
    masses = torch.tensor([1.0, 2.0], device=DEVICE, dtype=DTYPE)
    velocities = torch.tensor(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], device=DEVICE, dtype=DTYPE
    )
    momenta = velocities * masses.unsqueeze(-1)
    return {
        "masses": masses,
        "velocities": velocities,
        "momenta": momenta,
        "ke": torch.tensor(13.5, device=DEVICE, dtype=DTYPE),
        "kt": torch.tensor(4.5, device=DEVICE, dtype=DTYPE),
    }


@pytest.fixture
def batched_system_data() -> dict[str, Tensor]:
    masses = torch.tensor([1.0, 1.0, 2.0, 2.0], device=DEVICE, dtype=DTYPE)
    velocities = torch.tensor(
        [[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]], device=DEVICE, dtype=DTYPE
    )
    momenta = velocities * masses.unsqueeze(-1)
    system_idx = torch.tensor([0, 0, 1, 1], device=DEVICE)
    return {
        "masses": masses,
        "velocities": velocities,
        "momenta": momenta,
        "system_idx": system_idx,
        "ke": torch.tensor([3.0, 24.0], device=DEVICE, dtype=DTYPE),
        "kt": torch.tensor([1.0, 8.0], device=DEVICE, dtype=DTYPE),
    }


def test_calc_kinetic_energy_single_system(single_system_data: dict[str, Tensor]) -> None:
    # With velocities
    ke_vel = calc_kinetic_energy(
        masses=single_system_data["masses"],
        velocities=single_system_data["velocities"],
    )
    assert torch.allclose(ke_vel, single_system_data["ke"])

    # With momenta
    ke_mom = calc_kinetic_energy(
        masses=single_system_data["masses"], momenta=single_system_data["momenta"]
    )
    assert torch.allclose(ke_mom, single_system_data["ke"])


def test_calc_kinetic_energy_batched_system(
    batched_system_data: dict[str, Tensor],
) -> None:
    # With velocities
    ke_vel = calc_kinetic_energy(
        masses=batched_system_data["masses"],
        velocities=batched_system_data["velocities"],
        system_idx=batched_system_data["system_idx"],
    )
    assert torch.allclose(ke_vel, batched_system_data["ke"])

    # With momenta
    ke_mom = calc_kinetic_energy(
        masses=batched_system_data["masses"],
        momenta=batched_system_data["momenta"],
        system_idx=batched_system_data["system_idx"],
    )
    assert torch.allclose(ke_mom, batched_system_data["ke"])


@pytest.mark.parametrize("calc_fn", [calc_kinetic_energy, calc_kT])
def test_per_system_sums_with_unsorted_system_idx(calc_fn: Callable[..., Tensor]) -> None:
    """Per-system reductions must not assume atoms are grouped by system."""
    torch.manual_seed(0)
    system_idx = torch.tensor([1, 0, 1, 2, 0, 2, 1], device=DEVICE)
    masses = torch.rand(7, device=DEVICE, dtype=DTYPE) + 0.5
    velocities = torch.randn(7, 3, device=DEVICE, dtype=DTYPE)

    result = calc_fn(masses=masses, velocities=velocities, system_idx=system_idx)
    expected = torch.stack(
        [
            calc_fn(masses=masses[mask], velocities=velocities[mask])
            for mask in (system_idx == sys_id for sys_id in range(3))
        ]
    )
    assert torch.allclose(result, expected)


def test_calc_kinetic_energy_errors(single_system_data: dict[str, Tensor]) -> None:
    with pytest.raises(ValueError, match="Must pass either one of momenta or velocities"):
        calc_kinetic_energy(
            masses=single_system_data["masses"],
            momenta=single_system_data["momenta"],
            velocities=single_system_data["velocities"],
        )

    with pytest.raises(ValueError, match="Must pass either one of momenta or velocities"):
        calc_kinetic_energy(masses=single_system_data["masses"])


def test_calc_kt_single_system(single_system_data: dict[str, Tensor]) -> None:
    # With velocities
    kt_vel = calc_kT(
        masses=single_system_data["masses"],
        velocities=single_system_data["velocities"],
    )
    assert torch.allclose(kt_vel, single_system_data["kt"])

    # With momenta
    kt_mom = calc_kT(
        masses=single_system_data["masses"], momenta=single_system_data["momenta"]
    )
    assert torch.allclose(kt_mom, single_system_data["kt"])


def test_calc_kt_batched_system(batched_system_data: dict[str, Tensor]) -> None:
    # With velocities
    kt_vel = calc_kT(
        masses=batched_system_data["masses"],
        velocities=batched_system_data["velocities"],
        system_idx=batched_system_data["system_idx"],
    )
    assert torch.allclose(kt_vel, batched_system_data["kt"])

    # With momenta
    kt_mom = calc_kT(
        masses=batched_system_data["masses"],
        momenta=batched_system_data["momenta"],
        system_idx=batched_system_data["system_idx"],
    )
    assert torch.allclose(kt_mom, batched_system_data["kt"])


def test_calc_temperature(single_system_data: dict[str, Tensor]) -> None:
    temp = calc_temperature(
        masses=single_system_data["masses"],
        velocities=single_system_data["velocities"],
    )
    kt = calc_kT(
        masses=single_system_data["masses"],
        velocities=single_system_data["velocities"],
    )
    assert torch.allclose(temp, kt / MetalUnits.temperature)


def test_cell_volume_handedness_invariant() -> None:
    """Volume is positive and identical for left- and right-handed cells."""
    cell = torch.tensor([[[2.0, 0, 0], [0, 3.0, 0], [0, 0, 4.0]]] * 2)
    cell[1, 2] *= -1  # flip handedness of the second cell
    volumes = cell_volume(cell)
    torch.testing.assert_close(volumes, torch.full((2,), 24.0))
    assert cell_volume(cell[0]).shape == ()
