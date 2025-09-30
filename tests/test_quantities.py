"""Tests for quantities module functions."""

import pytest
import torch
from numpy.testing import assert_allclose
from torch._tensor import Tensor

from torch_sim import quantities
from torch_sim.quantities import calc_heat_flux
from torch_sim.units import MetalUnits


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


class TestHeatFlux:
    """Test suite for heat flux calculations."""

    @pytest.fixture
    def mock_simple_system(
        self,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Simple system with known values."""
        return {
            "velocities": torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 3.0],
                ],
                device=device,
            ),
            "energies": torch.tensor([1.0, 2.0, 3.0], device=device),
            "stress": torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
                ],
                device=device,
            ),
            "masses": torch.ones(3, device=device),
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
            stress=mock_simple_system["stress"],
            is_virial_only=False,
        )

        # Heat flux parts should cancel out
        expected = torch.zeros(3, device=flux.device)
        assert_allclose(flux.cpu().numpy(), expected.cpu().numpy())

    def test_unbatched_virial_only(
        self, mock_simple_system: dict[str, torch.Tensor]
    ) -> None:
        """Test virial-only heat flux calculation for unbatched case."""
        virial = calc_heat_flux(
            momenta=None,
            masses=mock_simple_system["masses"],
            velocities=mock_simple_system["velocities"],
            energies=mock_simple_system["energies"],
            stress=mock_simple_system["stress"],
            is_virial_only=True,
        )

        expected = -torch.tensor([1.0, 4.0, 9.0], device=virial.device)
        assert_allclose(virial.cpu().numpy(), expected.cpu().numpy())

    def test_batched_calculation(self, device: torch.device) -> None:
        """Test heat flux calculation with batched data."""
        velocities = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            device=device,
        )
        energies = torch.tensor([1.0, 2.0, 3.0], device=device)
        stress = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            ],
            device=device,
        )
        batch = torch.tensor([0, 0, 1], device=device)

        flux = calc_heat_flux(
            momenta=None,
            masses=torch.ones(3, device=device),
            velocities=velocities,
            energies=energies,
            stress=stress,
            batch=batch,
        )

        # Each batch should cancel heat flux parts
        expected = torch.zeros((2, 3), device=device)
        assert_allclose(flux.cpu().numpy(), expected.cpu().numpy())

    def test_centroid_stress(self, device: torch.device) -> None:
        """Test heat flux with centroid stress formulation."""
        velocities = torch.tensor([[1.0, 1.0, 1.0]], device=device)
        energies = torch.tensor([1.0], device=device)

        # Symmetric cross-terms
        stress = torch.tensor(
            [[1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], device=device
        )

        flux = calc_heat_flux(
            momenta=None,
            masses=torch.ones(1, device=device),
            velocities=velocities,
            energies=energies,
            stress=stress,
            is_centroid_stress=True,
        )

        # Heatflux should be [-1,-1,-1]
        expected = torch.full((3,), -1.0, device=device)
        assert_allclose(flux.cpu().numpy(), expected.cpu().numpy())

    def test_momenta_input(self, device: torch.device) -> None:
        """Test heat flux calculation using momenta instead."""
        momenta = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        masses = torch.tensor([2.0], device=device)
        energies = torch.tensor([1.0], device=device)
        stress = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)

        flux = calc_heat_flux(
            momenta=momenta,
            masses=masses,
            velocities=None,
            energies=energies,
            stress=stress,
        )

        # Heat flux terms should cancel out
        expected = torch.zeros(3, device=device)
        assert_allclose(flux.cpu().numpy(), expected.cpu().numpy())


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
    ke_vel = quantities.calc_kinetic_energy(
        masses=single_system_data["masses"],
        velocities=single_system_data["velocities"],
    )
    assert torch.allclose(ke_vel, single_system_data["ke"])

    # With momenta
    ke_mom = quantities.calc_kinetic_energy(
        masses=single_system_data["masses"], momenta=single_system_data["momenta"]
    )
    assert torch.allclose(ke_mom, single_system_data["ke"])


def test_calc_kinetic_energy_batched_system(
    batched_system_data: dict[str, Tensor],
) -> None:
    # With velocities
    ke_vel = quantities.calc_kinetic_energy(
        masses=batched_system_data["masses"],
        velocities=batched_system_data["velocities"],
        system_idx=batched_system_data["system_idx"],
    )
    assert torch.allclose(ke_vel, batched_system_data["ke"])

    # With momenta
    ke_mom = quantities.calc_kinetic_energy(
        masses=batched_system_data["masses"],
        momenta=batched_system_data["momenta"],
        system_idx=batched_system_data["system_idx"],
    )
    assert torch.allclose(ke_mom, batched_system_data["ke"])


def test_calc_kinetic_energy_errors(single_system_data: dict[str, Tensor]) -> None:
    with pytest.raises(ValueError, match="Must pass either one of momenta or velocities"):
        quantities.calc_kinetic_energy(
            masses=single_system_data["masses"],
            momenta=single_system_data["momenta"],
            velocities=single_system_data["velocities"],
        )

    with pytest.raises(ValueError, match="Must pass either one of momenta or velocities"):
        quantities.calc_kinetic_energy(masses=single_system_data["masses"])


def test_calc_kt_single_system(single_system_data: dict[str, Tensor]) -> None:
    # With velocities
    kt_vel = quantities.calc_kT(
        masses=single_system_data["masses"],
        velocities=single_system_data["velocities"],
    )
    assert torch.allclose(kt_vel, single_system_data["kt"])

    # With momenta
    kt_mom = quantities.calc_kT(
        masses=single_system_data["masses"], momenta=single_system_data["momenta"]
    )
    assert torch.allclose(kt_mom, single_system_data["kt"])


def test_calc_kt_batched_system(batched_system_data: dict[str, Tensor]) -> None:
    # With velocities
    kt_vel = quantities.calc_kT(
        masses=batched_system_data["masses"],
        velocities=batched_system_data["velocities"],
        system_idx=batched_system_data["system_idx"],
    )
    assert torch.allclose(kt_vel, batched_system_data["kt"])

    # With momenta
    kt_mom = quantities.calc_kT(
        masses=batched_system_data["masses"],
        momenta=batched_system_data["momenta"],
        system_idx=batched_system_data["system_idx"],
    )
    assert torch.allclose(kt_mom, batched_system_data["kt"])


def test_calc_temperature(single_system_data: dict[str, Tensor]) -> None:
    temp = quantities.calc_temperature(
        masses=single_system_data["masses"],
        velocities=single_system_data["velocities"],
    )
    kt = quantities.calc_kT(
        masses=single_system_data["masses"],
        velocities=single_system_data["velocities"],
    )
    assert torch.allclose(temp, kt / MetalUnits.temperature)
