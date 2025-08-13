import pytest
import torch
from torch._tensor import Tensor

from torch_sim import quantities
from torch_sim.units import MetalUnits


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


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
