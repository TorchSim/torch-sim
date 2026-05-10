import numpy as np
import torch
from ase import Atoms
from ase.mep import NEB as ASENEB
from ase.mep.neb import ImprovedTangentMethod, NEBState

import torch_sim as ts
from tests.conftest import DEVICE, DTYPE
from torch_sim.models.interface import ModelInterface
from torch_sim.workflows.neb import NEB, calculate_neb_forces, interpolate_path


class HarmonicModel(ModelInterface):
    def __init__(self) -> None:
        super().__init__()
        self._device = DEVICE
        self._dtype = DTYPE
        self._compute_forces = True
        self._compute_stress = True

    def forward(self, state: ts.SimState, **kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        per_atom_energy = 0.5 * (state.positions**2).sum(dim=1)
        energy = torch.zeros(state.n_systems, device=state.device, dtype=state.dtype)
        energy.scatter_add_(0, state.system_idx, per_atom_energy)
        return {
            "energy": energy,
            "forces": -state.positions,
            "stress": torch.zeros(
                state.n_systems, 3, 3, device=state.device, dtype=state.dtype
            ),
        }


def _single_atom_state(position: float) -> ts.SimState:
    return ts.SimState(
        positions=torch.tensor([[position, 0.0, 0.0]], device=DEVICE, dtype=DTYPE),
        masses=torch.ones(1, device=DEVICE, dtype=DTYPE),
        cell=torch.eye(3, device=DEVICE, dtype=DTYPE).unsqueeze(0) * 10.0,
        pbc=False,
        atomic_numbers=torch.tensor([18], device=DEVICE),
        system_idx=torch.zeros(1, device=DEVICE, dtype=torch.long),
    )


def test_interpolate_path_uses_movable_images_only() -> None:
    initial = _single_atom_state(0.0)
    final = _single_atom_state(1.0)

    path = interpolate_path(initial, final, n_images=3)

    assert path.n_systems == 3
    assert path.n_groups == 1
    assert torch.equal(path.group_idx, torch.zeros(3, device=DEVICE, dtype=torch.long))
    assert torch.allclose(
        path.positions[:, 0],
        torch.tensor([0.25, 0.5, 0.75], device=DEVICE, dtype=DTYPE),
    )


def test_calculate_neb_forces_matches_ase_step0_components() -> None:
    n_images = 5
    n_atoms = 2
    spring_constant = 0.1
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.8, 0.0]],
            [[0.2, 0.1, 0.0], [0.1, 0.9, 0.0]],
            [[0.5, 0.25, 0.0], [0.2, 1.0, 0.1]],
            [[0.8, 0.35, 0.0], [0.25, 1.05, 0.2]],
            [[1.0, 0.5, 0.0], [0.4, 1.2, 0.3]],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )
    energies = torch.tensor([0.0, 0.2, 0.7, 0.4, 0.1], device=DEVICE, dtype=DTYPE)
    true_forces = torch.tensor(
        [
            [[0.1, -0.2, 0.0], [0.0, 0.3, -0.1]],
            [[-0.2, 0.1, 0.2], [0.2, -0.1, 0.0]],
            [[0.3, 0.0, -0.1], [-0.1, 0.2, 0.1]],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )
    path_state = ts.SimState(
        positions=positions.reshape(-1, 3),
        masses=torch.ones(n_images * n_atoms, device=DEVICE, dtype=DTYPE),
        cell=torch.eye(3, device=DEVICE, dtype=DTYPE).unsqueeze(0).repeat(n_images, 1, 1)
        * 10.0,
        pbc=False,
        atomic_numbers=torch.tensor([18, 18], device=DEVICE).repeat(n_images),
        system_idx=torch.repeat_interleave(
            torch.arange(n_images, device=DEVICE), repeats=n_atoms
        ),
    )

    torch_forces = calculate_neb_forces(
        path_state,
        true_forces.reshape(-1, 3),
        energies[1:-1],
        energies[0],
        energies[-1],
        spring_constant=spring_constant,
        use_climbing_image=True,
    ).reshape(n_images - 2, n_atoms, 3)

    ase_images = [
        Atoms(
            "Ar2",
            positions=image_positions.detach().cpu().numpy(),
            cell=np.eye(3) * 10.0,
            pbc=False,
        )
        for image_positions in positions
    ]
    ase_neb = ASENEB(ase_images, k=spring_constant, climb=True, method="improvedtangent")
    ase_state = NEBState(ase_neb, ase_neb.images, energies.detach().cpu().numpy())
    tangent_method = ImprovedTangentMethod(ase_neb)
    ase_forces = []
    true_forces_np = true_forces.detach().cpu().numpy()
    for image_index in range(1, n_images - 1):
        spring1 = ase_state.spring(image_index - 1)
        spring2 = ase_state.spring(image_index)
        tangent = tangent_method.get_tangent(ase_state, spring1, spring2, image_index)
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-15:
            tangent = tangent / tangent_norm
        force = true_forces_np[image_index - 1]
        force_dot_tangent = np.vdot(force, tangent)
        if ase_neb.climb and image_index == ase_state.imax:
            ase_forces.append(force - 2 * force_dot_tangent * tangent)
        else:
            spring_force = (spring2.nt * spring2.k - spring1.nt * spring1.k) * tangent
            ase_forces.append(force - force_dot_tangent * tangent + spring_force)

    assert torch.allclose(
        torch_forces,
        torch.tensor(np.array(ase_forces), device=DEVICE, dtype=DTYPE),
        atol=1e-12,
        rtol=1e-12,
    )


def test_neb_run_uses_single_chain_optimize_without_moving_endpoints() -> None:
    initial = _single_atom_state(0.0)
    final = _single_atom_state(1.0)
    neb = NEB(
        model=HarmonicModel(),
        n_images=1,
        optimizer_type="gd",
        optimizer_params={"pos_lr": 0.1},
    )

    result = neb.run(initial, final, max_steps=3, fmax=1e-12)

    assert result.n_systems == 3
    assert torch.allclose(
        result.positions[:, 0],
        torch.tensor([0.0, 0.5, 1.0], device=DEVICE, dtype=DTYPE),
    )
