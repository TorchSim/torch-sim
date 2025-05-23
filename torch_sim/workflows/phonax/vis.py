"""visualization utils."""

from collections.abc import Callable

import ase
import ase.spectrum.band_structure
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm


def predict_hessian(
    energy_fn: Callable[[torch.Tensor], torch.Tensor], positions: torch.Tensor
) -> np.ndarray:
    """Computes the Hessian matrix for the given `model` and `graph`
    in a manner analogous to the JAX version.

    Returns:
        A Hessian tensor of shape (num_nodes, dim, num_nodes, dim).
        (I.e., positions.shape + positions.shape)
    """
    # Clone positions so we can autograd through them safely.
    # Make sure they require gradients for the Hessian computation.
    positions = positions.clone().detach().requires_grad_(True)

    # 1) Compute the total energy
    energy = energy_fn(positions)

    # 2) Compute the gradient of energy wrt positions
    #    We set create_graph=True to allow second derivatives.
    grad_e = torch.autograd.grad(energy, positions, create_graph=True)[0]
    # grad_e has the same shape as positions (e.g., [num_nodes, dim])

    # 3) Allocate tensor for Hessian
    numel = positions.numel()  # total number of elements in positions
    hessian = torch.zeros((numel, numel), dtype=positions.dtype, device=positions.device)

    # 4) Loop over each component of the gradient to get second derivatives
    grad_e_flat = grad_e.reshape(numel)  # flatten
    for i in range(numel):
        # Take derivative of grad_e_flat[i] wrt positions
        # retain_graph=True because we need the graph for multiple grads
        d2 = torch.autograd.grad(grad_e_flat[i], positions, retain_graph=True)[0]
        hessian[i, :] = d2.reshape(-1)

    # 5) Reshape Hessian to (positions.shape + positions.shape)
    #    so that if positions is [num_nodes, dim], Hessian is [num_nodes, dim, num_nodes, dim].
    hessian = hessian.reshape(positions.shape + positions.shape)

    return hessian.cpu().numpy()


def predict_hessian_from_forces(
    forces_fn: Callable[[torch.Tensor], torch.Tensor], positions: torch.Tensor
) -> np.ndarray:
    # Clone positions so we can autograd through them safely.
    # Make sure they require gradients for the Hessian computation.
    positions = positions.clone().detach().requires_grad_(True)

    # 1) Compute forces from the positions
    forces = forces_fn(positions)

    # 2) Allocate tensor for Hessian
    numel = positions.numel()  # total number of elements in positions
    hessian = torch.zeros((numel, numel), dtype=positions.dtype, device=positions.device)

    # 3) Loop over each component of the gradient to get second derivatives
    grad_e = (
        -forces.flatten()  # invert since forces is the inverted derivative of energy
    )
    # grad_e = -forces
    for i in range(grad_e.shape[0]):
        # Take derivative of grad_e_flat[i] wrt positions
        # retain_graph=True because we need the graph for multiple grads
        d2 = torch.autograd.grad(grad_e[i], positions, retain_graph=True)[0]
        hessian[i, :] = d2.reshape(-1)

    # 4) Reshape Hessian to (positions.shape + positions.shape)
    #    so that if positions is [num_nodes, dim], Hessian is [num_nodes, dim, num_nodes, dim].
    hessian = hessian.reshape(positions.shape + positions.shape)

    return hessian.numpy()


def hessian_k_np(
    kpt: np.ndarray,
    positions: torch.Tensor,
    H: torch.Tensor,
    n_atoms: int,
    nodes_index_cell0: np.ndarray,
):
    """Compute the Hessian matrix at a given k-point, using NumPy.

    Args:
        kpt: 1D array of shape (3,)
        graph: an object with:
               - graph.nodes.positions (numpy array of shape (N, 3))
               - graph.nodes.index_cell0 (numpy array of shape (N,))
        H: Hessian matrix of shape (N, 3, N, 3)
        n_atoms: number of primitive-cell atoms (int)

    Returns:
        Hk: Hessian matrix at kpt, shape (n_atoms * 3, n_atoms * 3)
    """
    # Positions of the extended graph
    r = positions  # shape (N, 3)
    # Phase factor: exp[-i k · (r - r)] = exp[- i * dot(...)]
    # shape: (N, N)
    phase = np.exp(-1j * np.einsum("nij,j->ni", r[:, None, :] - r[None, :, :], kpt))
    # Reshape to broadcast with H
    # original code did [:, None, :, None], making it shape (N, 1, N, 1)
    # which will broadcast correctly with H (N, 3, N, 3)
    phase = phase[:, None, :, None]

    # Index of the "primitive" atoms in the super/extended cell
    # a = graph.nodes.index_cell0  # shape (n_atoms,) with integer indices
    a = nodes_index_cell0
    # We want to address the x,y,z directions with i = [0,1,2]
    i = np.arange(3)

    # phase = np.array(phase, dtype=np.complex64)

    # Initialize the Hessian for just the n_atoms subset
    Hk = np.zeros((n_atoms, 3, n_atoms, 3), dtype=phase.dtype)
    # Accumulate phase * H into the appropriate block
    # We can use np.ix_ to do advanced indexing
    # Hk[np.ix_(a, i, a, i)] += phase * H
    np.add.at(Hk, np.ix_(a, i, a, i), phase * H)

    # Finally, reshape into (3*n_atoms, 3*n_atoms)
    return Hk.reshape(n_atoms * 3, n_atoms * 3)


def dynamical_matrix_np(
    kpt: np.ndarray,
    positions: torch.Tensor,
    H: torch.Tensor,
    masses: np.ndarray,
    nodes_index_cell0: np.ndarray,
):
    """Dynamical matrix at a given k-point, using NumPy.

    D(k) = Hk(k) / sqrt(m_i * m_j)

    Args:
        kpt: 1D array of shape (3,)
        graph: object with positions & index_cell0
        H: Hessian matrix of shape (N, 3, N, 3)
        masses: 1D array of shape (n_atoms,)

    Returns:
        D: Dynamical matrix at kpt, shape (3*n_atoms, 3*n_atoms)
    """
    Hk = hessian_k_np(
        kpt, positions, H, masses.size, nodes_index_cell0
    )  # shape (3*n_atoms, 3*n_atoms)

    # Reshape into (n_atoms, 3, n_atoms, 3) to apply the masses
    Hk = Hk.reshape((masses.size, 3, masses.size, 3))

    # 1 / sqrt(m)
    iM = 1.0 / np.sqrt(masses)

    # Apply the mass factors: Hk_ij / sqrt(m_i * m_j)
    # np.einsum("i,iujv,j->iujv", iM, Hk, iM)
    # shape of result is still (n_atoms, 3, n_atoms, 3)
    Hk = np.einsum("i,iujv,j->iujv", iM, Hk, iM)

    # Finally reshape back to (3*n_atoms, 3*n_atoms)
    D = Hk.reshape(3 * masses.size, 3 * masses.size)
    return D


# modified sqrt function to handle "negative/imaginary" modes in the spectrum
def safe_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def plot_spectrum_via_energy_preds(
    atoms: ase.Atoms,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    positions: torch.Tensor,
    nodes_index_cell0: torch.Tensor,
):
    hessian = predict_hessian(energy_fn, positions)
    with open("hessian.npy", "wb") as f:
        print("hessian shape", hessian.shape)
        np.save(f, hessian)
    hessian = np.load("hessian.npy")
    # print("hessian", hessian)
    return plot_bands(
        atoms,
        positions.cpu().numpy(),
        hessian,
        npoints=1000,
        nodes_index_cell0=nodes_index_cell0,
    )


def plot_spectrum_via_forces_preds(
    atoms: ase.Atoms,
    forces_fn: Callable[[torch.Tensor], torch.Tensor],
    positions: torch.Tensor,
    nodes_index_cell0: torch.Tensor,
):
    hessian = predict_hessian_from_forces(forces_fn, positions)
    with open("hessian_forces.npy", "wb") as f:
        print("hessian shape", hessian.shape)
        np.save(f, hessian)
    hessian = np.load("hessian_forces.npy")
    # print("hessian", hessian)
    return plot_bands(
        atoms, positions, hessian, npoints=1000, nodes_index_cell0=nodes_index_cell0
    )


def plot_bands(
    atoms: ase.Atoms,
    positions: np.ndarray,
    hessian: np.ndarray,
    nodes_index_cell0,
    npoints=1000,
):
    # create ase cell object
    cell = ase.Atoms(cell=atoms.cell.array, pbc=True).cell

    masses = ase.data.atomic_masses[atoms.get_atomic_numbers()]

    rec_vecs = 2 * np.pi * cell.reciprocal().real
    mp_band_path = cell.bandpath(npoints=npoints)

    all_kpts = mp_band_path.kpts @ rec_vecs
    all_eigs = []

    for kpt in tqdm.tqdm(all_kpts):
        Dk = dynamical_matrix_np(
            kpt, positions, hessian, masses=masses, nodes_index_cell0=nodes_index_cell0
        )
        Dk = np.asarray(Dk)
        all_eigs.append(np.sort(safe_sqrt(np.linalg.eigh(Dk)[0])))

    all_eigs = np.stack(all_eigs)

    eV_to_J = 1.60218e-19
    angstrom_to_m = 1e-10
    atom_mass = 1.660599e-27  # kg
    hbar = 1.05457182e-34
    cm_inv = (0.124e-3) * (1.60218e-19)  # in J
    conv_const = eV_to_J / (angstrom_to_m**2) / atom_mass

    all_eigs = all_eigs * np.sqrt(conv_const) * hbar / cm_inv

    bs = ase.spectrum.band_structure.BandStructure(mp_band_path, all_eigs[None])

    plt.figure(figsize=(7, 6), dpi=100)
    bs.plot(ax=plt.gca(), emin=1.1 * np.min(all_eigs), emax=1.1 * np.max(all_eigs))
    plt.ylabel("Phonon Frequency (cm$^{-1}$)")
    plt.tight_layout()
    return plt.gcf()
