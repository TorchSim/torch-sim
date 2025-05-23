"""Create and visualize supercells from torchsim atomic systems."""

from typing import TYPE_CHECKING

import torch
from ase.build import bulk
from pymatgen.vis.structure_vtk import StructureVis

import torch_sim as ts


if TYPE_CHECKING:
    from pymatgen.core import Structure

    from torch_sim.state import SimState

import torch_sim as ts


def create_supercell(
    state: "SimState", supercell_size: tuple[int, int, int] | int
) -> "SimState":
    """Create a supercell from a torchsim SimState.

    Args:
        state: The input SimState representing the unit cell
        supercell_size: Either a tuple of (nx, ny, nz) or a single integer for cubic supercells

    Returns:
        SimState: A new SimState containing the supercell
    """
    # Handle single integer input for cubic supercells
    if isinstance(supercell_size, int):
        supercell_size = (supercell_size, supercell_size, supercell_size)

    # Convert SimState to ASE atoms to use the built-in repeat functionality
    atoms_list = state.to_atoms()

    # Create supercells for each structure in the batch
    supercell_atoms_list = []
    for atoms in atoms_list:
        supercell_atoms = atoms.repeat(supercell_size)
        supercell_atoms_list.append(supercell_atoms)

    # Convert back to SimState
    supercell_state = ts.io.atoms_to_state(
        supercell_atoms_list, device=state.device, dtype=state.dtype
    )

    return supercell_state


def visualize_supercell_structure(
    state: "SimState", supercell_size: tuple[int, int, int] | int = 2, batch_idx: int = 0
) -> None:
    """Visualize a supercell using pymatgen's StructureVis.

    Args:
        state: The input SimState
        supercell_size: Size of supercell to create (default: 2x2x2)
        batch_idx: Which batch to visualize (default: 0)

    Raises:
        ImportError: If pymatgen or vtk are not installed
    """
    try:
        from pymatgen.vis.structure_vtk import StructureVis
    except ImportError as e:
        raise ImportError(
            "pymatgen and vtk are required for 3D visualization. "
            "Install with: uv add torch_sim_atomistic[visualization]"
        ) from e

    # Create the supercell
    supercell_state = create_supercell(state, supercell_size)

    # Convert to pymatgen structures
    structures = supercell_state.to_structures()

    # Get the specified batch
    if batch_idx >= len(structures):
        raise ValueError(
            f"batch_idx {batch_idx} is out of range. Available batches: 0-{len(structures) - 1}"
        )

    structure = structures[batch_idx]

    # Create visualizer and show
    vis = StructureVis()
    vis.set_structure(structure)
    vis.show()


def compare_unit_cell_and_supercell(
    state: "SimState",
    supercell_size: tuple[int, int, int] | int = 2,
    batch_idx: int = 0,
    use_vtk: bool = True,
) -> None:
    """Compare the original unit cell with its supercell.

    Args:
        state: The input SimState
        supercell_size: Size of supercell to create
        batch_idx: Which batch to compare
        use_vtk: If True, use pymatgen's StructureVis, otherwise use plotly
    """
    print(f"Original unit cell (batch {batch_idx}):")
    print(f"Number of atoms: {(state.batch == batch_idx).sum().item()}")

    # Show original structure
    structures = state.to_structures()
    structure = structures[batch_idx]

    print("Showing original unit cell...")
    vis = StructureVis()
    vis.set_structure(structure)
    vis.show()

    # Show supercell
    print(f"\nShowing {supercell_size} supercell...")
    visualize_supercell_structure(state, supercell_size, batch_idx)

    # Print supercell info
    supercell_state = create_supercell(state, supercell_size)
    supercell_atoms = (supercell_state.batch == batch_idx).sum().item()
    print(f"\nSupercell has {supercell_atoms} atoms")


def create_and_save_supercell(
    state: "SimState",
    supercell_size: tuple[int, int, int] | int = 2,
    output_file: str = "supercell.cif",
) -> "SimState":
    """Create a supercell and save it to a file.

    Args:
        state: The input SimState
        supercell_size: Size of supercell to create
        output_file: Output filename (supports .cif, .vasp, etc.)

    Returns:
        SimState: The supercell state
    """
    # Create the supercell
    supercell_state = create_supercell(state, supercell_size)

    # Convert to pymatgen structures and save
    structures = supercell_state.to_structures()

    for i, structure in enumerate(structures):
        if len(structures) > 1:
            # Multiple batches - save with batch index
            filename = output_file.replace(".", f"_batch_{i}.")
        else:
            filename = output_file

        structure.to(filename)
        print(f"Saved supercell to {filename}")

    return supercell_state


# Example usage functions
def example_silicon_supercell() -> None:
    """Example: Create and visualize a silicon supercell."""

    # Create a silicon unit cell
    si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)

    # Convert to SimState
    device = torch.device("cpu")
    dtype = torch.float32
    state = ts.io.atoms_to_state(si_atoms, device, dtype)

    print("Silicon unit cell created")
    print(f"Unit cell atoms: {state.n_atoms}")

    # Create and visualize 2x2x2 supercell
    visualize_supercell_structure(state, supercell_size=2)


def example_multiple_materials() -> None:
    """Example: Create supercells for multiple materials."""

    # Create multiple unit cells
    si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    cu_atoms = bulk("Cu", "fcc", a=3.61, cubic=True)

    # Convert to SimState (batched)
    device = torch.device("cpu")
    dtype = torch.float32
    state = ts.io.atoms_to_state([si_atoms, cu_atoms], device, dtype)

    print(f"Created batched state with {state.n_batches} structures")

    # Visualize Silicon supercell (batch 0)
    print("Visualizing Silicon 2x2x2 supercell...")
    visualize_supercell_structure(state, supercell_size=2, batch_idx=0)

    # Visualize Copper supercell (batch 1)
    print("Visualizing Copper 2x2x2 supercell...")
    visualize_supercell_structure(state, supercell_size=2, batch_idx=1)


if __name__ == "__main__":
    # Run example
    print("Creating silicon supercell example...")
    example_silicon_supercell()
