# %% [markdown]
# <details>
#   <summary>Dependencies</summary>
# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
#     "pymatgen>=2024.11.3",
#     "ase>=3.24",
#     "phonopy>=2.37.0",
# ]
# ///
# </details>


# %% [markdown]
"""
# Understanding State

This tutorial will guide you through the SimState class in TorchSim, which is the
core data structure for representing atomistic systems. You'll learn how to
create, manipulate, and work with states.

## Introduction

The SimState class is the core data structure in TorchSim that represents atomistic
systems. A SimState contains all the fundamental properties needed to describe
an atomistic system:

* Atomic positions
* Atomic masses
* Unit cell parameters
* Periodic boundary conditions
* Atomic numbers (elements)
* Batch indices (for processing multiple systems simultaneously)

"""

# %% [markdown]
"""
## Understanding the SimState Object

### A Simple State

New SimStates can be either created manually or from existing atomistic objects. Here
we'll start by creating an ase atoms object and converting it to a SimState. The `initialize_state` function
can take in pymatgen Structure, PhonopyAtoms, or other SimStates and convert them into a single SimState.

"""

# %%
import torch
import torch_sim as ts
from ase.build import bulk

# Create a crystal structure using ASE
si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)

# Convert to SimState
si_state = ts.initialize_state(si_atoms, device=torch.device("cpu"), dtype=torch.float64)

print(f"State has {si_state.n_atoms} atoms and {si_state.n_batches} batches")

# here we print all the attributes of the SimState
print(f"Positions shape: {si_state.positions.shape}")
print(f"Cell shape: {si_state.cell.shape}")
print(f"Atomic numbers shape: {si_state.atomic_numbers.shape}")
print(f"Masses shape: {si_state.masses.shape}")
print(f"PBC: {si_state.pbc}")
print(f"Batch indices shape: {si_state.batch.shape}")


# %% [markdown]
"""
SimState attributes fall into three categories: atomwise, batchwise, and global.

* Atomwise attributes are tensors with shape (n_atoms, ...), these are `positions`,
  `masses`, `atomic_numbers`, and `batch`. Names are plural.
* Batchwise attributes are tensors with shape (n_batches, ...), this is just `cell` for
  the base SimState. Names are singular.
* Global attributes have any other shape or type, just `pbc` here. Names are singular.

You can use the `infer_property_scope` function to analyze a state's properties. This
is mostly used internally but can be useful for debugging.
"""

# %%
from torch_sim.state import infer_property_scope

scope = infer_property_scope(si_state)
print(scope)


# %% [markdown]
"""
### A Batched State

A key advantage of TorchSim is its ability to simulate multiple systems simultaneously
via batching. To create a batch of multiple systems, you can simply pass
a list of atomistic objects to the `initialize_state` function.
"""

# %% Create multiple systems
cu_atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
al_atoms = bulk("Al", "fcc", a=4.05, cubic=True)
ag_atoms = bulk("Ag", "fcc", a=4.09, cubic=True)
# Initialize both as a single batched state
multi_state = ts.initialize_state(
    [cu_atoms, al_atoms, ag_atoms], device=torch.device("cpu"), dtype=torch.float64
)

print(
    f"Multi-state has {multi_state.n_atoms} total atoms across {multi_state.n_batches} batches"
)

# we can see how the shapes of batchwise, atomwise, and global properties change
print(f"Positions shape: {multi_state.positions.shape}")
print(f"Cell shape: {multi_state.cell.shape}")
print(f"PBC: {multi_state.pbc}")
print(f"Batch indices shape: {multi_state.batch.shape}")


# %% [markdown]
"""
## Manipulating SimState

### Datatype and Device

All SimState tensors are stored on the same device. Further, all floating point
tensors are stored as the same datatype, `torch.float64` by default.

We can change both the datatype and the state by calling the `to` method.
"""

# %%
if torch.cuda.is_available():
    multi_state = multi_state.to(device=torch.device("cuda"), dtype=torch.float32)


# %% [markdown]
"""
### Slicing, Splitting, Popping, and more

SimState supports many convenience operations for manipulating batched states. Slicing
is supported through fancy indexing, e.g. `state[[0, 1, 2]]` will return a new state
containing only the first three batches. The other operations are available through the
`pop`, `split`, `clone`, and `to` methods.
"""

# %% we can copy the state with the clone method
multi_state_copy = multi_state.clone()
print(f"This state has {multi_state_copy.n_batches} batches")

# we can pop states off while modifying the original state
popped_states = multi_state_copy.pop([0, 2])
print(
    f"We popped {len(popped_states)} states, leaving us with "
    f"{multi_state_copy.n_batches} batch in the original state"
)

# we can put them back together with concatenate
multi_state_full = ts.concatenate_states([*popped_states, multi_state_copy])
print(f"Again we have {multi_state_full.n_batches} batches in the full state")

# or if we don't want to modify the original state, we can instead index into it
# negative indexing
last_state = multi_state[-1]

# slicing
first_two_states = multi_state[:2]

# fancy indexing
sliced_state = multi_state[[0, 2]]

print(f"Unlike pop, slicing returns a {type(sliced_state)} instead of a list")

# but we could also get a list of states with the split method
list_of_sliced_states = sliced_state.split()

print(f"Which now is a list of {len(list_of_sliced_states)} states")


# %% [markdown]
"""

You can extract specific batches from a batched state using Python's slicing syntax.
This is extremely useful for analyzing specific systems or for implementing complex
workflows where different systems need separate processing:

The slicing interface follows Python's standard indexing conventions, making it
intuitive to use. Behind the scenes, TorchSim is creating a new SimState with only the
selected batches, maintaining all the necessary properties and relationships.

Note the difference between these operations:
- `split()` returns all batches as separate states but doesn't modify the original
- `pop()` removes specified batches from the original state and returns them as
separate states
- `__getitem__` (slicing) creates a new state with specified batches without modifying
the original

This flexibility allows you to structure your simulation workflows in the most
efficient way for your specific needs.

### Splitting and Popping Batches

SimState provides methods to split a batched state into separate states or to remove
specific batches:
"""

# %% [markdown]
"""
## Converting States to Other Formats

SimState objects can be converted back to other atomistic representations. This is
useful when you need to use external libraries for analysis or visualization:
"""

# %% Convert to ASE Atoms
atoms_list = multi_state.to_atoms()
print(f"Converted to {len(atoms_list)} ASE Atoms objects")
print(f"First atoms object has chemical formula: {atoms_list[0].get_chemical_formula()}")

# Convert to pymatgen Structure
structures = multi_state.to_structures()
print(f"Converted to {len(structures)} pymatgen Structure objects")
print(f"First structure has formula: {structures[0].formula}")

# Convert to PhonopyAtoms (for phonon calculations)
phonopy_atoms = multi_state.to_phonopy()
print(f"Converted to {len(phonopy_atoms)} PhonopyAtoms objects")
print(f"First PhonopyAtoms object has chemical symbols: {phonopy_atoms[0].symbols}")


# %% [markdown]
"""

## Extending SimState: The MDState

MDState is defined in the `torch_sim.integrators` module. It is a subclass of SimState
for molecular dynamics simulations. It includes additional properties like momenta,
forces, and energy. Here, we instantiate an MDState from a SimState by zeroing out the
additional properties.

Since it inherits from SimState, it supports all the same operations. In general,
all state objects in TorchSim support the operations described above.
"""

# %%
from torch_sim.integrators import MDState
from dataclasses import asdict

# Create an MDState from a SimState
md_state = MDState(
    **asdict(si_state),  # Copy all SimState properties
    momenta=torch.zeros_like(si_state.positions),  # Initial 0 momenta
    forces=torch.zeros_like(si_state.positions),  # Initial 0 forces
    energy=torch.zeros((si_state.n_batches,), device=si_state.device),  # Initial 0 energy
)

print("MDState properties:")
scope = infer_property_scope(md_state)
print("Global properties:", scope["global"])
print("Per-atom properties:", scope["per_atom"])
print("Per-batch properties:", scope["per_batch"])


# %% [markdown]
"""
## Conclusion

The SimState class is the foundation of atomistic simulations in TorchSim. It provides:

1. A flexible, GPU-compatible representation for atomistic systems
2. Support for batched operations to efficiently process multiple systems
3. Seamless conversion to and from common atomistic formats
4. Properties and methods for slicing, combining, and manipulating atomistic data

With this understanding of the SimState, you're now ready to build complex simulation
workflows using TorchSim's integrators, optimizers, and other modules.
"""
