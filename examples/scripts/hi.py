import ase


hi = ase.Atoms()
yo = ase.Atoms(hi)

# SimState(
#     positions=torch.randn(10, 3),
#     masses=torch.randn(10),
#     cell=torch.randn(3, 3),
#     pbc=True,
#     atomic_numbers=torch.randint(1, 100, (10,)),
#     batch=torch.randint(0, 10, (10,)),
#     system_idx=torch.randint(0, 10, (10,)),
# )
