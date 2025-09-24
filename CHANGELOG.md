## v0.2.1

2025-05-01

## What's Changed

### 💥 Breaking Changes

* Remove higher level model imports by @CompRhys in https://github.com/TorchSim/torch-sim/pull/179

### 🛠 Enhancements

* Add per atom energies and stresses for batched LJ by @abhijeetgangan in https://github.com/TorchSim/torch-sim/pull/144
* throw error if autobatcher type is wrong by @orionarcher in https://github.com/TorchSim/torch-sim/pull/167

### 🐛 Bug Fixes

* Fix column->row cell vector mismatch in integrators by @CompRhys in https://github.com/TorchSim/torch-sim/pull/175
* Mattersim fix tensors on wrong device (CPU->GPU) by @orionarcher in https://github.com/TorchSim/torch-sim/pull/154
* fix `npt_langevin` by @jla-gardner in https://github.com/TorchSim/torch-sim/pull/153
* Make sure to move data to CPU before calling vesin by @Luthaf in https://github.com/TorchSim/torch-sim/pull/156
* Fix virial calculations in `optimizers` and `integrators` by @janosh in https://github.com/TorchSim/torch-sim/pull/163
* Pad memory estimation by @orionarcher in https://github.com/TorchSim/torch-sim/pull/160
* Refactor sevennet model by @YutackPark in https://github.com/TorchSim/torch-sim/pull/172
* `io` optional dependencies in `pyproject.toml` by @curtischong in https://github.com/TorchSim/torch-sim/pull/185

### 📖 Documentation

* (tiny) add graph-pes to README by @jla-gardner in https://github.com/TorchSim/torch-sim/pull/149
* Better module fig by @janosh in https://github.com/TorchSim/torch-sim/pull/168

### 🚀 Performance

* More efficient Orb `state_to_atoms_graph` calculation by @AdeeshKolluru in https://github.com/TorchSim/torch-sim/pull/165

### 🚧 CI

* Refactor `test_math.py` and `test_transforms.py` by @janosh in https://github.com/TorchSim/torch-sim/pull/151

### 🏥 Package Health

* Try out hatchling for build vs setuptools by @CompRhys in https://github.com/TorchSim/torch-sim/pull/177

### 🏷️ Type Hints

* Add `torch_sim/typing.py` by @janosh in https://github.com/TorchSim/torch-sim/pull/157

### 📦 Dependencies

* Bump `mace-torch` to v0.3.12 by @janosh in https://github.com/TorchSim/torch-sim/pull/170
* Update metatrain dependency by @Luthaf in https://github.com/TorchSim/torch-sim/pull/186

## New Contributors

* @Luthaf made their first contribution in https://github.com/TorchSim/torch-sim/pull/156
* @YutackPark made their first contribution in https://github.com/TorchSim/torch-sim/pull/172
* @curtischong made their first contribution in https://github.com/TorchSim/torch-sim/pull/185

**Full Changelog**: https://github.com/TorchSim/torch-sim/compare/v0.2.0...v0.2.1

## v0.2.0

### Bug Fixes 🐛

* Fix integrate reporting kwarg to arg error, https://github.com/TorchSim/torch-sim/issues/113 (raised by @hn-yu)
* Allow runners to take large initial batches, https://github.com/TorchSim/torch-sim/issues/128 (raised by @YutackPark)
* Add Fairchem model support for PBC, https://github.com/TorchSim/torch-sim/issues/111 (raised by @ryanliu30)

### Enhancements 🛠

* **breaking** Rename `HotSwappingAutobatcher` to `InFlightAutobatcher` and `ChunkingAutoBatcher` to `BinningAutoBatcher`, https://github.com/TorchSim/torch-sim/pull/143 @orionarcher
* Support for Orbv3, https://github.com/TorchSim/torch-sim/pull/140, @AdeeshKolluru
* Support metatensor models, https://github.com/TorchSim/torch-sim/pull/141, @frostedoyter @Luthaf
* Support for graph-pes models, https://github.com/TorchSim/torch-sim/pull/118 @jla-gardner
* Support MatterSim and fix ASE cell convention issues, https://github.com/TorchSim/torch-sim/pull/112 @CompRhys
* Implement positions only FIRE optimization, https://github.com/TorchSim/torch-sim/pull/139 @abhijeetgangan
* Allow different temperatures in batches, https://github.com/TorchSim/torch-sim/pull/123 @orionarcher
* FairChem model updates: PBC handling, test on OMat24 e-trained model, https://github.com/TorchSim/torch-sim/pull/126 @AdeeshKolluru
* FairChem model from_data_list support, https://github.com/TorchSim/torch-sim/pull/138 @ryanliu30
* New correlation function module, https://github.com/TorchSim/torch-sim/pull/115 @stefanbringuier

### Documentation 📖

* Improved model documentation, https://github.com/TorchSim/torch-sim/pull/121 @orionarcher
* Plot of TorchSim module graph in docs, https://github.com/TorchSim/torch-sim/pull/132 @janosh

### House-Keeping 🧹

* Only install HF for fairchem tests, https://github.com/TorchSim/torch-sim/pull/134 @CompRhys
* Don't download MBD in CI, https://github.com/TorchSim/torch-sim/pull/135 @orionarcher
* Tighten graph-pes test bounds, https://github.com/TorchSim/torch-sim/pull/143 @orionarcher

## v0.1.0

Initial release.
