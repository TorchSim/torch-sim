<!-- markdownlint-disable -->
# Changelog

## v0.3.0

Thank you to everyone who contributed to this release! @t-reents, @curtischong, and @CompRhys did great work squashing an issue with `SimState` concatenation. @curtischong continued his crusade to type and improve the TorchSim API. @orionarcher, @kianpu34593, and @janosh all made contributions that continue to improve package quality and usability. 🚀

## What's Changed

### 🛠 Enhancements
* Define attribute scopes in `SimStates` by @curtischong, @CompRhys, @orionarcher in [#228](https://github.com/Radical-AI/torch-sim/pull/228)
* Improve typing of `ModelInterface` by @curtischong, @CompRhys in [#215](https://github.com/Radical-AI/torch-sim/pull/215)
* Make `system_idx` non-optional in `SimState` by @curtischong in [#231](https://github.com/Radical-AI/torch-sim/pull/231)
* Add new states when the `max_memory_scaler` is updated by @kianpu34593 in [#222](https://github.com/Radical-AI/torch-sim/pull/222)
* Rename `batch` to `system` by @curtischong in [#217](https://github.com/Radical-AI/torch-sim/pull/217), [#233](https://github.com/Radical-AI/torch-sim/pull/233)

### 🐛 Bug Fixes
* Initial fix for concatenation of states in `InFlightAutoBatcher` by @t-reents in [#219](https://github.com/Radical-AI/torch-sim/pull/219)
* Finish fix for `SimState` concatenation by @t-reents and @curtischong in [#232](https://github.com/Radical-AI/torch-sim/pull/232)
* Fix broken code block in low-level tutorial by @CompRhys in [#226](https://github.com/Radical-AI/torch-sim/pull/226)
* Update metatomic checkpoint to fix tests by @curtischong in [#223](https://github.com/Radical-AI/torch-sim/pull/223)
* Fix memory scaling in `determine_max_batch_size` by @t-reents, @janosh in [#212](https://github.com/Radical-AI/torch-sim/pull/212)

### 📖 Documentation
* Update README plot with more models by @orionarcher in [#236](https://github.com/Radical-AI/torch-sim/pull/236), [#237](https://github.com/Radical-AI/torch-sim/pull/237)
* Update `citation.cff` by @CompRhys in [#225](https://github.com/Radical-AI/torch-sim/pull/225)

**Full Changelog**: https://github.com/Radical-AI/torch-sim/compare/v0.2.2...v0.3.0

## v0.2.2

## What's Changed
### 💥 Breaking Changes
* Remove higher level model imports by @CompRhys in https://github.com/Radical-AI/torch-sim/pull/179
### 🛠 Enhancements
* Add per atom energies and stresses for batched LJ by @abhijeetgangan in https://github.com/Radical-AI/torch-sim/pull/144
* throw error if autobatcher type is wrong by @orionarcher in https://github.com/Radical-AI/torch-sim/pull/167
### 🐛 Bug Fixes
* Mattersim fix tensors on wrong device (CPU->GPU) by @orionarcher in https://github.com/Radical-AI/torch-sim/pull/154
* fix `npt_langevin` by @jla-gardner in https://github.com/Radical-AI/torch-sim/pull/153
* Make sure to move data to CPU before calling vesin by @Luthaf in https://github.com/Radical-AI/torch-sim/pull/156
* Fix virial calculations in `optimizers` and `integrators` by @janosh in https://github.com/Radical-AI/torch-sim/pull/163
* Pad memory estimation by @orionarcher in https://github.com/Radical-AI/torch-sim/pull/160
* Refactor sevennet model by @YutackPark in https://github.com/Radical-AI/torch-sim/pull/172
* `io` optional dependencies in `pyproject.toml` by @curtischong in https://github.com/Radical-AI/torch-sim/pull/185
* Fix column->row cell vector mismatch in integrators by @CompRhys in https://github.com/Radical-AI/torch-sim/pull/175
### 📖 Documentation
* (tiny) add graph-pes to README by @jla-gardner in https://github.com/Radical-AI/torch-sim/pull/149
* Better module fig by @janosh in https://github.com/Radical-AI/torch-sim/pull/168
### 🚀 Performance
* More efficient Orb `state_to_atoms_graph` calculation by @AdeeshKolluru in https://github.com/Radical-AI/torch-sim/pull/165
### 🚧 CI
* Refactor `test_math.py` and `test_transforms.py` by @janosh in https://github.com/Radical-AI/torch-sim/pull/151
### 🏥 Package Health
* Try out hatchling for build vs setuptools by @CompRhys in https://github.com/Radical-AI/torch-sim/pull/177
### 📦 Dependencies
* Bump `mace-torch` to v0.3.12 by @janosh in https://github.com/Radical-AI/torch-sim/pull/170
* Update metatrain dependency by @Luthaf in https://github.com/Radical-AI/torch-sim/pull/186
### 🏷️ Type Hints
* Add `torch_sim/typing.py` by @janosh in https://github.com/Radical-AI/torch-sim/pull/157

## New Contributors
* @Luthaf made their first contribution in https://github.com/Radical-AI/torch-sim/pull/156
* @YutackPark made their first contribution in https://github.com/Radical-AI/torch-sim/pull/172
* @curtischong made their first contribution in https://github.com/Radical-AI/torch-sim/pull/185

**Full Changelog**: https://github.com/Radical-AI/torch-sim/compare/v0.2.0...v0.2.1

## v0.2.1

## What's Changed

### 💥 Breaking Changes

* Remove higher level model imports by @CompRhys in [#179](https://github.com/TorchSim/torch-sim/pull/179)

### 🛠 Enhancements

* Add per atom energies and stresses for batched LJ by @abhijeetgangan in [#144](https://github.com/TorchSim/torch-sim/pull/144)
* throw error if autobatcher type is wrong by @orionarcher in [#167](https://github.com/TorchSim/torch-sim/pull/167)

### 🐛 Bug Fixes

* Fix column->row cell vector mismatch in integrators by @CompRhys in [#175](https://github.com/TorchSim/torch-sim/pull/175)
* Mattersim fix tensors on wrong device (CPU->GPU) by @orionarcher in [#154](https://github.com/TorchSim/torch-sim/pull/154)
* fix `npt_langevin` by @jla-gardner in [#153](https://github.com/TorchSim/torch-sim/pull/153)
* Make sure to move data to CPU before calling vesin by @Luthaf in [#156](https://github.com/TorchSim/torch-sim/pull/156)
* Fix virial calculations in `optimizers` and `integrators` by @janosh in [#163](https://github.com/TorchSim/torch-sim/pull/163)
* Pad memory estimation by @orionarcher in [#160](https://github.com/TorchSim/torch-sim/pull/160)
* Refactor sevennet model by @YutackPark in [#172](https://github.com/TorchSim/torch-sim/pull/172)
* `io` optional dependencies in `pyproject.toml` by @curtischong in [#185](https://github.com/TorchSim/torch-sim/pull/185)

### 📖 Documentation

* (tiny) add graph-pes to README by @jla-gardner in [#149](https://github.com/TorchSim/torch-sim/pull/149)
* Better module fig by @janosh in [#168](https://github.com/TorchSim/torch-sim/pull/168)

### 🚀 Performance

* More efficient Orb `state_to_atoms_graph` calculation by @AdeeshKolluru in [#165](https://github.com/TorchSim/torch-sim/pull/165)

### 🚧 CI

* Refactor `test_math.py` and `test_transforms.py` by @janosh in [#151](https://github.com/TorchSim/torch-sim/pull/151)

### 🏥 Package Health

* Try out hatchling for build vs setuptools by @CompRhys in [#177](https://github.com/TorchSim/torch-sim/pull/177)

### 🏷️ Type Hints

* Add `torch-sim/typing.py` by @janosh in [#157](https://github.com/TorchSim/torch-sim/pull/157)

### 📦 Dependencies

* Bump `mace-torch` to v0.3.12 by @janosh in [#170](https://github.com/TorchSim/torch-sim/pull/170)
* Update metatrain dependency by @Luthaf in [#186](https://github.com/TorchSim/torch-sim/pull/186)

## New Contributors

* @Luthaf made their first contribution in [#156](https://github.com/TorchSim/torch-sim/pull/156)
* @YutackPark made their first contribution in [#172](https://github.com/TorchSim/torch-sim/pull/172)
* @curtischong made their first contribution in [#185](https://github.com/TorchSim/torch-sim/pull/185)

**Full Changelog**: https://github.com/torchsim/torch-sim/compare/v0.2.0...v0.2.1

## v0.2.0

### Bug Fixes 🐛

* Fix integrate reporting kwarg to arg error, [#113](https://github.com/TorchSim/torch-sim/pull/113) (raised by @hn-yu)
* Allow runners to take large initial batches, [#128](https://github.com/TorchSim/torch-sim/pull/128) (raised by @YutackPark)
* Add Fairchem model support for PBC, [#111](https://github.com/TorchSim/torch-sim/pull/111) (raised by @ryanliu30)

### Enhancements 🛠

* **breaking** Rename `HotSwappingAutobatcher` to `InFlightAutobatcher` and `ChunkingAutoBatcher` to `BinningAutoBatcher`, [#143](https://github.com/TorchSim/torch-sim/pull/143) @orionarcher
* Support for Orbv3, [#140](https://github.com/TorchSim/torch-sim/pull/140), @AdeeshKolluru
* Support metatensor models, [#141](https://github.com/TorchSim/torch-sim/pull/141), @frostedoyter @Luthaf
* Support for graph-pes models, [#118](https://github.com/TorchSim/torch-sim/pull/118) @jla-gardner
* Support MatterSim and fix ASE cell convention issues, [#112](https://github.com/TorchSim/torch-sim/pull/112) @CompRhys
* Implement positions only FIRE optimization, [#139](https://github.com/TorchSim/torch-sim/pull/139) @abhijeetgangan
* Allow different temperatures in batches, [#123](https://github.com/TorchSim/torch-sim/pull/123) @orionarcher
* FairChem model updates: PBC handling, test on OMat24 e-trained model, [#126](https://github.com/TorchSim/torch-sim/pull/126) @AdeeshKolluru
* FairChem model from_data_list support, [#138](https://github.com/TorchSim/torch-sim/pull/138) @ryanliu30
* New correlation function module, [#115](https://github.com/TorchSim/torch-sim/pull/115) @stefanbringuier

### Documentation 📖

* Improved model documentation, [#121](https://github.com/TorchSim/torch-sim/pull/121) @orionarcher
* Plot of TorchSim module graph in docs, [#132](https://github.com/TorchSim/torch-sim/pull/132) @janosh

### House-Keeping 🧹

* Only install HF for fairchem tests, [#134](https://github.com/TorchSim/torch-sim/pull/134) @CompRhys
* Don't download MBD in CI, [#135](https://github.com/TorchSim/torch-sim/pull/135) @orionarcher
* Tighten graph-pes test bounds, [#143](https://github.com/TorchSim/torch-sim/pull/143) @orionarcher

## v0.1.0

Initial release.
