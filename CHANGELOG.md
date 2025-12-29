<!-- markdownlint-disable -->
# Changelog

## Unreleased

### 🎉 New Features
* Constraints support for molecular dynamics and optimization by @thomasloux in [#294](https://github.com/TorchSim/torch-sim/pull/294)
  - Added `FixAtoms` constraint to fix specific atoms in place
  - Added `FixCom` constraint to prevent center of mass drift
  - Constraints automatically adjust degrees of freedom for accurate temperature calculations
  - Full support across all integrators (NVE, NVT, NPT) and optimizers (FIRE, Gradient Descent)
  - Constraints preserved during state manipulation (slicing, splitting, concatenation)

## v0.5.0

This release focuses on improving batch processing capabilities across TorchSim. The neighbor list module has been completely refactored to support batched calculations with multiple backend implementations, elastic tensor calculations now leverage batched operations for improved performance, and a bug fix ensures Monte Carlo swaps work correctly with ragged (different-sized) systems.

### 🎉 New Features
* Refactor neighbor list module with batched support and multiple backends by @abhijeetgangan in [#348](https://github.com/TorchSim/torch-sim/pull/348)
  - New unified `torchsim_nl` function with automatic backend selection
  - Multiple implementations: Alchemiops (NVIDIA CUDA), Vesin, torch_nl, and pure PyTorch fallback
  - Support for both single-system and batched (multi-system) calculations
  - Automatic selection of best available implementation based on installed packages

### 🛠 Enhancements
* Batch elastic operations by @orionarcher in [#384](https://github.com/TorchSim/torch-sim/pull/384)
  - `calculate_elastic_tensor` now uses `ts.static` runner for batched calculations
  - Added `autobatcher` parameter for memory-efficient processing of deformations
  - Added `pbar` parameter for progress bar support

### 🐛 Bug Fixes
* Fix Monte Carlo swap for ragged systems by @curtischong in [#380](https://github.com/TorchSim/torch-sim/pull/380)
  - Fixed `generate_swaps` calculation of system start indices for systems with different atom counts

## v0.4.2

Thank you to everyone who contributed to this release! This release includes important bug fixes and new features. @thomasloux, @orionarcher, @WillEngler, @RishikeshMagar, @nh-univie, @andrewrm98, @danielzuegner, and others made valuable contributions. 🚀

### 🎉 New Features
* Add CSVR / V-Rescale thermostat and anisotropic C rescale barostat by @thomasloux in [#326](https://github.com/TorchSim/torch-sim/pull/326)
* Support for electrostatics by @orionarcher in [#373](https://github.com/TorchSim/torch-sim/pull/373)
* Add support for AMD GPUs (consumer/datacenter) by @amacbride in [#347](https://github.com/TorchSim/torch-sim/pull/347)

### 🐛 Bug Fixes
* Fix: add init_kwargs to ts.integrate by @danielzuegner in [#360](https://github.com/TorchSim/torch-sim/pull/360)
* Fix PBC extraction to CPU fairchem model by @nh-univie in [#368](https://github.com/TorchSim/torch-sim/pull/368)
* Handle tensor PBC input in FairChemV1Model by @WillEngler in [#372](https://github.com/TorchSim/torch-sim/pull/372)
* Fix Comments Issue#309 by @RishikeshMagar in [#378](https://github.com/TorchSim/torch-sim/pull/378)
* Fix fairchem-legacy tests by removing explicit Hugging Face login by @WillEngler in [#369](https://github.com/TorchSim/torch-sim/pull/369)

### 🛠 Enhancements
* Consolidate model and model_name args in FairchemModel by @orionarcher in [#377](https://github.com/TorchSim/torch-sim/pull/377)

## New Contributors
* @amacbride made their first contribution in [#347](https://github.com/TorchSim/torch-sim/pull/347)
* @danielzuegner made their first contribution in [#360](https://github.com/TorchSim/torch-sim/pull/360)
* @RishikeshMagar made their first contribution in [#378](https://github.com/TorchSim/torch-sim/pull/378)

## v0.4.1

Thank you to everyone who contributed to this release! This release includes important bug fixes, new features, and API improvements. @thomasloux, @curtischong, @CompRhys, @orionarcher, @WillEngler, @samanvya10, @hn-yu, @wendymak8, @chuin-wei, @pragnya17, and many others made valuable contributions. 🚀

### 💥 Breaking Changes
* Standardize argument order to (state, model) by @pragnya17 in [#341](https://github.com/TorchSim/torch-sim/pull/341)
* Deprecate pbc_wrap_general by @curtischong in [#305](https://github.com/TorchSim/torch-sim/pull/305)

### 🎉 New Features
* Mixed PBC Support by @curtischong in [#320](https://github.com/TorchSim/torch-sim/pull/320)
* Feature: Batched NVT Nose-Hoover by @thomasloux in [#265](https://github.com/TorchSim/torch-sim/pull/265)
* Add degrees of freedom (dof) in state and temperature computation by @thomasloux in [#328](https://github.com/TorchSim/torch-sim/pull/328)
* Add sources npt langevin by @thomasloux in [#298](https://github.com/TorchSim/torch-sim/pull/298)

### 🐛 Bug Fixes
* MACE: transfer atomic numbers to CPU before converting to numpy by @t-reents in [#289](https://github.com/TorchSim/torch-sim/pull/289)
* Fixed max atoms memory estimation by @nh-univie in [#279](https://github.com/TorchSim/torch-sim/pull/279)
* Fixing model loading logic (names and cache dir) for fairchem models by @nh-univie in [#278](https://github.com/TorchSim/torch-sim/pull/278)
* Fix fairchem and neighbors tests by @WillEngler in [#317](https://github.com/TorchSim/torch-sim/pull/317)
* Fix #293: State to device side effects by @samanvya10 in [#297](https://github.com/TorchSim/torch-sim/pull/297)
* Fix graph-pes key issue by @jla-gardner in [#303](https://github.com/TorchSim/torch-sim/pull/303)
* Fix calculate static state after relax by @curtischong in [#338](https://github.com/TorchSim/torch-sim/pull/338)
* Misc fixes by @orionarcher in [#336](https://github.com/TorchSim/torch-sim/pull/336)
* Fix cell to cellpar by @thomasloux in [#342](https://github.com/TorchSim/torch-sim/pull/342)
* Fix failing Docs build by @CompRhys in [#296](https://github.com/TorchSim/torch-sim/pull/296)
* Fix: correct speedup plot image path in README by @Joaqland in [#333](https://github.com/TorchSim/torch-sim/pull/333)

### 🛠 Enhancements
* Put SimState Init logic into __post_init__ (and enforce kw_args=true for SimState children) by @curtischong in [#335](https://github.com/TorchSim/torch-sim/pull/335)
* Replace vars(state) with state.attributes by @orionarcher in [#329](https://github.com/TorchSim/torch-sim/pull/329)
* Rename and reorder some variables by @orionarcher in [#316](https://github.com/TorchSim/torch-sim/pull/316)
* Add SevenNet path, str types for model arg by @YutackPark in [#322](https://github.com/TorchSim/torch-sim/pull/322)
* Add version attribute by @thomasloux in [#311](https://github.com/TorchSim/torch-sim/pull/311)
* Convert cell_to_cellpar from ase's numpy implementation to pytorch by @wendymak8 in [#306](https://github.com/TorchSim/torch-sim/pull/306)
* Autobatch OOM handling by @chuin-wei in [#337](https://github.com/TorchSim/torch-sim/pull/337)
* Allow Mace to be loaded from a model path by @orionarcher in [#349](https://github.com/TorchSim/torch-sim/pull/349)
* NPTLangevinState inherits from MDState by @hn-yu in [#299](https://github.com/TorchSim/torch-sim/pull/299)

### 📖 Documentation
* Add integrators in docs by @thomasloux in [#290](https://github.com/TorchSim/torch-sim/pull/290)

### 🏷️ Type Hints
* Create py.typed by @arosen93 in [#287](https://github.com/TorchSim/torch-sim/pull/287)

### 🧹 House-Keeping
* Remove unused imports orb and updates class type check by @thomasloux in [#292](https://github.com/TorchSim/torch-sim/pull/292)
* Skip FairChem tests on forks due to HF secret by @CompRhys in [#295](https://github.com/TorchSim/torch-sim/pull/295)

### 📦 Dependencies
* Loosen numpy dependency by @chuin-wei in [#321](https://github.com/TorchSim/torch-sim/pull/321)

## New Contributors
* @arosen93 made their first contribution in [#287](https://github.com/TorchSim/torch-sim/pull/287)
* @nh-univie made their first contribution in [#278](https://github.com/TorchSim/torch-sim/pull/278)
* @samanvya10 made their first contribution in [#297](https://github.com/TorchSim/torch-sim/pull/297)
* @wendymak8 made their first contribution in [#306](https://github.com/TorchSim/torch-sim/pull/306)
* @pragnya17 made their first contribution in [#341](https://github.com/TorchSim/torch-sim/pull/341)
* @chuin-wei made their first contribution in [#321](https://github.com/TorchSim/torch-sim/pull/321)
* @hn-yu made their first contribution in [#299](https://github.com/TorchSim/torch-sim/pull/299)
* @Joaqland made their first contribution in [#333](https://github.com/TorchSim/torch-sim/pull/333)

## v0.4.0

Thank you to everyone who contributed to this release! This release includes significant API improvements and breaking changes. @janosh led a major API redesign to improve usability. @stefanbringuier added heat flux calculations. @curtischong continued improving type safety across the codebase. @CompRhys, @orionarcher, @WillEngler, and @thomasloux all made valuable contributions. 🚀

## What's Changed

### 💥 Breaking Changes
* Fairchem v2 support by @janosh, @CompRhys, @abhijeetgangan, @orionarcher in [#211](https://github.com/TorchSim/torch-sim/pull/211)
* Big breaking API redesign by @janosh in [#264](https://github.com/TorchSim/torch-sim/pull/264)
* Rename Flavors to more descriptive names by @orionarcher in [#282](https://github.com/TorchSim/torch-sim/pull/282)

### 🎉 New Features
* Enhancement: Heat Flux Function by @stefanbringuier in [#127](https://github.com/TorchSim/torch-sim/pull/127)

### 🐛 Bug Fixes
* Fix: orb squeeze provides incorrect shape for energy tensor by @thomasloux in [#257](https://github.com/TorchSim/torch-sim/pull/257)
* Fix docs build by @WillEngler in [#271](https://github.com/TorchSim/torch-sim/pull/271)

### 🛠 Enhancements
* Fairchem legacy support by @CompRhys in [#270](https://github.com/TorchSim/torch-sim/pull/270)

### 📖 Documentation
* Update citation in README.md by @orionarcher in [#240](https://github.com/TorchSim/torch-sim/pull/240)
* Add GOVERNANCE.md and remove Contributor's Certification checkbox and language by @WillEngler in [#272](https://github.com/TorchSim/torch-sim/pull/272)
* Remove Contributor License Agreement (CLA) in favor of certification in contributing.md by @WillEngler in [#267](https://github.com/TorchSim/torch-sim/pull/267)
* Small update to README and CHANGELOG by @orionarcher in [#283](https://github.com/TorchSim/torch-sim/pull/283)

### 🏷️ Type Hints
* mypy type math.py and test_math.py by @curtischong in [#242](https://github.com/TorchSim/torch-sim/pull/242)
* Type test_io, neighbors, and transforms by @curtischong in [#243](https://github.com/TorchSim/torch-sim/pull/243)
* Type trajectory by @curtischong in [#244](https://github.com/TorchSim/torch-sim/pull/244)

### 🧹 House-Keeping
* MAINT: update pins in MACE phonons example. Remove misleading ty from PR template by @CompRhys in [#239](https://github.com/TorchSim/torch-sim/pull/239)

## New Contributors
* @thomasloux made their first contribution in [#257](https://github.com/TorchSim/torch-sim/pull/257)
* @WillEngler made their first contribution in [#267](https://github.com/TorchSim/torch-sim/pull/267)

**Full Changelog**: https://github.com/TorchSim/torch-sim/compare/v0.3.0...v0.4.0

## v0.3.0

Thank you to everyone who contributed to this release! @t-reents, @curtischong, and @CompRhys did great work squashing an issue with `SimState` concatenation. @curtischong continued his crusade to type and improve the TorchSim API. @orionarcher, @kianpu34593, and @janosh all made contributions that continue to improve package quality and usability. 🚀

## What's Changed

### 🛠 Enhancements
* Define attribute scopes in `SimStates` by @curtischong, @CompRhys, @orionarcher in [#228](https://github.com/TorchSim/torch-sim/pull/228)
* Improve typing of `ModelInterface` by @curtischong, @CompRhys in [#215](https://github.com/TorchSim/torch-sim/pull/215)
* Make `system_idx` non-optional in `SimState` by @curtischong in [#231](https://github.com/TorchSim/torch-sim/pull/231)
* Add new states when the `max_memory_scaler` is updated by @kianpu34593 in [#222](https://github.com/TorchSim/torch-sim/pull/222)
* Rename `batch` to `system` by @curtischong in [#217](https://github.com/TorchSim/torch-sim/pull/217), [#233](https://github.com/TorchSim/torch-sim/pull/233)

### 🐛 Bug Fixes
* Initial fix for concatenation of states in `InFlightAutoBatcher` by @t-reents in [#219](https://github.com/TorchSim/torch-sim/pull/219)
* Finish fix for `SimState` concatenation by @t-reents and @curtischong in [#232](https://github.com/TorchSim/torch-sim/pull/232)
* Fix broken code block in low-level tutorial by @CompRhys in [#226](https://github.com/TorchSim/torch-sim/pull/226)
* Update metatomic checkpoint to fix tests by @curtischong in [#223](https://github.com/TorchSim/torch-sim/pull/223)
* Fix memory scaling in `determine_max_batch_size` by @t-reents, @janosh in [#212](https://github.com/TorchSim/torch-sim/pull/212)

### 📖 Documentation
* Update README plot with more models by @orionarcher in [#236](https://github.com/TorchSim/torch-sim/pull/236), [#237](https://github.com/TorchSim/torch-sim/pull/237)
* Update `citation.cff` by @CompRhys in [#225](https://github.com/TorchSim/torch-sim/pull/225)

**Full Changelog**: https://github.com/TorchSim/torch-sim/compare/v0.2.2...v0.3.0

## v0.2.2

## What's Changed
### 💥 Breaking Changes
* Remove higher level model imports by @CompRhys in https://github.com/TorchSim/torch-sim/pull/179
### 🛠 Enhancements
* Add per atom energies and stresses for batched LJ by @abhijeetgangan in https://github.com/TorchSim/torch-sim/pull/144
* throw error if autobatcher type is wrong by @orionarcher in https://github.com/TorchSim/torch-sim/pull/167
### 🐛 Bug Fixes
* Mattersim fix tensors on wrong device (CPU->GPU) by @orionarcher in https://github.com/TorchSim/torch-sim/pull/154
* fix `npt_langevin` by @jla-gardner in https://github.com/TorchSim/torch-sim/pull/153
* Make sure to move data to CPU before calling vesin by @Luthaf in https://github.com/TorchSim/torch-sim/pull/156
* Fix virial calculations in `optimizers` and `integrators` by @janosh in https://github.com/TorchSim/torch-sim/pull/163
* Pad memory estimation by @orionarcher in https://github.com/TorchSim/torch-sim/pull/160
* Refactor sevennet model by @YutackPark in https://github.com/TorchSim/torch-sim/pull/172
* `io` optional dependencies in `pyproject.toml` by @curtischong in https://github.com/TorchSim/torch-sim/pull/185
* Fix column->row cell vector mismatch in integrators by @CompRhys in https://github.com/TorchSim/torch-sim/pull/175
### 📖 Documentation
* (tiny) add graph-pes to README by @jla-gardner in https://github.com/TorchSim/torch-sim/pull/149
* Better module fig by @janosh in https://github.com/TorchSim/torch-sim/pull/168
### 🚀 Performance
* More efficient Orb `state_to_atoms_graph` calculation by @AdeeshKolluru in https://github.com/TorchSim/torch-sim/pull/165
### 🚧 CI
* Refactor `test_math.py` and `test_transforms.py` by @janosh in https://github.com/TorchSim/torch-sim/pull/151
### 🏥 Package Health
* Try out hatchling for build vs setuptools by @CompRhys in https://github.com/TorchSim/torch-sim/pull/177
### 📦 Dependencies
* Bump `mace-torch` to v0.3.12 by @janosh in https://github.com/TorchSim/torch-sim/pull/170
* Update metatrain dependency by @Luthaf in https://github.com/TorchSim/torch-sim/pull/186
### 🏷️ Type Hints
* Add `torch_sim/typing.py` by @janosh in https://github.com/TorchSim/torch-sim/pull/157

## New Contributors
* @Luthaf made their first contribution in https://github.com/TorchSim/torch-sim/pull/156
* @YutackPark made their first contribution in https://github.com/TorchSim/torch-sim/pull/172
* @curtischong made their first contribution in https://github.com/TorchSim/torch-sim/pull/185

**Full Changelog**: https://github.com/TorchSim/torch-sim/compare/v0.2.0...v0.2.1

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
