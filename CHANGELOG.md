## v0.2.1

2025-05-01

## What's Changed

### 💥 Breaking Changes

* Remove higher level model imports by @CompRhys in #179

### 🛠 Enhancements

* Add per atom energies and stresses for batched LJ by @abhijeetgangan in #144
* throw error if autobatcher type is wrong by @orionarcher in #167

### 🐛 Bug Fixes

* Fix column->row cell vector mismatch in integrators by @CompRhys in #175
* Mattersim fix tensors on wrong device (CPU->GPU) by @orionarcher in #154
* fix `npt_langevin` by @jla-gardner in #153
* Make sure to move data to CPU before calling vesin by @Luthaf in #156
* Fix virial calculations in `optimizers` and `integrators` by @janosh in #163
* Pad memory estimation by @orionarcher in #160
* Refactor sevennet model by @YutackPark in #172
* `io` optional dependencies in `pyproject.toml` by @curtischong in #185

### 📖 Documentation

* (tiny) add graph-pes to README by @jla-gardner in #149
* Better module fig by @janosh in #168

### 🚀 Performance

* More efficient Orb `state_to_atoms_graph` calculation by @AdeeshKolluru in #165

### 🚧 CI

* Refactor `test_math.py` and `test_transforms.py` by @janosh in #151

### 🏥 Package Health

* Try out hatchling for build vs setuptools by @CompRhys in #177

### 🏷️ Type Hints

* Add `torch-sim/typing.py` by @janosh in #157

### 📦 Dependencies

* Bump `mace-torch` to v0.3.12 by @janosh in #170
* Update metatrain dependency by @Luthaf in #186

## New Contributors

* @Luthaf made their first contribution in #156
* @YutackPark made their first contribution in #172
* @curtischong made their first contribution in #185

**Full Changelog**: https://github.com/torchsim/torch-sim/compare/v0.2.0...v0.2.1

## v0.2.0

### Bug Fixes 🐛

* Fix integrate reporting kwarg to arg error, #113 (raised by @hn-yu)
* Allow runners to take large initial batches, #128 (raised by @YutackPark)
* Add Fairchem model support for PBC, #111 (raised by @ryanliu30)

### Enhancements 🛠

* **breaking** Rename `HotSwappingAutobatcher` to `InFlightAutobatcher` and `ChunkingAutoBatcher` to `BinningAutoBatcher`, #143 @orionarcher
* Support for Orbv3, #140, @AdeeshKolluru
* Support metatensor models, #141, @frostedoyter @Luthaf
* Support for graph-pes models, #118 @jla-gardner
* Support MatterSim and fix ASE cell convention issues, #112 @CompRhys
* Implement positions only FIRE optimization, #139 @abhijeetgangan
* Allow different temperatures in batches, #123 @orionarcher
* FairChem model updates: PBC handling, test on OMat24 e-trained model, #126 @AdeeshKolluru
* FairChem model from_data_list support, #138 @ryanliu30
* New correlation function module, #115 @stefanbringuier

### Documentation 📖

* Improved model documentation, #121 @orionarcher
* Plot of TorchSim module graph in docs, #132 @janosh

### House-Keeping 🧹

* Only install HF for fairchem tests, #134 @CompRhys
* Don't download MBD in CI, #135 @orionarcher
* Tighten graph-pes test bounds, #143 @orionarcher

## v0.1.0

Initial release.
