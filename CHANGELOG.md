## v0.2.0

### Bug Fixes 🐛
* Fix integrate reporting kwarg to arg error, https://github.com/Radical-AI/torch-sim/issues/113 (raised by @hn-yu)
* Allow runners to take large initial batches, https://github.com/Radical-AI/torch-sim/issues/128 (raised by @YutackPark)
* Add Fairchem model support for PBC, https://github.com/Radical-AI/torch-sim/issues/111 (raised by @ryanliu30)

### Enhancements 🛠
* **[breaking]** Rename `HotSwappingAutobatcher` to `InFlightAutobatcher` and `ChunkingAutoBatcher` to `BinningAutoBatcher`, https://github.com/Radical-AI/torch-sim/pull/143 @orionarcher
* Support for Orbv3, https://github.com/Radical-AI/torch-sim/pull/140, @AdeeshKolluru
* Support metatensor models, https://github.com/Radical-AI/torch-sim/pull/141, @frostedoyter @Luthaf
* Support for graph-pes models, https://github.com/Radical-AI/torch-sim/pull/118 @jla-gardner
* Support MatterSim and fix ASE cell convention issues, https://github.com/Radical-AI/torch-sim/pull/112 @CompRhys
* Implement positions only FIRE optimization, https://github.com/Radical-AI/torch-sim/pull/139 @abhijeetgangan
* Allow different temperatures in batches, https://github.com/Radical-AI/torch-sim/pull/123 @orionarcher
* FairChem model updates: PBC handling, test on OMat24 e-trained model, https://github.com/Radical-AI/torch-sim/pull/126 @AdeeshKolluru
* FairChem model from_data_list support, https://github.com/Radical-AI/torch-sim/pull/138 @ryanliu30
* New correlation function module, https://github.com/Radical-AI/torch-sim/pull/115 @stefanbringuier

### Documentation 📖
* Imoved model documentation, https://github.com/Radical-AI/torch-sim/pull/121 @orionarcher
* Plot of TorchSim module graph in docs, https://github.com/Radical-AI/torch-sim/pull/132 @janosh

### House-Keeping 🧹
* Only install HF for fairchem tests, https://github.com/Radical-AI/torch-sim/pull/134 @CompRhys
* Don't download MBD in CI, https://github.com/Radical-AI/torch-sim/pull/135 @orionarcher
* Tigthen graph-pes test bounds, https://github.com/Radical-AI/torch-sim/pull/143 @orionarcher

## v0.1.0

Initial release.
