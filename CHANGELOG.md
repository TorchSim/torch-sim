## v0.2.0

### Bug Fixes 🐛
* Fix integrate reporting kwarg to arg error, issue https://github.com/Radical-AI/torch-sim/issues/113 (raised by @hn-yu)
* Allow runners to take large initial batches, issue https://github.com/Radical-AI/torch-sim/issues/128 (raised by @YutackPark)
* Add Fairchem model support for PBC, issue https://github.com/Radical-AI/torch-sim/issues/111 (raised by @ryanliu30)

### Enhancements 🛠
* **[breaking]** Rename `HotSwappingAutobatcher` to `InFlightAutobatcher` and `ChunkingAutoBatcher` to `BinningAutoBatcher`, PR https://github.com/Radical-AI/torch-sim/pull/143 @orionarcher
* Support for Orbv3, PR#140, @AdeeshKolluru
* Support metatensor models, PR https://github.com/Radical-AI/torch-sim/pull/141, @frostedoyter @Luthaf
* Support for graph-pes models, PR https://github.com/Radical-AI/torch-sim/pull/118 @jla-gardner and @orionarcher
* Support MatterSim and fix ASE cell convention issues, PR https://github.com/Radical-AI/torch-sim/pull/112
* Implement positions only FIRE optimization, PR https://github.com/Radical-AI/torch-sim/pull/139 @abhijeetgangan
* Allow different temperatures in batches, PR https://github.com/Radical-AI/torch-sim/pull/123
* FairChem model updates: PBC handling, test on OMat24 pre-trained model, PR https://github.com/Radical-AI/torch-sim/pull/126 @AdeeshKolluru and @CompRhys
* FairChem model from_data_list support, PR https://github.com/Radical-AI/torch-sim/pull/138 @ryanliu30
* NewCorrelation function module, PR https://github.com/Radical-AI/torch-sim/pull/115 @stefanbringuier

### Documentation 📖
* Improved model documentation, PR https://github.com/Radical-AI/torch-sim/pull/121 @orionarcher
* Reduced test flakiness, PR https://github.com/Radical-AI/torch-sim/pull/143 @orionarcher
* Plot of TorchSim module graph in docs, PR https://github.com/Radical-AI/torch-sim/pull/132 @janosh

### House-Keeping 🧹
* Only install HF for fairchem tests, PR https://github.com/Radical-AI/torch-sim/pull/134 @CompRhys
* Don't download MBD in CI, PR https://github.com/Radical-AI/torch-sim/pull/135 @orionarcher

## v0.1.0

Initial release.

[contributors]: <> (CONTRIBUTOR SECTION)
[abhijeetgangan]: https://github.com/abhijeetgangan
[orionarcher]: https://github.com/orionarcher
[janosh]: https://github.com/janosh
[AdeeshKolluru]: https://github.com/AdeeshKolluru
[CompRhys]: https://github.com/CompRhys
[jla-gardner]: https://github.com/jla-gardner
[stefanbringuier]: https://github.com/stefanbringuier
[frostedoyter]: https://github.com/frostedoyter
[Luthaf]: https://github.com/Luthaf
[ryanliu30]: https://github.com/ryanliu30
