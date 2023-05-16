# AcousticFeatures Release Notes

## v0.1.13
* Replaced `mapslices` and `vcat` with `stack` for type stability
* Bumped compat for Julia to 1.9

## v0.1.12
* Implemented multi-channel data and sampled signals supports. `Subsequence` is replaced by `SignalAnalysis.partition`. The modifications result in the following breaking changes:
  * The output is a 3D Axis Array with dimensions of (number of sample indices, number of features, number of channels).
  * Sampling rate `fs` is not required in constructing AbstractAcousticFeature objects except `PSD`.
  * `Subsequence` is removed from the package.