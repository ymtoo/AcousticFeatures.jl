# AcousticFeatures
[![Build Status](https://travis-ci.org/ymtoo/AcousticFeatures.jl.svg?branch=master)](https://travis-ci.org/ymtoo/AcousticFeatures.jl)
[![codecov](https://codecov.io/gh/ymtoo/AcousticFeatures.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ymtoo/AcousticFeatures.jl)

This package implements a set of generic acoustic features for time series acoustic data.

The acoustic features are:
- Energy
- Myriad
- Vector Myriad
- Frequency Contours
- Sound Pressure Level
- Impulse Statistics (number of impulses, mean and variance of inter-impulse intervals)
- Alpha Stable Statistics (α and scale)
- Entropy (temporal entropy, spectral entropy and entropy index)
- [Zero Crossing Rate](https://en.wikipedia.org/wiki/Zero-crossing_rate)
- [Spectral Centroid](https://en.wikipedia.org/wiki/Spectral_centroid)
- [SpectralFlatness](https://en.wikipedia.org/wiki/Spectral_flatness)
- Sum of Absolute Autocorrelation,
- [Permutation Entropy](http://materias.df.uba.ar/mta2019v/files/2019/06/permutation_entropy1.pdf)

## Installation
```julia
using Pkg; pkg"add https://github.com/ymtoo/AcousticFeatures.jl.git"
```

## Usage
```julia
using AcousticFeatures, Plots
N  = 100_000
fs = 100_000
v  = randn(Float64, 3*N)
s  = chirp(10_000, 30_000, 1.0, fs)
x  = copy(v); x[N:2*N-1] += s
plot((1:3*N) / fs, x,
    xlabel = "Time (sec)",
    ylabel = "Pressure (uncalibrated)",
    legend = false,
    dpi    = 150,
    thickness_scaling = 1.5,
)
```
![window](timeseries.png)
```julia
n = 512; tnorm = 1.0; fd=1000.0; minhprc = 99.0; minfdist = 1000.0
mintlen = 0.05; winlen = 10_000; noverlap = 5_000
sc1 = Score(
    FrequencyContours(fs, n, tnorm, fd, minhprc, minfdist, mintlen),
    v,
    winlen = winlen,
    noverlap = noverlap,
)
sc2 = Score(
    FrequencyContours(fs, n, tnorm, fd, minhprc, minfdist, mintlen),
    x,
    winlen = winlen,
    noverlap = noverlap,
)
plot( sc1.indices / fs, sc1.s,
    xlabel = "Time (sec)",
    ylabel = "Frequency Contours",
    label  = "without chirp",
    color  = :blue,
    dpi    = 150,
    thickness_scaling = 1.5,
)
plot!( sc2.indices / fs, sc2.s,
    xlabel = "Time (sec)",
    ylabel = "Frequency Contours",
    label  = "with chirp",
    color  = :red,
    dpi    = 150,
    thickness_scaling = 1.5,
)
```
![window](frequencycontours.png)
