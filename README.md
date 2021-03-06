# AcousticFeatures
![CI](https://github.com/ymtoo/AcousticFeatures.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/ymtoo/AcousticFeatures.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ymtoo/AcousticFeatures.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://ymtoo.github.io/AcousticFeatures.jl/dev)

This package implements a set of methods to compute generic acoustic features in [AxisArrays](https://github.com/JuliaArrays/AxisArrays.jl.git) for 1-D time series acoustic data.

The acoustic features are:
- Energy
- [Myriad](https://link.springer.com/article/10.1155/S1110865702000483)
- [Frequency Contours](https://asa.scitation.org/doi/10.1121/1.3531926)
- [Sound Pressure Level](https://en.wikipedia.org/wiki/Sound_pressure#Sound_pressure_level)
- Impulse Statistics (number of impulses, mean and variance of inter-impulse intervals)
- [Alpha Stable Statistics (α and scale)](https://en.wikipedia.org/wiki/Stable_distribution)
- [Entropy (temporal entropy, spectral entropy and entropy index)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0004065)
- [Zero Crossing Rate](https://en.wikipedia.org/wiki/Zero-crossing_rate)
- [Spectral Centroid](https://en.wikipedia.org/wiki/Spectral_centroid)
- [Spectral Flatness](https://en.wikipedia.org/wiki/Spectral_flatness)
- [Permutation Entropy](http://materias.df.uba.ar/mta2019v/files/2019/06/permutation_entropy1.pdf)
- [PSD](https://en.wikipedia.org/wiki/Spectral_density)
- [Acoustic Complexity Index](https://www.sciencedirect.com/science/article/abs/pii/S1470160X10002037)

## Installation
```julia
using Pkg; pkg"add https://github.com/ymtoo/AcousticFeatures.jl.git"
```

## Usage
```julia
using AcousticFeatures, SignalAnalysis, Plots
N  = 100_000
fs = 100_000
v  = randn(Float64, 3*N)
s  = real(chirp(10_000, 30_000, 1.0, fs))
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
n = 512; nv=256; tnorm = 1.0; fd=1000.0; minhprc = 99.0; 
minfdist = 1000.0; mintlen = 0.05; winlen = 10_000; noverlap = 5_000
sc1 = Score(
    FrequencyContours(fs, n, nv, tnorm, fd, minhprc, minfdist, mintlen),
    v,
    winlen = winlen,
    noverlap = noverlap,
)
sc2 = Score(
    FrequencyContours(fs, n, nv, tnorm, fd, minhprc, minfdist, mintlen),
    x,
    winlen = winlen,
    noverlap = noverlap,
)
plot(sc1.axes[1] ./ fs, sc1.data,
     xlabel = "Time (sec)",
     ylabel = "Frequency Contours",
     label  = "without chirp",
     color  = :blue,
     dpi    = 150,
     thickness_scaling = 1.5,
)
plot!(sc2.axes[1] ./ fs, sc2.data,
      xlabel = "Time (sec)",
      ylabel = "Frequency Contours",
      label  = "with chirp",
      color  = :red,
      dpi    = 150,
      thickness_scaling = 1.5,
)
```
![window](frequencycontours.png)
