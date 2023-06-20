module AcousticFeatures

using AlphaStableDistributions
using AxisArrays
using DocStringExtensions
using DSP
using FFTW
using LinearAlgebra
using FindPeaks1D
using SignalAnalysis
using SignalAnalysis: SampledSignal
using Statistics
using StatsBase
using ProgressMeter

export

    # AcousticFeatures
    Energy,
    Myriad,
    FrequencyContours,
    SoundPressureLevel,
    ImpulseStats,
    SymmetricAlphaStableStats,
    Entropy,
    ZeroCrossingRate,
    SpectralCentroid,
    SpectralFlatness,
    PermutationEntropy,
    PSD,
    AcousticComplexityIndex,
    StatisticalComplexity,
    AcousticDiversityIndex,
    Score,

    # utils
    spectrumflatten,
    myriadconstant,
    pressure,
    envelope

include("utils.jl")

abstract type AbstractAcousticFeature end

################################################################################
#
#   AcousticFeature types
#
################################################################################
struct Energy <: AbstractAcousticFeature end
name(::Energy) = ["Energy"]

struct Myriad{T<:Union{Nothing,Real}} <: AbstractAcousticFeature
    sqKscale::T
end
Myriad() = Myriad{Nothing}(nothing)
name(::Myriad) = ["Myriad"]

struct FrequencyContours{T<:Real} <: AbstractAcousticFeature
    n::Int
    nv::Int # overlap
    tnorm::Union{Nothing,T} # time constant for normalization (sec)
    fd::T # frequency difference from one step to the next (Hz)
    minhprc::T
    minfdist::T
    mintlen::T
end
name(::FrequencyContours) = ["Frequency Contours"]

struct SoundPressureLevel{T<:Real} <: AbstractAcousticFeature
    ref::T
end
SoundPressureLevel() = SoundPressureLevel(1.0)
name(::SoundPressureLevel) = ["SPL"]

struct ImpulseStats{K<:Real,T<:Real,TT,H} <: AbstractAcousticFeature
    k::K
    tdist::T
    computeenvelope::Bool
    template::TT
    height::H
end
ImpulseStats(k, tdist) = ImpulseStats(k, tdist, true, nothing, nothing)
ImpulseStats(k, tdist, computeenvelope) = ImpulseStats(k, tdist, computeenvelope, nothing, nothing)
function ImpulseStats(k, tdist, computeenvelope, template)
    if computeenvelope && !isnothing(template)
        env = envelope(template)
        ImpulseStats(k, tdist, computeenvelope, env, nothing)
    else
        ImpulseStats(k, tdist, computeenvelope, template, nothing)
    end
end
name(::ImpulseStats) = ["Nᵢ", "μᵢᵢ", "varᵢᵢ"] 

struct SymmetricAlphaStableStats <: AbstractAcousticFeature end
name(::SymmetricAlphaStableStats) = ["α", "scale"]

struct Entropy <: AbstractAcousticFeature
    n::Int
    noverlap::Int
end
name(::Entropy) = ["Temporal Entropy","Spectral Entropy","Entropy Index"]

struct ZeroCrossingRate <: AbstractAcousticFeature end
name(::ZeroCrossingRate) = ["ZCR"]

struct SpectralCentroid <: AbstractAcousticFeature end
name(::SpectralCentroid) = ["Spectral Centroid"]

struct SpectralFlatness <: AbstractAcousticFeature end
name(::SpectralFlatness) = ["Spectral Flatness"]

struct PermutationEntropy <: AbstractAcousticFeature
    m::Int
    τ::Int
    normalization::Bool
    weighted::Bool
end
PermutationEntropy(m) = PermutationEntropy(m, 1, true, false)
PermutationEntropy(m, τ) = PermutationEntropy(m, τ, true, false)
name(::PermutationEntropy) = ["Permutation Entropy"]

struct PSD <: AbstractAcousticFeature
    n::Int
    noverlap::Int
    fs::Real
end
function name(f::PSD)
    "PSD-" .* string.(round.(FFTW.rfftfreq(f.n, f.fs); digits=1)) .* "Hz"
end

struct AcousticComplexityIndex <: AbstractAcousticFeature
    n::Int
    noverlap::Int
    jbin::Int
    amplitude::Bool
end
AcousticComplexityIndex(n, noverlap, jbin) = AcousticComplexityIndex(n, noverlap, jbin, true)
name(::AcousticComplexityIndex) = ["Acoustic Complexity Index"]

struct StatisticalComplexity <: AbstractAcousticFeature
    m::Int
    τ::Int
end
StatisticalComplexity(m) = StatisticalComplexity(m, 1)
name(::StatisticalComplexity) = ["Statistical Complexity"]

struct AcousticDiversityIndex{T<:Real} <: AbstractAcousticFeature
    n::Int
    noverlap::Int
    freqband_hz::T
    minmaxfreq_hz::Tuple{T,T}
    threshold_db::T
end
AcousticDiversityIndex(n, noverlap, freqband_hz::T, minmaxfreq::Tuple{T,T}) where {T<:Real} = 
    AcousticDiversityIndex(n, noverlap, freqband_hz, minmaxfreq, T(-50))
name(::AcousticDiversityIndex) = ["Acoustic Diversity Index"]

################################################################################
#
#   Implementations
#
################################################################################
"""
$(TYPEDSIGNATURES)
Score of `x` based on mean energy.

# Examples:
```julia-repl
julia> x = Score(Energy(), randn(9600))
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1]
    :feature, ["Energy"]
    :channel, [1]
And data, a 1×1×1 Array{Float64, 3}:
[:, :, 1] =
 1.001273328811565

julia> x = Score(Energy(), randn(9600); winlen=960, noverlap=480)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1, 481, 961, 1441, 1921, 2401, 2881, 3361, 3841, 4321, 4801, 5281, 5761, 6241, 6721, 7201, 7681, 8161, 8641]
    :feature, ["Energy"]
    :channel, [1]
And data, a 19×1×1 Array{Float64, 3}:
[:, :, 1] =
 0.9896401818968652
 ⋮
 0.9602549596080265
```
"""
score(::Energy, x::AbstractVector{T}) where T<:Real = [mean(abs2, x)]

"""
$(TYPEDSIGNATURES)
Score of `x` based on myriad.

# Reference:
Mahmood et. al., "Optimal and Near-Optimal Detection in Bursty Impulsive Noise,"
IEEE Journal of Oceanic Engineering, vol. 42, no. 3, pp. 639--653, 2016.

# Examples:
```julia-repl
julia> x = Score(Myriad(), randn(9600))
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1]
    :feature, ["Myriad"]
    :channel, [1]
And data, a 1×1×1 Array{Float64, 3}:
[:, :, 1] =
 30884.887026311182

julia> x = Score(Myriad(), randn(9600); winlen=960, noverlap=480)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1, 481, 961, 1441, 1921, 2401, 2881, 3361, 3841, 4321, 4801, 5281, 5761, 6241, 6721, 7201, 7681, 8161, 8641]
    :feature, ["Myriad"]
    :channel, [1]
And data, a 19×1×1 Array{Float64, 3}:
[:, :, 1] =
 34705.27101918274
 ⋮
 34616.48633137812
```
"""
function score(f::Myriad{S}, x::AbstractVector{T}) where {T<:Real,S<:Real}
    sqKscale = f.sqKscale
    [sum(x -> log(sqKscale + abs2(x)), x)]
end
score(::Myriad{Nothing}, x) = score(Myriad(myriadconstant(x)), x)

# """
#     Score of `x` based on vector myriad.

#     Reference:
#     Mahmood et. al., "Optimal and Near-Optimal Detection in Bursty Impulsive Noise,"
#     IEEE Journal of Oceanic Engineering, vol. 42, no. 3, pp. 639--653, 2016.
# """
# function score(f::VMyriad{S,M}, x::AbstractVector{T}) where {T<:Real,S<:Real,M<:AbstractMatrix}
#     K², Rₘ = f.K², f.Rₘ
#     m = size(Rₘ, 1)-1
#     N = length(x)
#     N < m && throw(ArgumentError("`m` has to be larger than length of `x`"))
#     mplusonedivtwo = (m+1)/2
#     mplustwodivtwo = (m+2)/2
#     onetom = 1:m
#     invRₘ = inv(cholesky(Rₘ).L)
#     invRₘonetom = inv(cholesky(Rₘ[onetom, onetom]).L)
#     s = mplusonedivtwo*log(K²+norm²(invRₘonetom*x[onetom]))
#     for n in m+1:N
#         s += @views mplustwodivtwo*log(K²+norm²(invRₘ*x[n-m:n])) - mplusonedivtwo*log(K²+norm²(invRₘonetom*x[n-m:n-1]))
#     end
#     s
# end
# score(::VMyriad{Nothing,Nothing}, x) = score(VMyriad(vmyriadconstant(x)...), x)

"""
$(TYPEDSIGNATURES)
Score of `x` based on frequency contours count.

# Reference:
D. Mellinger, R. Morrissey, L. Thomas, J. Yosco, "A method for detecting whistles, moans, and other frequency
contour sounds", 2011 J. Acoust. Soc. Am. 129 4055

# Examples:
```julia-repl
julia> x = Score(FrequencyContours(512, 256, 1.0, 1000.0, 99.0, 1000.0, 0.05), randn(9600); fs=9600)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1]
    :feature, ["Frequency Contours"]
    :channel, [1]
And data, a 1×1×1 Array{Float64, 3}:
[:, :, 1] =
 0.004539559014267186

julia> x = Score(FrequencyContours(512, 256, 1.0, 1000.0, 99.0, 1000.0, 0.05), randn(9600); winlen=960, noverlap=480, fs=9600)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1, 481, 961, 1441, 1921, 2401, 2881, 3361, 3841, 4321, 4801, 5281, 5761, 6241, 6721, 7201, 7681, 8161, 8641]
    :feature, ["Frequency Contours"]
    :channel, [1]
And data, a 19×1×1 Array{Float64, 3}:
[:, :, 1] =
 0.0
 ⋮
 0.0
```
"""
function score(f::FrequencyContours, x::SampledSignal{T}) where T<:Real
    spec = spectrogram(x, f.n, f.nv; fs=framerate(x), window=DSP.hamming)
    p  = spec.power; frequency=spec.freq; t=spec.time
    δt = t[2]-t[1]
    δf = frequency[2]-frequency[1]
    f.tnorm === nothing ? Nnorm = size(p, 2) : Nnorm = f.tnorm÷(δt) |> Int
    p    = spectrumflatten(p, Nnorm) #noise-flattened spectrogram
    crds, _ = findpeaks1d(p[:, 1]; height=eps(T)+percentile(p[:, 1], f.minhprc), distance=trunc(Int, f.minfdist/δf))
    ctrs = [[(crd, 1)] for crd in crds]
    for (i, col) ∈ enumerate(eachcol(p[:, 2:end]))
        col = collect(col)
        crds, _ = findpeaks1d(col; height=eps(T)+percentile(col, f.minhprc), distance=trunc(Int, f.minfdist/δf))
        for crd in crds
            if iszero(length(ctrs))
                ctrs = [[(crd, 1)] for crd in crds]
            else
                idxselect = Int64[]
                costselect = Float64[]
                for (j, ctr) ∈ enumerate(ctrs)
                    if (ctr[end][2] == i-1) && abs(frequency[ctr[end][1]]-frequency[crd]) <= f.fd
                        push!(idxselect, j)
                        push!(costselect, abs(frequency[ctr[end][1]]-frequency[crd]))
                    end
                end
                if isempty(idxselect)
                    push!(ctrs, [(crd, i)])
                else
                    idxopt = idxselect[argmin(costselect)]
                    push!(ctrs[idxopt], (crd, i))
                end
            end
        end
    end
    idxdelete = Int64[]
    for (i, ctr) in enumerate(ctrs)
        (length(ctr)-1)*(δt) < f.mintlen && push!(idxdelete, i)
    end
    deleteat!(ctrs, idxdelete)
    count = isempty(ctrs) ? 0 : sum(length, ctrs)
    [count/length(p)]
end

"""
$(TYPEDSIGNATURES)
Score of `x` based on Sound Pressure Level (SPL). `x` is in micropascal.
In water, the common reference `ref` is 1 micropascal. In air, the
common reference `ref` is 20 micropascal.

# Examples:
```julia-repl
julia> x = Score(SoundPressureLevel(), randn(9600))
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1]
    :feature, ["SPL"]
    :channel, [1]
And data, a 1×1×1 Array{Float64, 3}:
[:, :, 1] =
 -0.023514452445490917

julia> x = Score(SoundPressureLevel(), randn(9600); winlen=960, noverlap=480)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1, 481, 961, 1441, 1921, 2401, 2881, 3361, 3841, 4321, 4801, 5281, 5761, 6241, 6721, 7201, 7681, 8161, 8641]
    :feature, ["SPL"]
    :channel, [1]
And data, a 19×1×1 Array{Float64, 3}:
[:, :, 1] =
  0.1844558527671365
  ⋮
 -0.12184606185171237
```
"""
function score(f::SoundPressureLevel, x::AbstractVector{T}) where T<:Real
    rmsx = sqrt(mean(abs2, x))
    [20*log10(rmsx/f.ref)]
end

"""
$(TYPEDSIGNATURES)
Score of `x` based on number of impulses, mean and variance of inter-impulse intervals.
The minimum height of impulses is defined by `a+k*b` where `a` is median of the envelope
of `x` and `b` is median absolute deviation (MAD) of the envelope of `x`.

# Reference:
Matthew W Legg et al., "Analysis of impulsive biological noise due to snapping shrimp as a
point process in time", 2007.

# Examples:
```julia-repl
julia> x = Score(ImpulseStats(10, 0.01), randn(9600); fs = 9600)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1]
    :feature, ["Nᵢ", "μᵢᵢ", "varᵢᵢ"]
    :channel, [1]
And data, a 1×3×1 Array{Float64, 3}:
[:, :, 1] =
 0.0  NaN  NaN

julia> x = Score(ImpulseStats(10, 0.01), randn(9600); winlen=960, noverlap=480, fs=9600)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1, 481, 961, 1441, 1921, 2401, 2881, 3361, 3841, 4321, 4801, 5281, 5761, 6241, 6721, 7201, 7681, 8161, 8641]
    :feature, ["Nᵢ", "μᵢᵢ", "varᵢᵢ"]
    :channel, [1]
And data, a 19×3×1 Array{Float64, 3}:
[:, :, 1] =
 0.0  NaN  NaN
 ⋮               
 0.0  NaN  NaN
```
"""
function score(f::ImpulseStats, x::SampledSignal{T}) where T<:Real
    if f.computeenvelope
        x = envelope(x)
    end
    if !isnothing(f.template)
        x = normcrosscorr(x, f.template)
    end
    height = if isnothing(f.height)
        center = Statistics.median(filter(!isnan, x)) # median, ignoring NaNs
        center + f.k * mad(filter(!isnan, x), center=center, normalize=true) # mad, ignoring NaNs
    else
        f.height
    end
    distance = trunc(Int, f.tdist * framerate(x))
    crds,_ = findpeaks1d(samples(x); height=height, distance=distance)
    timeintervals = diff(crds)
    [convert(Float64, length(crds)), mean(timeintervals) / framerate(x), var(timeintervals) / framerate(x)]
end

"""
$(TYPEDSIGNATURES)
Score of `x` based on the parameters of Symmetric Alpha Stable Distributions.
The parameter α measures the impulsiveness while the parameter scale measures
the width of the distributions.

# Reference:
https://github.com/org-arl/AlphaStableDistributions.jl

# Examples:
```julia-repl
julia> x = Score(SymmetricAlphaStableStats(), randn(9600))
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1]
    :feature, ["α", "scale"]
    :channel, [1]
And data, a 1×2×1 Array{Float64, 3}:
[:, :, 1] =
 1.97123  0.701582

julia> x = Score(SymmetricAlphaStableStats(), randn(9600); winlen=960, noverlap=480)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1, 481, 961, 1441, 1921, 2401, 2881, 3361, 3841, 4321, 4801, 5281, 5761, 6241, 6721, 7201, 7681, 8161, 8641]
    :feature, ["α", "scale"]
    :channel, [1]
And data, a 19×2×1 Array{Float64, 3}:
[:, :, 1] =
 1.86815  0.66576
 ⋮        
 2.0      0.678246
```
"""
function score(::SymmetricAlphaStableStats, x::AbstractVector{T}) where T<:Real
    d = fit(SymmetricAlphaStable, x)
    [d.α, d.scale]
end

"""
$(TYPEDSIGNATURES)
Score of `x` based on temporal entropy, spectral entropy and entropy index.

# Reference:
J. Sueur, A. Farina, A. Gasc, N. Pieretti, S. Pavoine, Acoustic Indices for Biodiversity
Assessment and Landscape Investigation, 2014.

# Examples:
```julia-repl
julia> x = Score(Entropy(96, 48), randn(9600); fs=9600)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1]
    :feature, ["Temporal Entropy", "Spectral Entropy", "Entropy Index"]
    :channel, [1]
And data, a 1×3×1 Array{Float64, 3}:
[:, :, 1] =
 0.984822  0.998127  0.982977

julia> x = Score(Entropy(96, 48), randn(9600); winlen=960, noverlap=480, fs=9600)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1, 481, 961, 1441, 1921, 2401, 2881, 3361, 3841, 4321, 4801, 5281, 5761, 6241, 6721, 7201, 7681, 8161, 8641]
    :feature, ["Temporal Entropy", "Spectral Entropy", "Entropy Index"]
    :channel, [1]
And data, a 19×3×1 Array{Float64, 3}:
[:, :, 1] =
 0.980493  0.988433  0.969152
 ⋮                   
 0.979868  0.987179  0.967305
```
"""
function score(f::Entropy, x::SampledSignal{T}) where T<:Real
    sp = power(spectrogram(x, f.n, f.noverlap; fs=framerate(x)))
    ne = normalize_envelope(x)
    n = length(ne)
    Ht = -sum(ne .* log2.(ne)) ./ log2(n)
    ns = normalize_spectrum(sp)
    N = length(ns)
    Hf = -sum(ns .* log2.(ns)) ./ log2.(N)
    H = Ht*Hf
    [Ht, Hf, H]
end

"""
$(TYPEDSIGNATURES)
Score of `x` based on zero crossing rate.

# Refernce:
https://en.wikipedia.org/wiki/Zero-crossing_rate

# Examples:
```julia-repl
julia> x = Score(ZeroCrossingRate(), randn(9600))
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1]
    :feature, ["ZCR"]
    :channel, [1]
And data, a 1×1×1 Array{Float64, 3}:
[:, :, 1] =
 0.497239295759975

julia> x = Score(ZeroCrossingRate(), randn(9600); winlen=960, noverlap=480)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1, 481, 961, 1441, 1921, 2401, 2881, 3361, 3841, 4321, 4801, 5281, 5761, 6241, 6721, 7201, 7681, 8161, 8641]
    :feature, ["ZCR"]
    :channel, [1]
And data, a 19×1×1 Array{Float64, 3}:
[:, :, 1] =
 0.516162669447341
 ⋮
 0.470281543274244
```
"""
function score(::ZeroCrossingRate, x::AbstractVector{T}) where T<:Real
    [count(!iszero, diff(x .> zero(T))) / (length(x) - 1)]
end

"""
$(TYPEDSIGNATURES)
Score of `x` based on spectral centroid.

# Reference:
https://en.wikipedia.org/wiki/Spectral_centroid

# Examples:
```julia-repl
julia> x = Score(SpectralCentroid(), randn(9600); fs=9600)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1]
    :feature, ["Spectral Centroid"]
    :channel, [1]
And data, a 1×1×1 Array{Float64, 3}:
[:, :, 1] =
 2406.918897946181

julia> x = Score(SpectralCentroid(), randn(9600); winlen=960, noverlap=480, fs=9600)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1, 481, 961, 1441, 1921, 2401, 2881, 3361, 3841, 4321, 4801, 5281, 5761, 6241, 6721, 7201, 7681, 8161, 8641]
    :feature, ["Spectral Centroid"]
    :channel, [1]
And data, a 19×1×1 Array{Float64, 3}:
[:, :, 1] =
 2470.7611082156754
 ⋮
 2421.4182047609647
```
"""
function score(::SpectralCentroid, x::SampledSignal{T}) where T<:Real
    magnitudes = abs.(rfft(x))
    freqs = FFTW.rfftfreq(length(x), framerate(x))
    [sum(magnitudes .* freqs) / sum(magnitudes)]
end

"""
$(TYPEDSIGNATURES)
Score of `x` based on spectral flatness.

# Reference:
https://en.wikipedia.org/wiki/Spectral_flatness

# Examples:
```julia-repl
julia> x = Score(SpectralFlatness(), randn(9600))
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1]
    :feature, ["Spectral Flatness"]
    :channel, [1]
And data, a 1×1×1 Array{Float64, 3}:
[:, :, 1] =
 0.5703906724982125

julia> x = Score(SpectralFlatness(), randn(9600); winlen=960, noverlap=480)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1, 481, 961, 1441, 1921, 2401, 2881, 3361, 3841, 4321, 4801, 5281, 5761, 6241, 6721, 7201, 7681, 8161, 8641]
    :feature, ["Spectral Flatness"]
    :channel, [1]
And data, a 19×1×1 Array{Float64, 3}:
[:, :, 1] =
 0.5338443405802898
 ⋮
 0.5704666324591952
```
"""
function score(::SpectralFlatness, x::AbstractVector{T}) where T<:Real
    magnitudes² = (abs.(rfft(x))).^2
    [geomean(magnitudes²) / mean(magnitudes²)]
end

"""
$(TYPEDSIGNATURES)
Score of `x` based on permutation entropy.

# Reference:
- C. Bandt, B. Pompe, "Permutation entropy: a natural complexity measure for time series", 
    Phys. Rev. Lett., 88 (17), 2002

- B. Fadlallah, B. Chen, A. Keil, and J. Príncipe, “Weighted-permutation entropy: a complexity 
    measure for time series incorporating amplitude information,” Physical Review E: Statistical, 
    Nonlinear, and Soft Matter Physics, vol. 87, no. 2, Article ID 022911, 2013. 

# Examples:
```julia-repl
julia> x = Score(PermutationEntropy(7), randn(9600))
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1]
    :feature, ["Permutation Entropy"]
    :channel, [1]
And data, a 1×1×1 Array{Float64, 3}:
[:, :, 1] =
 0.964174724342896

julia> x = Score(PermutationEntropy(7), randn(9600), winlen=960, noverlap=480)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1, 481, 961, 1441, 1921, 2401, 2881, 3361, 3841, 4321, 4801, 5281, 5761, 6241, 6721, 7201, 7681, 8161, 8641]
    :feature, ["Permutation Entropy"]
    :channel, [1]
And data, a 19×1×1 Array{Float64, 3}:
[:, :, 1] =
 0.7892209267573311
 ⋮
 0.7887095679636025
```
"""
function score(f::PermutationEntropy, x::AbstractVector{T}) where T<:Real
    p = ordinalpatterns(x, f.m, f.τ, f.weighted)
    pe = -sum(p .* log2.(p))
    if f.normalization
        [pe / convert(eltype(pe), log2(factorial(big(f.m))))]
    else
        [pe]
    end
end

"""
$(TYPEDSIGNATURES)
Score of `x` based on power spectral density in dB scale.

# Examples:
```julia-repl
julia> x = Score(PSD(64, 32, 96000), randn(9600); fs=96000)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1]
    :feature, ["PSD-0.0Hz", "PSD-1500.0Hz", "PSD-3000.0Hz", "PSD-4500.0Hz", "PSD-6000.0Hz", "PSD-7500.0Hz", "PSD-9000.0Hz", "PSD-10500.0Hz", "PSD-12000.0Hz", "PSD-13500.0Hz"  …  "PSD-34500.0Hz", "PSD-36000.0Hz", "PSD-37500.0Hz", "PSD-39000.0Hz", "PSD-40500.0Hz", "PSD-42000.0Hz", "PSD-43500.0Hz", "PSD-45000.0Hz", "PSD-46500.0Hz", "PSD-48000.0Hz"]
    :channel, [1]
And data, a 1×33×1 Array{Float64, 3}:
[:, :, 1] =
 -50.1861  …  -46.536  -49.8058

julia> x = Score(PSD(64, 32, 96000), randn(9600), winlen=960, noverlap=480, fs=96000)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1, 481, 961, 1441, 1921, 2401, 2881, 3361, 3841, 4321, 4801, 5281, 5761, 6241, 6721, 7201, 7681, 8161, 8641]
    :feature, ["PSD-0.0Hz", "PSD-1500.0Hz", "PSD-3000.0Hz", "PSD-4500.0Hz", "PSD-6000.0Hz", "PSD-7500.0Hz", "PSD-9000.0Hz", "PSD-10500.0Hz", "PSD-12000.0Hz", "PSD-13500.0Hz"  …  "PSD-34500.0Hz", "PSD-36000.0Hz", "PSD-37500.0Hz", "PSD-39000.0Hz", "PSD-40500.0Hz", "PSD-42000.0Hz", "PSD-43500.0Hz", "PSD-45000.0Hz", "PSD-46500.0Hz", "PSD-48000.0Hz"]
    :channel, [1]
And data, a 19×33×1 Array{Float64, 3}:
[:, :, 1] =
 -52.0638  -48.6761  …  -49.5882
  ⋮                                                                        
 -47.6196  -48.6223     -49.3313
```
"""
function score(f::PSD, x::SampledSignal{T}) where T<:Real
    p = welch_pgram(x, f.n, f.noverlap; fs=framerate(x))
    pow2db.(power(p))
end

"""
$(TYPEDSIGNATURES)
Score of `x` based on Acoustic Complexity Index. `jbin` is the temporal size of a sub-spectrogram. 

# Reference:
N. Pieretti, A. Farina, D. Morri, "A new methodology to infer the singing activity of an avian community: The Acoustic Complexity Index (ACI)", 2011.

# Examples:
```julia-repl
julia> x = Score(AcousticComplexityIndex(1024, 0, 30), randn(960000); fs = 96000)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1]
    :feature, ["Acoustic Complexity Index"]
    :channel, [1]
And data, a 1×1×1 Array{Float64, 3}:
[:, :, 1] =
 9055.209860502793

 julia> x = Score(AcousticComplexityIndex(1024, 0, 30), randn(960000), winlen=96000, noverlap=48000, fs=96000)
 3-dimensional AxisArray{Float64,3,...} with axes:
     :sample, [1, 48001, 96001, 144001, 192001, 240001, 288001, 336001, 384001, 432001, 480001, 528001, 576001, 624001, 672001, 720001, 768001, 816001, 864001]
     :feature, ["Acoustic Complexity Index"]
     :channel, [1]
 And data, a 19×1×1 Array{Float64, 3}:
 [:, :, 1] =
 874.3303017743206
 ⋮
 875.986826332567
```
"""
function score(f::AcousticComplexityIndex, x::SampledSignal{T}) where T<:Real
    scale_fn = f.amplitude ? x -> sqrt.(x) : identity
    sp = spectrogram(x, f.n, f.noverlap; fs=framerate(x)) |> power |> scale_fn
    n = size(sp, 2)
    starts = range(1, n-f.jbin+1; step=f.jbin)
    aci = zero(T)
    for start ∈ starts
        @views subsp = sp[:,start:start+f.jbin-1]
        aci += sum(sum(abs, diff(subsp; dims=2); dims=2) ./ 
            (sum(subsp; dims=2) .+ eps(T)))
    end
    [aci]
end

"""
$(TYPEDSIGNATURES)
Score of `x` based on statistical complexity.

# Reference:
- Lopez-Ruiz, R., Mancini, H. L., & Calbet, X. (1995). A Statistical Measure of Complexity. 
    Physics Letters A, 209, 321-326.

- A. A. B. Pessa, H. V. Ribeiro, ordpy: A Python package for data analysis with permutation entropy 
    and ordinal network methods, Chaos 31, 063110 (2021).

# Examples:
```julia-repl
julia> x = Score(StatisticalComplexity(7), randn(9600))
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1]
    :feature, ["Statistical Complexity"]
    :channel, [1]
And data, a 1×1×1 Array{Float64, 3}:
[:, :, 1] =
 0.1203920737205638

julia> x = Score(StatisticalComplexity(7), randn(9600); winlen=960, noverlap=480)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1, 481, 961, 1441, 1921, 2401, 2881, 3361, 3841, 4321, 4801, 5281, 5761, 6241, 6721, 7201, 7681, 8161, 8641]
    :feature, ["Statistical Complexity"]
    :channel, [1]
And data, a 19×1×1 Array{Float64, 3}:
[:, :, 1] =
 0.5155574606285804
 ⋮
 0.515770942278282
```
"""
function score(f::StatisticalComplexity, x::SignalAnalysis.SampledSignal{T}) where T<:Real
    p = ordinalpatterns(x, f.m, f.τ, false)
    pe = -sum(p .* log2.(p))
    n = factorial(f.m)
    pe /= convert(eltype(pe), log2(n))    
    
    pᵤ = 1 / n
    a = (pᵤ .+ p) ./ 2
    S₁ = -sum(a .* log.(a)) - (pᵤ ./ 2) .* log.(pᵤ ./ 2) .* (n - length(p))
    S₂ = -sum(p .* log.(p)) / 2
    S₃ = log(n) / 2

    js_div_max = -(((n + 1) / n) * log(n + 1) + log(n) - 2 * log(2 * n)) / 2
    js_div = S₁ - S₂ - S₃

    [pe * js_div / js_div_max]
end

"""
$(TYPEDSIGNATURES)
Score of `x` based on acoustic diversity index.

# Reference
- Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011. A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.

- https://github.com/patriceguyot/Acoustic_Indices

# Examples:
```julia-repl
julia> x = Score(AcousticDiversityIndex(256, 128, 50, (50, 1000)), randn(9600); fs = 2000)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1]
    :feature, ["Acoustic Diversity Index"]
    :channel, [1]
And data, a 1×1×1 Array{Float64, 3}:
[:, :, 1] =
 2.833213240809075

julia> x = Score(AcousticDiversityIndex(256, 128, 50, (50, 1000)), randn(9600); winlen = 960, noverlap = 480, fs = 2000)
3-dimensional AxisArray{Float64,3,...} with axes:
    :sample, [1, 481, 961, 1441, 1921, 2401, 2881, 3361, 3841, 4321, 4801, 5281, 5761, 6241, 6721, 7201, 7681, 8161, 8641]
    :feature, ["Acoustic Diversity Index"]
    :channel, [1]
And data, a 19×1×1 Array{Float64, 3}:
[:, :, 1] =
 2.833213344056216
 ⋮
 2.833213344056216
"""
function score(f::AcousticDiversityIndex, x::SignalAnalysis.SampledSignal{T}) where T<:Real
    minfreq, maxfreq = f.minmaxfreq_hz
    spec = spectrogram(x, f.n, f.noverlap; fs=framerate(x))
    freqs = freq(spec)
    freq_step = freqs[2]
    freq_step > f.freqband_hz && throw(ArgumentError("The frequency band size of one bin` has to be 
        larger than the frequency step of the spectrogram."))
    num_freqsteps = Int(f.freqband_hz ÷ freq_step)
    sp = power(spec)
    sp_db = pow2db.(sp ./ maximum(sp))
    adi_bands = typeof(sp_db)[]
    istart = 1
    while true
        istop = istart + num_freqsteps
        istop > length(freqs) && break
        if (freqs[istart] ≥ minfreq) && (freqs[istop] ≤ maxfreq)
            push!(adi_bands, sp_db[istart:istop,:])
            istart = istop + 1
        else
            istart += 1
        end
    end
    vals = [sum(adi_band .> f.threshold_db) ./ length(adi_band)
            for adi_band ∈ adi_bands]
    filter!(!=(0), vals) # remove zeros
    vals_sum = sum(vals)
    [mapreduce(+, vals) do val
        -val / vals_sum * log(val / vals_sum)
    end]
end    

"""
$(TYPEDSIGNATURES) 
Compute acoustic feature `f` scores of a SampledSignal `x` using sliding windows. 

By default, window length `winlen` is the length of `x`, i.e., the whole signal is used to compute 
a score, and overlapping samples `noverlap` is 0. The `padtype` specifies the form of padding, and
for more information, refer to `ImageFiltering.jl`. The signal is subject to preliminary processing
`preprocess`. Acoustic feature scores of subseqences can be computed through mapping 
`map`. `showprogress` is used to monitor the computations.
"""
function Score(f::AbstractAcousticFeature,
               x::SignalAnalysis.SampledSignal{T};
               winlen::Int = size(x, 1),
               noverlap::Int = zero(Int),
               preprocess::Function = identity,
               showprogress::Bool = false) where {T<:Real}
    (noverlap < 0) && (return throw(ArgumentError("`noverlap` must be larger or equal to zero.")))
    xlen = size(x, 1)
    numsensors = size(x, 2)
    fs = framerate(x)
    stepsize = winlen - noverlap

    winlen > xlen && throw(ArgumentError("`winlen` must be smaller or equal to the length of `x`."))
    prog = Progress(numsensors; enabled = showprogress)
    # s = progress_map(Base.axes(x, 2); progress = prog) do i
    s = map(Base.axes(x, 2)) do i
        next!(prog)
        @views ps = SignalAnalysis.partition(x[:,i], winlen; step = stepsize, flush = false)
        af1 = [convert.(Float64, score(f, preprocess(p))) for p ∈ ps]
        stack(af1; dims = 1)
    end |> stack 

    return AxisArray(s; 
                     sample = collect(1:stepsize:(size(s, 1)*stepsize)), 
                     feature = name(f), 
                     channel = collect(1:numsensors))
end

"""
$(TYPEDSIGNATURES) 
Compute acoustic feature `f` scores of a signal `x` using sliding windows. 

By default, sampling rate `fs` is one, window length `winlen` is the length of `x`, i.e., the whole signal is used to compute 
a score, and overlapping samples `noverlap` is 0. The `padtype` specifies the form of padding, and
for more information, refer to `ImageFiltering.jl`. The signal is subject to preliminary processing
`preprocess`. Acoustic feature scores of subseqences can be computed through mapping 
`map`. `showprogress` is used to monitor the computations.

"""
function Score(f::AbstractAcousticFeature,
               x::AbstractVecOrMat{T};
               fs::Real = one(T),
               winlen::Int = size(x, 1),
               noverlap::Int = zero(Int),
               preprocess::Function = identity,
               showprogress::Bool = false) where {T<:Real}
    Score(f, 
          signal(x, fs); 
          winlen = winlen, 
          noverlap = noverlap, 
          preprocess = preprocess, 
          showprogress = showprogress)
end

score(f::AbstractAcousticFeature, x::AbstractVector, fs) = score(f, signal(x, fs))
(f::AbstractAcousticFeature)(x; kwargs...) = Score(f, x; kwargs...)

end
