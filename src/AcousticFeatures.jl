module AcousticFeatures

using AlphaStableDistributions
using AxisArrays
using DSP
using FFTW
using ImageFiltering: BorderArray, Fill, Pad
using LinearAlgebra
using FindPeaks1D
using Statistics
using StatsBase
using ProgressMeter

export

    # AcousticFeatures
    Energy,
    Myriad,
    # VMyriad,
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
    Score,

    # subsequences
    Subsequence,

    # utils
    spectrumflatten,
    myriadconstant,
    vmyriadconstant,
    pressure,
    envelope

include("subsequences.jl")
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

# struct VMyriad{T<:Union{Nothing,Real},M<:Union{Nothing,AbstractMatrix}} <: AbstractAcousticFeature
#     K²::T
#     Rₘ::M
# end
# VMyriad() = VMyriad(nothing, nothing)

struct FrequencyContours{FT<:Real,T<:Real} <: AbstractAcousticFeature
    fs::FT
    n::Int
    nv::Int # overlap
    tnorm::Union{Nothing,T} #time constant for normalization (sec)
    fd::T #frequency difference from one step to the next (Hz)
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

struct ImpulseStats{FT<:Real,K<:Real,T<:Real,TT,H} <: AbstractAcousticFeature
    fs::FT
    k::K
    tdist::T
    computeenvelope::Bool
    template::TT
    height::H
end
ImpulseStats(fs) = ImpulseStats(fs, 10.0, 1e-3, true, nothing, nothing)
ImpulseStats(fs, k, tdist) = ImpulseStats(fs, k, tdist, true, nothing, nothing)
ImpulseStats(fs, k, tdist, computeenvelope) = ImpulseStats(fs, k, tdist, computeenvelope, nothing, nothing)
function ImpulseStats(fs, k, tdist, computeenvelope, template)
    if computeenvelope && !isnothing(template)
        env = envelope(template)
        ImpulseStats(fs, k, tdist, computeenvelope, env, nothing)
    else
        ImpulseStats(fs, k, tdist, computeenvelope, template, nothing)
    end
end
name(::ImpulseStats) = ["Nᵢ", "μᵢᵢ", "varᵢᵢ"] 

struct SymmetricAlphaStableStats <: AbstractAcousticFeature end
name(::SymmetricAlphaStableStats) = ["α", "scale"]

struct Entropy{FT<:Real} <: AbstractAcousticFeature
    fs::FT
    n::Int
    noverlap::Int
end
name(::Entropy) = ["Temporal Entropy","Spectral Entropy","Entropy Index"]

struct ZeroCrossingRate <: AbstractAcousticFeature end
name(::ZeroCrossingRate) = ["ZCR"]

struct SpectralCentroid{FT<:Real} <: AbstractAcousticFeature
    fs::FT
end
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

struct PSD{FT<:Real} <: AbstractAcousticFeature
    fs::FT
    n::Int
    noverlap::Int
end
function name(f::PSD)
    "PSD-" .* string.(round.(FFTW.rfftfreq(f.n, f.fs); digits=1)) .* "Hz"
end

struct AcousticComplexityIndex{FT<:Real} <: AbstractAcousticFeature
    fs::FT
    n::Int
    noverlap::Int
    jbin::Int
end
name(::AcousticComplexityIndex) = ["Acoustic Complexity Index"]

struct StatisticalComplexity <: AbstractAcousticFeature
    m::Int
    τ::Int
end
StatisticalComplexity(m) = StatisticalComplexity(m, 1)
name(::StatisticalComplexity) = ["Statistical Complexity"]

################################################################################
#
#   Implementations
#
################################################################################
"""
    score(::Energy, x::AbstractVector{T}) where {T<:Real}

Score of `x` based on mean energy.

# Examples:
```julia-repl
julia> x = Score(Energy(), randn(9600))
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:1
    :col, ["Energy"]
And data, a 1×1 Array{Float64,2}:
 0.9960607967861373

julia> x = Score(Energy(), randn(9600); winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["Energy"]
And data, a 20×1 Array{Float64,2}:
 0.5280804987356663
 ⋮
 0.9988797206321275
```
"""
score(::Energy, x::AbstractVector{T}) where T<:Real = [mean(abs2, x)]

"""
    Score(f::Myriad{S}, x::AbstractVector{T})

Score of `x` based on myriad.

# Reference:
Mahmood et. al., "Optimal and Near-Optimal Detection in Bursty Impulsive Noise,"
IEEE Journal of Oceanic Engineering, vol. 42, no. 3, pp. 639--653, 2016.

# Examples:
```julia-repl
julia> x = Score(Myriad(), randn(9600))
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:1
    :col, ["Myriad"]
And data, a 1×1 Array{Float64,2}:
 27691.956992339285

julia> x = Score(Myriad(), randn(9600); winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["Myriad"]
And data, a 20×1 Array{Float64,2}:
 -5487.396124602646
  1977.7969182956683
  3216.396756712948
     ⋮
  2651.158251224668
  3246.7097026864853
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
    score(f::FrequencyContours, x::AbstractVector{T})

Score of `x` based on frequency contours count.

# Reference:
D. Mellinger, R. Morrissey, L. Thomas, J. Yosco, "A method for detecting whistles, moans, and other frequency
contour sounds", 2011 J. Acoust. Soc. Am. 129 4055

# Examples:
```julia-repl
julia> x = Score(FrequencyContours(9600, 512, 256, 1.0, 1000.0, 99.0, 1000.0, 0.05), randn(9600))
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:1
    :col, ["Frequency Contours"]
And data, a 1×1 Array{Float64,2}:
 0.0038910505836575876

julia> x = Score(FrequencyContours(9600, 512, 256, 1.0, 1000.0, 99.0, 1000.0, 0.05), randn(9600); winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["Frequency Contours"]
And data, a 20×1 Array{Float64,2}:
 0.0
 0.0
 0.0
 0.0
 ⋮
 0.0
 0.0
 0.0
```
"""
function score(f::FrequencyContours, x::AbstractVector{T}) where T<:Real
    spec = spectrogram(x, f.n, f.nv; fs=f.fs, window=DSP.hamming)
    p  = spec.power; frequency=spec.freq; t=spec.time
    δt = t[2]-t[1]
    δf = frequency[2]-frequency[1]
    f.tnorm === nothing ? Nnorm = size(p, 2) : Nnorm = f.tnorm÷(δt) |> Int
    p    = spectrumflatten(p, Nnorm) #noise-flattened spectrogram
    # crds, _ = peakprom(Maxima(), p[:, 1], trunc(Int, f.minfdist÷δf); minprom=eps(T)+percentile(p[:, 1], f.minhprc))
    crds, _ = findpeaks1d(p[:, 1]; height=eps(T)+percentile(p[:, 1], f.minhprc), distance=trunc(Int, f.minfdist/δf))
    ctrs = [[(crd, 1)] for crd in crds]
    for (i, col) ∈ enumerate(eachcol(p[:, 2:end]))
        col = collect(col)
        # crds,_ = peakprom(Maxima(), col, trunc(Int, f.minfdist/δf); minprom=eps(T)+percentile(col, f.minhprc))
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
    score(f::SoundPressureLevel, x::AbstractVector{T})

Score of `x` based on Sound Pressure Level (SPL). `x` is in micropascal.
In water, the common reference `ref` is 1 micropascal. In air, the
common reference `ref` is 20 micropascal.

# Examples:
```julia-repl
julia> x = Score(SoundPressureLevel(), randn(9600))
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:1
    :col, ["SPL"]
And data, a 1×1 Array{Float64,2}:
 -0.08307636105819256

julia> x = Score(SoundPressureLevel(), randn(9600); winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["SPL"]
And data, a 20×1 Array{Float64,2}:
 -2.369874304880999
  0.2795371978218069
  ⋮
 -0.04985476533237352
  0.11412307503113574
```
"""
function score(f::SoundPressureLevel, x::AbstractVector{T}) where T<:Real
    rmsx = sqrt(mean(abs2, x))
    [20*log10(rmsx/f.ref)]
end

"""
    score(f::ImpulseStats, x::AbstractVector{T})

Score of `x` based on number of impulses, mean and variance of inter-impulse intervals.
The minimum height of impulses is defined by `a+k*b` where `a` is median of the envelope
of `x` and `b` is median absolute deviation (MAD) of the envelope of `x`.

# Reference:
Matthew W Legg et al., "Analysis of impulsive biological noise due to snapping shrimp as a
point process in time", 2007.

# Examples:
```julia-repl
julia> x = Score(ImpulseStats(9600, 10, 0.01), randn(9600))
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:1
    :col, ["Nᵢ", "μᵢᵢ", "varᵢᵢ"]
And data, a 1×3 Array{Float64,2}:
 0.0  0.0  0.0

julia> x = Score(ImpulseStats(9600, 10, 0.01), randn(9600); winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["Nᵢ", "μᵢᵢ", "varᵢᵢ"]
And data, a 20×3 Array{Float64,2}:
 4.0  0.0140972  0.0621181
 0.0  0.0        0.0
 0.0  0.0        0.0
 0.0  0.0        0.0
 ⋮               
 0.0  0.0        0.0
 0.0  0.0        0.0
 0.0  0.0        0.0
```
"""
function score(f::ImpulseStats, x::AbstractVector{T}) where T<:Real
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
    distance = trunc(Int, f.tdist*f.fs)
    crds,_ = findpeaks1d(x; height=height, distance=distance)
    timeintervals = diff(crds)
    [convert(Float64, length(crds)), mean(timeintervals)/f.fs, var(timeintervals)/f.fs]
end

"""
    score(::SymmetricAlphaStableStats, x::AbstractVector{T})

Score of `x` based on the parameters of Symmetric Alpha Stable Distributions.
The parameter α measures the impulsiveness while the parameter scale measures
the width of the distributions.

# Reference:
https://github.com/org-arl/AlphaStableDistributions.jl

# Examples:
```julia-repl
julia> x = Score(SymmetricAlphaStableStats(), randn(9600))
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:1
    :col, ["α", "scale"]
And data, a 1×2 Array{Float64,2}:
 2.0  0.714388

julia> x = Score(SymmetricAlphaStableStats(), randn(9600); winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["α", "scale"]
And data, a 20×2 Array{Float64,2}:
 0.5      0.0
 1.90067  0.663918
 1.83559  0.614218
 ⋮        
 1.80072  0.676852
 1.94506  0.677581
```
"""
function score(::SymmetricAlphaStableStats, x::AbstractVector{T}) where T<:Real
    d = fit(SymmetricAlphaStable, x)
    [d.α, d.scale]
end

"""
    score(f::Entropy, x::AbstractVector{T})

Score of `x` based on temporal entropy, spectral entropy and entropy index.

# Reference:
J. Sueur, A. Farina, A. Gasc, N. Pieretti, S. Pavoine, Acoustic Indices for Biodiversity
Assessment and Landscape Investigation, 2014.

# Examples:
```julia-repl
julia> x = Score(Entropy(9600, 96, 48), randn(9600))
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:1
    :col, ["Temporal Entropy", "Spectral Entropy", "Entropy Index"]
And data, a 1×3 Array{Float64,2}:
 0.984457  0.997608  0.982103

julia> x = Score(Entropy(9600, 96, 48), randn(9600); winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["Temporal Entropy", "Spectral Entropy", "Entropy Index"]
And data, a 20×3 Array{Float64,2}:
 0.903053  0.982299  0.887068
 0.980151  0.986018  0.966446
 0.981492  0.984845  0.966618
 0.980283  0.986635  0.967182
 0.978714  0.987383  0.966366
 ⋮                   
 0.97895   0.986322  0.96556
 0.980274  0.983338  0.963941
 0.980822  0.99296   0.973917
 0.979092  0.989817  0.969122
```
"""
function score(f::Entropy, x::AbstractVector{T}) where T<:Real
    sp = spectrogram(x, f.n, f.noverlap; fs=f.fs).power
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
    score(::ZeroCrossingRate, x::AbstractVector{T})

Score of `x` based on zero crossing rate.

# Refernce:
https://en.wikipedia.org/wiki/Zero-crossing_rate

# Examples:
```julia-repl
julia> x = Score(ZeroCrossingRate(), randn(9600))
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:1
    :col, ["ZCR"]
And data, a 1×1 Array{Float64,2}:
 0.5027083333333333

julia> x = Score(ZeroCrossingRate(), randn(9600); winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["ZCR"]
And data, a 20×1 Array{Float64,2}:
 0.2375
 0.46979166666666666
 0.4708333333333333
 0.49583333333333335
 0.5072916666666667
 ⋮
 0.5145833333333333
 0.49375
 0.5052083333333334
 0.5125
 0.4947916666666667
```
"""
function score(::ZeroCrossingRate, x::AbstractVector{T}) where T<:Real
    [count(!iszero, diff(x .> zero(T))) / (length(x) - 1)]
end

"""
    score(f::SpectralCentroid, x::AbstractVector{T})

Score of `x` based on spectral centroid.

# Reference:
https://en.wikipedia.org/wiki/Spectral_centroid

# Examples:
```julia-repl
julia> x = Score(SpectralCentroid(9600), randn(9600))
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:1
    :col, ["Spectral Centroid"]
And data, a 1×1 Array{Float64,2}:
 2387.4592177121676

julia> x = Score(SpectralCentroid(9600), randn(9600); winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["Spectral Centroid"]
And data, a 20×1 Array{Float64,2}:
 2398.6889311658415
 2362.570125358973
 2342.6919660952803
 2375.5903954079977
    ⋮
 2415.431353476017
 2453.7105902333437
 2449.222535628719
 2380.319011105224
```
"""
function score(f::SpectralCentroid, x::AbstractVector{T}) where T<:Real
    magnitudes = abs.(rfft(x))
    freqs = FFTW.rfftfreq(length(x), f.fs)
    [sum(magnitudes .* freqs) / sum(magnitudes)]
end

"""
    score(::SpectralFlatness, x::AbstractVector{T})

Score of `x` based on spectral flatness.

# Reference:
https://en.wikipedia.org/wiki/Spectral_flatness

# Examples:
```julia-repl
julia> x = Score(SpectralFlatness(), randn(9600))
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:1
    :col, ["Spectral Flatness"]
And data, a 1×1 Array{Float64,2}:
 0.5598932661540399

julia> x = Score(SpectralFlatness(), randn(9600); winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["Spectral Flatness"]
And data, a 20×1 Array{Float64,2}:
 0.5661636483227057
 0.543740942647357
 0.5854629162797854
 0.532148471407988
 ⋮
 0.5451002001200566
 0.5496170417608265
 0.537335859768483
 0.535056705285523
```
"""
function score(::SpectralFlatness, x::AbstractVector{T}) where T<:Real
    magnitudes² = (abs.(rfft(x))).^2
    [geomean(magnitudes²) / mean(magnitudes²)]
end

"""
    score(f::PermutationEntropy, x::AbstractVector{T})

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
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:1
    :col, ["Permutation Entropy"]
And data, a 1×1 Array{Float64,2}:
 0.9637270776723836

julia> x = Score(PermutationEntropy(7), randn(9600), winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["Permutation Entropy"]
And data, a 20×1 Array{Float64,2}:
 0.4432867336969194
 0.7896679491573086
 0.7914368148634888
 0.790455877419772
 ⋮
 0.7894106035823039
 0.7886452315698513
 0.7897322855510599
 0.7884747786386084
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
    score(f::PSD, x::AbstractVector{T})

Score of `x` based on power spectral density in dB scale.

# Examples:
```julia-repl
julia> x = Score(PSD(96000, 64, 32), randn(9600))
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:1
    :col, ["PSD-0Hz", "PSD-1500Hz", "PSD-3000Hz", "PSD-4500Hz", "PSD-6000Hz", "PSD-7500Hz", "PSD-9000Hz", "PSD-10500Hz", "PSD-12000Hz", "PSD-13500Hz"  …  "PSD-34500Hz", "PSD-36000Hz", "PSD-37500Hz", "PSD-39000Hz", "PSD-40500Hz", "PSD-42000Hz", "PSD-43500Hz", "PSD-45000Hz", "PSD-46500Hz", "PSD-48000Hz"]
And data, a 1×33 Array{Float64,2}:
 -49.611  -47.1275  -46.7286  -46.4742  -46.6452  …  -47.0801  -47.2065  -46.5577  -46.4154  -50.4786

julia> x = Score(PSD(96000, 64, 32), randn(9600), winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["PSD-0Hz", "PSD-1500Hz", "PSD-3000Hz", "PSD-4500Hz", "PSD-6000Hz", "PSD-7500Hz", "PSD-9000Hz", "PSD-10500Hz", "PSD-12000Hz", "PSD-13500Hz"  …  "PSD-34500Hz", "PSD-36000Hz", "PSD-37500Hz", "PSD-39000Hz", "PSD-40500Hz", "PSD-42000Hz", "PSD-43500Hz", "PSD-45000Hz", "PSD-46500Hz", "PSD-48000Hz"]
And data, a 20×33 Array{Float64,2}:
 -54.3507  -49.2203  -47.9715  -50.4869  -51.4884  …  -48.8581  -49.8427  -50.804   -47.5865  -51.502
 -48.4897  -45.3653  -46.2605  -47.4234  -47.4109     -46.3487  -46.3496  -48.243   -45.3489  -49.5331
 -48.0083  -45.1101  -45.8474  -47.284   -45.2843     -46.2456  -46.5622  -47.2382  -46.6219  -47.9124
 -49.023   -45.4548  -44.5266  -45.9787  -44.7397     -47.6285  -48.4443  -47.2613  -47.7996  -48.1423
   ⋮                                               ⋱                        ⋮                 
 -49.9071  -46.6817  -47.1582  -45.9655  -48.3396     -46.986   -46.8983  -45.6008  -47.0211  -48.4817
 -49.0467  -47.1668  -46.9087  -47.0215  -47.8279     -46.8043  -47.2044  -45.6053  -47.0023  -48.222
 -49.7118  -47.3381  -47.219   -45.3647  -45.6587     -47.3541  -47.4126  -46.1465  -46.491   -48.1833
```
"""
function score(f::PSD, x::AbstractVector{T}) where T<:Real
    p = welch_pgram(x, f.n, f.noverlap; fs=f.fs)
    pow2db.(power(p))
end

"""
    score(f::AcousticComplexityIndex, x::AbstractVector{T})

Score of `x` based on Acoustic Complexity Index. `jbin` is the temporal size of a sub-spectrogram. 

# Reference:
N. Pieretti, A. Farina, D. Morri, "A new methodology to infer the singing activity of an avian community: The Acoustic Complexity Index (ACI)", 2011.

# Examples:
```julia-repl
julia> x = Score(AcousticComplexityIndex(96000, 1024, 0, 30), randn(960000))
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:1
    :col, ["Acoustic Complexity Index"]
And data, a 1×1 Matrix{Float64}:
 15394.052148047322

julia> x = Score(AcousticComplexityIndex(96000, 1024, 0, 30), randn(960000), winlen=96000, noverlap=48000)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:48000:912001
    :col, ["Acoustic Complexity Index"]
And data, a 20×1 Matrix{Float64}:
 998.0694761443425
 1493.1632775077805
    ⋮
 1496.4075980628913
 1486.5057244355821
```
"""
function score(f::AcousticComplexityIndex, x::AbstractVector{T}) where T<:Real
    sp = spectrogram(x, f.n, f.noverlap; fs=f.fs).power
    m, n = size(sp)
    starts = range(1, n-f.jbin+1; step=f.jbin)
    nstarts = length(starts)
    acitmp = Matrix{T}(undef, m, nstarts)
    for i ∈ 1:nstarts
        subsp = sp[:,starts[i]:starts[i]+f.jbin-1]
        D = sum(abs.(diff(subsp; dims=2)); dims=2)
        acitmp[:,i] = D ./ (sum(subsp; dims=2) .+ eps(T))
    end
    [sum(acitmp)]
end

"""
    score(f::StatisticalComplexity, x::AbstractVector{T})

Score of `x` based on statistical complexity.

# Reference:
- Lopez-Ruiz, R., Mancini, H. L., & Calbet, X. (1995). A Statistical Measure of Complexity. 
    Physics Letters A, 209, 321-326.

- A. A. B. Pessa, H. V. Ribeiro, ordpy: A Python package for data analysis with permutation entropy 
    and ordinal network methods, Chaos 31, 063110 (2021).

# Examples:
```julia-repl
julia> x = Score(StatisticalComplexity(7), randn(9600))
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, [1]
    :col, ["Statistical Complexity"]
And data, a 1×1 Matrix{Float64}:
 0.1204028612063487

julia> x = Score(StatisticalComplexity(7), randn(9600); winlen=960, noverlap=480)
 2-dimensional AxisArray{Float64,2,...} with axes:
     :row, [1, 481, 961, 1441, 1921, 2401, 2881, 3361, 3841, 4321, 4801, 5281, 5761, 6241, 6721, 7201, 7681, 8161, 8641, 9121]
     :col, ["Statistical Complexity"]
 And data, a 20×1 Matrix{Float64}:
  0.36175533549782896
  0.51246451919022
  0.5132583800072719
  0.5130145699228823
  ⋮
  0.5144948633431746
  0.5126295900830187
  0.5133615455161591
```
"""
function score(f::StatisticalComplexity, x::AbstractVector{T}) where T<:Real
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
    Score(f, 
          x; 
          winlen=length(x), 
          noverlap=0, 
          padtype=:fillzeros, 
          subseqtype=Float64,
          preprocess=identity,
          map=map,
          showprogress=true)

Compute acoustic feature `f` scores of a time series signal `x` using sliding windows. 

By default, window length `winlen` is the length of `x`, i.e., the whole signal is used to compute 
a score, and overlapping samples `noverlap` is 0. The `padtype` specifies the form of padding, and
for more information, refer to `ImageFiltering.jl`. The signal is subject to preliminary processing
`preprocess`. Acoustic feature scores of subseqences can be computed through mapping 
`map`. `showprogress` is used to monitor the computations.
"""
function Score(f::AbstractAcousticFeature,
               x::AbstractVector{T};
               winlen::Int = length(x),
               noverlap::Int = 0,
               padtype::Symbol = :fillzeros,
               subseqtype::DataType = Float64,
               preprocess::Function = identity,
               map::Function = map,
               showprogress::Bool = false) where {T<:Real}
    (noverlap < 0) && (return throw(ArgumentError("`noverlap` must be larger or equal to zero.")))
    xlen = length(x)
    if winlen < xlen
        subseq = Subsequence(x, winlen, noverlap; padtype=padtype)
        if showprogress
            s = (@showprogress map(x -> score(f, preprocess(convert.(subseqtype, x))), subseq))
        else
            s = map(x -> score(f, preprocess(convert.(subseqtype, x))), subseq)
        end
        return AxisArray(vcat(reshape.(s, 1, :)...)::Matrix{subseqtype}; 
                         row=collect(1:step(subseq):xlen), 
                         col=name(f))
    elseif winlen == xlen
        stmp = score(f, preprocess(convert.(subseqtype, x)))
        return AxisArray(reshape(stmp, (1, length(stmp))); row=ones(Int,1), col=name(f))
    else
        return throw(ArgumentError("`winlen` must be smaller or equal to the length of `x`."))
    end
end

(f::AbstractAcousticFeature)(x; kwargs...) = Score(f, x; kwargs...)

end