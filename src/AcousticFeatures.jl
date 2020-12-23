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
name(::FrequencyContours) = ["FrequencyContours"]

"""
In water, the common reference `ref` is 1 micropascal. In air, the
common reference `ref` is 20 micropascal.
"""
struct SoundPressureLevel{T<:Real} <: AbstractAcousticFeature
    ref::T
end
SoundPressureLevel() = SoundPressureLevel(1.0)
name(::SoundPressureLevel) = ["SPL"]

struct ImpulseStats{FT<:Real,T<:Real} <: AbstractAcousticFeature
    fs::FT
    k::Int
    tdist::T
    computeenvelope::Bool
end
ImpulseStats(fs) = ImpulseStats(fs, 10, 1e-3, true)
ImpulseStats(fs, k, tdist) = ImpulseStats(fs, k, tdist, true)
name(::ImpulseStats) = ["Nᵢ", "μᵢᵢ", "varᵢᵢ"] 

struct SymmetricAlphaStableStats <: AbstractAcousticFeature end
name(::SymmetricAlphaStableStats) = ["α", "scale"]

struct Entropy{FT<:Real} <: AbstractAcousticFeature
    fs::FT
    n::Int
    noverlap::Int
end
name(::Entropy) = ["TemporalEntropy","SpectralEntropy","EntropyIndex"]

struct ZeroCrossingRate <: AbstractAcousticFeature end
name(::ZeroCrossingRate) = ["ZCR"]

struct SpectralCentroid{FT<:Real} <: AbstractAcousticFeature
    fs::FT
end
name(::SpectralCentroid) = ["SpectralCentroid"]

struct SpectralFlatness <: AbstractAcousticFeature end
name(::SpectralFlatness) = ["SpectralFlatness"]

struct PermutationEntropy <: AbstractAcousticFeature
    m::Int
    τ::Int
    normalization::Bool
end
PermutationEntropy(m) = PermutationEntropy(m, 1, true)
name(::PermutationEntropy) = ["PermutationEntropy"]

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
    :col, ["FrequencyContours"]
And data, a 1×1 Array{Float64,2}:
 0.0038910505836575876

julia> x = Score(FrequencyContours(9600, 512, 256, 1.0, 1000.0, 99.0, 1000.0, 0.05), randn(9600); winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["FrequencyContours"]
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
    for (i, col) in enumerate(eachcol(p[:, 2:end]))
        col = collect(col)
        # crds,_ = peakprom(Maxima(), col, trunc(Int, f.minfdist/δf); minprom=eps(T)+percentile(col, f.minhprc))
        crds, _ = findpeaks1d(col; height=eps(T)+percentile(col, f.minhprc), distance=trunc(Int, f.minfdist/δf))
        for crd in crds
            if length(ctrs) == 0
                ctrs = [[(crd, 1)] for crd in crds]
            else
                idxselect = Int64[]
                costselect = Float64[]
                for (j, ctr) in enumerate(ctrs)
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

# Examples:
julia> x = Score(SoundPressureLevel(), randn(9600))2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:1
    :col, ["SPL"]
And data, a 1×1 Array{Float64,2}:
 -0.08307636105819256

julia> x = Score(SoundPressureLevel(), randn(9600); winlen=960, noverlap=480)2-dimensional AxisArray{Float64,2,...} with axes:
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
    center = Statistics.median(x)
    height = center+f.k*mad(x, center=center, normalize=true)
    distance = trunc(Int, f.tdist*f.fs)
    # crds, _ = peakprom(Maxima(), x, distance; minprom=height)
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
    :col, ["TemporalEntropy", "SpectralEntropy", "EntropyIndex"]
And data, a 1×3 Array{Float64,2}:
 0.984457  0.997608  0.982103

julia> x = Score(Entropy(9600, 96, 48), randn(9600); winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["TemporalEntropy", "SpectralEntropy", "EntropyIndex"]
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
    [count(!iszero, diff(x .> 0))/length(x)]
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
    :col, ["SpectralCentroid"]
And data, a 1×1 Array{Float64,2}:
 2387.4592177121676

julia> x = Score(SpectralCentroid(9600), randn(9600); winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["SpectralCentroid"]
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
    :col, ["SpectralFlatness"]
And data, a 1×1 Array{Float64,2}:
 0.5598932661540399

julia> x = Score(SpectralFlatness(), randn(9600); winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["SpectralFlatness"]
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
C. Bandt, B. Pompe, "Permutation entropy: a natural complexity measure for time series",
Phys. Rev. Lett., 88 (17), 2002

# Examples:
```julia-repl
julia> x = Score(PermutationEntropy(7), randn(9600))
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:1
    :col, ["PermutationEntropy"]
And data, a 1×1 Array{Float64,2}:
 0.9637270776723836

julia> x = Score(PermutationEntropy(7), randn(9600), winlen=960, noverlap=480)
2-dimensional AxisArray{Float64,2,...} with axes:
    :row, 1:480:9121
    :col, ["PermutationEntropy"]
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
    p = ordinalpatterns(x, f.m, f.τ)
    pe = -sum(p .* log2.(p))
    if f.normalization
        [pe / convert(eltype(pe), log2(factorial(big(f.m))))]
    else
        [pe]
    end
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
               winlen::Int=length(x),
               noverlap::Int=0,
               padtype::Symbol=:fillzeros,
               subseqtype::DataType=Float64,
               preprocess::Function=identity,
               map::Function=map,
               showprogress::Bool=false) where {T<:Real}
    xlen = length(x)
    if winlen < xlen
        (noverlap < 0) && throw(ArgumentError("`noverlap` must be larger or equal to zero."))
        subseqs = Subsequence(x, winlen, noverlap; padtype=padtype)
    elseif winlen == xlen
        stmp = score(f, preprocess(convert.(subseqtype, x)))
        return AxisArray(reshape([stmp...], (1, length(stmp))); row=1:1, col=name(f))
    else
        throw(ArgumentError("`winlen` must be smaller or equal to the length of `x`."))
    end
    if showprogress
        s = @showprogress map(x -> score(f, preprocess(convert.(subseqtype, x))), subseqs)
    else
        s = map(x -> score(f, preprocess(convert.(subseqtype, x))), subseqs)
    end
    AxisArray(mapreduce(transpose, vcat, s); row=1:subseqs.step:xlen, col=name(f))
end

(f::AbstractAcousticFeature)(x; kwargs...) = Score(f, x; kwargs...)

end