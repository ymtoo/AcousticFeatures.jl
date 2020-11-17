module AcousticFeatures

using AlphaStableDistributions
using DSP
using FFTW
using ImageFiltering: BorderArray, Fill, Pad
using Peaks
using LinearAlgebra
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
    nvelope

include("subsequences.jl")
include("utils.jl")

abstract type AbstractAcousticFeature end

################################################################################
#
#   AcousticFeature types
#
################################################################################
struct Energy <: AbstractAcousticFeature end

struct Myriad{T<:Union{Nothing,Real}} <: AbstractAcousticFeature
    sqKscale::T
end
Myriad() = Myriad{Nothing}(nothing)

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

struct SoundPressureLevel{T<:Real} <: AbstractAcousticFeature
    ref::T
end
SoundPressureLevel() = SoundPressureLevel(1.0)

struct ImpulseStats{FT<:Real,T<:Real} <: AbstractAcousticFeature
    fs::FT
    k::Int
    tdist::T
    computeenvelope::Bool
end
ImpulseStats(fs) = ImpulseStats(fs, 10, 1e-3, true)
ImpulseStats(fs, k, tdist) = ImpulseStats(fs, k, tdist, true)

struct SymmetricAlphaStableStats <: AbstractAcousticFeature end

struct Entropy{FT<:Real} <: AbstractAcousticFeature
    n::Int
    noverlap::Int
    fs::FT
    isspectrumflatten::Bool
end
Entropy(n, noverlap, fs) = Entropy(n, noverlap, fs, true)

struct ZeroCrossingRate <: AbstractAcousticFeature end

struct SpectralCentroid{FT<:Real} <: AbstractAcousticFeature
    fs::FT
end

struct SpectralFlatness <: AbstractAcousticFeature
end

# struct SumAbsAutocor <: AbstractAcousticFeature
#     demean::Bool
# end
# SumAbsAutocor() = SumAbsAutocor(true)

struct PermutationEntropy <: AbstractAcousticFeature
    m::Integer
    τ::Integer
    normalization::Bool
end
PermutationEntropy(m) = PermutationEntropy(m, 1, true)

mutable struct Score{VT1<:AbstractArray{<:Real},VT2<:AbstractRange{Int}}
    s::VT1
    indices::VT2
end
################################################################################
#
#   Implementations
#
################################################################################
"""
    Score of `x` based on mean energy.
"""
score(::Energy, x::AbstractVector{T}) where T<:Real = mean(abs2, x)

"""
    Score of `x` based on myriad.

    Reference:
    Mahmood et. al., "Optimal and Near-Optimal Detection in Bursty Impulsive Noise,"
    IEEE Journal of Oceanic Engineering, vol. 42, no. 3, pp. 639--653, 2016.
"""
function score(f::Myriad{S}, x::AbstractVector{T}) where {T<:Real, S<:Real}
    sqKscale = f.sqKscale
    sum(x -> log(sqKscale + abs2(x)), x)
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
    Score of `x` based on frequency contours count.

    Reference:
    D. Mellinger, R. Morrissey, L. Thomas, J. Yosco, "A method for detecting whistles, moans, and other frequency
    contour sounds", 2011 J. Acoust. Soc. Am. 129 4055
"""
function score(f::FrequencyContours, x::AbstractVector{T}) where T<:Real
    spec = spectrogram(x, f.n, f.nv; fs=f.fs, window=DSP.hamming)
    p  = spec.power; frequency=spec.freq; t=spec.time
    δt = t[2]-t[1]
    δf = frequency[2]-frequency[1]
    f.tnorm === nothing ? Nnorm = size(p, 2) : Nnorm = f.tnorm÷(δt) |> Int
    p    = spectrumflatten(p, Nnorm) #noise-flattened spectrogram
    crds, _ = peakprom(Maxima(), p[:, 1], trunc(Int, f.minfdist÷δf); minprom=eps(T)+percentile(p[:, 1], f.minhprc))
    # crds, _ = findpeaks1d(p[:, 1]; height=eps(T)+percentile(p[:, 1], f.minhprc), distance=trunc(Int, f.minfdist/δf))
    ctrs = [[(crd, 1)] for crd in crds]
    for (i, col) in enumerate(eachcol(p[:, 2:end]))
        col = collect(col)
        crds,_ = peakprom(Maxima(), col, trunc(Int, f.minfdist/δf); minprom=eps(T)+percentile(col, f.minhprc))
        # crds, _ = findpeaks1d(col; height=eps(T)+percentile(col, f.minhprc), distance=trunc(Int, f.minfdist/δf))
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
    count/length(p)
end

"""
Score of `x` based on Sound Pressure Level (SPL). `x` is in micropascal. In water, the common reference is 1 micropascal. In air, the common reference is 20 micropascal.
"""
function score(f::SoundPressureLevel, x::AbstractVector{T}) where T<:Real
    rmsx = sqrt(mean(abs2, x))
    20*log10(rmsx/f.ref)
end

"""
Score of `x` based on number of impulses, mean and variance of inter-impulse intervals. The minimum height of impulses is defined by `a+k*b` where `a` is median of the envelope of `x` and `b` is median absolute deviation (MAD) of the envelope of `x`.
"""
function score(f::ImpulseStats, x::AbstractVector{T}) where T<:Real
    if f.computeenvelope
        x = envelope(x)
    end
    center = Statistics.median(x)
    height = center+f.k*mad(x, center=center, normalize=false)
    distance = trunc(Int, f.tdist*f.fs)
    crds, _ = peakprom(Maxima(), x, distance; minprom=height)
    # crds,_ = findpeaks1d(x; height=height, distance=distance)
    timeintervals = diff(crds)
    [length(crds) mean(timeintervals)/f.fs var(timeintervals)/f.fs]
end

"""
Score of `x` based on the parameters of Symmetric Alpha Stable Distributions. The parameter α measures the impulsiveness while the parameter scale measures the width of the distributions.
"""
function score(::SymmetricAlphaStableStats, x::AbstractVector{T}) where T<:Real
    d = fit(SymmetricAlphaStable, x)
    [d.α d.scale]
end

"""
Score of `x` based on temporal entropy, spectral entropy and entropy index.

Reference:
J. Sueur, A. Farina, A. Gasc, N. Pieretti, S. Pavoine, Acoustic Indices for Biodiversity Assessment and Landscape Investigation, 2014.
"""
function score(f::Entropy, x::AbstractVector{T}) where T<:Real
    sp = spectrogram(x, f.n, f.noverlap; fs=f.fs).power
    f.isspectrumflatten && (sp = spectrumflatten(sp, size(sp, 2)))
    ne = normalize_envelope(x)
    n = length(ne)
    Ht = -sum(ne .* log2.(ne)) ./ log2(n)
    ns = normalize_spectrum(sp)
    N = length(ns)
    Hf = -sum(ns .* log2.(ns)) ./ log2.(N)
    H = Ht*Hf
    [Ht Hf H]
end

"""
Score of `x` based on zero crossing rate.

https://en.wikipedia.org/wiki/Zero-crossing_rate
"""
function score(::ZeroCrossingRate, x::AbstractVector{T}) where T<:Real
    count(!iszero, diff(x .> 0))/length(x)
end

"""
Score of `x` based on spectral centroid.

https://en.wikipedia.org/wiki/Spectral_centroid
"""
function score(f::SpectralCentroid, x::AbstractVector{T}) where T<:Real
    magnitudes = abs.(rfft(x))
    freqs = FFTW.rfftfreq(length(x), f.fs)
    sum(magnitudes .* freqs) / sum(magnitudes)
end

"""
Score of `x` based on spectral flatness.

https://en.wikipedia.org/wiki/Spectral_flatness
"""
function score(::SpectralFlatness, x::AbstractVector{T}) where T<:Real
    magnitudes² = (abs.(rfft(x))).^2
    geomean(magnitudes²) / mean(magnitudes²)
end

# """
# Score of `x` based on sum of absolute autocorrelation. 
# """
# function score(f::SumAbsAutocor, x::AbstractVector{T}) where T<:Real
# #    ac = autocor(x, 0:length(x)-1; demean=f.demean)
#     if f.demean
#         x .-= mean(x)
#     end
#     actmp = xcorr(x, x)
#     ac = actmp[length(x):end] / actmp[length(x)]
#     sum(abs, ac)
# end

"""
Score of `x` based on permutation entropy.

C. Bandt, B. Pompe, "Permutation entropy: a natural complexity measure for time series",
Phys. Rev. Lett., 88 (17), 2002
"""
function score(f::PermutationEntropy, x::AbstractVector{T}) where T<:Real
    p = ordinalpatterns(x, f.m, f.τ)
    pe = -sum(p .* log2.(p))
    if f.normalization
        pe / convert(eltype(pe), log2(factorial(big(f.m))))
    else
        pe
    end
end

"""
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
               preprocess::Function=x->x,
               map::Function=map,
               showprogress::Bool=true) where {T<:Real}
    xlen = length(x)
    if winlen < xlen
        (noverlap < 0) && throw(ArgumentError("`noverlap` must be larger or equal to zero."))
        subseqs = Subsequence(x, winlen, noverlap; padtype=padtype)
#        sc = Score(zeros(outputeltype(f), length(subseqs), outputndims(f)), 1:subseqs.step:xlen)
    elseif winlen == xlen
        stmp = score(f, preprocess(convert.(subseqtype, x)))
        if stmp isa Number
            return Score(reshape([stmp], (1, 1)), 1:1)
        else
            return Score(stmp, 1:1)
        end
    else
        throw(ArgumentError("`winlen` must be smaller or equal to the length of `x`."))
    end
    if showprogress
        s = @showprogress map(x -> score(f, preprocess(convert.(subseqtype, x))), subseqs)
    else
        s = map(x -> score(f, preprocess(convert.(subseqtype, x))), subseqs)
    end
    Score(reshape(vcat(s...), (length(s), length(s[1]))), 1:subseqs.step:xlen)
    # @inbounds for (i, subseq) in enumerate(subseqs)
    #     sc.s[i, :] = score(f, preprocess(convert.(subseqtype, subseq)))
    # end
    # sc
end

(f::AbstractAcousticFeature)(x) = Score(f, x)

end
