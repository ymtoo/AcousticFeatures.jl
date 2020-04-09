module AcousticFeatures

using AlphaStableDistributions, DSP, Peaks, Statistics, StatsBase

include("subsequences.jl")
include("utils.jl")

export Energy, Myriad, FrequencyContours, SoundPressureLevel, ImpulseStats, AlphaStableStats, Score

export Subsequence

export spectrumflatten, myriadconstant, pressure, envelope

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

struct FrequencyContours{FT<:Real,T<:Real} <: AbstractAcousticFeature
    fs::FT
    n::Int
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
end
ImpulseStats(fs) = ImpulseStats(fs, 10, 1e-3)

struct AlphaStableStats <: AbstractAcousticFeature end

struct MaxDemonSpectrum{FT<:Real} <: AbstractAcousticFeature
    fs::FT
end

mutable struct Score{VT1<:AbstractArray{<:Real},VT2<:AbstractRange{Int}}
    s::VT1
    indices::VT2
end
################################################################################
#
#   Implementations
#
################################################################################

outputlength(::Energy) = 1
outputeltype(::Energy) = Float64
"""
    Score of `x` based on mean energy.
"""
score(::Energy, x::AbstractVector{T}) where T<:Real = [mean(abs2, x)]

outputlength(::Myriad{S}) where S<:Real = 1
outputeltype(::Myriad{S}) where S<:Real = Float64
"""
    Score of `x` based on myriad

    Reference:
    Mahmood et. al., "Optimal and Near-Optimal Detection in Bursty Impulsive Noise,"
    IEEE Journal of Oceanic Engineering, vol. 42, no. 3, pp. 639--653, 2016.
"""
function score(f::Myriad{S}, x::AbstractVector{T}) where {T<:Real, S<:Real}
    sqKscale = f.sqKscale
    [sum(x -> log(sqKscale + abs2(x)), x)]
end


score(f::Myriad{Nothing}, x) = score(Myriad(myriadconstant(x)), x)

outputlength(::FrequencyContours) = 1
outputeltype(::FrequencyContours) = Float64
"""
    Score of `x` based on frequency contours count

    Reference:
    D. Mellinger, R. Morrissey, L. Thomas, J. Yosco, "A method for detecting whistles, moans, and other frequency
    contour sounds", 2011 J. Acoust. Soc. Am. 129 4055
"""
function score(f::FrequencyContours, x::AbstractVector{T}) where T<:Real
    spec = spectrogram(x, f.n, f.n÷2; fs=f.fs, window=DSP.hamming)
    p  = spec.power; frequency=spec.freq; t=spec.time
    δt = t[2]-t[1]
    δf = frequency[2]-frequency[1]
    f.tnorm === nothing ? Nnorm = size(p, 2) : Nnorm = f.tnorm÷(δt) |> Int
    p    = spectrumflatten(p, Nnorm) #noise-flattened spectrogram
    crds,_ = peakprom(p[:, 1], Maxima(), trunc(Int, f.minfdist÷δf), eps(T)+percentile(p[:, 1], f.minhprc))
    ctrs = [[(crd[1], 1)] for crd in crds]
    for (i, col) in enumerate(eachcol(p[:, 2:end]))
        col = collect(col)
        crds,_ = Peaks.peakprom(col, Maxima(), trunc(Int, f.minfdist/δf), eps(T)+percentile(col, f.minhprc))
        for crd in crds
            if length(ctrs) == 0
                ctrs = [[(crd[1], 1)] for crd in crds]
            else
                idxselect = Int64[]
                costselect = Float64[]
                for (j, ctr) in enumerate(ctrs)
                    if (ctr[end][2] == i-1) && abs(frequency[ctr[end][1]]-frequency[crd[1]]) <= f.fd
                        push!(idxselect, j)
                        push!(costselect, abs(frequency[ctr[end][1]]-frequency[crd[1]]))
                    end
                end
                if isempty(idxselect)
                    push!(ctrs, [(crd[1], i)])
                else
                    idxopt = idxselect[argmin(costselect)]
                    push!(ctrs[idxopt], (crd[1], i))
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

outputlength(::SoundPressureLevel) = 1
outputeltype(::SoundPressureLevel) = Float64
"""
Score of `x` based on Sound Pressure Level (SPL). `x` is in micropascal. In water, the common reference is 1 micropascal. In air, the common reference is 20 micropascal.
"""
function score(f::SoundPressureLevel, x::AbstractVector{T}) where T<:Real
    rmsx = sqrt(mean(abs2.(x)))
    [20*log10(rmsx/f.ref)]
end

outputlength(::ImpulseStats) = 3
outputeltype(::ImpulseStats) = Float64
"""
Score of `x` based on number of impulses, mean and variance of inter-impulse intervals. The minimum height of impulses is defined by `a+k*b` where `a` is median of the envelope of `x` and `b` is median absolute deviation (MAD) of the envelope of `x`.
"""
function score(f::ImpulseStats, x::AbstractVector{T}) where T<:Real
    env = envelope(x)
    center = Statistics.median(x)
    height = center+f.k*mad(x, center=center, normalize=false)
    distance = trunc(Int, f.tdist*f.fs)
    crds, _ = Peaks.peakprom(env, Maxima(), distance, height)
    timeintervals = diff(crds)
    [length(crds) mean(timeintervals)/f.fs var(timeintervals)/f.fs]
end

outputlength(::AlphaStableStats) = 2
outputeltype(::AlphaStableStats) = Float64
"""
Score of `x` based on the parameters of Symmetric Alpha Stable Distributions. The parameter α measures the impulsiveness while the parameter scale measures the width of the distributions.
"""
function score(f::AlphaStableStats, x::AbstractVector{T}) where T<:Real
    d = fit(AlphaStable, x)
    [d.α d.scale]
end

outputlength(::MaxDemonSpectrum) = 1
"""
"""
function score(f::MaxDemonSpectrum, x::AbstractVector{T}) where T<:Real
    xd = demon(x, fs=f.fs)
end

function Score(f::AbstractAcousticFeature, x::AbstractVector{T}; winlen::Int=length(x), noverlap::Int=0, subseqtype::DataType=Float64, preprocess::Function=x->x) where {T<:Real, N, L}
    xlen = length(x)
    if winlen < xlen
        (noverlap < 0) && throw(ArgumentError("`noverlap` must be larger or equal to zero."))
        subseqs = Subsequence(x, winlen, noverlap)
        sc = Score(zeros(outputeltype(f), length(subseqs), outputlength(f)), 1:subseqs.step:xlen)
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
    @inbounds for (i, subseq) in enumerate(subseqs)
        sc.s[i, :] = score(f, preprocess(convert.(subseqtype, subseq)))
    end
    sc
end

(f::AbstractAcousticFeature)(x) = Score(f, x)

end
