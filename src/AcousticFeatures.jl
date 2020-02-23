module AcousticFeatures

include("subsequences.jl")
include("utils.jl")

using .Subsequences, .Utils

using AlphaStableDistributions, DSP, Findpeaks, StatsBase

export Energy, Myriad, FrequencyContours, Score, Subsequence, spectrumflatten, chirp

abstract type AbstractAcousticFeature end

################################################################################
#
#   AcousticFeature types
#
################################################################################
struct Energy <: AbstractAcousticFeature end

struct Myriad <: AbstractAcousticFeature
    α::Union{Nothing, Float64}
    scale::Union{Nothing, Float64}
end
Myriad() = Myriad(nothing, nothing)

struct FrequencyContours <: AbstractAcousticFeature
    fs::Real
    n::Int
    tnorm::Union{Nothing, Real} #time constant for normalization (sec)
    fd::Real #frequency difference from one step to the next (Hz)
    minhprc::Real
    minfdist::Real
    mintlen::Real
end
#FrequencyContours(fs, n, tnorm) = FrequencyContours(fs, n ,nothing ,0.0 ,fs/2)

mutable struct Score{T}
    s::AbstractArray{T, 1}
    index::AbstractArray{Int, 1}
end
################################################################################
#
#   Implementations
#
################################################################################

"""
    Score of `x` based on mean energy.
"""
score(::Energy, x::AbstractArray{T, 1}) where T = mean(abs2, x)

"""
    Score of `x` based on myriad

    Reference:
    Mahmood et. al., "Optimal and Near-Optimal Detection in Bursty Impulsive Noise,"
    IEEE Journal of Oceanic Engineering, vol. 42, no. 3, pp. 639--653, 2016.
"""
function score(f::Myriad, x::AbstractArray{T, 1}) where T
    α=f.α; scale=f.scale
    if any([α, scale] .== [nothing, nothing])
        d = fit(AlphaStable, x)
        α=d.α; scale=d.scale
    end
    sq_Kscale = (α/(2-α+eps()))*(scale^2)
    sum(x -> log(sq_Kscale + x^2), x)
end

"""
    Score of `x` based on frequency contours count

    Reference:
    D. Mellinger, R. Morrissey, L. Thomas, J. Yosco, "A method for detecting whistles, moans, and other frequency
    contour sounds", 2011 J. Acoust. Soc. Am. 129 4055
"""
function score(f::FrequencyContours, x::AbstractArray{T, 1}) where T
    spec = spectrogram(x, f.n, f.n÷2; fs=f.fs, window=DSP.hamming)
    p=spec.power; frequency=spec.freq; t=spec.time
    δt = t[2]-t[1]
    δf = frequency[2]-frequency[1]
    f.tnorm == nothing ? Nnorm = size(p, 2) : Nnorm = f.tnorm÷(δt) |> Int
    p = spectrumflatten(p, Nnorm) #noise-flattened spectrogram
    crds = findpeaks(p[:, 1], minHeight=percentile(p[:, 1], f.minhprc), minDist=trunc(Int, f.minfdist÷δf))
    ctrs = [[(crd[1], 1)] for crd in crds]
    for (i, col) in enumerate(eachcol(p[:, 2:end]))
        col = collect(col)
        crds = findpeaks(col, minHeight=percentile(col, f.minhprc), minDist=trunc(Int, f.minfdist÷δf))
        for crd in crds
            if length(ctrs) == 0
                ctrs = [[(crd[1], 1)] for crd in crds]
            else
                idxselect = Int64[]
                costselect = Float64[]
                for (j, ctr) in enumerate(ctrs)
                    if (ctr[end][2] == i-1) && abs(frequency[ctr[end][1]]-frequency[crd[1]]) <= f.fd
                        append!(idxselect, j)
                        append!(costselect, abs(frequency[ctr[end][1]]-frequency[crd[1]]))
                    end
                end
                if isempty(idxselect)
                    append!(ctrs, [[(crd[1], i)]])
                else
                    idxopt = idxselect[argmin(costselect)]
                    append!(ctrs[idxopt], [(crd[1], i)])
                end
            end
        end
    end
    idxdelete = Int64[]
    for (i, ctr) in enumerate(ctrs)
        (length(ctr)-1)*(δt) < f.mintlen && append!(idxdelete, i)
    end
    deleteat!(ctrs, idxdelete)
    count = 0
    [count += length(ctr) for ctr in ctrs]
    count/length(p)
end


function Score(f::AbstractAcousticFeature, x:: AbstractArray{T, 1}; winlen::Int=length(x), noverlap::Int=0) where {T, N, L}
    xlen = length(x)
    if winlen < xlen
        sc = Score(zeros(Float64, 0), zeros(Int64, 0))
    elseif winlen == xlen
        return Score([score(f, x)], [1])
    else
        throw(ArgumentError("`winlen` must be smaller or equal to the length of `x`."))
    end
    (noverlap < 0) && throw(ArgumentError("`noverlap` must be larger or equal to zero."))
    subseqs = Subsequence(x, winlen, noverlap)
    state = 1
    step = subseqs.winlen-subseqs.noverlap
    for subseq in subseqs
        append!(sc.s, score(f, subseq))
        append!(sc.index, state)
        state += step
    end
    sc
end

(f::AbstractAcousticFeature)(x) = Score(f, x)

end
