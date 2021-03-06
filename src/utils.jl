"""Utility functions"""

# """
# Compute the squared L2 norm.
# """
# norm²(x) = sum(abs2, x)
"""
Spectral flattening.
"""
function spectrumflatten(x::AbstractArray{T,1}, Nnorm::Int) where T <:Real
    if Nnorm >= length(x)
        xfilt = x.-median(x)
        xfilt[xfilt.<0] .= 0
        return xfilt
    end
    M = map(median, Subsequence(x, Nnorm, Nnorm-1))
    xfilt = x.-M
    xfilt[xfilt.<0] .= 0
    xfilt
end
function spectrumflatten(x::AbstractArray{T,2}, Nnorm::Int; dims::Int=2) where T <: Real
    mapslices(v->spectrumflatten(v, Nnorm), x, dims=dims)
end

"""
Get myriad constant given α and scale.
"""
function myriadconstant(α, scale)
    (α/(2-α+eps()))*(scale^2)
end

"""
Get myriad constant given `x`.
"""
function myriadconstant(x::AbstractVector{T}) where T<:Real
    d = fit(AlphaStable, x)
    myriadconstant(d.α, d.scale)
end

# """
# Get vector myriad constants given α and Rₘ.
# """
# vmyriadconstant(α::T, Rₘ::AbstractMatrix{T}) where T = (α/(2-α+eps())), Rₘ

# """
# Get vector myriad constants given `x`.
# """
# function vmyriadconstant(x::AbstractVector{T}, m::Integer=4) where T<:Real
#     d = fit(AlphaSubGaussian, x, m)
#     vmyriadconstant(d.α, d.R)
# end

"""
Convert a real signal `x` to an acoustic pressure signal in micropascal.
"""
function pressure(x::AbstractVector{T}, sensitivity::T, gain::T; voltparams::Union{Nothing, Tuple{Int, T}}=nothing) where T<:Real
    ν = exp10(sensitivity/20)
    G = exp10(gain/20)
    if voltparams !== nothing
        nbits, vref = voltparams
        x .*= vref/(2^(nbits-1))
    end
    x./(ν*G)
end

"""
Generate a Hilbert envelope of a real signal `x`.
"""
function envelope(x::AbstractVector{T}) where T<:Real
    abs.(hilbert(x))
end

"""
Get the normalized envelope of of a real signal `x`.
"""
function normalize_envelope(x::AbstractVector{T}) where T<:Real
    env = envelope(x)
    env/sum(env)
end

"""
Get the normalized spectrum of a real signal `x`.
"""
function normalize_spectrum(s::AbstractMatrix{T}) where T<:Real
    sf = sum(s, dims=2)
    sf/sum(sf)
end

"""
Compute ordinal patterns of a real signal `x`.
"""
function ordinalpatterns(x::AbstractVector{T}, m::Integer, τ::Integer=1) where T<:Real
    n = length(x) - τ*m + 1
    ps = Vector{Int}[]
    counts = Int[]
    for t in 1:n
        s = @view x[t:τ:t+τ*(m-1)]
        p = sortperm(sortperm(s))
#        v = [p == ptmp for ptmp in ps]
        cntindex = 0
        for (i, ptmp) in enumerate(ps)
            if p == ptmp
                cntindex = i
                break
            end
        end
        if cntindex != 0 #any(v)
            counts[cntindex] += 1
        else
            push!(ps, p)
            push!(counts, 1)
        end
    end
    counts / sum(counts) 
end