"""Utility functions"""

function spectrumflatten(x::AbstractArray{T,1}, Nnorm::Int) where T <:Real
    if Nnorm >= length(x)
        xfilt = x.-median(x)
        xfilt[xfilt.<0] .= 0
        return xfilt
    end
    M = zeros(length(x))
    for (i, xpart) in enumerate(Subsequence(x, Nnorm, Nnorm-1))
        M[i] = median(xpart)
    end
    xfilt = x.-M
    xfilt[xfilt.<0] .= 0
    xfilt
end

function spectrumflatten(x::AbstractArray{T,2}, Nnorm::Int) where T <: Real
    xfilt = zeros(Float64, size(x))
    for (i, row) in enumerate(eachrow(x))
        xfilt[i, :] = spectrumflatten(row, Nnorm)
    end
    xfilt
end

"""
Get myriad constant given α and scale
"""
function myriadconstant(α, scale)
    (α/(2-α+eps()))*(scale^2)
end

"""
Get myriad constant given estimated α and scale
"""
function myriadconstant(x::AbstractArray{T,1}) where T<:Real
    d = fit(AlphaStable, x)
    myriadconstant(d.α, d.scale)
end

"""
Convert the real signal `x` to an acoustic pressure signal in micropascal.
"""
function pressure(x::AbstractVector{T}, sensitivity::T, gain::T; voltparams::Union{Nothing, Tuple{Int, T}}=nothing) where T<:Real
    ν = 10^(sensitivity/20)
    G = 10^(gain/20)
    if voltparams != nothing
        nbits, vref = voltparams
        x .*= vref/(2^(nbits-1))
    end
    x./(ν*G)
end

"""
Generate a Hilbert envelope of the real signal `x`.
"""
function envelope(x::AbstractVector{T}) where T<:Real
    abs.(hilbert(x))
end
