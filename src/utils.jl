"""Utility functions"""

"""
Spectral flattening.
"""
function spectrumflatten(x::AbstractVector{T}, Nnorm::Int) where T <:Real
    if Nnorm >= length(x)
        xfilt = x.-median(x)
        xfilt[xfilt.<0] .= 0
        return xfilt
    end
    lpadlen, rpadlen = getpadlen(Nnorm)
    xpad = padded(x, (lpadlen, rpadlen)) #vcat(zeros(T, lpadlen), x, zeros(T, rpadlen)) #BorderArray(x, Fill(zero(T), (lpadlen,), (rpadlen,)))
    M = map(median, SignalAnalysis.partition(signal(xpad, 1), Nnorm; step = 1, flush = false))
    xfilt = x .- M
    xfilt[xfilt.<0] .= 0
    xfilt
end
function spectrumflatten(x::AbstractMatrix{T}, Nnorm::Int; dims::Int=2) where T <: Real
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

"""
Convert a real signal `x` to an acoustic pressure signal in micropascal.
"""
function pressure(x::T, sensitivity::T, gain::T; voltparams::Union{Nothing, Tuple{Int, T}}=nothing) where T<:Real
    ν = exp10(sensitivity / 20)
    G = exp10(gain / 20)
    if voltparams !== nothing
        nbits, vref = voltparams
        x *= vref / (2 ^ (nbits - 1))
    end
    x / (ν * G)
end

"""
Generate a Hilbert envelope of a real signal `x`.
"""
function envelope(x::AbstractVecOrMat{T}) where T<:Real
    abs.(analytic(x))
end

"""
Get the normalized envelope of of a real signal `x`.
"""
function normalize_envelope(x::AbstractVector{T}) where T<:Real
    env = envelope(x)
    env ./ (sum(env) + eps(T))
end

"""
Get the normalized spectrum of a real signal `x`.
"""
function normalize_spectrum(s::AbstractMatrix{T}) where T<:Real
    sf = sum(s, dims=2)
    sf ./ sum(sf)
end

"""
Get left pad and right pad length.
"""
function getpadlen(winlen)
    if mod(winlen, 2) == 0
        lpadlen = (winlen-1)÷2
        rpadlen = winlen÷2
    else
        lpadlen = rpadlen = winlen÷2
    end
    lpadlen, rpadlen
end

"""
Compute ordinal patterns of a real signal `x`.
"""
function ordinalpatterns(x::AbstractVector{T}, 
                         m::Integer, 
                         τ::Integer=1, 
                         weighted::Bool=false) where T<:Real
    n = length(x) - τ*m + τ  
    ps = UInt[]
    counts = Float64[]
    @inbounds for t ∈ 1:n
        s = @view x[t:τ:t+τ*(m-1)]
        p = sortperm(s) |> hash
        cntindex = !isempty(ps) ? findfirst(p .== ps) : nothing
        count = weighted ? var(s) : 1.0
        if !isnothing(cntindex)
            counts[cntindex] += count
        else
            push!(ps, p)
            push!(counts, count)
        end
    end
    counts ./ sum(counts) 
end

function normcrosscorr(x::AbstractVector{T}, template::AbstractVector{T}) where {T}
    s = similar(x)
    m = length(template)
    lpadlen, rpadlen = getpadlen(m)
    xpad = padded(x, (lpadlen, rpadlen))
    for i ∈ eachindex(x)
        @views s[i] = cor(xpad[i-lpadlen:i+rpadlen], template)
    end
    s
end
