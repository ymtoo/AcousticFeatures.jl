module Utils

using Statistics

include("subsequences.jl")

using .Subsequences

export spectrumflatten, chirp

function spectrumflatten(x::AbstractArray{T, 1}, Nnorm::Int) where T <:Real
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

function spectrumflatten(x::AbstractArray{T, 2}, Nnorm::Int) where T <: Real
    xfilt = zeros(Float64, size(x))
    for (i, row) in enumerate(eachrow(x))
        xfilt[i, :] = spectrumflatten(row, Nnorm)
    end
    xfilt
end

"""
    Genetate frequency modulated sweep. The implementation is based on https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/waveforms.py#L16
"""
function chirp(f1, f2, duration, fs; method="linear", phi=0)
    n = round(Int64, duration*fs)
    t = (0:n-1)/fs
    if method in ["linear", "lin", "li"]
        beta = (f2-f1)/duration
        phase = 2π.*(f1 .* t .+ 0.5 .* beta .* t .* t)
    else
        ArgumentError("Method must be linear, but a value of $method was given.")
    end
    cos.(phase.+phi)
end

end
