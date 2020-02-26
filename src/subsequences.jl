module Subsequences

export Subsequence

struct Subsequence
    c::AbstractVector
    winlen::Int
    noverlap::Int
    step::Int
    padlen::Int
end

function Subsequence(c, winlen, noverlap)
    winlen = (mod(winlen, 2) == 0) ? winlen+1 : winlen
    winlen > length(c) && throw(ArgumentError("`winlen` has to be smaller than the signal length."))
    step = winlen-noverlap
    padlen = winlen÷2
    Subsequence(c, winlen, noverlap, step, padlen)
end

function Base.iterate(itr::Subsequence, state=1)
    state > length(itr.c) && return nothing
    if state <= itr.padlen
        return vcat(zeros(eltype(itr.c), itr.padlen-(state-1)), itr.c[1:state+itr.padlen]), state+itr.step
    elseif state >= length(itr.c)-itr.padlen
        return vcat(itr.c[state-itr.padlen:end], zeros(eltype(itr.c), itr.padlen-(length(itr.c)-state))), state+itr.step
    else
        return itr.c[state-itr.padlen:state+itr.padlen], state+itr.step
    end
end

Base.length(itr::Subsequence) = ceil(Int64, length(itr.c)/itr.step)

end
