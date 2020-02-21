module Subsequences

export Subsequence

struct Subsequence
    c
    winlen::Int
    noverlap::Int
    Subsequence(c, winlen, noverlap) = (mod(winlen, 2) == 0) ? new(c, winlen+1, noverlap) : new(c, winlen, noverlap)
end

function Base.iterate(itr::Subsequence, state=1)
    state > length(itr.c) && return nothing
    itr.winlen > length(itr.c) && throw(ArgumentError("`winlen` has to be smaller than the signal length."))
    step = itr.winlen-itr.noverlap
    padlen = itr.winlen÷2
    if state <= padlen
        return vcat(zeros(eltype(itr.c), padlen-(state-1)), itr.c[1:state+padlen]), state+step
    elseif state >= length(itr.c)-padlen
        return vcat(itr.c[state-padlen:end], zeros(eltype(itr.c), padlen-(length(itr.c)-state))), state+step
    else
        return itr.c[state-padlen:state+padlen], state+step
    end
end

end
