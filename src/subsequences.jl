struct Subsequence{VT<:AbstractVector}
    c::VT
#    fillvalue::T
    winlen::Int
    noverlap::Int
    step::Int
    lpadlen::Int
    rpadlen::Int
end

function getpadlen(winlen)
    if mod(winlen, 2) == 0
        lpadlen = (winlen-1)÷2
        rpadlen = winlen÷2
    else
        lpadlen = rpadlen = winlen÷2
    end
    lpadlen, rpadlen
end

function Subsequence(c::AbstractVector{T}, winlen, noverlap; fillvalue::T=zero(T)) where T
    winlen > length(c) && throw(ArgumentError("`winlen` has to be smaller than the signal length."))
    step = winlen-noverlap
    lpadlen, rpadlen = getpadlen(winlen)
    # Subsequence(c,
    #             fillvalue,
    #             winlen,
    #             noverlap,
    #             step,
    #             lpadlen,
    #             rpadlen)
    Subsequence(PaddedView(fillvalue, c, (1-lpadlen:length(c)+rpadlen,)), winlen, noverlap, step, lpadlen, rpadlen)
end

function Base.iterate(subseq::Subsequence, state=1)
    lenc = length(subseq.c.data)
    state > lenc && return nothing
    return @view(subseq.c[state-subseq.lpadlen:state+subseq.rpadlen]), state+subseq.step
    # if state <= subseq.lpadlen
    #     return vcat(fill(subseq.fillvalue, subseq.lpadlen-(state-1)), @view(subseq.c[1:state+subseq.rpadlen])), state+subseq.step
    # elseif state >= lenc-subseq.rpadlen
    #     return vcat(@view(subseq.c[state-subseq.lpadlen:end]), fill(subseq.fillvalue, subseq.rpadlen-(lenc-state))), state+subseq.step
    # else
    #     return @view(subseq.c[state-subseq.lpadlen:state+subseq.rpadlen]), state+subseq.step
    # end
end

Base.length(subseq::Subsequence) = ceil(Int64, length(subseq.c.data)/subseq.step)

Base.getindex(subseq::Subsequence, i::Number) = iterate(subseq, (1:subseq.step:length(subseq.c.data))[i])[1]
