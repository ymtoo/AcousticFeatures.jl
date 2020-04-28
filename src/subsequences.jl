using PaddedViews

struct Subsequence
    c::AbstractVector
    winlen::Int
    noverlap::Int
    step::Int
    lpadlen::Int
    rpadlen::Int
end

function Subsequence(c::AbstractVector{T}, winlen, noverlap; fill=zero(T)) where T
    winlen > length(c) && throw(ArgumentError("`winlen` has to be smaller than the signal length."))
    step = winlen-noverlap
    if mod(winlen, 2) == 0
        lpadlen = (winlen-1)÷2
        rpadlen = winlen÷2
    else
        lpadlen = rpadlen = winlen÷2
    end
    Subsequence(PaddedView(fill, c, (1-lpadlen:length(c)+rpadlen,)), winlen, noverlap, step, lpadlen, rpadlen)
end

function Base.iterate(subseq::Subsequence, state=1)
    state > length(subseq.c.data) && return nothing
    return subseq.c[state-subseq.lpadlen:state+subseq.rpadlen], state+subseq.step
    # if state <= subseq.lpadlen
    #     return vcat(zeros(eltype(subseq.c), subseq.lpadlen-(state-1)), subseq.c[1:state+subseq.rpadlen]), state+subseq.step
    # elseif state >= lenc-subseq.rpadlen
    #     return vcat(subseq.c[state-subseq.lpadlen:end], zeros(eltype(subseq.c), subseq.rpadlen-(lenc-state))), state+subseq.step
    # else
    #     return subseq.c[state-subseq.lpadlen:state+subseq.rpadlen], state+subseq.step
    # end
end

Base.length(subseq::Subsequence) = ceil(Int64, length(subseq.c.data)/subseq.step)

Base.getindex(subseq::Subsequence, i::Number) = iterate(subseq, (1:subseq.step:length(subseq.c.data))[i])[1]
