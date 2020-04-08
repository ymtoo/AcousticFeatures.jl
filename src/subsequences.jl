struct Subsequence
    c::AbstractVector
    winlen::Int
    noverlap::Int
    step::Int
    lpadlen::Int
    rpadlen::Int
end

function Subsequence(c, winlen, noverlap)
#    winlen = (mod(winlen, 2) == 0) ? winlen+1 : winlen
    winlen > length(c) && throw(ArgumentError("`winlen` has to be smaller than the signal length."))
    step = winlen-noverlap
    if mod(winlen, 2) == 0
        lpadlen = (winlen-1)÷2
        rpadlen = winlen÷2
    else
        lpadlen = rpadlen = winlen÷2
    end
#    padlen = winlen÷2
    Subsequence(c, winlen, noverlap, step, lpadlen, rpadlen)
end

function Base.iterate(subseq::Subsequence, state=1)
    lenc = length(subseq.c)
    state > lenc && return nothing
    if state <= subseq.lpadlen
#        return vcat(zeros(eltype(itr.c), itr.padlen-(state-1)), view(itr.c, 1:state+itr.padlen)), state+itr.step
        return vcat(zeros(eltype(subseq.c), subseq.lpadlen-(state-1)), subseq.c[1:state+subseq.rpadlen]), state+subseq.step
    elseif state >= lenc-subseq.rpadlen
#        return vcat(view(itr.c, state-itr.padlen:lenc), zeros(eltype(itr.c), itr.padlen-(lenc-state))), state+itr.step
        return vcat(subseq.c[state-subseq.lpadlen:end], zeros(eltype(subseq.c), subseq.rpadlen-(lenc-state))), state+subseq.step
    else
#        return view(itr.c, state-itr.padlen:state+itr.padlen), state+itr.step
        return subseq.c[state-subseq.lpadlen:state+subseq.rpadlen], state+subseq.step
    end
end

Base.length(subseq::Subsequence) = ceil(Int64, length(subseq.c)/subseq.step)

Base.getindex(subseq::Subsequence, i::Number) = iterate(subseq, (1:subseq.step:length(subseq.c))[i])[1]
