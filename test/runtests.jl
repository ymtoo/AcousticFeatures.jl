using AcousticFeatures
using AcousticFeatures: name

using AlphaStableDistributions
using BenchmarkTools
using Distributions
using LazyWAVFiles
using LinearAlgebra
using SignalAnalysis
using Test
using WAV

tmpdir = mktempdir()
fs = 100_000
N = 100_000
A = 1.0
frequency = 10_000
t = (0:N-1)./fs

@testset "AcousticFeatures" begin

    @testset "Energy" begin
        @info "Testing Energy"

        x = reshape(A.*sin.(2π*frequency*t), :, 1)

        @test Score(Energy(), x)[1] ≈ (A^2)/2
        winlens = [1_000, 10_000, 1_001, 10_001]
        noverlaps = [0, 100, 500]
        for winlen ∈ winlens, noverlap ∈ noverlaps
            sc1 = Score(Energy(), x; fs=fs, winlen=winlen, noverlap=noverlap)
            @test all(isapprox.(sc1, repeat([(A^2)/2], length(sc1)), atol=0.001))
            sc2 = Score(Energy(), signal(x, fs); winlen=winlen, noverlap=noverlap)
            @test all(isapprox.(sc2, repeat([(A^2)/2], length(sc2)), atol=0.001))
        end

        WAV.wavwrite(x[1:N÷2], joinpath(tmpdir, "1.wav"), Fs=fs)
        WAV.wavwrite(x[N÷2:end], joinpath(tmpdir, "2.wav"), Fs=fs)
        dfile = DistributedWAVFile(tmpdir)
        for winlen in winlens, noverlap in noverlaps
            sc1 = Score(Energy(), dfile; fs=fs, winlen=winlen, noverlap=noverlap)
            @test all(isapprox.(sc1, repeat([(A^2)/2], length(sc1)), atol=0.001))
            sc2 = Score(Energy(), signal(dfile, fs); winlen=winlen, noverlap=noverlap)
            @test all(isapprox.(sc2, repeat([(A^2)/2], length(sc2)), atol=0.001))
        end

        @test name(Energy()) == ["Energy"]
    end

    @testset "Myriad" begin
        @info "Testing Myriad"

        α = 1.9999
        scale = 1.0
        x = rand(AlphaStable(α=α, scale=scale), N)
        d = fit(AlphaStable, x)
        sqKscale = myriadconstant(d.α, d.scale)
        @test Score(Myriad(), x)[1]/N ≈ (log((d.α/(2-d.α+eps()))*(d.scale^2))) atol=0.1
        winlens = [1_000, 10_000, 1_001, 10_001]
        noverlaps = [0, 100, 500]
        for winlen in winlens, noverlap in noverlaps
            sc1 = Score(Myriad(sqKscale), x; fs=fs, winlen=winlen, noverlap=noverlap)
            @test all(isapprox.(sc1 ./ winlen, repeat([(log((d.α/(2-d.α+eps()))*(d.scale^2)))], length(sc1)), atol=0.1))
            sc2 = Score(Myriad(sqKscale), signal(x, fs); winlen=winlen, noverlap=noverlap)
            @test all(isapprox.(sc2 ./ winlen, repeat([(log((d.α/(2-d.α+eps()))*(d.scale^2)))], length(sc2)), atol=0.1))
        end

        WAV.wavwrite(x[1:N÷2], joinpath(tmpdir, "1.wav"), Fs=fs)
        WAV.wavwrite(x[N÷2:end], joinpath(tmpdir, "2.wav"), Fs=fs)
        dfile = DistributedWAVFile(tmpdir)
        for winlen in winlens, noverlap in noverlaps
            sc1 = Score(Myriad(sqKscale), dfile; fs=fs, winlen=winlen, noverlap=noverlap)
            @test all(isapprox.(sc1 ./ winlen, repeat([(log((d.α/(2-d.α+eps()))*(d.scale^2)))], length(sc1)), atol=0.1))
            sc2 = Score(Myriad(sqKscale), signal(dfile, fs); winlen=winlen, noverlap=noverlap)
            @test all(isapprox.(sc2 ./ winlen, repeat([(log((d.α/(2-d.α+eps()))*(d.scale^2)))], length(sc2)), atol=0.1))
        end

        @test name(Myriad()) == ["Myriad"]
    end

    @testset "FrequencyContours" begin
        @info "Testing FrequencyContours"

        duration = N/fs
        f11 = 10_000; f21 = 50_000
        f12 = 1_000; f22 = 20_000
        x1 = real(samples(chirp(f11, f21, duration, fs)))+real(samples(chirp(f12, f22, duration, fs)))
        x1 = reshape(x1, :, 1)
        x2 = real(samples(chirp(f11, f21, duration, fs)))
        x2 = reshape(x2, :, 1)
        n = 512
        nv = 256
        tnorm = 1.0
        fd = 1000.0
        minhprc = 99.0
        minfdist = 1000.0
        mintlen = 0.05
        sc1 = Score(FrequencyContours(n, nv, tnorm, fd, minhprc, minfdist, mintlen), x1; fs=fs)
        ssc1 = Score(FrequencyContours(n, nv, tnorm, fd, minhprc, minfdist, mintlen), signal(x1, fs))
        @test sc1 == ssc1
        sc2 = Score(FrequencyContours(n, nv, tnorm, fd, minhprc, minfdist, mintlen), x2; fs=fs)
        ssc2 = Score(FrequencyContours(n, nv, tnorm, fd, minhprc, minfdist, mintlen), signal(x2, fs))
        @test sc2 == ssc2
        @test sc1[1] > ssc2[1]
        winlens = [10_000, 10_001]
        noverlaps = [0, 100, 500]
        for winlen ∈ winlens, noverlap ∈ noverlaps
            sc1 = Score(FrequencyContours(n, nv, tnorm, fd, minhprc, minfdist, mintlen), x1; fs=fs, winlen=winlen, noverlap=noverlap)
            ssc1 = Score(FrequencyContours(n, nv, tnorm, fd, minhprc, minfdist, mintlen), signal(x1, fs); winlen=winlen, noverlap=noverlap)
            sc2 = Score(FrequencyContours(n, nv, tnorm, fd, minhprc, minfdist, mintlen), x2; fs=fs, winlen=winlen, noverlap=noverlap)
            ssc2 = Score(FrequencyContours(n, nv, tnorm, fd, minhprc, minfdist, mintlen), signal(x2, fs); winlen=winlen, noverlap=noverlap)
            @test sc1 == ssc1
            @test sc2 == ssc2
            @test all(isless.(sc2, sc1))
        end

        tmpdir1 = mktempdir()
        tmpdir2 = mktempdir()
        WAV.wavwrite(x1[1:N÷2], joinpath(tmpdir1, "1.wav"), Fs=fs)
        WAV.wavwrite(x1[N÷2:end], joinpath(tmpdir1, "2.wav"), Fs=fs)
        WAV.wavwrite(x2[1:N÷2], joinpath(tmpdir2, "1.wav"), Fs=fs)
        WAV.wavwrite(x2[N÷2:end], joinpath(tmpdir2, "2.wav"), Fs=fs)
        dfile1 = DistributedWAVFile(tmpdir1)
        dfile2 = DistributedWAVFile(tmpdir2)
        for winlen in winlens, noverlap in noverlaps
            sc1 = Score(FrequencyContours(n, nv, tnorm, fd, minhprc, minfdist, mintlen), dfile1; fs=fs, winlen=winlen, noverlap=noverlap)
            ssc1 = Score(FrequencyContours(n, nv, tnorm, fd, minhprc, minfdist, mintlen), signal(dfile1, fs); winlen=winlen, noverlap=noverlap)
            sc2 = Score(FrequencyContours(n, nv, tnorm, fd, minhprc, minfdist, mintlen), dfile2; fs=fs, winlen=winlen, noverlap=noverlap)
            ssc2 = Score(FrequencyContours(n, nv, tnorm, fd, minhprc, minfdist, mintlen), signal(dfile2, fs); winlen=winlen, noverlap=noverlap)
            @test sc1 == ssc1
            @test sc2 == ssc2
            @test all(isless.(sc2, sc1))
        end

        @test name(FrequencyContours(n, nv, tnorm, fd, minhprc, minfdist, mintlen)) == ["Frequency Contours"]
    end

    @testset "SoundPressureLevel" begin
        @info "Testing SoundPressureLevel"

        x = reshape(A.*sin.(2π*frequency*t), :, 1)
        x = pressure(x, 0.0, 0.0)
        sc = Score(SoundPressureLevel(), x)
        @test sc[1] ≈ 20*log10(1/sqrt(2))
        winlens = [1_000, 10_000, 1_001, 10_001]
        noverlaps = [0, 100, 500]
        for winlen in winlens, noverlap in noverlaps
            sc = Score(SoundPressureLevel(), x; fs=fs, winlen=winlen, noverlap=noverlap)
            ssc = Score(SoundPressureLevel(), signal(x, fs); winlen=winlen, noverlap=noverlap)
            @test sc == ssc
            @test all(isapprox.(sc, repeat([20*log10(1/sqrt(2))], length(sc)), atol=0.01))
        end

        WAV.wavwrite(x[1:length(x)÷2], joinpath(tmpdir, "1.wav"), Fs=fs)
        WAV.wavwrite(x[length(x)÷2:end], joinpath(tmpdir, "2.wav"), Fs=fs)
        dfile = DistributedWAVFile(tmpdir)
        for winlen in winlens, noverlap in noverlaps
            sc = Score(SoundPressureLevel(), dfile; fs=fs, winlen=winlen, noverlap=noverlap)
            ssc = Score(SoundPressureLevel(), signal(dfile, fs); winlen=winlen, noverlap=noverlap)
            @test sc == ssc
            @test all(isapprox.(sc, repeat([20*log10(1/sqrt(2))], length(sc)), atol=0.01))
        end

        @test name(SoundPressureLevel()) == ["SPL"]
    end

    @testset "ImpulseStats" begin
        @info "Testing ImpulseStats"

        trueindices = [201,2254,5322,8888]
        truetimeintervals = diff(trueindices)
        Nᵢ = length(trueindices)
        μᵢᵢ = mean(truetimeintervals)/fs
        varᵢᵢ = var(truetimeintervals)/fs
        x = zeros(N)
        x[trueindices] .= 10.0
        x += 0.1 .* randn(N)

        sc1 = Score(ImpulseStats(10, 1e-3), x; fs=fs)
        ssc1 = Score(ImpulseStats(10, 1e-3), signal(x, fs))
        @test sc1 == ssc1
        sc2 = Score(ImpulseStats(10, 1e-3, true), x; fs=fs)
        ssc2 = Score(ImpulseStats(10, 1e-3, true), signal(x, fs))
        @test sc2 == ssc2
        @test sc1[1,1] == sc2[1,1] == Nᵢ
        @test sc1[1,2] == sc2[1,2] == μᵢᵢ
        @test sc1[1,3] == sc2[1,3] == varᵢᵢ

        m = 100
        lpadlen, rpadlen = AcousticFeatures.getpadlen(m)
        x = zeros(N)
        template = randn(m)
        for trueindex ∈ trueindices
            x[trueindex-lpadlen:trueindex+rpadlen] = template
        end
        x += 0.1 .* randn(N)
        for height ∈ [nothing,0.85]
            impulsestats = ImpulseStats(5, 1e-3, false, template, height)
            @test impulsestats.template == template
            sc3 = Score(impulsestats, x; fs=fs)
            @test sc3[1,1] == Nᵢ
            @test sc3[1,2] == μᵢᵢ
            @test sc3[1,3] == varᵢᵢ
        end


        # with NaNs
        x = [1,2,100,2,1,50,1,-1,3,150,3,1,NaN,5]
        impulsestats = ImpulseStats(0.1, 0.1, false, [1.0,2.0,1.0])
        sc4 = Score(impulsestats, x; fs=1.0)
        @test sc4[1,1] == 3
        @test sc4[1,2] == 3.5
        @test sc4[1,3] == 0.5

        @test name(ImpulseStats(10, 0.1)) == ["Nᵢ", "μᵢᵢ", "varᵢᵢ"]
    end

    @testset "SymmetricAlphaStableStats" begin
        @info "Testing SymmetricAlphaStableStats"

        α = 1.6
        scale = 2.0
        d = AlphaStable(α=α, scale=scale)
        x = rand(d, N, 1)
        sc = Score(SymmetricAlphaStableStats(), x; fs=fs)
        ssc = Score(SymmetricAlphaStableStats(), signal(x, fs))
        @test sc == ssc
        @test sc[1,1] ≈ α atol=0.1
        @test sc[1,2] ≈ scale atol=0.1

        @test name(SymmetricAlphaStableStats()) == ["α", "scale"]
    end

    @testset "Entropy" begin
        @info "Testing Entropy"

        x = reshape(A.*sin.(2π*6250*t), :, 1)
        sc = Score(Entropy(256, 128), x; fs=fs)
        ssc = Score(Entropy(256, 128), signal(x, fs))
        @test sc == ssc
        @test sc[1] ≈ 1.0 atol=1e-2
        @test sc[2] ≈ 0.0 atol=1e-2
        @test sc[3] ≈ 0.0 atol=1e-2

        @test name(Entropy(256, 128)) == ["Temporal Entropy","Spectral Entropy","Entropy Index"]
    end

    @testset "ZeroCrossingRate" begin
        @info "Testing ZeroCrossingRate"
        x = reshape([100.0, 1.0, -2.0, 2.0, -100, 0.0, 10.0], :, 1)
        sc = Score(ZeroCrossingRate(), x; fs=fs)
        ssc = Score(ZeroCrossingRate(), signal(x, fs))
        @test sc == ssc
        @test sc[1] == 4 / (length(x) - 1)
        x = [1.0, -1.0, 1.0, -1.0, 1.0]
        sc = Score(ZeroCrossingRate(), x; fs=fs)
        ssc = Score(ZeroCrossingRate(), signal(x, fs))
        @test sc == ssc
        @test sc[1] == 1.0

        @test name(ZeroCrossingRate()) == ["ZCR"]
    end

    @testset "SpectralCentroid" begin
        @info "Testing SpectralCentroid"

        x = reshape(A.*sin.(2π*6250*t), :, 1)
        sc = Score(SpectralCentroid(), x; fs=fs)
        ssc = Score(SpectralCentroid(), signal(x, fs))
        @test sc == ssc
        @test sc[1] ≈ 6250 atol=0.0001

        @test name(SpectralCentroid()) == ["Spectral Centroid"]
    end

    @testset "SpectralFlatness" begin
        @info "Testing SpectralFlatness"

        x = reshape(A.*sin.(2π*6250*t), :, 1)
        sc = Score(SpectralFlatness(), x; fs=fs)
        ssc = Score(SpectralFlatness(), signal(x, fs))
        @test sc == ssc
        @test sc[1] ≈ 0.0 atol=0.0001

        x = randn(N)
        scnormal = Score(SpectralFlatness(), x; fs=fs)
        sscnormal = Score(SpectralFlatness(), signal(x, fs))
        @test scnormal == sscnormal
        @test scnormal[1] > sc[1]

        @test name(SpectralFlatness()) == ["Spectral Flatness"]
    end

    @testset "PermutationEntropy" begin
        @info "Testing PermutationEntropy"

        x = reshape([4,7,9,10,6,11,3], :, 1)
        m = 3
        τ1 = 1
        τ2 = 2
        norm1 = false
        norm2 = true
        weighted1 = false
        weighted2 = true

        sc111 = Score(PermutationEntropy(m, τ1, norm1, weighted1), x; fs=fs)
        ssc111 = Score(PermutationEntropy(m, τ1, norm1, weighted1), signal(x, fs))
        @test sc111 == ssc111
        sc121 = Score(PermutationEntropy(m, τ1, norm2, weighted1), x; fs=fs)
        ssc121 = Score(PermutationEntropy(m, τ1, norm2, weighted1), signal(x, fs))
        @test sc121 == ssc121
        sc131 = Score(PermutationEntropy(m), x; fs=fs)
        ssc131 = Score(PermutationEntropy(m), signal(x, fs))
        @test sc131 == ssc131
        @test sc111[1] ≈ 1.5219 atol=0.0001
        @test sc121[1] ≈ 0.5887 atol=0.0001
        @test sc121[1] == sc131[1] 

        sc211 = Score(PermutationEntropy(m, τ2, norm1, weighted1), x; fs=fs)
        sc221 = Score(PermutationEntropy(m, τ2, norm2, weighted1), x; fs=fs)
        sc231 = Score(PermutationEntropy(m, τ2), x; fs=fs)
        @test sc211[1] ≈ 1.5850 atol=0.0001
        @test sc221[1] ≈ 0.6131 atol=0.0001
        @test sc221[1] == sc231[1] 

        sc112 = Score(PermutationEntropy(m, τ1, norm1, weighted2), x; fs=fs)
        sc122 = Score(PermutationEntropy(m, τ1, norm2, weighted2), x; fs=fs)
        @test sc112[1] ≈ 1.4140 atol=0.0001
        @test sc122[1] ≈ 0.5470 atol=0.0001

        sc212 = Score(PermutationEntropy(m, τ2, norm1, weighted2), x; fs=fs)
        sc222 = Score(PermutationEntropy(m, τ2, norm2, weighted2), x; fs=fs)
        @test sc212[1] ≈ 1.5233 atol=0.0001
        @test sc222[1] ≈ 0.5893 atol=0.0001

        @test name(PermutationEntropy(m)) == ["Permutation Entropy"]
    end

    @testset "PSD" begin
        @info "Testing PSD"

        freq = 3000
        fs = 96000
        s = cw(freq, 0.1, fs) |> real 
        sc = Score(PSD(64, 32, fs), samples(s); fs=framerate(s))
        ssc = Score(PSD(64, 32, fs), s)
        @test sc == ssc
        @test sc.axes[2][argmax(sc)[2]] == "PSD-$(round(freq; digits=1))Hz"
    end

    @testset "AcousticComplexityIndex" begin
        @info "Testing AcousticComplexityIndex"

        freq1, freq2 = 1000, 48000
        fs = 96000
        s1 = chirp(freq1, freq2, 5.0, fs) |> real 
        s2 = cw(freq1, 5.0, fs) |> real 
        s1 += randn(length(s1))
        s2 += randn(length(s2))
        sc1 = Score(AcousticComplexityIndex(1048, 0, 30), samples(s1); fs=framerate(s1))
        ssc1 = Score(AcousticComplexityIndex(1048, 0, 30), s1)
        @test sc1 == ssc1
        sc2 = Score(AcousticComplexityIndex(1048, 0, 30), samples(s2); fs=framerate(s2))
        ssc2 = Score(AcousticComplexityIndex(1048, 0, 30), s2)
        @test sc2 == ssc2
        @test sc1[1] > sc2[1]
    end

    @testset "StatisticalComplexity" begin
        @info "Testing StatisticalComplexity"

        x = reshape([4,7,9,10,6,11,3], :, 1)
        m = 3
        τ1 = 1
        τ2 = 2

        sc11 = Score(StatisticalComplexity(m, τ1), x; fs=fs)
        ssc11 = Score(StatisticalComplexity(m, τ1), signal(x, fs))
        @test sc11 == ssc11
        sc12 = Score(StatisticalComplexity(m, τ2), x; fs=fs)
        ssc12 = Score(StatisticalComplexity(m, τ2), signal(x, fs))
        @test sc12 == ssc12
        sc13 = Score(StatisticalComplexity(m), x; fs=fs)
        ssc13 = Score(StatisticalComplexity(m), signal(x, fs))
        @test sc13 == ssc13
        @test sc11[1] ≈ 0.2899 atol=0.0001
        @test sc12[1] ≈ 0.2915 atol=0.0001
        @test sc11[1] == sc13[1] 

    end

    @testset "Score" begin
        @info "Testing Score"
        
        @test_throws ArgumentError Score(Energy(), randn(1000); winlen=1001)
        f = Energy()
        @test Score(f, randn(100000); fs=fs)[1] ≈ f(randn(100000); fs=fs)[1] atol=0.1
    end

    @testset "Utils" begin
        @info "Testing Utils"

        x = [1, 2, 3, 4, 5, 6, 7]
        Nnorm = 3
        xfilt = x - [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0]
        @test spectrumflatten(x, Nnorm) == xfilt

        x = [[1 2 3 4 5 6 7];
             [8 9 10 11 12 13 14];
             [15 16 17 18 19 20 21]]
        xfiltrow = x - [[1.0 2.0 3.0 4.0 5.0 6.0 6.0]; 
                        [8.0 9.0 10.0 11.0 12.0 13.0 13.0];
                        [15.0 16.0 17.0 18.0 19.0 20.0 20.0]]
        xfiltcol = x - [[1.0 2.0 3.0 4.0 5.0 6.0 7.0];
                        [8.0 9.0 10.0 11.0 12.0 13.0 14.0];
                        [8.0 9.0 10.0 11.0 12.0 13.0 14.0]]                
        @test spectrumflatten(x, Nnorm) == xfiltrow
        @test spectrumflatten(x, Nnorm; dims=1) == xfiltcol

        nbits = 16
        vref = 1.0
        s = signal(reshape(cw(64, 1, 512), :, 1), 512)
        xvolt = vref .* real(s)
        xbit = xvolt*(2^(nbits-1))
        sensitivity = 0.0
        gain = 0.0
        p1 = pressure(xvolt, sensitivity, gain)
        p2 = pressure(xbit, sensitivity, gain, voltparams=(nbits, vref))
        @test p1 == p2

        even_ms = 2:2:10
        for even_m ∈ even_ms
            @test AcousticFeatures.getpadlen(even_m) == ((even_m-1) ÷ 2, even_m ÷ 2)
        end
        odd_ms = 1:2:10
        for odd_m ∈ odd_ms
            @test AcousticFeatures.getpadlen(odd_m) == (odd_m ÷ 2, odd_m ÷ 2)
        end

        p = [1,2,3,4,5,6,7]
        @test AcousticFeatures.ordinalpatterns(p,3,1) == [1.0]
        @test AcousticFeatures.ordinalpatterns(p,3,2) == [1.0]
        p = [1,2,1,2,1,2,1]
        @test AcousticFeatures.ordinalpatterns(p,3,1) == [0.6,0.4]
        @test AcousticFeatures.ordinalpatterns(p,3,2) == [1.0]

        n = 1000
        m = 10
        index = rand(1:n-m)
        lpadlen, rpadlen = AcousticFeatures.getpadlen(m)
        x = randn(1000)
        template = x[index-lpadlen:index+rpadlen]
        s = AcousticFeatures.normcrosscorr(x, template)
        @test s[index] == 1.0
        @test all(s[1:n .!= index] .< 1.0)

    end

end
