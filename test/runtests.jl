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

        x = A.*sin.(2π*frequency*t)
        @test Score(Energy(), x)[1] ≈ (A^2)/2
        winlens = [1_000, 10_000, 1_001, 10_001]
        noverlaps = [0, 100, 500]
        for winlen in winlens, noverlap in noverlaps
            subseq = Subsequence(x, winlen, noverlap)
            sc = Score(Energy(), x; winlen=winlen, noverlap=noverlap)
            spart = sc[(sc.axes[1] .> subseq.winlen÷2) .& (sc.axes[1] .< length(x)-subseq.winlen÷2)]
            @test all(isapprox.(spart, repeat([(A^2)/2], length(spart)), atol=0.001))
        end
        WAV.wavwrite(x[1:N÷2], joinpath(tmpdir, "1.wav"), Fs=fs)
        WAV.wavwrite(x[N÷2:end], joinpath(tmpdir, "2.wav"), Fs=fs)
        dfile = DistributedWAVFile(tmpdir)
        for winlen in winlens, noverlap in noverlaps
            subseq = Subsequence(dfile, winlen, noverlap)
            sc = Score(Energy(), dfile; winlen=winlen, noverlap=noverlap)
            spart = sc[(sc.axes[1] .> subseq.winlen÷2) .& (sc.axes[1] .< length(x)-subseq.winlen÷2)]
            @test all(isapprox.(spart, repeat([(A^2)/2], length(spart)), atol=0.001))
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
            subseq = Subsequence(x, winlen, noverlap)
            sc = Score(Myriad(sqKscale), x; winlen=winlen, noverlap=noverlap)
            spart = sc[(sc.axes[1] .> subseq.winlen÷2) .& (sc.axes[1] .< length(x)-subseq.winlen÷2)]
            @test all(isapprox.(spart./subseq.winlen, repeat([(log((d.α/(2-d.α+eps()))*(d.scale^2)))], length(spart)), atol=0.1))
        end
        WAV.wavwrite(x[1:N÷2], joinpath(tmpdir, "1.wav"), Fs=fs)
        WAV.wavwrite(x[N÷2:end], joinpath(tmpdir, "2.wav"), Fs=fs)
        dfile = DistributedWAVFile(tmpdir)
        for winlen in winlens, noverlap in noverlaps
            subseq = Subsequence(dfile, winlen, noverlap)
            sc = Score(Myriad(sqKscale), dfile; winlen=winlen, noverlap=noverlap)
            spart = sc[(sc.axes[1] .> subseq.winlen÷2) .& (sc.axes[1] .< length(x)-subseq.winlen÷2)]
            @test all(isapprox.(spart./subseq.winlen, repeat([(log((d.α/(2-d.α+eps()))*(d.scale^2)))], length(spart)), atol=0.1))
        end

        @test name(Myriad()) == ["Myriad"]
    end

    # @testset "VMyriad" begin
    #     @info "Testing VMyriad"

    #     # α = 1.3
    #     # identitymatrix = zeros(5, 5)
    #     # identitymatrix[diagind(identitymatrix)] .= 1.0
    #     # d = AlphaSubGaussian(α=α, n=N)
    #     # x = rand(d)
    #     # d̂ = fit(AlphaSubGaussian, x, 4)

    #     α = 1.3
    #     d = AlphaSubGaussian(;α=α, n=N)
    #     x = rand(d)
    #     d̂ = fit(AlphaSubGaussian, x, 4)

    #     f0 = VMyriad(vmyriadconstant(d̂.α, d̂.R)...)
    #     f1 = VMyriad(vmyriadconstant(1.8, d̂.R)...)
    #     identitymatrix = zeros(5, 5)
    #     identitymatrix[diagind(identitymatrix)] .= 1.0
    #     f2 = VMyriad(vmyriadconstant(1.8, identitymatrix)...)

    #     @test Score(f0, x).s[1] ≈ Score(VMyriad(vmyriadconstant(d̂.α, d̂.R)...), x).s[1]
    #     @test Score(f0, x).s[1] < Score(f1, x).s[1]
    #     @test Score(f0, x).s[1] < Score(f2, x).s[1]
    # end

    @testset "FrequencyContours" begin
        @info "Testing FrequencyContours"

        duration = N/fs
        f11 = 10_000; f21 = 50_000
        f12 = 1_000; f22 = 20_000
        x1 = real(samples(chirp(f11, f21, duration, fs)))+real(samples(chirp(f12, f22, duration, fs)))
        x2 = real(samples(chirp(f11, f21, duration, fs)))
        n = 512
        nv = 256
        tnorm = 1.0
        fd = 1000.0
        minhprc = 99.0
        minfdist = 1000.0
        mintlen = 0.05
        sc1 = Score(FrequencyContours(fs, n, nv, tnorm, fd, minhprc, minfdist, mintlen), x1)
        sc2 = Score(FrequencyContours(fs, n, nv, tnorm, fd, minhprc, minfdist, mintlen), x2)
        @test sc1[1] > sc2[1]
        winlens = [10_000, 10_001]
        noverlaps = [0, 100, 500]
        for winlen in winlens, noverlap in noverlaps
            subseq = Subsequence(x1, winlen, noverlap)
            sc1 = Score(FrequencyContours(fs, n, nv, tnorm, fd, minhprc, minfdist, mintlen), x1, winlen=winlen, noverlap=noverlap)
            sc2 = Score(FrequencyContours(fs, n, nv, tnorm, fd, minhprc, minfdist, mintlen), x2, winlen=winlen, noverlap=noverlap)
            spart1 = sc1[(sc1.axes[1] .> subseq.winlen÷2) .& (sc1.axes[1] .< length(x1)-subseq.winlen÷2)]
            spart2 = sc2[(sc2.axes[1] .> subseq.winlen÷2) .& (sc2.axes[1] .< length(x1)-subseq.winlen÷2)]
            @test all(isless.(spart2, spart1))
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
            subseq = Subsequence(dfile1, winlen, noverlap)
            sc1 = Score(FrequencyContours(fs, n, nv, tnorm, fd, minhprc, minfdist, mintlen), dfile1, winlen=winlen, noverlap=noverlap)
            sc2 = Score(FrequencyContours(fs, n, nv, tnorm, fd, minhprc, minfdist, mintlen), dfile2, winlen=winlen, noverlap=noverlap)
            spart1 = sc1[(sc1.axes[1] .> subseq.winlen÷2) .& (sc1.axes[1] .< length(dfile1)-subseq.winlen÷2)]
            spart2 = sc2[(sc2.axes[1] .> subseq.winlen÷2) .& (sc2.axes[1] .< length(dfile1)-subseq.winlen÷2)]
            @test all(isless.(spart2, spart1))
        end

        @test name(FrequencyContours(fs, n, nv, tnorm, fd, minhprc, minfdist, mintlen)) == ["Frequency Contours"]
    end

    @testset "SoundPressureLevel" begin
        @info "Testing SoundPressureLevel"

        x = A.*sin.(2π*frequency*t)
        x = pressure(x, 0.0, 0.0)
        sc = Score(SoundPressureLevel(), x)
        @test sc[1] ≈ 20*log10(1/sqrt(2))
        winlens = [1_000, 10_000, 1_001, 10_001]
        noverlaps = [0, 100, 500]
        for winlen in winlens, noverlap in noverlaps
            subseq = Subsequence(x, winlen, noverlap)
            sc = Score(SoundPressureLevel(), x; winlen=winlen, noverlap=noverlap)
            spart = sc[(sc.axes[1] .> subseq.winlen÷2) .& (sc.axes[1] .< length(x)-subseq.winlen÷2)]
            @test all(isapprox.(spart, repeat([20*log10(1/sqrt(2))], length(spart)), atol=0.01))
        end
        WAV.wavwrite(x[1:length(x)÷2], joinpath(tmpdir, "1.wav"), Fs=fs)
        WAV.wavwrite(x[length(x)÷2:end], joinpath(tmpdir, "2.wav"), Fs=fs)
        dfile = DistributedWAVFile(tmpdir)
        for winlen in winlens, noverlap in noverlaps
            subseq = Subsequence(dfile, winlen, noverlap)
            sc = Score(SoundPressureLevel(), dfile; winlen=winlen, noverlap=noverlap)
            spart = sc[(sc.axes[1] .> subseq.winlen÷2) .& (sc.axes[1] .< length(x)-subseq.winlen÷2)]
            @test all(isapprox.(spart, repeat([20*log10(1/sqrt(2))], length(spart)), atol=0.01))
        end

        @test name(SoundPressureLevel()) == ["SPL"]
    end

    @testset "ImpulseStats" begin
        @info "Testing ImpulseStats"

        trueindices = [201, 2254, 5322, 8888]
        truetimeintervals = diff(trueindices)
        Nᵢ = length(trueindices)
        μᵢᵢ = mean(truetimeintervals)/fs
        varᵢᵢ = var(truetimeintervals)/fs
        x = zeros(N)
        x[trueindices] .= 10.0
        x += 0.1 .* randn(N)

        sc1 = Score(ImpulseStats(fs), x)
        sc2 = Score(ImpulseStats(fs, 10, 1e-3), x)
        @test sc1[1, 1] == sc2[1, 1] == Nᵢ
        @test sc1[1, 2] == sc2[1, 2] == μᵢᵢ
        @test sc1[1, 3] == sc2[1, 3] == varᵢᵢ

        
        m = 100
        lpadlen, rpadlen = AcousticFeatures.getpadlen(m)
        x = zeros(N)
        template = randn(m)
        for trueindex ∈ trueindices
            x[trueindex-lpadlen:trueindex+rpadlen] = template
        end
        x += 0.1 .* randn(N)
        impulsestats = ImpulseStats(fs, 5, 1e-3, false, template)
        @test impulsestats.template == template
        sc3 = Score(impulsestats, x)
        @test sc3[1, 1] == Nᵢ
        @test sc3[1, 2] == μᵢᵢ
        @test sc3[1, 3] == varᵢᵢ

        @test name(ImpulseStats(fs)) == ["Nᵢ", "μᵢᵢ", "varᵢᵢ"]
    end

    @testset "SymmetricAlphaStableStats" begin
        @info "Testing SymmetricAlphaStableStats"

        α = 1.6
        scale = 2.0
        d = AlphaStable(α=α, scale=scale)
        x = rand(d, N)
        sc = Score(SymmetricAlphaStableStats(), x)
        @test sc[1,1] ≈ α atol=0.1
        @test sc[1,2] ≈ scale atol=0.1

        @test name(SymmetricAlphaStableStats()) == ["α", "scale"]
    end

    @testset "Entropy" begin
        @info "Testing Entropy"

        x = A.*sin.(2π*6250*t)
        sc = Score(Entropy(fs, 256, 128), x)
        @test sc[1] ≈ 1.0 atol=1e-2
        @test sc[2] ≈ 0.0 atol=1e-2
        @test sc[3] ≈ 0.0 atol=1e-2

        @test name(Entropy(fs, 256, 128)) == ["Temporal Entropy","Spectral Entropy","Entropy Index"]
    end

    @testset "ZeroCrossingRate" begin
        @info "Testing ZeroCrossingRate"
        x = [100.0, 1.0, -2.0, 2.0, -100, 0.0, 10.0]
        sc = Score(ZeroCrossingRate(), x)
        @test sc[1] == 4 / length(x)

        @test name(ZeroCrossingRate()) == ["ZCR"]
    end

    @testset "SpectralCentroid" begin
        @info "Testing SpectralCentroid"

        x = A.*sin.(2π*6250*t)
        sc = Score(SpectralCentroid(fs), x)
        @test sc[1] ≈ 6250 atol=0.0001

        @test name(SpectralCentroid(fs)) == ["Spectral Centroid"]
    end

    @testset "SpectralFlatness" begin
        @info "Testing SpectralFlatness"

        x = A.*sin.(2π*6250*t)
        sc = Score(SpectralFlatness(), x)
        @test sc[1] ≈ 0.0 atol=0.0001

        x = randn(N)
        scnormal = Score(SpectralFlatness(), x)
        @test scnormal[1] > sc[1]

        @test name(SpectralFlatness()) == ["Spectral Flatness"]
    end

    @testset "PermutationEntropy" begin
        @info "Testing PermutationEntropy"

        x = [4,7,9,10,6,11,3]
        m = 3
        τ1 = 1
        τ2 = 2
        norm1 = false
        norm2 = true
        weighted1 = false
        weighted2 = true

        sc111 = Score(PermutationEntropy(m, τ1, norm1, weighted1), x)
        sc121 = Score(PermutationEntropy(m, τ1, norm2, weighted1), x)
        sc131 = Score(PermutationEntropy(m), x)
        @test sc111[1] ≈ 1.5219 atol=0.0001
        @test sc121[1] ≈ 0.5887 atol=0.0001
        @test sc121[1] == sc131[1] 

        sc211 = Score(PermutationEntropy(m, τ2, norm1, weighted1), x)
        sc221 = Score(PermutationEntropy(m, τ2, norm2, weighted1), x)
        sc231 = Score(PermutationEntropy(m, τ2), x)
        @test sc211[1] ≈ 1.5850 atol=0.0001
        @test sc221[1] ≈ 0.6131 atol=0.0001
        @test sc221[1] == sc231[1] 

        sc112 = Score(PermutationEntropy(m, τ1, norm1, weighted2), x)
        sc122 = Score(PermutationEntropy(m, τ1, norm2, weighted2), x)
        @test sc112[1] ≈ 1.4140 atol=0.0001
        @test sc122[1] ≈ 0.5470 atol=0.0001

        sc212 = Score(PermutationEntropy(m, τ2, norm1, weighted2), x)
        sc222 = Score(PermutationEntropy(m, τ2, norm2, weighted2), x)
        @test sc212[1] ≈ 1.5233 atol=0.0001
        @test sc222[1] ≈ 0.5893 atol=0.0001

        @test name(PermutationEntropy(m)) == ["Permutation Entropy"]
    end

    @testset "PSD" begin
        @info "Testing PSD"

        freq = 3000
        fs = 96000
        x = cw(freq, 0.1, fs) |> real |> samples
        sc = Score(PSD(fs, 64, 32), x)
        @test sc.axes[2][argmax(sc)[2]] == "PSD-$(round(freq; digits=1))Hz"
    end

    @testset "AcousticComplexityIndex" begin
        @info "Testing AcousticComplexityIndex"

        freq1, freq2 = 1000, 48000
        fs = 96000
        x1 = chirp(freq1, freq2, 5.0, fs) |> real |> samples
        x2 = cw(freq1, 5.0, fs) |> real |> samples
        x1 += randn(length(x1))
        x2 += randn(length(x2))
        sc1 = Score(AcousticComplexityIndex(fs, 1048, 0, 30), x1)
        sc2 = Score(AcousticComplexityIndex(fs, 1048, 0, 30), x2)
        @test sc1[1] > sc2[1]
    end

    @testset "Score" begin
        @info "Testing Score"
        
        @test_throws ArgumentError Score(Energy(), randn(1000); winlen=1001)
        f = Energy()
        @test Score(f, randn(100000))[1] ≈ f(randn(100000))[1] atol=0.1
    end

    @testset "Subsequences" begin
        @info "Testing Subsequences"

        x = [1, 2, 3, 4, 5, 6, 7]
        tmpdir = mktempdir()
        a, b = [1, 2, 3, 4], [5, 6, 7]
        WAV.wavwrite(a, joinpath(tmpdir, "1.wav"), Fs=100)
        WAV.wavwrite(b, joinpath(tmpdir, "2.wav"), Fs=100)
        dfile = DistributedWAVFile(tmpdir)

        winlen = 3
        noverlap = 1
        subseq1 = [[0, 1, 2],
                    [2, 3, 4],
                    [4, 5, 6],
                    [6, 7, 0]]
        subseq2 = Subsequence(x, winlen, noverlap)
        subseq3 = Subsequence(dfile, winlen, noverlap)
        @test length(subseq1) == length(subseq2) == length(subseq3)
        for (s1, s2, s3) in zip(subseq1, subseq2, subseq3)
            @test s1 == s2 == s3
        end

        winlen = 3
        noverlap = 0
        subseq1 = [[0, 1, 2],
        [3, 4, 5],
        [6, 7, 0]]
        subseq2 = Subsequence(x, winlen, noverlap)
        subseq3 = Subsequence(dfile, winlen, noverlap)
        @test length(subseq1) == length(subseq2) == length(subseq3)
        for (i, (subseq1, subseq2, subseq3)) in enumerate(zip(subseq1, subseq2, subseq3))
            @test subseq1 == subseq2 == subseq3
            @test subseq1[i] == subseq2[i] == subseq3[i]
        end

        winlen = 4
        noverlap = 1
        subseq1 = [[0, 1, 2, 3],
                    [3, 4, 5, 6],
                    [6, 7, 0, 0]]
        subseq2 = Subsequence(x, winlen, noverlap)
        subseq3 = Subsequence(dfile, winlen, noverlap)
        @test length(subseq1) == length(subseq2) == length(subseq3)
        for (i, (subseq1, subseq2, subseq3)) in enumerate(zip(subseq1, subseq2, subseq3))
            @test subseq1 == subseq2 == subseq3
            @test subseq1[i] == subseq2[i] == subseq3[i]
        end

        winlen = 4
        noverlap = 1
        subseq1 = [[1, 1, 2, 3],
                    [3, 4, 5, 6],
                    [6, 7, 7, 7]]
        subseq2 = Subsequence(x, winlen, noverlap; padtype=:replicate)
        subseq3 = Subsequence(dfile, winlen, noverlap; padtype=:replicate)
        @test length(subseq1) == length(subseq2) == length(subseq3)
        for (i, (subseq1, subseq2, subseq3)) in enumerate(zip(subseq1, subseq2, subseq3))
            @test subseq1 == subseq2 == subseq3
            @test subseq1[i] == subseq2[i] == subseq3[i]
        end

        winlen = 4
        noverlap = 1
        subseq1 = [[7, 1, 2, 3],
                    [3, 4, 5, 6],
                    [6, 7, 1, 2]]
        subseq2 = Subsequence(x, winlen, noverlap; padtype=:circular)
        subseq3 = Subsequence(dfile, winlen, noverlap; padtype=:circular)
        @test length(subseq1) == length(subseq2) == length(subseq3)
        for (i, (subseq1, subseq2, subseq3)) in enumerate(zip(subseq1, subseq2, subseq3))
            @test subseq1 == subseq2 == subseq3
            @test subseq1[i] == subseq2[i] == subseq3[i]
        end

        winlen = 4
        noverlap = 1
        subseq1 = [[1, 1, 2, 3],
                    [3, 4, 5, 6],
                    [6, 7, 7, 6]]
        subseq2 = Subsequence(x, winlen, noverlap; padtype=:symmetric)
        subseq3 = Subsequence(dfile, winlen, noverlap; padtype=:symmetric)
        @test length(subseq1) == length(subseq2) == length(subseq3)
        for (i, (subseq1, subseq2, subseq3)) in enumerate(zip(subseq1, subseq2, subseq3))
            @test subseq1 == subseq2 == subseq3
            @test subseq1[i] == subseq2[i] == subseq3[i]
        end

        winlen = 4
        noverlap = 1
        subseq1 = [[2, 1, 2, 3],
                    [3, 4, 5, 6],
                    [6, 7, 6, 5]]
        subseq2 = Subsequence(x, winlen, noverlap; padtype=:reflect)
        subseq3 = Subsequence(dfile, winlen, noverlap; padtype=:reflect)
        @test length(subseq1) == length(subseq2) == length(subseq3)
        for (i, (subseq1, subseq2, subseq3)) in enumerate(zip(subseq1, subseq2, subseq3))
            @test subseq1 == subseq2 == subseq3
            @test subseq1[i] == subseq2[i] == subseq3[i]
        end

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
        xvolt = vref.*real(samples(cw(64, 1, 512)))
        xbit = xvolt*(2^(nbits-1))
        sensitivity = 0.0
        gain = 0.0
        p1 = pressure(xvolt, sensitivity, gain)
        p2 = pressure(xbit, sensitivity, gain, voltparams=(nbits, vref))
        @test p1 == p2

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

    @testset "Benchmarks" begin
        @info "Benchmarks"

        path = mktempdir()
        y = sin.((0:99999999)/48000*2pi*440);
        wavwrite(y, joinpath(path, "test1.wav"), Fs=48000)

        dfile = DistributedWAVFile(path)
        subseqdf = Subsequence(dfile, 96000, 0)

        t = @belapsed $subseqdf[1]
        @test t < 0.1
        t = @belapsed $subseqdf[10]
        @test t < 0.1
        t = @belapsed $subseqdf[100]
        @test t < 0.1

        filepath = joinpath(path, "test1.wav")
        lfile = LazyWAVFile(filepath)
        subseqlf = Subsequence(lfile, 96000, 0)

        t = @belapsed $subseqlf[1]
        @test t < 0.01
        t = @belapsed $subseqlf[10]
        @test t < 0.01
        t = @belapsed $subseqlf[100]
        @test t < 0.01
    end

end
