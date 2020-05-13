using AcousticFeatures

using AlphaStableDistributions, Distributions, LazyWAVFiles, LinearAlgebra, SignalAnalysis, Test, WAV

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
        @test Score(Energy(), x).s[1] ≈ (A^2)/2
        winlens = [1_000, 10_000, 1_001, 10_001]
        noverlaps = [0, 100, 500]
        for winlen in winlens, noverlap in noverlaps
            subseq = Subsequence(x, winlen, noverlap)
            sc = Score(Energy(), x; winlen=winlen, noverlap=noverlap)
            spart = sc.s[(sc.indices .> subseq.winlen÷2) .& (sc.indices .< length(x)-subseq.winlen÷2)]
            @test all(isapprox.(spart, repeat([(A^2)/2], length(spart)), atol=0.001))
        end
        WAV.wavwrite(x[1:N÷2], joinpath(tmpdir, "1.wav"), Fs=fs)
        WAV.wavwrite(x[N÷2:end], joinpath(tmpdir, "2.wav"), Fs=fs)
        dfile = DistributedWAVFile(tmpdir)
        for winlen in winlens, noverlap in noverlaps
            subseq = Subsequence(dfile, winlen, noverlap)
            sc = Score(Energy(), dfile; winlen=winlen, noverlap=noverlap)
            spart = sc.s[(sc.indices .> subseq.winlen÷2) .& (sc.indices .< length(x)-subseq.winlen÷2)]
            @test all(isapprox.(spart, repeat([(A^2)/2], length(spart)), atol=0.001))
        end
    end

    @testset "Myriad" begin
        @info "Testing Myriad"

        α = 1.9999
        scale = 1.0
        x = rand(AlphaStable(α=α, scale=scale), N)
        d = fit(AlphaStable, x)
        sqKscale = myriadconstant(d.α, d.scale)
        @test Score(Myriad(), x).s[1]/N ≈ (log((d.α/(2-d.α+eps()))*(d.scale^2))) atol=0.1
        winlens = [1_000, 10_000, 1_001, 10_001]
        noverlaps = [0, 100, 500]
        for winlen in winlens, noverlap in noverlaps
            subseq = Subsequence(x, winlen, noverlap)
            sc = Score(Myriad(sqKscale), x; winlen=winlen, noverlap=noverlap)
            spart = sc.s[(sc.indices .> subseq.winlen÷2) .& (sc.indices .< length(x)-subseq.winlen÷2)]
            @test all(isapprox.(spart./subseq.winlen, repeat([(log((d.α/(2-d.α+eps()))*(d.scale^2)))], length(spart)), atol=0.1))
        end
        WAV.wavwrite(x[1:N÷2], joinpath(tmpdir, "1.wav"), Fs=fs)
        WAV.wavwrite(x[N÷2:end], joinpath(tmpdir, "2.wav"), Fs=fs)
        dfile = DistributedWAVFile(tmpdir)
        for winlen in winlens, noverlap in noverlaps
            subseq = Subsequence(dfile, winlen, noverlap)
            sc = Score(Myriad(sqKscale), dfile; winlen=winlen, noverlap=noverlap)
            spart = sc.s[(sc.indices .> subseq.winlen÷2) .& (sc.indices .< length(x)-subseq.winlen÷2)]
            @test all(isapprox.(spart./subseq.winlen, repeat([(log((d.α/(2-d.α+eps()))*(d.scale^2)))], length(spart)), atol=0.1))
        end
    end

    @testset "VMyriad" begin
        @info "Testing VMyriad"

        # α = 1.3
        # identitymatrix = zeros(5, 5)
        # identitymatrix[diagind(identitymatrix)] .= 1.0
        # d = AlphaSubGaussian(α=α, n=N)
        # x = rand(d)
        # d̂ = fit(AlphaSubGaussian, x, 4)

        α = 1.3
        d = AlphaSubGaussian(;α=α, n=N)
        x = rand(d)
        d̂ = fit(AlphaSubGaussian, x, 4)

        f0 = VMyriad(vmyriadconstant(d̂.α, d̂.R)...)
        f1 = VMyriad(vmyriadconstant(1.8, d̂.R)...)
        identitymatrix = zeros(5, 5)
        identitymatrix[diagind(identitymatrix)] .= 1.0
        f2 = VMyriad(vmyriadconstant(1.8, identitymatrix)...)

        @test Score(f0, x).s[1] ≈ Score(VMyriad(vmyriadconstant(d̂.α, d̂.R)...), x).s[1]
        @test Score(f0, x).s[1] < Score(f1, x).s[1]
        @test Score(f0, x).s[1] < Score(f2, x).s[1]
    end

    @testset "FrequencyContours" begin
        @info "Testing FrequencyContours"

        duration = N/fs
        f11 = 10_000; f21 = 50_000
        f12 = 1_000; f22 = 20_000
        x1 = real(chirp(f11, f21, duration, fs).data)+real(chirp(f12, f22, duration, fs).data)
        x2 = real(chirp(f11, f21, duration, fs).data)
        n = 512
        tnorm = 1.0
        fd = 1000.0
        minhprc = 99.0
        minfdist = 1000.0
        mintlen = 0.05
        sc1 = Score(FrequencyContours(fs, n, tnorm, fd, minhprc, minfdist, mintlen), x1)
        sc2 = Score(FrequencyContours(fs, n, tnorm, fd, minhprc, minfdist, mintlen), x2)
        @test sc1.s[1] > sc2.s[1]
        winlens = [10_000, 10_001]
        noverlaps = [0, 100, 500]
        for winlen in winlens, noverlap in noverlaps
            subseq = Subsequence(x1, winlen, noverlap)
            sc1 = Score(FrequencyContours(fs, n, tnorm, fd, minhprc, minfdist, mintlen), x1, winlen=winlen, noverlap=noverlap)
            sc2 = Score(FrequencyContours(fs, n, tnorm, fd, minhprc, minfdist, mintlen), x2, winlen=winlen, noverlap=noverlap)
            spart1 = sc1.s[(sc1.indices .> subseq.winlen÷2) .& (sc1.indices .< length(x1)-subseq.winlen÷2)]
            spart2 = sc2.s[(sc2.indices .> subseq.winlen÷2) .& (sc2.indices .< length(x1)-subseq.winlen÷2)]
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
            sc1 = Score(FrequencyContours(fs, n, tnorm, fd, minhprc, minfdist, mintlen), dfile1, winlen=winlen, noverlap=noverlap)
            sc2 = Score(FrequencyContours(fs, n, tnorm, fd, minhprc, minfdist, mintlen), dfile2, winlen=winlen, noverlap=noverlap)
            spart1 = sc1.s[(sc1.indices .> subseq.winlen÷2) .& (sc1.indices .< length(dfile1)-subseq.winlen÷2)]
            spart2 = sc2.s[(sc2.indices .> subseq.winlen÷2) .& (sc2.indices .< length(dfile1)-subseq.winlen÷2)]
            @test all(isless.(spart2, spart1))
        end
    end

    @testset "SoundPressureLevel" begin
        @info "Testing SoundPressureLevel"

        x = A.*sin.(2π*frequency*t)
        x = pressure(x, 0.0, 0.0)
        sc = Score(SoundPressureLevel(), x)
        @test sc.s[1] ≈ 20*log10(1/sqrt(2))
        winlens = [1_000, 10_000, 1_001, 10_001]
        noverlaps = [0, 100, 500]
        for winlen in winlens, noverlap in noverlaps
            subseq = Subsequence(x, winlen, noverlap)
            sc = Score(SoundPressureLevel(), x; winlen=winlen, noverlap=noverlap)
            spart = sc.s[(sc.indices .> subseq.winlen÷2) .& (sc.indices .< length(x)-subseq.winlen÷2)]
            @test all(isapprox.(spart, repeat([20*log10(1/sqrt(2))], length(spart)), atol=0.01))
        end
        WAV.wavwrite(x[1:length(x)÷2], joinpath(tmpdir, "1.wav"), Fs=fs)
        WAV.wavwrite(x[length(x)÷2:end], joinpath(tmpdir, "2.wav"), Fs=fs)
        dfile = DistributedWAVFile(tmpdir)
        for winlen in winlens, noverlap in noverlaps
            subseq = Subsequence(dfile, winlen, noverlap)
            sc = Score(SoundPressureLevel(), dfile; winlen=winlen, noverlap=noverlap)
            spart = sc.s[(sc.indices .> subseq.winlen÷2) .& (sc.indices .< length(x)-subseq.winlen÷2)]
            @test all(isapprox.(spart, repeat([20*log10(1/sqrt(2))], length(spart)), atol=0.01))
        end
    end

    @testset "ImpulseStats" begin
        @info "Testing ImpulseStats"

        trueindices = [101, 2254, 5322, 8888]
        x = zeros(N)
        x[trueindices] .= 1.0
        x += 0.1 .* randn(N)
        sc = Score(ImpulseStats(fs), x)
        @test sc.s[1, 1] == length(trueindices)
        truetimeintervals = diff(trueindices)
        @test sc.s[1, 2] == mean(truetimeintervals)/fs
        @test sc.s[1, 3] == var(truetimeintervals)/fs
    end

    @testset "AlphaStableStats" begin
        @info "Testing AlphaStableStats"

        α = 1.6
        scale = 2.0
        d = AlphaStable(α=α, scale=scale)
        x = rand(d, N)
        sc = Score(AlphaStableStats(), x).s[1, :]
        @test sc[1] ≈ α atol=0.1
        @test sc[2] ≈ scale atol=0.1

    end

    @testset "Subsequences" begin
        x = [1, 2, 3, 4, 5, 6, 7]
        tmpdir = mktempdir()
        a, b = [1, 2, 3, 4], [5, 6, 7]
        WAV.wavwrite(a, joinpath(tmpdir, "1.wav"), Fs=100)
        WAV.wavwrite(b, joinpath(tmpdir, "2.wav"), Fs=100)
        dfile = DistributedWAVFile(tmpdir)

        winlen = 3
        noverlap = 1
        subseqs1 = [[0, 1, 2],
        [2, 3, 4],
        [4, 5, 6],
        [6, 7, 0]]
        subseqs2 = Subsequence(x, winlen, noverlap)
        subseqs3 = Subsequence(dfile, winlen, noverlap)
        @test length(subseqs1) == length(subseqs2) == length(subseqs3)
        for (i, (subseq1, subseq2, subseq3)) in enumerate(zip(subseqs1, subseqs2, subseqs3))
            @test subseq1 == subseq2 == subseq3
            @test subseqs1[i] == subseqs2[i] == subseqs3[i]
        end

        winlen = 3
        noverlap = 0
        subseqs1 = [[0, 1, 2],
        [3, 4, 5],
        [6, 7, 0]]
        subseqs2 = Subsequence(x, winlen, noverlap)
        subseqs3 = Subsequence(dfile, winlen, noverlap)
        @test length(subseqs1) == length(subseqs2) == length(subseqs3)
        for (i, (subseq1, subseq2, subseq3)) in enumerate(zip(subseqs1, subseqs2, subseqs3))
            @test subseq1 == subseq2 == subseq3
            @test subseqs1[i] == subseqs2[i] == subseqs3[i]
        end

        winlen = 4
        noverlap = 1
        subseqs1 = [[0, 1, 2, 3],
                    [3, 4, 5, 6],
                    [6, 7, 0, 0]]
        subseqs2 = Subsequence(x, winlen, noverlap)
        subseqs3 = Subsequence(dfile, winlen, noverlap)
        @test length(subseqs1) == length(subseqs2) == length(subseqs3)
        for (i, (subseq1, subseq2, subseq3)) in enumerate(zip(subseqs1, subseqs2, subseqs3))
            @test subseq1 == subseq2 == subseq3
            @test subseqs1[i] == subseqs2[i] == subseqs3[i]
        end

    end

    @testset "Utils" begin
        x = [1, 2, 3, 4, 5, 6, 7]
        Nnorm = 3
        xfilt = x - [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0]
        @test spectrumflatten(x, Nnorm) == xfilt

        x = [[1 2 3 4 5 6 7];
        [8 9 10 11 12 13 14]]
        xfilt = x - [[1.0 2.0 3.0 4.0 5.0 6.0 6.0]; [8.0 9.0 10.0 11.0 12.0 13.0 13.0]]
        @test spectrumflatten(x, Nnorm) == xfilt

        nbits = 16
        vref = 1.0
        xvolt = vref.*real(cw(64, 1, 512).data)
        xbit = xvolt*(2^(nbits-1))
        sensitivity = 0.0
        gain = 0.0
        p1 = pressure(xvolt, sensitivity, gain)
        p2 = pressure(xbit, sensitivity, gain, voltparams=(nbits, vref))
        @test p1 == p2
    end

end
