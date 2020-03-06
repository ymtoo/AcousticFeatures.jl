using AcousticFeatures

using AcousticFeatures.Utils

using AlphaStableDistributions, Distributions, LazyWAVFiles, Test, WAV
tmpdir = mktempdir()
fs = 100_000
A = 1.0
frequency = 10_000
N = 100_000
t = (0:N-1)./fs
x = A.*sin.(2π*frequency*t)

@testset "AcousticFeatures" begin



    @testset "Energy" begin
        @info "Testing Energy"

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
        N = 100_000
        x = rand(AlphaStable(α=α, scale=scale), N)
        N = length(x)
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

    @testset "FrequencyContours" begin
        @info "Testing FrequencyContours"


        fs = 100_000
        N = 100_000
        duration = N/fs
        f11 = 10_000; f21 = 50_000
        f12 = 1_000; f22 = 20_000
        x1 = chirp(f11, f21, duration, fs)+chirp(f12, f22, duration, fs)
        x2 = chirp(f11, f21, duration, fs)
        n = 512
        tnorm = 1.0
        fd = 1000.0
        minhprc = 99.0
        minfdist = 1000.0
        mintlen = 0.05
        sc1 = Score(FrequencyContours(fs, n, tnorm, fd, minhprc, minfdist, mintlen), x1)
        sc2 = Score(FrequencyContours(fs, n, tnorm, fd, minhprc, minfdist, mintlen), x2)
        @test sc1.s > sc2.s
        winlens = [10_000, 10_001]
        noverlaps = [0, 100, 500]
        for winlen in winlens, noverlap in noverlaps
            subseq = Subsequence(x, winlen, noverlap)
            sc1 = Score(FrequencyContours(fs, n, tnorm, fd, minhprc, minfdist, mintlen), x1, winlen=winlen, noverlap=noverlap)
            sc2 = Score(FrequencyContours(fs, n, tnorm, fd, minhprc, minfdist, mintlen), x2, winlen=winlen, noverlap=noverlap)
            spart1 = sc1.s[(sc1.indices .> subseq.winlen÷2) .& (sc1.indices .< length(x)-subseq.winlen÷2)]
            spart2 = sc2.s[(sc2.indices .> subseq.winlen÷2) .& (sc2.indices .< length(x)-subseq.winlen÷2)]
            @test all(isless.(spart2, spart1))
        end
        WAV.wavwrite(x[1:N÷2], joinpath(tmpdir, "1.wav"), Fs=fs)
        WAV.wavwrite(x[N÷2:end], joinpath(tmpdir, "2.wav"), Fs=fs)
        dfile = DistributedWAVFile(tmpdir)
        for winlen in winlens, noverlap in noverlaps
            subseq = Subsequence(dfile, winlen, noverlap)
            sc1 = Score(FrequencyContours(fs, n, tnorm, fd, minhprc, minfdist, mintlen), x1, winlen=winlen, noverlap=noverlap)
            sc2 = Score(FrequencyContours(fs, n, tnorm, fd, minhprc, minfdist, mintlen), x2, winlen=winlen, noverlap=noverlap)
            spart1 = sc1.s[(sc1.indices .> subseq.winlen÷2) .& (sc1.indices .< length(x)-subseq.winlen÷2)]
            spart2 = sc2.s[(sc2.indices .> subseq.winlen÷2) .& (sc2.indices .< length(x)-subseq.winlen÷2)]
            @test all(isless.(spart2, spart1))
        end
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
        for i in 1:length(subseqs1)

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
        subseqs1 = [[0, 0, 1, 2, 3],
        [3, 4, 5, 6, 7]]
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

    end

end
