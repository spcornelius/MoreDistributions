using Test
using Distributions
using MoreDistributions
using ForwardDiff
using SpecialFunctions

@testset "GeneralizedNormal" begin
    μ = -1.2

    for β in [0.7, 1.5, 2.3]
        d = GeneralizedNormal(μ, sqrt(2), β)

        @test μ === mode(d)
        @test μ === mean(d)
        @test μ === median(d)

        @test μ === quantile(d, 0.5)
        @test Inf === quantile(d, 1)
        @test -Inf === quantile(d, 0)

        @test μ === cquantile(d, 0.5)
        @test -Inf === cquantile(d, 1)
        @test Inf === cquantile(d, 0)

        @test 0.5 === cdf(d, μ)
        @test 1.0 === cdf(d, Inf)
        @test 0.0 === cdf(d, -Inf)
    end
end

@testset "GeneralizedNormal β = 2" begin
    μ = 5.4
    σ = 1.4
    d_normal = Normal(μ, σ)
    d_gn = GeneralizedNormal(μ, σ * sqrt(gamma(1/2)/gamma(3/2)), 2.0)

    x = rand(d_normal)
    p = cdf(d_normal, x)

    @test pdf(d_gn, x) === pdf(d_normal, x)
    @test isapprox(cdf(d_gn, x), p)
    @test isapprox(quantile(d_gn, p), x)
    @test isapprox(cquantile(d_gn, 1 - p), x)
    
    @test isapprox(std(d_gn), σ)
    @test ismesokurtic(d_gn)
end

@testset "GeneralizedNormal β = 1" begin
    μ = -0.5
    b =  3.9
    d_laplace = Laplace(μ, b)
    d_gn = GeneralizedNormal(μ, b, 1.0)

    x = rand(d_laplace)
    p = cdf(d_laplace, x)

    @test pdf(d_gn, x) === pdf(d_laplace, x)
    @test isapprox(cdf(d_gn, x), p)
    @test isapprox(quantile(d_gn, p), x)
    @test isapprox(cquantile(d_gn, 1 - p), x)
    
    @test isapprox(std(d_gn), sqrt(2) * b)
    @test isleptokurtic(d_gn)
end

@testset "GeneralizedNormal MLE fits work" begin
    x1 = rand(Normal(), 1000)
    x2 = rand(Laplace(), 1000)
    x3 = rand(Uniform(), 1000)
    x4 = rand(GeneralizedNormal(-2.0, 1.0, 0.6), 1000)
    
    x = (x1, x2, x3, x4)
    algs = (:LN_NEWUOA, :LN_NELDERMEAD, :LN_COBYLA, :LN_PRAXIS)

    for x in x, alg in algs
        @test isa(fit_mle(GeneralizedNormal, x, alg=alg), GeneralizedNormal)
    end
end

@testset "GeneralizedNormal MLE fits alg consistency" begin
    x = rand(GeneralizedNormal(4.6, 2.0, 0.55), 10^5)

    algs = (:LN_NEWUOA, :LN_NELDERMEAD, :LN_COBYLA, :LN_PRAXIS)
    @time p = collect((params(fit_mle(GeneralizedNormal, x, alg=alg)) for alg in algs))
    μ, α, β = collect(zip(p...))
    
    @test all(isapprox.(maximum(μ), minimum(μ), rtol=1e-3))
    @test all(isapprox.(maximum(α), minimum(α), rtol=1e-2))
    @test all(isapprox.(maximum(β), minimum(β), rtol=1e-2))
end