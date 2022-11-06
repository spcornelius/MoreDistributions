const Γ = gamma
const ψ₀ = digamma

struct GeneralizedNormal{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    α::T
    β::T

    function GeneralizedNormal{T}(μ, α, β) where {T}
        new{T}(μ, α, β)
    end
end

#### Outer constructors
function GeneralizedNormal(μ::T, α::T, β::T; check_args::Bool=true) where {T<:Real}
    @check_args GeneralizedNormal (α, α > zero(α)) (β, β > zero(β))
    return GeneralizedNormal{T}(μ, α, β)
end

GeneralizedNormal(μ::Real, α::Real, β::Real) = GeneralizedNormal(promote(μ, α, β)...)
GeneralizedNormal(μ::Integer, α::Integer, β::Integer) = GeneralizedNormal(float(μ), float(α), float(β))

# standard GeneralizedNormal (mean 0, variance 1) with shape parameter β
GeneralizedNormal(β::T) where {T <: Real} = GeneralizedNormal(zero(T), sqrt(Γ(1/β)/Γ(3/β)), β)

# standard normal
GeneralizedNormal() = GeneralizedNormal(2.0)

# #### Conversions
convert(::Type{GeneralizedNormal{T}}, μ::S, α::S, β::S) where 
    {T <: Real, S <: Real} = GeneralizedNormal(T(μ), T(α), T(β))
convert(::Type{GeneralizedNormal{T}}, d::GeneralizedNormal{S}) where 
    {T <: Real, S <: Real} = GeneralizedNormal(T(d.μ), T(d.α), T(d.β))

@distr_support GeneralizedNormal -Inf Inf

params(d::GeneralizedNormal) = (d.μ, d.α, d.β)
@inline partype(::GeneralizedNormal{T}) where {T<:Real} = T

location(d::GeneralizedNormal) = d.μ
scale(d::GeneralizedNormal) = d.α
shape(d::GeneralizedNormal) = d.β

#### Statistics
mean(d::GeneralizedNormal) = d.μ
median(d::GeneralizedNormal) = d.μ
mode(d::GeneralizedNormal) = d.μ

var(d::GeneralizedNormal) = d.α^2*Γ(3/d.β)/Γ(1/d.β)

skewness(::GeneralizedNormal{T}) where {T<:Real} = zero(T)
kurtosis(d::GeneralizedNormal) = Γ(5/d.β)*Γ(1/d.β)/Γ(3/d.β)^2 - 3

isleptokurtic(d::GeneralizedNormal) = d.β < 2
isplatykurtic(d::GeneralizedNormal) = d.β > 2
ismesokurtic(d::GeneralizedNormal) = d.β == 2

entropy(d::GeneralizedNormal) = 1/d.β - log(d.β/(2*d.α*Γ(1/d.β)))

#### Evaluation

function pdf(d::GeneralizedNormal, x::Real)
    (μ, α, β) = params(d)
    β/(2*α*Γ(1/β)) * exp(-(abs(x-μ)/α)^β)
end

function logpdf(d::GeneralizedNormal, x::Real)
    (μ, α, β) = params(d)
    log(β/2α) - loggamma(1/β) - (abs(x-μ)/α)^β
end

function cdf(d::GeneralizedNormal, x::Real)
    (μ, α, β) = params(d)
    0.5 + sign(x-μ) * 0.5*gamma_inc(1/β, (abs(x-μ)/α)^β)[1]
end

function quantile(d::GeneralizedNormal, p::Real)
    (μ, α, β) = params(d)
    if p == 0.5
        return μ
    end
    c = sign(p - 0.5)
    y = 2.0*abs(p - 0.5)
    μ + c * α * gamma_inc_inv(1/β, y, 1-y)^(1/β)
end

struct GeneralizedNormalSampler{S<:Sampleable{Univariate,Continuous}, 
                                T<:Real} <: Sampleable{Univariate,Continuous}
    gs::S # gamma sampler
    β::T
    μ::T
end

function sampler(d::GeneralizedNormal)
    gs = sampler(Gamma(1/d.β, d.α^d.β))
    return GeneralizedNormalSampler(gs, d.β, d.μ)
end

function rand(rng::AbstractRNG, s::GeneralizedNormalSampler)
    y = rand(rng, s.gs)^(1/s.β)
    pm = ifelse(rand(rng, Bool), 1, -1)
    return s.μ + pm*y
end

function rand(rng::AbstractRNG, d::GeneralizedNormal)
    return rand(rng, sampler(d))
end

# range of β for lookup table
const β_MIN = 1.0e-6
const β_MAX = 100.0
const dβ = 0.001

# Lookup table for initial guess of shape parameter β.
const β₀_range = collect(β_MIN:dβ:β_MAX)
const outputs = map(β -> loggamma(5/β) + loggamma(1/β) - 2*loggamma(3/β),
                    β₀_range)
const β_lookup_tbl = hcat(β₀_range, outputs)[sortperm(outputs), :]

function guess_initial_params(x::AbstractVector{<:Real})
    n = length(x)
    # following Varanasi & Aazhang (1989)

    # initial guesses for variance and mean (μ) and scale (α) come directly
    # from  1st & 2nd (central) sample moments
    μ, σ² = mean_and_var(x)

    # Quantity resulting from moment-matching on 4th central moment, and
    # taking a log.
    c = log(moment(x, 4)/σ²^2)

    # ...c must be equal to loggamma(5/β) + loggamma(1/β) - 2*loggamma(3/β).
    # Invert to find β via precomputed lookup table.
    i = searchsortedfirst(view(β_lookup_tbl, :, 2), c)
    β = i > length(β_lookup_tbl) ? β_MAX : β_lookup_tbl[i, 1]

    # relation between scale parameter α and variance of the 
    # GeneralizedNormal
    α = sqrt(σ²*Γ(1/β)/Γ(3/β))

    # work in log-space for the parameters α, β
    # that way, all optimization params are unbounded
    return [μ, log(α), log(β)]
end

function fit_mle(::Type{<:GeneralizedNormal}, x::AbstractVector{T};
                 alg = DEFAULT_NLOPT_ALG, kwargs...) where {T <: Real}

    # Step 1: Get inital guesses for all parameters using moment matching
    p₀ = guess_initial_params(x)

    y = similar(x)
    yβ = similar(x)

    function objective(p, grad)
        μ, log_α, log_β = p
        α = exp(log_α)
        β = exp(log_β)
        β⁻¹ = 1/β
        
        @. y = abs(x - μ)/α
        @. yβ = y^β
        mean_yβ = mean(yβ)

        obj = log(β/2α) - loggamma(1/β) - mean_yβ

        if length(grad) > 0
            grad[1] = β/α * mean(@~ @. sign(x - μ) * y^(β-1))

            # extra factors of α, and β in these are to correct for the fact that
            # we're working in log-space
            grad[2] = α * (β/α * mean_yβ - 1/α)
            grad[3] = β * (β⁻¹*(β⁻¹*ψ₀(β⁻¹) + 1) - mean(@~ @. yβ * log(y)))
        end

        # (normalized) log-likelihood
        return obj
    end

    # Step 2: Polish initial estimates by numerically maximizing log-likelihood
    opt = Opt(alg, 3)
    opt.max_objective = objective
    params = merge(DEFAULT_NLOPT_OPTIONS, kwargs)
    for (k, v) in pairs(params)
        setproperty!(opt, Symbol(k), v)
    end

    _, p, ret = optimize(opt, p₀)
    if ret ∉ (:SUCCESS, :XTOL_REACHED, :FTOL_REACHED)
        error("Numerical optimization failed.")
    end

    μ = p[1]
    α = exp(p[2])
    β = exp(p[3])
    GeneralizedNormal{T}(μ, α, β)
end # fit_mle
