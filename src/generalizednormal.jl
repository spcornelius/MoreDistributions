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
convert(::Type{GeneralizedNormal{T}}, μ::S, α::S, β::S) where {T <: Real, S <: Real} = GeneralizedNormal(T(μ), T(α), T(β))
convert(::Type{GeneralizedNormal{T}}, d::GeneralizedNormal{S}) where {T <: Real, S <: Real} = GeneralizedNormal(T(d.μ), T(d.α), T(d.β))

@distr_support GeneralizedNormal -Inf Inf

params(d::GeneralizedNormal) = (d.μ, d.α, d.β)
@inline partype(d::GeneralizedNormal{T}) where {T<:Real} = T

location(d::GeneralizedNormal) = d.μ
scale(d::GeneralizedNormal) = d.α
shape(d::GeneralizedNormal) = d.β

#### Statistics
mean(d::GeneralizedNormal) = d.μ
median(d::GeneralizedNormal) = d.μ
mode(d::GeneralizedNormal) = d.μ

var(d::GeneralizedNormal) = d.α^2*Γ(3/d.β)/Γ(1/d.β)

skewness(d::GeneralizedNormal{T}) where {T<:Real} = zero(T)
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

struct GeneralizedNormalSampler{S<:Sampleable{Univariate,Continuous}, T<:Real} <: Sampleable{Univariate,Continuous}
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

const α_MIN = 1.0e-16
# const α_MAX = 1.0e8
const β_MIN = 1.0e-6
const β_MAX = 100.0
const dβ = 0.001

# Lookup table for initial guess of shape parameter β.
# Used in fit_mle.
const β₀_range = collect(β_MIN:dβ:β_MAX)
const outputs = map(β -> loggamma(5/β) + loggamma(1/β) - 2*loggamma(3/β),
                    β₀_range)
const lookup_tbl = hcat(β₀_range, outputs)[sortperm(outputs), :]

# gradient of log likelihood function w.r.t. parameters Θ, for the sample x
# @inbounds function update_ℒ′!(ℒ′, x, μ, α, β)
#     β⁻¹ = 1/β
#     n = length(x)
#
#     #y = x[x .!= μ]
#     ℒ′[1] = β/α^β * sum(@~ @. sign(x - μ)*abs(x - μ)^(β-1))
#     ℒ′[2] = β/(α^(β+1)) * sum(@~  @. abs(x - μ)^β) - n/α
#     ℒ′[3] = n*β⁻¹*(β⁻¹*ψ₀(β⁻¹) + 1) -
#                 sum(@~ @. (abs(x - μ)/α)^β*log(abs(x - μ)/α))
#     return ℒ′
# end

# expected Fisher information matrix
# @inbounds function update_ℐ!(ℐ, μ, α, β)
#     β⁻¹ = 1/β
#
#     ###################
#     # diagonal elements
#     ###################
#
#     # Fill ℐ_μ with a dummy value of 1 if β ≤ 1, as Γ(2-β⁻¹)
#     # can't be evaluated at β = 1. Doesn't matter anyway, as
#     # μ is not updated via Newton iteration when β ≤ 1.
#     ℐ_μ = β > 1 ? β^2/(Γ(β⁻¹)*α^2)*Γ(2-β⁻¹) : 1.0
#     ℐ_α = β/α^2
#
#     # diagonal element for β is a long expression; split calculation
#     c1 = β⁻¹*(β⁻¹*ψ₀(β⁻¹) + 1)
#     c2 = β⁻¹^2*ψ₀(1 + β⁻¹)
#     c3 = β⁻¹^2* Γ(2 + β⁻¹)/Γ(β⁻¹) * (ψ₀(2 + β⁻¹)^2 + ψ₁(2 + β⁻¹))
#
#     ℐ_β = c1^2 - 2*c1*c2 + c3
#
#     # lone nonzero off-diagonal element
#     ℐ_αβ = β⁻¹^2/α*ψ₀(1 + β⁻¹) -
#            β⁻¹/α*(1 + β⁻¹)*ψ₀(2 + β⁻¹)
#
#     fill!(ℐ, 0.0)
#     ℐ[1, 1] = ℐ_μ
#     ℐ[2, 2] = ℐ_α
#     ℐ[3, 3] = ℐ_β
#     ℐ[2, 3] = ℐ[3, 2] = ℐ_αβ
#     return ℐ
# end

function guess_initial_params(x::AbstractVector{T}) where {T <: Real}
    n = length(x)
    # following Varanasi & Aazhang (1989)

    # Step 1: Get inital guesses for all parameters using moment matching

    # initial guesses for variance and mean (μ) and scale (α) come directly
    # from  1st & 2nd (central) sample moments
    μ, σ² = mean_and_var(x)

    # Quantity resulting from moment-matching on 4th central moment, and
    # taking a log.
    c = log(moment(x, 4)/σ²^2)

    # Must be equal to loggamma(5/β) + loggamma(1/β) - 2*loggamma(3/β).
    # Use to find β₀ via precomputed lookup table.
    i = searchsortedfirst(view(lookup_tbl, :, 2), c)
    if i > length(lookup_tbl)
        β = β_MAX
    else
        β = lookup_tbl[i, 1]
    end

    # relation between scale parameter α and variance
    α = sqrt(σ²*Γ(1/β)/Γ(3/β))

    return [μ, α, β]
end

# gradient of (normalized) log-likelihood w.r.t. the parameters
# μ, log_α, and log_β
# function update_ℒ′!(ℒ′, x, μ, log_α, log_β)
#     α = exp(log_α)
#     β = exp(log_β)

#     β⁻¹ = 1/β
#     n = length(x)

#     ℒ′[1] = β/α^β * sum(@~ @. sign(x - μ)*abs(x - μ)^(β-1))
#     ℒ′[2] = exp(α) * (β/(α^(β+1)) * sum(@~  @. abs(x - μ)^β) - n/α)
#     ℒ′[3] = exp(β) * (n*β⁻¹*(β⁻¹*ψ₀(β⁻¹) + 1) -
#              sum(@~ @. (abs(x - μ)/α)^β*log(abs(x - μ)/α)))
#     ℒ′ ./= n
#     return ℒ′
# end

function fit_mle(::Type{<:GeneralizedNormal}, x::AbstractVector{T};
                 alg = :LN_NEWUOA,
                 reltol::Real = 1.0e-6) where {T <: Real}
    n = length(x)

    # Step 1: Get inital guesses for all parameters using moment matching
    p₀ = guess_initial_params(x)

    # work in log-space for the parameters α, β
    # that way, all parameters are unbounded (between -Inf and +Inf)
    p₀[2:3] .= log.(p₀[2:3])

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

    opt = Opt(alg, 3)
    opt.max_objective = objective
    opt.xtol_rel = reltol

    _, p, ret = optimize(opt, p₀)
    if ret ∉ (:SUCCESS, :XTOL_REACHED, :FTOL_REACHED)
        error("Numerical optimization failed.")
    end

    μ = p[1]
    α = exp(p[2])
    β = exp(p[3])
    GeneralizedNormal{T}(μ, α, β)
end # fit_mle
