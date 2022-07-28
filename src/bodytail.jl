const Γ = gamma

# NON-normalized upper incomplete gamma function
@inline function Γ_inc_upper(a, x)
    _, q = gamma_inc(a, x)
    return Γ(a) * q
end

struct BodyTailGeneralizedNormal{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    α::T
    β::T

    function BodyTailGeneralizedNormal{T}(μ, σ, α, β) where {T}
        new{T}(μ, σ, α, β)
    end
end

#### Outer constructors
function BodyTailGeneralizedNormal(μ::T, σ::T, α::T, β::T; check_args::Bool=true) where {T<:Real}
    @check_args BodyTailGeneralizedNormal (σ, σ > zero(σ)) (α, α > zero(α)) (β, β > zero(β))
    return BodyTailGeneralizedNormal{T}(μ, σ, α, β)
end

BodyTailGeneralizedNormal(μ::Real, σ::Real, α::Real,β::Real) =
    BodyTailGeneralizedNormal(promote(μ, σ, α, β)...)

BodyTailGeneralizedNormal(μ::Integer, σ::Integer, α::Integer, β::Integer) = 
    BodyTailGeneralizedNormal(float(μ), float(σ), float(α), float(β))

# standard BodyTailGeneralizedNormal (mean 0, variance 1) with shape parameters  β
BodyTailGeneralizedNormal(α::T, β::T) where {T <: Real} = 
    BodyTailGeneralizedNormal(zero(T), sqrt(Γ(1/β)/Γ(3/β)), β)

# #### Conversions
convert(::Type{BodyTailGeneralizedNormal{T}}, μ::S, σ::S, α::S, β::S) where 
    {T <: Real, S <: Real} = BodyTailGeneralizedNormal(T(μ), T(σ), T(α), T(β))

convert(::Type{BodyTailGeneralizedNormal{T}}, d::BodyTailGeneralizedNormal{S}) where 
    {T <: Real, S <: Real} = BodyTailGeneralizedNormal(T(d.μ), T(d.σ), T(d.α), T(d.β))

@distr_support BodyTailGeneralizedNormal -Inf Inf

params(d::BodyTailGeneralizedNormal) = (d.μ, d.σ, d.α, d.β)
@inline partype(d::BodyTailGeneralizedNormal{T}) where {T <: Real} = T

#### Statistics
mean(d::BodyTailGeneralizedNormal) = d.μ
median(d::BodyTailGeneralizedNormal) = d.μ
mode(d::BodyTailGeneralizedNormal) = d.μ

var(d::BodyTailGeneralizedNormal) = d.σ^2*Γ((d.α + 3)/d.β)/(3 * Γ((d.α + 1)/d.β))
skewness(d::BodyTailGeneralizedNormal{T}) where {T <: Real} = zero(T)
kurtosis(d::BodyTailGeneralizedNormal) = 9*Γ((d.α + 1)/d.β)*Γ((d.α + 5)/d.β)/(5*Γ((d.α + 3)/d.β)^2) - 3

function pdf(d::BodyTailGeneralizedNormal, x::Real)
    (μ, σ, α, β) = params(d)
    return Γ_inc_upper(α/β, abs((x - μ)/σ)^β)/(2*σ*Γ((α + 1)/β))
end

function cdf(d::BodyTailGeneralizedNormal, x::Real)
    μ, σ, α, β = params(d)
    z = (x - μ)/σ
    z_abs = abs(z)
    z_abs_β = z_abs^β
    a1  = (α + 1)/β
    a2 = α/β
    c = (Γ_inc_upper(a1, z_abs_β) - z_abs * Γ_inc_upper(a2, z_abs_β))/(2 * Γ(a1))
    return 1/2 - sign(z)*(c - 1/2)
end

# @quantile_newton BodyTailGeneralizedNormal

function quantile(d::BodyTailGeneralizedNormal, p::Real)
    if p == 0.5
        return mode(d)
    end
    s = sign(p - 0.5)
    if p < 0.5
        p = 1 - p
    end
    lb = 0.0
    ub = std(d)
    while cdf(d, ub) < p
        ub *= 2
    end
    return s * quantile_bisect(d, p, lb, ub)
end

function fit_mle(::Type{<:BodyTailGeneralizedNormal}, x::AbstractVector{T};
                 reltol::Real=1.0e-6) where {T <: Real}

    μ₀ = mean(x)
    α₀ = β₀ = 1.0
    σ₀ = sqrt(3*var(x)*gamma((α₀ + 1)/β₀)/gamma((α₀ + 3)/β₀))

    n = length(x)
    y = similar(x)

    function log_likelihood(p)
        μ, σ, α, β = p
        s = 0
        @. y = abs((x - μ)/σ)^β

        a = α/β
        @inbounds for i=1:n
            _, q = gamma_inc(a, y[i])
            s += log(q)
        end
        s /= n

        return s - log(2) - log(σ) - loggamma((α+1)/β) + loggamma(α/β)
    end

    opt = Opt(:LN_NEWUOA_BOUND, 4)
    opt.lower_bounds = [-Inf, 0., 0., 0.]
    opt.upper_bounds = [Inf, Inf, Inf, Inf]
    opt.xtol_rel = reltol
    opt.max_objective = (p, grad) -> log_likelihood(p)

    _, p, ret = optimize(opt, [μ₀, σ₀, α₀, β₀])
    if ret ∉ (:SUCCESS, :XTOL_REACHED, :FTOL_REACHED)
        error("Numerical optimization failed.")
    end
    
    BodyTailGeneralizedNormal{T}(p...)
end