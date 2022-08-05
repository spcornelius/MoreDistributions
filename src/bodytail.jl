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
    return gamma(α/β, abs((x - μ)/σ)^β)/(2*σ*Γ((α + 1)/β))
end

function cdf(d::BodyTailGeneralizedNormal, x::Real)
    μ, σ, α, β = params(d)
    z = (x - μ)/σ
    z_abs = abs(z)
    z_abs_β = z_abs^β
    a1  = (α + 1)/β
    a2 = α/β
    c = (gamma(a1, z_abs_β) - z_abs * gamma(a2, z_abs_β)) / gamma(a1) / 2
    return 1/2 - sign(z)*(c - 1/2)
end

function quantile(d::BodyTailGeneralizedNormal, p::Real)
    if p == 0.5
        return mode(d)
    end
    # distribution is symmetric around μ, so focus on finding
    # a root of cdf(x) = p for x > μ (i.e., p > 0.5) and flip to the
    # other side later if necessary
    s = sign(p - 0.5)
    if p < 0.5
        p = 1 - p
    end

    # if p > 0.5, then the associated quantile x will satisfy cdf(μ, x) < p
    # so find an upper bound for x, i.e. a pt where cdf(ub) > p. Then x
    # will be somewhere in the range [μ, ub]
    lb = d.μ
    Δ = std(d)
    ub = lb + Δ
    while cdf(d, ub) < p
        ub += Δ
    end
    return s * quantile_bisect(d, p, lb, ub) + (1 - s) * d.μ
end

struct BTGNFitCache{T}
    z::AbstractVector{T}
    z_abs::AbstractVector{T}
    z_absα::AbstractVector{T}
    z_absβ::AbstractVector{T}
    log_z_abs::AbstractVector{T}
    Γ::AbstractVector{T}
    y1::AbstractVector{T}
    y2::AbstractVector{T}

    function BTGNFitCache(x::AbstractVector{T}) where {T <: Real}
        arrays = (similar(x) for _ in 1:8)
        new{T}(arrays...)
    end
end

function btgn_fit_objective(p, grad, x::AbstractVector{T},
                            cache::BTGNFitCache) where {T <: Real}
    μ, log_σ, log_α, log_β = p
    @unpack z, z_abs, z_absα, z_absβ, log_z_abs, Γ, y1, y2 = cache

    σ = exp(log_σ)
    α = exp(log_α)
    β = exp(log_β)

    c1 = α/β
    c2 = (α + 1.)/β
    c3 = c1 + 1.
    c4 = gamma(c1)

    # parameters for hypergeoemtric function
    a = [c1, c1]
    b = [c3, c3]

    # according to Mathematica, this is equivalent to G(x), where G is the Meijer-G 
    # function with m = 3, n = 0, p = 2, q = 3 and parameters a = [1, 1], b = [0, 0, α/β]
    # (as of this coding, there is no direct way to evaluate the Meijer-G function in Julia)
    ψ1 = digamma(c1)
    A(x) = x^c1 * pFq(a, b, -x) / c1^2 + c4 * (ψ1 - log(x))

    @. z = (x - μ)/σ
    @. z_abs = abs(z)
    @. z_absβ = z_abs^β
    @. Γ = gamma(c1, z_absβ)

    obj =  mean(@~ @. log(Γ)) - log(2) - log(σ) - loggamma(c2)

    if length(grad) > 0
        @. log_z_abs = log(z_abs)
        @. z_absα = z_abs^α
        @. y1 = A(z_absβ) / Γ
        @. y2 = exp(-z_absβ) / Γ

        ψ2 = digamma(c2)

        grad[1] = β / σ * mean(@~ @. sign(z) * z_abs^(α - 1) * y2)

        # extra factors of σ, α, and β in these are to correct for the fact that
        # we're working in log-space
        grad[2] = σ * (β * mean(@~ @. z_absα * y2) - 1) / σ
        grad[3] = α * (mean(@~ @. log_z_abs + y1 / β) - ψ2 / β)
        grad[4] = β * (mean(@~ @. -log_z_abs * (c1 + z_absα * y2) - c1 * y1/β) + (c2 * ψ2)/β)
    end

    return obj
end

function fit_mle(::Type{<:BodyTailGeneralizedNormal}, x::AbstractVector{T};
                 alg = :LN_NEWUOA,
                 reltol::Real=1.0e-6) where {T <: Real}

    μ₀ = mean(x)

    α₀ = 2.0
    β₀ = 2.0
    σ₀ = sqrt(3*var(x)*gamma((α₀ + 1)/β₀)/gamma((α₀ + 3)/β₀))

    cache = BTGNFitCache(x)

    opt = Opt(alg, 4)
    opt.xtol_rel = reltol
    opt.max_objective = (p, grad) -> btgn_fit_objective(p, grad, x, cache)

    _, p, ret = optimize(opt, [μ₀, log(σ₀), log(α₀), log(β₀)])
    if ret ∉ (:SUCCESS, :XTOL_REACHED, :FTOL_REACHED)
        error("Numerical optimization failed.")
    end
    μ = p[1]
    σ = exp(p[2])
    α = exp(p[3])
    β = exp(p[4])
    BodyTailGeneralizedNormal{T}(μ, σ, α, β)
end