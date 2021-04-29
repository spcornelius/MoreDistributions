# MoreDistributions

More probability distributions for Julia. Implements the same interface as in [Distributions.jl](https://juliastats.org/Distributions.jl/stable/). For example:

```julia
julia> using MoreDistributions

julia> d = GeneralizedNormal(-1.0, 1.0, 1.5)
GeneralizedNormal{Float64}(μ=-1.0, α=1.0, β=1.5)

julia> mean(d)
0.5

julia> std(d)
0.8593533101243331

julia> rand(d, 3)
3-element Vector{Float64}:
 -1.0934595524895587
 -0.8887376183322262
 -0.6168054325590637

julia> cdf(d, -1.0)
0.5
```

Right now, only implements the  [Generalized Normal](https://en.wikipedia.org/wiki/Generalized_normal_distribution), so maybe should be called "OneMoreDistribution".
