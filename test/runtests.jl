using Test
using Distributions
using MoreDistributions
using Random

const tests = [
    "generalizednormal",
]

printstyled("Running tests:\n", color=:blue)

Random.seed!(345679)

for t in tests
    @testset "Test $t" begin
        include("$t.jl")
    end
end