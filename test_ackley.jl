#!/usr/bin/env julia
import Random
import Statistics

include("model_ackley.jl")
include("solver_eks.jl")
include("solver_cbo.jl")

objective = Ackley.ackley
constraint = Ackley.sphere_constraint

Random.seed!(0);

β = 1
λ = 1
σ = 2
Δ = .001
ε = .001
config = Cbo.Config(β, λ, σ, Δ, ε)

# nsimuls = 100
# successes = zeros(nsimuls)
# errors = zeros(nsimuls)
# niters = zeros(nsimuls)

J = 1000
Random.seed!(0)
ensembles = 3*Random.randn(Ackley.n, J)
distance, spread, niter = 10, 1, 0
while spread > 1e-10
    global mean
    mean = Statistics.mean(ensembles, dims=2)
    distance = maximum(abs.(mean - Ackley.exact))
    spread = sqrt(sum(abs2, Statistics.cov(ensembles, dims=2)))
    ensembles = Cbo.step(objective, config, ensembles; 
                         verbose=false, eq_constraint=false)
    println("Constraint: $(constraint(mean)), Distance: $distance, Spread: $spread")
    niter += 1
end
println(mean)
