#!/usr/bin/env julia
import Random
import Statistics
import DelimitedFiles

include("model_ackley.jl")
include("solver_eks.jl")
include("solver_cbo.jl")

objective = Ackley.ackley
constraint = Ackley.sphere_constraint

β = 1
λ = 1
σ = .7
Δ = .1
ε = 2^-4
config = Cbo.Config(β, λ, σ, Δ, ε)

J = 1000
nsimuls = 100
limits = zeros(Ackley.n, nsimuls)

Random.seed!(0);
for s in 1:nsimuls
    global mean
    ensembles = 10*Random.randn(Ackley.n, J)
    distance, spread, niter = 10, 1, 0
    while spread > 1e-10
        mean = Statistics.mean(ensembles, dims=2)
        spread = sqrt(sum(abs2, Statistics.cov(ensembles, dims=2)))
        ensembles = Cbo.step(objective, config, ensembles;
                             verbose=false, eq_constraint=constraint)
        # println("$niter: Constraint: $(constraint(mean)), Distance: $distance, Spread: $spread")
        niter += 1
    end
    println(mean, " ", constraint(mean))
    limits[:, s] = mean
end

writedlm = DelimitedFiles.writedlm

datadir = "data/"
run(`mkdir -p $datadir`)
writedlm(datadir * "limits-epsilon=$ε-J=$J.txt", limits)
writedlm(datadir * "limits.txt", limits)
