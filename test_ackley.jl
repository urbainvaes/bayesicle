#!/usr/bin/env julia
import Random
import Statistics
import DelimitedFiles

include("model_ackley.jl")
include("solver_eks.jl")
include("solver_cbo.jl")

objective = Ackley.ackley
constraint = Ackley.sphere_constraint
grad_constraint = Ackley.grad_sphere_constraint

β = 1
λ = 1
σ = .7
Δ = .01

writedlm = DelimitedFiles.writedlm
datadir = "data/"
run(`mkdir -p $datadir`)

# ε = .1
J = 100
nsimuls = 100

ν, ε, nsimuls = 1, 1, 1
νs = [ν]

# νs = [10., 1., .1]

for ν in νs
    config = Cbo.Config(β, λ, σ, Δ, ν, ε)
    limits = zeros(Ackley.n, nsimuls)
    Random.seed!(0);
    for s in 1:nsimuls
        global mean
        ensembles = 10*Random.randn(Ackley.n, J)
        distance, spread, niter = 10, 1, 0
        while spread > 1e-10
            # println("iteration $spread")
            mean = Statistics.mean(ensembles, dims=2)
            spread = sqrt(sum(abs2, Statistics.cov(ensembles, dims=2)))
            ensembles = Cbo.step(objective, config, ensembles;
                                 verbose=false, ineq_constraint=constraint,
                                 # verbose=false, eq_constraint=constraint,
                                 # grad_eq_constraint=nothing)
                                 grad_ineq_constraint=grad_constraint)
            # ensembles = Cbo.step(objective, config, ensembles;
            #                      verbose=false, ineq_constraint=constraint)
            println("$niter: Constraint: $(constraint(mean)), Distance: $distance, Spread: $spread")
            niter += 1
            writedlm(datadir * "niter=$niter.txt", ensembles)
        end
        println(s, mean, " ", constraint(mean))
        limits[:, s] = mean
    end

    writedlm(datadir * "limits-nu=$ν-epsilon=$ε-J=$J.txt", limits)
    writedlm(datadir * "limits.txt", limits)
end
