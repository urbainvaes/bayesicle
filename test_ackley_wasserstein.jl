#!/usr/bin/env julia
import Random
import Statistics
import DelimitedFiles
import OptimalTransport
import Tulip
ot = OptimalTransport

include("model_ackley.jl")
include("solver_eks.jl")
include("solver_cbo.jl")

objective = Ackley.ackley
constraint = Ackley.sphere_constraint
grad_constraint = Ackley.grad_sphere_constraint

β = 1
λ = 1
ε = .2
σ = .7
Δ = .1

writedlm = DelimitedFiles.writedlm
datadir = "data/"
run(`mkdir -p $datadir`)

Js = [250,500,1000,8000]
ensembles = []
for J in Js
    append!(ensembles, [10*Random.randn(Ackley.n, J)])
end

Random.seed!(0);
nsteps = 1000
config = Cbo.Config(β, λ, σ, Δ, ε)
for i in 1:nsteps
    for (i, ensemble) in enumerate(ensembles)
        ensembles[i] = Cbo.step(objective, config, ensemble;
                                verbose=false, eq_constraint=constraint,
                                grad_eq_constraint=grad_constraint)
    end

    μreference = fill(1/Js[end], Js[end])
    reference_ensemble = ensembles[end]
    for (i, J) in enumerate(Js[1:end-1])
        μ = fill(1/Js[i], Js[i])
        C = ot.pairwise(ot.SqEuclidean(), ensembles[i], reference_ensemble; dims=2);
        γ = ot.emd2(μ, μreference, C, Tulip.Optimizer());
    end
end
