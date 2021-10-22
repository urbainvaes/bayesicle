module Cbo
import Random
include("lib_inverse_problem.jl")

export Config, step

struct Config
    β::Float64
    λ::Float64
    σ::Float64
    Δ::Float64
    ν::Float64
    ε::Float64
end

function step(problem, config, ensembles;
              eq_constraint=nothing, ineq_constraint=nothing,
              grad_eq_constraint=nothing, grad_ineq_constraint=nothing,
              verbose=false)

    if occursin("InverseProblem", string(typeof(problem)))
        objective(u) = Ip.reg_least_squares(problem, u)
    else
        objective = problem
    end

    ν = config.ν
    ε = config.ε
    function extra_objective(x)
        result = 0
        if ! (eq_constraint == nothing)
            result += (1/ν) * eq_constraint(x)^2
        elseif ! (ineq_constraint == nothing)
            val = ineq_constraint(x)
            result += (val > 0) ? 0 : (1/ν) * val^2
        end
        return result
    end

    λ = config.λ
    σ = config.σ
    Δ = config.Δ

    d, J = size(ensembles)
    fensembles = zeros(J)
    for i in 1:length(fensembles)
        fensembles[i] = objective(ensembles[:, i])
        fensembles[i] += extra_objective(ensembles[:, i])
        verbose && print(".")
    end
    verbose && println("")

    fensembles = fensembles .- minimum(fensembles)
    weights = exp.(- config.β*fensembles)
    weights = weights / sum(weights)

    mean = sum(ensembles.*weights', dims=2)
    diff = ensembles .- mean
    new_ensembles = ensembles .- λ*Δ*diff .+ σ*sqrt(2Δ)*(diff .* Random.randn(d, J))
    if ! (grad_eq_constraint == nothing)
        for i in 1:length(fensembles)
            # SPECIAL CASE !!!
            new_ensembles[:, i] = new_ensembles[:, i] / (1 + 4*(Δ/ε)*eq_constraint(ensembles[:, i]))
            # new_ensembles[:, i] -= (2/ε)*Δ * eq_constraint(ensembles[:, i]) * grad_eq_constraint(ensembles[:, i])
        end
    end
    if ! (grad_ineq_constraint == nothing)
        for i in 1:length(fensembles)
            # SPECIAL CASE !!!
            # if ineq_constraint(ensembles[:, i]) < 0
            #     new_ensembles[:, i] = new_ensembles[:, i] / (1 + 4*(Δ/ε)*ineq_constraint(ensembles[:, i]))
            # end
            # new_ensembles[:, i] -= (2/ε)*Δ * eq_constraint(ensembles[:, i]) * grad_eq_constraint(ensembles[:, i])
        end
    end
    return new_ensembles
end

end
