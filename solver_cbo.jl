module Cbo
import Random
include("lib_inverse_problem.jl")

export Config, step

struct Config
    β::Float64
    λ::Float64
    σ::Float64
    Δ::Float64
    ε::Float64
end

function step(problem, config, ensembles; 
              eq_constraint=false, ineq_constraint=false, verbose=false)

    if occursin("InverseProblem", string(typeof(problem)))
        objective(u) = Ip.reg_least_squares(problem, u)
    else
        objective = problem
    end

    ε = config.ε
    function extra_objective(x)
        result = 0
        if ! (eq_constraint == false)
            result += (1/ε) * eq_constraint(x)^2
        elseif ! (ineq_constraint == false) 
            val = ineq_constraint(x)
            result += (val < 0) ? 0 : (1/ε) *val
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
    new_ensembles = ensembles .- λ*Δ*diff .+ σ*sqrt(Δ)*(diff .* Random.randn(d, J))
end

end

