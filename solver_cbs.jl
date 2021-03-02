module Cbs
import LinearAlgebra
import QuadGK
import Random
import SpecialFunctions
import Statistics
using SharedArrays
using Distributed

export Config, step, proba_further

struct Config
    alpha::Float64
    beta::Float64
    opti::Bool
    adaptive::Bool
    ess::Float64
end

function find_beta(fensembles, ess)
    J = length(fensembles)
    function get_ess(β)
        weights = exp.(- β*fensembles)
        return sum(weights)^2/sum(weights.^2)/J
    end
    β1, β2 = 0, 1e20
    β, e = β2, get_ess(β2)
    if e > ess
        println("Can't find β")
        return β
    end
    while abs(e - ess) > 1e-2
        β = (β1 + β2)/2
        e = get_ess(β)
        if e > ess
            β1 = β
        else
            β2 = β
        end
    end
    return β
end

function step(objective, config, ensembles)
    alpha = config.alpha
    opti = config.opti
    adaptive = config.adaptive
    ess = config.ess

    d, J = size(ensembles)
    fensembles = zeros(J)
    for i in 1:length(fensembles)
        fensembles[i] = objective(ensembles[:, i])
        print(".")
    end
    println("")

    fensembles = fensembles .- minimum(fensembles)
    beta = adaptive ? find_beta(fensembles, ess) : config.beta
    weights = exp.(- beta*fensembles)
    weights = weights / sum(weights)

    mean = sum(ensembles.*weights', dims=2)
    diff = ensembles .- mean
    cov = zeros(d, d)
    for j in 1:J
        cov += weights[j] * diff[:, j]*diff[:, j]'
    end
    coeff_noise = sqrt(1 - alpha^2) * sqrt(cov*(opti ? 1 : (1+beta)))
    coeff_noise = real(coeff_noise)
    new_ensembles = mean .+ alpha.*diff .+ coeff_noise*Random.randn(d, J)
end

function proba_further(ensembles, u)
    mean = Statistics.mean(ensembles, dims=2)
    cov = Statistics.cov(ensembles, dims=2)
    inv_cov = LinearAlgebra.inv(cov)
    weighted_distance = sqrt((u - mean)' * (inv_cov * (u - mean)))[1]
    d = length(mean)
    factor = 2 / 2^(d/2) / SpecialFunctions.gamma(d/2)
    integral = QuadGK.quadgk(z -> exp(-z^2/2) * z^(d-1), 0, weighted_distance)[1];
    proba_ball = factor*integral
    return 1 - proba_ball
end

end
