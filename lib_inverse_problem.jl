module Ip

import QuadGK
import SpecialFunctions
import SparseArrays
import LinearAlgebra
import Statistics
la = LinearAlgebra
sparse = SparseArrays

struct InverseProblem
    forward::Function
    observation::Array{Float64,1}
    noise_cov::Array{Float64,2}
    prior_cov::Array{Float64,2}
    inv_noise_cov::Array{Float64,2}
    inv_prior_cov::Array{Float64,2}
    d::Int
    InverseProblem(forward, observation, noise_cov, prior_cov) =
    begin
        inv_noise_cov = la.inv(noise_cov);
        inv_prior_cov = la.inv(prior_cov);
        new(forward, observation, noise_cov, prior_cov, inv_noise_cov,
            inv_prior_cov, size(prior_cov)[1])
    end
end

function reg_least_squares(ip, u)
    misfit = ip.observation - ip.forward(u)
    return 1/2 * (misfit' * (ip.inv_noise_cov*misfit) + u' * (ip.inv_prior_cov*u))
end

function least_squares(ip, u)
    misfit = ip.observation - ip.forward(u)
    return 1/2 * misfit' * (ip.inv_noise_cov*misfit)
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
