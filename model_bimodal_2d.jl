module ModelBimodal2d
include("lib_inverse_problem.jl")

import LinearAlgebra
la = LinearAlgebra

# Dimensions of the model
d, K = 2, 1

# Forward model
function forward(u)
    return [(u[1] - u[2])^2]
end

# Covariance of noise and prior
γ, σ = 1, 1e20
noise_cov = la.diagm(1 .+ zeros(K))
prior_cov = la.diagm(1 .+ zeros(d))

# Observation
y = [4.2297]

ip = Ip.InverseProblem(forward, y, noise_cov, prior_cov)
end
