module ModelSimple1d
include("lib_inverse_problem.jl")

import LinearAlgebra
la = LinearAlgebra

# Dimensions of the model
d, K = 1, 1

# Forward model
function forward(u)
    return [u[1]]
end

# Covariance of noise and prior
γ, σ = 1, 1e20
noise_cov = la.diagm(γ^2 .+ zeros(K))
prior_cov = la.diagm(σ^2 .+ zeros(d))

# Unknown
utruth = zeros(1)

# Observation
y = forward(utruth)

ip = Ip.InverseProblem(forward, y, noise_cov, prior_cov)
end

