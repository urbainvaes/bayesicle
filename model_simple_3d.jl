module ModelSimple3d
include("lib_inverse_problem.jl")

import LinearAlgebra
la = LinearAlgebra

# Dimensions of the model
d, K, k = 3, 3, 5

# Forward model
function forward(u)
    return [u[1]; k*u[2]; k^2*u[3]]
end

# Covariance of noise and prior
γ, σ = 1, 1e20
noise_cov = la.diagm(γ^2 .+ zeros(K))
prior_cov = la.diagm(σ^2 .+ zeros(d))

# Unknown
utruth = 1 .+ zeros(d)

# Observation
y = forward(utruth)

ip = Ip.InverseProblem(forward, y, noise_cov, prior_cov)
end
