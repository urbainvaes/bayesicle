module ModelSimple1d
include("lib_inverse_problem.jl")

import LinearAlgebra
la = LinearAlgebra

# Dimensions of the model
d, K, f = 1, 2, 2

# Forward model
function forward(u)
    return [u[1], sin(.6*π*u[1])]
end

# Covariance of noise and prior
γ, σ = .4, 1e20
noise_cov = la.diagm([1, .1])
prior_cov = la.diagm(σ^2 .+ zeros(d))

# Unknown
utruth = zeros(1)

# Observation
y = [1.353, 0.080]

ip = Ip.InverseProblem(forward, y, noise_cov, prior_cov)
end

# if abspath(PROGRAM_FILE) == @__FILE__
    include("lib_inverse_problem.jl")
    import Plots
    x = -5:.01:5
    y = (u -> Ip.reg_least_squares(ModelSimple1d.ip, [u])).(x)
    Plots.plot(x, exp.(-y))
# end
