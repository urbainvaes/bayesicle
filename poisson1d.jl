import LinearAlgebra
import Random
import Statistics
import SpecialFunctions
import QuadGK

include("solver_cbs.jl")
include("model_elliptic_2d.jl")

# Shorthand names for modules
la = LinearAlgebra

objective = ModelElliptic2d.least_squares
utruth = ModelElliptic2d.utruth
d = ModelElliptic2d.d

Random.seed!(0);
alpha = 0
beta = 1
opti = false
adaptive = true
ess = 1/2.
config = Cbs.Config(alpha, beta, opti, adaptive, ess)

J = 256
ensembles = 3*Random.randn(d, J)
distance, spread, niter = 10, 1, 0
while spread > 1e-12
    mean = Statistics.mean(ensembles, dims=2)
    cov = Statistics.cov(ensembles, dims=2)
    spread = sqrt(la.norm(cov, 2))
    weighted_distance = sqrt((utruth - mean)' * (la.inv(cov) * (utruth - mean)))[1]
    ensembles = Cbs.step(objective, config, ensembles)
    niter += 1
    # Is truth likely to be from the posterior?
    factor = 2 / 2^(d/2) / SpecialFunctions.gamma(d/2)
    integral = QuadGK.quadgk(z -> exp(-z^2/2) * z^(d-1), 0, weighted_distance)[1];
    proba_sphere = factor*integral
    println("")
    println(la.diag(cov))
    println("Spread = $spread, Error=$weighted_distance, Proba = $proba_sphere")
end
