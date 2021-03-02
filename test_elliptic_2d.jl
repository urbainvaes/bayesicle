import LinearAlgebra
import Random
import Statistics
import DelimitedFiles

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

datadir = "data_julia/cbs/model_elliptic_2d"
run(`mkdir -p "$datadir"`);

ensembles = (J = 512; 3*Random.randn(d, J))
DelimitedFiles.writedlm("$datadir/utruth.txt", utruth);
DelimitedFiles.writedlm("$datadir/ensemble-0.txt", ensembles);

distance, spread, iter = 10, 1, 0
while spread > 1e-12
    mean = Statistics.mean(ensembles, dims=2)
    cov = Statistics.cov(ensembles, dims=2)
    spread = sqrt(la.norm(cov, 2))
    ensembles = Cbs.step(objective, config, ensembles); iter += 1
    distance = la.norm(mean - utruth)
    proba_truth = Cbs.proba_further(ensembles, utruth)
    println("Spread = $spread, Error=$distance, Proba = $proba_truth")
    DelimitedFiles.writedlm("$datadir/ensemble-$iter.txt", ensembles);
end
