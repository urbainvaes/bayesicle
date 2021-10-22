import LinearAlgebra
import Random
import Statistics
import DelimitedFiles
import DelimitedFiles
Random.seed!(0)

# Shorthand names for modules
la = LinearAlgebra
dlm = DelimitedFiles

include("solver_cbs_metro.jl")
include("lib_inverse_problem.jl")
include("model_bimodal_2d.jl")
model = ModelBimodal2d

alpha = .5
beta = 1
adaptive = true
ess = .5
config = CbsMetro.Config(alpha, beta, adaptive, adaptive)

J = 10000
fensembles = nothing
ensembles = Random.randn(model.ip.d, J)

niter = 1000000
nburnin = 1000
datadir = "data_julia/model_bimodal_2d/cbs_metro/"
run(`mkdir -p "$datadir"`);
naccepts = 0
# all_ensembles = zeros(model.ip.d, (niter - nburnin)*J)
for iter in 1:niter
    global naccepts, ensembles, fensembles
    accept, ensembles, fensembles = CbsMetro.step(model.ip, config, ensembles, fensembles; verbose=false);
    accept && (naccepts += 1)
    if iter % 1000 == 0
        println("$iter $naccepts")
        DelimitedFiles.writedlm("$datadir/ensembles-$iter.txt", ensembles);
    end
    if iter > nburnin
        # all_ensembles[:,(iter - nburnin - 1)*J+1:(iter-nburnin)*J] = ensembles
    end
end
DelimitedFiles.writedlm("$datadir/all_ensembles.txt", all_ensembles);
DelimitedFiles.writedlm("$datadir/last_ensemble.txt", ensembles);
