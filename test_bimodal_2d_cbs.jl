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

J = 5000
fensembles = nothing
ensembles = Random.randn(model.ip.d, J)

niter = 100000
datadir = "data_julia/model_bimodal_2d/cbs_metro/"
run(`mkdir -p "$datadir"`);
for iter in 1:niter
    global ensembles
    accept, ensembles, fensembles = CbsMetro.step(model.ip, config, ensembles, fensembles; verbose=false);
    if iter % 1000 == 0
        println("$iter $accept")
        DelimitedFiles.writedlm("$datadir/ensembles-$iter.txt", ensembles);
    end
end
