import LinearAlgebra
import Random
import Statistics
import DelimitedFiles
import DelimitedFiles
Random.seed!(0)

# Shorthand names for modules
la = LinearAlgebra
dlm = DelimitedFiles

include("lib_inverse_problem.jl")
include("solver_multiscale.jl")
include("solver_aldi.jl")
include("model_simple_3d.jl")

model = ModelSimple3d

J = 5
dtmax = 1
reg = false
opti = false
adaptive = false
config = Aldi.Config(.001, dtmax, reg , opti,  adaptive)

# Initial ensembles
all_ensembles = zeros(model.ip.d, 0)
ensembles = Random.randn(model.ip.d, J)

for iter in 1:1e4
    ensembles = Aldi.step(model.ip, config, ensembles);
end

mean = Statistics.mean(ensembles, dims=2)
cov = Statistics.cov(ensembles, dims=2)
config = Aldi.Config(.01, dtmax, reg , opti,  adaptive)

for iter in 1:9e4
    ensembles = Aldi.step(model.ip, config, ensembles);
    mean = (iter*mean + Statistics.mean(ensembles, dims=2))/(iter + 1)
    cov = (iter*cov + Statistics.cov(ensembles, dims=2))/(iter + 1)
    if iter % 1000 == 0
        println("Iter = $iter Mean = $(mean[1]), Cov = $(cov[1,1])")
    end
end

DelimitedFiles.writedlm("data_julia/precond_mat.txt", cov);
DelimitedFiles.writedlm("data_julia/precond_vec.txt", mean);
