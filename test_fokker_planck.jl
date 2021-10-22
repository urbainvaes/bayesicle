import LinearAlgebra
import Random
import Statistics
import DelimitedFiles
import Formatting

# Shorthand names for modules
la = LinearAlgebra
dlm = DelimitedFiles

include("lib_inverse_problem.jl")
include("model_fokker_planck.jl")
include("solver_eks.jl")

solver = "eks"
model = "fokker_planck"
Solver = Eks
Model = FokkerPlanck

datadir = "data_julia/$solver/model_$model"
datadir_eks = "data_julia/eks/model_$model"
run(`mkdir -p "$datadir"`);

ip = Model.ip
objective(u) = Ip.reg_least_squares(ip, u)

dt = 1
dtmax = 1e200
reg = false
opti = true
adaptive = true
config = Eks.Config(dt, dtmax, reg , opti,  adaptive)

J = 1000
ensembles = 5*Random.randn(ip.d, J)

niter = 20000
for iter in 1:niter
    global mean, ensembles

    new_ensembles = Solver.step(ip, config, ensembles);
    println("Change = $(la.norm(new_ensembles -  ensembles))");
    ensembles = new_ensembles
    mean = Statistics.mean(ensembles, dims=2)
    cov = Statistics.cov(ensembles, dims=2)
    spread = sqrt(la.norm(cov, 2))
    sum_weights = sum(mean[1:2:end])
    error = la.norm(Model.utruth - mean) / la.norm(Model.utruth)
    Formatting.printfmt("$iter: Spread: {:.3e}, Sum of weights: {:.3e}, error: {:.3e}\n",
                        spread, sum_weights, error)
    # print(mean)

    if iter % 5 == 0
        DelimitedFiles.writedlm("$datadir/ensemble-$iter.txt", ensembles);
    end
end
