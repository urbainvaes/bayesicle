import LinearAlgebra
import Random
import Statistics
import DelimitedFiles
using Distributed

# Shorthand names for modules
la = LinearAlgebra

# solver, model = "eks", "elliptic_2d"
# solver, model = "cbs", "elliptic_2d"
solver, model = "aldi", "elliptic_2d"

include("lib_inverse_problem.jl")
include("solver_$solver.jl")
include("model_$model.jl")

if solver == "eks"
    Solver = Eks
elseif solver == "cbs"
    Solver = Cbs
elseif solver == "aldi"
    Solver = Aldi
end

Model = ModelElliptic2d

ip = ModelElliptic2d.ip
objective(u) = Ip.reg_least_squares(ip, u)

Random.seed!(0);

J = 0;
if solver == "cbs"
    alpha = 0
    beta = 1
    opti = false
    adaptive = true
    ess = 1/2.
    config = Cbs.Config(alpha, beta, opti, adaptive, ess)
elseif solver == "eks"
    J = 2*ip.d
    dt = 1
    dtmax = 10
    reg = true
    opti = false
    adaptive = true
    config = Eks.Config(dt, dtmax, reg , opti,  adaptive)
elseif solver == "aldi"
    J = 2*ip.d
    dt = .5
    dtmax = 10
    reg = true
    opti = false
    adaptive = true
    config = Aldi.Config(dt, dtmax, reg , opti,  adaptive)
end

datadir = "data_julia/$solver/model_$model"
run(`mkdir -p "$datadir"`);

# init_iter = 100
init_iter = 0
if init_iter > 0
    ensembles = DelimitedFiles.readdlm("$datadir/ensemble-$init_iter.txt");
else
    ensembles = 3*Random.randn(ip.d, J)
    DelimitedFiles.writedlm("$datadir/ensemble-0.txt", ensembles);
end

DelimitedFiles.writedlm("$datadir/utruth.txt", Model.utruth);
all_ensembles = zeros((ip.d, 0))

niter = 400
for iter in init_iter+1:init_iter+niter
    global ensembles
    mean = Statistics.mean(ensembles, dims=2)
    cov = Statistics.cov(ensembles, dims=2)
    spread = sqrt(la.norm(cov, 2))
    distance = la.norm(mean - Model.utruth)
    proba_truth = Ip.proba_further(ensembles, Model.utruth)
    println("Spread = $spread, Error=$distance, Proba = $proba_truth")
    ensembles = Solver.step(ip, config, ensembles);
    DelimitedFiles.writedlm("$datadir/ensemble-$iter.txt", ensembles);
    if iter > 0 && iter % 10 == 0
        global all_ensembles
        all_ensembles = [all_ensembles ensembles]
        DelimitedFiles.writedlm("$datadir/all_ensembles-$iter.txt", all_ensembles);
    end
end
