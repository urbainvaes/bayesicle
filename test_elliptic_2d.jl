# Parse arguments {{{1
import ArgParse
s = ArgParse.ArgParseSettings("2-dimensional elliptic problem")

@ArgParse.add_arg_table! s begin
    "--solver", "-s"
        help = "The solver to use (aldi, cbs, …)"     # used by the help screen
        default = "aldi"
        required = false
    "--dataroot", "-d"
        help = "Directory for the data"
        default = "data_julia"
        required = false
    "--nparticles", "-n"
        help = "Number of particles"
        arg_type = Int64
        default = 256
        required = true
end

parsed_args = ArgParse.parse_args(ARGS, s)
solver = parsed_args["solver"]
dataroot = parsed_args["dataroot"]
nparticles = parsed_args["nparticles"]
model = "elliptic_2d"

import LinearAlgebra
import Random
import Statistics
import DelimitedFiles
include("lib_inverse_problem.jl")
include("solver_$solver.jl")
include("model_$model.jl")

Model = ModelElliptic2d
datadir = "$dataroot/$solver/model_$model"
run(`mkdir -p "$datadir"`);

if solver == "eks"
    Solver = Eks
elseif solver == "cbs"
    Solver = Cbs
elseif solver == "aldi"
    Solver = Aldi
elseif solver == "pCN"
    Solver = pCN
elseif solver == "multiscale"
    Solver = Multiscale
end

# Method {{{1
Random.seed!(0);

if solver == "cbs"
    alpha = 0
    beta = 1
    opti = false
    adaptive = true
    ess = 1/2.
    config = Cbs.Config(alpha, beta, opti, adaptive, ess)
    ensembles = Random.randn(Model.ip.d, nparticles)

elseif solver == "eks"
    dt = 1
    dtmax = 10
    reg = true
    opti = false
    adaptive = true
    config = Eks.Config(dt, dtmax, reg , opti,  adaptive)

elseif solver == "aldi"
    dtmax = 10
    reg = true
    opti = false

    dt = .5
    adaptive = true

    # dt = .1
    # adaptive = false

    config = Aldi.Config(dt, dtmax, reg , opti,  adaptive)

    # Calculated based on 100 iterations of gfALDI with adaptive time step
    initial_condition = "data_julia/aldi/model_$model/initial-100.txt"
    if isfile(initial_condition)
        println("Using existing initial condition")
        ensembles = DelimitedFiles.readdlm(initial_condition)
    else
        ensembles = Random.randn(Model.ip.d, nparticles)
        DelimitedFiles.writedlm("$datadir/ensemble-0.txt", ensembles);
    end
end

# init_iter = 100
init_iter = 0
if init_iter > 0
    ensembles = DelimitedFiles.readdlm("$datadir/ensemble-$init_iter.txt");
end

DelimitedFiles.writedlm("$datadir/utruth.txt", Model.utruth);
all_ensembles = zeros((Model.ip.d, 0))

niter = 400
for iter in init_iter+1:init_iter+niter
    global ensembles
    mean = Statistics.mean(ensembles, dims=2)
    cov = Statistics.cov(ensembles, dims=2)
    spread = sqrt(LinearAlgebra.norm(cov, 2))
    distance = LinearAlgebra.norm(mean - Model.utruth)
    proba_truth = Ip.proba_further(ensembles, Model.utruth)
    println("Spread = $spread, Error=$distance, Proba = $proba_truth")
    ensembles = Solver.step(Model.ip, config, ensembles; verbose=true);
    DelimitedFiles.writedlm("$datadir/ensemble-$iter.txt", ensembles);
    if iter > 0 && iter % 10 == 0
        global all_ensembles
        all_ensembles = [all_ensembles ensembles]
        DelimitedFiles.writedlm("$datadir/all_ensembles-$iter.txt", all_ensembles);
    end
end
