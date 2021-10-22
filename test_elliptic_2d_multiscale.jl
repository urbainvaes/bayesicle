# Parse arguments {{{1
import ArgParse
s = ArgParse.ArgParseSettings("2-dimensional elliptic problem")

@ArgParse.add_arg_table! s begin
    "--solver", "-s"
        default = "pCN"
    # "--solver", "-s"
        # default = "pCN"
end

import LinearAlgebra
import Random
import Statistics
import DelimitedFiles

# Shorthand names for modules
la = LinearAlgebra
dlm = DelimitedFiles

include("lib_inverse_problem.jl")
include("solver_multiscale.jl")
include("solver_pCN.jl")
include("model_elliptic_2d.jl")

parsed_args = ArgParse.parse_args(ARGS, s)
solver = parsed_args["solver"]

if solver == "pCN"
    Solver = pCN
elseif solver == "multiscale"
    Solver = Multiscale
end

model = "elliptic_2d"
Model = ModelElliptic2d

datadir = "data_julia/$solver/model_$model"
run(`mkdir -p "$datadir"`);

# Method {{{1
ip = ModelElliptic2d.ip
objective(u) = Ip.reg_least_squares(ip, u)

# Use EKS as preconditioner
datadir_aldi = "data_julia/aldi/model_$model"

ensembles_aldi = zeros(ip.d, 0)
for i in 101:200
    global ensembles_aldi
    ensembles = dlm.readdlm("$datadir_aldi/ensemble-$i.txt");
    ensembles_aldi = [ensembles_aldi ensembles];
end
cov_aldi = Statistics.cov(ensembles_aldi, dims=2)
mean_aldi = Statistics.mean(ensembles_aldi, dims=2)

dlm.writedlm("precond_mat.txt", cov_aldi);
dlm.writedlm("precond_vec.txt", mean_aldi);

mean_aldi = dlm.readdlm("precond_vec.txt");
cov_aldi = dlm.readdlm("precond_mat.txt");

J = 8
sigma = 1e-5
delta = 1e-5
dt = .02
dtmax = 1
reg = true
opti = true
adaptive = false
precond_mat = cov_aldi
config = Multiscale.Config(sigma, delta, dt, dtmax, reg, opti, adaptive, precond_mat)

beta = .1
precond_vec = mean_aldi
precond_mat = 4*cov_aldi
# precond_mat = diag
config_pCN = pCN.Config(beta, precond_vec, precond_mat)

# Save truth
DelimitedFiles.writedlm("$datadir/utruth.txt", Model.utruth);

Random.seed!(0);
theta = mean_aldi;
xis = Random.randn(Model.d, J);

# Save iterations
ensembles = theta

# For pCN
ftheta = -1

niter = 20000
global naccepts = 0
for iter in 1:niter

    if solver == "multiscale"
        global ensembles, theta, xis
        theta, xis = Solver.step(ip, config, theta, xis)
        accept = 1
    end

    if solver == "pCN"
        global ftheta, naccepts
        accept, theta, ftheta = pCN.step(ip, config_pCN, theta, ftheta)
        naccepts += (accept ? 1 : 0)
        println("$naccepts / $iter")
    end

    ensembles = [ensembles theta]
    distance = la.norm(theta - Model.utruth)
    proba_truth = Ip.proba_further(ensembles, Model.utruth)
    cov = Statistics.cov(ensembles, dims=2)
    spread = sqrt(la.norm(cov, 2))
    println("Accept=$accept Error=$distance, Proba=$proba_truth, Spread=$spread")

    if iter % 50 == 0
        DelimitedFiles.writedlm("$datadir/ensemble-$iter.txt", ensembles);
    end
end
