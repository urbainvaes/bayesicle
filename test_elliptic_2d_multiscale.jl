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

solver = "multiscale"
model = "elliptic_2d"
Solver = Multiscale
Model = ModelElliptic2d

datadir = "data_julia/$solver/model_$model"
datadir_eks = "data_julia/eks/model_$model"
run(`mkdir -p "$datadir"`);

ip = ModelElliptic2d.ip
objective(u) = Ip.reg_least_squares(ip, u)

# Use EKS as preconditioner
iter_precond = 100
ensembles_eks = dlm.readdlm("$datadir_eks/ensemble-100.txt");
cov_eks = Statistics.cov(ensembles_eks, dims=2)
mean_eks = Statistics.mean(ensembles_eks, dims=2)

J = 8
sigma = 1e-5
delta = 1e-5
dt = .02
dtmax = 1
reg = true
opti = false
adaptive = false
precond_mat = cov_eks
config = Multiscale.Config(sigma, delta, dt, dtmax, reg, opti, adaptive, precond_mat)

beta = .1
precond_vec = mean_eks
precond_mat = 4*cov_eks
config_pCN = pCN.Config(beta, precond_vec, precond_mat)

# Save truth
DelimitedFiles.writedlm("$datadir/utruth.txt", Model.utruth);

Random.seed!(0);
theta = mean_eks;
xis = Random.randn(Model.d, J);

# Save iterations
ensembles = theta

# For pCN
ftheta = -1

niter = 20000
for iter in 1:niter
    # Multiscale
    global ensembles, theta, xis

    # pCN
    global ftheta

    # theta, xis = Solver.step(ip, config, theta, xis)
    accept, theta, ftheta = pCN.step(ip, config_pCN, theta, ftheta)
    ensembles = [ensembles theta]
    distance = la.norm(theta - Model.utruth)
    proba_truth = Ip.proba_further(ensembles, Model.utruth)
    println("Accept=$accept Error=$distance, Proba = $proba_truth")

    if iter % 50 == 0
        DelimitedFiles.writedlm("$datadir/ensemble-$iter.txt", ensembles);
    end
end
