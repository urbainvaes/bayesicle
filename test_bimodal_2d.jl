import LinearAlgebra
import Random
import Statistics
import DelimitedFiles
import DelimitedFiles
Random.seed!(0)

# Shorthand names for modules
la = LinearAlgebra
dlm = DelimitedFiles

include("solver_aldi.jl")
include("lib_inverse_problem.jl")
include("solver_multiscale.jl")
include("model_bimodal_2d.jl")

model = ModelBimodal2d

J = 4
dtmax = 1
reg = true
opti = false
adaptive = false
config = Aldi.Config(.001, dtmax, reg , opti,  adaptive)

# Initial ensembles
all_ensembles = zeros(model.ip.d, 0)
ensembles = Random.randn(model.ip.d, J)
for iter in 1:1e4
    ensembles = Aldi.step(model.ip, config, ensembles);
    if iter % 1000 == 0
        println("Iter = $iter")
    end
    all_ensembles = [all_ensembles ensembles]
end

datadir = "data_julia/model_bimodal_2d/aldi/"
run(`mkdir -p "$datadir"`);
DelimitedFiles.writedlm("$datadir/all_ensembles.txt", all_ensembles);

sigma = 1e-5
delta = 1e-5
dt = .01
dtmax = 1
reg = true
opti = false
adaptive = false
precond_mat = la.diagm(1 .+ zeros(model.ip.d))
config = Multiscale.Config(sigma, delta, dt, dtmax, reg, opti, adaptive, precond_mat)

J, niter = 8, 5*10^5
theta = zeros(model.ip.d)
xis = Random.randn(model.ip.d, J)
all_ensembles = zeros(model.ip.d, niter)


for iter in 1:niter
    global theta, xis
    theta, xis = Multiscale.step(model.ip, config, theta, xis)
    all_ensembles[:, iter] = theta
    if iter % 1000 == 0
        println("Iter = $iter")
    end
end

datadir = "data_julia/model_bimodal_2d/multiscale/"
run(`mkdir -p "$datadir"`);
DelimitedFiles.writedlm("$datadir/all_ensembles.txt", all_ensembles);

s = sign.(all_ensembles[1, :] - all_ensembles[2, :])
sum(1 .+ s)./2/niter
sum(abs.(s[2:end] - s[1:end-1]))
