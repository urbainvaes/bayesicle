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

sigma = 1e-5
delta = 1e-5
dt = .01
dtmax = 1
reg = true
opti = false
adaptive = false
precond_mat = la.diagm(1 .+ zeros(model.ip.d))
config = Multiscale.Config(sigma, delta, dt, dtmax, reg, opti, adaptive, precond_mat)

J, niter = 8, 10^6
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

reg = true
opti = false
adaptive = false
config_aldi = Aldi.Config(.01, 0, reg , opti, adaptive)
ensembles = Random.randn(model.ip.d, J)
niter_aldi = niter ÷ J
all_ensembles = zeros(model.ip.d, J*niter_aldi)
for iter in 1:niter_aldi
    println(iter)
    global ensembles
    new_ensembles = Aldi.step(model.ip, config_aldi, ensembles; verbose=false);
    if la.norm(new_ensembles - new_ensembles) != 0.0
        error("Ensembles contain NaNs!")
        break
    end
    ensembles = new_ensembles
    all_ensembles[:,(iter-1)*J+1:iter*J] = ensembles
    if iter % 1000 == 0
        println("Iter = $iter")
    end
end
datadir = "data_julia/model_bimodal_2d/aldi/"
run(`mkdir -p "$datadir"`);
DelimitedFiles.writedlm("$datadir/all_ensembles.txt", all_ensembles);
