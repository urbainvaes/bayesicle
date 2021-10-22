import LinearAlgebra
import Random
import Statistics
import DelimitedFiles
import Formatting
import Plots
using LaTeXStrings

include("lib_inverse_problem.jl")

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
opti = false
adaptive = true
config = Eks.Config(dt, dtmax, reg , opti,  adaptive)

J = 1000
ensembles = 5*Random.randn(ip.d, J)

niter = 1000
anim = Plots.Animation()

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

    if iter % 100 == 0
        Lx, Rx = minimum(ensembles[1,:]), maximum(ensembles[1,:])
        Ly, Ry = minimum(ensembles[2,:]), maximum(ensembles[2,:])
        gridx = range(Lx, stop=Rx, length=50)
        gridy = range(Ly, stop=Ry, length=50)
        objective(u) = Ip.reg_least_squares(ip, u)
        values = zeros(length(gridx), length(gridy))
        for i in 1:length(gridx)
            for j in 1:length(gridy)
                values[i, j] = objective([gridx[i]; gridy[j]])
            end
        end
        Plots.contourf(gridx, gridy, values)
        Plots.scatter!(ensembles[1,:], ensembles[2,:], legend=false)
        Plots.xlims!(Lx, Rx)
        Plots.ylims!(Ly, Ry)
        Plots.xlabel!(L"w_1")
        Plots.ylabel!(L"w_2")
        Plots.frame(anim)
    end
end
Plots.gif(anim, "fokker_planck.gif", fps=4)
