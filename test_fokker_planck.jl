import LinearAlgebra
import Random
import Statistics
import DelimitedFiles
import Formatting
import Plots
import PyCall
using LaTeXStrings

Plots.resetfontsizes()
Plots.scalefontsizes(2)

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
opti = true
adaptive = true
config = Eks.Config(dt, dtmax, reg , opti,  adaptive)

J = 100
ensembles = Random.randn(ip.d, J) .+ 1

niter = 200
errors = zeros(niter)
spreads = zeros(niter)
anim = Plots.Animation()

for iter in 1:niter
    global mean, ensembles, sum_weights

    new_ensembles = Solver.step(ip, config, ensembles, semi_implicit=false);
    println("Change = $(la.norm(new_ensembles -  ensembles))");
    ensembles = new_ensembles
    mean = Statistics.mean(ensembles, dims=2)
    weights = mean[1:Model.N_per_c:end]
    cov = Statistics.cov(ensembles, dims=2)
    spread = sqrt(la.norm(cov, 2))
    sum_weights = sum(weights)
    # error = la.norm(Model.utruth - mean) / la.norm(Model.utruth)
    error = sum(abs.(mean - Model.utruth))
    Formatting.printfmt("$iter: Spread: {:.3e}, Sum of weights: {:.3e}, error: {:.3e}\n",
                        spread, sum_weights, error)
    errors[iter] = error
    spreads[iter] = spread
    # print(mean)

    if iter % 5 == 0
        DelimitedFiles.writedlm("$datadir/ensemble-$iter.txt", ensembles);
    end

    # if iter % (niter ÷ niter) == 0
    #     Lx, Rx = minimum(ensembles[1,:]), maximum(ensembles[1,:])
    #     Ly, Ry = minimum(ensembles[2,:]), maximum(ensembles[2,:])
    #     Lx, Rx = 0, 1
    #     Ly, Ry = 0, 1
    #     gridx = range(Lx, stop=Rx, length=50)
    #     gridy = range(Ly, stop=Ry, length=50)
    #     objective(u) = Ip.least_squares(ip, u)
    #     values = zeros(length(gridx), length(gridy))
    #     for i in 1:length(gridx)
    #         for j in 1:length(gridy)
    #             values[j, i] = objective([gridx[i]; gridy[j]])
    #         end
    #     end
    #     Plots.contourf(gridx, gridy, values, size=(1200,800), c=:viridis, legend=:none,
    #                    left_margin=4Plots.mm, right_margin=4Plots.mm, bottom_margin=3Plots.mm)
    #     Plots.scatter!(ensembles[1,:], ensembles[2,:], legend=false, markersize=12)
    #     Plots.xlims!(Lx, Rx)
    #     Plots.ylims!(Ly, Ry)
    #     Plots.xlabel!(L"w_1")
    #     Plots.ylabel!(L"w_2")
    #     # Plots.title!(latexstring("Iteration $iter, \$ \\varepsilon = $(Model.ε) \$"))
    #     Plots.title!(latexstring("\\mathrm{Iteration}~$iter,~\\varepsilon = $(Model.ε)"))
    #     # Plots.frame(anim)
    #     Plots.savefig("data/fokker-planck_iter=$iter-eps=$(Model.ε).png")
    # end
end

Model.plot(mean)

Plots.plot(1:niter, errors_explicit, label="Explicit", size=(1200,800),
           left_margin=4Plots.mm, bottom_margin=3Plots.mm, line=(3,))
Plots.plot!(1:niter, errors_implicit, label="Semi-implicit", line=(3,))
Plots.xlabel!("Iteration index")
Plots.ylabel!("Error")
Plots.ylims!(0, 3)
Plots.savefig("data/errors.png")
Plots.plot(1:niter, spreads_explicit, yaxis=:log, label="Explicit", size=(1200,800),
           left_margin=4Plots.mm, bottom_margin=3Plots.mm, line=(3,))
Plots.plot!(1:niter, spreads_implicit, yaxis=:log, label="Semi-implicit", line=(3,))
Plots.xlabel!("Iteration index")
Plots.ylabel!("2-norm of sample covariance")
Plots.savefig("data/spreads.png")

sum(abs.(mean - Model.utruth))
Formatting.printfmt("\$\\varepsilon = $(Model.ε)\$ & {:.8f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\",
                    sum_weights, mean[1], mean[3], mean[5], mean[2], mean[4], mean[6])
sum_weights
# Plots.gif(anim, "fokker_planck.gif", fps=4)
