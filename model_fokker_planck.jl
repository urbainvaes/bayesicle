module FokkerPlanck

import Random
import LinearAlgebra
using LaTeXStrings
la = LinearAlgebra

include("lib_inverse_problem.jl")
export utruth, ip, ε, N_per_c, plot

# Number of Gaussians in the mixture
N = 3 ;
N_per_c = 2;
means = [-5, 0, 5];
variances = [.4, .1, .5];

# N = 2;
# N_per_c = 1;
# means = [-4, 4]
# variances = [.1, .1]

# Truth
Random.seed!(2)
weights = Random.rand(N);
weights = weights / sum(weights)

# Dimension of state space
d = N*N_per_c;

# Final time
T = .5;
L = 10;
xobservations = range(-L, stop=L, length=100)

function forward(u)
    # weights, variances = u[1:2:end], u[2:2:end]
    weights = u[1:N_per_c:end]
    variances = u[2:N_per_c:end]

    # dx = - θx dt + σ dWt
    θ, σ = 1, sqrt(2)
    new_means = exp(-θ*T)*means
    new_variances = variances*exp(-2θ*T) .+ (σ^2/2θ)*(1 - exp(-2θ*T))

    function solution(y)
        result = 0
        for i in 1:length(weights)
            mean, var = new_means[i], abs(new_variances[i])
            result += weights[i]/sqrt(2π*var) * exp(-(y-mean)^2/2var)
        end
        return result
    end

    return solution.(xobservations)
    # return [u[1]; u[2]]
end


utruth = zeros(d)
utruth[1:N_per_c:end] = weights[1:N];
if N_per_c > 1
    utruth[2:N_per_c:end] = variances[1:N];
end

# Covariance of noise and prior (prior will not be used)
γ = .01;

# Observations
y = forward(utruth)

# Square root of covariance matrix
K = length(y)
rtΓ = la.diagm(γ .+ zeros(K))

# Noisy observation
y = y + rtΓ*randn(K)

import Plots
Plots.plot(xobservations, y, line=(:dot, 3), marker=(:circle, 5),
           legend=:none, size=(1200,500),
           top_margin=2Plots.mm, bottom_margin=4Plots.mm)
Plots.xlabel!(L"x")
Plots.title!("Noisy observations")
Plots.savefig("data/observations.png")

function plot(u)
    Plots.plot(xobservations, y, line=(:dot, 3), marker=(:circle, 5),
               size=(1200,500), label="Observation",
               top_margin=3Plots.mm, bottom_margin=6Plots.mm)
    yu = forward(u)
    Plots.plot!(xobservations, forward(u), line=(3),
               label="Reconstruction")
    Plots.xlabel!(L"x")
    # Plots.title!(L"Comparison o")
    Plots.savefig("data/error.png")
end

# Constraint
function constraint_eq(u)
    weights = u[1:N_per_c:end]
    return sum(weights) - 1
end

function constraint_ineq(u)
    return min.(u, 0)
end

ε = .0001
forward_constraint(u) = [forward(u); constraint_eq(u); constraint_ineq(u)]
y_constraint = [y; 0; zeros(d)]
diag_noise_cov = [γ^2 .+ zeros(K); ε^2; ε^2 .+ zeros(d)]

# forward_constraint = forward
# y_constraint = y
# diag_noise_cov = γ^2 .+ zeros(K)

# Covariance matrices
noise_cov = la.diagm(diag_noise_cov)

# EKI will be used without regularization anyway
prior_cov = la.diagm(zeros(d) .+ 1)

ip = Ip.InverseProblem(forward_constraint, y_constraint, noise_cov, prior_cov)
end
