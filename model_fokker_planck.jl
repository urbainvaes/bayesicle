module FokkerPlanck

import Random
import LinearAlgebra
la = LinearAlgebra

include("lib_inverse_problem.jl")
export utruth, ip

# Number of Gaussians in the mixture
N = 2; N_per_c = 1; d = N*N_per_c;
means = [-4, 0, 4]
variances = [.1, .1]

# Final time
T = .5;
xobservations = -10:.05:10

function forward(u)
    # weights, variances = u[1:2:end], u[2:2:end]
    weights = u[1:N_per_c:end]
    # variances = u[2:N_per_c:end]

    # dx = - θx dt + σ dWt
    θ = 1
    σ = sqrt(2)
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
end

# Truth
Random.seed!(0)
weights = Random.rand(N);
weights = weights / sum(weights)
# variances = [.1, .2, .3]

utruth = zeros(d)
utruth[1:N_per_c:end] = weights[1:N];
# utruth[2:2:end] = variances[1:N];

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
Plots.plot(xobservations, y)

# Constraint
function constraint_eq(u)
    weights = u[1:N_per_c:end]
    return sum(weights) - 1
end

function constraint_ineq(u)
    return min.(u, 0)
end

ε = .1
# forward_constraint(u) = [forward(u); constraint_eq(u); constraint_ineq(u)]
# y_constraint = [y; 0; zeros(d)]
# diag_noise_cov = [γ^2 .+ zeros(K); ε^2; ε^2 .+ zeros(d)]

forward_constraint = forward
y_constraint = y
diag_noise_cov = γ^2 .+ zeros(K)


# ε = .01
# forward_constraint = forward
# y_constraint = y
# diag_noise_cov = γ^2 .+ zeros(K)

# Covariance matrices
noise_cov = la.diagm(diag_noise_cov)
prior_cov = la.diagm(zeros(d) .+ Inf)
# prior_cov = la.diagm(zeros(d) .+ 2)

ip = Ip.InverseProblem(forward_constraint, y_constraint, noise_cov, prior_cov)
end
