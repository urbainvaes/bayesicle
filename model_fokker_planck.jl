module FokkerPlanck

import Random
import LinearAlgebra
la = LinearAlgebra

include("lib_inverse_problem.jl")
export utruth, ip

# Number of Gaussians in the mixture
N = 2; d = 3N;

# Final time
T = .5;

function forward(u)
    weights, means, variances = u[1:3:end], u[2:3:end], u[3:3:end]
    variances = [.1, .2]
    # .+ zeros(N)

    new_means = exp(-T)*means
    new_variances = variances*exp(-2T) .+ (1 - exp(-2T))

    function solution(y)
        result = 0
        for i in 1:length(weights)
            mean, var = new_means[i], abs(new_variances[i])
            result += weights[i]/sqrt(2π*var) * exp(-(y-mean)^2/2var)
        end
        return result
    end

    observations = -5:.1:5
    return solution.(observations)
end

# Truth
Random.seed!(0)
weights = Random.randn(N);
weights = weights / sqrt(weights'weights)
weights = weights.^2

means = [-2, 0, 1]
# variances = .1 .+ zeros(N)
variances = [.1, .2]
# variances = [.1, .1, .1]

utruth = zeros(d)
utruth[1:3:end] = weights[1:N];
utruth[2:3:end] = means[1:N];
utruth[3:3:end] = variances[1:N];

# Covariance of noise and prior
γ, σ = .1, 1;

# Observations
y = forward(utruth)

# Square root of covariance matrix
K = length(y)
rtΓ = la.diagm(γ .+ zeros(K))

# Noisy observation
# y = y + rtΓ*randn(K)

# Constraint
function constraint_eq(u)
    weights = u[1:3:end]
    return sum(weights)
end

function constraint_ineq(u)
    weights = u[1:3:end]
    vars = u[3:3:end]
    m = minimum([weights; vars])
    return (m < 0) ? m : 0
end

forward_constraint(u) = [forward(u); constraint_eq(u); constraint_ineq(u)]
y_constraint = [y; 1; 0]

# Covariance matrices
ε = .01
diag_noise_cov = [γ^2 .+ zeros(K); ε^2; ε^2]
noise_cov = la.diagm(diag_noise_cov)
prior_cov = la.diagm(σ^2 .+ zeros(d))

ip = Ip.InverseProblem(forward_constraint, y_constraint, noise_cov, prior_cov)
end
