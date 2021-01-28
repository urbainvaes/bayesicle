#!/usr/bin/env julia
import Random
import Statistics
Random.seed!(0);

n = 2;
rootpow = 1;
shift = 0;
exact = shift * ones(n, 1);

function rastrigin(x)
    A = 10
    result = A*n
    for i in 1:n
        z = rootpow^i * (x[i] - shift)
        result += z^2 - A*cos(2π*z)
    end
    return result
end

function ackley(x)
    A = 20 + exp(1)
    z = x .- shift
    result = 20 + exp(1) - 20 * exp(-.2*sqrt(1/n*sum(abs2, z))) - exp(1/n*sum(cos.(2π*z)))
end


struct CbsConfig
    alpha::Float64
    beta::Float64
    opti::Bool
    adaptive::Bool
    frac_min::Float64
    frac_max::Float64
end

function cbs_step(objective, config, ensembles)
    α = config.alpha
    beta = config.beta
    opti = config.opti
    adaptive = config.adaptive
    frac_min = 1/5.
    frac_max = 1/2.

    d, J = size(ensembles)
    fensembles = zeros(J)
    for i in 1:length(fensembles)
        fensembles[i] = objective(ensembles[:, i])
    end

    nitermax = 1000
    fensembles = fensembles .- minimum(fensembles)
    converged = false
    for i in 1:nitermax
        weights = exp.(- beta*fensembles)
        sum_ess = sum(weights.^2)
        ess = sum_ess == 0 ? 0 : sum(weights)^2/sum_ess
        if adaptive && ess < frac_min*J
            beta /= 1.1
            # println("ESS = $ess too small, decreasing β to $beta")
        elseif adaptive && ess > frac_max*J
            beta *= (beta < 1e15 ? 1.1 : 1)
            # println("ESS = $ess too large, increasing β to $beta")
        else
            converged = true
            break
        end
    end
    if !converged
        println("Could not find suitable β")
    end
    weights = weights / sum(weights)

    mean = sum(ensembles.*weights', dims=2)
    diff = ensembles .- mean
    cov = zeros(d, d)
    for j in 1:J
        cov += weights[j] * diff[:, j]*diff[:, j]'
    end
    coeff_noise = sqrt(1 - α^2) * sqrt(cov*(opti ? 1 : (1+beta)))
    coeff_noise = real(coeff_noise)
    new_ensembles = mean .+ α.*diff .+ coeff_noise*Random.randn(d, J)
end

J = 100

alpha = .001
beta = 1
opti = true
adaptive = false
frac_min = 1/5.
frac_max = 1/2.
objective = ackley
config = CbsConfig(alpha, beta, opti, adaptive, 0, 0)

nsimuls = 1000
successes = zeros(1000)
errors = zeros(1000)

for i in 1:nsimuls
    spread, niter = 1, 0
    # Julia is column-major!
    # ensembles = -3*ones(n, 1) .+ 3*Random.rand(n, J)
    ensembles = 5*Random.randn(n, J)
    distance = 10
    while spread > 1e-12
        mean = Statistics.mean(ensembles, dims=2)
        distance = maximum(abs.(mean - exact))
        spread = sqrt(sum(abs2, Statistics.cov(ensembles, dims=2)))
        # println("Spread = $spread, L^∞ distance = $distance")
        ensembles = cbs_step(objective, config, ensembles)
        niter += 1
    end
    success = distance < .25
    successes[i] = success
    errors[i] = distance
    println("Iteration $i — success: $success, distance: $distance, niter: $niter") 
end
