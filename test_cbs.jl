#!/usr/bin/env julia
import Random
import Statistics

shift = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 2;
fun = length(ARGS) > 1 ? ARGS[2] : "rastrigin";
n = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 10;
exact = shift * ones(n, 1);

function rastrigin(x)
    A = 10
    result = A*n
    rootpow = 1
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

objective = rastrigin
if fun == "ackley"
    objective = ackley
end

struct CbsConfig
    alpha::Float64
    beta::Float64
    opti::Bool
    adaptive::Bool
    ess::Float64
end

function find_beta(fensembles, ess)
    J = length(fensembles)
    function get_ess(β)
        weights = exp.(- β*fensembles)
        return sum(weights)^2/sum(weights.^2)/J
    end
    β1, β2 = 0, 1e15
    β, e = β2, get_ess(β2)
    if e > ess
        println("Can't find β")
        return β
    end
    while abs(e - ess) > 1e-2
        β = (β1 + β2)/2
        e = get_ess(β)
        if e > ess
            β1 = β
        else
            β2 = β
        end
    end
    return β
end

function cbs_step(objective, config, ensembles)
    alpha = config.alpha
    opti = config.opti
    adaptive = config.adaptive
    ess = config.ess

    d, J = size(ensembles)
    fensembles = zeros(J)
    for i in 1:length(fensembles)
        fensembles[i] = objective(ensembles[:, i])
    end

    fensembles = fensembles .- minimum(fensembles)
    beta = adaptive ? find_beta(fensembles, ess) : config.beta
    weights = exp.(- beta*fensembles)
    weights = weights / sum(weights)

    mean = sum(ensembles.*weights', dims=2)
    diff = ensembles .- mean
    cov = zeros(d, d)
    for j in 1:J
        cov += weights[j] * diff[:, j]*diff[:, j]'
    end
    coeff_noise = sqrt(1 - alpha^2) * sqrt(cov*(opti ? 1 : (1+beta)))
    coeff_noise = real(coeff_noise)
    new_ensembles = mean .+ alpha.*diff .+ coeff_noise*Random.randn(d, J)
end

function run_simulation(alpha, J)
    Random.seed!(0);
    beta = 1
    opti = true
    adaptive = true
    ess = 1/2.
    config = CbsConfig(alpha, beta, opti, adaptive, ess)

    nsimuls = 100
    successes = zeros(nsimuls)
    errors = zeros(nsimuls)
    niters = zeros(nsimuls)

    for i in 1:nsimuls
        # Julia is column-major!
        ensembles = 3*Random.randn(n, J)
        distance, spread, niter = 10, 1, 0
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
        errors[i] = success ? distance : 0
        niters[i] = niter
        # println("Iteration $i — success: $success, distance: $distance, niter: $niter")
    end

    success_rate = round(Int, sum(successes)/length(successes) * 100);
    mean_niter = ceil(Int, Statistics.mean(niters));
    mean_error = sum(errors)/sum(successes);
    if mean_error > 0
        exp_mean_error = floor(log10(mean_error));
        root_mean_error = round(mean_error / 10^exp_mean_error, digits=2);
        exp_mean_error = round(Int, exp_mean_error);
        mean_error = "$root_mean_error \\times 10^{$exp_mean_error}"
    else
        mean_error = "-"
    end
    # println("J=$J, d=$n, α=$alpha, b=$shift, Success rate: $success_rate, mean niter: $mean_niter, mean error: $mean_error");
    println("& \$ $success_rate \\% \\,|\\, $mean_niter \\,|\\, $mean_error \$          J=$J, d=$n, α=$alpha, b=$shift");
end

alphas = [0, .5]
Js = [100, 500, 1000]

for alpha in alphas
    for J in Js
        run_simulation(alpha, J)
    end
end
