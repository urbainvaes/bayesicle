module CbsMetro
import Random
include("lib_inverse_problem.jl")

export Config, step

struct Config
    alpha::Float64
    beta::Float64
    adaptive::Bool
    ess::Float64
end

function find_beta(fensembles, ess)
    J = length(fensembles)
    function get_ess(β)
        weights = exp.(- β*fensembles)
        return sum(weights)^2/sum(weights.^2)/J
    end
    β1, β2 = 0, 1e20
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

function step(problem, config, ensembles, fensembles=nothing; verbose=false)
    if occursin("InverseProblem", string(typeof(problem)))
        objective(u) = Ip.reg_least_squares(problem, u)
    else
        objective = problem
    end

    alpha = config.alpha
    adaptive = config.adaptive
    ess = config.ess

    d, J = size(ensembles)
    if fensembles == nothing
        fensembles = zeros(J)
        for i in 1:length(fensembles)
            fensembles[i] = objective(ensembles[:, i])
            print(".")
        end
        println("")
    end

    fensembles = fensembles .- minimum(fensembles);
    beta = adaptive ? find_beta(fensembles, ess) : config.beta
    weights = exp.(- beta*fensembles);
    weights = weights / sum(weights);

    mean = sum(ensembles.*weights', dims=2) |> vec
    difference = ensembles .- mean
    cov = zeros(d, d)
    for j in 1:J
        cov += weights[j] * difference[:, j]*difference[:, j]'
    end
    coeff_noise = sqrt(1 - alpha^2) * sqrt(cov*(1+beta))
    coeff_noise = real(coeff_noise)

    index = ceil(Int, Random.rand()*J)
    theta = ensembles[:, index]
    ftheta = fensembles[index]
    proposal = mean + alpha*(theta - mean) + coeff_noise*Random.randn(d)
    fproposal = objective(proposal)

    # Calculate acceptance probability
    inv_cov = inv(cov)

    # This is approximate: when J >> 1, moving a particle does not change
    # the proposal much
    log_acceptance_proba_1 = 1/2 * (proposal-mean)'inv_cov*(proposal-mean) -
                             1/2 * (theta-mean)'inv_cov*(theta-mean)
    log_acceptance_proba_2 = ftheta - fproposal
    acceptance_proba = min(1, exp(log_acceptance_proba_1 + log_acceptance_proba_2))

    accept = Random.rand() <= acceptance_proba
    if accept
        ensembles[:, index] = proposal
        fensembles[index] = fproposal
    end

    return accept, ensembles, fensembles
end

end
