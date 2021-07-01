module pCN
import Statistics
import Random
import LinearAlgebra
la = LinearAlgebra

include("lib_inverse_problem.jl")

export Config, step

# Only multiplicative noise for now
struct Config
    beta::Float64
    precond_vec::Array{Float64}
    precond_mat::Matrix{Float64}
end

function step(ip, config, theta, ftheta)

    # Dimensions
    d = length(theta)
    theta = reshape(theta, d)

    # For convenience
    m = reshape(config.precond_vec, d)
    C = config.precond_mat
    β = config.beta

    # Precomputations
    sqrt_C = real(sqrt(C))
    inv_C = inv(C)

    # Generate proposal
    proposal = m + sqrt(1 - β^2) * reshape((theta - m), d) +
               β * sqrt_C * Random.randn(d)

    # First iteration
    if ftheta == -1
        ftheta = Ip.reg_least_squares(ip, theta)
    end

    # Calculate value of least-squares functional at proposal
    fproposal = Ip.reg_least_squares(ip, proposal)

    # Calculate acceptance probability
    log_acceptance_proba_1 = 1/2 * (proposal-m)'inv_C*(proposal-m) -
                             1/2 * (theta-m)'inv_C*(theta-m)
    log_acceptance_proba_2 = ftheta - fproposal
    acceptance_proba = min(1, exp(log_acceptance_proba_1 + log_acceptance_proba_2))

    if Random.rand() <= acceptance_proba
        return true, proposal, fproposal
    end

    return false, theta, ftheta
end

end
