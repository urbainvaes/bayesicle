module Multiscale
import Statistics
import Random
import LinearAlgebra
la = LinearAlgebra

export Config, step

# Only multiplicative noise for now
struct Config
    sigma::Float64
    delta::Float64
    dt::Float64
    dtmax::Float64
    reg::Bool
    opti::Bool
    adaptive::Bool
    precond_mat::Matrix{Float64}
end

function step(ip, config, theta, xis)
    d = length(theta)
    K = size(ip.noise_cov)[1]
    J = size(xis)[2]

    sqrt_precond_mat = real(sqrt(config.precond_mat))
    thetas = theta .+ config.sigma * sqrt_precond_mat * xis

    ftheta = ip.forward(theta)
    fensembles = zeros(K, J)
    for (i, e) in enumerate(eachcol(thetas))
        fensembles[:, i] = ip.forward(e)
        print(".")
    end
    println("")

    if config.opti || config.reg
        Cxi = (1/J) * xis * xis'
        Dxi = sqrt_precond_mat * Cxi * sqrt_precond_mat
        sqrt_Dxi = real(sqrt(Dxi))
    end

    diff_data = ftheta .- ip.observation
    diff_ftheta = fensembles .- ftheta
    coeffs = 1/(J*config.sigma^2) * diff_ftheta' * (ip.inv_noise_cov * diff_data)
    diff_thetas = thetas .- theta
    drift = - diff_thetas * coeffs

    if config.reg
        drift += - Dxi * ip.inv_prior_cov * theta
    end

    effective_dt = config.dt
    if config.adaptive
        norm_mat_drift = la.norm(coeffs, 2)
        effective_dt = config.dt/(config.dt/config.dtmax + norm_mat_drift)
        println("Norm of drift: $norm_mat_drift")
        println("New time step: $effective_dt")
    end

    new_theta = theta + effective_dt*drift
    if !config.opti
        dw = sqrt(effective_dt) * Random.randn(d)
        new_theta += sqrt(2)*sqrt_Dxi * dw
    end

    alpha = exp(-effective_dt/config.delta^2)
    new_xis = alpha*xis + sqrt(1-alpha^2) * Random.randn(d, J)

    return new_theta, new_xis
end

end
