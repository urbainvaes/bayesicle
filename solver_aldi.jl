module Aldi
import Statistics
import Random
import LinearAlgebra
la = LinearAlgebra

struct Config
    dt::Float64
    dtmax::Float64
    reg::Bool
    opti::Bool
    adaptive::Bool
end

function step(ip, config, ensembles; verbose=false)
    if la.norm(ensembles - ensembles) != 0.0
        error("Ensembles contain NaNs!")
    end

    d, J = size(ensembles)
    K = size(ip.noise_cov)[1]

    function vprint(arg)
        if verbose
            print(arg)
        end
    end


    fensembles = zeros(K, J)
    for (i, e) in enumerate(eachcol(ensembles))
        fensembles[:, i] = ip.forward(e)
        vprint(".")
    end
    vprint("\n")

    mean_ensembles = Statistics.mean(ensembles, dims=2)
    mean_fensembles = Statistics.mean(fensembles, dims=2)

    sqrt_cov_theta = (1/sqrt(J)) * (ensembles .- mean_ensembles)
    cov_theta = sqrt_cov_theta*sqrt_cov_theta'

    drifts = zeros(d, J)
    diff_data = fensembles .- ip.observation
    diff_mean_forward = fensembles .- mean_fensembles
    diff_mean = ensembles .- mean_ensembles
    coeffs = (1/J) * diff_mean_forward' * (ip.inv_noise_cov * diff_data)
    drifts = - ensembles * coeffs
    if config.reg
        drifts += - cov_theta * ip.inv_prior_cov * ensembles
    end

    # Aldi part
    drifts += (d+1)/J * (ensembles .- mean_ensembles)

    effective_dt = config.dt
    if config.adaptive
        println("Adapting")
        norm_mat_drift = la.norm(coeffs, 2)
        effective_dt = config.dt/(config.dt/config.dtmax + norm_mat_drift)
        println("Norm of drift: $norm_mat_drift")
        println("New time step: $effective_dt")
    end

    new_ensembles = ensembles + effective_dt*drifts
    if !config.opti
        dw = sqrt(effective_dt) * Random.randn(J, J)
        new_ensembles += sqrt(2) * sqrt_cov_theta * dw
    end

    return new_ensembles
end

end
