using Gridap
import LinearAlgebra
import Random
import Statistics

# Shorthand names for modules
la = LinearAlgebra

N = 2
indices = [(m, n) for m in 0:N for n in 0:N];
indices = sort(indices, by=maximum)

function assemble_diffusivity(u)
    # Eigenvalues of the covariance operator
    τ, α = 3, 2
    eig_v = [(π^2*norm(i)^2 + τ^2)^(-α) for i in indices]
    function diffusivity(x)
        result = 0.
        for (i, ind) in enumerate(indices)
            eigf = cos(ind[1]*π*x[1]) * cos(ind[2]*π*x[2])
            result += u[i] * sqrt(eig_v[i]) * eigf
        end
        return exp(result)
    end
    return diffusivity
end

d = length(indices)
utruth = zeros(d) .+ 1.

rhs(x) = 50
domain = (0,1,0,1)
partition = (20,20)
model = CartesianDiscreteModel(domain,partition)

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V0 = TestFESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
U = TrialFESpace(V0, x -> 0)

degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

# Find node indices of observations
function get_obs_nodes()
    M = 10
    grid = 1/M * (1:M-1) |> collect
    xobs = grid * ones(M-1)'
    yobs = ones(M-1)  * grid'
    xobs = reshape(xobs, ((M - 1)^2, 1))
    yobs = reshape(yobs, ((M - 1)^2, 1))
    nodes = Ω.node_coords

    obs_nodes = zeros(Int, (M-1)^2)
    iobs = 1

    for (inode, node) in enumerate(nodes)
        if iobs > length(obs_nodes)
            return obs_nodes
        end
        vec = [node[1], node[2]]
        obs = [xobs[iobs], yobs[iobs]]
        if norm(vec - obs) < 1e-10
            obs_nodes[iobs] = inode
            iobs += 1
        end
    end
end

# Evaluate finite element function at nodes
function eval_at_nodes(uh)
    nodes = Ω.node_coords
    elements = Ω.cell_node_ids;
    dof_values = uh.cell_dof_values
    node_values = zeros(length(nodes))
    for (ielem, dof_values) in enumerate(dof_values)
        for (idof, dofvalue) in enumerate(dof_values)
            inode = elements[ielem][idof]
            node_values[inode] = dofvalue
        end
    end
    return node_values
end

obs_nodes = get_obs_nodes()
function forward(u)
    print(".")
    if length(u) != length(indices)
        println("Wrong input!")
    end
    diff = assemble_diffusivity(u)
    a(u,v) = ∫( diff * ∇(v) ⋅ ∇(u) )*dΩ
    b(v) = ∫( v*rhs )*dΩ
    op = AffineFEOperator(a,b,U,V0)
    uh = solve(op)
    uh_nodes = eval_at_nodes(uh)
    uh_obs = uh_nodes[obs_nodes]
end

# Covariance of noise and prior
γ, σ = .01, 1

# Truth
Random.seed!(3)
utruth = randn(d)
y = forward(utruth)

# Square root of covariance matrix
K = length(y)
rtΓ = la.diagm(γ .+ zeros(K))

# Noisy observation
y = y + rtΓ*randn(K)

function least_squares(u)
    invΣ = la.diagm(1/σ^2 .+ zeros(d))
    invΓ = la.diagm(1/γ^2 .+ zeros(K))
    misfit = y - forward(u)
    return misfit' * (invΓ*misfit) + u' * (invΣ*u)
end

objective = least_squares

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

Random.seed!(0);
alpha = 0
beta = 1
opti = false
adaptive = true
ess = 1/2.
config = CbsConfig(alpha, beta, opti, adaptive, ess)

# Julia is column-major!
J = 256
ensembles = 3*Random.randn(d, J)
distance, spread, niter = 10, 1, 0
while spread > 1e-12
    mean = Statistics.mean(ensembles, dims=2)
    cov = Statistics.cov(ensembles, dims=2)
    inv_cov = la.inv(cov)
    weighted_distance = sqrt((utruth - mean)' * (inv_cov * (utruth - mean)))
    spread = sqrt(sum(abs2, Statistics.cov(ensembles, dims=2)))
    println("Spread = $spread, Error=$weighted_distance")
    ensembles = cbs_step(objective, config, ensembles)
    niter += 1
end
