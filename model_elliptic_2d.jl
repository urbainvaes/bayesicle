
module ModelElliptic2d

using Gridap
import Random
import LinearAlgebra
la = LinearAlgebra

include("lib_inverse_problem.jl")
export utruth, least_squares, d, y, noise_cov, prior_cov

N = 3
# N = 1
indices = [(m, n) for m in 0:N for n in 0:N];
indices = sort(indices, by=x->(maximum(x), sum(x), x[1]))

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
partition = (100,100)
# partition = (20,20)
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
utruth = σ*randn(d)
y = forward(utruth)

# Square root of covariance matrix
K = length(y)
rtΓ = la.diagm(γ .+ zeros(K))

# Noisy observation
y = y + rtΓ*randn(K)

# Covariance matrices
noise_cov = la.diagm(γ^2 .+ zeros(K))
prior_cov = la.diagm(σ^2 .+ zeros(d))

ip = Ip.InverseProblem(forward, y, noise_cov, prior_cov)
end
