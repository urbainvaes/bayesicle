using Gridap
using SharedArrays
using Distributed

f(x) = 50
domain = (0,1,0,1)
partition = (100,100)
model = CartesianDiscreteModel(domain,partition)

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V0 = TestFESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
U = TrialFESpace(V0, x -> 0)

degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

A = SharedArray{Int}(10)
@distributed for i = 1:10
    A[i] = i
end

k(x) = 1 + x[1]^2
a(u,v) = ∫( k * ∇(v) ⋅ ∇(u) )*dΩ
b(v) = ∫( v*f )*dΩ
op = AffineFEOperator(a,b,U,V0)
uh = solve(op)

@sync @distributed for k0 = 1:100
    println(k0)
    k(x) = k0 + sin(8π*x[1])
    a(u,v) = ∫( k * ∇(v) ⋅ ∇(u) )*dΩ
    b(v) = ∫( v*f )*dΩ
    op = AffineFEOperator(a,b,U,V0)
    uh = solve(op)
end
println("done")

# writevtk(Ω,"uh",cellfields=["uh" => uh])
