using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfinitePartitionFunction, LocalOperator, vertices

D = 3
χenv = 20
χenv_approx = 20

symmetry = "C4"
critical = false

setprecision(128)
T = Complex{BigFloat}

if critical
    J = 1.0
    g = 3.1
    e = -1.6417 * 2
    mˣ = 0.91
else
    J = 1.0
    g = 2.0
    e = -1.2379 * 2
    mˣ = 0.524
end

g = 1.0

pspace = ℂ^2
vspace = ℂ^D
twosite_op = rmul!(σᶻᶻ(T), -J)
onesite_op = rmul!(σˣ(T), g * -J)

spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : ℂ^30


c = [(0, 0), (0, 1), (1, 0), (2, 0), (2, -1), (2, 1), (3, 0), (4, 0)]
cluster = Cluster(c)


βs = [10.0^(x) for x in LinRange(-2, -1, 10)]
ps = [2 3 4 5]

errors = zeros(length(βs), length(ps))

for (j,p) = enumerate(ps)
    for (i,β) = enumerate(βs)
        println("beta = $(β), p = $p")
        O, O_clust = clusterexpansion(T, p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = "C4", verbosity = 0)

        exp_exact = exponentiate_hamiltonian(T, twosite_op, cluster, β)
        exp_approx = contract_PEPO(T, cluster, O, spaces)

        error = norm(exp_exact-exp_approx)/norm(exp_approx)
        println("error = $(error)")
        errors[i,j] = error
    end
end

using Plots
plt = scatter(βs, errors[:,1], label = "p = 2")
scatter!(βs, errors[:,2], label = "p = 3")
scatter!(βs, errors[:,3], label = "p = 4")
scatter!(βs, errors[:,4], label = "p = 5")
scatter!(xscale=:log10, yscale=:log10)
xlabel!("β*J")
ylabel!("ϵ")
display(plt)