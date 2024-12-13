using TensorKit
using MPSKitModels
using Graphs
using LongestPaths
using ClusterExpansions
using OptimKit
import PEPSKit: rmul!, σᶻᶻ, σˣ
p = 2
β = 1
D = 2
χenv = 12

J = 1.0
g = 1.0
N1, N2 = (1,1)

# PEPO_dict = get_all_indices(3, β)
# O = get_PEPO(ℂ^2, PEPO_dict)
# println("done")

twosite_op = rmul!(σᶻᶻ(), -1.0)
onesite_op = rmul!(σˣ(), g * -J)

pspace = ℂ^2

cluster = [(0,0),(1,0),(1,1),(0,1)]

α = 40
ϵ = 1e-15
max_iter = 200
exp_H = exponentiate_hamiltonian(twosite_op, cluster, β, length(cluster))
levels_to_update, solution, err = solve_4_loop(α, exp_H; ϵ = ϵ, max_iter = max_iter);
tensors, err = solve_4_loop_periodic(α, exp_H; ϵ = ϵ, max_iter = max_iter);

t = ncon([tensors[1], tensors[2]], [[-1 -2 -3 1], [-4 -5 1 -6]])
t = permute(t, ((1,2,3),(4,5,6)))
U, Σ, V = tsvd(t)

D = [Σ[i,i] for i = 1:size(convert(Array,Σ))[1]]

plt = plot(1:160, D)
display(plt)
