using Plots
using TensorKit
using MPSKitModels
using Graphs
using OptimKit
using ClusterExpansions
import PEPSKit: rmul!, σᶻᶻ, σˣ

p = 4
β = 1
D = 2
χenv = 12

J = 1.0
g = 1.0
N1, N2 = (1,1)

twosite_op = rmul!(σᶻᶻ(), -1.0)
onesite_op = rmul!(σˣ(), g * -J)

pspace = ℂ^2
α = 10

c = [(0,0),(0,1),(0,2),(1,2),(1,1),(1,0)]
cluster = Cluster(c)
exp_H = exponentiate_hamiltonian(twosite_op, cluster, β)

Ns = [1, 0, 1, 0]
C = nothing
errors = [solve_N_loop(Ns, C, exp_H; α = α, line_search = true, linesearch_options = n-1)[2] for n = 1:11]

plt = plot(1:length(errors[2]), log.(errors[2]), label = "noo = 1")
for n = 3:10
    plot!(1:length(errors[n]), log.(errors[n]), label = "noo = $(n)")
end
display(plt)