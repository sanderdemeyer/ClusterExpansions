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
H = transverse_field_ising(ComplexF64, Trivial, InfiniteSquare(N1,N2); J = J, g = g)
ψ₀ = InfinitePEPS(2, D; unitcell=(N1,N2))

O = clusterexpansion(p, β, twosite_op, onesite_op)
Magn = TensorMap([1.0 0.0; 0.0 -1.0], pspace, pspace)

@tensor Z[-1 -2 -3 -4] := O[1 1; -1 -2 -3 -4]
env = CTMRGEnv(O, ComplexSpace(χenv));

