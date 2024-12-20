using TensorKit
using MPSKitModels
using Graphs
using OptimKit
using ClusterExpansions
import PEPSKit: rmul!, σᶻᶻ, σˣ

p = 2
β = 1
D = 2
χenv = 12

J = 1.0
g = 1.0
N1, N2 = (1,1)

twosite_op = rmul!(σᶻᶻ(), -1.0)
onesite_op = rmul!(σˣ(), g * -J)

pspace = ℂ^2

O = clusterexpansion(p, β, twosite_op, onesite_op)
