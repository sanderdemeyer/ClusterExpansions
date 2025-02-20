using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ

p = 4
β = 1e-4

levels_convention = "tree_depth"
J = 1.0
g = 0.0
N1, N2 = (1,1)

cluster_1 = [(0,0), (1,0), (1,-1), (2,-1), (1,1), (2,1), (3,1),(4,1), (2,2),(2,3)]
cluster_2 = [(0,0), (1,0), (1,-1), (2,-1), (1,1), (1,2), (2,2), (3,2),(4,2), (2,3),(2,4)]

twosite_op = rmul!(σᶻᶻ(), -1.0)
onesite_op = rmul!(σˣ(), g * -J)


O = clusterexpansion(p, β, twosite_op, onesite_op; levels_convention = "tree_depth");

