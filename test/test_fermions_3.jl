using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfinitePartitionFunction, LocalOperator, vertices
using JLD2

symmetry = "C4"
critical = false

setprecision(128)
T = Complex{BigFloat}

p = 3
β = 1e-2

t = T(1.0)
V = T(-2.5)
# V = T(0.0)
μ = 2*V

c = [(0,0),(0,1),(1,1)]
cluster = Cluster(c)

kinetic_operator = -t * (c_plusmin(T) + c_minplus(T))
number_operator = c_number(T)
@tensor number_twosite[-1 -2; -3 -4] := number_operator[-1; -3] * number_operator[-2; -4]
onesite_op = rmul!(number_operator, -μ)
twosite_op = rmul!(kinetic_operator, -t) + rmul!(number_twosite, V)

I = fℤ₂
function spaces_dict(I, i)
    if i == 0
        return Vect[I](0 => 1)
    elseif i < 0
        return Vect[I](0 => 5, 1 => 5)
    else
        return Vect[I](0 => 2^(2*i-1), 1 => 2^(2*i-1))
    end
end

spaces = i -> spaces_dict(I, i)

O, O_clust = clusterexpansion(T, p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = "C4", verbosity = 2);
println("Done")
