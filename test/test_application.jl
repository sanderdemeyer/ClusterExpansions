using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare
using Test

function compare_steps(O, O_full, A)
    A_full = copy(A)

    W, A, error = find_truncation(A_full, O_full; verbosity = 0, c = 0);
    A_full = apply(A_full, O_full)
    
    for _ = 1:steps
        W, A, error = find_truncation(A, O; verbosity = 0, c = 0);
        A = applly(A, O)
    end
    
    return norm(A_full - A)
end

p = 3
β = 1e-3
steps = 5
D = 2
χenv = 8

symmetry = "C4"

J = 1.0
g = 0.0
N1, N2 = (1,1)

pspace = ℂ^2
vspace = ℂ^2
twosite_op = rmul!(σᶻᶻ(), -1.0)
onesite_op = rmul!(σˣ(), g * -J)

spaces = i -> (i >= 0) ? ℂ^(2^(i)) : ℂ^10
_, O_clust_full = clusterexpansion(p, 5*β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = symmetry)
_, O_clust = clusterexpansion(p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = symmetry)

A = TensorMap(randn, pspace, vspace ⊗ vspace ⊗ vspace' ⊗ vspace')
A = flip_arrows(make_translationally_invariant(flip_arrows(A)))

error = compare_steps(O_clust, O_clust_full, A)

@test error < 1e-5
