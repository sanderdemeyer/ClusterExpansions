using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
using Test
using Random
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare
import ClusterExpansions: rotl90_fermionic

function check_rotational_invariance(O)
    for (key, value) = O
        rotated_key = Tuple(circshift(collect(key), 1))
        error = norm(value - rotl90_fermionic(O[rotated_key]))/norm(value)
        println("Error for key = $(key) is $(error)")
        if error > 1e-10
            println("Assymetric for key = $(key), with error $(error)")
        end
    end
end

function compare_steps(O, O_full, A, steps; verbosity = 2)
    A_full = copy(A)
    A_steps = copy(A)

    Ws, _, error = find_truncation(A_full, O_full; verbosity = verbosity, c = 0);
    A_full = apply(A_full, O_full, Ws)
    
    for _ = 1:steps
        Ws, _, error = find_truncation(A_steps, O; verbosity = verbosity, c = 0);
        A_steps = apply(A_steps, O, Ws; spaces = [vspace], verbosity = 3)
    end
    return norm(A_full - A_steps)/norm(A_full - A), A_full, A_steps
end

Random.seed!(156489748)

p = 4
β = 1e-1
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
O_full, O_clust_full = clusterexpansion(p, 5*β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = symmetry, verbosity = 0)
O, O_clust = clusterexpansion(p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = symmetry, verbosity = 0)

A = TensorMap(randn, pspace, vspace ⊗ vspace ⊗ vspace' ⊗ vspace')
A = flip_arrows(make_translationally_invariant(flip_arrows(A)))

error, B_full, B = compare_steps(O_clust, O_clust_full, A, steps)

println("Error = $(error)")
@test error < 1e-5
