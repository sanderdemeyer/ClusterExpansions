using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare
using Test
using Random
using LinearAlgebra

import ClusterExpansions: contract_tensors_symmetric

function flip_arrows(A::AbstractTensorMap{S,E,2,2} 
    ) where {S<:ElementarySpace,E}
    I2 = isometry(A.codom[2]', (A.codom[2]))
    I4 = isometry(A.dom[1], (A.dom[1])')
    @tensor A_flipped[-1 -2; -3 -4] := A[-1 2; 1 -4] * I4[1; -3] * I2[2; -2]
    return A_flipped
end

Random.seed!(1654843513)
setprecision(128)

T = Complex{BigFloat}
β = 1e-2
twosite_op = σᶻᶻ(T)

cluster = Cluster([(0,0), (1,0), (1,1), (0,1)])

exp_H = exponentiate_hamiltonian(T, twosite_op, cluster, β)
Vspace = ℂ^12
levels_to_update = [(-1, -1, 0, 0) (0, -1, -1, 0) (0, 0, -1, -1) (-1, 0, 0, -1)]

spaces = [2 3]
truncations = [truncdim(s) for s = spaces]

As, error, _ = solve_4_loop(exp_H, Vspace, levels_to_update; verbosity = 0);

println("Error in loop solver is $(error)")
@test error < 1e-36
