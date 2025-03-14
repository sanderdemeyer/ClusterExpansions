using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfiniteSquareNetwork, InfinitePartitionFunction, LocalOperator, vertices
using BlockTensorKit
using BlockTensorKit: ⊕



p = 5
β = 1.0
χenv = 4

symmetry = "C4"

setprecision(128)
T = Complex{BigFloat}

# SF model
t = T(1.0)
V = T(-2.5)
V = T(0.0)
μ = 2*V

kinetic_operator = -t * (c_plusmin(T) + c_minplus(T))
number_operator = c_number(T)
@tensor number_twosite[-1 -2; -3 -4] := number_operator[-1; -3] * number_operator[-2; -4]
onesite_op = rmul!(number_operator, -μ)
twosite_op = rmul!(kinetic_operator, -t) + rmul!(number_twosite, V)

pspace = domain(onesite_op)[1]

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

O, O_clust = clusterexpansion(T, p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = "C4", verbosity = 3)

O_clust = convert(TensorMap, O_clust)
# O_clust_approx = TensorMap(O_clust.data, ComplexF64, codomain(O_clust), domain(O_clust))
O_clust_approx = TensorMap(convert(Array{ComplexF64}, O_clust.data), codomain(O_clust), domain(O_clust))

# O_clust_approx = TensorMap(convert(Array{ComplexF64}, convert(Array, O_clust)), codomain(O_clust), domain(O_clust))

@tensor Z[-3 -4; -1 -2] := O_clust[1 1; -1 -2 -3 -4]
@tensor A_M[-3 -4; -1 -2] := O_clust[1 2; -1 -2 -3 -4] * number_operator[2; 1]

partfunc = InfinitePartitionFunction(Z)
partfunc_M = InfinitePartitionFunction(A_M)

envspace = Vect[I](0 => χenv / 2, 1 => χenv / 2)

env0 = CTMRGEnv(partfunc, envspace)

ctm_alg = SimultaneousCTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=1000,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
)
env, = leading_boundary(env0, partfunc, ctm_alg)

occupancy = network_value(partfunc, env) / network_value(partfunc_M, env);
println("Occupancy of this model is $(occupancy)")
