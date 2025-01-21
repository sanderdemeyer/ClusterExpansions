using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
using Plots
using JLD2
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfinitePartitionFunction

function test_ising(p, β, χenv; g = 2.5)
    J = 1.0

    # twosite_op = rmul!(σᶻᶻ(), -J)
    # onesite_op = rmul!(σˣ(), g * -J)
    twosite_op = rmul!(σˣˣ(), -J)
    onesite_op = rmul!(σᶻ(), g)

    pspace = ℂ^2
    spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : ℂ^10

    O = clusterexpansion(p, β, twosite_op, onesite_op, spaces = spaces, verbosity = 0)
    Magn = TensorMap([1.0 0.0; 0.0 -1.0], pspace, pspace)
    # Magn = TensorMap([0.0 1.0; 1.0 0.0], pspace, pspace)
    Magn = σᶻ()

    @tensor Z[-3 -4; -1 -2] := O[1 1; -1 -2 -3 -4]

    println("Dimension of the PEPO = $(summary(Z))")
    pspace = ℂ^2
    T = TensorMap(randn, pspace, pspace ⊗ pspace ⊗ pspace' ⊗ pspace')

    partfunc = InfinitePartitionFunction(Z)
    @tensor A_M[-3 -4; -1 -2] := O[1 2; -1 -2 -3 -4] * Magn[2; 1]
    partfunc_M = InfinitePartitionFunction(A_M)

    env0 = CTMRGEnv(partfunc, ComplexSpace(χenv))

    ctm_alg = CTMRG(;
        tol=1e-10,
        miniter=4,
        maxiter=100,
        verbosity=2,
        svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
        ctmrgscheme=:simultaneous,
    )
    env = leading_boundary(env0, partfunc, ctm_alg)

    Z = norm(partfunc, env);
    magn = norm(partfunc_M, env)
    println("For p = $p, β = $(β), T = $(1/β), magn = $(magn)/$(Z) = $(magn/Z)")
    return Z, magn
end

Ts = LinRange(0.7, 1.8, 10)
βs = [1/T for T = Ts]

p = 3
χenv = 16
g = 2.5

Zs = [test_ising(p, β, χenv; g = g) for β = βs]

Magn = [i[2]/i[1] for i = Zs]

file = jldopen("ClusterExpansion_g_$(g)_p_$(p)_chienv_$(χenv).jld2", "w")
file["Ts"] = Ts
file["Zs"] = Zs
file["Magn"] = Magn
close(file)