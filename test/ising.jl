using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfinitePartitionFunction

function test_ising(p, β, χenv)
    J = 1.0
    g = 0.0

    twosite_op = rmul!(σᶻᶻ(), -J)
    onesite_op = rmul!(σˣ(), g * -J)

    pspace = ℂ^2

    O = clusterexpansion(p, β, twosite_op, onesite_op)
    Magn = TensorMap([1.0 0.0; 0.0 -1.0], pspace, pspace)

    @tensor Z[-3 -4; -1 -2] := O[1 1; -1 -2 -3 -4]

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
    error_Z = abs(Z-2*(cosh(β*J)^2))
    magn = norm(partfunc_M, env)
    println("error = $(error_Z)/$(Z)")
    return Z, error_Z, magn
end

βs = [10.0^(-i/3) for i = -3:10]

Z_num_12 = [test_ising(2, β, 12)[1] for β = βs]
Z_num_24 = [test_ising(4, β, 24)[1] for β = βs]
Z_exact = [2*(cosh(β)^2) for β = βs]

using Plots
plt = scatter(log.(βs), real.(Z_num_12), label = "p = 2, chi = 12", ylims = (1,5), xlabel = "log(β)", ylabel = "Partition function Z")
# scatter!(log.(βs), real.(Z_num_24), label = "p = 2, chi = 24", ylims = (1,5))
scatter!(log.(βs), Z_exact, label = "exact", ylims = (1,5))
display(plt)