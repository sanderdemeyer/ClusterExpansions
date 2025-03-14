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

function test_ising(T, p, β, χenv; g = 2.5)
    trunc = β^(p)
    J = T(1.0)

    # twosite_op = rmul!(σᶻᶻ(), -J)
    # onesite_op = rmul!(σˣ(), g * -J)
    twosite_op = rmul!(σˣˣ(T), -J)
    onesite_op = rmul!(σᶻ(T), g)

    pspace = ℂ^2
    spaces = i -> (i >= 0) ? ℂ^(2^(i)) : ℂ^10

    O, O_clust_full = clusterexpansion(T, p, β, twosite_op, onesite_op; spaces = spaces, verbosity = 0, symmetry = "C4")
    O_clust = zeros(ComplexF64, codomain(O_clust_full), domain(O_clust_full))
    O_clust[] = O_clust_full[]

    Magn = TensorMap([1.0 0.0; 0.0 -1.0], pspace, pspace)
    # Magn = TensorMap([0.0 1.0; 1.0 0.0], pspace, pspace)
    # Magn = σᶻ(T)
    Magn = σˣ()
    println(typeof(O_clust))
    println(scalartype(O_clust))
    @tensor Z[-3 -4; -1 -2] := O_clust[1 1; -1 -2 -3 -4]

    println("Dimension of the PEPO = $(summary(Z))")
    pspace = ℂ^2

    partfunc = InfinitePartitionFunction(Z)
    @tensor A_M[-3 -4; -1 -2] := O_clust[1 2; -1 -2 -3 -4] * Magn[2; 1]
    partfunc_M = InfinitePartitionFunction(A_M)

    env0 = CTMRGEnv(partfunc, ComplexSpace(χenv))

    ctm_alg = SimultaneousCTMRG(;
        tol=1e-10,
        miniter=4,
        maxiter=1000,
        verbosity=2,
        svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
    )
    env, = leading_boundary(env0, partfunc, ctm_alg)

    Z = network_value(partfunc, env);
    magn = network_value(partfunc_M, env)
    println("For p = $p, β = $(β), T = $(1/β), magn = $(magn)/$(Z) = $(magn/Z)")
    return Z, magn
end

setprecision(128)
T = Complex{BigFloat}

Ts = LinRange(T(2.2), T(2.35), 10)
βs = [1/T for T = Ts]

p = 6
χenv = 45
g = T(0.0)

Zs = [test_ising(T, p, β, χenv; g = g) for β = βs]

Magn = [i[2]/i[1] for i = Zs]

plt = scatter(Float64.(Ts), abs.((Magn)), label = "p = 6, χ = $(χenv)")
xlabel!("T")
ylabel!("Magnetization")
title!("Ising model with g = $(real(g))")
savefig(plt, "Ising_phase_transition_g_$(real(g))_p_$(p)_chienv_$(χenv).png")
display(plt)

# file = jldopen("ClusterExpansion_g_$(g)_p_$(p)_chienv_$(χenv).jld2", "w")
# file["Ts"] = Ts
# file["Zs"] = Zs
# file["Magn"] = Magn
# close(file)