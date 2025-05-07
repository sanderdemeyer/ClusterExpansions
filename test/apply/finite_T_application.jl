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

function calculate_magnetization(O, χenv, Magn, ctm_alg)
    println(summary(O))
    @tensor Z[-3 -4; -1 -2] := O[1 1; -1 -2 -3 -4]
    partfunc = InfinitePartitionFunction(Z)
    @tensor A_M[-3 -4; -1 -2] := O[1 2; -1 -2 -3 -4] * Magn[2; 1]
    partfunc_M = InfinitePartitionFunction(A_M)

    env0 = CTMRGEnv(partfunc, ComplexSpace(χenv))
    env, = leading_boundary(env0, partfunc, ctm_alg)

    Z = network_value(partfunc, env);
    magn = network_value(partfunc_M, env)
    println("Z = $(Z), magn = $(magn)")
    return Z/magn
end

function get_phase_transition(β₀, O, Magn, trunc_alg, ctm_alg, χenv, iterations)
    β = β₀
    βs = []
    Zs = []    
    for _ = 1:iterations
        println("For β = $(β), T = $(1/β), O  = $(summary(O))")
        Z = calculate_magnetization(O, χenv, Magn, ctm_alg)
        push!(βs, β₀)
        push!(Zs, Z)
        println("For β = $(β), T = $(1/β), magn  = $(Z)")
        O, _ = approximate_state((O,O), trunc_alg)
        β *= 2
    end
    return βs, Zs
end

setprecision(128)
T = Complex{BigFloat}

β₀ = 1e-3
iterations = ceil(log(2, 1/(2.2*β₀)))
p = 5
χenv_approx = 30
χenv = 40
envspace_approx = ℂ^χenv_approx
Dcut = 3
g = T(0.0)

ctm_alg = SimultaneousCTMRG(;
tol=1e-10,
miniter=4,
maxiter=1000,
verbosity=2,
svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),)

trunc_alg = ApproximateEnvTruncation(ctm_alg, envspace_approx, truncdim(Dcut); check_fidelity = false, maxiter = 30, verbosity = 0)

J = T(1.0)
twosite_op = rmul!(σˣˣ(T), -J)
onesite_op = rmul!(σᶻ(T), g)
Magn = σˣ()

spaces = i -> (i >= 0) ? ℂ^(2^(i)) : ℂ^10

O, O_clust_full = clusterexpansion(T, p, β₀, twosite_op, onesite_op; spaces = spaces, verbosity = 0, symmetry = "C4")
O_clust_full = convert(TensorMap, O_clust_full)
O_clust = zeros(ComplexF64, codomain(O_clust_full), domain(O_clust_full))
O_clust[] = O_clust_full[]

βs, Zs = get_phase_transition(β₀, O_clust, Magn, trunc_alg, ctm_alg, χenv, iterations)

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
