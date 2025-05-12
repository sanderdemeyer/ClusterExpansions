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
    return magn / Z
end

function get_phase_transition(expon1, expon2, Magn, trunc_alg, ctm_alg, χenv, iterations)
    (O₀, β₀) = expon1
    (O₊, β₊) = expon2
    β = β₀
    βs = []
    Zs = []    
    O = copy(O₀)
    for _ = 1:iterations
        println("For β = $(β), T = $(1/β), O  = $(summary(O))")
        Z = calculate_magnetization(O, χenv, Magn, ctm_alg)
        push!(βs, β)
        push!(Zs, Z)
        println("For β = $(β), T = $(1/β), magn  = $(Z)")
        O, _ = approximate_state((O,O₊), trunc_alg)
        β += β₊
    end
    return βs, Zs
end

setprecision(128)
T = Complex{BigFloat}

β₀ = T(0.1)
β₊ = T(0.05)
iterations = ceil(Int,(0.5-β₀)/β₊)
p = 3
χenv_approx = 10
χenv = 12
envspace_approx = ℂ^χenv_approx
Dcut = 5
g = T(0.0)

ctm_alg = SimultaneousCTMRG(;
tol=1e-10,
miniter=4,
maxiter=1000,
verbosity=2,
svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),)

trunc_alg = ApproximateEnvTruncation(ctm_alg, envspace_approx, truncdim(Dcut); check_fidelity = false, maxiter = 1, verbosity = 0)

J = T(1.0)
twosite_op = rmul!(σˣˣ(T), -J)
onesite_op = rmul!(σᶻ(T), g)
Magn = σˣ()

spaces = i -> (i >= 0) ? ℂ^(2^(i)) : ℂ^10

O, O_clust_full = clusterexpansion(T, p, β₀, twosite_op, onesite_op; spaces = spaces, verbosity = 0, symmetry = "C4")
O_clust_full = convert(TensorMap, O_clust_full)
O_clust = zeros(ComplexF64, codomain(O_clust_full), domain(O_clust_full))
O_clust[] = O_clust_full[]
O₀ = copy(O_clust)

# O, O_clust_full = clusterexpansion(T, p, β₊, twosite_op, onesite_op; spaces = spaces, verbosity = 0, symmetry = "C4")
# O_clust_full = convert(TensorMap, O_clust_full)
# O_clust = zeros(ComplexF64, codomain(O_clust_full), domain(O_clust_full))
# O_clust[] = O_clust_full[]
# O₊ = copy(O_clust)
O₊ = copy(O₀)

βs, Zs = get_phase_transition((O₀, β₀), (O₊, β₊), Magn, trunc_alg, ctm_alg, χenv, iterations)

Ts = [1/β for β = βs]
plt = scatter(Float64.(Ts), abs.((Zs)), label = "p = 6, χ = $(χenv)")
xlabel!("T")
ylabel!("Magnetization")
title!("Ising model with g = $(real(g))")
# savefig(plt, "Ising_phase_transition_g_$(real(g))_p_$(p)_chienv_$(χenv).png")
display(plt)

file = jldopen("ClusterExpansion_g_$(g)_p_$(p)_chienv_$(χenv).jld2", "w")
file["Ts"] = Ts
file["Zs"] = Zs
file["Magn"] = Magn
file["β₀"] = β₀
file["β₊"] = β₊
close(file)
