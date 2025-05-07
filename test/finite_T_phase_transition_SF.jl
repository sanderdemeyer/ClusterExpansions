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

function test_SF(onesite_op, twosite_op, spaces, T, p, β, envspace)
    ctm_alg = SimultaneousCTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=1000,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
)

    O, O_clust = clusterexpansion(T, p, β, twosite_op, onesite_op; spaces = spaces, verbosity = 0, symmetry = "C4")
    O_clust = convert(TensorMap, O_clust)
    # O_clust = zeros(ComplexF64, codomain(O_clust), domain(O_clust))
    # O_clust.data = O_clust_full.data

    Magn = c_number(T)

    @tensor Z[-3 -4; -1 -2] := O_clust[1 1; -1 -2 -3 -4]
    partfunc = InfinitePartitionFunction(Z)
    @tensor A_M[-3 -4; -1 -2] := O_clust[1 2; -1 -2 -3 -4] * Magn[2; 1]
    partfunc_M = InfinitePartitionFunction(A_M)

    env0 = CTMRGEnv(partfunc, envspace)
    env, = leading_boundary(env0, partfunc, ctm_alg)

    Z = network_value(partfunc, env);
    magn = network_value(partfunc_M, env)
    m = magn/Z
    println("For p = $p, β = $(β), T = $(1/β), magn = $(magn)/$(Z) = $(magn/Z)")

    return Z, magn
end

setprecision(128)
T = Complex{Float64}

Ts = LinRange(T(0.5), T(3.0), 4)
βs = [1/T for T = Ts]

p = 5
χenv = 44
envspace = Vect[I](0 => χenv/2, 1 => χenv/2)

t = T(1.0)
V = T(-2.0)
μ = 2*V

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

Zs = [test_SF(onesite_op, twosite_op, spaces, T, p, β, envspace) for β = βs]

Magn = [i[2]/i[1] for i = Zs]

plt = scatter(Float64.(Ts), abs.((Magn)), label = "p = 6, χ = $(χenv)")
xlabel!("T")
ylabel!("Magnetization")
title!("Ising model with V = $(real(V))")
display(plt)

# file = jldopen("ClusterExpansion_g_$(g)_p_$(p)_chienv_$(χenv).jld2", "w")
# file["Ts"] = Ts
# file["Zs"] = Zs
# file["Magn"] = Magn
# close(file)