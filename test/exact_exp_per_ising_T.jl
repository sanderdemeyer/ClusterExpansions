using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfinitePartitionFunction, LocalOperator, vertices
using JLD2

symmetry = "C4"
critical = false

setprecision(128)
T = Complex{BigFloat}

if critical
    J = T(1.0)
    g = T(3.1)
    e = -1.6417 * 2
    mˣ = 0.91
else
    J = T(1.0)
    g = T(2.0)
    e = -1.2379 * 2
    mˣ = 0.524
end

g = T(0.0)

pspace = ℂ^2
twosite_op = rmul!(σᶻᶻ(T), -J)
onesite_op = rmul!(σˣ(T), g * -J)

# spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : ℂ^20
spaces = i -> (i >= 0) ? ℂ^(2^(i)) : ℂ^10



# c = [(0, 0), (0, 1), (1, 0), (2, 0), (2, -1), (2, 1), (3, 0), (4, 0)]
# N = 2
# c = []
# for i = 1:N
#     for j = 1:N
#         push!(c, (i,j))
#     end
# end
# cluster = Cluster(c)

function exponentiate_hamiltonian_periodic(T, onesite_op, twosite_op, β, L)
    N = L^2
    pspace = domain(twosite_op)[1]
    H = []
    for x₁ = 1:L
        for y₁ = 1:L
            i = L*(y₁-1) + x₁
            term = ncon([onesite_op, [id(pspace) for _ = 1:N-1]...], [[-i, -N-i], [[-k, -N-k] for k = setdiff(1:N, i)]...], [false for _ = 1:N])
            push!(H, permute(term, Tuple(1:N), Tuple(N+1:2*N)))

            x₂ = mod1(x₁+1, L)
            y₂ = mod1(y₁, L)
            (i,j) = (L*(y₁-1) + x₁, L*(y₂-1) + x₂)
            term = ncon([twosite_op, [id(pspace) for _ = 1:N-2]...], [[-i, -j, -N-i, -N-j], [[-k, -N-k] for k = setdiff(1:N, (i,j))]...], [false for _ = 1:N-1])
            push!(H, permute(term, Tuple(1:N), Tuple(N+1:2*N)))

            x₂ = mod1(x₁, L)
            y₂ = mod1(y₁+1, L)
            (i,j) = (L*(y₁-1) + x₁, L*(y₂-1) + x₂)
            term = ncon([twosite_op, [id(pspace) for _ = 1:N-2]...], [[-i, -j, -N-i, -N-j], [[-k, -N-k] for k = setdiff(1:N, (i,j))]...], [false for _ = 1:N-1])
            push!(H, permute(term, Tuple(1:N), Tuple(N+1:2*N)))
        end
    end
    exp_H = exp(-β*sum(H)) 
    return exp_H
end

function contract_PEPO_periodic(O, L)
    N = L^2
    indices = fill(0, N, 6)
    bond = 1
    for x₁ = 1:L
        for y₁ = 1:L
            x₂ = mod1(x₁+1, L)
            y₂ = mod1(y₁, L)
            (i,j) = (L*(y₁-1) + x₁, L*(y₂-1) + x₂)
            if indices[i,4] == 0
                indices[i,4] = bond
                indices[j,6] = bond
                bond += 1
            end

            x₂ = mod1(x₁, L)
            y₂ = mod1(y₁+1, L)
            (i,j) = (L*(y₁-1) + x₁, L*(y₂-1) + x₂)
            if indices[i,5] == 0
                indices[i,5] = bond
                indices[j,3] = bond
                bond += 1
            end
            indices[i,1] = -i
            indices[i,2] = -N-i
        end
    end
    indices = [[indices[n,i] for i = 1:6] for n = 1:N]
    term = ncon([O for _ = 1:N], indices, [false for _ = 1:N])

    return permute(term, Tuple(1:N), Tuple(N+1:2*N))
end

bmin = -4
bmax = -1
pmax = 4
βs = [10.0^(x) for x in LinRange(bmin, bmax, 10)]
ps = [i for i = 2:pmax]

errors = zeros(length(βs), length(ps))

# O, O_clust = clusterexpansion(T, 4, 1e-2, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = "C4", verbosity = 2)
# println(a)
L = 2

# O, O_clust_full = clusterexpansion([0 0 0], T, 3, 1e-3, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = nothing, verbosity = 2)
# print(fdjsa)
for (j,p) = enumerate(ps)
    for (i,β) = enumerate(βs)
        @warn "beta = $(β) (number i = $(i)), p = $p"
        O, O_clust_full = clusterexpansion([0 0 0], T, p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = "C4", verbosity = 2)
        # O_clust = convert(TensorMap, O_clust_full)
        # O_clust = TensorMap(convert(Array{ComplexF64}, O_clust.data), codomain(O_clust), domain(O_clust))

        println("Cluster expansion done")
        exp_exact = exponentiate_hamiltonian_periodic(T, onesite_op, twosite_op, β, L)
        println("Exact exponential done")
        exp_approx = contract_PEPO_periodic(O_clust_full, L)
        println("Approx exponential done")

        error = norm(exp_exact-exp_approx)/norm(exp_approx)
        println("error = $(error)")
        errors[i,j] = error
        file = jldopen("exact_exponential_periodic_intermediate_g_$(g)_p_$(pmax)_betas_$(bmin)_$(bmax)_T_$(T).jld2", "w")
        file["errors"] = errors
        close(file)
    end
end

colors = [:red, :green, :blue]

using Plots
plt = scatter(βs, errors_bigfloat[:,1], label = "p = 2, T = BigFloat", markercolor = colors[1], markershape=:xcross)
for i = 3:pmax
    scatter!(βs, errors_bigfloat[:,i-1], label = "p = $(i), T = BigFloat", markercolor = colors[i-1], markershape=:xcross)
end
# for i = 2:pmax
#     scatter!(βs, errors_double[:,i-1], label = "p = $(i), T = Float64", markercolor = colors[i-1], markershape=:cross)
# end
for i = 2:pmax
    scatter!(βs, errors_single[:,i-1], label = "p = $(i), T = Float32", markercolor = colors[i-1], markershape=:cross)
end

scatter!(xscale=:log10, yscale=:log10)
xlabel!("β*J")
ylabel!("Error on the exact exp for PBE with L = $(L)")
title!("Ising model with g = $(real(g))")
savefig(plt, "Exact_exponential_PBE_L_$(L)_g_$(real(g))_p_$(pmax)_betas_$(bmin)_$(bmax)_comparison_32.png")
scatter!(legend=:bottomright)

display(plt)

# file = jldopen("exact_exponential_PBE_L_$(L)_g_$(real(g))_p_$(pmax)_betas_$(bmin)_$(bmax)_comparison.jld2", "w")
# file["errors_Float64"] = errors_double
# file["errors_BigFloat"] = errors_bigfloat
# close(file)
