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

symmetry = nothing
critical = false
solving_loops = false

setprecision(128)
T = Complex{Float64}

t = T(1.0)
V = T(0.0)
# μ = 2*V
μ = T(0.0)

kinetic_operator = -t * (c_plusmin(T) - c_minplus(T))
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



# c = [(0, 0), (0, 1), (1, 0), (2, 0), (2, -1), (2, 1), (3, 0), (4, 0)]
# N = 2
# c = []
# for i = 1:N
#     for j = 1:N
#         push!(c, (i,j))
#     end
# end
# cluster = Cluster(c)

function exponentiate_hamiltonian_open(T, onesite_op, twosite_op, β, L)
    N = L^2
    pspace = domain(twosite_op)[1]
    H = []
    for x₁ = 1:L
        for y₁ = 1:L
            i = L*(y₁-1) + x₁
            term = ncon([onesite_op, [id(pspace) for _ = 1:N-1]...], [[-i, -N-i], [[-k, -N-k] for k = setdiff(1:N, i)]...], [false for _ = 1:N])
            push!(H, permute(term, Tuple(1:N), Tuple(N+1:2*N)))

            if x₁ < L
                x₂ = x₁+1
                y₂ = y₁
                (i,j) = (L*(y₁-1) + x₁, L*(y₂-1) + x₂)
                term = ncon([twosite_op, [id(pspace) for _ = 1:N-2]...], [[-i, -j, -N-i, -N-j], [[-k, -N-k] for k = setdiff(1:N, (i,j))]...], [false for _ = 1:N-1])
                push!(H, permute(term, Tuple(1:N), Tuple(N+1:2*N)))
            end

            if y₁ < L
                x₂ = mod1(x₁, L)
                y₂ = mod1(y₁+1, L)
                (i,j) = (L*(y₁-1) + x₁, L*(y₂-1) + x₂)
                term = ncon([twosite_op, [id(pspace) for _ = 1:N-2]...], [[-i, -j, -N-i, -N-j], [[-k, -N-k] for k = setdiff(1:N, (i,j))]...], [false for _ = 1:N-1])
                push!(H, permute(term, Tuple(1:N), Tuple(N+1:2*N)))
            end
        end
    end
    exp_H = exp(-β*sum(H)) 
    return exp_H
end

function contract_PEPO_open(O, L)
    N = L^2
    trivspace = Vect[fℤ₂](0 => 1)
    triv_tensor = zeros(ComplexF64, domain(O)[1])
    triv_tensor_conj = zeros(ComplexF64, domain(O)[3])
    triv_tensor[1] = ones(trivspace)
    triv_tensor_conj[1] = ones(trivspace')

    indices = fill(0, N, 6)
    for x = 1:L
        for y = 1:L
            base = (y-1)*(2*L+1)
            i = L*(y-1) + x
            indices[i,1] = -i
            indices[i,2] = -N-i
            indices[i,3] = base + x
            indices[i,4] = base + L + x + 1
            indices[i,5] = base + 2*L + x + 1
            indices[i,6] = base + L + x
        end
    end
    indices = [[indices[n,i] for i = 1:6] for n = 1:N]
    triv_indices = vcat(1:L, [y*(2*L+1) for y = 1:L], (1:L) .+ L*(2*L+1), [y*(2*L+1)-L for y = 1:L])
    triv_indices = [[i] for i = triv_indices]
    triv_tensors = vcat([triv_tensor for _ = 1:2*L], [triv_tensor_conj for _ = 1:2*L])
    tensors = [i <= N ? O : triv_tensors[i-N] for i = 1:N+4*L]
    all_indices = vcat(indices, triv_indices)
    println(all_indices)
    term = ncon(tensors, all_indices)

    return permute(term, Tuple(1:N), Tuple(N+1:2*N))
end

bmin = -1.5
bmax = -1.5
pmax = 4
βs = [10.0^(x) for x in LinRange(bmin, bmax, 1)]
ps = [i for i = 2:pmax]

errors = zeros(length(βs), length(ps))

L = 3

for (j,p) = enumerate(ps)
    for (i,β) = enumerate(βs)
        @warn "beta = $(β) (number i = $(i)), p = $p"
        O, O_clust_full = clusterexpansion(T, p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = symmetry, verbosity = 2, solving_loops = solving_loops)
        O_clust = convert(TensorMap, O_clust_full)
        O_clust = TensorMap(convert(Array{ComplexF64}, O_clust.data), codomain(O_clust), domain(O_clust))

        println("Cluster expansion done")
        exp_exact = exponentiate_hamiltonian_open(T, onesite_op, twosite_op, β, L)
        println("Exact exponential done")
        exp_approx = contract_PEPO_open(O_clust_full, L)
        println(summary(O_clust_full))
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
plt = scatter(βs, errors[:,1], label = "p = 2, T = BigFloat", markercolor = colors[1], markershape=:xcross)
for i = 3:pmax
    scatter!(βs, errors[:,i-1], label = "p = $(i), T = BigFloat", markercolor = colors[i-1], markershape=:xcross)
end

scatter!(xscale=:log10, yscale=:log10)
xlabel!("β*J")
ylabel!("Error on the exact exp for OBE with L = $(L)")
title!("SF model")
# savefig(plt, "Exact_exponential_PBE_L_$(L)_g_$(real(g))_p_$(pmax)_betas_$(bmin)_$(bmax)_comparison.png")
display(plt)

# file = jldopen("exact_exponential_PBE_L_$(L)_g_$(real(g))_p_$(pmax)_betas_$(bmin)_$(bmax)_comparison.jld2", "w")
# file["errors_Float64"] = errors_double
# file["errors_BigFloat"] = errors_bigfloat
# close(file)
