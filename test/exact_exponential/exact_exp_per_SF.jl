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

setprecision(128)
T = Complex{BigFloat}

t = T(1.0)
V = T(0.0)
# μ = 2*V
μ = T(0.0)

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

function exponentiate_hamiltonian_periodic_fermionic(T, onesite_op, twosite_op, β, L)
    N = L^2
    pspace = domain(twosite_op)[1]
    U, S, R_op = tsvd(twosite_op, (1,3), (2,4))
    L_op = U * S

    H = []
    for x₁ = 1:L
        for y₁ = 1:L
            println("Starting for x₁ = $x₁, y₁ = $y₁")
            i = L*(y₁-1) + x₁
            term = ncon([onesite_op, [id(pspace) for _ = 1:N-1]...], [[-i, -N-i], [[-k, -N-k] for k = setdiff(1:N, i)]...], [false for _ = 1:N])
            push!(H, permute(term, Tuple(1:N), Tuple(N+1:2*N)))

            x₂ = mod1(x₁+1, L)
            y₂ = mod1(y₁, L)
            if x₁ == L && false
                twosite_op = permute(L_op * twist(R_op, 1), ((1,3),(2,4)))
                println("twist for x₁ = $x₁, y₁ = $y₁, x₂ = $x₂, y₂ = $y₂")
            else
                twosite_op = permute(L_op * R_op, ((1,3),(2,4)))
                println("no twist for x₁ = $x₁, y₁ = $y₁, x₂ = $x₂, y₂ = $y₂")
            end
            (i,j) = (L*(y₁-1) + x₁, L*(y₂-1) + x₂)
            term = ncon([twosite_op, [id(pspace) for _ = 1:N-2]...], [[-i, -j, -N-i, -N-j], [[-k, -N-k] for k = setdiff(1:N, (i,j))]...], [false for _ = 1:N-1])
            push!(H, permute(term, Tuple(1:N), Tuple(N+1:2*N)))

            x₂ = mod1(x₁, L)
            y₂ = mod1(y₁+1, L)
            if y₁ == L && false
                twosite_op = permute(L_op * twist(R_op, 1), ((1,3),(2,4)))
                println("twist for x₁ = $x₁, y₁ = $y₁, x₂ = $x₂, y₂ = $y₂")
            else
                twosite_op = permute(L_op * R_op, ((1,3),(2,4)))
                println("no twist for x₁ = $x₁, y₁ = $y₁, x₂ = $x₂, y₂ = $y₂")
            end
            (i,j) = (L*(y₁-1) + x₁, L*(y₂-1) + x₂)
            term = ncon([twosite_op, [id(pspace) for _ = 1:N-2]...], [[-j, -i, -N-j, -N-i], [[-k, -N-k] for k = setdiff(1:N, (i,j))]...], [false for _ = 1:N-1])
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
    tensors = [O for _ = 1:N]
    # Top twists
    for j = 1:L
        (x,y) = (j,1)
        i = L*(y-1) + x
        println("For j = $j, i = $i")
        tensors[i] = twist(tensors[i], [3])
    end
    # Left twists
    for j = 1:L
        (x,y) = (1,j)
        i = L*(y-1) + x
        println("For j = $j, i = $i")
        tensors[i] = twist(tensors[i], [6])
    end
    println("indices = $indices")
    term = ncon(tensors, indices, [false for _ = 1:N])

    return permute(term, Tuple(1:N), Tuple(N+1:2*N))
end

bmin = -3
bmax = -3
pmax = 3
βs = [10.0^(x) for x in LinRange(bmin, bmax, 1)]
ps = [i for i = 2:pmax]

cluster_index = 4

errors = zeros(length(βs), length(ps))

# O, O_clust = clusterexpansion(T, 4, 1e-2, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = symmetry, verbosity = 2)
# println(a)
L = 2
for (j,p) = enumerate(ps)
    for (i,β) = enumerate(βs)
        @warn "beta = $(β) (number i = $(i)), p = $p"
        O, O_clust_full = clusterexpansion(T, p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = symmetry, verbosity = 2)
        # O_clust = zeros(ComplexF64, codomain(O_clust_full), domain(O_clust_full))
        O_clust = convert(TensorMap, O_clust_full)
        # O_clust_approx = TensorMap(O_clust.data, ComplexF64, codomain(O_clust), domain(O_clust))
        # O_clust = TensorMap(convert(Array{ComplexF64}, O_clust.data), codomain(O_clust), domain(O_clust))
        
        println("bond dimension of PEPO = $(dim(domain(O_clust)[1]))")

        println("Cluster expansion done")
        exp_exact = exponentiate_hamiltonian_periodic(T, onesite_op, twosite_op, β, L)
        println("Exact exponential done")
        exp_approx = contract_PEPO_periodic(O_clust, L)
        println("Approx exponential done")
        println("norms: $(norm(exp_exact)), $(norm(exp_approx))")
        error = norm(exp_exact-exp_approx)/norm(exp_approx)
        println("error = $(error)")
        errors[i,j] = error
        file = jldopen("exact_exp_PBC_SF_intermediate_t_$(t)_V_$(V)_p_$(pmax)_betas_$(bmin)_$(bmax)_T_$(T).jld2", "w")
        file["errors"] = errors
        close(file)
    end
end

using Plots
plt = scatter(βs, errors[:,1], label = "p = 2")
scatter!(βs, errors[:,2], label = "p = 3")
# scatter!(βs, errors[:,3], label = "p = 4")
# scatter!(βs, errors[:,4], label = "p = 5")
# scatter!(βs, errors[:,4], label = "p = 6")
scatter!(xscale=:log10, yscale=:log10)
xlabel!("β*J")
ylabel!("Error on the exact exp for PBE with L = $(L)")
# title!("SF model with t = $(t), V = $(V), without loops")
title!("SF model cluster $cluster_index")
# savefig(plt, "Exact_exponential_PBE_L_$(L)_g_$(real(g))_p_$(pmax)_betas_$(bmin)_$(bmax)_withoutloops.png")
display(plt)

# file = jldopen("exact_exp_PBC_SF_L_$(L)_t_$(t)_V_$(V)_p_$(pmax)_betas_$(bmin)_$(bmax)_T_$(T).jld2", "w")
# file["errors"] = errors
# close(file)



