using Test
using TensorKit
using MPSKitModels
using ClusterExpansions
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ

symmetry = nothing
critical = false

T = ComplexF64
J = T(1.0)
g = T(2.5)

pspace = ℂ^2
twosite_op = rmul!(σᶻᶻ(T), -J)
onesite_op = rmul!(σˣ(T), g * -J)

# spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : ℂ^20
spaces = i -> (i >= 0) ? ℂ^(2^(i)) : ℂ^10

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

bmin = -1
bmax = -3
pmax = 4
nodp = 4
βs = [T(10.0)^(x) for x in LinRange(bmin, bmax, nodp)]
ps = [i for i = 2:pmax]

errors = zeros(length(βs), length(ps))

L = 2
for (j,p) = enumerate(ps)
    ce_alg = ising_operators(J, g, 0.0; p, verbosity = 0, symmetry)
    for (i,β) = enumerate(βs)
        O_clust_full = evolution_operator(ce_alg, β)

        exp_exact = exponentiate_hamiltonian_periodic(T, onesite_op, twosite_op, β, L)
        exp_approx = contract_PEPO_periodic(O_clust_full, L)

        error = norm(exp_exact-exp_approx)/norm(exp_approx)
        errors[i,j] = error
    end
end

@testset "Order of the cluster expansion" begin
    for p = 2:pmax
        for n = 1:nodp-1
            @test log(errors[n+1,p-1]/errors[n,p-1]) / log(βs[n+1]/βs[n]) / p ≈ 1 atol=5e-2
        end
    end
end