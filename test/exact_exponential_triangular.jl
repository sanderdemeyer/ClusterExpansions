using Test
using TensorKit
using TensorKitTensors
using MPSKitModels
using ClusterExpansions
using PEPSKit
import PEPSKit: rmul!
using Plots

symmetry = nothing
critical = false

setprecision(128)

T = BigFloat
J = T(1.0)
g = T(2.5)

pspace = ℂ^2
twosite_op = rmul!(SpinOperators.S_z_S_z(T), -J)
onesite_op = rmul!(SpinOperators.S_z(T), g * -J)

spaces = i -> (i >= 0) ? ℂ^(2^(i)) : ℂ^4

function exponentiate_hamiltonian_periodic_triangular(onesite_op, twosite_op, β, Lx, Ly)
    N = Lx*Ly
    pspace = domain(twosite_op)[1]
    H = []
    for x₁ = 1:Lx
        for y₁ = 1:Ly
            i = Lx*(y₁-1) + x₁
            term = ncon([onesite_op, [id(pspace) for _ = 1:N-1]...], [[-i, -N-i], [[-k, -N-k] for k = setdiff(1:N, i)]...], [false for _ = 1:N])
            push!(H, permute(term, Tuple(1:N), Tuple(N+1:2*N)))

            # --
            x₂ = mod1(x₁+1, Lx)
            y₂ = mod1(y₁, Ly)
            (i,j) = (Lx*(y₁-1) + x₁, Lx*(y₂-1) + x₂)
            term = ncon([twosite_op, [id(pspace) for _ = 1:N-2]...], [[-i, -j, -N-i, -N-j], [[-k, -N-k] for k = setdiff(1:N, (i,j))]...], [false for _ = 1:N-1])
            push!(H, permute(term, Tuple(1:N), Tuple(N+1:2*N)))

            # /
            if mod(y₁, 2) == 0
                x₂ = mod1(x₁, Lx)
            else
                x₂ = mod1(x₁-1, Lx)
            end
            y₂ = mod1(y₁+1, Ly)
            (i,j) = (Lx*(y₁-1) + x₁, Lx*(y₂-1) + x₂)
            term = ncon([twosite_op, [id(pspace) for _ = 1:N-2]...], [[-i, -j, -N-i, -N-j], [[-k, -N-k] for k = setdiff(1:N, (i,j))]...], [false for _ = 1:N-1])
            push!(H, permute(term, Tuple(1:N), Tuple(N+1:2*N)))

            # \
            if mod(y₁, 2) == 0
                x₂ = mod1(x₁+1, Lx)
            else
                x₂ = mod1(x₁, Lx)
            end
            y₂ = mod1(y₁+1, Ly)
            (i,j) = (Lx*(y₁-1) + x₁, Lx*(y₂-1) + x₂)
            term = ncon([twosite_op, [id(pspace) for _ = 1:N-2]...], [[-i, -j, -N-i, -N-j], [[-k, -N-k] for k = setdiff(1:N, (i,j))]...], [false for _ = 1:N-1])
            push!(H, permute(term, Tuple(1:N), Tuple(N+1:2*N)))
        end
    end
    exp_H = exp(-β*sum(H))
    return exp_H
end

function contract_PEPO_periodic_triangular(O, Lx, Ly)
    N = Lx*Ly
    indices = fill(0, N, 8)
    bond = 1

    for x₁ = 1:Lx
        for y₁ = 1:Ly
            x₂ = mod1(x₁+1, Lx)
            y₂ = mod1(y₁, Ly)
            (i,j) = (Lx*(y₁-1) + x₁, Lx*(y₂-1) + x₂)
            if indices[i,5] == 0
                indices[i,5] = bond
                indices[j,8] = bond
                bond += 1
            end

            # /
            if mod(y₁, 2) == 0
                x₂ = mod1(x₁, Lx)
            else
                x₂ = mod1(x₁-1, Lx)
            end
            y₂ = mod1(y₁+1, Ly)
            (i,j) = (Lx*(y₁-1) + x₁, Ly*(y₂-1) + x₂)
            if indices[i,7] == 0
                indices[i,7] = bond
                indices[j,4] = bond
                bond += 1
            end

            # \
            if mod(y₁, 2) == 0
                x₂ = mod1(x₁+1, Lx)
            else
                x₂ = mod1(x₁, Lx)
            end
            y₂ = mod1(y₁+1, Ly)
            (i,j) = (Lx*(y₁-1) + x₁, Ly*(y₂-1) + x₂)
            if indices[i,6] == 0
                indices[i,6] = bond
                indices[j,3] = bond
                bond += 1
            end

            indices[i,1] = -i
            indices[i,2] = -N-i

        end
    end
    indices = [[indices[n,i] for i = 1:8] for n = 1:N]
    term = ncon([O for _ = 1:N], indices, [false for _ = 1:N])
    return permute(term, Tuple(1:N), Tuple(N+1:2*N))
end

bmin = -1
bmax = -3
pmax = 3
nodp = 4
βs = [T(10.0)^(x) for x in LinRange(bmin, bmax, nodp)]
ps = [i for i = 2:pmax]

errors = zeros(length(βs), length(ps))

Lx = 2
Ly = 2
for (j,p) = enumerate(ps)
    ce_alg = ising_operators(J, g, 0.0; p, verbosity = 1, symmetry)
    for (i,β) = enumerate(βs)
        println("Started with p = $p and β = $β")
        O = evolution_operator_triangular(ce_alg, β)
        println("Calculating the exact exponential")
        exp_exact = exponentiate_hamiltonian_periodic_triangular(onesite_op, twosite_op, β, Lx, Ly)
        println("Contracting the PEPOs")
        exp_approx = contract_PEPO_periodic_triangular(O, Lx, Ly)

        error = norm(exp_exact-exp_approx)/norm(exp_approx)
        println("error = $error")
        errors[i,j] = error
    end
end

plt = plot()
for (j,p) = enumerate(2:pmax)
    plot!(plt, βs, errors[:,j], label = "p = $p", xaxis = :log, yaxis = :log)
end
display(plt)
