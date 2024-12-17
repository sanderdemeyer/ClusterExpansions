function contract_tensors_symmetric(A)
    loop = ncon([A, A, A, A], [[-1 -5 4 1], [-2 -6 1 2], [-3 -7 2 3], [-4 -8 3 4]])
    return permute(loop, ((1,2,3,4),(5,6,7,8)))
end

function contract_tensors_N_loop(Ns, C, A)
    N = sum(Ns) + 4
    tens = vcat([C], fill(A, Ns[1]), [C], fill(A, Ns[2]), [C], fill(A, Ns[3]), [C], fill(A, Ns[4]))
    contractions = [[-i -N-i mod1(i-1,N) i] for i = 1:N]
    return permute(ncon(tens, contractions), ((Tuple(1:N)), (Tuple(N+1:2*N))))
end

function get_gradient(A)
    return ncon([A, A, A], [[-3 -6 -2 1], [-4 -7 1 2], [-5 -8 2 -1]])
end

function get_gradient_N_loop(Ns, C, A, dir, i)
    N = sum(Ns) + 4
    Ns_g = copy(Ns)
    Ns_g[dir] -= 1
    tens = vcat([C], fill(A, Ns_g[1]), [C], fill(A, Ns_g[2]), [C], fill(A, Ns_g[3]), [C], fill(A, Ns_g[4]))
    base = sum((Ns.+1)[1:dir-1])+i
    contractions = [[-j-2 -j-1-N mod1(j-1,N-1) j] for j = 1:N-1]
    contractions[base][4] = -1
    contractions[base+1][3] = -2
    g = ncon(tens, contractions)
    return g
end

function get_A(A, exp_H)
    D = contract_tensors_symmetric(A) - exp_H
    g = get_gradient(A)
    x = ncon([D, g], [[-1 1 2 3 -2 4 5 6], [-3 -4 1 2 3 4 5 6]], [false true])
    return permute(x, ((1,2),(3,4))), norm(D)
end

function get_A_N_loop(Ns, C, A, exp_H, dir, i)
    N = sum(Ns) + 4
    D = contract_tensors_N_loop(Ns, C, A) - exp_H
    g = get_gradient_N_loop(Ns, C, A, dir, i)
    base = sum((Ns.+1)[1:dir-1])+i+1

    contractions_D = zeros(Int64, 2*N)
    contractions_D[base] = -1
    contractions_D[N+base] = -2
    count = 1
    for (i,e) = enumerate(contractions_D)
        if e == 0
            contractions_D[i] = count
            count += 1
        end
    end
    x = ncon([D, g], [contractions_D, vcat([-3, -4], 1:2*N-2)], [false true])
    return permute(x, ((1,2),(3,4))), norm(D)
end

function solve_4_loop(exp_H; α = 10, step_size = 1e-7, ϵ = 1e-10, max_iter = 1000)
    pspace = ℂ^2
    space = ℂ^α
    A = TensorMap(randn, pspace ⊗ pspace', space ⊗ space')
    for i = 1:max_iter
        A_nudge, error = get_A(A, exp_H)
        A = A - step_size*A_nudge
        if error < ϵ
            println("Converged after $(i) iterations - error = $(error)")
            return A
        end
    end
    error = norm(contract_tensors_symmetric(A) - exp_H)
    @warn "Not converged after $(max_iter) iterations - error = $(error)"    
    return A
end

function solve_N_loop(Ns, C, exp_H; α = 10, step_size = 1e-7, ϵ = 1e-10, max_iter = 1000)
    pspace = ℂ^2
    space = ℂ^α
    C = TensorMap(randn, pspace ⊗ pspace', space ⊗ space')
    A = TensorMap(randn, pspace ⊗ pspace', space ⊗ space')
    for i = 1:max_iter
        for dir = 1:4
            for j = 1:Ns[dir]
                A_nudge, error = get_A_N_loop(Ns, C, A, exp_H, dir, j)
                println("norm A = $(norm(A)), norm A_nudge = $(norm(A_nudge)), error = $(error)")
                println("relative error = $(error/norm(exp_H))")
                A = A - step_size*A_nudge
            end
        end
        if error < ϵ
            println("Converged after $(i) iterations - error = $(error)")
            return A
        end
    end
    error = norm(contract_tensors_N_loop(Ns, C, A) - exp_H)
    @warn "Not converged after $(max_iter) iterations - error = $(error)"    
    return A
end

using TensorKit
using MPSKitModels
using Graphs
using LongestPaths
# using ClusterExpansions
using OptimKit
import PEPSKit: rmul!, σᶻᶻ, σˣ

p = 2
β = 1
D = 2
χenv = 12

J = 1.0
g = 1.0
N1, N2 = (1,1)

twosite_op = rmul!(σᶻᶻ(), -1.0)
onesite_op = rmul!(σˣ(), g * -J)

pspace = ℂ^2

α = 6
pspace = ℂ^2
space = ℂ^α
A = TensorMap(randn, pspace ⊗ pspace', space ⊗ space')
α = 40
ϵ = 1e-10
max_iter = 2000

step_size = 1e-5
Ns = [1, 0, 1, 0]

cluster = [(0,0)]
for dir = 1:4
    for i = 1:Ns[dir]+1
        latest = cluster[end]
        if dir == 1
            push!(cluster, (latest[1]+1, latest[2]))
        elseif dir == 2
            push!(cluster, (latest[1], latest[2]+1))
        elseif dir == 3
            push!(cluster, (latest[1]-1, latest[2]))
        else
            push!(cluster, (latest[1], latest[2]-1))
        end
    end
end      
cluster = sort(cluster[1:end-1])      
println("cluster = $(cluster)")
exp_H = exponentiate_hamiltonian(twosite_op, cluster, β, length(cluster))


# loop = contract_tensors_N_loop([1, 0, 1, 0], A, A);
# g = get_gradient_N_loop([1, 0, 1, 0], A, A);
Anew = solve_N_loop(Ns, A, exp_H; α = 10, step_size = 1e-8, ϵ = 1e-9, max_iter = 1000)

println(summary(loop))
println(summary(g))