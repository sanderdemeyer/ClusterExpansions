using TensorKit
using MPSKitModels
using Graphs
using LongestPaths
using ClusterExpansions
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

cluster = [(0,0),(1,0),(1,1),(0,1)]

function contract_tensors_symmetric(A)
    RHS = ncon([A, A, A, A], [[-1 -5 4 1], [-2 -6 1 2], [-3 -7 2 3], [-4 -8 3 4]])
    return permute(RHS, ((1,2,3,4),(5,6,7,8)))
end

function get_gradient(A)
    RHS = ncon([A, A, A], [[-3 -6 -2 1], [-4 -7 1 2], [-5 -8 2 -1]])
    return RHS
end

function get_A(A, exp_H)
    D = contract_tensors_symmetric(A) - exp_H
    g = get_gradient(A)
    x = ncon([D, g], [[-1 1 2 3 -2 4 5 6], [-3 -4 1 2 3 4 5 6]], [false true])
    return permute(x, ((1,2),(3,4))), norm(D)
end

function gd(α, exp_H; step_size = 1e-3, ϵ = 1e-15, max_iter = 200)
    pspace = ℂ^2
    space = ℂ^α
    x = TensorMap(randn, pspace ⊗ pspace', space ⊗ space')
    for i = 1:max_iter
        x_nudge, error = get_A(x, exp_H)
        # println("error = $(error)")
        # sz = step_size*norm(x)/norm(x_nudge)
        x = x - step_size*x_nudge
        if error < ϵ
            println("Converged after $(i) iterations - error = $(error)")
            return x
        end
    end
    error = norm(contract_tensors_symmetric(x) - exp_H)
    @warn "Not converged after $(max_iter) iterations - error = $(error)"    
    return x
end


α = 40
ϵ = 1e-10
max_iter = 2000

step_size = 1e-5
exp_H = exponentiate_hamiltonian(twosite_op, cluster, β, length(cluster))

for i = 4:8
    println("i = $(i)")
    step_size = 10.0^(-i)
    A = gd(α, exp_H; step_size = step_size, ϵ = ϵ, max_iter = max_iter);
end