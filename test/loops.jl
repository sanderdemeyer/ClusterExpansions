function apply_A_loop(A, x::TensorMap, ::Val{false})
    Ax = ncon([A, x], [[1 2 -2 -3 -4 -6 -7 -8], [-1 -5 2 1]])
    return permute(Ax, ((1,2,3,4), (5,6,7,8)))
end

function apply_A_loop(A, Ax::TensorMap, ::Val{true})
    x = ncon([A, Ax], [[-4 -3 1 2 3 4 5 6], [-1 1 2 3 -2 4 5 6]], [true false])
    return permute(x, ((1,2),(3,4)))
end

function update(tensors, exp_H)
    F = ncon(tensors[2:end],  [[-3 -6 -1 1], [-4 -7 1 2], [-5 -8 2 -2]])
    x0 = copy(tensors[1])
    apply_A = (x, val) -> apply_A_loop(F, x, val)
    x, _ = linsolve(apply_A, exp_H, x0, LSMR(verbosity = 0))
    tensors[1] = copy(x)
    circshift!(tensors, -1)
    return norm(x-x0)
end

function update_symmetric(A, exp_H)
    F = ncon([A, A, A],  [[-3 -6 -1 1], [-4 -7 1 2], [-5 -8 2 -2]])
    x0 = copy(A)
    apply_A = (x, val) -> apply_A_loop(F, x, val)
    Anew, _ = linsolve(apply_A, exp_H, x0, LSMR(verbosity = 1))
    return copy(Anew), norm(Anew-x0)
end

function contract_tensors(tensors)
    RHS = ncon(tensors, [[-1 -5 4 1], [-2 -6 1 2], [-3 -7 2 3], [-4 -8 3 4]])
    return permute(RHS, ((1,2,3,4),(5,6,7,8)))
end

function contract_tensors_symmetric(A)
    RHS = ncon([A, A, A, A], [[-1 -5 4 1], [-2 -6 1 2], [-3 -7 2 3], [-4 -8 3 4]])
    return permute(RHS, ((1,2,3,4),(5,6,7,8)))
end

function solve_4_loop_periodic(α, exp_H; ϵ = 1e-15, max_iter = 200)
    pspace = ℂ^2
    space = ℂ^α
    tensors = fill(TensorMap(randn, pspace ⊗ pspace', space ⊗ space'),4)
    for _ = 1:max_iter
        error = update(tensors, exp_H)
        if error < ϵ
            final_error = norm(contract_tensors(tensors)-exp_H)
            @assert final_error < 1e-10
            return tensors, final_error
        end
    end
    final_error = norm(contract_tensors(tensors)-exp_H)
    @warn "Not converged after $(max_iter) iterations - error is $(error), final error is $(final_error)"
    return tensors, final_error
end

function solve_4_loop(α, exp_H; ϵ = 1e-15, max_iter = 200)
    pspace = ℂ^2
    trivspace = ℂ^1
    space = ℂ^α
    tensors, err = solve_4_loop_periodic(α, exp_H; ϵ = ϵ, max_iter = max_iter)
    A = TensorMap(zeros, pspace ⊗ pspace', trivspace ⊗ space ⊗ space' ⊗ trivspace')
    B = TensorMap(zeros, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ space' ⊗ space')
    C = TensorMap(zeros, pspace ⊗ pspace', space ⊗ trivspace ⊗ trivspace' ⊗ space')
    D = TensorMap(zeros, pspace ⊗ pspace', space ⊗ space ⊗ trivspace' ⊗ trivspace')
    A[][:,:,1,:,:,1] = tensors[1][]
    B[][:,:,1,1,:,:] = tensors[2][]
    C[][:,:,:,1,1,:] = tensors[3][]
    D[][:,:,:,:,1,1] = tensors[4][]
    levels_to_update = [(0, -1, -1, 0), (0, 0, -1, -1), (-1, 0, 0, -1), (-1, -1, 0, 0)]
    solution = [A, B, C, D]
    return levels_to_update, solution, err
end

function solve_4_loop_periodic_symmetric(α, exp_H; ϵ = 1e-15, max_iter = 200)
    pspace = ℂ^2
    space = ℂ^α
    A = TensorMap(randn, pspace ⊗ pspace', space ⊗ space')
    for _ = 1:max_iter
        println("norm of A = $(norm(A))")
        Anew, error = update_symmetric(A, exp_H)
        A = copy(Anew)
        println("error = $(error)")
        final_error = norm(contract_tensors_symmetric(A)-exp_H)
        println("final error = $(final_error)")
        if false && (error < ϵ)
            final_error = norm(contract_tensors_symmetric(A)-exp_H)
            println("final error = $(final_error)")
            @assert final_error < 1e-10
            return tensors, final_error
        end
    end
    final_error = norm(contract_tensors_symmetric(A)-exp_H)
    @warn "Not converged after $(max_iter) iterations - error is $(error), final error is $(final_error)"
    return A, final_error
end

# β = 1.0
# α = 10
# cluster = [(0, 0), (0, 1), (1, 1), (1, 0)]
# exp_H = exponentiate_hamiltonian(cluster, β)
# A, final_error = solve_4_loop_symmetric(α, exp_H);
# norm(tensors[2]-tensors[3])