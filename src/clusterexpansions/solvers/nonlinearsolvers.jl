"""
    This is not yet updated to deal with BigFloat
"""

function contract_tensors_symmetric(A)
    loop = ncon([A, A, A, A], [[-1 -5 4 1], [-2 -6 1 2], [-3 -7 2 3], [-4 -8 3 4]])
    return permute(loop, ((1,2,3,4),(5,6,7,8)))
end

function construct_PEPO_loop(A, pspace, vspace, trivspace)
    A_NW = TensorMap(zeros, pspace ⊗ pspace', trivspace ⊗ vspace ⊗ vspace' ⊗ trivspace')
    A_NE = TensorMap(zeros, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ vspace' ⊗ vspace')
    A_SE = TensorMap(zeros, pspace ⊗ pspace', vspace ⊗ trivspace ⊗ trivspace' ⊗ vspace')
    A_SW = TensorMap(zeros, pspace ⊗ pspace', vspace ⊗ vspace ⊗ trivspace' ⊗ trivspace')
    A_NW[][:,:,1,:,:,1] = A[]
    A_NE[][:,:,1,1,:,:] = A[]
    A_SE[][:,:,:,1,1,:] = A[]
    A_SW[][:,:,:,:,1,1] = A[]
    return [A_NW, A_NE, A_SE, A_SW]
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

function get_step_size(A, A_nudge, exp_H, step_size, linesearch_options::Int)
    αs = step_size .* [10.0^(i) for i = -linesearch_options:linesearch_options]
    errors = [norm(contract_tensors_symmetric(A-α*A_nudge)-exp_H) for α in αs]
    return αs[argmin(errors)]
end

function get_step_size_N_loop(Ns, C, A, A_nudge, exp_H, step_size, linesearch_options::Int)
    αs = step_size .* [10.0^(i) for i = -linesearch_options:linesearch_options]
    errors = [norm(contract_tensors_N_loop(Ns, C, A-α*A_nudge)-exp_H) for α in αs]
    return αs[argmin(errors)]
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

function get_A_nudge(A, exp_H; c = 1)
    D = contract_tensors_symmetric(A) - exp_H
    g = get_gradient(A)
    x1 = ncon([D, g], [[-1 1 2 3 -2 4 5 6], [-3 -4 1 2 3 4 5 6]], [false true])
    x2 = ncon([D, g], [[1 -1 2 3 4 -2 5 6], [-3 -4 1 2 3 4 5 6]], [false true])
    x3 = ncon([D, g], [[1 2 -1 3 4 5 -2 6], [-3 -4 1 2 3 4 5 6]], [false true])
    x4 = ncon([D, g], [[1 2 3 -1 4 5 6 -2], [-3 -4 1 2 3 4 5 6]], [false true])

    return permute(x1+x2+x3+x4, ((1,2),(3,4))) + c*A, norm(D)/norm(exp_H)
end

function get_A_nudge_N_loop(Ns, C, A, exp_H, dir, i)
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
    return permute(x, ((1,2),(3,4))), norm(D)/norm(exp_H)
end

function solve_4_loop(RHS, space, levels_to_update; step_size = 1e-3, ϵ = 1e-10, max_iter = 10000, line_search = false, linesearch_options = 1, verbosity = 2)
    base_step_size = step_size
    exp_H = RHS / norm(RHS)
    pspace = ℂ^2
    trivspace = ℂ^1
    A = TensorMap(randn, pspace ⊗ pspace', space ⊗ space')
    A = A / norm(A) * (norm(exp_H))^(1/4) * sqrt(10)
    errors = [Inf]
    c = 1e-2
    for i = 1:max_iter
        A_nudge, error = get_A_nudge(A, exp_H; c = c)
        A_nudge = A_nudge * (norm(A) / norm(A_nudge))
        step_size = base_step_size*error
        if line_search
            step_size = get_step_size(A, A_nudge, exp_H, step_size, linesearch_options)
        end
        A = A - step_size*A_nudge
        if error < ϵ
            if verbosity >= 2
                @info "Converged after $(i) iterations - error = $(error)"
            end
            As = construct_PEPO_loop(A, pspace, space, trivspace)
            dict = Dict((0, -1, -1, 0) => 1, (0, 0, -1, -1) => 2, (-1, 0, 0, -1) => 3, (-1, -1, 0, 0) => 4)
            values = [dict[key] for key in levels_to_update]
            return [As[values[1]], As[values[2]], As[values[3]], As[values[4]]] .* norm(RHS)^(1/4), errors
        end
        if verbosity >= 3
            @info "Iteration $(i) of loop solver: error = $(error) - step size = $(step_size)"
        end
        push!(errors, error)
        if errors[end] > errors[end-1]
            if verbosity >= 1
                @info "Error starting increasing after iteration $(i): Early stopping at error = $(error)"
            end
            As = construct_PEPO_loop(A, pspace, space, trivspace)
            dict = Dict((0, -1, -1, 0) => 1, (0, 0, -1, -1) => 2, (-1, 0, 0, -1) => 3, (-1, -1, 0, 0) => 4)
            values = [dict[key] for key in levels_to_update]
            return [As[values[1]], As[values[2]], As[values[3]], As[values[4]]] .* norm(RHS)^(1/4), errors
        end
    end
    error = norm(contract_tensors_symmetric(A) - exp_H)/norm(exp_H)
    if verbosity >= 1
        @warn "Not converged after $(max_iter) iterations - error = $(error)"
    end
    As = construct_PEPO_loop(A, pspace, space, trivspace)
    dict = Dict((0, -1, -1, 0) => 1, (0, 0, -1, -1) => 2, (-1, 0, 0, -1) => 3, (-1, -1, 0, 0) => 4)
    values = [dict[key] for key in levels_to_update]
    return [As[values[1]], As[values[2]], As[values[3]], As[values[4]]] .* norm(RHS)^(1/4), errors
end

function solve_N_loop(Ns, C, exp_H; α = 10, step_size = 1e-7, ϵ = 1e-10, max_iter = 1000, line_search = false, linesearch_options = 3)
    pspace = ℂ^2
    space = ℂ^α
    # C = TensorMap(randn, pspace ⊗ pspace', space ⊗ space')
    A = TensorMap(randn, pspace ⊗ pspace', space ⊗ space')
    errors = []
    error = ϵ + 1
    for i = 1:max_iter
        for dir = 1:4
            for j = 1:Ns[dir]
                A_nudge, error = get_A_nudge_N_loop(Ns, C, A, exp_H, dir, j)
                if line_search
                    step_size = get_step_size_N_loop(Ns, C, A, A_nudge, exp_H, step_size, linesearch_options)
                end
                A = A - step_size*A_nudge
                push!(errors, error)
            end
        end
        if error < ϵ
            println("Converged after $(i) iterations - error = $(error)")
            return A, errors
        end
    end
    error = norm(contract_tensors_N_loop(Ns, C, A) - exp_H)
    @warn "Not converged after $(max_iter) iterations - error = $(error)"    
    return A, errors
end