function get_triv_tensors(T, conjugated, trivspace)
    tensor = ones(T, trivspace)
    tensor_conj = ones(T, trivspace')
    return [conj ? tensor_conj : tensor for conj = conjugated] 
end

function get_graph(cluster)
    graph = fill(0, cluster.N, 4)
    for (bond_s, bond_i) = zip(cluster.bonds_sites,cluster.bonds_indices)
        dir = get_direction(bond_s[1], bond_s[2])
        graph[bond_i[1], dir[1]] = bond_i[2]
        graph[bond_i[2], dir[2]] = bond_i[1]
    end
    return graph
end

function symmetrize_cluster(x1, x2)
    @error "Not yet implemented"
    apply_A = (x, val) -> apply_A_N_2(x2, x, val)
    x, info = lssolve(apply_A, x1, LSMR(verbosity = 1, maxiter = 1000))

    return [x1 x2]
end

function apply_A_N_2(A, x, ::Val{false})
    @tensor Ax[-1 -2 -3 -4 -5; -6] := A[1; -1 -2 -4 -5 -3] * x[-6; 1]
    return Ax
end

function apply_A_N_2(x1, y ::Val{true})
    @tensor x2_new[-1 -2; -3 -4 -5 -6] := conj(y[1; -6]) * x1[-1 -2; -5 1 -3 -4] * conj(I[-4; 2])
    return flip(x2_new, 4)
end

function get_A(T, cluster, PEPO, sites_to_update)
    updates = length(sites_to_update)
    fixed_tensors = cluster.N - updates 
    graph = get_graph(cluster)
    contraction_indices = fill(0, fixed_tensors, 6)
    for i = 1:fixed_tensors
        contraction_indices[i,1] = -i
        contraction_indices[i,2] = -(fixed_tensors+i)
    end
    included_sites = setdiff(1:cluster.N, sites_to_update)
    conv = i -> i-sum([j ∈ sites_to_update for j = 1:i])
    m = 1
    for i = included_sites
        for j = 1:4
            if !(graph[i,j] ∈ (0, sites_to_update...)) && (contraction_indices[conv(i),j+2] == 0)
                ind = findall(x -> x == i, graph[graph[i,j],:])[1]
                contraction_indices[conv(i),j+2] = m 
                contraction_indices[conv(graph[i,j]),ind+2] = m
                m += 1
            end
        end
    end
    number_of_bonds = copy(m)-1
    conjugated = Bool[]
    for i = included_sites
        for j = 1:4
            if (graph[i,j] == 0)
                contraction_indices[conv(i),j+2] = m
                push!(conjugated, j > 2)
                m += 1
            end
        end
    end
    open_conjugated = Bool[]
    open_indices = Int[]
    spaces = fill((0,0), updates, 5-updates)
    m = 2*fixed_tensors+1
    opposite = dir -> 4 - dir + 2*(dir ∈ (2,4))
    for (enum,i) = enumerate(sites_to_update)
        kⱼ = 0
        for j = 1:4
            if graph[i,j] ∉ sites_to_update
                kⱼ += 1
                if graph[i,j] == 0
                    push!(open_conjugated, j > 2)
                    push!(open_indices, -m)
                else
                    ind = findall(x -> x == i, graph[graph[i,j],:])[1]
                    contraction_indices[conv(graph[i,j]),ind+2] = -m
                    spaces[enum, kⱼ] = (graph[i,j],opposite(j))
                end
                m += 1
            end
        end
    end

    trivspace = ℂ^1
    included_sites = setdiff(1:length(cluster.levels_sites)[1], sites_to_update)
    nontriv_tensors = [PEPO[cluster.levels_sites[site]] for site = included_sites]
    triv_tensors = vcat(get_triv_tensors(T, conjugated, trivspace), get_triv_tensors(T, open_conjugated, trivspace))
    triv_contractions = vcat([[number_of_bonds+i] for i = 1:length(conjugated)], [[i] for i = open_indices])
    all_contractions = vcat([contraction_indices[i,:] for i = 1:size(contraction_indices)[1]], triv_contractions)
    all_tensors = vcat(nontriv_tensors, triv_tensors)

    return ncon(all_tensors, all_contractions)
end

function apply_A_onesite(A, x::TensorMap, sites_to_update, N, ::Val{false})
    i = sites_to_update[1]
    included_sites = setdiff(1:N, sites_to_update)
    contraction_indices = vcat(-included_sites, -included_sites .- N, [1, 2, 3, 4])
    Ax = ncon([A, x], [contraction_indices, [-i, -N-i, 1, 2, 3, 4]])
    Ax = permute(Ax, ((Tuple(1:N)), (Tuple(N+1:2*N))))
    return Ax
end

function apply_A_onesite(A, Ax::TensorMap, sites_to_update, N, ::Val{true})
    included_sites = setdiff(1:N, sites_to_update)
    updates = length(sites_to_update)

    contracted_b = zeros(Int, 2*N)
    for (i,s) = enumerate(sites_to_update)
        contracted_b[s] = -i
        contracted_b[s+N] = -i-length(sites_to_update)
    end
    count = 1
    for i = 1:2*N
        if contracted_b[i] == 0
            contracted_b[i] = count
            count += 1
        end
    end
    x = ncon([A, Ax], [vcat(1:2*length(included_sites), [-3, -4, -5, -6]), contracted_b], [true false])
    x = permute(x, ((Tuple(1:2*updates)), (Tuple(1+2*updates:4+2*updates))))
    return x
end

function apply_A_twosite(A, x::TensorMap, sites_to_update, N, ::Val{false})
    (i,j) = sites_to_update
    included_sites = setdiff(1:N, sites_to_update)
    contraction_indices = vcat(-included_sites, -included_sites .- N, [1, 2, 3, 4, 5, 6])

    Ax = ncon([A, x], [contraction_indices, [-i, -N-i, 1, 2, 3, -j, -N-j, 4, 5, 6]])
    Ax = permute(Ax, ((Tuple(1:N)), (Tuple(N+1:2*N))))
    return Ax
end

function apply_A_twosite(A, Ax::TensorMap, sites_to_update, N, ::Val{true})
    included_sites = setdiff(1:N, sites_to_update)

    contracted_b = zeros(Int, 2*N)
    contracted_b[sites_to_update[1]] = -1
    contracted_b[sites_to_update[2]] = -6
    contracted_b[sites_to_update[1]+N] = -2
    contracted_b[sites_to_update[2]+N] = -7

    count = 1
    for i = 1:2*N
        if contracted_b[i] == 0
            contracted_b[i] = count
            count += 1
        end
    end

    x = ncon([A, Ax], [vcat(1:2*length(included_sites), [-3, -4, -5, -8, -9, -10]), contracted_b], [true false])
    x = permute(x, ((Tuple(1:5)), (Tuple(6:10))))    
    return x
end

function permute_dir(x, dir, second)
    tup = zeros(Int,4)
    tup[dir] = 6-5*second
    count = 1
    for i = 1:4
        if tup[i] == 0
            tup[i] = count+2+second
            count += 1
        end
    end
    return permute(x, ((1,2).+second, (Tuple(tup))))
end

function eig_with_truncation(x, space)
    T = scalartype(x)
    D = dim(space)
    eigval, eigvec = eig(x)
    # x_permuted = flip(permute(x, ((1,6,3,4,5),(7,2,8,9,10))), (2,7))
    # @assert norm(x_permuted' - x_permuted) < eps(real(T))*1e2 "Permutation made x nonhermitian: $(norm(x_permuted' - x_permuted))"
    # x_permuted = (x_permuted + x_permuted')/2
    # eigval, eigvec = eig(x_permuted)
    # println("eigvals = $(eigval)")
    eigval_trunc = zeros(T, space, space)
    eigvec_trunc = zeros(T, codomain(x), space)
    eigval_trunc[] = eigval[][1:D,1:D]
    eigvec_trunc[] = eigvec[][:,:,:,:,:,1:D]
    return eigval_trunc, eigvec_trunc
end

function solve_index(T, A, exp_H, conjugated, sites_to_update, levels_to_update, dir, N, spaces; verbosity = 2)
    pspace = ℂ^2
    trivspace = ℂ^1
    if N == 2
        x = zeros(T, pspace ⊗ pspace' ⊗ prod([conj ? trivspace : trivspace' for conj = conjugated[1]]), pspace' ⊗ pspace ⊗ prod([conj ? trivspace' : trivspace for conj = conjugated[2]]))
        b = permute(exp_H, ((1,3), (2,4)))
        x[][:,:,1,1,1,:,:,1,1,1] = b[]
    elseif length(sites_to_update) == 2
        apply_A = (x, val) -> apply_A_twosite(A, x, sites_to_update, N, val)

        if scalartype(exp_H) == Complex{BigFloat}
            x, info = lssolve(apply_A, exp_H, LSMR(verbosity = verbosity, maxiter = 2000, tol = BigFloat(1e-36)))
            # x = permute(x, ((1,6,3,4,8), (2,7,9,5,10)))
            # x = (x + x')/2
            # x = permute(x, ((1,6,3,4,9), (2,7,5,8,10)))
        else
            x, info = lssolve(apply_A, exp_H, LSMR(verbosity = verbosity, maxiter = 1000))
        end
    elseif length(sites_to_update) == 1
        apply_A = (x, val) -> apply_A_onesite(A, x, sites_to_update, N, val)
        
        if scalartype(exp_H) == Complex{BigFloat}
            x, info = lssolve(apply_A, exp_H, LSMR(verbosity = verbosity, maxiter = 2000, tol = BigFloat(1e-36)))
        else
            x, info = lssolve(apply_A, exp_H, LSMR(verbosity = verbosity, maxiter = 1000))
        end
        x = [x,]
    else
        error("Impossible to use a linear solver when the number of tensors to update is equal to $(length(sites_to_update))")
    end

    if length(sites_to_update) == 2
        svd = true
        if svd
            U, Σ, V = tsvd(x, trunc = truncspace(spaces(levels_to_update[1][dir[1]])))
            x1 = U * sqrt(Σ)
            x2 = sqrt(Σ) * V
            println("Norm of exp_H = $(norm(exp_H))")
            @assert norm(x - x1 * x2)/norm(x) < eps(real(T))*1e2 "Error made on the SVD is of the order $(norm(x - x1 * x2)/norm(x))"
        else
            if dir == (3,1)
                x = flip(x, (2,3,4,6,9,10))
                x = permute(x, ((1,2,3,4,5), (6,7,9,10,8)))
            else
                @error "TBA"
            end
            if !(ishermitian(x))
                x = (x + x')/2
                if verbosity >= 1
                    @warn "Hermiticicing: Change is of order $(norm(x-x')/(2*norm(x)))"
                end
            end
            eigval, eigvec = eig_with_truncation(x, spaces(levels_to_update[1][dir[1]]))
            x1 = eigvec * sqrt(eigval)
            x2 = sqrt(eigval) * eigvec'
            @assert norm(x - x1 * x2) < eps(real(T))*1e2 "Error made on the eigenvalue decomposition is of the order $(norm(x - x1 * x2))"
            # @assert norm(x - x1 * x2)/norm(x) < eps(real(T))*1e2 "Error made on the eigenvalue decomposition is of the order $(norm(x - x1 * x2)/norm(x))"
            x1 = flip(x1, (2,3,4))
            x2 = permute(x2, ((1,), (2,3,6,4,5)))
            x2 = flip(x2, (2,5,6))
        end
        x1 = permute_dir(x1, dir[1], 0)
        x2 = permute_dir(x2, dir[2], 1)
        if (dir == (3,1) || dir == (4,2))
            x1 = permute(flip(x1, 2+dir[1]), ((1,2),(3,4,5,6)))
            x2 = permute(flip(x2, 2+dir[2]), ((1,2),(3,4,5,6)))
        end

        x2_rot = rotl180_fermionic(x2)
        x = [x1, x2]
        # x = symmetrize_cluster!(x1, x2, dir)
    end
    if length(sites_to_update) == 3
        @warn "This happens because the error in the previous levels is too small - be warned, things can go wrong"
        @warn "Using eigendecomposition regardless of what you want"
        if dir == (3,1)
            x = flip(x, (2,3,4,6,9,10))
            x = permute(x, ((1,2,3,4,5), (6,7,9,10,8)))
        else
            @error "TBA"
        end
        if !(ishermitian(x))
            x = (x + x')/2
            @warn "Hermiticicing: Change is of order $(norm(x-x')/(2*norm(x)))"
        end
        eigval, eigvec = eig_with_truncation(x, spaces(levels_to_update[1][dir[1]]))
        x1 = eigvec * sqrt(eigval)
        x2 = sqrt(eigval) * eigvec'
        @assert norm(x - x1 * x2)/norm(x) < 1e-10 "Error made on the eigenvalue decomposition is of the order $(norm(x - x1 * x2)/norm(x))"
        println("Error made on the eigenvalue decomposition is of the order $(norm(x - x1 * x2)/norm(x))")
        println("Eigenvalues are $(eigval)")
        x1 = flip(x1, (2,3,4))
        x2 = permute(x2, ((1,), (2,3,6,4,5)))
        x2 = flip(x2, (2,5,6))
        println("norm of exp_H = $(norm(exp_H))")
        println("norm of x = $(norm(x))")

        x1 = permute_dir(x1, dir[1], 0)
        x2 = permute_dir(x2, dir[2], 1)
        if (dir == (3,1) || dir == (4,2))
            x1 = permute(flip(x1, 2+dir[1]), ((1,2),(3,4,5,6)))
            x2 = permute(flip(x2, 2+dir[2]), ((1,2),(3,4,5,6)))
        end

        x2_rot = rotl180_fermionic(x2)
        x = [x1, x2]
        # x = symmetrize_cluster!(x1, x2, dir)
    end
    return x
end