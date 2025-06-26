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

function get_twists(dir)
    if dir == (3,1) || dir == (4,2)
        return [1 2 4]
    elseif dir == (1,3) || dir == (2,4)
        return [1 4 5]
    else
        @error "Unexpected value for dir"
    end
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
    trivspace = domain(PEPO[0,0,0,0])[1]
    included_sites = setdiff(1:length(cluster.levels_sites)[1], sites_to_update)
    nontriv_tensors = [PEPO[cluster.levels_sites[site]] for site = included_sites]
    triv_tensors = vcat(get_triv_tensors(T, conjugated, trivspace), get_triv_tensors(T, open_conjugated, trivspace))
    triv_contractions = vcat([[number_of_bonds+i] for i = 1:length(conjugated)], [[i] for i = open_indices])
    all_contractions = vcat([contraction_indices[i,:] for i = 1:size(contraction_indices)[1]], triv_contractions)
    all_tensors = vcat(nontriv_tensors, triv_tensors)

    A = ncon(all_tensors, all_contractions)
    len = length(domain(A)) + length(codomain(A))
    A = permute(A, (Tuple(1:len-4), Tuple(len-3:len)))
    return A
end

function apply_A_onesite(A, x::TensorMap, sites_to_update, N, ::Val{false})
    Ax = ncon([A, x], [hcat(transpose(-1:-1:-2*N+2), [1 2 3 4]), [1 2 3 4 -2*N+1 -2*N]])
    len = length(domain(Ax)) + length(codomain(Ax))
    Ax = permute(Ax, (Tuple(1:len-2), Tuple(len-1:len)))
    return Ax
end

function apply_A_onesite(A, Ax::TensorMap, sites_to_update, N, ::Val{true})
    # f = x -> apply_A_onesite(A, x, sites_to_update, N, Val(true))
    # _, g = Zygote.pullback(f)
    # return g(Ax)

    Ax′ = twist(Ax, N:2*N-2)
    x = ncon([A, Ax′], [hcat(transpose(1:2*N-2), [-1 -2 -3 -4]), hcat(transpose(1:2*N-2), [-5 -6])], [true false])

    x = twist(x, [1 2])
    x = permute(x, ((1,2,3,4),(5,6)))
    return x
end

function apply_A_twosite(A, x::TensorMap, sites_to_update, N, dir, ::Val{false})
    (i,j) = sites_to_update
    included_sites = setdiff(1:N, sites_to_update)
    contraction_indices = vcat(-included_sites, -included_sites .- N, [1, 2, 3, 4, 5, 6])

    Ax = ncon([A, x], [hcat(transpose(-1:-1:-2*N+4), [1 2 3 4 5 6]), [1 2 3 4 5 6 -2*N+3 -2*N+2 -2*N+1 -2*N]])
    len = length(domain(Ax)) + length(codomain(Ax))
    Ax = permute(Ax, (Tuple(1:len-4), Tuple(len-3:len)))
    return Ax
end

function apply_A_twosite(A, Ax::TensorMap, sites_to_update, N, dir, ::Val{true})
    # y, back = Zygote.pullback(apply_A_twosite)
    
    Ax′ = twist(Ax, N-1:2*N-4)

    x = ncon([A, Ax′], [hcat(transpose(1:2*N-4), [-1 -2 -3 -4 -5 -6]), hcat(transpose(1:2*N-4), [-7 -8 -9 -10])], [true false])

    x = twist(x, get_twists(dir))
    x = permute(x, ((1,2,3,4,5,6),(7,8,9,10)))
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

    eigval_trunc = zeros(T, space, space)
    eigvec_trunc = zeros(T, codomain(x), space)
    eigval_trunc[] = eigval[][1:D,1:D]
    eigvec_trunc[] = eigvec[][:,:,:,:,:,1:D]
    return eigval_trunc, eigvec_trunc
end

function solve_index(T, A, exp_H, conjugated, sites_to_update, levels_to_update, dir, N, spaces; verbosity = 2)
    # pspace = codomain(exp_H)[1]
    trivspace = spaces(0)
    if N == 2
        # xtry = zeros(T, pspace ⊗ pspace' ⊗ prod([conj ? trivspace : trivspace' for conj = conjugated[1]]), pspace' ⊗ pspace ⊗ prod([conj ? trivspace' : trivspace for conj = conjugated[2]]))
        b = permute(exp_H, ((1,3), (2,4)))
        if dir == (3,1)
            Isom_domain = isomorphism(domain(b), domain(b) ⊗ trivspace ⊗ trivspace' ⊗ trivspace')
            Isom_codomain = isomorphism(codomain(b) ⊗ trivspace' ⊗ trivspace' ⊗ trivspace, codomain(b))
        elseif dir == (2,4)
            Isom_domain = isomorphism(domain(b), domain(b) ⊗ trivspace ⊗ trivspace ⊗ trivspace')
            Isom_codomain = isomorphism(codomain(b) ⊗ trivspace' ⊗ trivspace ⊗ trivspace, codomain(b))
        else
            @error "Unexpected value for dir"
        end
        x = Isom_codomain * b * Isom_domain
        # x.data = b.data
        # x[][:,:,1,1,1,:,:,1,1,1] = b[]
    elseif length(sites_to_update) == 2
        A = permute(A, (Tuple(1:2*N-4),Tuple(2*N-3:2*N+2)))

        apply_A = (x, val) -> apply_A_twosite(A, x, sites_to_update, N, dir, val)

        included_sites = setdiff(1:N, sites_to_update)
        exp_H_flipped = permute(exp_H, ((included_sites..., (included_sites .+ N)...), (sites_to_update..., (sites_to_update .+ N)...)))

        if scalartype(exp_H) == Complex{BigFloat}
            x, info = lssolve(apply_A, exp_H_flipped, LSMR(verbosity = verbosity, maxiter = 2000, tol = BigFloat(1e-36)))
            # x = permute(x, ((1,6,3,4,8), (2,7,9,5,10)))
            # x = (x + x')/2
            # x = permute(x, ((1,6,3,4,9), (2,7,5,8,10)))
        else
            x, info = lssolve(apply_A, exp_H_flipped, LSMR(verbosity = verbosity, maxiter = 1000))
        end
        error = norm(apply_A(x, Val(false)) - exp_H_flipped)/norm(exp_H_flipped)
        if error > 1e-10 && verbosity > 0
            @warn "Error made on the solution is of the order $(error)"
        end
        x = permute(x, ((7,9,1,2,3),(8,10,4,5,6)))

    elseif length(sites_to_update) == 1
        A = permute(A, (Tuple(1:2*N-2),Tuple(2*N-1:2*N+2)))

        apply_A = (x, val) -> apply_A_onesite(A, x, sites_to_update, N, val)
        
        included_sites = setdiff(1:N, sites_to_update[1])
        exp_H_flipped = permute(exp_H, ((included_sites..., (included_sites .+ N)...), (sites_to_update[1], sites_to_update[1]+N)))

        if scalartype(exp_H) == Complex{BigFloat}
            x, info = lssolve(apply_A, exp_H_flipped, LSMR(verbosity = verbosity, maxiter = 2000, tol = BigFloat(1e-36)))
        else
            x, info = lssolve(apply_A, exp_H_flipped, LSMR(verbosity = verbosity, maxiter = 1000))
        end
        error = norm(apply_A(x, Val(false)) - exp_H_flipped)/norm(exp_H_flipped)
        if error > 1e-10 && verbosity > 0
            @warn "Error made on the solution is of the order $(error)"
        end
        x = permute(x, ((5,6),(1,2,3,4)))
        x = [x,]
    else
        error("Impossible to use a linear solver when the number of tensors to update is equal to $(length(sites_to_update))")
    end
    if norm(x) == 0
        if verbosity > 0
            @warn "Norm of the solution is zero"
        end
        return nothing
    end
    if length(sites_to_update) == 2
        svd = true
        if svd
            U, Σ, V = tsvd(x)
            U, Σ, V = tsvd(x, trunc = truncspace(spaces(levels_to_update[1][dir[1]])))
            x1 = U * sqrt(Σ)
            x2 = sqrt(Σ) * V
            # @assert norm(x - x1 * x2)/norm(x) < eps(real(T))*1e2 "Error made on the SVD is of the order $(norm(x - x1 * x2)/norm(x))"
            if norm(x - x1 * x2)/norm(x) > eps(real(T))*1e2 && verbosity > 0
                @warn "Error made on the SVD is of the order $(norm(x - x1 * x2)/norm(x))"
            end
        else
            if dir == (3,1)
                x = flip(x, (2,3,4,6,9,10))
                x = permute(x, ((1,2,3,4,5), (6,7,9,10,8)))
            else
                @error "TBA"
            end
            if !(ishermitian(x))
                x = (x + x')/2
                if verbosity > 0
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
        # x2_rot = rotl180_fermionic(x2)
        x = [x1, x2]
        # x = symmetrize_cluster!(x1, x2, dir)
    elseif length(sites_to_update) == 3
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
        x1 = flip(x1, (2,3,4))
        x2 = permute(x2, ((1,), (2,3,6,4,5)))
        x2 = flip(x2, (2,5,6))

        x1 = permute_dir(x1, dir[1], 0)
        x2 = permute_dir(x2, dir[2], 1)
        if (dir == (3,1) || dir == (4,2))
            x1 = permute(flip(x1, 2+dir[1]), ((1,2),(3,4,5,6)))
            x2 = permute(flip(x2, 2+dir[2]), ((1,2),(3,4,5,6)))
        end

        x = [x1, x2]
        # x = symmetrize_cluster!(x1, x2, dir)
    end
    return x
end