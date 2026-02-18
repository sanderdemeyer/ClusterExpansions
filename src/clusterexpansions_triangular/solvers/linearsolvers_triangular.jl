function get_graph(lattice::Triangular, cluster)
    graph = fill(0, cluster.N, 6)
    for (bond_s, bond_i) = zip(cluster.bonds_sites,cluster.bonds_indices)
        dir = get_direction(lattice, bond_s[1], bond_s[2])
        graph[bond_i[1], dir[1]] = bond_i[2]
        graph[bond_i[2], dir[2]] = bond_i[1]
    end
    return graph
end

function get_twists(lattice::Triangular, dir)
    if dir == (1,4) || dir == (2,5) || dir == (3,6)
        return [1 2 6 7 8]
    elseif dir == (4,1) || dir == (5,2) || dir == (6,3)
        return [1 2 3 6 7]
    else
        @error "Unexpected value for dir"
    end
end

function get_A(lattice::Triangular, T, cluster, PEPO, sites_to_update)
    updates = length(sites_to_update)
    fixed_tensors = cluster.N - updates
    graph = get_graph(lattice, cluster)
    contraction_indices = fill(0, fixed_tensors, 8)
    for i = 1:fixed_tensors
        contraction_indices[i,1] = -i
        contraction_indices[i,2] = -(fixed_tensors+i)
    end
    included_sites = setdiff(1:cluster.N, sites_to_update)
    conv = i -> i-sum([j ∈ sites_to_update for j = 1:i])
    m = 1
    for i = included_sites
        for j = 1:6
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
        for j = 1:6
            if (graph[i,j] == 0)
                contraction_indices[conv(i),j+2] = m
                push!(conjugated, j > 3)
                m += 1
            end
        end
    end
    open_conjugated = Bool[]
    open_indices = Int[]
    # spaces = fill((0,0), updates, 5-updates)
    m = 2*fixed_tensors+1
    # opposite = dir -> 4 - dir + 2*(dir ∈ (2,4))
    for (enum,i) = enumerate(sites_to_update)
        kⱼ = 0
        for j = 1:6
            if graph[i,j] ∉ sites_to_update
                kⱼ += 1
                if graph[i,j] == 0
                    push!(open_conjugated, j > 3)
                    push!(open_indices, -m)
                else
                    ind = findall(x -> x == i, graph[graph[i,j],:])[1]
                    contraction_indices[conv(graph[i,j]),ind+2] = -m
                    # spaces[enum, kⱼ] = (graph[i,j],opposite(j))
                end
                m += 1
            end
        end
    end
    trivspace = domain(PEPO[0,0,0,0,0,0])[1]
    # included_sites = setdiff(1:length(cluster.levels_sites)[1], sites_to_update)
    included_sites = setdiff(1:cluster.N, sites_to_update)
    nontriv_tensors = [PEPO[cluster.levels_sites[site]] for site = included_sites]
    triv_tensors = vcat(get_triv_tensors(T, conjugated, trivspace), get_triv_tensors(T, open_conjugated, trivspace))
    triv_contractions = vcat([[number_of_bonds+i] for i = 1:length(conjugated)], [[i] for i = open_indices])
    all_contractions = vcat([contraction_indices[i,:] for i = 1:size(contraction_indices)[1]], triv_contractions)
    all_tensors = vcat(nontriv_tensors, triv_tensors)

    A = ncon(all_tensors, all_contractions)
    len = length(domain(A)) + length(codomain(A))
    A = permute(A, (Tuple(1:len-6), Tuple(len-5:len)))
    return A
end

function apply_A_onesite(lattice::Triangular, A, x::TensorMap, sites_to_update, N, ::Val{false})
    Ax = ncon([A, x], [hcat(transpose(-1:-1:-2*N+2), [1 2 3 4 5 6]), [1 2 3 4 5 6 -2*N+1 -2*N]])
    len = length(domain(Ax)) + length(codomain(Ax))
    Ax = permute(Ax, (Tuple(1:len-2), Tuple(len-1:len)))
    return Ax
end

function apply_A_onesite(lattice::Triangular, A, Ax::TensorMap, sites_to_update, N, ::Val{true})
    Ax′ = twist(Ax, N:2*N-2)
    x = ncon([A, Ax′], [hcat(transpose(1:2*N-2), [-1 -2 -3 -4 -5 -6]), hcat(transpose(1:2*N-2), [-7 -8])], [true false])

    x = twist(x, [1 2 3])
    x = permute(x, ((1,2,3,4,5,6),(7,8)))
    return x
end

function apply_A_twosite(lattice::Triangular, A, x::TensorMap, N, dir, ::Val{false})
    Ax = ncon([A, x], [hcat(transpose(-1:-1:-2*N+4), [1 2 3 4 5 6 7 8 9 10]), [1 2 3 4 5 6 7 8 9 10 -2*N+3 -2*N+2 -2*N+1 -2*N]])
    len = length(domain(Ax)) + length(codomain(Ax))
    Ax = permute(Ax, (Tuple(1:len-4), Tuple(len-3:len)))
    return Ax
end

function apply_A_twosite(lattice::Triangular, A, Ax::TensorMap, N, dir, ::Val{true})
    Ax′ = twist(Ax, N-1:2*N-4)

    x = ncon([A, Ax′], [hcat(transpose(1:2*N-4), [-1 -2 -3 -4 -5 -6 -7 -8 -9 -10]), hcat(transpose(1:2*N-4), [-11 -12 -13 -14])], [true false])

    x = twist(x, get_twists(lattice, dir))
    x = permute(x, ((1,2,3,4,5,6,7,8,9,10),(11,12,13,14)))
    return x
end

function permute_dir(lattice::Triangular, x, dir, second)
    tup = zeros(Int,6)
    tup[dir] = 8-7*second
    count = 1
    for i = 1:6
        if tup[i] == 0
            tup[i] = count+2+second
            count += 1
        end
    end
    return permute(x, ((1,2).+second, (Tuple(tup))))
end

function solve_index(lattice::Triangular, T, A, exp_H, conjugated, sites_to_update, levels_to_update, dir, N, spaces; verbosity = 2, svd = true)
    trivspace = spaces(0)
    if N == 2
        b = permute(exp_H, ((1,3), (2,4)))
        if dir == (4,1) || dir == (5,2)
            Isom_domain = isomorphism(domain(b), domain(b) ⊗ trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace' ⊗ trivspace')
            Isom_codomain = isomorphism(codomain(b) ⊗ trivspace' ⊗ trivspace' ⊗ trivspace' ⊗ trivspace ⊗ trivspace, codomain(b))
        elseif dir == (3,6)
            Isom_domain = isomorphism(domain(b), domain(b) ⊗ trivspace ⊗ trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace')
            Isom_codomain = isomorphism(codomain(b) ⊗ trivspace' ⊗ trivspace' ⊗ trivspace ⊗ trivspace ⊗ trivspace, codomain(b))
        else
            @error "Unexpected value for dir"
        end
        x = Isom_codomain * b * Isom_domain
        # x.data = b.data
        # x[][:,:,1,1,1,:,:,1,1,1] = b[]
    elseif length(sites_to_update) == 2
        A = permute(A, (Tuple(1:2*N-4),Tuple(2*N-3:2*N+6)))

        apply_A = (x, val) -> apply_A_twosite(lattice, A, x, N, dir, val)

        included_sites = setdiff(1:N, sites_to_update)
        exp_H_flipped = permute(exp_H, ((included_sites..., (included_sites .+ N)...), (sites_to_update..., (sites_to_update .+ N)...)))

        if scalartype(exp_H) ∈ [Complex{BigFloat}, BigFloat]
            x, info = lssolve(apply_A, exp_H_flipped, LSMR(verbosity = verbosity, maxiter = 2000, tol = BigFloat(1e-70)), real(T(0.0)))
            # x = permute(x, ((1,6,3,4,8), (2,7,9,5,10)))
            # x = (x + x')/2
            # x = permute(x, ((1,6,3,4,9), (2,7,5,8,10)))
        else
            x, info = lssolve(apply_A, exp_H_flipped, LSMR(verbosity = verbosity, maxiter = 1000), real(T(0.0)))
        end
        error = norm(apply_A(x, Val(false)) - exp_H_flipped)/norm(exp_H_flipped)
        if error > 1e-10 && verbosity > 0
            @warn "Error made on the solution is of the order $(error)"
        end
        # x = permute(x, ((7,9,1,2,3),(8,10,4,5,6)))
        x = permute(x, ((11,13,1,2,3,4,5),(12,14,6,7,8,9,10)))

    elseif length(sites_to_update) == 1
        A = permute(A, (Tuple(1:2*N-2),Tuple(2*N-1:2*N+4)))

        apply_A = (x, val) -> apply_A_onesite(lattice, A, x, sites_to_update, N, val)
        
        included_sites = setdiff(1:N, sites_to_update[1])
        exp_H_flipped = permute(exp_H, ((included_sites..., (included_sites .+ N)...), (sites_to_update[1], sites_to_update[1]+N)))

        if scalartype(exp_H) ∈ [Complex{BigFloat}, BigFloat]
            x, info = lssolve(apply_A, exp_H_flipped, LSMR(verbosity = verbosity, maxiter = 2000, tol = BigFloat(1e-70)), real(T(0.0)))
        else
            x, info = lssolve(apply_A, exp_H_flipped, LSMR(verbosity = verbosity, maxiter = 1000), real(T(0.0)))
        end
        error = norm(apply_A(x, Val(false)) - exp_H_flipped)/norm(exp_H_flipped)
        if error > 1e-10 && verbosity > 0
            @warn "Error made on the solution is of the order $(error) = $(norm(apply_A(x, Val(false)) - exp_H_flipped)) / $(norm(exp_H_flipped))"
        end
        x = permute(x, ((7,8),(1,2,3,4,5,6)))
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
        if svd
            U, Σ, V = tsvd(x, trunc = truncspace(spaces(levels_to_update[1][dir[1]])))
            x1 = U * sqrt(Σ)
            x2 = sqrt(Σ) * V
            if norm(x - x1 * x2)/norm(x) > eps(real(T))*1e2 && verbosity > 0
                @warn "Error made on the SVD is of the order $(norm(x - x1 * x2)/norm(x))"
            end
        else
            if dir == (4,1)
                x = flip(x, (2,3,4,5,8,12,13,14))
                x = permute(x, ((1,2,3,4,5,6,7), (8,9,12,13,14,10,11)))
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
            
            x1 = flip(x1, (2,3,4,5))
            x2 = permute(x2, ((1,), (2,3,7,8,4,5,6)))
            x2 = flip(x2, (2,6,7,8))
        end
        x1 = permute_dir(lattice, x1, dir[1], 0)
        x2 = permute_dir(lattice, x2, dir[2], 1)
        if (dir == (4,1) || dir == (5,2))
            x1 = flip(x1, 2+dir[1])
            x2 = flip(x2, 2+dir[2]; inv = true)
        end
        # x2_rot = rotl180_fermionic(x2)
        x = [x1, x2]
        # x = symmetrize_cluster!(x1, x2, dir)
    elseif length(sites_to_update) >= 3
        @error "This happens because the error in the previous levels is too small."
    end
    return x
end