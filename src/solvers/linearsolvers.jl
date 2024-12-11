function apply_contraction(x, PEPO, cluster_levels)
    return nothing
end

function get_triv_tensors(conjugated, trivspace)
    tensor = Tensor([1.0], trivspace)
    tensor_conj = Tensor([1.0], trivspace')
    return [conj ? tensor_conj : tensor for conj = conjugated] 
end

function get_A(PEPO, levels_sites, number_of_bonds, sites_to_update, conjugated, open_conjugated, open_indices, contraction_indices)
    trivspace = ℂ^1
    included_sites = setdiff(1:length(levels_sites)[1], sites_to_update)
    nontriv_tensors = [PEPO[levels_sites[site]] for site = included_sites]
    triv_tensors = vcat(get_triv_tensors(conjugated, trivspace), get_triv_tensors(open_conjugated, trivspace))
    triv_contractions = vcat([[number_of_bonds+i] for i = 1:length(conjugated)], [[i] for i = open_indices])
    all_contractions = vcat([contraction_indices[i,:] for i = 1:size(contraction_indices)[1]], triv_contractions)
    all_tensors = vcat(nontriv_tensors, triv_tensors)

    A = ncon(all_tensors, all_contractions)
    return A
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
    # N is length of the cluster
    (i,j) = sites_to_update
    included_sites = setdiff(1:N, sites_to_update)
    contraction_indices = vcat(-included_sites, -included_sites .- N, [1, 2, 3, 4, 5, 6])

    Ax = ncon([A, x], [contraction_indices, [-i, -N-i, 1, 2, 3, -j, -N-j, 4, 5, 6]])
    Ax = permute(Ax, ((Tuple(1:N)), (Tuple(N+1:2*N))))
    # println("forward - contractions are $(contraction_indices)\n and $([-i, -N-i, 1, 2, 3, -j, -N-j, 4, 5, 6])")
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
    # println("backward - contractions are $(vcat(1:2*length(included_sites), [-3, -4, -5, -8, -9, -10]))\n and $(contracted_b)")
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

function solve_index(updates, A, exp_H, conjugated, sites_to_update, levels_to_update, dir, N; spaces = i -> ℂ^(2^(2*i)))
    pspace = ℂ^2
    trivspace = ℂ^1
    # x0 = TensorMap(randn, pspace ⊗ pspace' ⊗ prod([conj ? trivspace' : trivspace for conj = conjugated[1]]), pspace ⊗ pspace' ⊗ prod([conj ? trivspace' : trivspace for conj = conjugated[2]]))
    if N == 2
        x = TensorMap(randn, pspace ⊗ pspace' ⊗ prod([conj ? trivspace : trivspace' for conj = conjugated[1]]), pspace' ⊗ pspace ⊗ prod([conj ? trivspace' : trivspace for conj = conjugated[2]]))
        b = permute(exp_H, ((1,3), (2,4)))
        x[][:,:,1,1,1,:,:,1,1,1] = b[]
    elseif length(sites_to_update) == 2
        init_spaces = [[spaces(levels_to_update[1][i]) for i = 1:4 if (i != dir[1])], [spaces(levels_to_update[2][i]) for i = 1:4 if (i != dir[2])]]
        # x0 = TensorMap(randn, pspace ⊗ pspace' ⊗ prod([conj ? trivspace : trivspace' for conj = conjugated[1]]), pspace' ⊗ pspace ⊗ prod([conj ? trivspace' : trivspace for conj = conjugated[2]]))
        # x0 = TensorMap(randn, pspace ⊗ pspace' ⊗ prod([conj ? spaces(levels_to_update[1][i])' : spaces(levels_to_update[1][i]) for (i,conj) = enumerate(conjugated[1]) if (i != dir[1])]), pspace ⊗ pspace' ⊗ prod([conj ? spaces(levels_to_update[2][i])' : spaces(levels_to_update[2][i]) for (i,conj) = enumerate(conjugated[2]) if (i != dir[2])]))
        x0 = TensorMap(randn, pspace ⊗ pspace' ⊗ prod([conj ? space : space' for (conj,space) = zip(conjugated[1], init_spaces[1])]), pspace' ⊗ pspace ⊗ prod([conj ? space' : space for (conj,space) = zip(conjugated[2], init_spaces[2])]))

        apply_A = (x, val) -> apply_A_twosite(A, x, sites_to_update, N, val)
        Ax = apply_A_twosite(A, x0, sites_to_update, N, Val{false}())
        x1 = apply_A_twosite(A, Ax, sites_to_update, N, Val{true}())
        @assert (x0.dom == x1.dom) && (x0.codom == x1.codom)
        @assert (Ax.dom == exp_H.dom) && (Ax.codom == exp_H.codom)
        # println("x0 = $(summary(x0))")
        # println("for length = 2, A = $(summary(A))")
        # println("exp_H = $(summary(exp_H))")
        x, info = linsolve(apply_A, exp_H, x0, LSMR(verbosity = 1))
    elseif length(sites_to_update) == 1
        # x0 = TensorMap(randn, pspace ⊗ pspace', prod([conj ? trivspace' : trivspace for conj = conjugated[1]]))
        # x0 = TensorMap(randn, pspace ⊗ pspace', prod([spaces(levels_to_update[1][i]) for i = 1:4]))
        x0 = TensorMap(randn, pspace ⊗ pspace', prod([conj ? spaces(levels_to_update[1][i])' : spaces(levels_to_update[1][i]) for (i,conj) = enumerate(conjugated[1])]))
        apply_A = (x, val) -> apply_A_onesite(A, x, sites_to_update, N, val)
        Ax = apply_A_onesite(A, x0, sites_to_update, N, Val{false}())
        x1 = apply_A_onesite(A, Ax, sites_to_update, N, Val{true}())
        @assert (x0.dom == x1.dom) && (x0.codom == x1.codom)
        @assert (Ax.dom == exp_H.dom) && (Ax.codom == exp_H.codom)
        # println("for length = 1, A = $(summary(A))")
        x, info = linsolve(apply_A, exp_H, x0, LSMR(verbosity = 1))
        x = [x]
    else
        error("Something went terribly wrong")
    end

    if length(sites_to_update) == 2
        x1, Σ, x2 = tsvd(x)
        x1 = x1 * sqrt(Σ)
        x2 = sqrt(Σ) * x2
        @assert norm(x - x1 * x2) < 1e-10

        x1 = permute_dir(x1, dir[1], 0)
        x2 = permute_dir(x2, dir[2], 1)
        if (dir == (3,1) || dir == (4,2))
            vspace = x1.dom[dir[1]]
            I₁ = isometry(vspace, vspace')
            I₂ = isometry(vspace', vspace)
            ind₁ = [(i==dir[1]) ? 1 : -i-2 for i=1:4]
            ind₂ = [(i==dir[2]) ? 1 : -i-2 for i=1:4]
            x1 = permute(ncon([x1, I₁], [vcat([-1, -2], ind₁), [1, -2-dir[1]]]), ((1,2),(3,4,5,6)))
            x2 = permute(ncon([x2, I₂], [vcat([-1, -2], ind₂), [1, -2-dir[2]]]), ((1,2),(3,4,5,6)))
        end
        x = [x1 x2]
    end
    return x
    spaces = [(0, 0, 0, 0)]
    sp = spaces[1]
    return [TensorMap(randn, ℂ^2 ⊗ (ℂ^2)' ← ℂ^(1) ⊗ ℂ^(1) ⊗ (ℂ^(1))' ⊗ (ℂ^(1))') for sp in spaces]
    return [TensorMap(randn, ℂ^2 ⊗ (ℂ^2)' ← ℂ^(sp[1]+1) ⊗ ℂ^(sp[2]+1) ⊗ (ℂ^(sp[3]+1))' ⊗ (ℂ^(sp[4]+1))') for sp in spaces]
end

function get_other_tensors(cluster, PEPO, levels_sites, sites_to_update, contraction_indices, conjugated, number_of_bonds)
    pspace = ℂ^2
    edge = Tensor([1.0], ℂ^1)
    conj_edge = Tensor([1.0], (ℂ^1)')
    edge_tensors = [i ? conj_edge : edge for i = conjugated]
    cluster_tensors = [PEPO[lev] for (i,lev) = enumerate(levels_sites) if !(i ∈ sites_to_update)]
    contraction_indices = collect.(contraction_indices)
    contraction_indices_edges = [[i] for i = number_of_bonds:number_of_bonds+length(conjugated)]
    final_tensor = ncon(vcat(cluster_tensors, edge_tensors), vcat(contraction_indices, contraction_indices_edges))
    return final_tensor
end

function fixed_tensors(cluster, indices_to_update, bonds_indices, levels_sites, PEPO)
    tensors = []
    indices = []
    for (i,c) = enumerate(cluster)
        if !(i ∈ indices_to_update)
            push!(tensors, PEPO[levels_sites[i]])
        end
    end
end

function twosite_update(cluster, β, sites_to_update, contraction_indices, conjugated)
    bonds_sites, bonds_indices = get_bonds(cluster)

    # F = apply_contraction(x, PEPO, )

    current_tensors = []

    RHS = exponentiate_hamiltonian(cluster, β)
    pspace = ℂ^2
    edge = Tensor([1.0], ℂ^1)
    conj_edge = Tensor([1.0], (ℂ^1)')
    x0 = TensorMap(randn, pspace ⊗ pspace, pspace ⊗ pspace)
    f = x -> apply_contraction(x, PEPO, cluster_levels)
    x = linsolve(f, RHS, x0)
end