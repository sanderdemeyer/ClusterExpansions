function get_triv_tensors(conjugated, trivspace)
    tensor = Tensor([1.0], trivspace)
    tensor_conj = Tensor([1.0], trivspace')
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
    bond_space = x2.codom[1]
    x0 = TensorMap(bond_space', bond_space)

    println("this is the summary:")
    println(summary(x1))
    println(summary(x2))
    println(summary(x0))

    result = apply_A_N_2(x2, x0, Val(false))
    println("result = $(summary(result))")

    apply_A = (x, val) -> apply_A_N_2(x2, x, val)
    x, info = linsolve(apply_A, x1, x0, LSMR(verbosity = 1, maxiter = 1000))
    println("done - x = $(x)")

    return [x1 x2]
end

function apply_A_N_2(A, x, ::Val{false})
    @tensor Ax[-1 -2 -3 -4 -5; -6] := A[1; -1 -2 -4 -5 -3] * x[-6; 1]
    return Ax
end

function apply_A_N_2(x1, y ::Val{true})
    trivspace = x1.dom[1]
    I = isometry(trivspace', trivspace)
    @tensor x2_new[-1 -2; -3 -4 -5 -6] := conj(y[1; -6]) * x1[-1 -2; -5 1 -3 2] * conj(I[-4; 2])
    return x2_new
end

function get_A(cluster, PEPO, sites_to_update)
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
    triv_tensors = vcat(get_triv_tensors(conjugated, trivspace), get_triv_tensors(open_conjugated, trivspace))
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

function solve_index(A, exp_H, conjugated, sites_to_update, levels_to_update, dir, N; spaces = i -> ℂ^(2^(2*i)))
    pspace = ℂ^2
    trivspace = ℂ^1
    
    # x0 = TensorMap(randn, pspace ⊗ pspace' ⊗ prod([conj ? trivspace' : trivspace for conj = conjugated[1]]), pspace ⊗ pspace' ⊗ prod([conj ? trivspace' : trivspace for conj = conjugated[2]]))
    if N == 2
        x = TensorMap(zeros, pspace ⊗ pspace' ⊗ prod([conj ? trivspace : trivspace' for conj = conjugated[1]]), pspace' ⊗ pspace ⊗ prod([conj ? trivspace' : trivspace for conj = conjugated[2]]))
        b = permute(exp_H, ((1,3), (2,4)))
        x[][:,:,1,1,1,:,:,1,1,1] = b[]
    elseif length(sites_to_update) == 2
        init_spaces = [[spaces(levels_to_update[1][i]) for i = 1:4 if (i != dir[1])], [spaces(levels_to_update[2][i]) for i = 1:4 if (i != dir[2])]]
        x0 = TensorMap(zeros, pspace ⊗ pspace' ⊗ prod([conj ? space : space' for (conj,space) = zip(conjugated[1], init_spaces[1])]), pspace' ⊗ pspace ⊗ prod([conj ? space' : space for (conj,space) = zip(conjugated[2], init_spaces[2])]))
        apply_A = (x, val) -> apply_A_twosite(A, x, sites_to_update, N, val)

        Ax = apply_A_twosite(A, x0, sites_to_update, N, Val{false}())
        x1 = apply_A_twosite(A, Ax, sites_to_update, N, Val{true}())
        @assert (x0.dom == x1.dom) && (x0.codom == x1.codom)
        @assert (Ax.dom == exp_H.dom) && (Ax.codom == exp_H.codom)

        x, info = linsolve(apply_A, exp_H, x0, LSMR(verbosity = 1, maxiter = 1000))
    elseif length(sites_to_update) == 1
        x0 = TensorMap(zeros, pspace ⊗ pspace', prod([conj ? spaces(levels_to_update[1][i])' : spaces(levels_to_update[1][i]) for (i,conj) = enumerate(conjugated[1])]))
        apply_A = (x, val) -> apply_A_onesite(A, x, sites_to_update, N, val)
        
        Ax = apply_A_onesite(A, x0, sites_to_update, N, Val{false}())
        x1 = apply_A_onesite(A, Ax, sites_to_update, N, Val{true}())
        @assert (x0.dom == x1.dom) && (x0.codom == x1.codom)
        @assert (Ax.dom == exp_H.dom) && (Ax.codom == exp_H.codom)
        x, info = linsolve(apply_A, exp_H, x0, LSMR(verbosity = 1, maxiter = 1000))
        x = [x]
    else
        error("Something went terribly wrong")
    end

    if length(sites_to_update) == 2
        x1, Σ, x2 = tsvd(x)
        x1 = x1 * sqrt(Σ)
        x2 = sqrt(Σ) * x2
        @assert norm(x - x1 * x2) < 1e-10
        println("summaries")
        println(summary(x1))
        println(summary(x2))
        x1, x2 = symmetrize_cluster(x1, x2)

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
        # x = symmetrize_cluster!(x1, x2, dir)
    end

    return x
end