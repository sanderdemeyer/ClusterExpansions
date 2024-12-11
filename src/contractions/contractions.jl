function exponentiate_hamiltonian(twosite_op, cluster, β, N)
    pspace = twosite_op.dom[1]
    _, bond_indices = get_bonds(cluster)
    H = []
    for bond = bond_indices
        (i,j) = bond
        term = ncon([twosite_op, [id(pspace) for _ = 1:N-2]...], [[-i, -j, -N-i, -N-j], [[-k, -N-k] for k = setdiff(1:N, (i,j))]...], [false for _ = 1:N-1])
        push!(H, permute(term, Tuple(1:N), Tuple(N+1:2*N)))
    end
    return exp(-β*sum(H))
end

function exponentiate_hamiltonian(cluster, β)
    N = length(cluster)
    twosite_op = -S_zz()
    return exponentiate_hamiltonian(twosite_op, cluster, β, N)
end

function contract_PEPS(PEPO, cluster)
    println("cluster = $(cluster)")

    highest = [get_size_level(maximum([i[dir] for i = keys(PEPO)])) for dir = 1:4]
    println("highest")
    tensor = Tensor(ℂ^highest)
    tensor_conj = Tensor((ℂ^highest)')
    tensor[][1] = 1.0
    tensor_conj[][1] = 1.0
    # tensor = Tensor([1.0], trivspace)
    # tensor_conj = Tensor([1.0], trivspace')

    _, bonds_indices = get_bonds(cluster)
    levels_sites = get_levels_sites(cluster)
    println("levels = $(levels_sites)")
    println(bonds_indices)
    O = get_PEPO(ℂ^2, PEPO)

    N = length(cluster)
    contraction_indices = fill(0, N, 6)
    for i = 1:N
        contraction_indices[i,1] = -i
        contraction_indices[i,2] = -i-N
    end
    for (count,(i,j)) = enumerate(bonds_indices)
        dir = get_direction(cluster[i],cluster[j])
        contraction_indices[i,dir[1]+2] = count
        contraction_indices[j,dir[2]+2] = count
    end    
    start_count = length(bonds_indices)
    count = 1
    conjugated = Bool[]
    for t = 1:N
        for d = 1:4
            if contraction_indices[t,d+2] == 0
                contraction_indices[t,d+2] = start_count + count 
                push!(conjugated, d > 2)
                count += 1
            end
        end
    end
    println("conj = $(conjugated)")
    println(start_count+1:start_count+count-1)
    println("contraction_indices = $(contraction_indices)")
    triv_tensors = [conj ? tensor_conj : tensor for conj = conjugated]
    nontriv_tensors = fill(O, N)
    triv_contractions = [[i] for i = start_count+1:start_count+count-1]

    println("lengths are $(length([contraction_indices[i,:] for i = 1:size(contraction_indices)[1]])) = $(length(triv_contractions))")
    println("and $(length(nontriv_tensors)) and $(length(triv_tensors))")

    all_contractions = vcat([contraction_indices[i,:] for i = 1:size(contraction_indices)[1]], triv_contractions)
    all_tensors = vcat(nontriv_tensors, triv_tensors)

    println("contr = $(all_contractions)")
    for t = all_tensors
        println(summary(t))
    end

    return ncon(all_tensors, all_contractions)
end


cont = contract_PEPS(result, cluster)
