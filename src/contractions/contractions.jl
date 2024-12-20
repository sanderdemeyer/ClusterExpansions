function exponentiate_hamiltonian(twosite_op, cluster, β, N)
    pspace = twosite_op.dom[1]
    _, bond_indices = get_bonds(cluster)
    H = []
    for bond = bond_indices
        (i,j) = bond
        term = ncon([twosite_op, [id(pspace) for _ = 1:N-2]...], [[-i, -j, -N-i, -N-j], [[-k, -N-k] for k = setdiff(1:N, (i,j))]...], [false for _ = 1:N-1])
        push!(H, permute(term, Tuple(1:N), Tuple(N+1:2*N)))
    end
    exp_H = exp(-β*sum(H)) 
    return exp_H
end

function exponentiate_hamiltonian(cluster, β)
    N = length(cluster)
    twosite_op = -S_xx() + S_yy() - S_zz()
    return exponentiate_hamiltonian(twosite_op, cluster, β, N)
end

function contract_PEPS(cluster, PEPO)
    sizes = [get_size_level(maximum([i[dir] for i = keys(PEPO)])) for dir = 1:4]
    triv_tensors_dir = [Tensor(zeros,ℂ^sizes[1]), Tensor(zeros,ℂ^sizes[2]), Tensor(zeros,(ℂ^sizes[3])'), Tensor(zeros,(ℂ^sizes[4])')]
    for dir = 1:4
        triv_tensors_dir[dir][][1] = 1.0
    end

    _, bonds_indices = get_bonds(cluster)
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
    triv_tensors = []
    for t = 1:N
        for d = 1:4
            if contraction_indices[t,d+2] == 0
                contraction_indices[t,d+2] = start_count + count 
                push!(triv_tensors, triv_tensors_dir[d])
                count += 1
            end
        end
    end
    # triv_tensors = [conj ? tensor_conj : tensor for conj = conjugated]
    nontriv_tensors = fill(O, N)
    # triv_contractions = [[i] for i = start_count+1:start_count+count-1]
    triv_contractions = [[i] for i = 1:count-1]

    # Swap the order, first do the trivial contractions to improve efficiency
    conversion = merge!(Dict(j => j+count-1 for j = 1:start_count), Dict(j => j-start_count for j = start_count+1:start_count+count-1))
    nontriv_contractions_base = [contraction_indices[i,:] for i = 1:size(contraction_indices)[1]]
    nontriv_contractions = [[(j > 0) ? conversion[j] : j for j = contraction] for contraction = nontriv_contractions_base]

    all_contractions = vcat(nontriv_contractions, triv_contractions)
    all_tensors = vcat(nontriv_tensors, triv_tensors)

    contracted_tens = ncon(all_tensors, all_contractions)
    return permute(contracted_tens, (Tuple(1:N),Tuple(N+1:2*N)))
end