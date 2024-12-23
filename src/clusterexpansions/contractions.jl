function exponentiate_hamiltonian(twosite_op, cluster, β)
    N = cluster.N
    pspace = twosite_op.dom[1]
    H = []
    for bond = cluster.bonds_indices
        (i,j) = bond
        term = ncon([twosite_op, [id(pspace) for _ = 1:N-2]...], [[-i, -j, -N-i, -N-j], [[-k, -N-k] for k = setdiff(1:N, (i,j))]...], [false for _ = 1:N-1])
        push!(H, permute(term, Tuple(1:N), Tuple(N+1:2*N)))
    end
    exp_H = exp(-β*sum(H)) 
    return exp_H
end

function exponentiate_hamiltonian(cluster, β)
    twosite_op = -S_xx() + S_yy() - S_zz()
    return exponentiate_hamiltonian(twosite_op, cluster, β)
end

function contract_PEPO(cluster, PEPO)
    highest = [get_size_level(maximum([i[dir] for i = keys(PEPO)])) for dir = 1:4]
    highest_loop = [get_size_level_loop(minimum([i[dir] for i = keys(PEPO)])) for dir = 1:4]
    sizes = [h+h_loop for (h,h_loop) = zip(highest, highest_loop)]

    triv_tensors_dir = [Tensor(zeros,ℂ^sizes[1]), Tensor(zeros,ℂ^sizes[2]), Tensor(zeros,(ℂ^sizes[3])'), Tensor(zeros,(ℂ^sizes[4])')]
    for dir = 1:4
        triv_tensors_dir[dir][][1] = 1.0
    end

    O = get_PEPO(ℂ^2, PEPO)

    N = cluster.N
    contraction_indices = fill(0, N, 6)
    for i = 1:cluster.N
        contraction_indices[i,1] = -i
        contraction_indices[i,2] = -i-N
    end
    for (count,(i,j)) = enumerate(cluster.bonds_indices)
        dir = get_direction(cluster.cluster[i],cluster.cluster[j])
        contraction_indices[i,dir[1]+2] = count
        contraction_indices[j,dir[2]+2] = count
    end    
    start_count = length(cluster.bonds_indices)
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
    nontriv_tensors = fill(O, N)
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