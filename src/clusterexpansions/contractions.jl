function exponentiate_hamiltonian(twosite_op, onesite_op, cluster, β; nn_term = nothing)
    N = cluster.N
    pspace = domain(twosite_op)[1]
    H = []
    for (i, j) in cluster.bonds_indices
        term = ncon([twosite_op, [id(pspace) for _ in 1:(N - 2)]...], [[-i, -j, -N - i, -N - j], [[-k, -N - k] for k in setdiff(1:N, (i, j))]...], [false for _ in 1:(N - 1)])
        push!(H, permute(term, (Tuple(1:N), Tuple((N + 1):(2 * N)))))
    end
    for site in 1:N
        term = ncon([onesite_op, [id(pspace) for _ in 1:(N - 1)]...], [[-site, -N - site], [[-k, -N - k] for k in setdiff(1:N, site)]...], [false for _ in 1:N])
        push!(H, permute(term, (Tuple(1:N), Tuple((N + 1):(2 * N)))))
    end
    if !isnothing(nn_term)
        for (i, j) in cluster.diag_bonds_indices
            term = ncon([nn_term, [id(pspace) for _ in 1:(N - 2)]...], [[-i, -j, -N - i, -N - j], [[-k, -N - k] for k in setdiff(1:N, (i, j))]...], [false for _ in 1:(N - 1)])
            push!(H, permute(term, (Tuple(1:N), Tuple((N + 1):(2 * N)))))
        end
    end
    exp_H = exp(-β * sum(H))
    return exp_H
end

function contract_PEPO(T, cluster, PEPO, spaces)
    highest = [maximum([i[dir] for i in keys(PEPO)]) for dir in 1:4]
    highest_loop = [minimum([i[dir] for i in keys(PEPO)]) for dir in 1:4]
    total_spaces = [get_sum_space(h, hloop, spaces) for (h, hloop) in zip(highest, highest_loop)]
    triv_tensors_dir = [zeros(T, total_spaces[1]), zeros(T, total_spaces[2]), zeros(T, (total_spaces[3])'), zeros(T, (total_spaces[4])')]
    for dir in 1:4
        if dir > 2
            triv_tensors_dir[dir][1] = ones(T, total_spaces[dir][1]')
        else
            triv_tensors_dir[dir][1] = ones(T, total_spaces[dir][1])
        end
    end
    pspace = codomain(PEPO[(0, 0, 0, 0)])[1]
    O = get_PEPO(T, pspace, PEPO, spaces)
    # O_rot = rotl90_fermionic(O)
    # for (key, value) = PEPO
    #     shifted = tuple(circshift(collect(key), 2)...)
    #     println("key = $(key) has shifted $(shifted)")
    #     if shifted in PEPO.keys
    #         error = norm(value-rotl180_fermionic(PEPO[shifted]))/norm(value)
    #         println("error on key $(key) is $(error)")
    #     else
    #         println("key $(key) has no equivalent")
    #     end
    # end
    # println("rot invariance of PEPO = $(norm(O-O_rot))/$(norm(O))")

    N = cluster.N
    contraction_indices = fill(0, N, 6)
    for i in 1:cluster.N
        contraction_indices[i, 1] = -i
        contraction_indices[i, 2] = -i - N
    end
    for (count, (i, j)) in enumerate(cluster.bonds_indices)
        dir = get_direction(cluster.cluster[i], cluster.cluster[j])
        contraction_indices[i, dir[1] + 2] = count
        contraction_indices[j, dir[2] + 2] = count
    end
    start_count = length(cluster.bonds_indices)
    count = 1
    triv_tensors = []
    for t in 1:N
        for d in 1:4
            if contraction_indices[t, d + 2] == 0
                contraction_indices[t, d + 2] = start_count + count
                push!(triv_tensors, triv_tensors_dir[d])
                count += 1
            end
        end
    end
    nontriv_tensors = fill(O, N)
    triv_contractions = [[i] for i in 1:(count - 1)]

    # Swap the order, first do the trivial contractions to improve efficiency
    conversion = merge!(Dict(j => j + count - 1 for j in 1:start_count), Dict(j => j - start_count for j in (start_count + 1):(start_count + count - 1)))
    nontriv_contractions_base = [contraction_indices[i, :] for i in 1:size(contraction_indices)[1]]
    nontriv_contractions = [[(j > 0) ? conversion[j] : j for j in contraction] for contraction in nontriv_contractions_base]

    all_contractions = vcat(nontriv_contractions, triv_contractions)
    all_tensors = vcat(nontriv_tensors, triv_tensors)

    contracted_tens = ncon(all_tensors, all_contractions)
    return permute(contracted_tens, (Tuple(1:N), Tuple((N + 1):(2 * N))))
end
