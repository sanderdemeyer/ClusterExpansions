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