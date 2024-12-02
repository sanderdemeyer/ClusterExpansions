function exponentiate_hamiltonian(twosite_op, cluster, β, N)
    bonds = get_bonds(cluster)
    H = []
    for bond = bonds
        # terms = [(ind in bond) ? ... : id() for ind = cluster]
        term = ncon(hcat([twosite_op], [id() for _ = 1:N-2]), hcat([], []))
        push!(H, term)
    end
    return exp(-β*sum(H))
end

current_indices = [(0,0,0,0)]

clusters = get_nontrivial_terms(4)
exponentiate_hamiltonian(clusters[27])