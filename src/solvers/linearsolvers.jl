function apply_contraction(x, PEPO, cluster_levels)
    return nothing
end

function get_other_tensors(cluster, PEPO, levels_sites, sites_to_update, contraction_indices, conjugated)
    pspace = ℂ^2
    edge = Tensor([1.0], ℂ^1)
    conj_edge = Tensor([1.0], (ℂ^1)')
    edge_tensors = [i ? conj_edge : edge for i = conjugated]
    cluster_tensors = [PEPO[lev] for (i,lev) = enumerate(levels_sites) if !(i ∈ sites_to_update)]
    println("edge tensors = $(edge_tensors)")
    println("cluster tensors = $(cluster_tensors)")
end

function fixed_tensors(cluster, indices_to_update, bonds_indices, levels_sites, PEPO)
    tensors = []
    indices = []
    for (i,c) = enumerate(cluster)
        if !(i ∈ indices_to_update)
            push!(tensors, PEPO[levels_sites[i]])
            push!
        end
    end
end

function twosite_update(cluster, β, sites_to_update, contraction_indices, conjugated)
    bonds_sites, bonds_indices = get_bonds(cluster)

    F = apply_contraction(x, PEPO, )

    current_tensors = []

    RHS = exponentiate_hamiltonian(cluster, β)
    pspace = ℂ^2
    edge = Tensor([1.0], ℂ^1)
    conj_edge = Tensor([1.0], (ℂ^1)')
    x0 = TensorMap(randn, pspace ⊗ pspace, pspace ⊗ pspace)
    f = x -> apply_contraction(x, PEPO, cluster_levels)
    x = linsolve(f, RHS, x0)
end