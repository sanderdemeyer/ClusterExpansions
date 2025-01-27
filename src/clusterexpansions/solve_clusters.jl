function get_update_dir(c, sites_to_update)
    if length(sites_to_update) == 2
        dir = (c[sites_to_update[2]][1] - c[sites_to_update[1]][1], c[sites_to_update[2]][2] - c[sites_to_update[1]][2])
        conjugated = get_conjugated(dir)
    elseif length(sites_to_update) == 1
        dir = 0
        conjugated = [Bool[0, 0, 1, 1]]
    else
        error("Number of sites to update $(length(sites_to_update)) not implemented")
    end
    return dir, conjugated
end

function solve_cluster(c, PEPO, β, twosite_op, spaces; levels_convention = "tree_depth", symmetry = nothing, verbosity = 2)
    cluster = Cluster(c; levels_convention = levels_convention)
    exp_H = exponentiate_hamiltonian(twosite_op, cluster, β)
    residual = contract_PEPO(cluster, PEPO, spaces)
    RHS = exp_H - residual
    @assert !(any(isnan.(convert(Array,RHS[][:])))) "RHS contains elements that are NaN"
    sites_to_update = [i for (i,levels) = enumerate(cluster.levels_sites) if !(levels ∈ keys(PEPO))]
    length(sites_to_update) == 0 && return
    levels_to_update = cluster.levels_sites[sites_to_update]

    println("Norm of RHS = $(norm(RHS)): Difference of $(norm(exp_H)) and $(norm(residual))")

    if length(sites_to_update) == 4
        solutions, _ = solve_4_loop(RHS, spaces(-1), levels_to_update; verbosity = verbosity)
    elseif length(sites_to_update) ∈ [1, 2]
        A = get_A(cluster, PEPO, sites_to_update)
        dir, conjugated = get_update_dir(c, sites_to_update)
        solutions = solve_index(A, exp_H-residual, conjugated, sites_to_update, levels_to_update, get_direction(dir), cluster.N, spaces; verbosity = verbosity)
    end
    levels_to_update, solutions = symmetrize(symmetry, levels_to_update, solutions)
    merge!(PEPO, Dict(zip(levels_to_update, solutions)))
end

function get_nontrivial_terms(N; prev_clusters = [[(0,0)]])
    if N == 1
        return [[(0,0)]]
    end
    
    # initialize new list of clusters
    clusters = []
    for cluster_indices = prev_clusters # iterate over all previous clusters
        new_indices = []
        # get all the possible new values for the new index
        for (k₁,j₁) in cluster_indices
            for (k₂,j₂) = [(k₁+1,j₁), (k₁-1,j₁), (k₁,j₁+1), (k₁,j₁-1)]
                proposed_cluster = sort(vcat(cluster_indices, ((k₂,j₂))))
                if !(((k₂,j₂) in new_indices) || ((k₂,j₂) in cluster_indices) || proposed_cluster in clusters)
                    push!(clusters, proposed_cluster)
                    push!(new_indices, (k₂,j₂))
                end
            end
        end
    end
    return clusters
end

function get_all_indices(PEPO, p, β, twosite_op, spaces; levels_convention = "tree_depth", symmetry = nothing, verbosity = 2)
    previous_clusters = [[(0,0)]]
    for N = 2:p
        if verbosity >= 1
            println("N = $(N)")
        end
        clusters = get_nontrivial_terms(N; prev_clusters = previous_clusters)
        for cluster = clusters
            solve_cluster(cluster, PEPO, β, twosite_op, spaces; levels_convention = levels_convention, symmetry = symmetry, verbosity = verbosity)
        end
        previous_clusters = clusters
    end
    if verbosity >= 1
        for (key, tens) = PEPO
            println("key = $(key)")
            println("norm of tensor = $(norm(tens))")
            println("summary of tensor: $(summary(tens))")
        end
    end
return PEPO
end    

function clusterexpansion(p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : ℂ^10, symmetry = nothing, verbosity = 2)
    (p < 10) || error("Only cluster up until 9th order are implemented correctly")
    dim(spaces(0)) == 1 || error("The zeroth space should be of dimension 1")
    pspace = onesite_op.dom[1]
    PEPO₀ = init_PEPO(β, onesite_op)
    PEPO = get_all_indices(PEPO₀, p, β, twosite_op, spaces; levels_convention = levels_convention, symmetry = symmetry, verbosity = verbosity)
    return PEPO, get_PEPO(pspace, PEPO, spaces)
end