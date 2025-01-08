function solve_cluster(c, PEPO, β, twosite_op; levels_convention = "initial")
    cluster = Cluster(c; levels_convention = levels_convention)
    exp_H = exponentiate_hamiltonian(twosite_op, cluster, β)
    residual = contract_PEPO(cluster, PEPO)
    RHS = exp_H - residual
    @assert !(any(isnan.(convert(Array,RHS[][:])))) "RHS contains elements that are NaN"
    # levels_sites, m = get_levels_sites(cluster)

    if cluster.m >= 1
        loops = true
        if loops
            levels_to_update = [(0, -1, -1, 0), (0, 0, -1, -1), (-1, 0, 0, -1), (-1, -1, 0, 0)]
            solution, errors = solve_4_loop(RHS; α = 10)
            merge!(PEPO, Dict(zip(levels_to_update, solution)))
            return
        else
            levels_sites = [(0, 1, 0, 0), (0, 0, -2, 1), (-2, 0, 0, 1), (0, 1, 0, 0)]
        end
    end

    sites_to_update = [i for (i,levels) = enumerate(cluster.levels_sites) if !(levels ∈ keys(PEPO))]
    length(sites_to_update) == 0 && return
    A = get_A(cluster, PEPO, sites_to_update)

    if length(sites_to_update) == 2
        dir = (c[sites_to_update[2]][1] - c[sites_to_update[1]][1], c[sites_to_update[2]][2] - c[sites_to_update[1]][2])
        conjugated = get_conjugated(dir)
    elseif length(sites_to_update) == 1
        dir = 0
        conjugated = [Bool[0, 0, 1, 1]]
    else
        error("Something went terribly wrong, sites_to_update = $(sites_to_update)")
    end
    levels_to_update = cluster.levels_sites[sites_to_update]
    solution = solve_index(A, exp_H-residual, conjugated, sites_to_update, levels_to_update, get_direction(dir), cluster.N; spaces = i -> ℂ^(2^(2*i)))
    merge!(PEPO, Dict(zip(levels_to_update, solution)))
end

function get_nontrivial_terms(N; prev_clusters = [[(0,0)]])
    if N == 1
        return [[(0,0)]]
    end
    
    # prev_clusters = get_nontrivial_terms(N-1) # get all the clusters of size N-1

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

function get_all_indices(PEPO, p, β, twosite_op; levels_convention = "initial")
    previous_clusters = [[(0,0)]]
    for N = 2:p
        println("N = $(N)")
        clusters = get_nontrivial_terms(N; prev_clusters = previous_clusters)
        for cluster = clusters
            solve_cluster(cluster, PEPO, β, twosite_op; levels_convention = levels_convention)
        end
        for (key, tens) = PEPO
            println("key = $(key)")
            println("norm of tensor = $(norm(tens))")
        end
        previous_clusters = clusters
    end
    return PEPO
end    

function clusterexpansion(p, β, twosite_op, onesite_op; levels_convention = "initial")
    (p < 10) || error("Only cluster up until 9th order are implemented correctly")
    pspace = onesite_op.dom[1]
    PEPO₀ = init_PEPO(onesite_op)
    PEPO = get_all_indices(PEPO₀, p, β, twosite_op; levels_convention = levels_convention)
    return get_PEPO(pspace, PEPO)
end