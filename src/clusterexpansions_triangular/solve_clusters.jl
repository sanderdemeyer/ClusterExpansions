function get_nontrivial_terms(lattice::Triangular, N; prev_clusters = [[(0,0)]])
    if N == 1
        return [[(0,0)]]
    end
    # initialize new list of clusters
    clusters = []
    for cluster_indices = prev_clusters # iterate over all previous clusters
        new_indices = []
        # get all the possible new values for the new index
        for (k₁,j₁) in cluster_indices
            for (k₂,j₂) = [(k₁+1,j₁), (k₁-1,j₁), (k₁,j₁+1), (k₁,j₁-1), (k₁+1,j₁-1), (k₁-1,j₁+1)]
                proposed_cluster = sort(vcat(cluster_indices, ((k₂,j₂))))
                if !(((k₂,j₂) in new_indices) || ((k₂,j₂) in cluster_indices) || proposed_cluster in clusters)
                    if (proposed_cluster ∉ clusters) && (move(proposed_cluster, -1, 0) ∉ clusters) && (move(proposed_cluster, 1, 0) ∉ clusters) && (move(proposed_cluster, 0, 1) ∉ clusters) && (move(proposed_cluster, 0, -1) ∉ clusters) && (move(proposed_cluster, 1, -1) ∉ clusters) && (move(proposed_cluster, -1, 1) ∉ clusters)
                        push!(clusters, proposed_cluster)
                        push!(new_indices, (k₂,j₂))
                    end
                end
            end
        end
    end
    return clusters
end

function get_all_indices(lattice::Triangular, T, PEPO, p, β, twosite_op, onesite_op, spaces; nn_term = nothing, levels_convention = "tree_depth", symmetry = nothing, verbosity = 2, solving_loops = true, svd = true)
    previous_clusters = [[(0,0)]]
    for N = 2:p
        if verbosity >= 2
            @info "N = $(N)"
        end
        cluster_indices = get_nontrivial_terms(lattice, N; prev_clusters = previous_clusters)
        clusters = [TriangularCluster(lattice, c; levels_convention = levels_convention, symmetry = symmetry) for c in cluster_indices]
        # clusters = [TriangularCluster(c; levels_convention = levels_convention, symmetry = symmetry) for c in cluster_indices]
        for cluster = clusters
            println(cluster)
        end
        sort!(clusters, by = p -> (p.m, p.n)) # Sort the clusters such that the loops and higher levels are solved last
        # for cluster = clusters
        #     spaces = solve_cluster(T, cluster, PEPO, β, twosite_op, onesite_op, spaces; nn_term, symmetry, verbosity, solving_loops, svd)
        # end
        previous_clusters = cluster_indices
    end
    # if verbosity >= 2
    #     for (key, tens) = PEPO
    #         @info "key = $(key)"
    #         @info "Maximum is $(maximum(abs.(tens.data))), norm is $(norm(tens))"
    #         @info "Summary = $(summary(tens))"
    #     end
    # end    
    return PEPO
end    
