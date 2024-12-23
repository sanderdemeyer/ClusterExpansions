function get_nontrivial_terms(N)
    if N == 1
        return [[(0,0)]]
    end
    prev_clusters = get_nontrivial_terms(N-1) # get all the clusters of size N-1
    
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

# function get_levels(cluster)
#     # Find the levels of all the bonds of the cluster (in the Type A construction?)
#     _, bonds_indices = get_bonds(cluster)
#     g = SimpleGraph(Graphs.SimpleEdge.(bonds_indices))

#     longest_path, n = get_longest_path(cluster)
#     # longest_cycle = find_longest_cycle(g, log_level = 0)
#     longest_cycle, m = get_longest_cycle(cluster)
#     coo = get_coordination_number(cluster)
#     println("m = $(m)")
#     if m >= 1
#         println(longest_cycle)
#         return nothing, m
#         if longest_cycle.lower_bound >= 6
#             @warn "Cycles of size >= 6 not yet implemented"
#             return nothing
#         end
#         if length(cluster) >= 5
#             @warn "Cycles not implemented for N = $(length(cluster)) >= 5"
#         end
#         return nothing
#     end

#     lp = longest_path
# end

function get_graph(cluster)
    graph = fill(0, cluster.N, 4)
    for (bond_s, bond_i) = zip(cluster.bonds_sites,cluster.bonds_indices)
        dir = get_direction(bond_s[1], bond_s[2])
        graph[bond_i[1], dir[1]] = bond_i[2]
        graph[bond_i[2], dir[2]] = bond_i[1]
    end
    return graph
end