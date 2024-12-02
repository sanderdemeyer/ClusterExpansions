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

function distance(ind₁, ind₂)
    return abs(ind₁[1] - ind₂[1]) + abs(ind₁[2] - ind₂[2])
end

function get_bonds(cluster)
    bonds = []
    for (i,ind₁) = enumerate(cluster)
        for (j,ind₂) = enumerate(cluster)
            if (j > i) && (distance(ind₁, ind₂) == 1)
                push!(bonds, (ind₁, ind₂))
            end
        end
    end
    return bonds
end

function is_horizontal(bond)
    return (abs(bond[1][2]-bond[2][2]) == 1)*1
end

function get_central_site(cluster)
    bonds = get_bonds(cluster)
    println(bonds)
    println(is_horizontal.(bonds))
    findindex
    corner = bonds[findindex()]
end

function get_level(cluster, current_indices)
    return 0
end
