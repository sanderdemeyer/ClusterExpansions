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
    bonds_sites = []
    bonds_indices = []
    for (i,ind₁) = enumerate(cluster)
        for (j,ind₂) = enumerate(cluster)
            if (j > i) && (distance(ind₁, ind₂) == 1)
                push!(bonds_sites, (ind₁, ind₂))
                push!(bonds_indices, (i, j))
            end
        end
    end
    return bonds_sites, bonds_indices
end

function is_horizontal(bond)
    return (abs(bond[1][2]-bond[2][2]) == 1)*1
end

function get_central_site(cluster)
    bonds = get_bonds(cluster)
    findindex
    corner = bonds[findindex()]
end

function get_levels_line(n)
    return Int.([(n+1)/2 - abs((n+1)/2 - i) for i = 1:n])
end

function get_loops(cluster)
    _, bonds_indices = get_bonds(cluster)
    g = SimpleGraph(Graphs.SimpleEdge.(bonds_indices))
    longest_path = find_longest_path(g)
    longest_cycle = find_longest_cycle(g)
    contains_cycle = (longest_cycle.lower_bound > 2)
    if contains_cycle
        @warn "Fuck"
    end

    return cycle_basis(g)
end

function contains_loops(cluster)
    return (length(get_loops(cluster)) != 0)
end

function get_coordination_number(cluster)
    # Get the coordination number of each site of the cluster
    _, bonds_indices = get_bonds(cluster)
    return [sum(i .∈ bonds_indices) for i = 1:length(cluster)]
end

function get_branch(site1, site2, bonds)
    branch = [site1, site2]
    edge_bond = site2
    prev_bond = site1
    surroundings = [setdiff(bond, edge_bond)[1] for bond = bonds if ((edge_bond ∈ bond) && !(prev_bond ∈ bond))]
    while length(surroundings) != 0
        if length(surroundings) > 1
            @error "Multiple branches - currently not implemented"
        end
        push!(branch, surroundings[1])
        prev_bond = site2
        edge_bond = surroundings[1]
        surroundings = [setdiff(bond, edge_bond)[1] for bond = bonds if ((edge_bond ∈ bond) && !(prev_bond ∈ bond))]
    end
    return branch
end

function get_levels(cluster)
    # Find the levels of all the bonds of the cluster (in the Type A construction?)
    _, bonds_indices = get_bonds(cluster)
    g = SimpleGraph(Graphs.SimpleEdge.(bonds_indices))
    longest_path = find_longest_path(g, log_level = 0)
    longest_cycle = find_longest_cycle(g, log_level = 0)

    if (longest_cycle.lower_bound > 2)
        @warn "Cycles not implemented"
    end

    lp = longest_path.longest_path
    n = length(lp)-1
    levels_line = get_levels_line(n)
    coo = get_coordination_number(cluster)

    levels_dict = Dict()
    for i = 1:n
        indices = [lp[i],lp[i+1]]
        levels_dict[Tuple(sort(indices))] = levels_line[i]
    end
    
    for (ind,site) in enumerate(lp)
        if coo[site] > 2
            start_branches = [setdiff(bond, site)[1] for bond = bonds_indices if ((site ∈ bond) && !(lp[ind-1] ∈ bond) && !(lp[ind+1] ∈ bond))]
            for start_branch = start_branches
                branch = get_branch(site, start_branch, bonds_indices)
                n = length(branch)
                for i = 1:n-1
                    indices = [branch[i], branch[i+1]]
                    levels_dict[Tuple(sort(indices))] = n-i
                end
            end
        end
    end
    return [levels_dict[bond] for bond = bonds_indices]
end

function get_direction(site₁, site₂)
    dir = site₂ - site₁
    if dir == (0, 1) # to do - other stuff
        return (2, 4)
    end
end

function get_levels_sites(cluster)
    bonds_sites, bonds_indices = get_bonds(cluster)
    levels = get_levels(cluster)
    levels_sites = fill((0,0,0,0), length(cluster))

    for (i,(bond_s, bond_i)) = enumerate(zip(bonds_sites,bonds_indices))
        dir = get_direction(bond_s[1], bond_s[2])
        levels_sites[bond_i[1]][dir[1]] = levels[i]
        levels_sites[bond_i[2]][dir[2]] = levels[i]
    end
    return levels_sites
end

# Some Manual Test
cluster = sort([(-2, 0), (-1, 0), (0, -1), (0, 0), (1, 0), (2, 0)])
cluster = sort([(-2, 0), (-1, 0), (0, -2), (0, -1), (0, 0), (1, 0), (2, 0)])
levels_indices = get_levels(cluster)
println(levels_indices)