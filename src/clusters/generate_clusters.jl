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

function get_longest_path(g, p)
    longest_path = find_longest_path(g, 1, 0; log_level = 0, solver_mode = "lp+ip")
    n = length(longest_path.longest_path) - 1

    for first_vertex = 2:p
        longest_path_vertex = find_longest_path(g, first_vertex, 0; log_level = 0, solver_mode = "lp+ip")
        if length(longest_path_vertex.longest_path) - 1 > n
            longest_path = longest_path_vertex
            n = length(longest_path_vertex.longest_path) - 1
        end
    end
    return longest_path, n
end

function get_levels(cluster)
    # Find the levels of all the bonds of the cluster (in the Type A construction?)
    _, bonds_indices = get_bonds(cluster)
    # all_bonds = sort(vcat([(u, v) for (u, v) in bonds_indices], [(v, u) for (u, v) in bonds_indices]))
    all_bonds = bonds_indices
    g = SimpleGraph(Graphs.SimpleEdge.(all_bonds))

    longest_path, n = get_longest_path(g, length(cluster))

    longest_cycle = find_longest_cycle(g, log_level = 0)

    coo = get_coordination_number(cluster)

    if (longest_cycle.lower_bound > 2)
        if longest_cycle.lower_bound >= 6
            @warn "Cycles of size >= 6 not yet implemented"
            return nothing
        end
        @warn "Cycles not implemented"
        return nothing
        # branch_starts = [i for i = longest_cycle.longest_path if coo[i] > 2]
        # println("branch_starts = $(branch_starts)")

        # (length(branch_starts) == 2) && (return nothing)

        # branches = []
        # for site₁ = branch_starts
        #     site₂ = [i for i = 1:length(cluster) if ((Tuple(sort([site₁, i])) ∈ bonds_indices) && !(i ∈ longest_cycle.longest_path))]
        #     push!(branches, get_branch(site₁, site₂, bonds_indices)...)
        # end

        # println(branches)
        # return nothing
    end

    lp = longest_path.longest_path
    levels_line = get_levels_line(n)
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
    dir = (site₂[1] - site₁[1], site₂[2] - site₁[2])
    if dir == (-1, 0)
        return (1, 3)
    elseif dir == (0, 1)
        return (2, 4)
    elseif dir == (1, 0)
        return (3, 1)
    elseif dir == (0, -1)
        return (4, 2)
    end
end

function get_levels_sites(cluster)
    bonds_sites, bonds_indices = get_bonds(cluster)
    levels = get_levels(cluster)
    (levels === nothing) && (return nothing)
    levels_sites = fill(0, length(cluster), 4) # fill([0,0,0,0], length(cluster))
    contraction_indices = fill(0, length(cluster), 4)
    conjugated = Bool[]

    for (i,(bond_s, bond_i)) = enumerate(zip(bonds_sites,bonds_indices))
        dir = get_direction(bond_s[1], bond_s[2])
        levels_sites[bond_i[1], dir[1]] = levels[i]
        levels_sites[bond_i[2], dir[2]] = levels[i]
        contraction_indices[bond_i[1], dir[1]] = contraction_indices[bond_i[2], dir[2]] = i
    end
    n = length(bonds_sites)+1 
    for c = 1:length(cluster)
        for dir = 1:4
            if contraction_indices[c, dir] == 0
                contraction_indices[c, dir] = n
                n += 1
                push!(conjugated, dir > 2)
            end
        end
    end
    levels_sites = [Tuple(levels_sites[i,:]) for i = 1:length(cluster)]
    contraction_indices = [Tuple(contraction_indices[i,:]) for i = 1:length(cluster)]
    println("cluster = $(cluster)")
    println("contraction indices = $(contraction_indices)")
    println("conjugated = $(conjugated)")
    return levels_sites, contraction_indices, conjugated
end

function new_clusters(cluster, current_indices)
    levels_sites, contraction_indices, conjugated = get_levels_sites(cluster)
    (levels_sites === nothing) && (return (current_indices, 0))
    new_indices = [levels for levels = levels_sites if !(levels ∈ current_indices)]
    sites_to_update = [i for (i,levels) = enumerate(levels_sites) if !(levels ∈ current_indices)]
    
    push!(current_indices, new_indices...)
    return current_indices, sites_to_update, length(new_indices), contraction_indices, conjugated
end

function index_to_solve(clusters, current_indices)
    β = 1
    pspace = ℂ^2
    trivspace = ℂ^1
    PEPO = Dict((0,0,0,0) => TensorMap([1.0 0.0; 0.0 1.0], pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace ⊗ trivspace))   
    for c = clusters
        println("cluster = $(c)")
        current_indices, sites_to_update, nnew, contraction_indices, conjugated = new_clusters(c, current_indices)
        levels_sites, contraction_indices, conjugated = get_levels_sites(c)
        println("site to update = $(sites_to_update)")
        if nnew == 1
            @error "TBA"
        elseif nnew == 2
            F = get_other_tensors(cluster, PEPO, levels_sites, sites_to_update, contraction_indices, conjugated)
        elseif nnew > 2
            error("Too many new levels to implement")
        end
        println("New clusters = $(current_indices[end-nnew+1:end])")
        println("All current indices = $(current_indices)")
    end
    return current_indices
end

function get_all_indices(p)
    current_indices = Tuple{Int64, Int64, Int64, Int64}[]

    for N = 2:p
        clusters = get_nontrivial_terms(N)
        
        current_indices = index_to_solve(clusters, current_indices)
    end
    return current_indices
end    

# all_indices = get_all_indices(4)


# # Some Manual Test
# cluster = sort([(-2, 0), (-1, 0), (0, -1), (0, 0), (1, 0), (2, 0)])
# cluster = sort([(-2, 0), (-1, 0), (0, -2), (0, -1), (0, 0), (1, 0), (2, 0)])
# levels_indices = get_levels(cluster)
# println(levels_indices)

"""
Test loop clusters
# cluster = sort([(0, 0), (1, 0), (0, 1), (1, 1)])
# cluster = sort([(0, 0), (1, 0), (0, 1), (1, 1), (1, 2)])
# get_levels(cluster)
"""

all_indices = get_all_indices(2)