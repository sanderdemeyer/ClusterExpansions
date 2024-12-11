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
        error("cycles not implemented")
    end

    return cycle_basis(g)
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
            error("Multiple branches - currently not implemented")
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
    g = SimpleGraph(Graphs.SimpleEdge.(bonds_indices))

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

function get_direction(dir::Int64)
    return 0
end

function get_direction(dir::Tuple{Int64,Int64})
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

function get_graph(cluster)
    bonds_sites, bonds_indices = get_bonds(cluster)
    graph = fill(0, length(cluster), 4)
    for (bond_s, bond_i) = zip(bonds_sites,bonds_indices)
        dir = get_direction(bond_s[1], bond_s[2])
        graph[bond_i[1], dir[1]] = bond_i[2]
        graph[bond_i[2], dir[2]] = bond_i[1]
    end
    return graph
end

function get_contraction_order(cluster, sites_to_update)
    updates = length(sites_to_update)
    fixed_tensors = length(cluster) - updates 
    graph = get_graph(cluster)
    contraction_indices = fill(0, fixed_tensors, 6)
    for i = 1:fixed_tensors
        contraction_indices[i,1] = -i
        contraction_indices[i,2] = -(fixed_tensors+i)
    end
    included_sites = setdiff(1:length(cluster)[1], sites_to_update)
    conv = i -> i-sum([j ∈ sites_to_update for j = 1:i])
    m = 1
    for i = included_sites
        for j = 1:4
            if !(graph[i,j] ∈ (0, sites_to_update...)) && (contraction_indices[conv(i),j+2] == 0)
                ind = findall(x -> x == i, graph[graph[i,j],:])[1]
                contraction_indices[conv(i),j+2] = m 
                contraction_indices[conv(graph[i,j]),ind+2] = m
                m += 1
            end
        end
    end
    number_of_bonds = copy(m)-1
    conjugated = Bool[]
    for i = included_sites
        for j = 1:4
            if (graph[i,j] == 0)
                contraction_indices[conv(i),j+2] = m
                push!(conjugated, j > 2)
                m += 1
            end
        end
    end
    open_conjugated = Bool[]
    open_indices = Int[]
    spaces = fill((0,0), updates, 5-updates)
    m = 2*fixed_tensors+1
    opposite = dir -> 4 - dir + 2*(dir ∈ (2,4))
    for (enum,i) = enumerate(sites_to_update)
        kⱼ = 0
        for j = 1:4
            if graph[i,j] ∉ sites_to_update
                kⱼ += 1
                if graph[i,j] == 0
                    push!(open_conjugated, j > 2)
                    push!(open_indices, -m)
                else
                    ind = findall(x -> x == i, graph[graph[i,j],:])[1]
                    contraction_indices[conv(graph[i,j]),ind+2] = -m
                    spaces[enum, kⱼ] = (graph[i,j],opposite(j))
                end
                m += 1
            end
        end
    end
    return contraction_indices, conjugated, open_conjugated, open_indices, spaces, number_of_bonds
end

function get_levels_sites(cluster)
    bonds_sites, bonds_indices = get_bonds(cluster)
    levels = get_levels(cluster)
    (levels === nothing) && (return nothing)
    levels_sites = fill(0, length(cluster), 4)

    for (i,(bond_s, bond_i)) = enumerate(zip(bonds_sites,bonds_indices))
        dir = get_direction(bond_s[1], bond_s[2])
        levels_sites[bond_i[1], dir[1]] = levels[i]
        levels_sites[bond_i[2], dir[2]] = levels[i]
    end

    levels_sites = [Tuple(levels_sites[i,:]) for i = 1:length(cluster)]
    return levels_sites
end

function init_PEPO(pspace, trivspace)
    return Dict((0,0,0,0) => TensorMap([1.0 0.0; 0.0 1.0], pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace'))   
end

function init_PEPO()
    return init_PEPO(ℂ^2, ℂ^1)
end

function get_size_level(highest)
    return sum([2^(2*i) for i = 0:highest])
end

function get_location_PEPO(ind)
    if ind == 0
        return 1
    end
    h = get_size_level(ind-1)
    return h+1:h+2^(2*ind)
end

function get_PEPO(pspace, PEPO)
    highest = [get_size_level(maximum([i[dir] for i = keys(PEPO)])) for dir = 1:4]
    conjugated = Bool[0, 0, 1, 1]
    O = TensorMap(pspace ⊗ pspace', prod([conj ? (ℂ^h)' : ℂ^h for (conj,h) = zip(conjugated,highest)]))

    for (key, tens) = PEPO
        places = [get_location_PEPO(ind) for ind = key]
        O[][:,:,places[1],places[2],places[3],places[4]] = tens[]
    end

    return O
end

function get_conjugated(dir)
    if dir == (-1, 0) || dir == (0, 1)
        return [Bool[0, 1, 1],Bool[0, 0, 1]]
    else
        return [Bool[0, 0, 1], Bool[0, 1, 1]]
    end
end

function solve_cluster(cluster, PEPO, β)
    println("in solve cluster - cluster = $(cluster)")
    exp_H = exponentiate_hamiltonian(cluster, β)

    levels_sites = get_levels_sites(cluster)
    (levels_sites === nothing) && (return PEPO)

    sites_to_update = [i for (i,levels) = enumerate(levels_sites) if !(levels ∈ PEPO.keys)]
    length(sites_to_update) == 0 && return
    contraction_indices, conjugated, open_conjugated, open_indices, spaces, number_of_bonds = get_contraction_order(cluster, sites_to_update)

    A = get_A(PEPO, levels_sites, number_of_bonds, sites_to_update, conjugated, open_conjugated, open_indices, contraction_indices)

    if length(sites_to_update) == 2
        dir = (cluster[sites_to_update[2]][1] - cluster[sites_to_update[1]][1], cluster[sites_to_update[2]][2] - cluster[sites_to_update[1]][2])
        conjugated = get_conjugated(dir)
    elseif length(sites_to_update) == 1
        dir = 0
        conjugated = [Bool[0, 0, 1, 1]]
    else
        error("Something went terribly wrong")
    end
    levels_to_update = levels_sites[sites_to_update]
    solution = solve_index(length(sites_to_update), A, exp_H, conjugated, sites_to_update, levels_to_update, get_direction(dir), length(cluster); spaces = i -> ℂ^(2^(2*i)))
    merge!(PEPO, Dict(zip(levels_to_update, solution)))
end

function get_all_indices(p, β)
    PEPO = init_PEPO()
    for N = 2:p
        println("N = $(N)")
        clusters = get_nontrivial_terms(N)
        for cluster = clusters
            solve_cluster(cluster, PEPO, β)
        end
    end
    return PEPO
end    

β = 1
result = get_all_indices(3, β)
O = get_PEPO(ℂ^2, result)
println("done")

# x0 = TensorMap(randn, ℂ^2 ⊗ (ℂ^2)' ← ℂ^(7) ⊗ ℂ^(6) ⊗ (ℂ^(8))' ⊗ (ℂ^(4))')
# summary(x0)

# sol = TensorMap(randn, ℂ^2 ⊗ (ℂ^2)' ← ℂ^(4) ⊗ ℂ^(1) ⊗ (ℂ^(4))' ⊗ (ℂ^(1))')

# x0[][:,:,1:4,1,5:8,1] = sol[];
