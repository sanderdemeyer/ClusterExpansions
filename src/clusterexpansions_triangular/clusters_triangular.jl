struct TriangularCluster
    N::Int
    cluster
    bonds_sites
    bonds_indices
    diag_bonds_sites
    diag_bonds_indices
    levels_sites
    m
    n
end

function TriangularCluster(lattice::Triangular, cluster; levels_convention = "tree_depth", symmetry = nothing)
    @assert levels_convention == "tree_depth" "Only 'tree_depth' levels convention is implemented for Triangular lattices"
    N = length(cluster)
    bonds_sites, bonds_indices = get_bonds(lattice, cluster)
    diag_bonds_sites, diag_bonds_indices = get_diag_bonds(lattice, cluster)

    g = SimpleGraph(Graphs.SimpleEdge.(bonds_indices))
    g_dir = get_directed_graph(bonds_indices)

    longest_path, n = get_longest_path(g_dir, N)
    # longest_cycle, m = get_longest_cycle(g_dir)
    cycles = cycle_basis(g_dir)
    m = length(cycles)

    if m >= 1
        levels_sites = nothing
        # if !isnothing(symmetry)
        #     permutation = vcat(cycles[1], setdiff(1:N, cycles[1]))
        #     permute!(cluster, permutation)
        #     bonds_sites, bonds_indices = get_bonds(lattice, cluster)
        #     g = SimpleGraph(Graphs.SimpleEdge.(bonds_indices))
        #     g_dir = get_directed_graph(bonds_indices)

        #     longest_path, n = get_longest_path(g_dir, N)
        #     cycles = cycle_basis(g_dir)
        #     m = length(cycles)
        #     @assert m >= 1
        # end
        coo = get_coordination_number(bonds_indices, N)
        levels = get_tree_depths(g, bonds_indices, vcat(cycles...))
        levels = update_levels_loops(lattice, levels, cycles, m, bonds_indices, cluster)
        # middle_cycle = nothing
        # if m >= 3
        #     for (ic, cycle) = enumerate(cycles)
        #         if sum([check_connectedness(cycle, c) for c = cycles]) == 2
        #             middle_cycle = ic
        #             break
        #         end
        #     end
        # end
        # for (i,(u,v)) = enumerate(bonds_indices)
        #     if (m >= 3) && (u ∈ cycles[middle_cycle] && v ∈ cycles[middle_cycle]) # ladder
        #         isedge = false
        #         for ic = setdiff(1:m, middle_cycle)
        #             if u ∈ cycles[ic] && v ∈ cycles[ic]
        #                 isedge = true
        #                 break
        #             end
        #         end
        #         levels[i] = -1 - isedge
        #     elseif u ∈ cycles[1] && v ∈ cycles[1] # all other cycles
        #         levels[i] = -1
        #     end
        # end
        levels_sites = get_levels_sites(lattice, bonds_sites, bonds_indices, levels, N)
    else
        coo = get_coordination_number(bonds_indices, N)
        levels = get_tree_depths(g, bonds_indices)
        levels_sites = get_levels_sites(lattice, bonds_sites, bonds_indices, levels, N)

    end
    c = TriangularCluster(N, cluster, bonds_sites, bonds_indices, diag_bonds_sites, diag_bonds_indices, levels_sites, m, n)
    return c
end

function isdiagonal(lattice, site₁, site₂)
    return (distance(lattice, site₁, site₂) == 3)
end

function distance(::Triangular, ind₁, ind₂)
    c1 = ind₁[1] - ind₂[1]
    c2 = ind₁[2] - ind₂[2]
    return c1^2 + c1 * c2 + c2^2
end

function get_direction(lattice::Triangular, site₁, site₂) # should not have to be restricted to Triangular lattices
    dir = (site₂[1] - site₁[1], site₂[2] - site₁[2])
    return get_direction(lattice, dir)
end

function get_direction(lattice::Triangular, dir::Int64) # should not have to be restricted to Triangular lattices
    return 0
end

function get_direction(lattice::Triangular, dir::Tuple{Int64, Int64})
    if dir == (0, 1)
        return (3, 6)
    elseif dir == (0, -1)
        return (6, 3)
    elseif dir == (-1, 1)
        return (2, 5)
    elseif dir == (1, -1)
        return (5, 2)
    elseif dir == (1, 0)
        return (4, 1)
    elseif dir == (-1, 0)
        return (1, 4)
    end
end

function get_bonds(lattice::Lattice, cluster)
    bonds_sites = Vector{Tuple{Tuple{Int, Int}, Tuple{Int, Int}}}()
    bonds_indices = Vector{Tuple{Int, Int}}()
    for (i, ind₁) in enumerate(cluster)
        for (j, ind₂) in enumerate(cluster)
            if (j > i) && (distance(lattice, ind₁, ind₂) == 1)
                push!(bonds_sites, (ind₁, ind₂))
                push!(bonds_indices, (i, j))
            end
        end
    end
    return bonds_sites, bonds_indices
end

function get_diag_bonds(lattice::Lattice, cluster)
    diag_bonds_sites = Vector{Tuple{Tuple{Int, Int}, Tuple{Int, Int}}}()
    diag_bonds_indices = Vector{Tuple{Int, Int}}()
    for (i, ind₁) in enumerate(cluster)
        for (j, ind₂) in enumerate(cluster)
            if (j > i) && isdiagonal(lattice, ind₁, ind₂)
                push!(diag_bonds_sites, (ind₁, ind₂))
                push!(diag_bonds_indices, (i, j))
            end
        end
    end
    return diag_bonds_sites, diag_bonds_indices
end

function get_levels_sites(lattice::Triangular, bonds_sites, bonds_indices, levels, N)
    levels_sites = fill(0, N, 6)

    for (i, (bond_s, bond_i)) in enumerate(zip(bonds_sites, bonds_indices))
        dir = get_direction(lattice, bond_s[1], bond_s[2])
        levels_sites[bond_i[1], dir[1]] = levels[i]
        levels_sites[bond_i[2], dir[2]] = levels[i]
    end

    levels_sites = [Tuple(levels_sites[i, :]) for i in 1:N]
    return levels_sites
end

# Pretty printing of a cluster
function Base.show(io::IO, z::TriangularCluster)
    print(io, "Cluster =   ")
    coords = [(2 * y + x, -2 * x) for (x, y) in z.cluster]
    # Determine the size of the graph
    min_x, max_x = minimum(x for (x, _) in coords), maximum(x for (x, _) in coords)
    min_y, max_y = minimum(y for (_, y) in coords), maximum(y for (_, y) in coords)

    offset = 6
    offset_cluster = 6
    max_grid = 20

    # grid = fill("  ", max_y - min_y + 1, offset + max_x - min_x + 1)
    grid = fill("  ", max_y - min_y + 1, max_grid)
    # Plot points
    for (x, y) in coords
        if y == max_y
            grid[max_y - y + 1, offset - offset_cluster + x - min_x + 1] = "● "
        else
            grid[max_y - y + 1, offset + x - min_x + 1] = "● "
        end
    end
    if z.m == 1
        grid[1, max_grid] = "has $(z.N) sites and $(z.m) loop"
    else
        grid[1, max_grid] = "has $(z.N) sites and $(z.m) loops"
    end
    # Print the graph in one call to avoid newline issues
    print(io, join([join(row) for row in eachrow(grid)], "\n"))
    return print(io, "\n")  # Keep the text on the same line but move to next before graph
end
