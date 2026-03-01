# Function to calculate the depth of each subtree
function get_subtree_depths(tree::Graph, root::Int)
    subtree_depths = Dict{Int, Int}()

    function dfs(node, parent)
        max_depth = 0
        for neighbor in neighbors(tree, node)
            if neighbor != parent
                child_depth = dfs(neighbor, node)
                max_depth = max(max_depth, child_depth)
            end
        end
        subtree_depths[node] = max_depth + 1  # Depth = max child depth + 1
        return subtree_depths[node]
    end

    dfs(root, -1)
    return subtree_depths
end

# Function to convert a graph to a tree and label edges based on subtree depth
function get_tree_depths(graph::Graph, bonds_indices::Vector{Tuple{Int64, Int64}}, root::Int)
    # Perform BFS to create a spanning tree
    tree = SimpleGraph(nv(graph))  # Initialize an empty graph with the same number of vertices
    visited = [root]

    queue = [root]
    parent = Dict(root => -1) # Keep track of parents to avoid backtracking
    while !isempty(queue)
        current = popfirst!(queue)
        for neighbor in neighbors(graph, current)
            if !(neighbor in visited)
                add_edge!(tree, current, neighbor)
                push!(visited, neighbor)
                queue = push!(queue, neighbor)
                parent[neighbor] = current
            end
        end
    end

    # Calculate subtree depths
    subtree_depths = get_subtree_depths(tree, root)

    levels = Vector{Int}()
    for edge in bonds_indices
        u, v = edge
        if parent[v] == u  # Edge points from parent to child
            push!(levels, subtree_depths[v])
        else # if parent[u] == v  # Edge points from child to parent -- include cycles
            push!(levels, subtree_depths[u])
        end
    end

    return levels, maximum(levels)
end

function get_tree_depths(graph::Graph, bonds_indices::Vector{Tuple{Int64, Int64}})
    levels = Vector{Int}()
    max_depth = Inf
    for site in 1:nv(graph)
        new_levels, depth = get_tree_depths(graph, bonds_indices, site)
        if depth < max_depth
            levels = new_levels
            max_depth = depth
        elseif depth == max_depth
            levels = min.(levels, new_levels)
        end
    end
    return levels
end

function get_tree_depths(graph::Graph, bonds_indices::Vector{Tuple{Int64, Int64}}, roots::Vector{Int})
    return min.([get_tree_depths(graph, bonds_indices, root)[1] for root in roots]...)
end
