using Graphs

# Function to calculate the depth of each subtree
function calculate_subtree_depths(tree::Graph, root::Int)
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
function graph_to_tree_with_generation_labels(graph::Graph, root::Int)
    # Perform BFS to create a spanning tree
    tree = SimpleGraph(nv(graph))  # Initialize an empty graph with the same number of vertices
    visited = Set([root])

    queue = [root]
    parent = Dict{Int, Int}()  # Keep track of parents to avoid backtracking
    count = 0
    while !isempty(queue) && (count < 100)
        println("iteration - count = $count")
        count += 1
        current = popfirst!(queue)
        for neighbor in neighbors(graph, current)
            if !(neighbor in visited)
                add_edge!(tree, current, neighbor)
                # visited = Set([neighbor])
                visited = union(visited, Set([neighbor]))
                queue = push!(queue, neighbor)
                parent[neighbor] = current
            end
        end
    end

    # Calculate subtree depths
    subtree_depths = calculate_subtree_depths(tree, root)

    # Assign edge labels based on subtree depths
    edge_labels = Dict{Tuple{Int, Int}, Int}()
    for edge in edges(tree)
        u, v = src(edge), dst(edge)
        if parent[v] == u  # Edge points from parent to child
            edge_labels[(u, v)] = subtree_depths[v]
        elseif parent[u] == v  # Edge points from child to parent
            edge_labels[(v, u)] = subtree_depths[u]
        end
    end

    return tree, edge_labels
end

# Example Usage
g = SimpleGraph(10)  # Create a graph with 5 vertices
add_edge!(g, 1, 2)
add_edge!(g, 2, 3)
add_edge!(g, 2, 4)
add_edge!(g, 4, 5)
add_edge!(g, 1, 6)
add_edge!(g, 6, 7)
add_edge!(g, 7, 9)
add_edge!(g, 6, 8)
add_edge!(g, 8, 10)

tree, edge_labels = graph_to_tree_with_generation_labels(g, 1)

println("Tree edges: $(edges(tree))")
println("Edge labels: $edge_labels")
