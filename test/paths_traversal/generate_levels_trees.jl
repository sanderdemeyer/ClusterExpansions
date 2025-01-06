using Graphs
using Test

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

correct_result = Dict(
    (1, 2) => 3,
    (1, 6) => 3,
    (2, 4) => 2,
    (6, 7) => 2,
    (6, 8) => 2,
    (2, 3) => 1,
    (4, 5) => 1,
    (7, 9) => 1,
    (8, 10) => 1
)

@test correct_result == edge_labels
