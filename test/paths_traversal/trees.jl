using Graphs
using ClusterExpansions: get_bonds, get_longest_cycle, find_longest_path

cluster = [(0, 0), (0, 1), (0, -1), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (4, 3), (3, -1), (3, -2), (4, -2), (2, -2)]
# cluster = [(0, 1), (0, 0), (0, -1), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (4, 3), (3, -1), (3, -2), (4, -2), (2, -2)]
loop_4 = [(0, 0), (0, 1), (1, 1), (1, 0)]
loop_6 = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]

cluster = loop_4

_, bonds_indices = get_bonds(cluster)
bonds_rev = [(j,i) for (i,j) = bonds_indices]
bonds_bi = unique(vcat(bonds_indices, bonds_rev))
# g = SimpleGraph(Graphs.SimpleEdge.(bonds_indices))
g = SimpleDiGraph(Graphs.SimpleEdge.(bonds_indices))
g_dir = SimpleDiGraph(Graphs.SimpleEdge.(bonds_bi))
println("done")

# (mstt, weightst) = kruskal_mst(g, distmx =weights(g); minimize = true)
# (mstf, weightsf) = kruskal_mst(g; minimize = false)
N = length(cluster)
longest_path_graph, n = find_longest_path(g_dir, N)
longest_path = [Tuple(sort([l.src, l.dst])) for l = longest_path_graph]
longest_cycle_base = cycle_basis(g_dir)
longest_cycle = get_longest_cycle(cluster)