using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ

function solve_cluster(c, PEPO, β, twosite_op; levels_convention = "initial")
    cluster = Cluster(c; levels_convention = levels_convention)

    if cluster.m >= 1
        loops = true
        if loops
            levels_to_update = [(0, -1, -1, 0), (0, 0, -1, -1), (-1, 0, 0, -1), (-1, -1, 0, 0)]
            # solution, errors = solve_4_loop(RHS; α = 10)
            solution = fill(0, length(levels_to_update))
            merge!(PEPO, Dict(zip(levels_to_update, solution)))
            return
        else
            levels_sites = [(0, 1, 0, 0), (0, 0, -2, 1), (-2, 0, 0, 1), (0, 1, 0, 0)]
        end
    end

    sites_to_update = [i for (i,levels) = enumerate(cluster.levels_sites) if !(levels ∈ keys(PEPO))]
    length(sites_to_update) == 0 && return

    levels_to_update = cluster.levels_sites[sites_to_update]
    solution = fill(0, length(levels_to_update))
    merge!(PEPO, Dict(zip(levels_to_update, solution)))
end

function get_all_indices(PEPO, p, β, twosite_op; levels_convention = "initial")
    prev_clusters = [[(0,0)]]
    for N = 2:p
        println("N = $(N)")
        clusters = get_nontrivial_terms(N; prev_clusters = prev_clusters)
        for cluster = clusters
            solve_cluster(cluster, PEPO, β, twosite_op; levels_convention = levels_convention)
        end
        prev_clusters = clusters
    end
    return PEPO
end    

function clusterexpansion(p, β, twosite_op, onesite_op; levels_convention = "initial")
    (p < 10) || error("Only cluster up until 9th order are implemented correctly")
    pspace = onesite_op.dom[1]
    PEPO₀ = Dict((0,0,0,0) => 0)
    PEPO = get_all_indices(PEPO₀, p, β, twosite_op; levels_convention = levels_convention)
    return PEPO
end

p = 4
β = 1e-4

levels_convention = "tree_depth"
J = 1.0
g = 0.0
N1, N2 = (1,1)

twosite_op = rmul!(σᶻᶻ(), -1.0)
onesite_op = rmul!(σˣ(), g * -J)

PEPO = clusterexpansion(p, β, twosite_op, onesite_op; levels_convention = levels_convention)