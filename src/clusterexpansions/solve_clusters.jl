function solve_cluster(c, PEPO, β, twosite_op)
    cluster = Cluster(c)
    exp_H = exponentiate_hamiltonian(twosite_op, cluster, β)
    residual = contract_PEPO(cluster, PEPO)
    RHS = exp_H - residual
    @assert !(any(isnan.(convert(Array,RHS[][:])))) "RHS contains elements that are NaN"
    # levels_sites, m = get_levels_sites(cluster)

    if cluster.m >= 1
        loops = true
        if loops
            levels_to_update = [(0, -1, -1, 0), (0, 0, -1, -1), (-1, 0, 0, -1), (-1, -1, 0, 0)]
            solution = solve_4_loop(exp_H; α = 10)
            merge!(PEPO, Dict(zip(levels_to_update, solution)))
            return
        else
            levels_sites = [(0, 1, 0, 0), (0, 0, -2, 1), (-2, 0, 0, 1), (0, 1, 0, 0)]
        end
    end

    sites_to_update = [i for (i,levels) = enumerate(cluster.levels_sites) if !(levels ∈ PEPO.keys)]
    length(sites_to_update) == 0 && return

    A = get_F(cluster, PEPO, sites_to_update)

    if length(sites_to_update) == 2
        dir = (c[sites_to_update[2]][1] - c[sites_to_update[1]][1], c[sites_to_update[2]][2] - c[sites_to_update[1]][2])
        conjugated = get_conjugated(dir)
    elseif length(sites_to_update) == 1
        dir = 0
        conjugated = [Bool[0, 0, 1, 1]]
    else
        println(cluster.levels_sites)
        error("Something went terribly wrong, sites_to_update = $(sites_to_update)")
    end
    levels_to_update = cluster.levels_sites[sites_to_update]
    solution = solve_index(A, exp_H-residual, conjugated, sites_to_update, levels_to_update, get_direction(dir), cluster.N; spaces = i -> ℂ^(2^(2*i)))
    merge!(PEPO, Dict(zip(levels_to_update, solution)))
end

function get_all_indices(PEPO, p, β, twosite_op)
    for N = 2:p
        println("N = $(N)")
        clusters = get_nontrivial_terms(N)
        for cluster = clusters
            solve_cluster(cluster, PEPO, β, twosite_op)
        end
    end
    # cluster = [(0,0),(1,0),(1,1),(0,1)]
    # solve_cluster(cluster, PEPO, β, twosite_op)
    return PEPO
end    

function clusterexpansion(p, β, twosite_op, onesite_op)
    pspace = onesite_op.dom[1]
    PEPO₀ = init_PEPO(onesite_op)
    PEPO = get_all_indices(PEPO₀, p, β, twosite_op)
    return get_PEPO(pspace, PEPO)
end