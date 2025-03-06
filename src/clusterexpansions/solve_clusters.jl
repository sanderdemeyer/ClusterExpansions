function get_update_dir(c, sites_to_update)
    if length(sites_to_update) == 2
        dir = (c[sites_to_update[2]][1] - c[sites_to_update[1]][1], c[sites_to_update[2]][2] - c[sites_to_update[1]][2])
        conjugated = get_conjugated(dir)
    elseif length(sites_to_update) == 1
        dir = 0
        conjugated = [Bool[0, 0, 1, 1]]
    else
        error("Number of sites to update $(length(sites_to_update)) not implemented")
    end
    return dir, conjugated
end

function solve_cluster(T, cluster, PEPO, β, twosite_op, onesite_op, spaces; symmetry = nothing, verbosity = 2, solving_loops = true)
    if verbosity >= 2
        println(cluster)
    end
    sites_to_update = [i for (i,levels) = enumerate(cluster.levels_sites) if !(levels ∈ keys(PEPO))]
    (length(sites_to_update) == 0) && return spaces
    if !solving_loops
        @warn "Not solving loops"
        return spaces
    end
    levels_to_update = cluster.levels_sites[sites_to_update]


    exp_H = exponentiate_hamiltonian(twosite_op, onesite_op, cluster, β)
    residual = contract_PEPO(T, cluster, PEPO, spaces)
    # RHS = (exp_H/norm(exp_H) - residual/norm(exp_H))*norm(exp_H)
    RHS = exp_H - residual

    # if cluster.N == 6
    #     RHS2 = exp_H - residual
    #     exp_H_rot = permute(exp_H, ((4,1,2,3),(8,5,6,7)))
    #     residual_rot = permute(residual, ((4,1,2,3),(8,5,6,7)))
    #     RHS_rot = permute(RHS, ((4,1,2,3),(8,5,6,7)))
    #     RHS2_rot = permute(RHS2, ((4,1,2,3),(8,5,6,7)))

    #     println("exp_H norm is $(norm(exp_H))")
    #     println("residual norm is $(norm(residual))")
    #     println("RHS norm is $(norm(RHS))")
    #     println("RHS2 norm is $(norm(RHS2))")

    #     println("Rot invariance of exp_H: $(norm(exp_H - exp_H_rot) / norm(exp_H))")
    #     println("Rot invariance of residual: $(norm(residual - residual_rot) / norm(residual))")
    #     println("Rot invariance of RHS: $(norm(RHS - RHS_rot) / norm(RHS))")
    #     println("Rot invariance of RHS2: $(norm(RHS2 - RHS2_rot) / norm(RHS2))")
    # end

    @assert !(any(isnan.(convert(Array,RHS[][:])))) "RHS contains elements that are NaN"
    (norm(RHS) < eps(real(T))*1e5) && return spaces

    if length(sites_to_update) == 4
        solutions, _, _ = solve_4_loop_optim(RHS, spaces(-1), levels_to_update; verbosity = verbosity, symmetry = symmetry)
        # solutions, _, spaces = solve_4_loop_SVD(RHS, spaces(-1), levels_to_update; verbosity = verbosity, symmetry = symmetry)
    elseif length(sites_to_update) ∈ [1, 2]
        A = get_A(T, cluster, PEPO, sites_to_update)
        dir, conjugated = get_update_dir(cluster.cluster, sites_to_update)
        solutions = solve_index(T, A, exp_H-residual, conjugated, sites_to_update, levels_to_update, get_direction(dir), cluster.N, spaces; verbosity = verbosity)
    else
        @error "Number of sites to update = $(length(sites_to_update))"
    end
    levels_to_update, solutions = symmetrize(symmetry, levels_to_update, solutions)
    merge!(PEPO, Dict(zip(levels_to_update, solutions)))
    return spaces
end

function get_nontrivial_terms(N; prev_clusters = [[(0,0)]])
    if N == 1
        return [[(0,0)]]
    end
    
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

function get_all_indices(T, PEPO, p, β, twosite_op, onesite_op, spaces; levels_convention = "tree_depth", symmetry = nothing, verbosity = 2)
    previous_clusters = [[(0,0)]]
    for N = 2:p
        if verbosity >= 1
            println("N = $(N)")
        end
        cluster_indices = get_nontrivial_terms(N; prev_clusters = previous_clusters)
        clusters = [Cluster(c; levels_convention = levels_convention, symmetry = symmetry) for c in cluster_indices]
        sort!(clusters, by = p -> (p.m, p.n)) # Sort the clusters such that the loops and higher levels are solved last
        for cluster = clusters
            spaces = solve_cluster(T, cluster, PEPO, β, twosite_op, onesite_op, spaces; symmetry = symmetry, verbosity = verbosity)
        end
        previous_clusters = cluster_indices
        if verbosity >= 2
            for (key, tens) = PEPO
                println("key = $(key)")
                println("Maximum is $(maximum(abs.(tens[]))), norm is $(norm(tens))")
                println("Summary = $(summary(tens))")
            end
        end    
    end
    # if verbosity >= 1
    #     for (key, tens) = PEPO
    #         println("key = $(key)")
    #         println("Maximum is $(maximum(tens[])), norm is $(norm(tens))")
    #         println("Summary = $(summary(tens))")
    #     end
    # end
return PEPO
end    

function clusterexpansion(T, p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : ℂ^10, symmetry = nothing, verbosity = 2)
    (p < 10) || error("Only cluster up until 9th order are implemented correctly")
    dim(spaces(0)) == 1 || error("The zeroth space should be of dimension 1")
    pspace = domain(onesite_op)[1]
    PEPO₀ = init_PEPO(T, β, onesite_op)
    PEPO = get_all_indices(T, PEPO₀, p, β, twosite_op, onesite_op, spaces; levels_convention = levels_convention, symmetry = symmetry, verbosity = verbosity)
    return PEPO, get_PEPO(T, pspace, PEPO, spaces)
end

function clusterexpansion(p, β, twosite_op, onesite_op; kwargs...)
    return clusterexpansion(Complex{Float64}, p, β, twosite_op, onesite_op; kwargs...)
end