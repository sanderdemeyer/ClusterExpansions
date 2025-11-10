function get_update_dir(lattice, c, sites_to_update)
    if length(sites_to_update) == 2
        dir = (c[sites_to_update[2]][1] - c[sites_to_update[1]][1], c[sites_to_update[2]][2] - c[sites_to_update[1]][2])
        conjugated = get_conjugated(dir)
    elseif length(sites_to_update) == 1
        dir = 0
        conjugated = [Bool[0, 0, 0, 1, 1, 1]]
    else
        error("Number of sites to update $(length(sites_to_update)) not implemented")
    end
    return dir, conjugated
end

function solve_cluster(lattice, T, cluster, PEPO, β, twosite_op, onesite_op, spaces; nn_term = nothing, symmetry = nothing, verbosity = 2, solving_loops = true, svd = true)
    if verbosity >= 2
        println(cluster)
        println(cluster.cluster)
    end
    sites_to_update = [i for (i,levels) = enumerate(cluster.levels_sites) if !(levels ∈ keys(PEPO))]
    (length(sites_to_update) == 0) && return spaces
    if !solving_loops && cluster.m > 0
        @warn "Not solving loops"
        return spaces
    end
    levels_to_update = cluster.levels_sites[sites_to_update]

    exp_H = exponentiate_hamiltonian(twosite_op, onesite_op, cluster, β; nn_term)
    residual = contract_PEPO(lattice, T, cluster, PEPO, spaces)
    RHS = exp_H - residual

    @assert !(any(isnan.(RHS.data))) "RHS contains elements that are NaN"
    if (norm(RHS) < eps(real(T))*1e3)
        if verbosity >= 2
            @warn "Not solving this cluster: norm(RHS) = $(norm(RHS))"
        end
        return spaces
    end
    println("sites to update = $(sites_to_update)")
    if length(sites_to_update) == 3 && cluster.m > 0
        solutions, _ = solve_3_loop_optim(lattice, RHS, spaces, levels_to_update; verbosity, symmetry)
    elseif length(sites_to_update) ∈ [1, 2]
        A = get_A(lattice, T, cluster, PEPO, sites_to_update)
        dir, conjugated = get_update_dir(lattice, cluster.cluster, sites_to_update)
        solutions = solve_index(lattice, T, A, RHS, conjugated, sites_to_update, levels_to_update, get_direction(lattice, dir), cluster.N, spaces; verbosity, svd)
    else
        @error "Number of sites to update = $(length(sites_to_update))"
        return spaces
    end
    if isnothing(solutions)
        if verbosity >= 2
            @warn "Solution returned nothing"
        end
        return spaces
    end
    levels_to_update, solutions = symmetrize(lattice, symmetry, levels_to_update, solutions)
    merge!(PEPO, Dict(zip(levels_to_update, solutions)))
    return spaces
end


function get_nontrivial_terms(lattice::Triangular, N; prev_clusters = [[(0,0)]])
    if N == 1
        return [[(0,0)]]
    end
    # initialize new list of clusters
    clusters = []
    for cluster_indices = prev_clusters # iterate over all previous clusters
        new_indices = []
        # get all the possible new values for the new index
        for (k₁,j₁) in cluster_indices
            for (k₂,j₂) = [(k₁+1,j₁), (k₁-1,j₁), (k₁,j₁+1), (k₁,j₁-1), (k₁+1,j₁-1), (k₁-1,j₁+1)]
                proposed_cluster = sort(vcat(cluster_indices, ((k₂,j₂))))
                if !(((k₂,j₂) in new_indices) || ((k₂,j₂) in cluster_indices) || proposed_cluster in clusters)
                    if (proposed_cluster ∉ clusters) && (move(proposed_cluster, -1, 0) ∉ clusters) && (move(proposed_cluster, 1, 0) ∉ clusters) && (move(proposed_cluster, 0, 1) ∉ clusters) && (move(proposed_cluster, 0, -1) ∉ clusters) && (move(proposed_cluster, 1, -1) ∉ clusters) && (move(proposed_cluster, -1, 1) ∉ clusters)
                        push!(clusters, proposed_cluster)
                        push!(new_indices, (k₂,j₂))
                    end
                end
            end
        end
    end
    return clusters
end

function get_all_indices(lattice::Triangular, T, PEPO, p, β, twosite_op, onesite_op, spaces; nn_term = nothing, levels_convention = "tree_depth", symmetry = nothing, verbosity = 2, solving_loops = true, svd = true)
    previous_clusters = [[(0,0)]]
    for N = 2:p
        if verbosity >= 2
            @info "N = $(N)"
        end
        cluster_indices = get_nontrivial_terms(lattice, N; prev_clusters = previous_clusters)
        clusters = [TriangularCluster(lattice, c; levels_convention = levels_convention, symmetry = symmetry) for c in cluster_indices]
        sort!(clusters, by = p -> (p.m, p.n)) # Sort the clusters such that the loops and higher levels are solved last
        for cluster = clusters
            spaces = solve_cluster(lattice, T, cluster, PEPO, β, twosite_op, onesite_op, spaces; nn_term, symmetry, verbosity, solving_loops, svd)
        end
        previous_clusters = cluster_indices
    end
    if verbosity >= 2
        for (key, tens) = PEPO
            @info "key = $(key)"
            @info "Maximum is $(maximum(abs.(tens.data))), norm is $(norm(tens))"
            @info "Summary = $(summary(tens))"
        end
    end    
    return PEPO
end    

function clusterexpansion(lattice::Lattice, T, p, β, twosite_op, onesite_op; nn_term = nothing, levels_convention = "tree_depth", spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : ℂ^10, symmetry = nothing, verbosity = 2, solving_loops = true, svd = true)
    (p < 10) || error("Only cluster up until 9th order are implemented correctly")
    dim(spaces(0)) == 1 || error("The zeroth space should be of dimension 1")
    pspace = domain(onesite_op)[1]
    PEPO₀ = init_PEPO(lattice, T, β, onesite_op, spaces(0))
    PEPO = get_all_indices(lattice, T, PEPO₀, p, β, twosite_op, onesite_op, spaces; nn_term, levels_convention, symmetry, verbosity, solving_loops, svd)
    return PEPO, get_PEPO(lattice, T, pspace, PEPO, spaces)
end

function clusterexpansion(lattice::Lattice, p, β, twosite_op, onesite_op; kwargs...)
    return clusterexpansion(lattice, Complex{Float64}, p, β, twosite_op, onesite_op; kwargs...)
end