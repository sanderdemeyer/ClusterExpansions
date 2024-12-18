function init_PEPO(pspace::ElementarySpace, trivspace::ElementarySpace)
    return Dict((0,0,0,0) => TensorMap([1.0 0.0; 0.0 1.0], pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace'))   
end

function init_PEPO(pspace::ElementarySpace, trivspace::ElementarySpace, onesite_op::AbstractTensorMap)
    A = TensorMap(zeros, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace')
    A[][:,:,1,1,1,1] = exp(-onesite_op)[]
    return Dict((0,0,0,0) => A)
end

function init_PEPO()
    return init_PEPO(ℂ^2, ℂ^1)
end

function init_PEPO(onesite_op::AbstractTensorMap)
    return init_PEPO(ℂ^2, ℂ^1, onesite_op)
end

function get_PEPO(pspace, PEPO)
    highest = [get_size_level(maximum([i[dir] for i = keys(PEPO)])) for dir = 1:4]
    highest_loop = [get_size_level_loop(minimum([i[dir] for i = keys(PEPO)])) for dir = 1:4]
    conjugated = Bool[0, 0, 1, 1]
    O = TensorMap(zeros, pspace ⊗ pspace', prod([conj ? (ℂ^(h+hloop))' : ℂ^(h+hloop) for (conj,h,hloop) = zip(conjugated,highest, highest_loop)]))
    for (key, tens) = PEPO
        places = [get_location_PEPO(ind, highest) for ind = key]
        O[][:,:,places[1],places[2],places[3],places[4]] = tens[]
    end
    return O
end

function solve_cluster(cluster, PEPO, β, twosite_op)
    println("in solve cluster - cluster = $(cluster)")
    exp_H = exponentiate_hamiltonian(twosite_op, cluster, β, length(cluster))
    residual = contract_PEPS(cluster, PEPO)
    RHS = exp_H - residual
    @assert !(any(isnan.(convert(Array,RHS[][:])))) "RHS contains elements that are NaN"

    levels_sites = get_levels_sites(cluster)
    if levels_sites === nothing
        if N == 4.1
            levels_to_update, solution, err = solve_4_loop(α, RHS)
            merge!(PEPO, Dict(zip(levels_to_update, solution)))
        else
            (levels_sites === nothing) && (return PEPO)
        end
    end

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
    solution = solve_index(A, exp_H-residual, conjugated, sites_to_update, levels_to_update, get_direction(dir), length(cluster); spaces = i -> ℂ^(2^(2*i)))
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
    return PEPO
end    

function clusterexpansion(p, β, twosite_op, onesite_op)
    pspace = onesite_op.dom[1]
    PEPO₀ = init_PEPO(onesite_op)
    PEPO = get_all_indices(PEPO₀, p, β, twosite_op)
    return get_PEPO(pspace, PEPO)
end

# cluster = [(-1, 0), (0, 0), (1, 0)]

# x0 = TensorMap(randn, ℂ^2 ⊗ (ℂ^2)' ← ℂ^(7) ⊗ ℂ^(6) ⊗ (ℂ^(8))' ⊗ (ℂ^(4))')
# summary(x0)


# x0[][:,:,1:4,1,5:8,1] = sol[];

# loop = [(0,0),(0,1),(1,0),(1,1)]
# exp_H = exponentiate_hamiltonian(loop, β)