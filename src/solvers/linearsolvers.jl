function apply_contraction(x, PEPO, cluster_levels)
    return nothing
end

function get_triv_tensors(conjugated, trivspace)
    tensor = Tensor([1.0], trivspace)
    tensor_conj = Tensor([1.0], trivspace')
    return [conj ? tensor_conj : tensor for conj = conjugated] 
end

function get_A(PEPO, levels_sites, number_of_bonds, sites_to_update, conjugated, open_conjugated, open_indices, contraction_indices)
    trivspace = ℂ^1
    included_sites = setdiff(1:length(levels_sites)[1], sites_to_update)
    println("included = $(included_sites)")
    nontriv_tensors = [PEPO[levels_sites[site]] for site = included_sites]
    triv_tensors = vcat(get_triv_tensors(conjugated, trivspace), get_triv_tensors(open_conjugated, trivspace))
    triv_contractions = vcat([[number_of_bonds+i] for i = 1:length(conjugated)], [[i] for i = open_indices])
    all_contractions = vcat([contraction_indices[i,:] for i = 1:size(contraction_indices)[1]], triv_contractions)
    all_tensors = vcat(nontriv_tensors, triv_tensors)

    A = ncon(all_tensors, all_contractions)
    return A
end

function apply_A_twosite_update(A, x::TensorMap, sites_to_update, N) # ::Val{false}
    # N is length of the cluster
    (i,j) = sites_to_update
    included_sites = setdiff(1:N, sites_to_update)
    conv = i -> included_sites[i]
    contraction_indices = vcat(-included_sites, -included_sites .- N, [1, 2, 3, 4, 5, 6])

    println("ci_A = $(contraction_indices)")
    println("ci_x = $([-i, -N-i, 1, 2, 3, -j, -N-j, 4, 5, 6])")
    Ax = ncon([A, x], [contraction_indices, [-i, -N-i, 1, 2, 3, -j, -N-j, 4, 5, 6]])
    println("done with contraction")
    return @tensor Ax[-1 -2 -3; -4 -5 -6] := A_l[-1; -4 1] * x[1 -2; -5 2] * A_r[2 -3; -6]
end
function apply_A_twosite_update(x::TensorMap, ::Val{true})
    return @tensor Adagx[-1 -2; -3 -4] := adjoint(A_l)[3 -1; 1] * x[1 -2 2; 3 -3 4] * adjoint(A_r)[4; -4 2]
end

function solve_index(updates, A, exp_H, conjugated, sites_to_update, N; spaces = [(0, 0, 0, 0)])
    println("conjugated = $(conjugated)")
    pspace = ℂ^2
    trivspace = ℂ^1
    x0 = TensorMap(randn, pspace ⊗ pspace' ⊗ prod([conj ? trivspace' : trivspace for conj = conjugated[1]]), pspace ⊗ pspace' ⊗ prod([conj ? trivspace' : trivspace for conj = conjugated[2]]))

    println("exp_H = $(summary(exp_H))")
    println("A = $(summary(A))")
    println("x0 = $(summary(x0))")

    apply_A_twosite_update(A, x0, sites_to_update, N)

    return [TensorMap(randn, ℂ^2 ⊗ (ℂ^2)' ← ℂ^(sp[1]+1) ⊗ ℂ^(sp[2]+1) ⊗ (ℂ^(sp[3]+1))' ⊗ (ℂ^(sp[4]+1))') for sp in spaces]
end

function get_other_tensors(cluster, PEPO, levels_sites, sites_to_update, contraction_indices, conjugated, number_of_bonds)
    pspace = ℂ^2
    edge = Tensor([1.0], ℂ^1)
    conj_edge = Tensor([1.0], (ℂ^1)')
    edge_tensors = [i ? conj_edge : edge for i = conjugated]
    cluster_tensors = [PEPO[lev] for (i,lev) = enumerate(levels_sites) if !(i ∈ sites_to_update)]
    contraction_indices = collect.(contraction_indices)
    contraction_indices_edges = [[i] for i = number_of_bonds:number_of_bonds+length(conjugated)]
    println("edge tensors = $(edge_tensors)")
    println("cluster tensors = $(cluster_tensors)")
    println("contraction_indices = $(contraction_indices)")
    println("contraction_indices_edges = $(contraction_indices_edges)")
    println("first argum = $(vcat(cluster_tensors, edge_tensors))")
    println("second argum = $(vcat(contraction_indices, contraction_indices_edges))")
    final_tensor = ncon(vcat(cluster_tensors, edge_tensors), vcat(contraction_indices, contraction_indices_edges))
    println(final_tensor)
    error("test")
end

function fixed_tensors(cluster, indices_to_update, bonds_indices, levels_sites, PEPO)
    tensors = []
    indices = []
    for (i,c) = enumerate(cluster)
        if !(i ∈ indices_to_update)
            push!(tensors, PEPO[levels_sites[i]])
        end
    end
end

function twosite_update(cluster, β, sites_to_update, contraction_indices, conjugated)
    bonds_sites, bonds_indices = get_bonds(cluster)

    # F = apply_contraction(x, PEPO, )

    current_tensors = []

    RHS = exponentiate_hamiltonian(cluster, β)
    pspace = ℂ^2
    edge = Tensor([1.0], ℂ^1)
    conj_edge = Tensor([1.0], (ℂ^1)')
    x0 = TensorMap(randn, pspace ⊗ pspace, pspace ⊗ pspace)
    f = x -> apply_contraction(x, PEPO, cluster_levels)
    x = linsolve(f, RHS, x0)
end