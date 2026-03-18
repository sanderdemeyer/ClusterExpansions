function _apply_A_honeycomb(x::TensorMap, tens_01::TensorMap, tens_10::TensorMap, ::Val{false})
    Ax = ncon([tens_01, x, tens_10], [[-1 -4 1], [-2 -5 1 2], [-3 -6 2]])
    return permute(Ax, ((1,2,3),(4,5,6)))
end

function _apply_A_honeycomb(Ax::TensorMap, tens_01::TensorMap, tens_10::TensorMap, ::Val{true})
    x = ncon([tens_01, tens_10, Ax], [[1 2 -3], [3 4 -4], [1 -1 3 2 -2 4]], [true true false])
    return permute(x, ((1,2),(3,4)))
end

function get_2nd_order(ce_alg, β, residual, cluster_2, twosite_op)
    exp_H = ClusterExpansions.exponentiate_hamiltonian(twosite_op, ce_alg.onesite_op, cluster_2, β)
    RHS_2 = exp_H - residual
    b = permute(RHS_2, ((1,3), (2,4)))
    if ce_alg.svd
        U, S, V = tsvd(b; trunc = truncspace(ce_alg.spaces(1)))
        T01 = U * sqrt(S)
        T10 = permute(sqrt(S) * V, ((2,3),(1,)))
    else
        eigval, eigvec = eig_with_truncation_honeycomb(flip(b, [3 4]), ce_alg.spaces(1))
        T01 = eigvec * sqrt(eigval)
        T10 = flip(permute(sqrt(eigval) * eigvec', ((2,3),(1,))), [1 2])
    end
    return T01, T10
end

# function exponentiate_hamiltonian(twosite_op, onesite_op, cluster, β; nn_term = nothing)
function exponentiate_hamiltonian_honeycomb_manual(twosite_ops, onesite_op, cluster, β; nn_term = nothing)
    N = cluster.N
    pspace = domain(twosite_ops[1])[1]
    H = []
    for (ind,(i, j)) in enumerate(cluster.bonds_indices)
        println("ind = $ind, i = $i, j = $j")
        term = ncon([twosite_ops[ind], [id(pspace) for _ in 1:(N - 2)]...], [[-i, -j, -N - i, -N - j], [[-k, -N - k] for k in setdiff(1:N, (i, j))]...], [false for _ in 1:(N - 1)])
        push!(H, permute(term, (Tuple(1:N), Tuple((N + 1):(2 * N)))))
        println("bond index")
    end
    for site in 1:N
        term = ncon([onesite_op, [id(pspace) for _ in 1:(N - 1)]...], [[-site, -N - site], [[-k, -N - k] for k in setdiff(1:N, site)]...], [false for _ in 1:N])
        push!(H, permute(term, (Tuple(1:N), Tuple((N + 1):(2 * N)))))
        println("site index")
    end
    if !isnothing(nn_term)
        for (i, j) in cluster.diag_bonds_indices
            term = ncon([nn_term, [id(pspace) for _ in 1:(N - 2)]...], [[-i, -j, -N - i, -N - j], [[-k, -N - k] for k in setdiff(1:N, (i, j))]...], [false for _ in 1:(N - 1)])
            push!(H, permute(term, (Tuple(1:N), Tuple((N + 1):(2 * N)))))
            println("nn bond index")
        end
    end
    println("length of H = $(length(H))")
    exp_H = exp(-β * sum(H))
    return exp_H    
end

function get_PEPO_manual(ce_alg, β, operators)
    onesite_op = operators[1]
    twosite_op_x, twosite_op_y, twosite_op_z = operators[2:4]
    trivspace = ce_alg.spaces(0)

    T000 = permute(exp(-β*onesite_op), ((1,2),()))
    isometry000 = isometry(trivspace' ⊗ trivspace, trivspace')
    PEPO000 = T000 * permute(isometry000, ((),(1,2,3)))
    PEPO = Dict((0,0,0) => PEPO000)

    if ce_alg.p == 1
        return PEPO
    end

    cluster_2 = ClusterExpansions.Cluster([(0, 0), (0, 1)])
    @tensor residual[-1 -2; -3 -4] := T000[-1; -3] * T000[-2; -4]
    T01x, T10x = get_2nd_order(ce_alg, β, residual, cluster_2, twosite_op_x)
    T01y, T10y = get_2nd_order(ce_alg, β, residual, cluster_2, twosite_op_y)
    T01z, T10z = get_2nd_order(ce_alg, β, residual, cluster_2, twosite_op_z)

    @assert norm(T01x - flip(T10x, [3])) < 1e-14
    @assert norm(T01y - flip(T10y, [3])) < 1e-14
    @assert norm(T01z - flip(T10z, [3])) < 1e-14

    vspace = domain(T01z)
    isometry010 = isometry(vspace, vspace ⊗ trivspace ⊗ trivspace)
    PEPO010 = flip(permute(T01z * isometry010, ((1,2),(4,3,5))), [4 5])

    vspace = domain(T01x)
    isometry100 = isometry(vspace, vspace ⊗ trivspace ⊗ trivspace)
    PEPO100 = flip(permute(T01x * isometry100, ((1,2),(3,4,5))), [4 5])

    vspace = domain(T01y)
    isometry001 = isometry(vspace, vspace ⊗ trivspace ⊗ trivspace)
    PEPO001 = flip(permute(T01y * isometry001, ((1,2),(4,5,3))), [4 5])
    merge!(PEPO, Dict(zip([(0,1,0) (1,0,0) (0,0,1)], [PEPO010, PEPO100, PEPO001])))
    if ce_alg.p == 2
        return PEPO
    end

    # Third order
    # Residuals
    @tensor res00[-1 -2 -3; -4 -5 -6] := T000[-1; -4] * T000[-2; -5] * T000[-3; -6]
    # @tensor res10[-1 -2 -3; -4 -5 -6] := T01[-1 -4; 1] * T10[-2 -5; 1] * T000[-3; -6]
    # @tensor res01[-1 -2 -3; -4 -5 -6] := T000[-1; -4] * T01[-2 -5; 1] * T10[-3 -6; 1]

    # Calculate RHS's
    # \
    #  |
    # /
    @tensor res10[-1 -2 -3; -4 -5 -6] := T01x[-1 -4; 1] * T10x[-2 -5; 1] * T000[-3; -6]
    @tensor res01[-1 -2 -3; -4 -5 -6] := T000[-1; -4] * T01y[-2 -5; 1] * T10y[-3 -6; 1]
    cluster_3 = ClusterExpansions.Cluster([(0, 0), (0, 1), (1, 1)])
    exp_H_3 = exponentiate_hamiltonian_honeycomb_manual([twosite_op_x, twosite_op_y], onesite_op, cluster_3, β; nn_term = ce_alg.nn_term)
    RHS_3 = exp_H_3 - res00 - res10 - res01
    apply_A_AD = (x, val) -> _apply_A_honeycomb(x, T01y, T10x, val)
    T11_101, info = lssolve(apply_A_AD, RHS_3, LSMR(verbosity = 1, maxiter = 1000))

    #  --
    # /
    @tensor res10[-1 -2 -3; -4 -5 -6] := T01y[-1 -4; 1] * T10y[-2 -5; 1] * T000[-3; -6]
    @tensor res01[-1 -2 -3; -4 -5 -6] := T000[-1; -4] * T01z[-2 -5; 1] * T10z[-3 -6; 1]
    cluster_3 = ClusterExpansions.Cluster([(0, 0), (1, 0), (1, 1)])
    exp_H_3 = exponentiate_hamiltonian_honeycomb_manual([twosite_op_y, twosite_op_z], onesite_op, cluster_3, β; nn_term = ce_alg.nn_term)
    RHS_3 = exp_H_3 - res00 - res10 - res01
    apply_A_AD = (x, val) -> _apply_A_honeycomb(x, T01y, T10z, val)
    T11_011, info = lssolve(apply_A_AD, RHS_3, LSMR(verbosity = 1, maxiter = 1000))

    # \
    #  --
    @tensor res10[-1 -2 -3; -4 -5 -6] := T01x[-1 -4; 1] * T10x[-2 -5; 1] * T000[-3; -6]
    @tensor res01[-1 -2 -3; -4 -5 -6] := T000[-1; -4] * T01z[-2 -5; 1] * T10z[-3 -6; 1]
    cluster_3 = ClusterExpansions.Cluster([(0, 0), (1, 0), (1, 1)])
    exp_H_3 = exponentiate_hamiltonian_honeycomb_manual([twosite_op_x, twosite_op_z], onesite_op, cluster_3, β; nn_term = ce_alg.nn_term)
    RHS_3 = exp_H_3 - res00 - res10 - res01
    apply_A_AD = (x, val) -> _apply_A_honeycomb(x, T01x, T10z, val)
    T11_110, info = lssolve(apply_A_AD, RHS_3, LSMR(verbosity = 1, maxiter = 1000))


    vspace = domain(T11_101)
    isometry101 = isometry(vspace, vspace ⊗ trivspace)
    PEPO101 = flip(permute(T11_101 * isometry101, ((1,2),(3,5,4))), [3 4 5])

    vspace = domain(T11_011)
    isometry011 = isometry(vspace, vspace ⊗ trivspace)
    PEPO011 = flip(permute(T11_011 * isometry011, ((1,2),(5,3,4))), [5])

    vspace = domain(T11_110)
    isometry110 = isometry(vspace, vspace ⊗ trivspace)
    PEPO110 = flip(permute(T11_110 * isometry110, ((1,2),(3,4,5))), [3 4 5])

    println("PEPO101: space = $(PEPO101.space)")
    println("PEPO011: space = $(PEPO011.space)")
    println("PEPO110: space = $(PEPO110.space)")
    println("norms are $(norm(PEPO101)), $(norm(PEPO011)), $(norm(PEPO110))")
    # PEPO101 = flip(permute(PEPO110, ((1,2),(4,5,3))), [3 5])
    # PEPO011 = flip(permute(PEPO101, ((1,2),(4,5,3))), [3 5])

    merge!(PEPO, Dict(zip([(1,1,0) (1,0,1) (0,1,1)], [PEPO110, PEPO101, PEPO011])))

    for (key,tens) in PEPO
        println("key = $key, tens = $(tens.space)")
    end
    return PEPO
end

function clusterexpansion(lattice::Honeycomb, T, p, β, twosite_op, onesite_op; nn_term = nothing, levels_convention = "tree_depth", spaces = i -> (i >= 0) ? ℂ^(2^(2 * i)) : ℂ^10, symmetry = nothing, verbosity = 2, solving_loops = true, svd = true)
    (p < 4) || error("Only cluster up until 3th order are implemented for the honeycomb lattice")
    dim(spaces(0)) == 1 || error("The zeroth space should be of dimension 1")
    pspace = domain(onesite_op)[1]
    envspace = χ -> ℂ^χ
    ce_alg = ClusterExpansion(twosite_op, onesite_op, nn_term, p, verbosity, T, spaces, symmetry, solving_loops, svd, envspace)

    operators = [ce_alg.onesite_op, ce_alg.twosite_op...]

    PEPO = get_PEPO_manual(ce_alg, β, operators)
    return PEPO, get_PEPO(lattice, T, pspace, PEPO, spaces)
end

function clusterexpansion(lattice::Honeycomb, p, β, twosite_op, onesite_op; kwargs...)
    return clusterexpansion(lattice, Complex{Float64}, p, β, twosite_op, onesite_op; kwargs...)
end
