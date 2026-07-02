function get_PEPO_cubic(T, pspace, PEPO, spaces)
    highest = [maximum([i[dir] for i = keys(PEPO)]) for dir = 1:6]
    highest_loop = [minimum([i[dir] for i = keys(PEPO)]) for dir = 1:6]
    conjugated = Bool[0, 0, 0, 1, 1, 1]
    O = zeros(T, SumSpace(pspace) ⊗ SumSpace(pspace)', ⊗([conj ? (get_sum_space(h, hloop, spaces))' : get_sum_space(h, hloop, spaces) for (conj,h,hloop) = zip(conjugated,highest,highest_loop)]...))
    for (key, tens) = PEPO
        key = [i < 0 ? h - i + 1 : i + 1 for (i,h) = zip(key,highest)]
        O[1,1,key...] = tens
    end
    return O
end

function construct_levels(key)
    return [(0, key[1], key[2], 0, key[3], key[4]),
    (key[1], 0, key[2], key[3], 0, key[4]),
    (key[1], key[2], 0, key[3], key[4], 0)
    ]
end

function construct_tensors(tens, TS; T = scalartype(tens))
    VS = domain(tens)
    VS1 = [TS, VS[1], VS[2], TS', VS[3], VS[4]]
    VS2 = [VS[1], TS, VS[2], VS[3], TS', VS[4]]
    VS3 = [VS[1], VS[2], TS, VS[3], VS[4], TS']

    tens1 = zeros(T, codomain(tens), ⊗(VS1...))
    tens2 = zeros(T, codomain(tens), ⊗(VS2...))
    tens3 = zeros(T, codomain(tens), ⊗(VS3...))

    tens1[:,:,1,:,:,1,:,:] .= tens[:,:,:,:,:,:]
    tens2[:,:,:,1,:,:,1,:] .= tens[:,:,:,:,:,:]
    tens3[:,:,:,:,1,:,:,1] .= tens[:,:,:,:,:,:]

    return [tens1, tens2, tens3]
end

function construct_cubic_CE(O_2D)
    keys_3D = []
    tensors_3D = []

    TS = domain(O_2D[(0,0,0,0)])[1]
    for (key,tens) in O_2D
        push!(keys_3D, construct_levels(key)...)
        push!(tensors_3D, construct_tensors(tens, TS)...)
    end
    return Dict(zip(keys_3D, tensors_3D))
end

function evolution_operator_cubic(ce_alg::ClusterExpansion, β::Number; T_conv = ComplexF64, canoc_alg::Union{Nothing,Canonicalization} = nothing)
    pspace = domain(ce_alg.onesite_op)[1]
    if β == 0.0
        vspace = ce_alg.spaces(0)
        t = id(T_conv, pspace ⊗ vspace ⊗ vspace ⊗ vspace)
        return permute(t, ((1,5),(6,7,8,2,3,4)))
    end
    PEPO_2D, = clusterexpansion(ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; nn_term = ce_alg.nn_term, spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops, svd = ce_alg.svd)
    PEPO_3D = construct_cubic_CE(PEPO_2D)
    O_clust_full = get_PEPO_cubic(ce_alg.T, pspace, PEPO_3D, ce_alg.spaces)

    O_clust_full_tm = convert(TensorMap, O_clust_full)
    O_canoc = canonicalize(O_clust_full_tm, canoc_alg)
    O = zeros(T_conv, codomain(O_canoc), domain(O_canoc))
    for (f_full, f_conv) in zip(blocks(O_canoc), blocks(O))
        f_conv[2] .= f_full[2]
    end
    return O # Don't normalize, otherwise Atsushi will be mad.
end
