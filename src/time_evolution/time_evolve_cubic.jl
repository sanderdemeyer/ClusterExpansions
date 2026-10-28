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

    F1 = isomorphism(VS, TS ⊗ VS[1] ⊗ VS[2] ⊗ TS' ⊗ VS[3] ⊗ VS[4])
    F2 = isomorphism(VS, VS[1] ⊗ TS ⊗ VS[2] ⊗ VS[3] ⊗ TS' ⊗ VS[4])
    F3 = isomorphism(VS, VS[1] ⊗ VS[2] ⊗ TS ⊗ VS[3] ⊗ VS[4] ⊗ TS')

    tens1 = tens * F1
    tens2 = tens * F2
    tens3 = tens * F3
    
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
    PEPO_2D, = clusterexpansion(ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; nn_term = ce_alg.nn_term, spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = (ce_alg.symmetry == "Oh") ? "C4v" : nothing , solving_loops = ce_alg.solving_loops, svd = ce_alg.svd)
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

function time_evolve_cubic(
    ce_alg::Union{ClusterExpansion,TrotterDecomposition},
    time_alg::StaticTimeEvolution,
    trunc_alg::EnvTruncation,
    observable;
    finalize! = nothing,
    A0 = nothing,
    canoc_alg::Union{Canonicalization,Nothing} = nothing,
    skip_first::Bool = false,
    initial_guesses = i -> nothing,
    saving::Bool = true,
    saving_frequency::Int = 1,
    normalizing::Bool = true,
    trunc_alg_start = nothing
)
    if isnothing(trunc_alg_start)
        As = AbstractTensorMap[evolution_operator_cubic(ce_alg, β; canoc_alg) for β = time_alg.βs_helper]
    else
        As = AbstractTensorMap[approximate_state(evolution_operator_cubic(ce_alg, β; canoc_alg), trunc_alg_start; rot_invariance = trunc_alg.rot_invariance)[1] for β = time_alg.βs_helper]
    end
    times = copy(time_alg.βs_helper)
    if isnothing(A0)
        A = evolution_operator_cubic(ce_alg, time_alg.β₀; canoc_alg)
    else
        A = canonicalize(A0, canoc_alg)
    end
    if skip_first
        obs = nothing
        expvals = []
    else
        obs = observable(A, 0)
        expvals = [obs]
    end
    push!(As, copy(A))
    push!(times, time_alg.β₀)
    errors = []
    for (i,ind) in enumerate(time_alg.update_list)
        if ind <= length(As)
            A, ϵ = approximate_state((A, As[ind]), trunc_alg; rot_invariance = trunc_alg.rot_invariance)
        elseif ind == i
            A, ϵ = approximate_state((A, A), trunc_alg; rot_invariance = trunc_alg.rot_invariance)
        else
            error("Cannot perform time evolution without saving intermediaire steps for this time algorithm")
        end
        if normalizing
            A /= norm(A)
        end
        A = canonicalize(A, canoc_alg)
        obs = observable(A, i)
        push!(errors, ϵ)
        push!(times, times[end] + times[ind])
        if saving
            push!(expvals, obs)
            if mod(i, saving_frequency) == 0
                push!(As, copy(A))
            end
        end
        if time_alg.verbosity > 1
            @info "Time evolution step $(i) with β = $(times[end]), obs = $(obs)"
            @info "Bond dimension is now $(dim(domain(A)[1]))"
            if time_alg.verbosity > 2
                @info "Current norm is $(norm(A))"
            end
        end
        if !isnothing(finalize!)
            finalize!(As, expvals, i)
        end
    end
    if saving
        return times[length(time_alg.βs_helper)+1:end], expvals, As[length(time_alg.βs_helper)+1:end], errors
    else
        return times[end], obs, A, errors
    end
end
