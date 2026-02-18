function evolution_operator_triangular(ce_alg::ClusterExpansion, β::Number; T_conv = ComplexF64, canoc_alg::Union{Nothing,Canonicalization} = nothing)
    if β == 0.0
        pspace = domain(ce_alg.onesite_op)[1]
        vspace = ce_alg.spaces(0)
        t = id(T_conv, pspace ⊗ vspace ⊗ vspace ⊗ vspace)
        return permute(t, ((1,5),(6,7,8,2,3,4)))
    end
    lattice = ClusterExpansions.Triangular()
    _, O_clust_full = clusterexpansion(lattice, ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; nn_term = ce_alg.nn_term, spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops, svd = ce_alg.svd)
    O_clust_full = convert(TensorMap, O_clust_full)
    O_canoc = canonicalize(O_clust_full, canoc_alg)
    O = zeros(T_conv, codomain(O_canoc), domain(O_canoc))
    for (f_full, f_conv) in zip(blocks(O_canoc), blocks(O))
        f_conv[2] .= f_full[2]
    end
    return O
end

function time_evolve_triangular(
    ce_alg::Union{ClusterExpansion,TrotterDecomposition},
    time_alg::StaticTimeEvolution,
    trunc_alg::Union{EnvTruncation,VOPEPO},
    observable;
    finalize! = nothing,
    A0 = nothing,
    canoc_alg::Union{Canonicalization,Nothing} = nothing,
    skip_first::Bool = false,
    initial_guesses = i -> nothing,
    saving = true
)
    As = AbstractTensorMap[evolution_operator_triangular(ce_alg, β; canoc_alg) for β = time_alg.βs_helper]
    times = copy(time_alg.βs_helper)
    if isnothing(A0)
        A = evolution_operator_triangular(ce_alg, time_alg.β₀; canoc_alg)
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
    
    if trunc_alg isa VOPEPO
        env_double, env_triple = initialize_vomps_environments(domain(A)[1], domain(As[time_alg.update_list[1]])[1], trunc_alg)
    end
    for (i,ind) in enumerate(time_alg.update_list)
        if trunc_alg isa VOPEPO
            # if domain(env_triple.edges[1,1,1])[1] ≠ domain(A)[1] ⊗ domain(As[ind])[1] ⊗ trunc_alg.truncspace'
            if codomain(env_triple.edges[1,1,1])[2] ≠ domain(A)[1] || codomain(env_triple.edges[1,1,1])[3] ≠ domain(As[ind])[1]
                env_double, env_triple = initialize_vomps_environments(domain(A)[1], domain(As[ind])[1], trunc_alg)
            end
            if ind <= length(As)
                A, env_double, env_triple, = approximate_state((A, As[ind]), env_double, env_triple, trunc_alg; iter = i, B = initial_guesses(i))
            elseif ind == i
                A, env_double, env_triple, = approximate_state((A, A), env_double, env_triple, trunc_alg; iter = i, B = initial_guesses(i))
            else
                @error "Cannot perform time evolution without saving intermediaire steps for this time algorithm"
            end
        else
            if ind <= length(As)
                A, _ = approximate_state((A, As[ind]), trunc_alg)
            elseif ind == i
                A, _ = approximate_state((A, A), trunc_alg)
            else
                @error "Cannot perform time evolution without saving intermediaire steps for this time algorithm"
            end
        end
        A = canonicalize(A, canoc_alg)
        obs = observable(A, i)
        push!(times, times[end] + times[ind])
        if saving
            push!(expvals, obs)
            push!(As, copy(A))
        end
        if time_alg.verbosity > 1
            @info "Time evolution step $(i) with β = $(times[end]), obs = $(obs)"
            @info "Bond dimension is now $(dim(domain(A)[1]))"
            if time_alg.verbosity > 2
                @info "Current norm is $(norm(A))"
            end
        end
        if !isnothing(finalize!)
            A = finalize!(As, expvals, i)
        end
    end
    if saving
        return times[length(time_alg.βs_helper)+1:end], expvals, As[length(time_alg.βs_helper)+1:end]
    else
        return times[end], obs, A
    end
end
