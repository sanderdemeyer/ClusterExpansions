abstract type TimeEvolution end

# struct StaticTimeEvolution <: TimeEvolution
#     β₀
#     Δβ
#     maxiter
#     trunc_alg
#     verbosity
# end

struct TimeDependentTimeEvolution <: TimeEvolution
    β₀
    Δβ
    maxiter
    verbosity
    f₁
    f₂
end

struct StaticTimeEvolution <: TimeEvolution
    β₀
    βs_helper
    update_list
    verbosity
end

# function evolution_operator(ce_alg::ClusterExpansion, time_alg::StaticTimeEvolution)
#     _, O_clust_full = clusterexpansion(ce_alg.T, ce_alg.p, time_alg.Δβ, ce_alg.twosite_op, ce_alg.onesite_op; spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops)
#     O_clust_full = convert(TensorMap, O_clust_full)
#     O = zeros(ComplexF64, codomain(O_clust_full), domain(O_clust_full))
#     for (f_full, f_conv) in zip(blocks(O_clust_full), blocks(O))
#         f_conv[2] .= f_full[2]
#     end
#     return O
# end

function evolution_operator(ce_alg::ClusterExpansion, time_alg::TimeDependentTimeEvolution, β::Number; T_conv = ComplexF64, canoc_alg::Union{Nothing,Canonicalization} = nothing)
    _, O_clust_full = clusterexpansion(ce_alg.T, ce_alg.p, time_alg.Δβ, time_alg.f₂(β) * ce_alg.twosite_op, time_alg.f₁(β) * ce_alg.onesite_op; nn_term = ce_alg.nn_term, spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops)
    O_clust_full = convert(TensorMap, O_clust_full)
    O_canoc = canonicalize(O_clust_full, canoc_alg)
    O = zeros(T_conv, codomain(O_canoc), domain(O_canoc))
    for (f_full, f_conv) in zip(blocks(O_canoc), blocks(O))
        f_conv[2] .= f_full[2]
    end
    return O
end

function evolution_operator(ce_alg::ClusterExpansion, β::Number; T_conv = ComplexF64, canoc_alg::Union{Nothing,Canonicalization} = nothing)
    if β == 0.0
        pspace = domain(ce_alg.onesite_op)[1]
        vspace = ce_alg.spaces(0)
        t = id(T_conv, pspace ⊗ vspace ⊗ vspace)
        return permute(t, ((1,4),(5,6,2,3)))
    end
    _, O_clust_full = clusterexpansion(ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; nn_term = ce_alg.nn_term, spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops)
    O_clust_full = convert(TensorMap, O_clust_full)
    O_canoc = canonicalize(O_clust_full, canoc_alg)
    O = zeros(T_conv, codomain(O_canoc), domain(O_canoc))
    for (f_full, f_conv) in zip(blocks(O_canoc), blocks(O))
        f_conv[2] .= f_full[2]
    end
    return O
end

function StaticTimeEvolution(β₀, βs_helper, update_list; verbosity = 0)
    return StaticTimeEvolution(β₀, βs_helper, update_list, verbosity)
end

function UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 0)
    return StaticTimeEvolution(β₀, [Δβ], [1 for i = 1:maxiter], verbosity)
end

function SquaringTimeEvolution(β₀, maxiter; verbosity = 0)
    return StaticTimeEvolution(β₀, [], 1:maxiter, verbosity)
end

function TimeDependentTimeEvolution(β₀, Δβ, maxiter; verbosity = 0, f₁ = β -> 1.0, f₂ = β -> 1.0)
    return TimeDependentTimeEvolution(β₀, Δβ, maxiter, verbosity, f₁, f₂)
end

function time_evolve(
    ce_alg::ClusterExpansion,
    time_alg::StaticTimeEvolution,
    trunc_alg::Union{EnvTruncation,VOPEPO},
    observable;
    finalize! = nothing,
    A0 = nothing
)
    # canoc_alg = Canonicalization(; verbosity = 1)
    canoc_alg = nothing
    As = AbstractTensorMap[evolution_operator(ce_alg, β; canoc_alg) for β = time_alg.βs_helper]
    times = copy(time_alg.βs_helper)
    if isnothing(A0)
        A = evolution_operator(ce_alg, time_alg.β₀; canoc_alg)
    else
        A = canonicalize(A0, canoc_alg)
    end

    expvals = [observable(A)]
    push!(As, copy(A))
    push!(times, time_alg.β₀)
    
    if trunc_alg isa VOPEPO
        env_double, env_triple = initialize_vomps_environments(domain(A)[1], domain(As[time_alg.update_list[1]])[1], trunc_alg)
    end
    for (i,ind) in enumerate(time_alg.update_list)
        if trunc_alg isa VOPEPO
            if domain(env_triple.edges[1,1,1])[1] ≠ domain(A)[1] ⊗ domain(As[ind])[1] ⊗ trunc_alg.truncspace'
                env_double, env_triple = initialize_vomps_environments(domain(A)[1], domain(As[ind])[1], trunc_alg)
            end
            A, env_double, env_triple, _ = approximate_state((A, As[ind]), env_double, env_triple, trunc_alg; iter = i)
        else
            A, _ = approximate_state((A, As[ind]), trunc_alg)
        end
        A /= norm(A)
        A = canonicalize(A, canoc_alg)
        obs = observable(A)
        push!(times, times[end] + times[ind])
        push!(expvals, obs)
        push!(As, copy(A))
        
        if time_alg.verbosity > 1
            @info "Time evolution step $(i) with β = $(times[end]), obs = $(obs)"
            @info "Bond dimension is now $(dim(domain(A)[1]))"
            if time_alg.verbosity > 2
                @info "Current norm is $(norm(A))"
            end
        end
        if !isnothing(finalize!)
            A = finalize!(A, obs, i)
        end
    end
    return times[length(time_alg.βs_helper)+1:end], expvals, As[length(time_alg.βs_helper)+1:end]
end

function get_time_array(time_alg::StaticTimeEvolution)
    times = copy(time_alg.βs_helper)
    push!(times, time_alg.β₀)
    for ind in time_alg.update_list
        push!(times, times[end] + times[ind])
    end
    return times[length(time_alg.βs_helper)+1:end]
end

function time_scan(
    ce_alg::ClusterExpansion,
    times::Array,
    observable;
    verbosity::Int=0,
    finalize! = nothing
)
    expvals = []
    As = []
    for (i,t) = enumerate(times)
        O = evolution_operator(ce_alg, t)

        obs = observable(O)
        if verbosity > 0 && any([imag(ob) / real(ob) > 1e-5 for ob in obs])
            @warn "Complex value for observable: $([imag(ob) / real(ob) for ob in obs])"
        end

        push!(expvals, real(obs))
        push!(As, copy(O))
        if verbosity > 1
            @info "Time evolution step $(i) with β = $(t), obs = $(obs)"
        end
        if !isnothing(finalize!)
            finalize!(O, obs, i)
        end
    end
    return times, expvals, As
end
