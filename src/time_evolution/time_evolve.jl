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

struct GroundStateTimeEvolution <: TimeEvolution
    β₀
    βs_helper
    update_list
    tol_energy
    verbosity
end

struct GroundStateFillingTimeEvolution <: TimeEvolution
    β₀
    Δβ
    maxiter
    f_target
    μ₀
    α
    tol_energy
    verbosity
end

function StaticTimeEvolution(β₀, βs_helper, update_list; verbosity = 0)
    return StaticTimeEvolution(β₀, βs_helper, update_list, verbosity)
end

function UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 0)
    return StaticTimeEvolution(β₀, [Δβ], [1 for i = 1:maxiter], verbosity)
end

function UniformGroundStateTimeEvolution(β₀, Δβ, maxiter, tol_energy; verbosity = 0)
    return GroundStateTimeEvolution(β₀, [Δβ], [1 for i = 1:maxiter], tol_energy, verbosity)
end

function SquaringTimeEvolution(β₀, maxiter; verbosity = 0)
    return StaticTimeEvolution(β₀, [], 1:maxiter, verbosity)
end

function SquaringGroundStateTimeEvolution(β₀, maxiter, tol_energy; verbosity = 0)
    return GroundStateTimeEvolution(β₀, [], 1:maxiter, tol_energy, verbosity)
end

function TimeDependentTimeEvolution(β₀, Δβ, maxiter; verbosity = 0, f₁ = β -> 1.0, f₂ = β -> 1.0)
    return TimeDependentTimeEvolution(β₀, Δβ, maxiter, verbosity, f₁, f₂)
end

function UniformGroundStateFillingTimeEvolution(β₀, Δβ, maxiter, f_target; μ₀ = 0.0, α = 1e-3, tol_energy = 1e-5, verbosity = 0)
    return GroundStateFillingTimeEvolution(β₀, Δβ, maxiter, f_target, μ₀, α, tol_energy, verbosity)
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
    _, O_clust_full = clusterexpansion(ce_alg.T, ce_alg.p, time_alg.Δβ, time_alg.f₂(β) * ce_alg.twosite_op, time_alg.f₁(β) * ce_alg.onesite_op; nn_term = ce_alg.nn_term, spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops, svd = ce_alg.svd)
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
    _, O_clust_full = clusterexpansion(ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; nn_term = ce_alg.nn_term, spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops, svd = ce_alg.svd)
    O_clust_full = convert(TensorMap, O_clust_full)
    O_canoc = canonicalize(O_clust_full, canoc_alg)
    O = zeros(T_conv, codomain(O_canoc), domain(O_canoc))
    for (f_full, f_conv) in zip(blocks(O_canoc), blocks(O))
        f_conv[2] .= f_full[2]
    end
    return O # Don't normalize, otherwise Atsushi will be mad.
end

function evolution_operator(td_alg::GenericTrotterDecomposition, β::Number; T = ComplexF64, canoc_alg::Union{Nothing,Canonicalization} = nothing)
    if β == 0.0
        pspace = domain(td_alg.onesite_op)[1]
        vspace = td_alg.spaces(0)
        t = id(T, pspace ⊗ vspace ⊗ vspace)
        return permute(t, ((1,4),(5,6,2,3)))
    end
    U_onesite = get_Trotter_onesite(td_alg.onesite_op, td_alg.g, β)
    U_twosite = get_Trotter_twosite(td_alg.twosite_op, td_alg.spaces(1), β)
    @tensor O_Trotter[-1 -2; -3 -4 -5 -6] := U_onesite[-1; 1] * U_twosite[1 2; -3 -4 -5 -6] * U_onesite[2; -2]
    O_canoc = canonicalize(O_Trotter, canoc_alg)
    return O_canoc
end

function evolution_operator(td_alg::TwositeTrotterDecomposition, β::Number; T = ComplexF64, canoc_alg::Union{Nothing,Canonicalization} = nothing)
    if β == 0.0
        pspace = domain(td_alg.onesite_op)[1]
        vspace = td_alg.spaces(0)
        t = id(T, pspace ⊗ vspace ⊗ vspace)
        return permute(t, ((1,4),(5,6,2,3)))
    end
    O_Trotter = get_Trotter_twosite(td_alg.twosite_op, td_alg.spaces(1), β)
    O_canoc = canonicalize(O_Trotter, canoc_alg)
    return O_canoc
end

function MPSKit.time_evolve(
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
    As = AbstractTensorMap[evolution_operator(ce_alg, β; canoc_alg) for β = time_alg.βs_helper]
    times = copy(time_alg.βs_helper)
    if isnothing(A0)
        A = evolution_operator(ce_alg, time_alg.β₀; canoc_alg)
    else
        A = canonicalize(A0, canoc_alg)
    end
    if skip_first
        obs = nothing
        expvals = []
    else
        obs = observable(A)
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
        A /= norm(A)
        A = canonicalize(A, canoc_alg)
        obs = observable(A)
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

function time_evolve_filling(
    ce_alg::F,
    time_alg::GroundStateFillingTimeEvolution,
    trunc_alg::Union{EnvTruncation,VOPEPO},
    observable;
    finalize! = nothing,
    canoc_alg::Union{Canonicalization,Nothing} = nothing,
    initial_guesses = i -> nothing,
    check_energy::Bool = false
) where {F <: Function}
    @assert !(trunc_alg isa VOPEPO) "VOPEPO not implemented for filling control"
    A = evolution_operator(ce_alg(time_alg.μ₀), time_alg.β₀; canoc_alg)
    As = AbstractTensorMap[A]
    μs = Float64[time_alg.μ₀]
    times = Float64[0.0]

    expvals = [observable(A)]
    μ = time_alg.μ₀

    if trunc_alg isa VOPEPO
        env_double, env_triple = initialize_vomps_environments(domain(A)[1], domain(As[time_alg.update_list[1]])[1], trunc_alg) # this will give errors due to time_alg.update_list
    end
    for i = 1:time_alg.maxiter
        μ -= real((expvals[end][1] - time_alg.f_target) * time_alg.α)
        println("μ = $(μ), n = $(expvals[end][1])")
        if trunc_alg isa VOPEPO
            if codomain(env_triple.edges[1,1,1])[2] ≠ domain(A)[1] || codomain(env_triple.edges[1,1,1])[3] ≠ domain(As[ind])[1]
                env_double, env_triple = initialize_vomps_environments(domain(A)[1], domain(As[ind])[1], trunc_alg)
            end
            A, env_double, _ = approximate_state((A, As[ind]), env_double, env_triple, trunc_alg; iter = i, B = initial_guesses(i))
        else
            A, _ = approximate_state((A, evolution_operator(ce_alg(μ), time_alg.Δβ)), trunc_alg)
        end
        A /= norm(A)
        A = canonicalize(A, canoc_alg)
        obs = observable(A)
        push!(times, times[end] + time_alg.Δβ)
        push!(expvals, obs)
        push!(As, copy(A))
        push!(μs, copy(μ))
        if time_alg.verbosity > 1
            @info "Time evolution step $(i) with β = $(times[end]), μ = $(μ) obs = $(obs)"
            @info "Bond dimension is now $(dim(domain(A)[1]))"
            if time_alg.verbosity > 2
                @info "Current norm is $(norm(A))"
            end
        end
        if !isnothing(finalize!)
            A = finalize!(As, expvals, i)
        end
        if check_energy && i > 2 && abs(expvals[end][2] - expvals[end-1][2]) < time_alg.tol_energy
            if time_alg.verbosity > 1
                @info "Ground state search converged after $(i) iterations. Energy is $(expvals[end][1])"
                return times, expvals, μs, As
            end
        end
    end
    if time_alg.verbosity > 0
        @warn "Ground state search did not converge after $(maxiter) iterations. Energy is $(expvals[end][1])"
    end
    return times, expvals, μs, As
end

function get_time_array(time_alg::StaticTimeEvolution)
    times = copy(time_alg.βs_helper)
    push!(times, time_alg.β₀)
    for ind in time_alg.update_list
        push!(times, times[end] + times[ind])
    end
    return times[length(time_alg.βs_helper)+1:end]
end

function PEPSKit.fixedpoint(
    ce_alg::ClusterExpansion,
    time_alg::GroundStateTimeEvolution,
    trunc_alg::Union{EnvTruncation,VOPEPO},
    observable;
    finalize! = nothing,
    A0 = nothing,
    canoc_alg::Union{Canonicalization,Nothing} = nothing,
    skip_first::Bool = false
)
    As = AbstractTensorMap[evolution_operator(ce_alg, β; canoc_alg) for β = time_alg.βs_helper]
    times = copy(time_alg.βs_helper)
    if isnothing(A0)
        A = evolution_operator(ce_alg, time_alg.β₀; canoc_alg)
    else
        A = canonicalize(A0, canoc_alg)
    end

    push!(As, copy(A))
    push!(times, time_alg.β₀)
    
    if skip_first
        expvals = []
    else
        expvals = [observable(A)]
    end

    if trunc_alg isa VOPEPO
        env_double, env_triple = initialize_vomps_environments(domain(A)[1], domain(As[time_alg.update_list[1]])[1], trunc_alg)
    end
    for (i,ind) in enumerate(time_alg.update_list)
        if trunc_alg isa VOPEPO
            if domain(env_triple.edges[1,1,1])[1][1] ≠ domain(A)[1] || domain(env_triple.edges[1,1,1])[1][2] ≠ domain(As[ind])[1]
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
        if !isnothing(finalize!)
            A = finalize!(As, expvals, i)
        end

        if time_alg.verbosity > 1
            @info "Time evolution step $(i) with β = $(times[end]), obs = $(obs)"
            @info "Bond dimension is now $(dim(domain(A)[1]))"
            if time_alg.verbosity > 2
                @info "Current norm is $(norm(A))"
            end
        end
        if i > 2 && abs(expvals[end][1] - expvals[end-1][1]) < time_alg.tol_energy
            if time_alg.verbosity > 1
                @info "Ground state search converged after $(i) iterations. Energy is $(expvals[end][1])"
                return As[end], expvals[end]
            end
        end
    end
    if time_alg.verbosity > 0
        @warn "Ground state search did not converge after $(length(time_alg.update_list)) iterations. Energy is $(expvals[end][1])"
    end
    return As[end], expvals[end]
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
        A = evolution_operator(ce_alg, t)
        obs = observable(A)

        push!(expvals, obs)
        push!(As, copy(A))

        if verbosity > 1
            @info "Time evolution step $(i) with β = $(t), obs = $(obs)"
            if verbosity > 2
                @info "Current norm is $(norm(A))"
            end
        end
        if !isnothing(finalize!)
            finalize!(A, obs, i)
        end
    end
    return times, expvals, As
end
