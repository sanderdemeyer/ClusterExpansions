abstract type TimeEvolution end

struct StaticTimeEvolution <: TimeEvolution
    Δβ
    maxiter
    trunc_alg
    verbosity
end

struct TimeDependentTimeEvolution <: TimeEvolution
    Δβ
    maxiter
    trunc_alg
    verbosity
    f₁
    f₂
end

function evolution_operator(ce_alg::ClusterExpansion, time_alg::StaticTimeEvolution)
    _, O_clust_full = clusterexpansion(ce_alg.T, ce_alg.p, time_alg.Δβ, ce_alg.twosite_op, ce_alg.onesite_op; spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops)
    O_clust_full = convert(TensorMap, O_clust_full)
    O = zeros(ComplexF64, codomain(O_clust_full), domain(O_clust_full))
    for (f_full, f_conv) in zip(blocks(O_clust_full), blocks(O))
        f_conv[2] .= f_full[2]
    end
    return O
end

function evolution_operator(ce_alg::ClusterExpansion, time_alg::TimeDependentTimeEvolution, β)
    _, O_clust_full = clusterexpansion(ce_alg.T, ce_alg.p, time_alg.Δβ, time_alg.f₂(β) * ce_alg.twosite_op, time_alg.f₁(β) * ce_alg.onesite_op; spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops)
    O_clust_full = convert(TensorMap, O_clust_full)
    O = zeros(ComplexF64, codomain(O_clust_full), domain(O_clust_full))
    O[] = O_clust_full[]
    return O
end

function evolution_operator(ce_alg::ClusterExpansion, β)
    _, O_clust_full = clusterexpansion(ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops)
    O_clust_full = convert(TensorMap, O_clust_full)
    O = zeros(ComplexF64, codomain(O_clust_full), domain(O_clust_full))
    O[] = O_clust_full[]
    return O
end

function StaticTimeEvolution(Δβ, maxiter, trunc_alg; verbosity = 0)
    return StaticTimeEvolution(Δβ, maxiter, trunc_alg, verbosity)
end

function TimeDependentTimeEvolution(Δβ, maxiter, trunc_alg; verbosity = 0, f₁ = β -> 1.0, f₂ = β -> 1.0)
    return TimeDependentTimeEvolution(Δβ, maxiter, trunc_alg, verbosity, f₁, f₂)
end

function time_evolve(
    A, 
    observable, # function A, env -> f(A, env)
    env::CTMRGEnv,
    ce_alg::ClusterExpansion,
    time_alg::StaticTimeEvolution,
    trunc_alg::EnvTruncation;
    finalize! = nothing
)
    if time_alg.verbosity > 2
        println("Time evolution with Δβ = $(time_alg.Δβ), maxiter = $(time_alg.maxiter)")
    end
    O = evolution_operator(ce_alg, time_alg)
    expvals = Float64[]
    times = ComplexF64[]
    for i = 1:time_alg.maxiter
        A, _ = approximate_state((A, O), trunc_alg)

        obs, env = observable(A, env)
        push!(times, i * time_alg.Δβ)
        push!(expvals, obs)
        if time_alg.verbosity > 1
            println("Time evolution step $(i) with β = $(i * time_alg.Δβ), obs = $(obs)")
        end
        if !isnothing(finalize!)
            finalize!(A, env, obs, i)
        end
    end
    return times, expvals, A
end

function time_evolve(
    A, 
    observable, # function A, env -> f(A, env)
    env::CTMRGEnv,
    ce_alg::ClusterExpansion,
    time_alg::TimeDependentTimeEvolution,
    trunc_alg::PEPSKit.CTMRGAlgorithm
)
    if time_alg.verbosity > 2
        println("Time evolution with Δβ = $(time_alg.Δβ), maxiter = $(time_alg.maxiter)")
    end
    expvals = Float64[]
    times = ComplexF64[]
    for i = 1:maxiter
        O = evolution_operator(time_alg, ce_alg, i * time_alg.Δβ)
        A, _ = approximate_state((A, O), trunc_alg)

        obs, env = observable(A, env)
        push!(times, i * time_alg.Δβ)
        push!(expvals, obs)
        if time_alg.verbosity > 1
            println("Time evolution step $(i) with β = $(i * time_alg.Δβ), obs = $(obs)")
        end
        if !isnothing(finalize!)
            finalize!(A, env, obs, i)
        end
    end
    return times, expvals, A
end

function time_scan(
    times::Array,
    observable,
    env::CTMRGEnv,
    ce_alg::ClusterExpansion,
)
    if time_alg.verbosity > 2
        println("Time evolution with Δβ = $(time_alg.Δβ), maxiter = $(time_alg.maxiter)")
    end
    expvals = Float64[]
    for (i,t) = enumerate(times)
        O = evolution_operator(ce_alg, t)

        obs, env = observable(O, env)
        push!(expvals, obs)
        if time_alg.verbosity > 1
            println("Time evolution step $(i) with t = $(t), obs = $(obs)")
        end
        if !isnothing(finalize!)
            finalize!(O, env, obs, i)
        end
    end
    return times, expvals
end
