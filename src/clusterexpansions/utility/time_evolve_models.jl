function pf_observable(O::PEPSKit.PEPOTensor, env0, M, ctm_alg)
    @tensor Z[-3 -4; -1 -2] := O[1 1; -1 -2 -3 -4]
    partfunc = InfinitePartitionFunction(Z)
    @tensor A_M[-3 -4; -1 -2] := O[1 2; -1 -2 -3 -4] * M[2; 1]
    partfunc_M = InfinitePartitionFunction(A_M)

    if codomain(env0.edges[1,1,1])[2] != domain(Z)[1]
        env0 = CTMRGEnv(partfunc, codomain(env0.edges[1,1,1])[1])
    end        
    env, = leading_boundary(env0, partfunc, ctm_alg)

    Z = network_value(partfunc, env);
    obs = network_value(partfunc_M, env)
    if abs(imag(obs) / real(obs)) > 1e-5
        @warn "complex value for observable: $(imag(obs) / real(obs))"
    end
    return real(obs / Z), env
end

function pf_observable(A::PEPSKit.PEPSTensor, env0, M, ctm_alg)
    peps = InfinitePEPS(A)
    pspace = codomain(A)[1]
    pspaces = fill(pspace, 1, 1)
    H = PEPSKit.LocalOperator(pspaces, (CartesianIndex(1, 1),) => M)

    if codomain(env0.edges[1,1,1])[2] != domain(A)[1]
        env0 = CTMRGEnv(peps, codomain(env0.edges[1,1,1])[1])
    end
    env, = leading_boundary(env0, peps, ctm_alg)
    obs = expectation_value(peps, H, env)
    if abs(imag(obs) / real(obs)) > 1e-5
        @warn "complex value for observable: $(imag(obs) / real(obs))"
    end
    return real(obs), env
end

function get_env0(O::PEPSKit.PEPOTensor, envspace::ElementarySpace)
    @tensor Z₀[-3 -4; -1 -2] := O[1 1; -1 -2 -3 -4]
    return CTMRGEnv(InfinitePartitionFunction(Z₀), envspace)
end

function get_env0(A::PEPSKit.PEPSTensor, envspace::ElementarySpace)
    return CTMRGEnv(InfinitePEPS(A), envspace)
end

function time_evolve_model(model, param, β₀, Δβ, maxiter, χenv; χenv_approx = χenv, Dcut = nothing, T = ComplexF64, M = σˣ(), verbosity = 0, O₀ = nothing)
    envspace_approx = ℂ^χenv_approx
    envspace = ℂ^χenv
    ce_alg = model(param...; T)

    if isnothing(O₀)
        O₀ = evolution_operator(ce_alg, β₀)
    end

    ctm_alg = SimultaneousCTMRG(;
        tol=1e-10,
        miniter=4,
        maxiter=100,
        verbosity=0,
        svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10))
    )

    if isnothing(Dcut)
        Dcut = codomain(O₀)[1]
    end
    trunc_alg = ExactEnvTruncation(ctm_alg, envspace_approx, truncdim(Dcut))
    time_alg = StaticTimeEvolution(Δβ, maxiter, trunc_alg; verbosity)

    env0 = get_env0(O₀, envspace)
    observable_time_evolve = (O, env) -> pf_observable(O, env, M, ctm_alg)

    return time_evolve(O₀, observable_time_evolve, env0, ce_alg, time_alg, trunc_alg)
end