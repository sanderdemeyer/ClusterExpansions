# different contraction methods have different environment initializations
initialize_environment(alg::Union{SimultaneousCTMRG, SequentialCTMRG}, args...; kwargs...) = CTMRGEnv(args...; kwargs...)
initialize_environment(alg::C4vCTMRG, args...; kwargs...) = initialize_random_c4v_env(args...; kwargs...)

function my_finalize_ctmrg_base!(x, f, g, T; reshuffle = false)
    if reshuffle
        n2 = InfiniteSquareNetwork(x[1])
        n3 = InfiniteSquareNetwork(x[1], T)
        chienv = dim(domain(x[2].corners[1,1,1]))
        ctm_alg_1 = SimultaneousCTMRG(; tol=1e-9, maxiter=30, verbosity=1, trunc = truncrank(chienv));
        env2_new, = leading_boundary(x[2], n2, ctm_alg_1);
        env3_new, = leading_boundary(x[3], n3, ctm_alg_1);

        return (x[1], env2_new, env3_new, norm(g)), f, g
    else
        return (x[1], x[2], x[3], norm(g)), f, g
    end
end

function get_initial_state(psi_unnormalized, O, trunc_alg, maxiter_init)
    for i = 1:maxiter_init
        psi_unnormalized, = approximate_state((psi_unnormalized, O), trunc_alg)
    end
    return InfinitePEPS(psi_unnormalized)
end    

function vbpeps(CE_PEPO, obs, Vpeps, Venv; boundary_alg = nothing, gradient_alg = nothing, optimizer_alg = nothing, symmetrization = nothing, maxiter_init = 20, verbosity_boundary_alg = 1, verbosity_optimizer = 3)
    # set symmetrization = RotateReflect() to use C4vCTMRG
    if isnothing(symmetrization)
        boundary_alg_type = SimultaneousCTMRG
    elseif symmetrization == RotateReflect()
        boundary_alg_type = C4vCTMRG
    end

    @tensor O[-1 -2; -3 -4 -5 -6] := twist(CE_PEPO, 2)[1 1; -2 -3 -4 -1 -5 -6]
    T = InfinitePEPO(O)

    @tensor M0[-1 -2; -3 -4 -5 -6] := CE_PEPO[1 2; -2 -3 -4 -1 -5 -6] * obs[2; 1]

    my_finalize_ctmrg! = (x,f,g,numiter) -> my_finalize_ctmrg_base!(x,f,g,T)

    # Define algorithms if they were not given.
    if isnothing(boundary_alg)
        boundary_alg = boundary_alg_type(;
            tol = 1.0e-8,
            maxiter = 400,
            verbosity = verbosity_boundary_alg)
    end
    if isnothing(gradient_alg)
        gradient_alg = FixedPointGradient(;
        solver_alg = KrylovKit.Arnoldi(;
            tol = 1.0e-6,
            maxiter = 5,
            krylovdim = 20,
            eager = true,
        ))
    end
    if isnothing(optimizer_alg)
        optimizer_alg = LBFGS(64;
        gradtol = 1.0e-5,
        maxiter = 75,
        verbosity = verbosity_optimizer)
    end

    my_retract, = pepo_symmetrize_retract_and_finalize!(
        symmetrization, pepo_retract_3layer
    )

    ## Initialize
    trunc_alg = NoEnvTruncation(truncrank(dim(Vpeps)))
    psi_unnormalized = get_initial_state(initializePEPS(T, Vpeps)[1,1], O, trunc_alg, maxiter_init)
    psi0 = PEPSKit.peps_normalize(symmetrize!(psi_unnormalized, symmetrization))

    env2_0 = initialize_environment(boundary_alg, InfiniteSquareNetwork(psi0), Venv)
    env3_0 = initialize_environment(boundary_alg, InfiniteSquareNetwork(psi0, T), Venv)

    pepo_costfun = generate_ctmrg_pepo_cost_function(
        T;
        symmetrization,
        boundary_alg,
        gradient_alg,
    )
    (psi_final, env2_final, env3_final), f, ng, normgradhistory = optimize(
        pepo_costfun,
        (psi0, env2_0, env3_0),
        optimizer_alg;
        inner = PEPSKit.real_inner,
        retract = my_retract,
        (transport!) = (pepo_transport_3layer!),
        (finalize!) = (my_finalize_ctmrg!),
    )

    n3_final = InfiniteSquareNetwork(psi_final, T)
    # n2_final = InfiniteSquareNetwork(psi_final)
    m0 = PEPSKit.contract_local_tensor((1, 1, 1), M0, n3_final, env3_final) / PEPSKit._contract_site((1, 1), n3_final, env3_final)

    return m0, psi_final
end

