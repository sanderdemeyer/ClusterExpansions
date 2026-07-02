function pepo_retract_3D(x, η, α)
    x´_partial, ξ = PEPSKit.peps_retract(x[1:2], η, α)
    x´ = (x´_partial..., deepcopy(x[3]))
    return x´, ξ
end
function pepo_transport_3D!(ξ, x, η, α, x´)
    return PEPSKit.peps_transport!(ξ, x[1:2], η, α, x´[1:2])
end

function get_correlation_length_optimkit(PEPO, χpeps, χenv)
    @tensor O[-1 -2; -3 -4 -5 -6] := twist(PEPO, 2)[1 1 -2 -3 -4 -1 -5 -6]
    envspace_peps = ℂ^χpeps
    envspace = ℂ^χenv

    # prep
    ctm_alg = SimultaneousCTMRG(; maxiter = 150, tol = 1.0e-8, verbosity = 1)
    vumps_alg = VUMPS(; maxiter = 500, verbosity = 1)
    # gradient_alg = GradientAlgorithm(;
    #     solver_alg = KrylovKit.Arnoldi(; maxiter = 30, tol = 1.0e-6, eager = true),
    # )
    gradient_alg = EigSolver(;
    solver_alg=KrylovKit.Arnoldi(; maxiter=30, tol=1e-6, eager=true), iterscheme=:diffgauge
    )

    opt_alg = LBFGS(32; maxiter = 5, gradtol = 1.0e-5, verbosity = 3)

    # contract
    T = InfinitePEPO(O; unitcell = (1, 1, 1))
    psi0 = initializePEPS(T, envspace_peps)
    env2_0 = CTMRGEnv(InfiniteSquareNetwork(psi0), envspace)
    env3_0 = CTMRGEnv(InfiniteSquareNetwork(psi0, T), envspace)

    # optimize free energy per site
    (psi_final, env2_final, env3_final), f, = optimize(
        (psi0, env2_0, env3_0),
        opt_alg;
        inner = PEPSKit.real_inner,
        retract = pepo_retract_3D,
        (transport!) = (pepo_transport_3D!),
    ) do (psi, env2, env3)
        E, gs = withgradient(psi) do ψ
            n2 = InfiniteSquareNetwork(ψ)
            env2′, info = PEPSKit.hook_pullback(
                leading_boundary, env2, n2, ctm_alg; alg_rrule = gradient_alg
            )
            n3 = InfiniteSquareNetwork(ψ, T)
            env3′, info = PEPSKit.hook_pullback(
                leading_boundary, env3, n3, ctm_alg; alg_rrule = gradient_alg
            )
            PEPSKit.ignore_derivatives() do
                PEPSKit.update!(env2, env2′)
                PEPSKit.update!(env3, env3′)
            end
            λ3 = network_value(n3, env3)
            λ2 = network_value(n2, env2)
            println("λ3 = $λ3, λ2 = $λ2")
            return -log(abs(λ3 / λ2))
            # return -log(real(λ3 / λ2))
        end
        g = only(gs)
        return E, g
    end
    return get_correlation_length_peps(psi_final[1,1], envspace, vumps_alg)
end

function get_correlation_length_peps(peps, envspace, vumps_alg)
    n = InfiniteSquareNetwork(InfinitePEPS(peps))
    T = InfiniteMPO([n[1,1]])
    pspace = domain(n[1,1][1])[1]
    mps = InfiniteMPS([
        randn(
            ComplexF64,
            envspace * pspace * pspace',
            envspace,
        )])
    # println("n = $(n[1,1][1].space), mps.AL = $(mps.AL[1].space)")
    mps, env, _ = leading_boundary(mps, T, vumps_alg)
    return marek_gap(mps; num_vals = 20)
end

function get_boundary_peps(PEPO, χpeps; maxiter = 10)
    @tensor O[-1 -2; -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -2 -3 -4 -1 -5 -6]

    pspace = codomain(O)[1]
    trivspace = ℂ^1
    peps = randn(scalartype(PEPO), pspace, trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace')

    trunc_alg = NoEnvTruncation(truncdim(χpeps))
    for i = 1:maxiter
        @info "in boundary PEPS calculation - step $i / $maxiter"
        peps, = approximate_state((peps, O), trunc_alg)
        peps /= norm(peps)
    end
    return peps
end

function get_boundary_peps(PEPO, χpeps, frequency, maxiter, obs)
    @tensor O[-1 -2; -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -2 -3 -4 -1 -5 -6]

    pspace = codomain(O)[1]
    trivspace = ℂ^1
    peps = randn(scalartype(PEPO), pspace, trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace')

    trunc_alg = NoEnvTruncation(truncdim(χpeps))
    obss = []
    for i = 1:frequency*maxiter
        @info "in boundary PEPS calculation - step $i / $(frequency*maxiter)"
        peps, = approximate_state((peps, O), trunc_alg)
        peps /= norm(peps)
        if i % frequency == 0
            push!(obss, obs(PEPO,peps))
        end
    end
    return peps, obss
end

function get_local_observable_3D(PEPO, op, boundary_peps, χenv, ctm_alg)
    @tensor O[-1 -2; -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -2 -3 -4 -1 -5 -6]
    @tensor O_op[-1 -2; -3 -4 -5 -6] := twist(PEPO, 2)[1 2; -2 -3 -4 -1 -5 -6] * op[2; 1]
    peps = InfinitePEPS(boundary_peps)
    nw = InfiniteSquareNetwork(peps, InfinitePEPO(O), peps)
    nw_op = InfiniteSquareNetwork(peps, InfinitePEPO(O_op), peps)
    env′, = leading_boundary(CTMRGEnv(nw, ℂ^χenv), nw, ctm_alg)

    return network_value(nw_op, env′) / network_value(nw, env′)
end