abstract type VOPEPO end

struct VOPEPO_CTMRG <: VOPEPO
    ctm_alg
    envspace::ElementarySpace
    truncspace::ElementarySpace
    ftol::Number
    gradnormtol::Number
    maxiter::Union{Int,Function}
    c₁::Number
    verbosity::Int
end

struct VOPEPO_VUMPS <: VOPEPO
    vumps_alg
    envspace::ElementarySpace
    truncspace::ElementarySpace
    ftol::Number
    gradnormtol::Number
    maxiter::Union{Int,Function}
    c₁::Number
    verbosity::Int
end

function VOPEPO_CTMRG(ctm_alg, envspace::ElementarySpace, truncspace::ElementarySpace, ftol::Number, gradnormtol::Number, maxiter::T;
    c₁::Number = 1e-5, verbosity::Int = 0) where {T <: Function}
    return VOPEPO_CTMRG(ctm_alg, envspace, truncspace, ftol, gradnormtol, maxiter, c₁, verbosity)
end

function VOPEPO_CTMRG(ctm_alg, envspace::ElementarySpace, truncspace::ElementarySpace, ftol::Number, gradnormtol::Number, maxiter::Int = 2;
    c₁::Number = 1e-5, verbosity::Int = 0)
    return VOPEPO_CTMRG(ctm_alg, envspace, truncspace, ftol, gradnormtol, i -> maxiter, c₁, verbosity)
end

function _stack_pepos(pepos)
    return InfinitePEPO(cat(pepos...; dims = 3))
end

function initialize_vomps_environments(A₁::T, A₂::T, B::T, trunc_alg; ctm_alg = nothing) where {T<:AbstractTensorMap}
    network = InfiniteSquareNetwork(_stack_pepos((B, PEPSKit._dag(B))))
    env_double = CTMRGEnv(network, trunc_alg.envspace)
    env_double, = leading_boundary(env_double, network, ctm_alg)
    network = InfiniteSquareNetwork(_stack_pepos((A₁, A₂, PEPSKit._dag(B))))
    env_triple = CTMRGEnv(network, trunc_alg.envspace)
    env_triple, = leading_boundary(env_triple, network, ctm_alg)
    return env_double, env_triple
end

function initialize_vomps_environments(A₁_space, A₂_space, trunc_alg)
    Ds_double = trunc_alg.truncspace ⊗ trunc_alg.truncspace'
    Ds_triple = A₁_space ⊗ A₂_space ⊗ trunc_alg.truncspace'
    env_double = CTMRGEnv(Ds_double, Ds_double, trunc_alg.envspace)
    env_triple = CTMRGEnv(Ds_triple, Ds_triple, trunc_alg.envspace)
    return env_double, env_triple
end

function initialize_vomps_environments(A::T, B::T, trunc_alg; ctm_alg = nothing) where {T<:AbstractTensorMap}
    network = InfiniteSquareNetwork(_stack_pepos((B, PEPSKit._dag(B))))
    env_double = CTMRGEnv(network, trunc_alg.envspace)
    env_double, = leading_boundary(env_double, network, ctm_alg)
    network = InfiniteSquareNetwork(_stack_pepos((A, PEPSKit._dag(B))))
    env_triple = CTMRGEnv(network, trunc_alg.envspace)
    env_triple, = leading_boundary(env_triple, network, ctm_alg)
    return env_double, env_triple
end

function initialize_vomps_environments(A_space, trunc_alg)
    Ds_double = trunc_alg.truncspace ⊗ trunc_alg.truncspace'
    Ds_triple = A_space ⊗ trunc_alg.truncspace'
    env_double = CTMRGEnv(Ds_double, Ds_double, trunc_alg.envspace)
    env_triple = CTMRGEnv(Ds_triple, Ds_triple, trunc_alg.envspace)
    return env_double, env_triple
end


function approximate_state(
    A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}},
    trunc_alg::T
) where {E,S,T<:VOPEPO}
    env_double, env_triple = initialize_vomps_environments(domain(A[1])[1], domain(A[2])[1], trunc_alg)
    return approximate_state(A, env_double, env_triple, trunc_alg)
end

function vopepo_costfun_exact_base((B₀, env_double_layer, env_triple_layer), A₁, A₂, boundary_alg)
    rrule_alg = EigSolver(;
    solver_alg=KrylovKit.Arnoldi(; maxiter=30, tol=1e-6, eager=true), iterscheme=:diffgauge
    )

    E, gs = withgradient(B₀) do B
        ## construct the double and triple layer networks
        n_double_layer = InfiniteSquareNetwork(_stack_pepos((B, PEPSKit._dag(B))))
        n_triple_layer = InfiniteSquareNetwork(_stack_pepos((A₁, A₂, PEPSKit._dag(B))))

        ## contract this network
        env_double_layer′, info = PEPSKit.hook_pullback(
            leading_boundary,
            env_double_layer,
            n_double_layer,
            boundary_alg;
            alg_rrule=rrule_alg,
        )
        ## contract this network
        env_triple_layer′, info = PEPSKit.hook_pullback(
            leading_boundary,
            env_triple_layer,
            n_triple_layer,
            boundary_alg;
            alg_rrule=rrule_alg,
        )
        ## update the environments for reuse
        PEPSKit.ignore_derivatives() do
            PEPSKit.update!(env_double_layer, env_double_layer′)
            PEPSKit.update!(env_triple_layer, env_triple_layer′)
        end
            
        ## update the environments for reuse
        LHS = network_value(n_double_layer, env_double_layer′)
        RHS = network_value(n_triple_layer, env_triple_layer′)
        return - abs(RHS / sqrt(LHS))
    end
    g = only(gs)
    return E, g
end

function vopepo_costfun_exact_base_single((B₀, env_double_layer, env_triple_layer), A, boundary_alg)
    rrule_alg = EigSolver(;
    solver_alg=KrylovKit.Arnoldi(; maxiter=30, tol=1e-6, eager=true), iterscheme=:diffgauge
    )

    E, gs = withgradient(B₀) do B
        ## construct the double and triple layer networks
        n_double_layer = InfiniteSquareNetwork(_stack_pepos((B, PEPSKit._dag(B))))
        n_triple_layer = InfiniteSquareNetwork(_stack_pepos((A, PEPSKit._dag(B))))

        ## contract this network
        env_double_layer′, info = PEPSKit.hook_pullback(
            leading_boundary,
            env_double_layer,
            n_double_layer,
            boundary_alg;
            alg_rrule=rrule_alg,
        )
        ## contract this network
        env_triple_layer′, info = PEPSKit.hook_pullback(
            leading_boundary,
            env_triple_layer,
            n_triple_layer,
            boundary_alg;
            alg_rrule=rrule_alg,
        )
        ## update the environments for reuse
        PEPSKit.ignore_derivatives() do
            PEPSKit.update!(env_double_layer, env_double_layer′)
            PEPSKit.update!(env_triple_layer, env_triple_layer′)
        end
            
        ## update the environments for reuse
        LHS = network_value(n_double_layer, env_double_layer′)
        RHS = network_value(n_triple_layer, env_triple_layer′)
        return - abs(RHS / sqrt(LHS))
    end
    g = only(gs)
    return E, g
end

function pepo_retract((peps, env_monolayer), η, α)
    peps´, ξ = PEPSKit.norm_preserving_retract(peps, η, α)

    env_monolayer′ = deepcopy(env_monolayer)
    return (peps´, env_monolayer′), ξ
end
function pepo_transport!(
    ξ,
    (peps, env_monolayer),
    η,
    α,
    (peps´, env_monolayer′),
)
    return PEPSKit.norm_preserving_transport!(
        ξ, peps, η, α, peps´
    )
end;

function pepo_retract_triple((peps, env_double_layer, env_triple_layer), η, α)
    peps´, ξ = PEPSKit.norm_preserving_retract(peps, η, α)

    env_double_layer´ = deepcopy(env_double_layer)
    env_triple_layer´ = deepcopy(env_triple_layer)
    return (peps´, env_double_layer´, env_triple_layer´), ξ
end

function pepo_transport_triple!(
    ξ,
    (peps, env_double_layer, env_triple_layer),
    η,
    α,
    (peps´, env_double_layer´, env_triple_layer´),
)
    return PEPSKit.norm_preserving_transport!(
        ξ, peps, η, α, peps´
    )
end;

# AD version of VOPEPO
function approximate_state(
    A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}},
    env_double::CTMRGEnv,
    env_triple::CTMRGEnv,
    trunc_alg::T;
    iter = 1,
    B::Union{AbstractTensorMap{E,S,2,4},Nothing} = nothing,
    finalize = (x, f, g, numiter) -> (x, f, g)
) where {E,S,T<:VOPEPO}
    if isnothing(B)
        B, _ = approximate_state(A, NoEnvTruncation(truncdim(dim(trunc_alg.truncspace))))
    end
    # Use a random initial guess
    # B = randn(scalartype(B), codomain(B), domain(B))
    if domain(B)[1] != codomain(env_double.edges[1,1,1])[2]
        if trunc_alg.verbosity > 0
            @warn "Other spaces, initializing with random environment"
        end
        env_double, env_triple = initialize_vomps_environments(A[1], A[2], B, trunc_alg; trunc_alg.ctm_alg)
    end
    ls_alg = HagerZhangLineSearch(; c₁ = trunc_alg.c₁, maxiter = 5, maxfg = 10)
    optimizer_alg = LBFGS(4; maxiter=trunc_alg.maxiter(iter), gradtol=1e-5, verbosity=3)#, linesearch = ls_alg)#, scalestep = false)

    vopepo_costfun = tup -> vopepo_costfun_exact_base(tup, A[1], A[2], trunc_alg.ctm_alg)

    (B_final, env_double_final, env_triple_final), f, = optimize(
    vopepo_costfun,
    (B, env_double, env_triple),
    finalize! = finalize,
    optimizer_alg;
    inner=PEPSKit.real_inner,
    retract=pepo_retract_triple,
    (transport!)=(pepo_transport_triple!),
    hasconverged = (x, fhistory, g, gradnormhistory) -> (length(fhistory) > 1 && abs(fhistory[end] - fhistory[end-1]) < trunc_alg.ftol) || (length(gradnormhistory) > 1 && abs(gradnormhistory[end] - gradnormhistory[end-1]) < trunc_alg.gradnormtol)
    );
    return B_final, env_double_final, env_triple_final, f
end

function approximate_state(
    A::AbstractTensorMap{E,S,2,4},
    env_double::CTMRGEnv,
    env_triple::CTMRGEnv,
    trunc_alg::T;
    iter = 1,
    B::Union{AbstractTensorMap{E,S,2,4},Nothing} = nothing,
    finalize = (x, f, g, numiter) -> (x, f, g)
) where {E,S,T<:VOPEPO}
    if isnothing(B)
        B, _ = approximate_state(A, NoEnvTruncation(truncdim(dim(trunc_alg.truncspace))))
    end
    # Use a random initial guess
    B = randn(scalartype(B), codomain(B), domain(B))
    if domain(B)[1] != codomain(env_double.edges[1,1,1])[2]
        if trunc_alg.verbosity > 0
            @warn "Other spaces, initializing with random environment"
        end
        env_double, env_triple = initialize_vomps_environments(A, B, trunc_alg; trunc_alg.ctm_alg)
    end
    ls_alg = HagerZhangLineSearch(; c₁ = trunc_alg.c₁, maxiter = 5, maxfg = 10)
    optimizer_alg = LBFGS(4; maxiter=trunc_alg.maxiter(iter), gradtol=1e-5, verbosity=3)#, linesearch = ls_alg)#, scalestep = false)

    vopepo_costfun = tup -> vopepo_costfun_exact_base_single(tup, A, trunc_alg.ctm_alg)

    (B_final, env_double_final, env_triple_final), f, = optimize(
    vopepo_costfun,
    (B, env_double, env_triple),
    finalize! = finalize,
    optimizer_alg;
    inner=PEPSKit.real_inner,
    retract=pepo_retract_triple,
    (transport!)=(pepo_transport_triple!),
    hasconverged = (x, fhistory, g, gradnormhistory) -> (length(fhistory) > 1 && abs(fhistory[end] - fhistory[end-1]) < trunc_alg.ftol) || (length(gradnormhistory) > 1 && abs(gradnormhistory[end] - gradnormhistory[end-1]) < trunc_alg.gradnormtol)
    );
    return B_final, env_double_final, env_triple_final, f
end
